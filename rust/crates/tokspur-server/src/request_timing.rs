//! The opt-in per-request phase record (`request_timing`), printed under `--telemetry-stdout` for every request that reached the generation thread, failures included.
//!
//! Every time is integer microseconds. Fields ending `_at_us` are offsets from the request's arrival (the accept instant). The others are durations. A field whose phase never happened is left out and named in `unavailable`. No token IDs and no text are recorded.
//!
//! Two fields have a fixed meaning that their names alone do not give. `first_sse_write_at_us` is the role delta, which is written before prefill starts. `first_tool_call_at_us` is written only after the whole generation and the final response parse. `arrival_to_dequeue_us` includes reading and routing the request on its reader thread, so it is an upper bound on time spent waiting for the slot.

use std::time::{Duration, Instant};

use tokspur_json::Json;

/// The record's schema number.
pub const REQUEST_TIMING_SCHEMA: u64 = 1;

/// How a request that reached the generation thread ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// The reply was fully written.
    Ok,
    /// Refused before any worker command started.
    Refused,
    /// The client disconnected, or the server is stopping.
    Cancelled,
    /// The request deadline passed.
    Deadline,
    /// A worker command started and the request then failed.
    Failed,
}

impl Outcome {
    pub const ALL: [Self; 5] = [
        Self::Ok,
        Self::Refused,
        Self::Cancelled,
        Self::Deadline,
        Self::Failed,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Refused => "refused",
            Self::Cancelled => "cancelled",
            Self::Deadline => "deadline",
            Self::Failed => "failed",
        }
    }

    /// Classifies a failed request in the order `run_job` chooses its status: an expired deadline first (408), then a disconnect or stop, then whether the target started (502 or 400).
    pub fn of_failure(deadline_passed: bool, stopped: bool, target_started: bool) -> Self {
        if deadline_passed {
            Self::Deadline
        } else if stopped {
            Self::Cancelled
        } else if target_started {
            Self::Failed
        } else {
            Self::Refused
        }
    }
}

/// Whether a request with this outcome gets a record. Every request that reached the generation thread does, so a reader can use the records as a denominator.
pub fn emits_record(telemetry_stdout: bool, _outcome: Outcome) -> bool {
    telemetry_stdout
}

/// The session's own phase stamps for the latest request, taken only when enabled.
#[derive(Debug, Clone, Copy, Default)]
pub struct PhaseStamps {
    /// Top of `ServeSession::complete`, before chat admission.
    pub request_start: Option<Instant>,
    pub prefill_start: Option<Instant>,
    pub generation_start: Option<Instant>,
    /// When the first output packet's worker command returned, before its tokens were parsed or written.
    pub first_packet_done: Option<Instant>,
    pub first_packet_tokens: u64,
    /// After the final parse and every finishing write.
    pub generation_end: Option<Instant>,
    pub output_packets: u64,
    pub max_packet_tokens: u64,
    pub completion_tokens: u64,
}

/// Which kinds of output one streamed chunk carries.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DeltaKinds {
    pub reasoning: bool,
    pub content: bool,
    pub tool_call: bool,
}

impl DeltaKinds {
    pub fn any(self) -> bool {
        self.reasoning || self.content || self.tool_call
    }
}

fn non_empty(value: Option<&Json>) -> bool {
    match value {
        Some(Json::String(text)) => !text.is_empty(),
        Some(Json::Array(items)) => !items.is_empty(),
        _ => false,
    }
}

/// The output kinds in a chat completion chunk's first choice. The role delta and the finish chunk carry none.
pub fn delta_kinds(event: &Json) -> DeltaKinds {
    let delta = event
        .find("choices")
        .ok()
        .flatten()
        .and_then(|choices| match choices {
            Json::Array(items) => items.first(),
            _ => None,
        })
        .and_then(|choice| choice.find("delta").ok().flatten());
    let Some(delta @ Json::Object(_)) = delta else {
        return DeltaKinds::default();
    };
    let field = |name: &str| delta.find(name).ok().flatten();
    DeltaKinds {
        reasoning: non_empty(field("reasoning_content")),
        content: non_empty(field("content")),
        tool_call: non_empty(field("tool_calls")),
    }
}

/// What `run_job` stamps around delivery, plus the session's stamps at the end.
#[derive(Debug, Clone)]
pub struct RequestTiming {
    arrival: Instant,
    dequeued: Instant,
    stream: bool,
    first_sse_write: Option<Instant>,
    last_sse_write: Option<Instant>,
    sse_writes: u64,
    sse_write_total: Duration,
    json_write: Option<Duration>,
    first_reasoning: Option<Instant>,
    first_content: Option<Instant>,
    first_tool_call: Option<Instant>,
    last_output: Option<Instant>,
    abort_drain: Option<Duration>,
}

fn micros(duration: Duration) -> Json {
    Json::integer(u64::try_from(duration.as_micros()).unwrap_or(u64::MAX))
}

impl RequestTiming {
    pub fn new(arrival: Instant, dequeued: Instant, stream: bool) -> Self {
        Self {
            arrival,
            dequeued,
            stream,
            first_sse_write: None,
            last_sse_write: None,
            sse_writes: 0,
            sse_write_total: Duration::ZERO,
            json_write: None,
            first_reasoning: None,
            first_content: None,
            first_tool_call: None,
            last_output: None,
            abort_drain: None,
        }
    }

    /// One SSE data write that started at `start` and returned at `end`, carrying `kinds`.
    pub fn sse_write(&mut self, start: Instant, end: Instant, kinds: DeltaKinds) {
        self.first_sse_write.get_or_insert(end);
        self.last_sse_write = Some(end);
        self.sse_writes += 1;
        self.sse_write_total += end.saturating_duration_since(start);
        if kinds.reasoning {
            self.first_reasoning.get_or_insert(end);
        }
        if kinds.content {
            self.first_content.get_or_insert(end);
        }
        if kinds.tool_call {
            self.first_tool_call.get_or_insert(end);
        }
        if kinds.any() {
            self.last_output = Some(end);
        }
    }

    /// The single JSON reply of a non-streaming request.
    pub fn json_write(&mut self, start: Instant, end: Instant) {
        self.json_write = Some(end.saturating_duration_since(start));
    }

    pub fn abort_drain(&mut self, start: Instant, end: Instant) {
        self.abort_drain = Some(end.saturating_duration_since(start));
    }

    /// The record, with `finished` as the end of the request on the generation thread.
    pub fn record(
        &self,
        response_id: Option<&[u8]>,
        outcome: Outcome,
        phases: &PhaseStamps,
        finished: Instant,
    ) -> Json {
        let mut fields: Vec<(Vec<u8>, Json)> = Vec::with_capacity(32);
        let mut unavailable: Vec<Json> = Vec::new();
        let mut put = |name: &str, value: Option<Json>| match value {
            Some(value) => fields.push((name.as_bytes().to_vec(), value)),
            None => unavailable.push(Json::from(name)),
        };
        let at = |instant: Option<Instant>| {
            instant.map(|instant| micros(instant.saturating_duration_since(self.arrival)))
        };
        let between = |start: Option<Instant>, end: Option<Instant>| match (start, end) {
            (Some(start), Some(end)) => Some(micros(end.saturating_duration_since(start))),
            _ => None,
        };
        put("event", Some(Json::from("request_timing")));
        put("schema", Some(Json::integer(REQUEST_TIMING_SCHEMA)));
        put("response_id", response_id.map(Json::from));
        put("outcome", Some(Json::from(outcome.name())));
        put("stream", Some(Json::Bool(self.stream)));
        put(
            "arrival_to_dequeue_us",
            Some(micros(
                self.dequeued.saturating_duration_since(self.arrival),
            )),
        );
        put(
            "admission_us",
            between(phases.request_start, phases.prefill_start),
        );
        put(
            "prefill_us",
            between(phases.prefill_start, phases.generation_start),
        );
        put(
            "generation_us",
            between(phases.generation_start, phases.generation_end),
        );
        put("first_packet_done_at_us", at(phases.first_packet_done));
        put(
            "first_packet_tokens",
            phases
                .first_packet_done
                .map(|_| Json::integer(phases.first_packet_tokens)),
        );
        put("output_packets", Some(Json::integer(phases.output_packets)));
        put(
            "max_packet_tokens",
            Some(Json::integer(phases.max_packet_tokens)),
        );
        put(
            "completion_tokens",
            Some(Json::integer(phases.completion_tokens)),
        );
        if self.stream {
            put("first_sse_write_at_us", at(self.first_sse_write));
            put("last_sse_write_at_us", at(self.last_sse_write));
            put("sse_writes", Some(Json::integer(self.sse_writes)));
            put("sse_write_us_total", Some(micros(self.sse_write_total)));
            put("first_reasoning_at_us", at(self.first_reasoning));
            put("first_content_at_us", at(self.first_content));
            put("first_tool_call_at_us", at(self.first_tool_call));
            put("last_output_at_us", at(self.last_output));
        } else {
            put("json_write_us", self.json_write.map(micros));
        }
        put("abort_drain_us", self.abort_drain.map(micros));
        put(
            "total_us",
            Some(micros(finished.saturating_duration_since(self.arrival))),
        );
        fields.push((b"unavailable".to_vec(), Json::Array(unavailable)));
        Json::Object(fields)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn ms(base: Instant, offset: u64) -> Instant {
        base + Duration::from_millis(offset)
    }

    fn number(record: &Json, name: &str) -> Option<u64> {
        record
            .find(name)
            .unwrap()
            .map(|value| value.unsigned_integer().unwrap())
    }

    fn unavailable(record: &Json) -> Vec<Vec<u8>> {
        record
            .find("unavailable")
            .unwrap()
            .unwrap()
            .array()
            .unwrap()
            .iter()
            .map(|name| name.string().unwrap().to_vec())
            .collect()
    }

    fn chunk(delta: Json) -> Json {
        Json::new_object([(
            "choices",
            Json::Array(vec![Json::new_object([
                ("index", Json::integer(0)),
                ("delta", delta),
                ("finish_reason", Json::Null),
            ])]),
        )])
    }

    fn role() -> Json {
        chunk(Json::new_object([("role", Json::from("assistant"))]))
    }

    fn content(text: &str) -> Json {
        chunk(Json::new_object([("content", Json::from(text))]))
    }

    fn reasoning(text: &str) -> Json {
        chunk(Json::new_object([("reasoning_content", Json::from(text))]))
    }

    /// A streamed request: queue `queue`, admission 1 ms, prefill 10 ms, first packet `packet` ms after generation starts, each write `write` ms.
    fn streamed(queue: u64, packet: u64, write: u64) -> Json {
        let t0 = Instant::now();
        let dequeued = ms(t0, queue);
        let prefill_start = ms(dequeued, 1);
        let generation_start = ms(prefill_start, 10);
        let packet_done = ms(generation_start, packet);
        let mut timing = RequestTiming::new(t0, dequeued, true);
        let role_start = ms(dequeued, 1);
        timing.sse_write(role_start, ms(role_start, write), delta_kinds(&role()));
        let mut clock = packet_done;
        for event in [reasoning("r"), content("a"), content("b")] {
            let end = ms(clock, write);
            timing.sse_write(clock, end, delta_kinds(&event));
            clock = end;
        }
        let phases = PhaseStamps {
            request_start: Some(dequeued),
            prefill_start: Some(prefill_start),
            generation_start: Some(generation_start),
            first_packet_done: Some(packet_done),
            first_packet_tokens: 3,
            generation_end: Some(clock),
            output_packets: 1,
            max_packet_tokens: 3,
            completion_tokens: 3,
        };
        timing.record(Some(b"chatcmpl-x"), Outcome::Ok, &phases, clock)
    }

    #[test]
    fn a_delayed_packet_moves_only_packet_and_output_times() {
        let fast = streamed(5, 2, 1);
        let slow = streamed(5, 50, 1);
        assert_eq!(
            number(&slow, "first_packet_done_at_us").unwrap()
                - number(&fast, "first_packet_done_at_us").unwrap(),
            48_000
        );
        for name in ["arrival_to_dequeue_us", "sse_write_us_total", "prefill_us"] {
            assert_eq!(number(&slow, name), number(&fast, name), "{name}");
        }
        assert!(number(&slow, "first_content_at_us") > number(&fast, "first_content_at_us"));
    }

    #[test]
    fn a_slow_reader_moves_only_delivery_time() {
        let fast = streamed(5, 2, 1);
        let slow = streamed(5, 2, 20);
        assert_eq!(number(&fast, "sse_write_us_total"), Some(4_000));
        assert_eq!(number(&slow, "sse_write_us_total"), Some(80_000));
        for name in [
            "arrival_to_dequeue_us",
            "first_packet_done_at_us",
            "prefill_us",
        ] {
            assert_eq!(number(&slow, name), number(&fast, name), "{name}");
        }
    }

    #[test]
    fn a_queued_request_moves_only_queue_and_absolute_times() {
        let first = streamed(0, 2, 1);
        let second = streamed(300, 2, 1);
        assert_eq!(number(&second, "arrival_to_dequeue_us"), Some(300_000));
        for name in [
            "admission_us",
            "prefill_us",
            "generation_us",
            "sse_write_us_total",
        ] {
            assert_eq!(number(&second, name), number(&first, name), "{name}");
        }
    }

    #[test]
    fn the_role_delta_is_not_first_output() {
        assert_eq!(delta_kinds(&role()), DeltaKinds::default());
        let record = streamed(5, 2, 1);
        let first_write = number(&record, "first_sse_write_at_us").unwrap();
        let first_content = number(&record, "first_content_at_us").unwrap();
        // Prefill ends 16 ms after arrival (5 queue, 1 admission, 10 prefill).
        assert!(first_write < 16_000);
        assert!(first_content > 16_000);
        assert!(number(&record, "first_reasoning_at_us").unwrap() < first_content);
    }

    #[test]
    fn delta_kinds_read_only_non_empty_members() {
        assert!(delta_kinds(&content("x")).content);
        assert!(!delta_kinds(&content("")).content);
        let tools = chunk(Json::new_object([(
            "tool_calls",
            Json::Array(vec![Json::Object(Vec::new())]),
        )]));
        assert!(delta_kinds(&tools).tool_call);
        assert!(!delta_kinds(&Json::Null).any());
        assert!(!delta_kinds(&chunk(Json::Null)).any());
    }

    #[test]
    fn every_outcome_is_counted_when_on_and_none_when_off() {
        for outcome in Outcome::ALL {
            assert!(emits_record(true, outcome), "{}", outcome.name());
            assert!(!emits_record(false, outcome), "{}", outcome.name());
        }
    }

    #[test]
    fn failure_classes_follow_the_status_order() {
        assert_eq!(Outcome::of_failure(true, true, true), Outcome::Deadline);
        assert_eq!(Outcome::of_failure(false, true, true), Outcome::Cancelled);
        assert_eq!(Outcome::of_failure(false, false, true), Outcome::Failed);
        assert_eq!(Outcome::of_failure(false, false, false), Outcome::Refused);
    }

    #[test]
    fn a_refused_request_names_its_missing_phases() {
        let t0 = Instant::now();
        let timing = RequestTiming::new(t0, ms(t0, 3), false);
        let phases = PhaseStamps {
            request_start: Some(ms(t0, 3)),
            ..PhaseStamps::default()
        };
        let record = timing.record(None, Outcome::Refused, &phases, ms(t0, 4));
        assert_eq!(
            record.find("outcome").unwrap().unwrap().string().unwrap(),
            b"refused"
        );
        let missing = unavailable(&record);
        for name in [
            "response_id",
            "admission_us",
            "prefill_us",
            "generation_us",
            "first_packet_done_at_us",
            "first_packet_tokens",
            "json_write_us",
            "abort_drain_us",
        ] {
            assert!(missing.contains(&name.as_bytes().to_vec()), "{name}");
            assert!(record.find(name).unwrap().is_none(), "{name}");
        }
        assert_eq!(number(&record, "total_us"), Some(4_000));
        assert!(record.find("sse_writes").unwrap().is_none());
    }

    #[test]
    fn the_record_holds_no_text_and_stays_small() {
        let record = streamed(5, 2, 1);
        let bytes = record.dump().unwrap();
        assert!(bytes.len() < 1024, "{}", bytes.len());
        for text in [b"\"r\"".as_slice(), b"\"a\"", b"\"b\""] {
            assert!(!bytes.windows(text.len()).any(|window| window == text));
        }
    }
}
