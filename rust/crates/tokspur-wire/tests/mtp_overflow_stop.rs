//! Output-packet boundary contract for an accepted MTP stop that lands in round overflow.
//!
//! The native depth-two round may commit past the packet end. Its accepted stop must stay queued until the next packet drains it: a packet that reports `stopped` without emitting the stop is refused, which invalidates the owned worker.

use tokspur_json::Json;
use tokspur_wire::mtp_wire::MtpWireReply;

const STOP: u64 = 248046;
const BASE: u64 = 4104;
const COUNTERS: [&str; 8] = [
    "native_mtp_head_calls",
    "native_mtp_prime_calls",
    "native_mtp_rounds",
    "native_mtp_proposed",
    "native_mtp_accepted",
    "native_mtp_recovery_rounds",
    "native_mtp_carried_anchors",
    "native_mtp_discarded_anchors",
];

fn int(value: u64) -> Json {
    Json::integer(value)
}

fn generation(active: bool, remaining: u64) -> Json {
    Json::new_object([
        ("id", int(1)),
        ("active", Json::Bool(active)),
        ("remaining", int(remaining)),
        ("pending", Json::Bool(false)),
        ("committed", int(0)),
        ("emitted", int(0)),
        ("overflow", int(0)),
        ("remaining_compute", int(remaining)),
        ("total_budget", int(256)),
        ("continuation_allowed", Json::Bool(true)),
        ("stop_queued", Json::Bool(false)),
        ("stop_drained", Json::Bool(false)),
    ])
}

fn ready() -> Json {
    Json::new_object([
        ("event", Json::from("ready")),
        ("native_mtp_depth", int(2)),
        ("native_mtp_carry_scope", Json::from("http-generation-v2")),
        ("native_mtp_generation", generation(false, 0)),
        ("native_mtp_output_packet", int(8)),
        ("native_mtp_output_packet_schema", int(1)),
    ])
}

fn sampling(prefix: u64, words: u64) -> Json {
    Json::new_object([
        ("valid", Json::Bool(true)),
        ("entropy_prefix", int(prefix)),
        ("entropy_words", int(words)),
    ])
}

struct Done<'a> {
    prefix: u64,
    generated: &'a [u64],
    remaining: u64,
    stop: Option<u64>,
    counters: [u64; 8],
}

fn done(d: &Done<'_>) -> Json {
    let stopped = d.stop.is_some();
    let mut fields = vec![
        ("event", Json::from("done")),
        ("committed_prefix", int(d.prefix)),
        (
            "generated_ids",
            Json::Array(d.generated.iter().copied().map(int).collect()),
        ),
        ("sampling", sampling(d.prefix, 12)),
        ("continuation_allowed", Json::Bool(!stopped)),
        (
            "status",
            Json::from(if stopped { "stopped" } else { "length" }),
        ),
        (
            "finish_reason",
            Json::from(if stopped { "stop" } else { "length" }),
        ),
        ("stop_id", d.stop.map_or(Json::Null, int)),
        (
            "native_mtp_generation",
            generation(d.remaining > 0, d.remaining),
        ),
    ];
    fields.extend(COUNTERS.iter().zip(d.counters).map(|(n, v)| (*n, int(v))));
    Json::new_object(fields)
}

fn previous() -> Json {
    done(&Done {
        prefix: BASE,
        generated: &[42],
        remaining: 248,
        stop: None,
        counters: [0; 8],
    })
}

fn output_job(remaining: u64, committed: u64, emitted: u64, pending: u64) -> Json {
    Json::new_object([
        ("tokens", Json::Array(Vec::new())),
        ("generate", int(1)),
        ("reset", Json::Bool(false)),
        ("stop_ids", Json::Array(vec![int(STOP)])),
        (
            "mtp_generation",
            Json::new_object([
                ("id", int(1)),
                ("begin", Json::Bool(false)),
                ("remaining", int(remaining)),
                ("committed", int(committed)),
                ("emitted", int(emitted)),
                ("pending", int(pending)),
                ("remaining_compute", int(256 - committed)),
                ("total_budget", int(256)),
            ]),
        ),
    ])
}

/// A depth-one round of the anchor plus an accepted draft stop, committing two tokens.
fn stop_round() -> Json {
    Json::new_object([
        ("event", Json::from("mtp_round")),
        ("base", int(BASE)),
        ("committed_prefix", int(BASE + 2)),
        ("count", int(2)),
        ("accepted", int(1)),
        ("proposed", int(1)),
        ("lookahead_anchor", Json::Bool(true)),
        ("used_carried_anchor", Json::Bool(false)),
        ("pending_anchor", Json::Bool(false)),
    ])
}

fn token(id: u64, token_prefix: u64) -> Json {
    Json::new_object([
        ("event", Json::from("token")),
        ("id", int(id)),
        ("committed_prefix", int(BASE + 2)),
        ("round_committed_prefix", int(BASE + 2)),
        ("token_prefix", int(token_prefix)),
        ("entropy_words", int(12)),
    ])
}

#[test]
fn stopped_packet_without_its_stop_token_is_refused() {
    // Arrange: only the anchor fits in this one-token packet, so the accepted stop is in overflow.
    let (ready, previous, job) = (ready(), previous(), output_job(248, 8, 8, 0));
    let mut reply = MtpWireReply::new(&ready, &previous, &job, false).expect("packet admission");
    reply.event(&stop_round()).expect("round admission");
    reply.event(&token(42, BASE + 1)).expect("anchor admission");

    // Act: the reply the round published before this fix.
    let result = reply.finish(&done(&Done {
        prefix: BASE + 2,
        generated: &[42],
        remaining: 0,
        stop: Some(STOP),
        counters: [1, 1, 1, 1, 1, 0, 0, 0],
    }));

    // Assert
    let error = result.expect_err("an unemitted stop must be refused");
    assert_eq!(error.to_string(), "MTP continuation flag differs");
}

#[test]
fn queued_stop_is_published_by_the_next_packet() {
    // Arrange
    let (ready, previous, first_job) = (ready(), previous(), output_job(248, 8, 8, 0));
    let mut first =
        MtpWireReply::new(&ready, &previous, &first_job, false).expect("first packet admission");
    first.event(&stop_round()).expect("round admission");
    first.event(&token(42, BASE + 1)).expect("anchor admission");
    let first_done = done(&Done {
        prefix: BASE + 2,
        generated: &[42],
        remaining: 247,
        stop: None,
        counters: [1, 1, 1, 1, 1, 0, 0, 0],
    });

    // Act: the first packet ends as length with the stop queued, then the next packet drains it.
    let first_result = first.finish(&first_done);
    let second_job = output_job(247, 10, 9, 1);
    let mut second = MtpWireReply::new(&ready, &first_done, &second_job, false)
        .expect("second packet admission");
    second
        .event(&token(STOP, BASE + 2))
        .expect("drained stop admission");
    let second_result = second.finish(&done(&Done {
        prefix: BASE + 2,
        generated: &[STOP],
        remaining: 0,
        stop: Some(STOP),
        counters: [1, 1, 1, 1, 1, 0, 0, 0],
    }));

    // Assert
    assert!(
        first_result.is_ok(),
        "length packet with a queued stop: {first_result:?}"
    );
    assert!(
        second_result.is_ok(),
        "drained stop completion: {second_result:?}"
    );
}
