//! Fake native targets: the process the owned worker drives, as a Shuttle thread that reads wire commands from one channel and writes events to another, one event per write.
//!
//! The plain broker target follows `BrokerFake` in `tests/serve_protocol.rs`; the branch8 target wraps `branch8::Fake` from `tests/http_private` exactly as that suite's `target` process does. Both check the wire fields the owned worker adds before answering, as the branch8 fake process does.

use std::sync::Arc;

use shuttle::sync::atomic::{AtomicUsize, Ordering};
use shuttle::sync::mpsc::{Receiver, Sender};
use tokspur_json::Json;

/// The plain broker readiness epoch.
pub const EPOCH: u64 = 77;
/// The plain codec's `<|im_end|>`.
pub const IM_END: u32 = 1000;
const SLOTS: usize = 2;
const STORAGE: [(&str, u64); 5] = [
    ("checkpoint_slots", SLOTS as u64),
    ("checkpoint_combined_budget_bytes", 1 << 20),
    ("checkpoint_host_bytes", 512),
    ("checkpoint_store_host_bytes", 256),
    ("checkpoint_gpu_bytes", 1024),
];
/// The fields the owned worker adds to each broker command.
const WIRE: [&str; 6] = [
    "broker_epoch",
    "command_id",
    "lease_id",
    "expected_prefix",
    "broker_protocol",
    "broker_batch_width",
];

/// How a misbehaving target answers one command (by command id).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fault {
    None,
    /// Every event of command `at` carries the previous epoch.
    StaleEpoch {
        at: u64,
    },
    /// Every event of command `at` carries the previous lease.
    StaleLease {
        at: u64,
    },
    /// Command `at` is answered with a replay of the previous done.
    DuplicateReply {
        at: u64,
    },
}

/// The target exited; its event pipe closes.
pub struct Exit;

pub trait Target {
    /// One wire command's events, the done last.
    fn command(&mut self, wire: &Json) -> Result<Vec<Json>, Exit>;
}

/// The target process: one command at a time, each event a separate write.
pub fn serve(
    mut target: Box<dyn Target + Send>,
    commands: Receiver<Json>,
    events: Sender<Json>,
    written: Arc<AtomicUsize>,
) {
    while let Ok(wire) = commands.recv() {
        let Ok(replies) = target.command(&wire) else {
            return;
        };
        for reply in replies {
            if events.send(reply).is_err() {
                return;
            }
            // Each event is its own pipe write: a scheduling point, so the owned worker may read, and the client cancel, between events.
            written.fetch_add(1, Ordering::SeqCst);
        }
    }
}

pub fn number(value: &Json, name: &str) -> u64 {
    value
        .find(name)
        .unwrap()
        .unwrap_or_else(|| panic!("fixture field {name}"))
        .unsigned_integer()
        .unwrap()
}

pub fn flag(value: &Json, name: &str) -> bool {
    value.find(name).unwrap().unwrap().boolean().unwrap()
}

/// Replaces the first `name` or appends it.
pub fn put(value: &mut Json, name: &str, replacement: Json) {
    let Json::Object(fields) = value else {
        panic!("fixture value is not an object");
    };
    match fields
        .iter_mut()
        .find(|(key, _)| key.as_slice() == name.as_bytes())
    {
        Some((_, item)) => *item = replacement,
        None => fields.push((name.as_bytes().to_vec(), replacement)),
    }
}

fn without(value: &Json, names: &[&str]) -> Json {
    let fields = value.object().unwrap();
    Json::Object(
        fields
            .iter()
            .filter(|(key, _)| !names.iter().any(|name| key.as_slice() == name.as_bytes()))
            .cloned()
            .collect(),
    )
}

/// The plain broker readiness, as `broker_ready()` in `serve_protocol.rs`.
pub fn broker_ready() -> Json {
    let mut ready = Json::new_object([
        ("event", Json::from("ready")),
        ("small_gemv_column_splits", Json::integer(16)),
        ("block_gemv_formats", Json::integer(3)),
        ("stack_execution", Json::from("direct")),
        ("maximum_generated_tokens", Json::integer(4096)),
        ("checkpoint_broker", Json::Bool(true)),
        ("broker_epoch", Json::integer(EPOCH)),
    ]);
    for (name, amount) in STORAGE {
        put(&mut ready, name, Json::integer(amount));
    }
    ready
}

struct Saved {
    prefix: u32,
    order: u64,
}

/// The plain broker target: exact history, two checkpoint slots, and a fixed reply ("ok<|im_end|>") restarted by every input and begin.
pub struct BrokerTarget {
    command: u64,
    lease: u64,
    position: u32,
    history: Vec<u32>,
    saved: Vec<Saved>,
    order: u64,
    sampling: Json,
    seed: u64,
    plan: Vec<u32>,
    next: usize,
    previous: Json,
    fault: Fault,
}

impl BrokerTarget {
    pub fn new(fault: Fault) -> Self {
        Self {
            command: 0,
            lease: 0,
            position: 0,
            history: Vec::new(),
            saved: Vec::new(),
            order: 0,
            sampling: Json::Null,
            seed: 0,
            plan: vec![u32::from(b'o') + 1, u32::from(b'k') + 1, IM_END],
            next: 0,
            previous: Json::Null,
            fault,
        }
    }

    fn stamp(&self, mut value: Json) -> Json {
        for (name, amount) in STORAGE {
            put(&mut value, name, Json::integer(amount));
        }
        put(
            &mut value,
            "checkpoint_retained",
            Json::integer(self.saved.len() as u64),
        );
        put(&mut value, "broker_epoch", Json::integer(EPOCH));
        put(&mut value, "command_id", Json::integer(self.command));
        put(&mut value, "lease_id", Json::integer(self.lease));
        value
    }

    fn sampled(&self) -> Json {
        let mut policy = self.sampling.clone();
        put(&mut policy, "valid", Json::Bool(true));
        put(
            &mut policy,
            "entropy_prefix",
            Json::integer(u64::from(self.position)),
        );
        put(&mut policy, "seed", Json::integer(self.seed));
        policy
    }

    fn control(&mut self, job: &Json) -> Vec<Json> {
        let op = job.find("op").unwrap().unwrap().string().unwrap().to_vec();
        let mut outcome = "captured";
        let mut common = self.position;
        if op == b"branch_begin" {
            self.lease += 1;
            let restore = flag(job, "allow_restore");
            let future = job.find("prompt_ids").unwrap().unwrap().array().unwrap();
            common = 0;
            if restore {
                while (common as usize) < future.len()
                    && (common as usize) < self.history.len()
                    && u64::from(self.history[common as usize])
                        == future[common as usize].unsigned_integer().unwrap()
                {
                    common += 1;
                }
            }
            // The newest of the longest saved prefixes within the common one.
            let selected = self
                .saved
                .iter()
                .filter(|saved| restore && saved.prefix <= common)
                .max_by_key(|saved| (saved.prefix, saved.order))
                .map(|saved| saved.prefix);
            if let Some(prefix) = selected {
                self.position = prefix;
                self.saved.retain(|saved| saved.prefix <= prefix);
                outcome = "restored";
            } else {
                self.position = 0;
                self.saved.clear();
                outcome = "cold_miss";
            }
            self.history.truncate(self.position as usize);
            self.sampling = job.find("sampling").unwrap().unwrap().clone();
            self.seed = number(job, "seed");
            self.next = 0;
        } else {
            assert!(
                op == b"branch_capture" && self.position != 0,
                "fixture capture"
            );
            if self.saved.len() == SLOTS
                && let Some(oldest) = (0..SLOTS).min_by_key(|&i| self.saved[i].order)
            {
                self.saved.remove(oldest);
            }
            self.order += 1;
            self.saved.push(Saved {
                prefix: self.position,
                order: self.order,
            });
        }
        let position = u64::from(self.position);
        vec![self.stamp(Json::new_object([
            ("event", Json::from("control_done")),
            ("op", Json::from(op.as_slice())),
            ("outcome", Json::from(outcome)),
            ("model_steps", Json::integer(0)),
            ("generated_ids", Json::Array(Vec::new())),
            ("committed_prefix", Json::integer(position)),
            ("common_prefix", Json::integer(u64::from(common))),
            ("sampling", self.sampled()),
            ("logits_available", Json::Bool(position != 0)),
            ("stop_id", Json::Null),
            ("continuation_allowed", Json::Bool(position != 0)),
        ]))]
    }

    fn step(&mut self, job: &Json) -> Vec<Json> {
        if flag(job, "reset") {
            self.lease += 1;
            self.position = 0;
            self.history.clear();
            self.saved.clear();
            self.sampling = job.find("sampling").unwrap().unwrap().clone();
            self.seed = number(job, "seed");
        }
        let generate = number(job, "generate") != 0;
        self.position += 1;
        let position = u64::from(self.position);
        let mut replies = Vec::new();
        let mut ids = Vec::new();
        let mut stopped = false;
        if generate {
            let id = self.plan[self.next.min(self.plan.len() - 1)];
            self.next += 1;
            stopped = id == IM_END;
            ids.push(Json::integer(u64::from(id)));
            replies.push(self.stamp(Json::new_object([
                ("event", Json::from("token")),
                ("id", Json::integer(u64::from(id))),
                ("committed_prefix", Json::integer(position)),
            ])));
            self.history.push(id);
        } else {
            let tokens = job.find("tokens").unwrap().unwrap().array().unwrap();
            self.history
                .push(tokens[0].unsigned_integer().unwrap() as u32);
            self.next = 0;
        }
        let (status, finish) = match (generate, stopped) {
            (false, _) => ("prefilled", Json::Null),
            (true, true) => ("stopped", Json::from("stop")),
            (true, false) => ("length", Json::from("length")),
        };
        replies.push(self.stamp(Json::new_object([
            ("event", Json::from("done")),
            ("committed_prefix", Json::integer(position)),
            ("generated_ids", Json::Array(ids)),
            ("status", Json::from(status)),
            ("finish_reason", finish),
            (
                "stop_id",
                if stopped {
                    Json::integer(u64::from(IM_END))
                } else {
                    Json::Null
                },
            ),
            ("continuation_allowed", Json::Bool(!stopped)),
            ("sampling", self.sampled()),
            ("small_gemv_column_splits", Json::integer(16)),
            ("block_gemv_formats", Json::integer(3)),
            ("stack_execution", Json::from("direct")),
        ])));
        replies
    }
}

impl Target for BrokerTarget {
    fn command(&mut self, wire: &Json) -> Result<Vec<Json>, Exit> {
        let control = wire.find("op").unwrap().is_some();
        let fresh = if control {
            wire.find("op").unwrap().unwrap().string().unwrap() == b"branch_begin"
        } else {
            flag(wire, "reset")
        };
        if number(wire, "broker_epoch") != EPOCH
            || number(wire, "command_id") != self.command + 1
            || number(wire, "lease_id") != self.lease + u64::from(fresh)
            || number(wire, "expected_prefix") != u64::from(self.position)
        {
            return Err(Exit);
        }
        let job = without(wire, &WIRE);
        self.command += 1;
        if let Fault::DuplicateReply { at } = self.fault
            && at == self.command
        {
            // The command is taken, and answered with the last done again.
            self.lease += u64::from(fresh);
            return Ok(vec![self.previous.clone()]);
        }
        let mut replies = if control {
            self.control(&job)
        } else {
            self.step(&job)
        };
        for reply in &mut replies {
            match self.fault {
                Fault::StaleEpoch { at } if at == self.command => {
                    put(reply, "broker_epoch", Json::integer(EPOCH - 1));
                }
                Fault::StaleLease { at } if at == self.command => {
                    put(reply, "lease_id", Json::integer(self.lease - 1));
                }
                _ => {}
            }
        }
        if let Some(done) = replies.last() {
            self.previous = done.clone();
        }
        Ok(replies)
    }
}

/// `branch8::target`'s per-line checks and dispatch, without the process.
#[cfg(feature = "branch8")]
pub struct Branch8Target {
    fake: crate::branch8::Fake,
    fault: Fault,
}

#[cfg(feature = "branch8")]
impl Branch8Target {
    pub fn new(fault: Fault) -> Self {
        Self {
            fake: crate::branch8::Fake::new(crate::branch8::prepared()),
            fault,
        }
    }
}

#[cfg(feature = "branch8")]
impl Target for Branch8Target {
    fn command(&mut self, wire: &Json) -> Result<Vec<Json>, Exit> {
        use crate::verify8::{array, integer, present, remove, text};
        let fake = &mut self.fake;
        if integer(wire, "broker_epoch") != 4242
            || integer(wire, "command_id") != fake.command + 1
            || integer(wire, "expected_prefix") != fake.history.len() as u64
            || text(wire, "broker_protocol") != b"branch8-v1"
            || integer(wire, "broker_batch_width") != 8
        {
            return Err(Exit);
        }
        let mut job = wire.clone();
        for name in crate::branch8::WIRE {
            remove(&mut job, name);
        }
        let control = present(&job, "op");
        let begin = control && text(&job, "op") == b"branch_begin";
        if integer(wire, "lease_id") != fake.lease + u64::from(begin) {
            return Err(Exit);
        }
        if let Fault::DuplicateReply { at } = self.fault
            && at == fake.command + 1
        {
            fake.command += 1;
            fake.lease += u64::from(begin);
            return Ok(vec![fake.previous.clone()]);
        }
        // `target` restarts the reply at each begin; restarting it at each input batch as well lets appended turns generate again.
        if begin || (!control && !array(&job, "tokens").is_empty()) {
            fake.plan("ok<|im_end|>");
        }
        let mut replies = Vec::new();
        let done = fake.apply(&job, |event| replies.push(event.clone()));
        replies.push(done);
        for reply in &mut replies {
            match self.fault {
                Fault::StaleEpoch { at } if at == fake.command => {
                    put(reply, "broker_epoch", Json::integer(4241));
                }
                Fault::StaleLease { at } if at == fake.command => {
                    put(reply, "lease_id", Json::integer(fake.lease - 1));
                }
                _ => {}
            }
        }
        Ok(replies)
    }
}
