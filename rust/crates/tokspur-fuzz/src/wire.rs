//! Worker wire target: the owned worker's readiness admission (`tokspur-server`'s `validate_worker_ready`) and `tokspur-wire`'s private V8 and branch8 reply admission and dispatch formula, on arbitrary JSON. Built with `verify8` or a later feature level; every validator behaves as the C++ build with the matching macros does.
//!
//! The input is one JSON document. Every validator sees the whole document. An array also drives the multi-document gates: `[ready, previous, job, policy, event..., done]` for `Verify8Reply`, `[prepared, previous, job, value]` for branch8 control, and ten shape numbers plus a position for the QSA dispatch count.

#![forbid(unsafe_code)]

use std::fmt::Debug;

use tokspur_json::{Json, parse_json};
use tokspur_server::admission::validate_worker_ready;
use tokspur_wire::qsa::{QsaError, QsaShape, qsa_block_budget, qsa_dispatches, validate_qsa_shape};
#[cfg(feature = "fast-math")]
use tokspur_wire::verify8::verify8_fast_math;
use tokspur_wire::verify8::{Verify8Context, Verify8Reply, validate_verify8_ready};
#[cfg(feature = "branch8")]
use tokspur_wire::verify8::{branch8_session, validate_branch8_control, validate_branch8_ready};

use crate::encode::escape;

/// Runs a validator twice: it must not panic, must give the same answer both times, and any refusal must carry a message.
fn stable<T: PartialEq + Debug, E: PartialEq + Debug + std::fmt::Display>(
    name: &str,
    input: &[u8],
    validator: impl Fn() -> Result<T, E>,
) -> Result<T, E> {
    let first = validator();
    let second = validator();
    assert!(
        first == second,
        "{name} is not deterministic: {}",
        escape(input, 512)
    );
    if let Err(error) = &first {
        assert!(
            !error.to_string().is_empty(),
            "{name} refused without a message"
        );
    }
    first
}

fn number(value: &Json, name: &str) -> Option<u64> {
    value.find(name).ok().flatten()?.unsigned_integer().ok()
}

/// `[ready, previous, job, policy, event..., done]` through `Verify8Reply` for both `reset_resync` values and both branch8 arms: events are admitted in order until the first refusal, and `events()` is available exactly when `finish` admitted the done.
fn reply_sequence(items: &[Json], input: &[u8]) {
    let [ready, previous, job, policy, rest @ ..] = items else {
        return;
    };
    let Some((done, events)) = rest.split_last() else {
        return;
    };
    let before = if previous.is_null() {
        0
    } else {
        number(previous, "committed_prefix")
            .map_or(u32::MAX, |n| u32::try_from(n).unwrap_or(u32::MAX))
    };
    let seed = number(job, "seed").unwrap_or(0);
    for (reset_resync, branch8) in [(false, false), (true, false), (false, true), (true, true)] {
        let context = Verify8Context {
            ready,
            previous,
            job,
            before,
            policy,
            seed,
            reset_resync,
            branch8,
        };
        let Ok(mut reply) = Verify8Reply::new(context) else {
            continue;
        };
        let mut admitted = true;
        for event in events {
            if reply.event(event).is_err() {
                admitted = false;
                break;
            }
        }
        let finished = admitted && reply.finish(done).is_ok();
        match reply.events() {
            Ok(records) => assert!(
                finished && records.len() <= 16 && records.len() <= events.len(),
                "events available without an admitted done: {}",
                escape(input, 512)
            ),
            Err(error) => assert!(
                !finished && error.to_string() == "private V8 events not admitted",
                "an admitted done withheld its events: {}",
                escape(input, 512)
            ),
        }
    }
}

/// Ten shape numbers and a position through the QSA dispatch formula: a valid shape has its block budget, and a position inside the cache has at least the 18 fixed dispatches.
fn dispatches(items: &[Json], input: &[u8]) {
    let numbers: Vec<u32> = items
        .iter()
        .map_while(|item| {
            item.unsigned_integer()
                .ok()
                .and_then(|n| u32::try_from(n).ok())
        })
        .collect();
    let [
        channels,
        query_heads,
        kv_heads,
        head_width,
        index_heads,
        index_width,
        rotary_width,
        compression,
        token_budget,
        capacity,
        position,
        ..,
    ] = numbers[..]
    else {
        return;
    };
    let shape = QsaShape {
        channels,
        query_heads,
        kv_heads,
        head_width,
        index_heads,
        index_width,
        rotary_width,
        compression,
        token_budget,
        capacity,
    };
    let valid = stable("validate_qsa_shape", input, || validate_qsa_shape(shape));
    let budget = stable("qsa_block_budget", input, || qsa_block_budget(shape));
    let count = stable("qsa_dispatches", input, || qsa_dispatches(shape, position));
    match valid {
        Err(error) => assert!(
            error == QsaError::Shape
                && budget == Err(QsaError::Shape)
                && count == Err(QsaError::Shape),
            "invalid shape accepted: {}",
            escape(input, 512)
        ),
        Ok(()) => {
            assert_eq!(budget, Ok(token_budget / compression));
            match count {
                Ok(count) => assert!(
                    position < capacity && count >= 18,
                    "dispatches: {}",
                    escape(input, 512)
                ),
                Err(error) => assert!(
                    error == QsaError::Capacity && position >= capacity,
                    "dispatches refused: {}",
                    escape(input, 512)
                ),
            }
        }
    }
}

/// Every wire validator on one JSON document (refusals are expected; the properties are no panic, determinism and the cross-checks above).
pub fn check(data: &[u8]) {
    let Ok(value) = parse_json(data) else {
        return;
    };
    for (stack, slots, budget) in [
        ("direct", 0, 0),
        ("recurrent-replay", 0, 0),
        ("direct", 2, 64 << 20),
    ] {
        stable("validate_worker_ready", data, || {
            validate_worker_ready(&value, stack, slots, budget)
        })
        .ok();
    }
    stable("validate_verify8_ready", data, || {
        validate_verify8_ready(&value)
    })
    .ok();
    #[cfg(feature = "fast-math")]
    stable("verify8_fast_math", data, || verify8_fast_math(&value)).ok();
    #[cfg(feature = "branch8")]
    {
        let session = stable("branch8_session", data, || branch8_session(&value));
        let ready = stable("validate_branch8_ready", data, || {
            validate_branch8_ready(&value)
        });
        assert!(
            ready.is_err() || session.is_ok(),
            "branch8 readiness without a session: {}",
            escape(data, 512)
        );
    }
    let Json::Array(items) = &value else {
        return;
    };
    #[cfg(feature = "branch8")]
    if let [prepared, previous, job, reply, ..] = &items[..] {
        for cold_resync in [false, true] {
            stable("validate_branch8_control", data, || {
                validate_branch8_control(prepared, previous, job, reply, cold_resync)
            })
            .ok();
        }
    }
    reply_sequence(items, data);
    dispatches(items, data);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validators_survive_documents_of_every_shape() {
        for document in [
            b"{}".as_slice(),
            b"[]",
            b"[{},{},{},{},{}]",
            b"[null,null,{\"tokens\":[],\"generate\":1,\"reset\":false},{},{\"event\":\"done\"}]",
            b"[4096,16,4,256,4,128,64,4,2048,65536,100]",
            b"[4096,16,4,256,4,128,64,4,2048,65536,65536]",
            b"[0,16,4,256,4,128,64,4,2048,65536,1]",
            b"{\"event\":\"ready\",\"maximum_generated_tokens\":4096}",
        ] {
            check(document);
        }
    }
}
