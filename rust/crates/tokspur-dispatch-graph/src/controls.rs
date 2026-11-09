//! Controls for the check, built from a real reference trace. Each is a candidate the check must accept (a legal reschedule) or refuse (one broken dependency, argument, dispatch or digest).

use crate::graph::{edges, levels};
use crate::trace::{Batch, Dispatch, Event, Item, Outcome, Trace};

/// What the check must say about a control.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Expect {
    Pass,
    Fail,
}

pub struct Control {
    pub name: &'static str,
    pub expect: Expect,
    pub candidate: Trace,
}

fn map_batches(trace: &Trace, mut change: impl FnMut(&Batch) -> Batch) -> Trace {
    let mut result = trace.clone();
    for event in &mut result.events {
        if let Event::Batch(batch) = event {
            *batch = change(batch);
        }
    }
    result
}

/// The first batch with a dependency, or none.
fn first_dependent(trace: &Trace) -> Option<usize> {
    trace.batches().position(|batch| {
        let dispatches: Vec<&Dispatch> = batch.dispatches().collect();
        !edges(&dispatches).is_empty()
    })
}

/// Every dispatch at its earliest level, a barrier only between levels; `reverse` reverses the order inside each level.
fn by_levels(batch: &Batch, reverse: bool) -> Batch {
    let dispatches: Vec<&Dispatch> = batch.dispatches().collect();
    let level = levels(dispatches.len(), &edges(&dispatches));
    let depth = level.iter().max().map_or(0, |deepest| deepest + 1);
    let mut items = Vec::new();
    for at in 0..depth {
        if at > 0 {
            items.push(Item::Barrier);
        }
        let mut members: Vec<usize> = (0..dispatches.len()).filter(|&i| level[i] == at).collect();
        if reverse {
            members.reverse();
        }
        items.extend(
            members
                .into_iter()
                .map(|i| Item::Dispatch(dispatches[i].clone())),
        );
    }
    if !items.is_empty() {
        items.push(Item::Barrier);
    }
    Batch {
        items,
        outcome: batch.outcome.clone(),
    }
}

/// Removes the barrier between levels 0 and 1 of the level schedule, so a level-1 dispatch (which depends on level 0 by construction) may overlap it.
fn merge_first_levels(batch: &Batch) -> Batch {
    let mut scheduled = by_levels(batch, false);
    if let Some(at) = scheduled
        .items
        .iter()
        .position(|item| matches!(item, Item::Barrier))
        && at + 1 < scheduled.items.len()
    {
        scheduled.items.remove(at);
    }
    scheduled
}

/// Moves the later dispatch of the first dependency to just before the earlier one, keeping one barrier after every dispatch.
fn swap_first_dependency(batch: &Batch) -> Batch {
    let mut dispatches: Vec<Dispatch> = batch.dispatches().cloned().collect();
    let references: Vec<&Dispatch> = dispatches.iter().collect();
    if let Some(&(earlier, later)) = edges(&references).first() {
        let moved = dispatches.remove(later);
        dispatches.insert(earlier, moved);
    }
    Batch {
        items: dispatches
            .into_iter()
            .flat_map(|dispatch| [Item::Dispatch(dispatch), Item::Barrier])
            .collect(),
        outcome: batch.outcome.clone(),
    }
}

fn first_dispatch_mut(batch: &mut Batch) -> Option<&mut Dispatch> {
    batch.items.iter_mut().find_map(|item| match item {
        Item::Dispatch(dispatch) => Some(dispatch),
        Item::Barrier => None,
    })
}

fn flip(digest: &str) -> String {
    let mut text = digest.to_owned();
    let last = if text.ends_with('0') { '1' } else { '0' };
    text.pop();
    text.push(last);
    text
}

/// The controls for `reference`, in a fixed order.
pub fn controls(reference: &Trace) -> Vec<Control> {
    let dependent = first_dependent(reference);
    let mut list = vec![
        Control {
            name: "identity",
            expect: Expect::Pass,
            candidate: reference.clone(),
        },
        Control {
            name: "level schedule: barriers only between dependency levels",
            expect: Expect::Pass,
            candidate: map_batches(reference, |batch| by_levels(batch, false)),
        },
        Control {
            name: "level schedule, each level reversed",
            expect: Expect::Pass,
            candidate: map_batches(reference, |batch| by_levels(batch, true)),
        },
    ];
    if let Some(target) = dependent {
        let mut index = 0;
        list.push(Control {
            name: "level schedule without the barrier after level 0",
            expect: Expect::Fail,
            candidate: map_batches(reference, |batch| {
                index += 1;
                if index - 1 == target {
                    merge_first_levels(batch)
                } else {
                    batch.clone()
                }
            }),
        });
        let mut index = 0;
        list.push(Control {
            name: "first dependency swapped",
            expect: Expect::Fail,
            candidate: map_batches(reference, |batch| {
                index += 1;
                if index - 1 == target {
                    swap_first_dependency(batch)
                } else {
                    batch.clone()
                }
            }),
        });
    }
    let mut once = true;
    list.push(Control {
        name: "first grid width changed",
        expect: Expect::Fail,
        candidate: map_batches(reference, |batch| {
            let mut changed = batch.clone();
            if once && let Some(dispatch) = first_dispatch_mut(&mut changed) {
                dispatch.grid[0] += 1;
                once = false;
            }
            changed
        }),
    });
    let mut once = true;
    list.push(Control {
        name: "first setBytes digest changed",
        expect: Expect::Fail,
        candidate: map_batches(reference, |batch| {
            let mut changed = batch.clone();
            if once {
                for item in &mut changed.items {
                    if let Item::Dispatch(dispatch) = item
                        && let Some(constant) = dispatch.constants.first_mut()
                    {
                        constant.sha256 = flip(&constant.sha256);
                        once = false;
                        break;
                    }
                }
            }
            changed
        }),
    });
    let mut once = true;
    list.push(Control {
        name: "first dispatch dropped",
        expect: Expect::Fail,
        candidate: map_batches(reference, |batch| {
            let mut changed = batch.clone();
            if once
                && let Some(at) = changed
                    .items
                    .iter()
                    .position(|item| matches!(item, Item::Dispatch(_)))
            {
                changed.items.remove(at);
                once = false;
            }
            changed
        }),
    });
    let mut once = true;
    list.push(Control {
        name: "first post-batch state digest changed",
        expect: Expect::Fail,
        candidate: map_batches(reference, |batch| {
            let mut changed = batch.clone();
            if once
                && let Outcome::Submitted { states, .. } = &mut changed.outcome
                && let Some((_, digest)) = states.first_mut()
            {
                *digest = flip(digest);
                once = false;
            }
            changed
        }),
    });
    let mut candidate = reference.clone();
    if let Some(Event::HostRead { sha256, .. }) = candidate
        .events
        .iter_mut()
        .find(|event| matches!(event, Event::HostRead { .. }))
    {
        *sha256 = flip(sha256);
        list.push(Control {
            name: "first host readback (logits) digest changed",
            expect: Expect::Fail,
            candidate,
        });
    }
    list
}
