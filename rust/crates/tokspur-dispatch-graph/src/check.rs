//! Dependency-graph equivalence of a candidate trace against a reference.
//!
//! The candidate passes when, batch by batch, it has the same host-side events (allocations, writes, reads with their digests, copies, releases), the same multiset of dispatches with identical arguments, every reference data dependency separated by at least one barrier in candidate order (any topologically compatible order is allowed), and identical post-batch state digests. Slots are compared as allocated, so the candidate must allocate in the reference order.

use std::collections::{BTreeMap, VecDeque};

use crate::graph::{Shape, edges, shape};
use crate::trace::{Batch, Dispatch, Event, Item, Outcome, Trace};

/// Stop listing after this many failures; the count stays exact.
const LISTED: usize = 32;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Report {
    pub failures: usize,
    /// The first [`LISTED`] failure messages.
    pub listed: Vec<String>,
    pub batches: Vec<BatchReport>,
}

impl Report {
    pub fn passed(&self) -> bool {
        self.failures == 0
    }

    fn fail(&mut self, message: String) {
        self.failures += 1;
        if self.listed.len() < LISTED {
            self.listed.push(message);
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchReport {
    pub reference: Shape,
    pub candidate_barriers: usize,
}

fn describe(event: &Event) -> &'static str {
    match event {
        Event::Allocation(_) => "allocation",
        Event::Release(_) => "release",
        Event::HostWrite { .. } => "host write",
        Event::HostRead { .. } => "host read",
        Event::HostCopy(_) => "host copy",
        Event::Batch(_) => "batch",
    }
}

/// Candidate position of each reference dispatch, matched by full argument signature in order of appearance.
fn match_dispatches(
    report: &mut Report,
    batch: usize,
    reference: &[&Dispatch],
    candidate: &[&Dispatch],
) -> Option<Vec<usize>> {
    let mut waiting: BTreeMap<String, VecDeque<usize>> = BTreeMap::new();
    for (index, dispatch) in reference.iter().enumerate() {
        waiting
            .entry(dispatch.signature())
            .or_default()
            .push_back(index);
    }
    let mut position = vec![usize::MAX; reference.len()];
    let mut complete = true;
    for (at, dispatch) in candidate.iter().enumerate() {
        match waiting
            .get_mut(&dispatch.signature())
            .and_then(VecDeque::pop_front)
        {
            Some(index) => position[index] = at,
            None => {
                complete = false;
                report.fail(format!(
                    "batch {batch}: candidate dispatch {at} ({}) has no identical reference dispatch",
                    dispatch.pipeline
                ));
            }
        }
    }
    for (index, &at) in position.iter().enumerate() {
        if at == usize::MAX {
            complete = false;
            report.fail(format!(
                "batch {batch}: reference dispatch {index} ({}) is missing from the candidate",
                reference[index].pipeline
            ));
        }
    }
    complete.then_some(position)
}

/// Barriers before each candidate item position.
fn barrier_prefix(batch: &Batch) -> Vec<usize> {
    let mut seen = 0;
    let mut before = Vec::new();
    for item in &batch.items {
        match item {
            Item::Dispatch(_) => before.push(seen),
            Item::Barrier => seen += 1,
        }
    }
    before
}

fn check_batch(report: &mut Report, index: usize, reference: &Batch, candidate: &Batch) {
    let ours: Vec<&Dispatch> = reference.dispatches().collect();
    let theirs: Vec<&Dispatch> = candidate.dispatches().collect();
    report.batches.push(BatchReport {
        reference: shape(reference),
        candidate_barriers: candidate.barriers(),
    });
    if let Some(position) = match_dispatches(report, index, &ours, &theirs) {
        let before = barrier_prefix(candidate);
        for (earlier, later) in edges(&ours) {
            let (first, second) = (position[earlier], position[later]);
            if first > second || before[second] == before[first] {
                report.fail(format!(
                    "batch {index}: dependency of reference dispatch {later} ({}) on {earlier} ({}) is not separated by a barrier",
                    ours[later].pipeline, ours[earlier].pipeline
                ));
            }
        }
    }
    match (&reference.outcome, &candidate.outcome) {
        (
            Outcome::Submitted {
                dispatches: count,
                states,
            },
            Outcome::Submitted {
                dispatches: other_count,
                states: other_states,
            },
        ) => {
            if count != other_count {
                report.fail(format!(
                    "batch {index}: {other_count} dispatches submitted, reference {count}"
                ));
            }
            if states != other_states {
                let differing = states
                    .iter()
                    .zip(other_states)
                    .filter(|(a, b)| a != b)
                    .count()
                    + states.len().abs_diff(other_states.len());
                report.fail(format!(
                    "batch {index}: {differing} post-batch state digests differ"
                ));
            }
        }
        (Outcome::Aborted, Outcome::Aborted) => {}
        _ => report.fail(format!("batch {index}: outcome differs")),
    }
}

pub fn check(reference: &Trace, candidate: &Trace) -> Report {
    let mut report = Report {
        failures: 0,
        listed: Vec::new(),
        batches: Vec::new(),
    };
    if reference.events.len() != candidate.events.len() {
        report.fail(format!(
            "candidate has {} host events and batches, reference {}",
            candidate.events.len(),
            reference.events.len()
        ));
    }
    let mut batch = 0;
    for (at, (ours, theirs)) in reference.events.iter().zip(&candidate.events).enumerate() {
        match (ours, theirs) {
            (Event::Batch(ours), Event::Batch(theirs)) => {
                check_batch(&mut report, batch, ours, theirs);
                batch += 1;
            }
            _ if ours == theirs => {}
            _ if describe(ours) == describe(theirs) => {
                report.fail(format!("event {at}: {} differs", describe(ours)));
            }
            _ => report.fail(format!(
                "event {at}: candidate has a {}, reference a {}",
                describe(theirs),
                describe(ours)
            )),
        }
    }
    report
}
