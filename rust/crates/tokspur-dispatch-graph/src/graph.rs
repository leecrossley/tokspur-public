//! The data dependencies of one batch. Two dispatches conflict when they touch overlapping bytes of one allocation and at least one writes them. The error flag (`buffer(7)`) is shared by every kernel: all OR into it with relaxed atomics, which commute, but the kernels in [`FLAG_READERS`] also load it and return early once it is set, so a flag reader is ordered against every other dispatch (and two readers against each other only through the writes they both make). `tests/kernels.rs` checks the list against the MSL sources.

use crate::trace::{Access, Batch, Dispatch, Target};

/// Kernels that load the error flag (`atomic_load_explicit(error, ...)`).
pub const FLAG_READERS: [&str; 6] = [
    "qsa_score_accumulate",
    "qsa_chronological_factors",
    "qsa_chronological_channels",
    "qsa_chronological_serial_factors",
    "qsa_verify_rows_factors",
    "qsa_verify_rows_channels",
];

pub fn reads_flag(dispatch: &Dispatch) -> bool {
    FLAG_READERS.contains(&dispatch.function())
}

fn touches_flag(dispatch: &Dispatch) -> bool {
    dispatch
        .bindings
        .iter()
        .any(|binding| binding.target == Target::ErrorFlag)
}

/// Whether `later` must stay ordered after `earlier`.
pub fn conflicts(earlier: &Dispatch, later: &Dispatch) -> bool {
    // Every flag-bound kernel writes the flag; ordering matters only when one side also reads it.
    if touches_flag(earlier) && touches_flag(later) && (reads_flag(earlier) || reads_flag(later)) {
        return true;
    }
    for a in &earlier.bindings {
        let Target::Range(first, first_access) = a.target else {
            continue;
        };
        for b in &later.bindings {
            let Target::Range(second, second_access) = b.target else {
                continue;
            };
            if (first_access == Access::Write || second_access == Access::Write)
                && first.overlaps(&second)
            {
                return true;
            }
        }
    }
    false
}

/// Dependency edges `(earlier, later)` between dispatch positions of a batch in its own order; not transitively reduced.
pub fn edges(dispatches: &[&Dispatch]) -> Vec<(usize, usize)> {
    let mut found = Vec::new();
    for later in 0..dispatches.len() {
        for earlier in 0..later {
            if conflicts(dispatches[earlier], dispatches[later]) {
                found.push((earlier, later));
            }
        }
    }
    found
}

/// The earliest level each dispatch can run at: one more than the highest level among the dispatches it depends on.
pub fn levels(count: usize, edges: &[(usize, usize)]) -> Vec<usize> {
    let mut level = vec![0; count];
    // Edges are generated with `earlier < later`, so one pass in `later` order sees every predecessor's final level.
    let mut sorted = edges.to_vec();
    sorted.sort_by_key(|&(earlier, later)| (later, earlier));
    for (earlier, later) in sorted {
        level[later] = level[later].max(level[earlier] + 1);
    }
    level
}

/// A batch's dependency shape: how much concurrency the data allow.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    pub dispatches: usize,
    pub barriers: usize,
    pub edges: usize,
    /// Dispatches on the longest dependency chain.
    pub depth: usize,
    /// The widest level of the as-soon-as-possible schedule.
    pub widest: usize,
}

pub fn shape(batch: &Batch) -> Shape {
    let dispatches: Vec<&Dispatch> = batch.dispatches().collect();
    let found = edges(&dispatches);
    let level = levels(dispatches.len(), &found);
    let depth = level.iter().max().map_or(0, |deepest| deepest + 1);
    let mut width = vec![0; depth];
    for &at in &level {
        width[at] += 1;
    }
    Shape {
        dispatches: dispatches.len(),
        barriers: batch.barriers(),
        edges: found.len(),
        depth,
        widest: width.into_iter().max().unwrap_or(0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::{Binding, Range};

    fn dispatch(function: &str, bindings: &[(u64, u64, u64, Access)]) -> Dispatch {
        let mut all = vec![Binding {
            index: 7,
            target: Target::ErrorFlag,
        }];
        for (index, &(slot, offset, bytes, access)) in bindings.iter().enumerate() {
            all.push(Binding {
                index: index as u64,
                target: Target::Range(
                    Range {
                        slot,
                        offset,
                        bytes,
                    },
                    access,
                ),
            });
        }
        Dispatch {
            pipeline: format!("{function}:4294967295"),
            aligned: true,
            grid: [1, 1, 1],
            group: [128, 1, 1],
            bindings: all,
            constants: Vec::new(),
        }
    }

    #[test]
    fn writes_order_overlapping_reads_and_writes_only() {
        // Arrange
        let writer = dispatch("a", &[(0, 0, 16, Access::Write)]);
        let overlapping_reader = dispatch("b", &[(0, 12, 8, Access::Read)]);
        let adjacent_reader = dispatch("c", &[(0, 16, 8, Access::Read)]);
        let other_reader = dispatch("d", &[(0, 0, 16, Access::Read)]);

        // Act / Assert
        assert!(conflicts(&writer, &overlapping_reader));
        assert!(conflicts(&overlapping_reader, &writer));
        assert!(!conflicts(&writer, &adjacent_reader));
        assert!(!conflicts(&overlapping_reader, &other_reader));
        assert!(conflicts(
            &writer,
            &dispatch("e", &[(0, 0, 4, Access::Write)])
        ));
    }

    #[test]
    fn the_shared_error_flag_orders_only_its_readers() {
        // Arrange
        let first = dispatch("a", &[(0, 0, 4, Access::Write)]);
        let second = dispatch("b", &[(1, 0, 4, Access::Write)]);
        let reader = dispatch("qsa_score_accumulate", &[(2, 0, 4, Access::Write)]);

        // Act / Assert
        assert!(!conflicts(&first, &second));
        assert!(conflicts(&first, &reader));
        assert!(conflicts(&reader, &second));
    }

    #[test]
    fn levels_follow_the_longest_chain() {
        // Arrange
        let a = dispatch("a", &[(0, 0, 4, Access::Write)]);
        let b = dispatch("b", &[(1, 0, 4, Access::Write)]);
        let c = dispatch("c", &[(0, 0, 4, Access::Read), (2, 0, 4, Access::Write)]);
        let d = dispatch("d", &[(2, 0, 4, Access::Read), (1, 0, 4, Access::Read)]);
        let all = [&a, &b, &c, &d];

        // Act
        let found = edges(&all);
        let level = levels(all.len(), &found);

        // Assert
        assert_eq!(found, vec![(0, 2), (1, 3), (2, 3)]);
        assert_eq!(level, vec![0, 0, 1, 2]);
    }
}
