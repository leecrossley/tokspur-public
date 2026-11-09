//! Dependency-graph equivalence for Metal dispatch traces: development tooling for the phase B encoder, never part of a shipped binary. A candidate encoder is equivalent to the C++ host when it runs the same dispatches with identical arguments, keeps every data dependency behind a barrier in any compatible order, and leaves identical state and readback digests.

#![forbid(unsafe_code)]

pub mod check;
pub mod controls;
pub mod graph;
pub mod trace;
