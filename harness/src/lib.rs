//! Shared pieces for every problem crate.
//!
//! Inputs are generated from a seed so every machine benchmarks the same data.
//! `environment` describes the machine so a result can say where it came from.

pub mod data;
pub mod environment;
pub mod rng;
