//! # libcola
//!
//! Constraint-based graph layout library. Rust port of adaptagrams/libcola.
//!
//! ## Architecture
//!
//! - `vpsc` - Variable Placement with Separation Constraints solver (foundation)
//! - `cola` - Constrained force-directed layout engine

pub mod vpsc;
pub mod cola;

#[cfg(feature = "wasm")]
pub mod wasm;
