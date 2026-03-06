//! # vpsc - Variable Placement with Separation Constraints
//!
//! A solver for the quadratic programming problem of placing variables
//! at positions close to their desired positions while satisfying a set
//! of separation constraints.
//!
//! ## C++ ref: libvpsc (adaptagrams)
//!
//! ## Key types
//!
//! - [`Variable`] - A variable with desired position, weight, and scale.
//! - [`Constraint`] - A separation constraint: left + gap <= right (or ==).
//! - [`Solver`] - Static VPSC solver.
//! - [`IncSolver`] - Incremental VPSC solver (preferred for interactive use).
//! - [`Rectangle`] - A movable rectangle for overlap removal.

mod variable;
mod constraint;
mod block;
mod solver;
mod rectangle;

pub use variable::Variable;
pub use constraint::{Constraint, remove_redundant_equalities};
pub use solver::{Solver, IncSolver, VpscError};
pub use rectangle::{
    Dim, Rectangle,
    generate_x_constraints, generate_y_constraints,
    remove_overlaps, remove_overlaps_with_fixed,
    no_rectangle_overlaps,
};
