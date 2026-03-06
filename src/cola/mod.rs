//! # cola - Constrained force-directed layout
//!
//! Force-directed network layout using stress-majorization subject to
//! separation constraints.
//!
//! C++ ref: libcola

pub mod commondefs;
pub mod pseudorandom;
pub mod shortest_paths;
pub mod sparse_matrix;
pub mod conjugate_gradient;
pub mod convex_hull;
pub mod connected_components;
pub mod r#box;
pub mod compound_constraints;
pub mod cluster;
pub mod gradient_projection;
pub mod layout;

pub use commondefs::{NonOverlapConstraintsMode, FixedList};
pub use pseudorandom::PseudoRandom;
pub use shortest_paths::{floyd_warshall, dijkstra, johnsons, neighbours as adjacency_matrix};
pub use sparse_matrix::{SparseMap, SparseMatrix};
pub use conjugate_gradient::conjugate_gradient;
pub use convex_hull::convex_hull;
pub use connected_components::{Component, connected_components, separate_components};
pub use cluster::{Cluster, ClusterData, ClusterVar, RootCluster, RectangularCluster, ConvexCluster};
pub use layout::{
    ConstrainedFDLayout, Lock, Resize, DesiredPosition, PreIteration,
    TestConvergence, Connectivity, ProjectionResult,
    project_onto_ccs, solve_constraints,
};
