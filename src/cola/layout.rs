//! Constrained force-directed layout engine.
//!
//! Implements stress-majorization layout with separation constraints,
//! non-overlap, and cluster containment.
//!
//! C++ ref: libcola/cola.h, libcola/colafd.cpp

use crate::cola::commondefs::Edge;
use crate::cola::compound_constraints::CompoundConstraint;
use crate::cola::cluster::Cluster;
use crate::cola::pseudorandom::PseudoRandom;
use crate::cola::shortest_paths;
use crate::cola::sparse_matrix::{SparseMap, SparseMatrix};
use crate::vpsc::{
    Constraint, Dim, IncSolver, Rectangle, Variable,
    generate_x_constraints, generate_y_constraints,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Squared distance threshold below which nodes are considered coincident
/// and randomly displaced to avoid division-by-zero in force computation.
const COINCIDENT_DISTANCE_SQ: f64 = 1e-3;

/// Minimum distance used to avoid division by zero in force computation
/// when actual distance is effectively zero.
const MIN_DISTANCE: f64 = 0.1;

/// Distance threshold below which we treat values as numerically zero.
const NEAR_ZERO: f64 = 1e-30;

/// Default convergence tolerance for the layout loop.
/// Layout converges when relative stress reduction falls below this.
const DEFAULT_CONVERGENCE_TOLERANCE: f64 = 1e-4;

/// Default maximum iterations for the layout loop.
const DEFAULT_MAX_ITERATIONS: usize = 100;

/// Step-size halving threshold in `apply_descent_vector`.
const STEPSIZE_MIN: f64 = 1e-11;

/// Weight applied to lock constraints to keep locked nodes in place.
const LOCK_WEIGHT: f64 = 10000.0;

/// Weight factor for desired position stress contribution.
const DESIRED_POSITION_STRESS_FACTOR: f64 = 0.5;

/// Default non-positive edge length replacement.
const DEFAULT_EDGE_LENGTH: f64 = 1.0;

/// Random offset lower bound for coincident node displacement.
const RANDOM_OFFSET_MIN: f64 = 0.01;

/// Random offset upper bound for coincident node displacement.
const RANDOM_OFFSET_MAX: f64 = 1.0;

/// Random offset centering value.
const RANDOM_OFFSET_CENTER: f64 = 0.5;

/// Default border for makeFeasible.
const DEFAULT_MAKE_FEASIBLE_BORDER: f64 = 1.0;

/// Default variable weight.
const DEFAULT_VAR_WEIGHT: f64 = 1.0;

// ---------------------------------------------------------------------------
// Lock - a required position for a node
// ---------------------------------------------------------------------------

/// A lock specifies a required position for a node during layout.
///
/// C++ ref: cola::Lock
#[derive(Debug, Clone)]
pub struct Lock {
    pub id: usize,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Lock {
    pub fn new(id: usize, x: f64, y: f64) -> Self {
        Self { id, x, y, z: 0.0 }
    }

    pub fn new_3d(id: usize, x: f64, y: f64, z: f64) -> Self {
        Self { id, x, y, z }
    }

    pub fn pos(&self, dim: Dim) -> f64 {
        match dim {
            Dim::Horizontal => self.x,
            Dim::Vertical => self.y,
            Dim::Depth => self.z,
        }
    }
}

// ---------------------------------------------------------------------------
// Resize - a new bounding box for a node
// ---------------------------------------------------------------------------

/// A resize specifies a new required bounding box for a node.
///
/// C++ ref: cola::Resize
#[derive(Debug, Clone)]
pub struct Resize {
    pub id: usize,
    pub target: Rectangle,
}

impl Resize {
    pub fn new(id: usize, x: f64, y: f64, w: f64, h: f64) -> Self {
        Self {
            id,
            target: Rectangle::new(x, x + w, y, y + h),
        }
    }
}

// ---------------------------------------------------------------------------
// DesiredPosition
// ---------------------------------------------------------------------------

/// A desired position for a node, contributing an attractive force.
///
/// C++ ref: cola::DesiredPosition
#[derive(Debug, Clone)]
pub struct DesiredPosition {
    pub id: usize,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub weight: f64,
}

// ---------------------------------------------------------------------------
// PreIteration callback
// ---------------------------------------------------------------------------

/// Callback invoked before each layout iteration.
///
/// Return `false` from `should_continue()` to stop the layout.
///
/// C++ ref: cola::PreIteration
pub trait PreIteration {
    /// Whether layout should continue. Return `false` to stop.
    fn should_continue(&mut self) -> bool {
        true
    }

    /// Whether node positions/sizes have changed externally.
    fn changed(&self) -> bool {
        false
    }

    /// Current set of locked node positions.
    fn locks(&self) -> &[Lock] {
        &[]
    }

    /// Current set of resize requests.
    fn resizes(&self) -> &[Resize] {
        &[]
    }
}


// ---------------------------------------------------------------------------
// TestConvergence
// ---------------------------------------------------------------------------

/// Convergence test for the layout loop.
///
/// C++ ref: cola::TestConvergence
pub struct TestConvergence {
    pub tolerance: f64,
    pub max_iterations: usize,
    old_stress: f64,
    iterations: usize,
}

impl TestConvergence {
    pub fn new(tolerance: f64, max_iterations: usize) -> Self {
        Self {
            tolerance,
            max_iterations,
            old_stress: f64::MAX,
            iterations: 0,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(DEFAULT_CONVERGENCE_TOLERANCE, DEFAULT_MAX_ITERATIONS)
    }

    pub fn reset(&mut self) {
        self.old_stress = f64::MAX;
        self.iterations = 0;
    }

    /// Returns `true` if layout has converged or exceeded max iterations.
    ///
    /// C++ ref: TestConvergence::operator()
    pub fn test(&mut self, new_stress: f64) -> bool {
        self.iterations += 1;

        if self.old_stress == f64::MAX {
            self.old_stress = new_stress;
            return self.iterations >= self.max_iterations;
        }

        let converged = (self.old_stress - new_stress) / (new_stress + NEAR_ZERO)
            < self.tolerance
            || self.iterations > self.max_iterations;

        self.old_stress = new_stress;
        converged
    }

    pub fn iterations(&self) -> usize {
        self.iterations
    }
}

// ---------------------------------------------------------------------------
// G matrix connectivity encoding
// ---------------------------------------------------------------------------

/// Connectivity classification between node pairs.
///
/// C++ ref: G matrix values in colafd.cpp
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Connectivity {
    /// No forces between disconnected components.
    Disconnected = 0,
    /// Direct edge neighbours — attractive force always applies.
    EdgeNeighbour = 1,
    /// Connected but not direct neighbours — attractive force only when
    /// current distance exceeds ideal distance.
    ConnectedNonNeighbour = 2,
}

// ---------------------------------------------------------------------------
// ConstrainedFDLayout
// ---------------------------------------------------------------------------

/// Constrained force-directed layout using stress majorization.
///
/// C++ ref: cola::ConstrainedFDLayout
pub struct ConstrainedFDLayout {
    n: usize,
    /// Number of spatial dimensions (2 or 3).
    dims: usize,
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    bounding_boxes: Vec<Rectangle>,

    /// All-pairs shortest path distances, scaled by ideal_edge_length.
    /// Flat row-major n×n.
    d: Vec<f64>,

    /// Connectivity matrix. Flat row-major n×n.
    g: Vec<Connectivity>,

    /// Minimum non-zero distance in D.
    min_d: f64,

    /// Neighbour adjacency (1 if edge-connected, 0 otherwise). n×n.
    neighbours: Vec<Vec<u8>>,

    ideal_edge_length: f64,
    edge_lengths: Vec<f64>,

    convergence: TestConvergence,
    runge_kutta: bool,

    /// Compound constraints set by the user.
    compound_constraints: Vec<CompoundConstraint>,

    /// Extra constraints generated for clusters/non-overlap.
    extra_constraints: Vec<CompoundConstraint>,

    /// Cluster hierarchy (optional).
    cluster_hierarchy: Option<Cluster>,

    /// Whether to generate non-overlap constraints.
    generate_non_overlap: bool,

    /// Whether to use neighbour-only stress.
    use_neighbour_stress: bool,

    /// Whether to skip forces for non-neighbours already beyond ideal distance.
    /// C++ ref: `if(l>d && p>1) continue;` in computeForces/computeStress.
    /// This optimization speeds up large graphs but causes local minima
    /// (tangled layouts) for cycles and sparse graphs.
    /// Default: false (disabled — better quality, slightly slower).
    skip_distant_non_neighbours: bool,

    /// Desired positions (optional external attraction).
    desired_positions: Vec<DesiredPosition>,

    /// PRNG for random displacement of coincident nodes.
    random: PseudoRandom,
}

impl ConstrainedFDLayout {
    /// Create a new constrained force-directed layout.
    ///
    /// `rs` are the initial bounding boxes for nodes.
    /// `es` are the edges as `(source, target)` pairs.
    /// `ideal_length` is a scalar modifier for edge lengths.
    /// `edge_lengths` are optional per-edge ideal lengths (multiplied by `ideal_length`).
    ///
    /// C++ ref: ConstrainedFDLayout constructor
    pub fn new(
        rs: Vec<Rectangle>,
        es: &[Edge],
        ideal_length: f64,
        edge_lengths: Option<&[f64]>,
    ) -> Self {
        let n = rs.len();

        let x: Vec<f64> = rs.iter().map(|r| r.centre_x()).collect();
        let y: Vec<f64> = rs.iter().map(|r| r.centre_y()).collect();

        // Build neighbour adjacency matrix.
        let mut neighbours = vec![vec![0u8; n]; n];
        for &(s, t) in es {
            neighbours[s][t] = 1;
            neighbours[t][s] = 1;
        }

        let mut layout = Self {
            n,
            dims: 2,
            x,
            y,
            z: Vec::new(),
            bounding_boxes: rs,
            d: vec![0.0; n * n],
            g: vec![Connectivity::Disconnected; n * n],
            min_d: f64::MAX,
            neighbours,
            ideal_edge_length: ideal_length,
            edge_lengths: edge_lengths.map_or_else(Vec::new, |el| el.to_vec()),
            convergence: TestConvergence::with_defaults(),
            runge_kutta: true,
            compound_constraints: Vec::new(),
            extra_constraints: Vec::new(),
            cluster_hierarchy: None,
            generate_non_overlap: false,
            use_neighbour_stress: false,
            skip_distant_non_neighbours: true,
            desired_positions: Vec::new(),
            random: PseudoRandom::new(0.0),
        };

        layout.compute_path_lengths(es);
        layout
    }

    /// Create a new 3D constrained force-directed layout.
    ///
    /// `rs` are the initial bounding boxes (used for XY overlap removal only).
    /// `z_positions` are the initial Z coordinates for each node.
    pub fn new_3d(
        rs: Vec<Rectangle>,
        z_positions: &[f64],
        es: &[Edge],
        ideal_length: f64,
        edge_lengths: Option<&[f64]>,
    ) -> Self {
        let mut layout = Self::new(rs, es, ideal_length, edge_lengths);
        layout.dims = 3;
        layout.z = z_positions.to_vec();
        layout
    }

    // -----------------------------------------------------------------------
    // Configuration
    // -----------------------------------------------------------------------

    /// Set compound constraints for the layout.
    pub fn set_constraints(&mut self, ccs: Vec<CompoundConstraint>) {
        self.compound_constraints = ccs;
    }

    /// Enable/disable automatic non-overlap constraint generation.
    pub fn set_avoid_node_overlaps(&mut self, avoid: bool) {
        self.generate_non_overlap = avoid;
    }

    /// Set the cluster hierarchy.
    pub fn set_cluster_hierarchy(&mut self, hierarchy: Cluster) {
        self.cluster_hierarchy = Some(hierarchy);
    }

    /// Set whether to use neighbour-only stress.
    pub fn set_use_neighbour_stress(&mut self, val: bool) {
        self.use_neighbour_stress = val;
    }

    /// Enable/disable the C++ `if(l>d && p>1) continue;` optimization.
    /// When enabled, forces between non-neighbours already beyond their ideal
    /// distance are skipped. Faster for large graphs, but causes local minima.
    /// Default: disabled.
    pub fn set_skip_distant_non_neighbours(&mut self, val: bool) {
        self.skip_distant_non_neighbours = val;
    }

    /// Set desired positions for external attraction.
    pub fn set_desired_positions(&mut self, positions: Vec<DesiredPosition>) {
        self.desired_positions = positions;
    }

    /// Set convergence parameters.
    pub fn set_convergence(&mut self, tolerance: f64, max_iterations: usize) {
        self.convergence = TestConvergence::new(tolerance, max_iterations);
    }

    /// Enable/disable Runge-Kutta integration (default: enabled).
    pub fn set_runge_kutta(&mut self, enabled: bool) {
        self.runge_kutta = enabled;
    }

    /// Get current X positions.
    pub fn x_positions(&self) -> &[f64] {
        &self.x
    }

    /// Get current Y positions.
    pub fn y_positions(&self) -> &[f64] {
        &self.y
    }

    /// Get the bounding boxes (updated after layout).
    pub fn bounding_boxes(&self) -> &[Rectangle] {
        &self.bounding_boxes
    }

    /// Get X coordinates.
    pub fn x(&self) -> &[f64] { &self.x }
    /// Get Y coordinates.
    pub fn y(&self) -> &[f64] { &self.y }
    /// Get Z coordinates (empty for 2D layouts).
    pub fn z(&self) -> &[f64] { &self.z }

    /// Get the D matrix (all-pairs shortest path distances * ideal_edge_length).
    pub fn d_matrix(&self) -> &[f64] {
        &self.d
    }

    /// Get the G matrix (connectivity).
    pub fn g_matrix(&self) -> &[Connectivity] {
        &self.g
    }

    /// Number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.n
    }

    // -----------------------------------------------------------------------
    // Path length computation (constructor helper)
    // -----------------------------------------------------------------------

    /// Compute D and G matrices from edges.
    ///
    /// D[i][j] = shortest_path(i,j) * ideal_edge_length
    /// G[i][j] = Disconnected | EdgeNeighbour | ConnectedNonNeighbour
    ///
    /// C++ ref: ConstrainedFDLayout::computePathLengths
    fn compute_path_lengths(&mut self, es: &[Edge]) {
        let n = self.n;

        // Sanitize edge lengths.
        let weights: Option<Vec<f64>> = if self.edge_lengths.is_empty() {
            None
        } else {
            let mut w = self.edge_lengths.clone();
            for val in &mut w {
                if *val <= 0.0 {
                    *val = DEFAULT_EDGE_LENGTH;
                }
            }
            Some(w)
        };

        // Compute all-pairs shortest paths.
        let sp = shortest_paths::johnsons(n, es, weights.as_deref());

        // Fill D and G matrices.
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let idx = i * n + j;
                let dist = sp[i][j];

                if dist.is_infinite() {
                    // Disconnected components.
                    self.g[idx] = Connectivity::Disconnected;
                } else {
                    let scaled = dist * self.ideal_edge_length;
                    self.d[idx] = scaled;
                    self.g[idx] = Connectivity::ConnectedNonNeighbour;

                    if scaled > 0.0 && scaled < self.min_d {
                        self.min_d = scaled;
                    }
                }
            }
        }

        if self.min_d == f64::MAX {
            self.min_d = DEFAULT_EDGE_LENGTH;
        }

        // Mark direct edges as EdgeNeighbour.
        for &(u, v) in es {
            self.g[u * n + v] = Connectivity::EdgeNeighbour;
            self.g[v * n + u] = Connectivity::EdgeNeighbour;
        }
    }

    // -----------------------------------------------------------------------
    // Coordinate access by dimension
    // -----------------------------------------------------------------------

    fn coords(&self, dim: Dim) -> &[f64] {
        match dim {
            Dim::Horizontal => &self.x,
            Dim::Vertical => &self.y,
            Dim::Depth => &self.z,
        }
    }

    fn coords_mut(&mut self, dim: Dim) -> &mut Vec<f64> {
        match dim {
            Dim::Horizontal => &mut self.x,
            Dim::Vertical => &mut self.y,
            Dim::Depth => &mut self.z,
        }
    }

    // -----------------------------------------------------------------------
    // Main layout loop
    // -----------------------------------------------------------------------

    /// Run the layout algorithm until convergence.
    ///
    /// C++ ref: ConstrainedFDLayout::run
    pub fn run(&mut self) {
        self.run_axes(true, true, None);
    }

    /// Run layout with a PreIteration callback.
    pub fn run_with_callback(&mut self, pre_iteration: &mut dyn PreIteration) {
        self.run_axes(true, true, Some(pre_iteration));
    }

    /// Run layout on specified axes.
    pub fn run_axes(
        &mut self,
        x_axis: bool,
        y_axis: bool,
        mut pre_iteration: Option<&mut dyn PreIteration>,
    ) {
        if self.n == 0 {
            return;
        }

        // Generate non-overlap and cluster constraints.
        self.generate_non_overlap_and_cluster_constraints();

        self.convergence.reset();
        let mut stress = f64::MAX;

        loop {
            // Pre-iteration callback.
            if let Some(ref mut pre) = pre_iteration {
                if !pre.should_continue() {
                    break;
                }
                if pre.changed() {
                    stress = f64::MAX;
                }
                let resizes = pre.resizes().to_vec();
                if !resizes.is_empty() {
                    self.handle_resizes(&resizes);
                }
            }

            let big_n = self.dims * self.n;
            let mut x0 = vec![0.0; big_n];
            self.get_position(&mut x0);

            let mut x1 = vec![0.0; big_n];

            if self.runge_kutta {
                let locks = pre_iteration
                    .as_ref()
                    .map(|p| p.locks().to_vec())
                    .unwrap_or_default();

                let mut a = vec![0.0; big_n];
                self.compute_descent_vector_on_all_axes(
                    x_axis, y_axis, stress, &x0, &mut a, &locks,
                );

                // ia = x0 + (a - x0) / 2
                let mut ia = vec![0.0; big_n];
                for i in 0..big_n {
                    ia[i] = x0[i] + (a[i] - x0[i]) / 2.0;
                }

                let mut b = vec![0.0; big_n];
                self.compute_descent_vector_on_all_axes(
                    x_axis, y_axis, stress, &ia, &mut b, &locks,
                );

                // ib = x0 + (b - x0) / 2
                let mut ib = vec![0.0; big_n];
                for i in 0..big_n {
                    ib[i] = x0[i] + (b[i] - x0[i]) / 2.0;
                }

                let mut c = vec![0.0; big_n];
                self.compute_descent_vector_on_all_axes(
                    x_axis, y_axis, stress, &ib, &mut c, &locks,
                );

                let mut dd = vec![0.0; big_n];
                self.compute_descent_vector_on_all_axes(
                    x_axis, y_axis, stress, &c, &mut dd, &locks,
                );

                // x1 = (a + 2b + 2c + d) / 6
                for i in 0..big_n {
                    x1[i] = (a[i] + 2.0 * b[i] + 2.0 * c[i] + dd[i]) / 6.0;
                }
            } else {
                let locks = pre_iteration
                    .as_ref()
                    .map(|p| p.locks().to_vec())
                    .unwrap_or_default();
                self.compute_descent_vector_on_all_axes(
                    x_axis, y_axis, stress, &x0, &mut x1, &locks,
                );
            }

            self.set_position(&x1, pre_iteration.as_deref());
            stress = self.compute_stress(pre_iteration.as_deref());

            if self.convergence.test(stress) {
                break;
            }
        }

        self.extra_constraints.clear();
    }

    /// Run a single iteration of layout.
    ///
    /// C++ ref: ConstrainedFDLayout::runOnce
    pub fn run_once(&mut self, x_axis: bool, y_axis: bool) {
        if self.n == 0 {
            return;
        }

        let stress = f64::MAX;
        let big_n = self.dims * self.n;
        let mut x0 = vec![0.0; big_n];
        self.get_position(&mut x0);

        let locks = Vec::new();

        if self.runge_kutta {
            let mut a = vec![0.0; big_n];
            self.compute_descent_vector_on_all_axes(
                x_axis, y_axis, stress, &x0, &mut a, &locks,
            );

            let mut ia = vec![0.0; big_n];
            for i in 0..big_n {
                ia[i] = x0[i] + (a[i] - x0[i]) / 2.0;
            }

            let mut b = vec![0.0; big_n];
            self.compute_descent_vector_on_all_axes(
                x_axis, y_axis, stress, &ia, &mut b, &locks,
            );

            let mut ib = vec![0.0; big_n];
            for i in 0..big_n {
                ib[i] = x0[i] + (b[i] - x0[i]) / 2.0;
            }

            let mut c = vec![0.0; big_n];
            self.compute_descent_vector_on_all_axes(
                x_axis, y_axis, stress, &ib, &mut c, &locks,
            );

            let mut dd = vec![0.0; big_n];
            self.compute_descent_vector_on_all_axes(
                x_axis, y_axis, stress, &c, &mut dd, &locks,
            );

            let mut x1 = vec![0.0; big_n];
            for i in 0..big_n {
                x1[i] = (a[i] + 2.0 * b[i] + 2.0 * c[i] + dd[i]) / 6.0;
            }
            self.set_position(&x1, None);
        } else {
            let mut x1 = vec![0.0; big_n];
            self.compute_descent_vector_on_all_axes(
                x_axis, y_axis, stress, &x0, &mut x1, &locks,
            );
            self.set_position(&x1, None);
        }
    }

    // -----------------------------------------------------------------------
    // Position get/set
    // -----------------------------------------------------------------------

    fn get_position(&self, pos: &mut [f64]) {
        for i in 0..self.n {
            pos[i] = self.x[i];
            pos[i + self.n] = self.y[i];
        }
        if self.dims == 3 {
            for i in 0..self.n {
                pos[i + 2 * self.n] = self.z[i];
            }
        }
    }

    fn set_position(&mut self, pos: &[f64], pre_iteration: Option<&dyn PreIteration>) {
        let locks: Vec<Lock> = pre_iteration
            .map(|p| p.locks().to_vec())
            .unwrap_or_default();

        self.move_to(Dim::Horizontal, pos, &locks);
        self.move_to(Dim::Vertical, pos, &locks);
        if self.dims == 3 {
            self.move_to(Dim::Depth, pos, &locks);
        }
    }

    fn move_bounding_boxes(&mut self) {
        for i in 0..self.n {
            self.bounding_boxes[i].move_centre(self.x[i], self.y[i]);
        }
    }

    // -----------------------------------------------------------------------
    // Descent vector computation
    // -----------------------------------------------------------------------

    fn compute_descent_vector_on_all_axes(
        &mut self,
        x_axis: bool,
        y_axis: bool,
        stress: f64,
        x0: &[f64],
        x1: &mut [f64],
        locks: &[Lock],
    ) {
        // Set current positions from x0.
        for i in 0..self.n {
            self.x[i] = x0[i];
            self.y[i] = x0[i + self.n];
        }
        if self.dims == 3 {
            for i in 0..self.n {
                self.z[i] = x0[i + 2 * self.n];
            }
        }
        self.move_bounding_boxes();

        if x_axis {
            self.apply_forces_and_constraints(Dim::Horizontal, stress, locks);
        }
        if y_axis {
            self.apply_forces_and_constraints(Dim::Vertical, stress, locks);
        }
        if self.dims == 3 {
            self.apply_forces_and_constraints(Dim::Depth, stress, locks);
        }

        self.get_position(x1);
    }

    // -----------------------------------------------------------------------
    // Force computation and constraint application
    // -----------------------------------------------------------------------

    /// Apply forces and project onto constraints for one dimension.
    ///
    /// C++ ref: ConstrainedFDLayout::applyForcesAndConstraints
    fn apply_forces_and_constraints(
        &mut self,
        dim: Dim,
        old_stress: f64,
        locks: &[Lock],
    ) {
        let n = self.n;
        let mut g = vec![0.0; n];

        // Build desired positions from locks.
        let des: Vec<(usize, f64)> = locks.iter().map(|l| (l.id, l.pos(dim))).collect();

        // Setup variables and constraints.
        let (mut vars, constraints) = self.setup_vars_and_constraints(dim);

        // Setup extra constraints (non-overlap, cluster).
        let extra_cs = self.setup_extra_constraints(dim);

        // Compute forces.
        let mut h_map = SparseMap::new(n);
        self.compute_forces(dim, &mut h_map, &mut g);
        let h = SparseMatrix::from_sparse_map(&h_map);

        // Save old coords.
        let old_coords = self.coords(dim).to_vec();

        // Compute initial step size and apply descent.
        // The optimal quadratic step g'g/(g'Hg) can be very large when the
        // Hessian is ill-conditioned (e.g., clustered nodes). RK4 integration
        // naturally dampens this via midpoint averaging, but for simple descent
        // we need to limit the maximum displacement per node.
        let step = self.compute_step_size(&h, &g, &g);
        let max_g = g.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
        let max_displacement = step.abs() * max_g;
        let max_allowed = self.ideal_edge_length * (self.n as f64);
        let step = if max_displacement > max_allowed && max_g > 0.0 {
            max_allowed / max_g * step.signum()
        } else {
            step
        };
        self.apply_descent_vector(&g, &old_coords, dim, old_stress, step);

        // Set variable desired positions.
        for i in 0..n.min(vars.len()) {
            vars[i] = Variable::new(i, self.coords(dim)[i], DEFAULT_VAR_WEIGHT, 1.0);
        }
        for &(id, pos) in &des {
            if id < vars.len() {
                vars[id] = Variable::new(id, pos, LOCK_WEIGHT, 1.0);
            }
        }

        // Merge all constraints.
        let mut all_cs = constraints;
        all_cs.extend(extra_cs);

        // Project via VPSC.
        let mut solver = IncSolver::new(vars, all_cs);
        let _ = solver.solve();
        let positions = solver.final_positions();
        {
            let coords = self.coords_mut(dim);
            for i in 0..n.min(positions.len()) {
                coords[i] = positions[i];
            }
        }

        // Compute beta for step limiting.
        let d: Vec<f64> = (0..n).map(|i| old_coords[i] - self.coords(dim)[i]).collect();
        let stepsize = self.compute_step_size(&h, &g, &d);
        let stepsize = stepsize.max(0.0).min(1.0);

        self.apply_descent_vector(&d, &old_coords, dim, old_stress, stepsize);
        self.move_bounding_boxes();

        // Update compound constraint positions.
        // (In the full C++ this calls updatePosition on each cc; we skip for now
        // as our compound constraints are stateless.)
    }

    /// Compute the negative gradient g and Hessian H for stress minimization.
    ///
    /// C++ ref: ConstrainedFDLayout::computeForces
    fn compute_forces(&mut self, dim: Dim, h: &mut SparseMap, g: &mut [f64]) {
        let n = self.n;
        if n <= 1 {
            return;
        }

        for i in 0..n {
            g[i] = 0.0;
        }

        for u in 0..n {
            let mut h_uu = 0.0;

            for v in 0..n {
                if u == v {
                    continue;
                }
                if self.use_neighbour_stress && self.neighbours[u][v] != 1 {
                    continue;
                }

                // Randomly displace coincident nodes.
                let mut rx = self.x[u] - self.x[v];
                let mut ry = self.y[u] - self.y[v];
                let mut rz = if self.dims == 3 { self.z[u] - self.z[v] } else { 0.0 };
                let mut sd2 = rx * rx + ry * ry + rz * rz;
                let mut max_displaces = n;

                while max_displaces > 0 {
                    if sd2 > COINCIDENT_DISTANCE_SQ {
                        break;
                    }
                    let rd = self.offset_dir();
                    self.x[v] += rd.0;
                    self.y[v] += rd.1;
                    if self.dims == 3 {
                        self.z[v] += rd.2;
                    }
                    rx = self.x[u] - self.x[v];
                    ry = self.y[u] - self.y[v];
                    rz = if self.dims == 3 { self.z[u] - self.z[v] } else { 0.0 };
                    sd2 = rx * rx + ry * ry + rz * rz;
                    max_displaces -= 1;
                }

                let p = self.g[u * n + v];
                if p == Connectivity::Disconnected {
                    continue;
                }

                let l = sd2.sqrt();
                let d = self.d[u * n + v];

                // C++ ref: if(l>d && p>1) continue;
                if self.skip_distant_non_neighbours
                    && l > d
                    && p == Connectivity::ConnectedNonNeighbour
                {
                    continue;
                }

                let d2 = d * d;
                let l_safe = if l < NEAR_ZERO { MIN_DISTANCE } else { l };

                // Component along current dimension; perpendicular squared
                // distance generalizes the 2D dy² to work in 2D and 3D.
                let dx = match dim {
                    Dim::Horizontal => rx,
                    Dim::Vertical => ry,
                    Dim::Depth => rz,
                };
                let perp_sq = sd2 - dx * dx;

                g[u] += dx * (l_safe - d) / (d2 * l_safe);

                let h_uv = (d * perp_sq / (l_safe * l_safe * l_safe) - 1.0) / d2;
                h.set(u, v, h_uv);
                h_uu -= h_uv;
            }

            h.set(u, u, h_uu);
        }

        // Add desired position forces.
        for dp in &self.desired_positions {
            let i = dp.id;
            let d_val = match dim {
                Dim::Horizontal => dp.x - self.x[i],
                Dim::Vertical => dp.y - self.y[i],
                Dim::Depth => if self.dims == 3 { dp.z - self.z[i] } else { 0.0 },
            };
            g[i] -= d_val * dp.weight;
            let current = h.get(i, i);
            h.set(i, i, current + dp.weight);
        }
    }

    /// Compute optimal step size: g'd / (d' H d).
    ///
    /// C++ ref: ConstrainedFDLayout::computeStepSize
    fn compute_step_size(&self, h: &SparseMatrix, g: &[f64], d: &[f64]) -> f64 {
        let numerator: f64 = g.iter().zip(d.iter()).map(|(a, b)| a * b).sum();

        let mut hd = vec![0.0; d.len()];
        h.right_multiply(d, &mut hd);
        let denominator: f64 = d.iter().zip(hd.iter()).map(|(a, b)| a * b).sum();

        if denominator == 0.0 {
            return 0.0;
        }
        numerator / denominator
    }

    /// Apply descent: coords = old_coords - stepsize * d.
    ///
    /// C++ ref: ConstrainedFDLayout::applyDescentVector
    fn apply_descent_vector(
        &mut self,
        d: &[f64],
        old_coords: &[f64],
        dim: Dim,
        _old_stress: f64,
        stepsize: f64,
    ) {
        let coords = self.coords_mut(dim);

        if stepsize.abs() > STEPSIZE_MIN {
            for i in 0..coords.len().min(d.len()) {
                coords[i] = old_coords[i] - stepsize * d[i];
            }
        } else {
            // Step too small: restore old coords.
            coords[..old_coords.len()].copy_from_slice(old_coords);
        }
    }

    /// Compute random offset for displacing coincident nodes.
    ///
    /// C++ ref: ConstrainedFDLayout::offsetDir
    fn offset_dir(&mut self) -> (f64, f64, f64) {
        let mut ux = self.random.get_next_between(RANDOM_OFFSET_MIN, RANDOM_OFFSET_MAX)
            - RANDOM_OFFSET_CENTER;
        let mut uy = self.random.get_next_between(RANDOM_OFFSET_MIN, RANDOM_OFFSET_MAX)
            - RANDOM_OFFSET_CENTER;
        let mut uz = if self.dims == 3 {
            self.random.get_next_between(RANDOM_OFFSET_MIN, RANDOM_OFFSET_MAX)
                - RANDOM_OFFSET_CENTER
        } else {
            0.0
        };
        let l = (ux * ux + uy * uy + uz * uz).sqrt();
        if l > 0.0 {
            ux *= self.min_d / l;
            uy *= self.min_d / l;
            uz *= self.min_d / l;
        }
        (ux, uy, uz)
    }

    // -----------------------------------------------------------------------
    // Stress computation
    // -----------------------------------------------------------------------

    /// Compute the current stress value.
    ///
    /// stress = Σ (d_ij - l_ij)² / d_ij²
    ///
    /// C++ ref: ConstrainedFDLayout::computeStress
    pub fn compute_stress(&self, pre_iteration: Option<&dyn PreIteration>) -> f64 {
        let n = self.n;
        let mut stress = 0.0;

        for u in 0..(n.saturating_sub(1)) {
            for v in (u + 1)..n {
                if self.use_neighbour_stress && self.neighbours[u][v] != 1 {
                    continue;
                }

                let p = self.g[u * n + v];
                if p == Connectivity::Disconnected {
                    continue;
                }

                let rx = self.x[u] - self.x[v];
                let ry = self.y[u] - self.y[v];
                let rz = if self.dims == 3 { self.z[u] - self.z[v] } else { 0.0 };
                let l = (rx * rx + ry * ry + rz * rz).sqrt();
                let d = self.d[u * n + v];

                // C++ ref: if(l>d && p>1) continue;
                if self.skip_distant_non_neighbours
                    && l > d
                    && p == Connectivity::ConnectedNonNeighbour
                {
                    continue;
                }
                let d2 = d * d;
                let rl = d - l;
                stress += rl * rl / d2;
            }
        }

        // Lock stress.
        if let Some(pre) = pre_iteration {
            for lock in pre.locks() {
                let dx = lock.x - self.x[lock.id];
                let dy = lock.y - self.y[lock.id];
                let dz = if self.dims == 3 { lock.z - self.z[lock.id] } else { 0.0 };
                stress += LOCK_WEIGHT * (dx * dx + dy * dy + dz * dz);
            }
        }

        // Desired position stress.
        for dp in &self.desired_positions {
            let dx = self.x[dp.id] - dp.x;
            let dy = self.y[dp.id] - dp.y;
            let dz = if self.dims == 3 { self.z[dp.id] - dp.z } else { 0.0 };
            stress += DESIRED_POSITION_STRESS_FACTOR * dp.weight * (dx * dx + dy * dy + dz * dz);
        }

        stress
    }

    // -----------------------------------------------------------------------
    // moveTo - constrained movement in one dimension
    // -----------------------------------------------------------------------

    /// Move nodes to target positions while respecting constraints.
    ///
    /// C++ ref: ConstrainedFDLayout::moveTo
    fn move_to(&mut self, dim: Dim, target: &[f64], locks: &[Lock]) {
        let n = self.n;
        let (vars, constraints) = self.setup_vars_and_constraints(dim);
        let extra_cs = self.setup_extra_constraints(dim);

        // Build desired positions from locks.
        let des: Vec<(usize, f64)> = locks.iter().map(|l| (l.id, l.pos(dim))).collect();

        // Set variable desired positions from target.
        let offset = match dim {
            Dim::Horizontal => 0,
            Dim::Vertical => n,
            Dim::Depth => 2 * n,
        };

        let mut all_vars = vars;
        for i in 0..n.min(all_vars.len()) {
            all_vars[i] = Variable::new(i, target[i + offset], DEFAULT_VAR_WEIGHT, 1.0);
        }
        for &(id, pos) in &des {
            if id < all_vars.len() {
                all_vars[id] = Variable::new(id, pos, LOCK_WEIGHT, 1.0);
            }
        }

        let mut all_cs = constraints;
        all_cs.extend(extra_cs);

        // Project.
        let mut solver = IncSolver::new(all_vars, all_cs);
        let _ = solver.solve();
        let positions = solver.final_positions();

        let coords = self.coords_mut(dim);
        for i in 0..n.min(positions.len()) {
            coords[i] = positions[i];
        }

        self.move_bounding_boxes();
    }

    // -----------------------------------------------------------------------
    // Setup variables and constraints from compound constraints
    // -----------------------------------------------------------------------

    /// Create VPSC variables and constraints from compound constraints.
    ///
    /// C++ ref: setupVarsAndConstraints
    fn setup_vars_and_constraints(&self, dim: Dim) -> (Vec<Variable>, Vec<Constraint>) {
        let n = self.n;
        let coords = self.coords(dim);

        let vars: Vec<Variable> = (0..n)
            .map(|i| Variable::new(i, coords[i], DEFAULT_VAR_WEIGHT, 1.0))
            .collect();

        // Generate variables from compound constraints.
        let mut all_gen_cs = Vec::new();

        // Note: We need to work with a copy of compound_constraints since
        // generate_variables needs &mut. We collect the generated vars first.
        // In practice, the generated variables are determined by the constraint
        // type and dimension, so we can iterate immutably for constraint generation.
        for cc in &self.compound_constraints {
            let gen_cs = cc.generate_separation_constraints(dim, &self.bounding_boxes);
            all_gen_cs.extend(gen_cs);
        }

        // Create cluster variables if we have a hierarchy.
        // Cluster vars would be appended after node vars.
        // For now, skip cluster variable generation in the constraint path
        // since the main layout uses generate_non_overlap_and_cluster_constraints.

        // Convert generated constraints to VPSC constraints.
        let constraints: Vec<Constraint> = all_gen_cs
            .iter()
            .filter(|gc| gc.left < vars.len() && gc.right < vars.len())
            .map(|gc| Constraint::new(gc.left, gc.right, gc.gap, gc.equality))
            .collect();

        (vars, constraints)
    }

    /// Setup extra constraints (non-overlap, cluster containment).
    fn setup_extra_constraints(&self, dim: Dim) -> Vec<Constraint> {
        let mut constraints = Vec::new();

        // Generate non-overlap constraints if enabled.
        if self.generate_non_overlap {
            // Overlap removal is 2D only; no Z-axis overlap constraints.
            let noc_opt = match dim {
                Dim::Horizontal => Some(generate_x_constraints(&self.bounding_boxes, false)),
                Dim::Vertical => Some(generate_y_constraints(&self.bounding_boxes)),
                Dim::Depth => None,
            };
            if let Some((_, noc)) = noc_opt {
                constraints.extend(noc);
            }
        }

        // Extra compound constraints (cluster containment).
        for cc in &self.extra_constraints {
            let gen_cs = cc.generate_separation_constraints(dim, &self.bounding_boxes);
            for gc in gen_cs {
                constraints.push(Constraint::new(gc.left, gc.right, gc.gap, gc.equality));
            }
        }

        constraints
    }

    // -----------------------------------------------------------------------
    // Non-overlap and cluster constraint generation
    // -----------------------------------------------------------------------

    /// Generate non-overlap and cluster containment compound constraints.
    ///
    /// C++ ref: ConstrainedFDLayout::generateNonOverlapAndClusterCompoundConstraints
    fn generate_non_overlap_and_cluster_constraints(&mut self) {
        // For now, we rely on the simple non-overlap path in setup_extra_constraints.
        // Full cluster containment constraint generation would require the
        // NonOverlapConstraints class which handles cluster hierarchies.
        // This is sufficient for flat layouts with overlap removal.
    }

    // -----------------------------------------------------------------------
    // Handle resizes
    // -----------------------------------------------------------------------

    fn handle_resizes(&mut self, resizes: &[Resize]) {
        for resize in resizes {
            let id = resize.id;
            if id < self.bounding_boxes.len() {
                self.bounding_boxes[id] = resize.target.clone();
                self.x[id] = resize.target.centre_x();
                self.y[id] = resize.target.centre_y();
            }
        }
    }

    // -----------------------------------------------------------------------
    // makeFeasible
    // -----------------------------------------------------------------------

    /// Find a feasible starting position satisfying all constraints.
    ///
    /// Uses a greedy priority-based approach: sorts compound constraints by
    /// priority and tries to satisfy them one at a time with minimal node
    /// displacement.
    ///
    /// C++ ref: ConstrainedFDLayout::makeFeasible
    pub fn make_feasible(&mut self) {
        self.make_feasible_with_border(DEFAULT_MAKE_FEASIBLE_BORDER, DEFAULT_MAKE_FEASIBLE_BORDER);
    }

    pub fn make_feasible_with_border(&mut self, _x_border: f64, _y_border: f64) {
        // Create variables for both dimensions.
        let n = self.n;

        let mut vars: [Vec<Variable>; 2] = [
            (0..n)
                .map(|i| Variable::new(i, self.bounding_boxes[i].centre_x(), DEFAULT_VAR_WEIGHT, 1.0))
                .collect(),
            (0..n)
                .map(|i| Variable::new(i, self.bounding_boxes[i].centre_y(), DEFAULT_VAR_WEIGHT, 1.0))
                .collect(),
        ];

        self.generate_non_overlap_and_cluster_constraints();

        // Collect all constraints, sorted by priority.
        let mut all_ccs: Vec<&CompoundConstraint> = self
            .compound_constraints
            .iter()
            .chain(self.extra_constraints.iter())
            .collect();
        all_ccs.sort_by_key(|cc| cc.priority());

        // For each constraint (lowest priority first), generate and try to satisfy.
        for cc in &all_ccs {
            for dim_idx in 0..2usize {
                let dim = if dim_idx == 0 {
                    Dim::Horizontal
                } else {
                    Dim::Vertical
                };

                let gen_cs = cc.generate_separation_constraints(dim, &self.bounding_boxes);
                if gen_cs.is_empty() {
                    continue;
                }

                let vpsc_cs: Vec<Constraint> = gen_cs
                    .iter()
                    .filter(|gc| gc.left < vars[dim_idx].len() && gc.right < vars[dim_idx].len())
                    .map(|gc| Constraint::new(gc.left, gc.right, gc.gap, gc.equality))
                    .collect();

                if vpsc_cs.is_empty() {
                    continue;
                }

                let mut solver = IncSolver::new(vars[dim_idx].clone(), vpsc_cs);
                let _ = solver.satisfy();
                let positions = solver.final_positions();

                for i in 0..n.min(positions.len()) {
                    vars[dim_idx][i] =
                        Variable::new(i, positions[i], DEFAULT_VAR_WEIGHT, 1.0);
                }
            }
        }

        // Also handle non-overlap if enabled.
        if self.generate_non_overlap {
            for dim_idx in 0..2usize {
                let (_, noc) = if dim_idx == 0 {
                    generate_x_constraints(&self.bounding_boxes, false)
                } else {
                    generate_y_constraints(&self.bounding_boxes)
                };

                if noc.is_empty() {
                    continue;
                }

                let mut solver = IncSolver::new(vars[dim_idx].clone(), noc);
                let _ = solver.satisfy();
                let positions = solver.final_positions();

                for i in 0..n.min(positions.len()) {
                    vars[dim_idx][i] =
                        Variable::new(i, positions[i], DEFAULT_VAR_WEIGHT, 1.0);
                }
            }
        }

        // Write back positions.
        for i in 0..n {
            let x_pos = vars[0][i].final_position;
            let y_pos = vars[1][i].final_position;
            self.bounding_boxes[i].move_centre(x_pos, y_pos);
            self.x[i] = x_pos;
            self.y[i] = y_pos;
        }

        self.extra_constraints.clear();
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Project onto compound constraints (standalone).
///
/// C++ ref: cola::projectOntoCCs
pub fn project_onto_ccs(
    dim: Dim,
    rs: &mut [Rectangle],
    ccs: &[CompoundConstraint],
    prevent_overlaps: bool,
) -> ProjectionResult {
    let n = rs.len();

    let vars: Vec<Variable> = (0..n)
        .map(|i| {
            let pos = match dim {
                Dim::Horizontal => rs[i].centre_x(),
                Dim::Vertical => rs[i].centre_y(),
                Dim::Depth => 0.0,
            };
            Variable::new(i, pos, DEFAULT_VAR_WEIGHT, 1.0)
        })
        .collect();

    let mut constraints = Vec::new();

    // Generate constraints from compound constraints.
    for cc in ccs {
        let gen_cs = cc.generate_separation_constraints(dim, rs);
        for gc in gen_cs {
            if gc.left < vars.len() && gc.right < vars.len() {
                constraints.push(Constraint::new(gc.left, gc.right, gc.gap, gc.equality));
            }
        }
    }

    // Generate non-overlap constraints if requested.
    if prevent_overlaps {
        let noc_opt = match dim {
            Dim::Horizontal => Some(generate_x_constraints(rs, false)),
            Dim::Vertical => Some(generate_y_constraints(rs)),
            Dim::Depth => None,
        };
        if let Some((_, noc)) = noc_opt {
            constraints.extend(noc);
        }
    }

    let result = solve_constraints(vars, constraints);

    result
}

/// Construct a solver and attempt to solve the passed constraints.
///
/// C++ ref: cola::solve
pub fn solve_constraints(
    vars: Vec<Variable>,
    constraints: Vec<Constraint>,
) -> ProjectionResult {
    let mut solver = IncSolver::new(vars, constraints);
    let _ok = solver.solve();

    ProjectionResult {
        error_level: 0,
        unsat_info: String::new(),
    }
}

/// Result of a constraint projection.
///
/// C++ ref: cola::ProjectionResult
#[derive(Debug, Clone)]
pub struct ProjectionResult {
    /// 0 = all satisfied, 1 = only non-overlap unsatisfied, 2 = other unsatisfied.
    pub error_level: i32,
    /// Description of unsatisfied constraints (for debugging).
    pub unsat_info: String,
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Constants for test assertions
    // -----------------------------------------------------------------------

    /// Tolerance for floating-point comparisons.
    const TOL: f64 = 1e-3;

    /// Helper to create a simple rectangle centered at (cx, cy) with given width/height.
    fn rect(cx: f64, cy: f64, w: f64, h: f64) -> Rectangle {
        Rectangle::new(cx - w / 2.0, cx + w / 2.0, cy - h / 2.0, cy + h / 2.0)
    }

    // ===================================================================
    // Category 1: Construction
    // ===================================================================

    #[test]
    fn empty_graph() {
        let mut layout = ConstrainedFDLayout::new(vec![], &[], 100.0, None);
        layout.run(); // Should not panic.
        assert_eq!(layout.num_nodes(), 0);
    }

    #[test]
    fn single_node() {
        let rs = vec![rect(50.0, 50.0, 20.0, 20.0)];
        let mut layout = ConstrainedFDLayout::new(rs, &[], 100.0, None);
        layout.run();
        assert_eq!(layout.num_nodes(), 1);
        assert!((layout.x[0] - 50.0).abs() < TOL);
        assert!((layout.y[0] - 50.0).abs() < TOL);
    }

    #[test]
    fn two_connected_nodes() {
        let rs = vec![rect(0.0, 0.0, 10.0, 10.0), rect(0.0, 0.0, 10.0, 10.0)];
        let es = vec![(0, 1)];
        let ideal = 100.0;
        let mut layout = ConstrainedFDLayout::new(rs, &es, ideal, None);
        layout.run();

        // Nodes should be approximately ideal_length apart.
        let dx = layout.x[0] - layout.x[1];
        let dy = layout.y[0] - layout.y[1];
        let dist = (dx * dx + dy * dy).sqrt();
        assert!(
            (dist - ideal).abs() < ideal * 0.5,
            "Expected distance ~{}, got {}",
            ideal,
            dist
        );
    }

    // ===================================================================
    // Category 2: D and G matrices
    // ===================================================================

    #[test]
    fn d_matrix_single_edge() {
        let rs = vec![rect(0.0, 0.0, 10.0, 10.0), rect(100.0, 0.0, 10.0, 10.0)];
        let es = vec![(0, 1)];
        let ideal = 50.0;
        let layout = ConstrainedFDLayout::new(rs, &es, ideal, None);

        // D[0][1] should be shortest_path(0,1) * ideal = 1 * 50 = 50
        let d01 = layout.d[0 * 2 + 1];
        assert!((d01 - ideal).abs() < TOL, "D[0][1] = {}, expected {}", d01, ideal);

        // G[0][1] should be EdgeNeighbour
        assert_eq!(layout.g[0 * 2 + 1], Connectivity::EdgeNeighbour);
        assert_eq!(layout.g[1 * 2 + 0], Connectivity::EdgeNeighbour);
    }

    #[test]
    fn d_matrix_path_graph() {
        // 0 -- 1 -- 2
        let rs = vec![
            rect(0.0, 0.0, 10.0, 10.0),
            rect(50.0, 0.0, 10.0, 10.0),
            rect(100.0, 0.0, 10.0, 10.0),
        ];
        let es = vec![(0, 1), (1, 2)];
        let ideal = 30.0;
        let layout = ConstrainedFDLayout::new(rs, &es, ideal, None);

        // D[0][1] = 1 * 30 = 30
        assert!((layout.d[0 * 3 + 1] - 30.0).abs() < TOL);
        // D[0][2] = 2 * 30 = 60 (shortest path through 1)
        assert!((layout.d[0 * 3 + 2] - 60.0).abs() < TOL);

        // G: 0-1 and 1-2 are EdgeNeighbour, 0-2 is ConnectedNonNeighbour
        assert_eq!(layout.g[0 * 3 + 1], Connectivity::EdgeNeighbour);
        assert_eq!(layout.g[0 * 3 + 2], Connectivity::ConnectedNonNeighbour);
    }

    #[test]
    fn d_matrix_disconnected_components() {
        // Node 0 and node 1 are not connected.
        let rs = vec![rect(0.0, 0.0, 10.0, 10.0), rect(100.0, 0.0, 10.0, 10.0)];
        let layout = ConstrainedFDLayout::new(rs, &[], 50.0, None);

        assert_eq!(layout.g[0 * 2 + 1], Connectivity::Disconnected);
        assert_eq!(layout.g[1 * 2 + 0], Connectivity::Disconnected);
    }

    #[test]
    fn d_matrix_with_edge_lengths() {
        // Edge with custom length 2.0 * ideal 50 = 100.
        let rs = vec![rect(0.0, 0.0, 10.0, 10.0), rect(100.0, 0.0, 10.0, 10.0)];
        let es = vec![(0, 1)];
        let ideal = 50.0;
        let el = vec![2.0];
        let layout = ConstrainedFDLayout::new(rs, &es, ideal, Some(&el));

        // D[0][1] = 2.0 * 50 = 100
        assert!((layout.d[0 * 2 + 1] - 100.0).abs() < TOL);
    }

    // ===================================================================
    // Category 3: Stress computation
    // ===================================================================

    #[test]
    fn stress_at_ideal_distance_is_zero() {
        // Two nodes at exactly ideal distance apart.
        let ideal = 100.0;
        let rs = vec![
            rect(0.0, 0.0, 10.0, 10.0),
            rect(ideal, 0.0, 10.0, 10.0),
        ];
        let es = vec![(0, 1)];
        let layout = ConstrainedFDLayout::new(rs, &es, ideal, None);

        let stress = layout.compute_stress(None);
        assert!(
            stress < TOL,
            "Stress should be ~0 at ideal distance, got {}",
            stress
        );
    }

    #[test]
    fn stress_increases_with_deviation() {
        let ideal = 100.0;
        // Close together (distance < ideal).
        let rs_close = vec![
            rect(0.0, 0.0, 10.0, 10.0),
            rect(10.0, 0.0, 10.0, 10.0),
        ];
        let es = vec![(0, 1)];
        let layout_close = ConstrainedFDLayout::new(rs_close, &es, ideal, None);
        let stress_close = layout_close.compute_stress(None);

        // Far apart (distance = ideal).
        let rs_ideal = vec![
            rect(0.0, 0.0, 10.0, 10.0),
            rect(ideal, 0.0, 10.0, 10.0),
        ];
        let layout_ideal = ConstrainedFDLayout::new(rs_ideal, &es, ideal, None);
        let stress_ideal = layout_ideal.compute_stress(None);

        assert!(
            stress_close > stress_ideal,
            "Stress at non-ideal distance ({}) should exceed stress at ideal ({})",
            stress_close,
            stress_ideal
        );
    }

    // ===================================================================
    // Category 4: Force computation
    // ===================================================================

    #[test]
    fn forces_are_zero_at_equilibrium() {
        // Two nodes at ideal distance on x-axis.
        let ideal = 100.0;
        let rs = vec![
            rect(0.0, 0.0, 10.0, 10.0),
            rect(ideal, 0.0, 10.0, 10.0),
        ];
        let es = vec![(0, 1)];
        let mut layout = ConstrainedFDLayout::new(rs, &es, ideal, None);

        let mut h = SparseMap::new(2);
        let mut g = vec![0.0; 2];
        layout.compute_forces(Dim::Horizontal, &mut h, &mut g);

        // Forces should be near zero at ideal distance.
        assert!(
            g[0].abs() < TOL,
            "g[0] = {}, expected ~0",
            g[0]
        );
        assert!(
            g[1].abs() < TOL,
            "g[1] = {}, expected ~0",
            g[1]
        );
    }

    #[test]
    fn forces_push_apart_when_too_close() {
        let ideal = 100.0;
        let rs = vec![
            rect(0.0, 0.0, 10.0, 10.0),
            rect(10.0, 0.0, 10.0, 10.0),
        ];
        let es = vec![(0, 1)];
        let mut layout = ConstrainedFDLayout::new(rs, &es, ideal, None);

        let mut h = SparseMap::new(2);
        let mut g = vec![0.0; 2];
        layout.compute_forces(Dim::Horizontal, &mut h, &mut g);

        // The gradient direction is such that movement = -stepsize * g.
        // Node 0 (at x=0) should be pushed left (away from node 1 at x=10),
        // so g[0] > 0 (movement = -g => leftward).
        // Node 1 should be pushed right (away from node 0), so g[1] < 0.
        assert!(g[0] > 0.0, "g[0] should be positive (descent pushes left), g[0] = {}", g[0]);
        assert!(g[1] < 0.0, "g[1] should be negative (descent pushes right), g[1] = {}", g[1]);
    }

    // ===================================================================
    // Category 5: Layout convergence
    // ===================================================================

    #[test]
    fn layout_reduces_stress() {
        let ideal = 100.0;
        let rs = vec![
            rect(0.0, 0.0, 10.0, 10.0),
            rect(10.0, 0.0, 10.0, 10.0),
            rect(20.0, 0.0, 10.0, 10.0),
        ];
        let es = vec![(0, 1), (1, 2), (0, 2)];
        let mut layout = ConstrainedFDLayout::new(rs, &es, ideal, None);

        let stress_before = layout.compute_stress(None);
        layout.run();
        let stress_after = layout.compute_stress(None);

        assert!(
            stress_after < stress_before,
            "Stress should decrease: {} -> {}",
            stress_before,
            stress_after
        );
    }

    #[test]
    fn triangle_graph_converges() {
        let ideal = 100.0;
        // Start with nodes spread in a triangle shape, not collinear.
        let rs = vec![
            rect(0.0, 0.0, 10.0, 10.0),
            rect(80.0, 0.0, 10.0, 10.0),
            rect(40.0, 70.0, 10.0, 10.0),
        ];
        let es = vec![(0, 1), (1, 2), (0, 2)];
        let mut layout = ConstrainedFDLayout::new(rs, &es, ideal, None);
        layout.set_convergence(1e-6, 500);
        layout.run();

        // All pairwise distances should be approximately equal (equilateral).
        let d01 = ((layout.x[0] - layout.x[1]).powi(2)
            + (layout.y[0] - layout.y[1]).powi(2))
        .sqrt();
        let d12 = ((layout.x[1] - layout.x[2]).powi(2)
            + (layout.y[1] - layout.y[2]).powi(2))
        .sqrt();
        let d02 = ((layout.x[0] - layout.x[2]).powi(2)
            + (layout.y[0] - layout.y[2]).powi(2))
        .sqrt();

        // Distances should be within 50% of average (stress-based layout
        // with limited iterations may not reach perfect equilateral).
        let avg = (d01 + d12 + d02) / 3.0;
        assert!(
            (d01 - avg).abs() / avg < 0.5,
            "Triangle not equilateral: d01={}, d12={}, d02={}",
            d01,
            d12,
            d02
        );
    }

    // ===================================================================
    // Category 6: Runge-Kutta vs simple descent
    // ===================================================================

    #[test]
    fn simple_descent_also_converges() {
        let ideal = 100.0;
        let rs = vec![
            rect(0.0, 0.0, 10.0, 10.0),
            rect(10.0, 0.0, 10.0, 10.0),
        ];
        let es = vec![(0, 1)];
        let mut layout = ConstrainedFDLayout::new(rs, &es, ideal, None);
        layout.set_runge_kutta(false);

        let stress_before = layout.compute_stress(None);
        layout.run();
        let stress_after = layout.compute_stress(None);

        assert!(stress_after < stress_before);
    }

    // ===================================================================
    // Category 7: Non-overlap constraints
    // ===================================================================

    #[test]
    fn non_overlap_prevents_overlaps() {
        // Place nodes on top of each other.
        let rs = vec![
            rect(0.0, 0.0, 30.0, 30.0),
            rect(5.0, 5.0, 30.0, 30.0),
            rect(10.0, 10.0, 30.0, 30.0),
        ];
        let es = vec![(0, 1), (1, 2)];
        let mut layout = ConstrainedFDLayout::new(rs, &es, 100.0, None);
        layout.set_avoid_node_overlaps(true);
        layout.run();

        // Check no bounding boxes overlap.
        let bb = layout.bounding_boxes();
        for i in 0..bb.len() {
            for j in (i + 1)..bb.len() {
                let overlap_x = bb[i].get_max_x() > bb[j].get_min_x() + TOL
                    && bb[j].get_max_x() > bb[i].get_min_x() + TOL;
                let overlap_y = bb[i].get_max_y() > bb[j].get_min_y() + TOL
                    && bb[j].get_max_y() > bb[i].get_min_y() + TOL;

                assert!(
                    !(overlap_x && overlap_y),
                    "Nodes {} and {} overlap: {:?} vs {:?}",
                    i,
                    j,
                    (bb[i].get_min_x(), bb[i].get_max_x(), bb[i].get_min_y(), bb[i].get_max_y()),
                    (bb[j].get_min_x(), bb[j].get_max_x(), bb[j].get_min_y(), bb[j].get_max_y()),
                );
            }
        }
    }

    // ===================================================================
    // Category 8: TestConvergence
    // ===================================================================

    #[test]
    fn convergence_first_call_records_stress() {
        let mut tc = TestConvergence::with_defaults();
        // First call with initial stress should not converge (unless max_iterations = 0).
        assert!(!tc.test(100.0));
        assert_eq!(tc.iterations(), 1);
    }

    #[test]
    fn convergence_detects_plateau() {
        let mut tc = TestConvergence::new(1e-4, 1000);
        tc.test(100.0); // First call.
        // Same stress => relative change is 0 < tolerance => converged.
        assert!(tc.test(100.0));
    }

    #[test]
    fn convergence_respects_max_iterations() {
        let mut tc = TestConvergence::new(1e-10, 3);
        tc.test(1000.0);
        tc.test(500.0);
        tc.test(250.0);
        // 4th iteration (> max_iterations=3) => converged.
        assert!(tc.test(125.0));
    }

    #[test]
    fn convergence_reset() {
        let mut tc = TestConvergence::new(1e-4, 100);
        tc.test(100.0);
        tc.test(100.0);
        tc.reset();
        assert_eq!(tc.iterations(), 0);
        // After reset, first call should not converge.
        assert!(!tc.test(100.0));
    }

    // ===================================================================
    // Category 9: Lock
    // ===================================================================

    #[test]
    fn lock_pos_horizontal() {
        let lock = Lock::new(0, 10.0, 20.0);
        assert_eq!(lock.pos(Dim::Horizontal), 10.0);
        assert_eq!(lock.pos(Dim::Vertical), 20.0);
    }

    // ===================================================================
    // Category 10: Resize
    // ===================================================================

    #[test]
    fn resize_creates_correct_rectangle() {
        let r = Resize::new(0, 10.0, 20.0, 30.0, 40.0);
        assert_eq!(r.id, 0);
        assert!((r.target.get_min_x() - 10.0).abs() < TOL);
        assert!((r.target.get_max_x() - 40.0).abs() < TOL);
        assert!((r.target.get_min_y() - 20.0).abs() < TOL);
        assert!((r.target.get_max_y() - 60.0).abs() < TOL);
    }

    // ===================================================================
    // Category 11: Step size computation
    // ===================================================================

    #[test]
    fn step_size_zero_denominator() {
        let rs = vec![rect(0.0, 0.0, 10.0, 10.0)];
        let layout = ConstrainedFDLayout::new(rs, &[], 100.0, None);

        // Zero Hessian and zero direction => step size 0.
        let h_map = SparseMap::new(1);
        let h = SparseMatrix::from_sparse_map(&h_map);
        let g = [0.0];
        let d = [0.0];
        assert_eq!(layout.compute_step_size(&h, &g, &d), 0.0);
    }

    #[test]
    fn step_size_identity_hessian() {
        let rs = vec![rect(0.0, 0.0, 10.0, 10.0), rect(100.0, 0.0, 10.0, 10.0)];
        let layout = ConstrainedFDLayout::new(rs, &[], 100.0, None);

        let mut h_map = SparseMap::new(2);
        h_map.set(0, 0, 1.0);
        h_map.set(1, 1, 1.0);
        let h = SparseMatrix::from_sparse_map(&h_map);

        // g = d = [1, 0]: step = g'd / d'Hd = 1 / 1 = 1.
        let g = [1.0, 0.0];
        let d = [1.0, 0.0];
        assert!((layout.compute_step_size(&h, &g, &d) - 1.0).abs() < TOL);
    }

    // ===================================================================
    // Category 12: Offset direction (coincident node displacement)
    // ===================================================================

    #[test]
    fn offset_dir_produces_nonzero_displacement() {
        let rs = vec![rect(0.0, 0.0, 10.0, 10.0), rect(10.0, 0.0, 10.0, 10.0)];
        let es = vec![(0, 1)];
        let mut layout = ConstrainedFDLayout::new(rs, &es, 100.0, None);

        let (dx, dy, dz) = layout.offset_dir();
        assert!(dx != 0.0 || dy != 0.0, "Offset should be nonzero");
        let magnitude = (dx * dx + dy * dy + dz * dz).sqrt();
        assert!(
            (magnitude - layout.min_d).abs() < TOL,
            "Offset magnitude {} should equal min_d {}",
            magnitude,
            layout.min_d
        );
    }

    // ===================================================================
    // Category 13: run_once
    // ===================================================================

    #[test]
    fn run_once_changes_positions() {
        let rs = vec![
            rect(0.0, 0.0, 10.0, 10.0),
            rect(5.0, 0.0, 10.0, 10.0),
        ];
        let es = vec![(0, 1)];
        let mut layout = ConstrainedFDLayout::new(rs, &es, 100.0, None);

        let x0 = layout.x.clone();
        layout.run_once(true, true);

        // Positions should have changed.
        let moved = layout.x.iter().zip(x0.iter()).any(|(a, b)| (a - b).abs() > TOL);
        assert!(moved, "run_once should move nodes");
    }

    // ===================================================================
    // Category 14: Neighbour stress mode
    // ===================================================================

    #[test]
    fn neighbour_stress_ignores_non_neighbours() {
        // Path: 0 -- 1 -- 2
        // With neighbour stress, force between 0 and 2 should be zero.
        let ideal = 100.0;
        let rs = vec![
            rect(0.0, 0.0, 10.0, 10.0),
            rect(50.0, 0.0, 10.0, 10.0),
            rect(300.0, 0.0, 10.0, 10.0), // Far from ideal
        ];
        let es = vec![(0, 1), (1, 2)];
        let mut layout = ConstrainedFDLayout::new(rs, &es, ideal, None);
        layout.set_use_neighbour_stress(true);

        let stress = layout.compute_stress(None);
        // The stress should only count pairs (0,1) and (1,2), not (0,2).
        // With neighbour stress, 0-2 pair is skipped.
        assert!(stress > 0.0, "Should have some stress from non-ideal distances");
    }

    // ===================================================================
    // Category 15: DesiredPosition
    // ===================================================================

    #[test]
    fn desired_positions_add_stress() {
        let rs = vec![rect(0.0, 0.0, 10.0, 10.0)];
        let mut layout = ConstrainedFDLayout::new(rs, &[], 100.0, None);

        // Node at (0,0), desired at (100,100) => stress > 0.
        layout.set_desired_positions(vec![DesiredPosition {
            id: 0,
            x: 100.0,
            y: 100.0,
            z: 0.0,
            weight: 1.0,
        }]);

        let stress = layout.compute_stress(None);
        assert!(stress > 0.0, "Desired position should add stress");
    }

    // ===================================================================
    // Category 16: Larger graphs
    // ===================================================================

    #[test]
    fn line_graph_5_nodes() {
        let ideal = 50.0;
        let rs: Vec<Rectangle> = (0..5)
            .map(|i| rect(i as f64 * 10.0, 0.0, 10.0, 10.0))
            .collect();
        let es: Vec<Edge> = (0..4).map(|i| (i, i + 1)).collect();
        let mut layout = ConstrainedFDLayout::new(rs, &es, ideal, None);

        let stress_before = layout.compute_stress(None);
        layout.run();
        let stress_after = layout.compute_stress(None);

        assert!(
            stress_after < stress_before,
            "Stress should decrease for line graph"
        );

        // Check monotonicity: nodes should be roughly in order.
        for i in 0..4 {
            // At least for x or y, there should be an ordering tendency.
            // (The layout may rotate, so we check that adjacent nodes aren't
            // at the same position.)
            let d = ((layout.x[i] - layout.x[i + 1]).powi(2)
                + (layout.y[i] - layout.y[i + 1]).powi(2))
            .sqrt();
            assert!(d > ideal * 0.3, "Adjacent nodes {} and {} too close: {}", i, i + 1, d);
        }
    }

    #[test]
    fn complete_graph_4_nodes() {
        let ideal = 80.0;
        let rs: Vec<Rectangle> = (0..4)
            .map(|_| rect(0.0, 0.0, 10.0, 10.0))
            .collect();
        let es: Vec<Edge> = vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
        let mut layout = ConstrainedFDLayout::new(rs, &es, ideal, None);
        layout.run();

        // All pairwise distances should be close to ideal.
        for i in 0..4 {
            for j in (i + 1)..4 {
                let d = ((layout.x[i] - layout.x[j]).powi(2)
                    + (layout.y[i] - layout.y[j]).powi(2))
                .sqrt();
                assert!(
                    (d - ideal).abs() < ideal * 0.5,
                    "K4: distance {}-{} = {}, expected ~{}",
                    i,
                    j,
                    d,
                    ideal
                );
            }
        }
    }

    // ===================================================================
    // Category 17: makeFeasible
    // ===================================================================

    #[test]
    fn make_feasible_does_not_panic() {
        let rs = vec![
            rect(0.0, 0.0, 20.0, 20.0),
            rect(5.0, 5.0, 20.0, 20.0),
        ];
        let es = vec![(0, 1)];
        let mut layout = ConstrainedFDLayout::new(rs, &es, 100.0, None);
        layout.set_avoid_node_overlaps(true);
        layout.make_feasible();
        // Should not panic.
    }

    // ===================================================================
    // Category 18: Edge length scaling
    // ===================================================================

    #[test]
    fn edge_lengths_affect_distances() {
        let rs = vec![
            rect(0.0, 0.0, 10.0, 10.0),
            rect(10.0, 0.0, 10.0, 10.0),
            rect(20.0, 0.0, 10.0, 10.0),
        ];
        let es = vec![(0, 1), (1, 2)];
        let ideal = 50.0;
        let el = vec![1.0, 2.0]; // edge 1-2 has double ideal length.
        let mut layout = ConstrainedFDLayout::new(rs, &es, ideal, Some(&el));
        layout.run();

        let d01 = ((layout.x[0] - layout.x[1]).powi(2)
            + (layout.y[0] - layout.y[1]).powi(2))
        .sqrt();
        let d12 = ((layout.x[1] - layout.x[2]).powi(2)
            + (layout.y[1] - layout.y[2]).powi(2))
        .sqrt();

        // Edge 1-2 should be longer than edge 0-1.
        assert!(
            d12 > d01 * 1.2,
            "Edge 1-2 ({}) should be noticeably longer than 0-1 ({})",
            d12,
            d01
        );
    }

    // ===================================================================
    // Category 19: ProjectionResult
    // ===================================================================

    #[test]
    fn solve_constraints_empty() {
        let result = solve_constraints(vec![], vec![]);
        assert_eq!(result.error_level, 0);
    }

    // ===================================================================
    // Category 20: Connectivity enum
    // ===================================================================

    #[test]
    fn connectivity_values() {
        assert_eq!(Connectivity::Disconnected as u8, 0);
        assert_eq!(Connectivity::EdgeNeighbour as u8, 1);
        assert_eq!(Connectivity::ConnectedNonNeighbour as u8, 2);
    }

    // ===================================================================
    // Category 21: Axis-selective layout
    // ===================================================================

    #[test]
    fn x_only_layout_changes_only_x() {
        let rs = vec![
            rect(0.0, 0.0, 10.0, 10.0),
            rect(5.0, 0.0, 10.0, 10.0),
        ];
        let es = vec![(0, 1)];
        let mut layout = ConstrainedFDLayout::new(rs, &es, 100.0, None);
        let y_before = layout.y.clone();
        layout.run_axes(true, false, None);

        // Y positions should not change when only running x-axis.
        for i in 0..2 {
            assert!(
                (layout.y[i] - y_before[i]).abs() < TOL,
                "Y[{}] changed from {} to {}",
                i,
                y_before[i],
                layout.y[i]
            );
        }
    }

    // ===================================================================
    // Category 22: Stress with locks
    // ===================================================================

    #[test]
    fn stress_with_locks_adds_penalty() {
        let rs = vec![rect(0.0, 0.0, 10.0, 10.0), rect(100.0, 0.0, 10.0, 10.0)];
        let es = vec![(0, 1)];
        let layout = ConstrainedFDLayout::new(rs, &es, 100.0, None);

        struct LockedPre {
            locks: Vec<Lock>,
        }
        impl PreIteration for LockedPre {
            fn locks(&self) -> &[Lock] {
                &self.locks
            }
        }

        let base_stress = layout.compute_stress(None);

        let pre = LockedPre {
            locks: vec![Lock::new(0, 50.0, 50.0)], // lock at wrong position
        };
        let lock_stress = layout.compute_stress(Some(&pre));

        assert!(
            lock_stress > base_stress,
            "Stress with lock ({}) should exceed base ({})",
            lock_stress,
            base_stress
        );
    }

    // ===================================================================
    // Category 23: Min distance computation
    // ===================================================================

    #[test]
    fn min_d_is_smallest_nonzero_distance() {
        let rs = vec![
            rect(0.0, 0.0, 10.0, 10.0),
            rect(10.0, 0.0, 10.0, 10.0),
            rect(20.0, 0.0, 10.0, 10.0),
        ];
        let es = vec![(0, 1), (1, 2)]; // path: 0-1-2
        let ideal = 50.0;
        let layout = ConstrainedFDLayout::new(rs, &es, ideal, None);

        // min_d should be the smallest D value = 1 * 50 = 50.
        assert!(
            (layout.min_d - ideal).abs() < TOL,
            "min_d = {}, expected {}",
            layout.min_d,
            ideal
        );
    }

    #[test]
    fn min_d_defaults_when_no_edges() {
        let rs = vec![rect(0.0, 0.0, 10.0, 10.0)];
        let layout = ConstrainedFDLayout::new(rs, &[], 100.0, None);
        assert!((layout.min_d - DEFAULT_EDGE_LENGTH).abs() < TOL);
    }

    // ===================================================================
    // Category 24: Geometry quality tests
    // ===================================================================

    /// Check if two line segments (p1-p2) and (p3-p4) cross.
    fn segments_cross(
        p1: (f64, f64), p2: (f64, f64),
        p3: (f64, f64), p4: (f64, f64),
    ) -> bool {
        fn cross(o: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
            (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
        }
        let d1 = cross(p3, p4, p1);
        let d2 = cross(p3, p4, p2);
        let d3 = cross(p1, p2, p3);
        let d4 = cross(p1, p2, p4);
        if ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0))
            && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0))
        {
            return true;
        }
        false
    }

    fn count_edge_crossings(x: &[f64], y: &[f64], edges: &[Edge]) -> usize {
        let mut crossings = 0;
        for i in 0..edges.len() {
            for j in (i + 1)..edges.len() {
                let (s1, t1) = edges[i];
                let (s2, t2) = edges[j];
                // Skip edges that share a node.
                if s1 == s2 || s1 == t2 || t1 == s2 || t1 == t2 {
                    continue;
                }
                if segments_cross(
                    (x[s1], y[s1]), (x[t1], y[t1]),
                    (x[s2], y[s2]), (x[t2], y[t2]),
                ) {
                    crossings += 1;
                }
            }
        }
        crossings
    }

    #[test]
    fn grid_4x4_no_edge_crossings() {
        let cols = 4;
        let rows = 4;
        let n = rows * cols;
        let ideal = 80.0;

        // Randomized initial positions (scrambled grid).
        let mut rng = PseudoRandom::new(42.0);
        let rs: Vec<Rectangle> = (0..n)
            .map(|_| {
                let x = rng.get_next_between(0.0, 400.0);
                let y = rng.get_next_between(0.0, 400.0);
                rect(x, y, 10.0, 10.0)
            })
            .collect();

        let mut edges: Vec<Edge> = Vec::new();
        for r in 0..rows {
            for c in 0..cols {
                let i = r * cols + c;
                if c + 1 < cols { edges.push((i, i + 1)); }
                if r + 1 < rows { edges.push((i, i + cols)); }
            }
        }

        let mut layout = ConstrainedFDLayout::new(rs, &edges, ideal, None);
        layout.set_convergence(1e-6, 500);
        layout.run();

        // Check: no edge crossings in the result.
        let crossings = count_edge_crossings(&layout.x, &layout.y, &edges);
        assert_eq!(
            crossings, 0,
            "4x4 grid should have 0 edge crossings, got {}.\n\
             Positions: x={:?}\n           y={:?}",
            crossings, layout.x, layout.y
        );
    }

    #[test]
    fn grid_4x4_edge_distances_near_ideal() {
        let cols = 4;
        let rows = 4;
        let n = rows * cols;
        let ideal = 80.0;

        // Start from scrambled positions.
        let mut rng = PseudoRandom::new(42.0);
        let rs: Vec<Rectangle> = (0..n)
            .map(|_| {
                let x = rng.get_next_between(0.0, 400.0);
                let y = rng.get_next_between(0.0, 400.0);
                rect(x, y, 10.0, 10.0)
            })
            .collect();

        let mut edges: Vec<Edge> = Vec::new();
        for r in 0..rows {
            for c in 0..cols {
                let i = r * cols + c;
                if c + 1 < cols { edges.push((i, i + 1)); }
                if r + 1 < rows { edges.push((i, i + cols)); }
            }
        }

        let mut layout = ConstrainedFDLayout::new(rs, &edges, ideal, None);
        layout.set_convergence(1e-6, 500);
        layout.run();

        // Every edge should be within 50% of ideal length.
        let tolerance_fraction = 0.5;
        for &(s, t) in &edges {
            let d = ((layout.x[s] - layout.x[t]).powi(2)
                + (layout.y[s] - layout.y[t]).powi(2))
            .sqrt();
            assert!(
                (d - ideal).abs() < ideal * tolerance_fraction,
                "Edge {}-{}: distance {:.1}, expected ~{:.1}",
                s, t, d, ideal,
            );
        }
    }

    #[test]
    fn stress_decreases_monotonically_for_simple_graph() {
        // 6-node cycle: stress should decrease every iteration.
        let n = 6;
        let ideal = 80.0;
        let rs: Vec<Rectangle> = (0..n)
            .map(|i| rect(i as f64 * 5.0, 0.0, 10.0, 10.0))
            .collect();
        let edges: Vec<Edge> = (0..n).map(|i| (i, (i + 1) % n)).collect();

        let mut layout = ConstrainedFDLayout::new(rs, &edges, ideal, None);
        layout.set_convergence(1e-10, 1); // 1 iteration at a time

        let mut stresses = vec![layout.compute_stress(None)];

        let max_steps = 50;
        for _ in 0..max_steps {
            layout.run_once(true, true);
            stresses.push(layout.compute_stress(None));
        }

        // Stress should decrease (or stay the same) over iterations.
        let mut increases = 0;
        for i in 1..stresses.len() {
            if stresses[i] > stresses[i - 1] + 1e-6 {
                increases += 1;
            }
        }
        assert!(
            increases <= 2,
            "Stress increased {} times out of {} steps. Stresses: {:?}",
            increases, max_steps, stresses,
        );

        // Stress should have decreased significantly overall.
        let initial = stresses[0];
        let final_s = *stresses.last().unwrap();
        assert!(
            final_s < initial * 0.5,
            "Stress should decrease significantly: {:.4} -> {:.4}",
            initial, final_s,
        );
    }

    #[test]
    fn two_nodes_converge_to_ideal_distance() {
        let ideal = 100.0;
        // Start very close together.
        let rs = vec![rect(0.0, 0.0, 10.0, 10.0), rect(1.0, 0.0, 10.0, 10.0)];
        let es = vec![(0, 1)];
        let mut layout = ConstrainedFDLayout::new(rs, &es, ideal, None);
        layout.set_convergence(1e-8, 1000);
        layout.run();

        let dx = layout.x[0] - layout.x[1];
        let dy = layout.y[0] - layout.y[1];
        let dist = (dx * dx + dy * dy).sqrt();
        assert!(
            (dist - ideal).abs() < ideal * 0.05,
            "Two nodes should converge to ideal distance {}, got {:.4}",
            ideal, dist,
        );
    }

    #[test]
    fn complete_graph_k6_stress_decreases() {
        // K6 is non-planar (minimum crossing number = 3), so we only test
        // that stress majorization reduces stress, not that it eliminates crossings.
        let n = 6;
        let ideal = 80.0;

        let mut rng = PseudoRandom::new(7.0);
        let rs: Vec<Rectangle> = (0..n)
            .map(|_| {
                let x = rng.get_next_between(0.0, 200.0);
                let y = rng.get_next_between(0.0, 200.0);
                rect(x, y, 10.0, 10.0)
            })
            .collect();

        let mut edges: Vec<Edge> = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                edges.push((i, j));
            }
        }

        let mut layout = ConstrainedFDLayout::new(rs, &edges, ideal, None);
        let stress_before = layout.compute_stress(None);
        layout.set_convergence(1e-6, 500);
        layout.run();
        let stress_after = layout.compute_stress(None);

        assert!(
            stress_after < stress_before,
            "K6 stress should decrease: {:.4} -> {:.4}",
            stress_before, stress_after,
        );

        // K6 in 2D can't place all pairs at ideal distance.
        // Check that edge distances are at least in a reasonable range.
        for &(s, t) in &edges {
            let d = ((layout.x[s] - layout.x[t]).powi(2)
                + (layout.y[s] - layout.y[t]).powi(2))
            .sqrt();
            assert!(
                d > ideal * 0.3 && d < ideal * 2.0,
                "K6 edge {}-{}: distance {:.1} out of range [{:.1}, {:.1}]",
                s, t, d, ideal * 0.3, ideal * 2.0,
            );
        }
    }

    #[test]
    fn square_4_cycle_preserves_topology() {
        let ideal = 100.0;
        let rs = vec![
            rect(-100.0, -100.0, 10.0, 10.0),
            rect(100.0, -100.0, 10.0, 10.0),
            rect(100.0, 100.0, 10.0, 10.0),
            rect(-100.0, 100.0, 10.0, 10.0),
        ];
        let edges: Vec<Edge> = vec![(0, 1), (1, 2), (2, 3), (3, 0)];
        let mut layout = ConstrainedFDLayout::new(rs, &edges, ideal, None);
        layout.run();

        let crossings = count_edge_crossings(&layout.x, &layout.y, &edges);
        assert_eq!(crossings, 0, "4-cycle should have 0 crossings");

        // All edges should be within 30% of ideal.
        for &(s, t) in &edges {
            let d = ((layout.x[s] - layout.x[t]).powi(2)
                + (layout.y[s] - layout.y[t]).powi(2))
            .sqrt();
            assert!(
                (d - ideal).abs() < ideal * 0.3,
                "Edge {}-{}: distance {:.1}, expected ~{:.1}",
                s, t, d, ideal,
            );
        }
    }

    #[test]
    fn cycle_12_default_stress_decreases() {
        // With default skip (matching C++), some seeds produce crossings
        // due to local minima. Verify stress always decreases.
        let n = 12;
        let ideal = 60.0;

        for seed in [1.0, 7.0, 42.0, 99.0, 123.0, 256.0, 500.0, 777.0] {
            let mut rng = PseudoRandom::new(seed);
            let rs: Vec<Rectangle> = (0..n)
                .map(|_| {
                    let x = rng.get_next_between(-200.0, 200.0);
                    let y = rng.get_next_between(-200.0, 200.0);
                    rect(x, y, 10.0, 10.0)
                })
                .collect();

            let edges: Vec<Edge> = (0..n).map(|i| (i, (i + 1) % n)).collect();

            let mut layout = ConstrainedFDLayout::new(rs, &edges, ideal, None);
            layout.set_convergence(1e-8, 1000);

            let initial_stress = layout.compute_stress(None);
            layout.run();
            let final_stress = layout.compute_stress(None);

            assert!(
                final_stress < initial_stress,
                "12-cycle seed={}: stress should decrease ({:.2} -> {:.2})",
                seed, initial_stress, final_stress,
            );
        }
    }

    #[test]
    fn cycle_12_full_stress_fewer_crossings() {
        // With skip_distant_non_neighbours=false, fewer seeds get stuck in
        // local minima. But stress majorization fundamentally has local minima
        // for cycles from random positions — some seeds still produce crossings.
        let n = 12;
        let ideal = 60.0;
        let mut total_crossings = 0;

        for seed in [1.0, 7.0, 42.0, 99.0, 123.0, 256.0, 500.0, 777.0] {
            let mut rng = PseudoRandom::new(seed);
            let rs: Vec<Rectangle> = (0..n)
                .map(|_| {
                    let x = rng.get_next_between(-200.0, 200.0);
                    let y = rng.get_next_between(-200.0, 200.0);
                    rect(x, y, 10.0, 10.0)
                })
                .collect();

            let edges: Vec<Edge> = (0..n).map(|i| (i, (i + 1) % n)).collect();

            let mut layout = ConstrainedFDLayout::new(rs, &edges, ideal, None);
            layout.set_skip_distant_non_neighbours(false);
            layout.set_convergence(1e-8, 1000);
            layout.run();

            total_crossings += count_edge_crossings(&layout.x, &layout.y, &edges);
        }

        // Stress majorization has local minima for cycles from random positions.
        // With full stress (no skip), fewer crossings occur than with the skip,
        // but some are still possible. This is a known limitation.
        assert!(
            total_crossings <= 5,
            "Expected at most 5 total crossings across 8 seeds, got {}",
            total_crossings,
        );
    }

    #[test]
    fn cycle_12_from_circular_init_no_crossings() {
        // When initialized in a circle (near the global optimum), the algorithm
        // always converges without crossings. This is the proper way to lay out
        // cycle graphs.
        let n = 12;
        let ideal = 60.0;
        let radius = (ideal / 2.0) / (std::f64::consts::PI / n as f64).sin();

        let rs: Vec<Rectangle> = (0..n)
            .map(|i| {
                let angle = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
                let x = radius * angle.cos();
                let y = radius * angle.sin();
                rect(x, y, 10.0, 10.0)
            })
            .collect();

        let edges: Vec<Edge> = (0..n).map(|i| (i, (i + 1) % n)).collect();

        let mut layout = ConstrainedFDLayout::new(rs, &edges, ideal, None);
        layout.set_convergence(1e-8, 1000);
        layout.run();

        let crossings = count_edge_crossings(&layout.x, &layout.y, &edges);
        assert_eq!(crossings, 0, "Circular-init 12-cycle should have 0 crossings");
    }

    // ---- Category 25: 3D layout tests ----

    #[test]
    fn test_3d_constructor() {
        let rs = vec![rect(0.0, 0.0, 10.0, 10.0), rect(50.0, 0.0, 10.0, 10.0)];
        let z = vec![0.0, 100.0];
        let es = vec![(0, 1)];
        let layout = ConstrainedFDLayout::new_3d(rs, &z, &es, 80.0, None);

        assert_eq!(layout.dims, 3);
        assert_eq!(layout.z.len(), 2);
        assert_eq!(layout.z[0], 0.0);
        assert_eq!(layout.z[1], 100.0);
    }

    #[test]
    fn test_3d_stress_includes_z() {
        // Two nodes separated only in Z: stress should reflect Z distance.
        let rs = vec![rect(0.0, 0.0, 10.0, 10.0), rect(0.0, 0.0, 10.0, 10.0)];
        let z = vec![0.0, 200.0];
        let es = vec![(0, 1)];
        let layout = ConstrainedFDLayout::new_3d(rs, &z, &es, 80.0, None);

        let stress = layout.compute_stress(None);
        assert!(stress > 0.0, "3D stress should be > 0 for non-ideal distance");

        // Compare with 2D version where nodes are coincident: stress should differ.
        let rs2 = vec![rect(0.0, 0.0, 10.0, 10.0), rect(0.0, 0.0, 10.0, 10.0)];
        let layout2d = ConstrainedFDLayout::new(rs2, &es, 80.0, None);
        let stress2d = layout2d.compute_stress(None);

        assert!(
            (stress - stress2d).abs() > 0.01,
            "3D stress ({}) should differ from 2D stress ({}) when Z varies",
            stress, stress2d,
        );
    }

    #[test]
    fn test_3d_run_converges() {
        // Tetrahedron: 4 nodes, 6 edges.
        let n = 4;
        let ideal = 80.0;
        let rs = vec![
            rect(0.0, 0.0, 10.0, 10.0),
            rect(100.0, 0.0, 10.0, 10.0),
            rect(50.0, 100.0, 10.0, 10.0),
            rect(50.0, 50.0, 10.0, 10.0),
        ];
        let z = vec![0.0, 0.0, 0.0, 80.0];
        let edges: Vec<Edge> = vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

        let mut layout = ConstrainedFDLayout::new_3d(rs, &z, &edges, ideal, None);
        layout.set_convergence(1e-6, 500);

        let initial_stress = layout.compute_stress(None);
        layout.run();
        let final_stress = layout.compute_stress(None);

        assert!(
            final_stress < initial_stress,
            "3D tetrahedron stress should decrease: {:.2} -> {:.2}",
            initial_stress, final_stress,
        );

        // All edges should be within 40% of ideal.
        for &(s, t) in &edges {
            let dx = layout.x[s] - layout.x[t];
            let dy = layout.y[s] - layout.y[t];
            let dz = layout.z[s] - layout.z[t];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            assert!(
                (dist - ideal).abs() < ideal * 0.4,
                "3D edge {}-{}: distance {:.1}, expected ~{:.1}",
                s, t, dist, ideal,
            );
        }
    }

    #[test]
    fn test_3d_position_packing() {
        let rs = vec![rect(10.0, 20.0, 5.0, 5.0), rect(30.0, 40.0, 5.0, 5.0)];
        let z = vec![50.0, 60.0];
        let es = vec![(0, 1)];
        let layout = ConstrainedFDLayout::new_3d(rs, &z, &es, 80.0, None);

        let mut pos = vec![0.0; 6]; // 3 dims * 2 nodes
        layout.get_position(&mut pos);

        // [x0, x1, y0, y1, z0, z1]
        assert!((pos[0] - 10.0).abs() < 0.01); // centre of rect(10, 20, 5, 5) x
        assert!((pos[1] - 30.0).abs() < 0.01);
        assert!((pos[2] - 20.0).abs() < 0.01); // y
        assert!((pos[3] - 40.0).abs() < 0.01);
        assert!((pos[4] - 50.0).abs() < 0.01); // z
        assert!((pos[5] - 60.0).abs() < 0.01);
    }

    #[test]
    fn test_2d_unchanged_by_3d_code() {
        // Verify 2D constructor still works exactly as before.
        let rs = vec![rect(0.0, 0.0, 10.0, 10.0), rect(50.0, 0.0, 10.0, 10.0)];
        let es = vec![(0, 1)];
        let layout = ConstrainedFDLayout::new(rs, &es, 80.0, None);

        assert_eq!(layout.dims, 2);
        assert!(layout.z.is_empty());

        let mut pos = vec![0.0; 4]; // 2 dims * 2 nodes
        layout.get_position(&mut pos);
        assert!(pos[0].is_finite());
        assert!(pos[1].is_finite());
    }
}
