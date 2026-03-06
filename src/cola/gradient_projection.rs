//! Gradient projection solver for constrained quadratic optimization.
//!
//! Solves Qx = b subject to separation constraints using gradient
//! descent with VPSC projection.
//!
//! C++ ref: libcola/gradient_projection.h, libcola/gradient_projection.cpp

use crate::cola::conjugate_gradient::inner;
use crate::cola::sparse_matrix::SparseMatrix;
use crate::vpsc::{Constraint, Dim, IncSolver, Variable};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Upper bound on beta for constrained step limiting. Values at or above this
/// indicate the unconstrained step was essentially feasible.
const BETA_UPPER_LIMIT: f64 = 0.99999;

/// Default convergence tolerance for the solve loop.
const DEFAULT_TOLERANCE: f64 = 1e-4;

/// Default maximum number of gradient projection iterations.
const DEFAULT_MAX_ITERATIONS: usize = 100;

/// Default weight for variables when none is specified.
const DEFAULT_WEIGHT: f64 = 1.0;

/// Denominator guard: if g'Qg is below this, skip the gradient step
/// (degenerate direction).
const DENOMINATOR_EPSILON: f64 = 1e-30;

/// Factor applied to g'Qg denominator in optimal step computation.
/// alpha = (g'g) / (STEP_DENOM_FACTOR * g'Qg)
const STEP_DENOM_FACTOR: f64 = 2.0;

/// Factor applied to step size in beta computation.
/// beta = BETA_SCALE * computeStepSize(g, d)
const BETA_SCALE: f64 = 0.5;

/// Weight assigned to fixed-position variables so VPSC keeps them in place.
const FIXED_WEIGHT: f64 = 1e8;

// ---------------------------------------------------------------------------
// ConstraintSpec
// ---------------------------------------------------------------------------

/// A constraint specification (not yet instantiated in a solver).
///
/// Describes left + gap <= right (or == if `equality` is true).
#[derive(Debug, Clone)]
pub struct ConstraintSpec {
    pub left: usize,
    pub right: usize,
    pub gap: f64,
    pub equality: bool,
}

// ---------------------------------------------------------------------------
// GradientProjection
// ---------------------------------------------------------------------------

/// Gradient projection solver for constrained quadratic optimization.
///
/// Minimises x'Qx - 2bx subject to separation constraints by alternating
/// gradient descent steps with constraint projection via VPSC.
pub struct GradientProjection {
    dim: Dim,
    dense_size: usize,
    dense_q: Vec<f64>,
    tolerance: f64,
    max_iterations: usize,
    num_static_vars: usize,
    var_weights: Vec<f64>,
    var_scales: Vec<f64>,
    var_fixed: Vec<bool>,
    var_desired: Vec<f64>,
    constraints: Vec<ConstraintSpec>,
    scaling: bool,
}

impl GradientProjection {
    /// Create a new gradient projection solver.
    ///
    /// `dense_q` is the dense n x n Laplacian matrix in row-major order.
    /// `dim` is the layout dimension this solver operates on.
    /// `scaling` enables diagonal scaling of Q for better conditioning.
    pub fn new(
        dim: Dim,
        dense_q: Vec<f64>,
        tolerance: f64,
        max_iterations: usize,
        scaling: bool,
    ) -> Self {
        let dense_size = (dense_q.len() as f64).sqrt() as usize;
        debug_assert_eq!(
            dense_size * dense_size,
            dense_q.len(),
            "dense_q must be a square matrix in row-major order"
        );

        let mut var_scales = vec![1.0; dense_size];

        if scaling {
            for i in 0..dense_size {
                let diag = dense_q[i * dense_size + i].abs();
                if diag > 0.0 {
                    var_scales[i] = 1.0 / diag.sqrt();
                }
            }
        }

        let var_weights = vec![DEFAULT_WEIGHT; dense_size];
        let var_fixed = vec![false; dense_size];
        let var_desired = vec![0.0; dense_size];

        Self {
            dim,
            dense_size,
            dense_q,
            tolerance,
            max_iterations,
            num_static_vars: dense_size,
            var_weights,
            var_scales,
            var_fixed,
            var_desired,
            constraints: Vec::new(),
            scaling,
        }
    }

    /// Create with default tolerance and max iterations.
    pub fn with_defaults(dim: Dim, dense_q: Vec<f64>, scaling: bool) -> Self {
        Self::new(dim, dense_q, DEFAULT_TOLERANCE, DEFAULT_MAX_ITERATIONS, scaling)
    }

    /// Add a separation constraint specification.
    pub fn add_constraint(&mut self, spec: ConstraintSpec) {
        self.constraints.push(spec);
    }

    /// Add multiple constraint specifications.
    pub fn add_constraints(&mut self, specs: impl IntoIterator<Item = ConstraintSpec>) {
        self.constraints.extend(specs);
    }

    /// Fix a variable's position. Fixed variables resist movement during
    /// the gradient step (they keep their desired position).
    pub fn fix_pos(&mut self, i: usize, pos: f64) {
        assert!(i < self.dense_size, "variable index out of range");
        self.var_fixed[i] = true;
        self.var_desired[i] = pos;
    }

    /// Unfix a variable, allowing it to move freely again.
    pub fn unfix_pos(&mut self, i: usize) {
        assert!(i < self.dense_size, "variable index out of range");
        self.var_fixed[i] = false;
    }

    /// Number of static (non-generated) variables.
    pub fn num_static_vars(&self) -> usize {
        self.num_static_vars
    }

    /// The dimension this solver operates on.
    pub fn dim(&self) -> Dim {
        self.dim
    }

    /// Solve Qx = b subject to separation constraints.
    ///
    /// `b` is the linear coefficient vector (length `dense_size`).
    /// `x` is the initial position guess on entry and the result on exit.
    ///
    /// Returns the number of iterations taken.
    pub fn solve(&mut self, b: &[f64], x: &mut [f64]) -> usize {
        self.solve_with_sparse(b, x, None)
    }

    /// Solve with an optional additional sparse Q term.
    ///
    /// The effective quadratic is (dense_q + sparse_q).
    pub fn solve_with_sparse(
        &mut self,
        b: &[f64],
        x: &mut [f64],
        sparse_q: Option<&SparseMatrix>,
    ) -> usize {
        let n = self.dense_size;
        debug_assert_eq!(b.len(), n);
        debug_assert!(x.len() >= n);

        if self.max_iterations == 0 {
            return 0;
        }

        // Apply scaling to initial positions.
        if self.scaling {
            for i in 0..n {
                x[i] /= self.var_scales[i];
            }
        }

        let mut g = vec![0.0; n];
        let mut previous = vec![0.0; n];
        let mut d = vec![0.0; n];

        // Scale b if scaling is enabled.
        let b_scaled: Vec<f64> = if self.scaling {
            (0..n).map(|i| b[i] * self.var_scales[i]).collect()
        } else {
            b.to_vec()
        };

        // Build the effective Q for steepest descent (with scaling applied).
        let effective_q = self.build_effective_q(sparse_q);

        let mut counter = 0;
        for _iter in 0..self.max_iterations {
            counter = _iter + 1;
            previous.copy_from_slice(&x[..n]);

            // Compute steepest descent direction and optimal step size.
            let alpha = self.compute_steepest_descent(
                &b_scaled,
                &x[..n],
                &mut g,
                &effective_q,
            );

            // Take unconstrained gradient step (even if alpha is zero or NaN,
            // we still project via VPSC to enforce constraints).
            if !alpha.is_nan() && alpha > 0.0 {
                for i in 0..n {
                    let step = alpha * g[i] / self.var_weights[i];
                    x[i] += step;
                }
            }

            // Build variables for VPSC.
            let mut vars = Vec::with_capacity(n);
            for i in 0..n {
                let desired = if self.var_fixed[i] {
                    self.var_desired[i]
                        / if self.scaling { self.var_scales[i] } else { 1.0 }
                } else {
                    x[i]
                };

                let weight = if self.var_fixed[i] {
                    FIXED_WEIGHT
                } else {
                    self.var_weights[i]
                };

                vars.push(Variable::new(i, desired, weight, 1.0));
            }

            // Build VPSC constraints from specs.
            let vpsc_constraints: Vec<Constraint> = self
                .constraints
                .iter()
                .map(|spec| {
                    let gap = if self.scaling {
                        spec.gap / self.var_scales[spec.left]
                    } else {
                        spec.gap
                    };
                    Constraint::new(spec.left, spec.right, gap, spec.equality)
                })
                .collect();

            // Create solver and project.
            let mut solver = IncSolver::new(vars, vpsc_constraints);
            let constrained_optimum = solver.satisfy().unwrap_or(false);

            // Read back projected positions.
            let positions = solver.final_positions();
            for i in 0..n {
                x[i] = positions[i];
            }

            // Compute step size for convergence check.
            let mut step_size = 0.0;
            for i in 0..n {
                let diff = previous[i] - x[i];
                step_size += diff * diff;
            }

            // If constraints were active, limit step via beta.
            if constrained_optimum {
                for i in 0..n {
                    d[i] = x[i] - previous[i];
                }
                let beta =
                    BETA_SCALE * self.compute_step_size(&g, &d, &effective_q);
                if beta > 0.0 && beta < BETA_UPPER_LIMIT {
                    for i in 0..n {
                        x[i] = previous[i] + beta * d[i];
                    }
                }
            }

            if step_size < self.tolerance {
                break;
            }
        }

        // Unscale result.
        if self.scaling {
            for i in 0..n {
                x[i] *= self.var_scales[i];
            }
        }

        counter
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Build the effective Q matrix incorporating scaling and an optional sparse
    /// term. Returns a dense n*n matrix in row-major order.
    ///
    /// When scaling is enabled: Q_eff = S * Q * S where S = diag(var_scales).
    fn build_effective_q(&self, sparse_q: Option<&SparseMatrix>) -> Vec<f64> {
        let n = self.dense_size;
        let mut q = self.dense_q.clone();

        // Add sparse Q contribution (before scaling).
        if let Some(sq) = sparse_q {
            for i in 0..n {
                for j in 0..n {
                    let val = sq.get(i, j);
                    if val != 0.0 {
                        q[i * n + j] += val;
                    }
                }
            }
        }

        // Apply scaling: Q_eff[i][j] = scale[i] * Q[i][j] * scale[j].
        if self.scaling {
            for i in 0..n {
                for j in 0..n {
                    q[i * n + j] *= self.var_scales[i] * self.var_scales[j];
                }
            }
        }

        q
    }

    /// Compute the steepest descent direction g = b - Q*x and the optimal
    /// unconstrained step size alpha = (g'g) / (2 * g'Qg).
    ///
    /// Returns alpha. Returns 0 if already at optimum (zero gradient).
    /// Returns NaN if g'Qg is degenerate.
    fn compute_steepest_descent(
        &self,
        b: &[f64],
        x: &[f64],
        g: &mut [f64],
        effective_q: &[f64],
    ) -> f64 {
        let n = self.dense_size;

        // g = b - Q * x
        for i in 0..n {
            let mut qx_i = 0.0;
            let row = i * n;
            for j in 0..n {
                qx_i += effective_q[row + j] * x[j];
            }
            g[i] = b[i] - qx_i;
        }

        let gg = inner(g, g);
        if gg < DENOMINATOR_EPSILON {
            return 0.0; // Already at the optimum.
        }

        // Compute g' Q g.
        let mut gqg = 0.0;
        for i in 0..n {
            let row = i * n;
            let mut qg_i = 0.0;
            for j in 0..n {
                qg_i += effective_q[row + j] * g[j];
            }
            gqg += g[i] * qg_i;
        }

        if gqg.abs() < DENOMINATOR_EPSILON {
            return f64::NAN;
        }

        gg / (STEP_DENOM_FACTOR * gqg)
    }

    /// Compute optimal step size along direction d: alpha = (g'd) / (d'Qd).
    fn compute_step_size(&self, g: &[f64], d: &[f64], effective_q: &[f64]) -> f64 {
        let n = self.dense_size;

        let gd = inner(g, d);

        // Compute d' Q d.
        let mut dqd = 0.0;
        for i in 0..n {
            let row = i * n;
            let mut qd_i = 0.0;
            for j in 0..n {
                qd_i += effective_q[row + j] * d[j];
            }
            dqd += d[i] * qd_i;
        }

        if dqd.abs() < DENOMINATOR_EPSILON {
            return 0.0;
        }

        gd / dqd
    }

    /// Compute cost = 2bx - x'Qx.
    pub fn compute_cost(
        &self,
        b: &[f64],
        x: &[f64],
        sparse_q: Option<&SparseMatrix>,
    ) -> f64 {
        let n = self.dense_size;
        let effective_q = self.build_effective_q(sparse_q);

        let bx = inner(b, x);
        let mut xqx = 0.0;
        for i in 0..n {
            let row = i * n;
            let mut qx_i = 0.0;
            for j in 0..n {
                qx_i += effective_q[row + j] * x[j];
            }
            xqx += x[i] * qx_i;
        }

        STEP_DENOM_FACTOR * bx - xqx
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cola::sparse_matrix::{SparseMap, SparseMatrix};
    use crate::vpsc::Dim;

    /// Tight tolerance for the solver to get close to the optimum.
    const TEST_TOLERANCE: f64 = 1e-8;
    /// More iterations so steepest descent converges for simple problems.
    const TEST_MAX_ITERATIONS: usize = 200;
    /// Assertion tolerance for comparing results.
    const ASSERT_TOLERANCE: f64 = 1e-2;

    /// Helper: build an identity matrix of size n as a flat row-major vec.
    fn identity(n: usize) -> Vec<f64> {
        let mut q = vec![0.0; n * n];
        for i in 0..n {
            q[i * n + i] = 1.0;
        }
        q
    }

    /// Helper: build a diagonal matrix from the given diagonal entries.
    fn diagonal(diag: &[f64]) -> Vec<f64> {
        let n = diag.len();
        let mut q = vec![0.0; n * n];
        for i in 0..n {
            q[i * n + i] = diag[i];
        }
        q
    }

    // =======================================================================
    // 1. Unconstrained solve - identity Q
    // =======================================================================

    #[test]
    fn unconstrained_identity_q() {
        // Q = I, b = [3, 7].  Solution: x = Q^{-1} b = [3, 7].
        let q = identity(2);
        let b = [3.0, 7.0];
        let mut x = [0.0, 0.0];

        let mut gp = GradientProjection::new(
            Dim::Horizontal,
            q,
            TEST_TOLERANCE,
            TEST_MAX_ITERATIONS,
            false,
        );
        gp.solve(&b, &mut x);

        assert!(
            (x[0] - 3.0).abs() < ASSERT_TOLERANCE,
            "x[0] = {}, expected 3.0",
            x[0]
        );
        assert!(
            (x[1] - 7.0).abs() < ASSERT_TOLERANCE,
            "x[1] = {}, expected 7.0",
            x[1]
        );
    }

    // =======================================================================
    // 2. Unconstrained solve - diagonal Q
    // =======================================================================

    #[test]
    fn unconstrained_diagonal_q() {
        // Q = diag(2, 5), b = [4, 10].  Solution: x = [2, 2].
        let q = diagonal(&[2.0, 5.0]);
        let b = [4.0, 10.0];
        let mut x = [0.0, 0.0];

        let mut gp = GradientProjection::new(
            Dim::Horizontal,
            q,
            TEST_TOLERANCE,
            TEST_MAX_ITERATIONS,
            false,
        );
        gp.solve(&b, &mut x);

        assert!(
            (x[0] - 2.0).abs() < ASSERT_TOLERANCE,
            "x[0] = {}, expected 2.0",
            x[0]
        );
        assert!(
            (x[1] - 2.0).abs() < ASSERT_TOLERANCE,
            "x[1] = {}, expected 2.0",
            x[1]
        );
    }

    // =======================================================================
    // 3. With constraints - separation forces minimum gap
    // =======================================================================

    #[test]
    fn constrained_separation() {
        // Q = I, b = [5, 5]. Without constraints: x = [5, 5].
        // Add x[0] + 3 <= x[1], forcing x[1] >= x[0] + 3.
        let q = identity(2);
        let b = [5.0, 5.0];
        let mut x = [0.0, 0.0];

        let mut gp = GradientProjection::new(
            Dim::Horizontal,
            q,
            TEST_TOLERANCE,
            TEST_MAX_ITERATIONS,
            false,
        );
        gp.add_constraint(ConstraintSpec {
            left: 0,
            right: 1,
            gap: 3.0,
            equality: false,
        });
        gp.solve(&b, &mut x);

        // The constraint x[1] >= x[0] + 3 should be active.
        assert!(
            x[1] >= x[0] + 3.0 - ASSERT_TOLERANCE,
            "constraint violated: x[0]={}, x[1]={}, gap={}",
            x[0],
            x[1],
            x[1] - x[0]
        );
    }

    // =======================================================================
    // 4. Scaling - solve with and without, same result for well-conditioned
    // =======================================================================

    #[test]
    fn scaling_same_result_well_conditioned() {
        let q = identity(2);
        let b = [3.0, 7.0];

        let mut x_unscaled = [0.0, 0.0];
        let mut gp_unscaled = GradientProjection::new(
            Dim::Horizontal,
            q.clone(),
            TEST_TOLERANCE,
            TEST_MAX_ITERATIONS,
            false,
        );
        gp_unscaled.solve(&b, &mut x_unscaled);

        let mut x_scaled = [0.0, 0.0];
        let mut gp_scaled = GradientProjection::new(
            Dim::Horizontal,
            q,
            TEST_TOLERANCE,
            TEST_MAX_ITERATIONS,
            true,
        );
        gp_scaled.solve(&b, &mut x_scaled);

        for i in 0..2 {
            assert!(
                (x_unscaled[i] - x_scaled[i]).abs() < ASSERT_TOLERANCE,
                "scaled and unscaled differ at {}: {} vs {}",
                i,
                x_unscaled[i],
                x_scaled[i]
            );
        }
    }

    // =======================================================================
    // 5. fix_pos / unfix_pos
    // =======================================================================

    #[test]
    fn fix_pos_holds_position() {
        // Q = I, b = [10, 10]. Fix x[0] at 0.
        // x[0] should stay near 0, x[1] should go to 10.
        let q = identity(2);
        let b = [10.0, 10.0];
        let mut x = [0.0, 0.0];

        let mut gp = GradientProjection::new(
            Dim::Horizontal,
            q,
            TEST_TOLERANCE,
            TEST_MAX_ITERATIONS,
            false,
        );
        gp.fix_pos(0, 0.0);
        gp.solve(&b, &mut x);

        assert!(
            x[0].abs() < ASSERT_TOLERANCE,
            "fixed var moved: x[0] = {}",
            x[0]
        );
        assert!(
            (x[1] - 10.0).abs() < ASSERT_TOLERANCE,
            "free var wrong: x[1] = {}",
            x[1]
        );
    }

    #[test]
    fn unfix_pos_allows_movement() {
        let q = identity(2);
        let b = [10.0, 10.0];
        let mut x = [0.0, 0.0];

        let mut gp = GradientProjection::new(
            Dim::Horizontal,
            q,
            TEST_TOLERANCE,
            TEST_MAX_ITERATIONS,
            false,
        );
        gp.fix_pos(0, 0.0);
        gp.unfix_pos(0);
        gp.solve(&b, &mut x);

        // After unfixing, both should converge to b.
        assert!(
            (x[0] - 10.0).abs() < ASSERT_TOLERANCE,
            "unfixed var should move: x[0] = {}",
            x[0]
        );
    }

    // =======================================================================
    // 6. Zero iterations - returns immediately
    // =======================================================================

    #[test]
    fn zero_iterations_unchanged() {
        let q = identity(2);
        let b = [3.0, 7.0];
        let mut x = [1.0, 2.0];
        let x_before = x;

        let mut gp = GradientProjection::new(
            Dim::Horizontal,
            q,
            TEST_TOLERANCE,
            0, // zero max iterations
            false,
        );
        let iters = gp.solve(&b, &mut x);

        assert_eq!(iters, 0);
        assert_eq!(x, x_before);
    }

    // =======================================================================
    // 7. Already at solution - converges quickly
    // =======================================================================

    #[test]
    fn already_at_solution() {
        // Q = I, b = [3, 7], start at x = [3, 7].
        let q = identity(2);
        let b = [3.0, 7.0];
        let mut x = [3.0, 7.0];

        let mut gp = GradientProjection::new(
            Dim::Horizontal,
            q,
            TEST_TOLERANCE,
            TEST_MAX_ITERATIONS,
            false,
        );
        let iters = gp.solve(&b, &mut x);

        assert!(iters <= 2, "should converge immediately, took {}", iters);
        assert!((x[0] - 3.0).abs() < ASSERT_TOLERANCE);
        assert!((x[1] - 7.0).abs() < ASSERT_TOLERANCE);
    }

    // =======================================================================
    // 8. Steepest descent computation
    // =======================================================================

    #[test]
    fn steepest_descent_gradient_direction() {
        // Q = I, x = [0, 0], b = [3, 7].
        // g = b - Q*x = [3, 7].
        // alpha = g'g / (2 * g'Qg) = 58 / (2*58) = 0.5.
        let gp = GradientProjection::new(
            Dim::Horizontal,
            identity(2),
            TEST_TOLERANCE,
            TEST_MAX_ITERATIONS,
            false,
        );

        let effective_q = gp.build_effective_q(None);
        let b = [3.0, 7.0];
        let x = [0.0, 0.0];
        let mut g = [0.0, 0.0];

        let alpha = gp.compute_steepest_descent(&b, &x, &mut g, &effective_q);

        assert!(
            (g[0] - 3.0).abs() < 1e-10,
            "g[0] = {}, expected 3.0",
            g[0]
        );
        assert!(
            (g[1] - 7.0).abs() < 1e-10,
            "g[1] = {}, expected 7.0",
            g[1]
        );
        assert!(
            (alpha - 0.5).abs() < 1e-10,
            "alpha = {}, expected 0.5",
            alpha
        );
    }

    // =======================================================================
    // 9. Step size computation
    // =======================================================================

    #[test]
    fn step_size_known_values() {
        // Q = I, g = [1, 0], d = [1, 0].
        // step = g'd / d'Qd = 1 / 1 = 1.
        let gp = GradientProjection::new(
            Dim::Horizontal,
            identity(2),
            TEST_TOLERANCE,
            TEST_MAX_ITERATIONS,
            false,
        );

        let effective_q = gp.build_effective_q(None);
        let g = [1.0, 0.0];
        let d = [1.0, 0.0];

        let step = gp.compute_step_size(&g, &d, &effective_q);
        assert!(
            (step - 1.0).abs() < 1e-10,
            "step = {}, expected 1.0",
            step
        );
    }

    #[test]
    fn step_size_diagonal_q() {
        // Q = diag(2, 4), g = [2, 4], d = [1, 1].
        // g'd = 2 + 4 = 6.
        // d'Qd = 1*2*1 + 1*4*1 = 6.
        // step = 6 / 6 = 1.
        let gp = GradientProjection::new(
            Dim::Horizontal,
            diagonal(&[2.0, 4.0]),
            TEST_TOLERANCE,
            TEST_MAX_ITERATIONS,
            false,
        );

        let effective_q = gp.build_effective_q(None);
        let g = [2.0, 4.0];
        let d = [1.0, 1.0];

        let step = gp.compute_step_size(&g, &d, &effective_q);
        assert!(
            (step - 1.0).abs() < 1e-10,
            "step = {}, expected 1.0",
            step
        );
    }

    // =======================================================================
    // 10. Beta limiting - constraint active limits step
    // =======================================================================

    #[test]
    fn beta_limiting_with_constraint() {
        // Q = I, b = [0, 0]. Unconstrained: x = [0, 0].
        // Start at x = [-5, 5]. Add x[0] + 20 <= x[1].
        // The solver should produce positions respecting the gap.
        let q = identity(2);
        let b = [0.0, 0.0];
        let mut x = [-5.0, 5.0];

        let mut gp = GradientProjection::new(
            Dim::Horizontal,
            q,
            TEST_TOLERANCE,
            TEST_MAX_ITERATIONS,
            false,
        );
        gp.add_constraint(ConstraintSpec {
            left: 0,
            right: 1,
            gap: 20.0,
            equality: false,
        });
        gp.solve(&b, &mut x);

        // Constraint must be satisfied.
        assert!(
            x[1] >= x[0] + 20.0 - ASSERT_TOLERANCE,
            "constraint violated: gap = {}",
            x[1] - x[0]
        );
    }

    // =======================================================================
    // 11. Cost computation
    // =======================================================================

    #[test]
    fn cost_computation_known_values() {
        // Q = I, b = [1, 1], x = [1, 1].
        // cost = 2*b'x - x'Qx = 2*2 - 2 = 2.
        let gp = GradientProjection::new(
            Dim::Horizontal,
            identity(2),
            TEST_TOLERANCE,
            TEST_MAX_ITERATIONS,
            false,
        );

        let cost = gp.compute_cost(&[1.0, 1.0], &[1.0, 1.0], None);
        assert!(
            (cost - 2.0).abs() < 1e-10,
            "cost = {}, expected 2.0",
            cost
        );
    }

    #[test]
    fn cost_computation_zero_x() {
        let gp = GradientProjection::new(
            Dim::Horizontal,
            identity(2),
            TEST_TOLERANCE,
            TEST_MAX_ITERATIONS,
            false,
        );

        let cost = gp.compute_cost(&[1.0, 1.0], &[0.0, 0.0], None);
        assert!((cost - 0.0).abs() < 1e-10, "cost = {}, expected 0.0", cost);
    }

    // =======================================================================
    // 12. Larger system - 5x5 diagonal with constraints
    // =======================================================================

    #[test]
    fn larger_system_5x5_diagonal() {
        let n = 5;
        let diag_vals: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let q = diagonal(&diag_vals);
        // b[i] = diag[i] * 10 so unconstrained solution is x[i] = 10 for all i.
        let b: Vec<f64> = diag_vals.iter().map(|d| d * 10.0).collect();
        let mut x = vec![0.0; n];

        let mut gp = GradientProjection::new(
            Dim::Horizontal,
            q,
            TEST_TOLERANCE,
            500,
            false,
        );
        gp.solve(&b, &mut x);

        for i in 0..n {
            assert!(
                (x[i] - 10.0).abs() < ASSERT_TOLERANCE,
                "x[{}] = {}, expected 10.0",
                i,
                x[i]
            );
        }
    }

    #[test]
    fn larger_system_5x5_with_chain_constraints() {
        // 5 variables, Q = I, b = [0; 5], start spread out to give gradient
        // something to work with once projected.
        // Chain: x[0]+5<=x[1], x[1]+5<=x[2], x[2]+5<=x[3], x[3]+5<=x[4].
        // Optimal: minimise sum(x_i^2) s.t. chain.
        // Should spread symmetrically around 0: [-10, -5, 0, 5, 10].
        let n = 5;
        let q = identity(n);
        let b = vec![0.0; n];
        // Start at positions that already satisfy the chain but are not optimal.
        let mut x = vec![0.0, 5.0, 10.0, 15.0, 20.0];

        let chain_gap = 5.0;
        let mut gp = GradientProjection::new(
            Dim::Horizontal,
            q,
            1e-8,
            1000,
            false,
        );
        for i in 0..(n - 1) {
            gp.add_constraint(ConstraintSpec {
                left: i,
                right: i + 1,
                gap: chain_gap,
                equality: false,
            });
        }
        gp.solve(&b, &mut x);

        // Verify chain constraints are satisfied.
        for i in 0..(n - 1) {
            assert!(
                x[i + 1] >= x[i] + chain_gap - ASSERT_TOLERANCE,
                "chain constraint violated at {}: x[{}]={}, x[{}]={}",
                i,
                i,
                x[i],
                i + 1,
                x[i + 1]
            );
        }

        // Check symmetry: positions should be approximately -10, -5, 0, 5, 10.
        let expected = [-10.0, -5.0, 0.0, 5.0, 10.0];
        for i in 0..n {
            assert!(
                (x[i] - expected[i]).abs() < 1.0,
                "x[{}] = {}, expected ~{}",
                i,
                x[i],
                expected[i]
            );
        }
    }

    // =======================================================================
    // 13. Convergence - reasonable iteration count
    // =======================================================================

    #[test]
    fn convergence_iteration_count() {
        // For identity Q with steepest descent, convergence rate is 0.5 per
        // iteration. With tolerance 1e-8, need about 27 iterations.
        let q = identity(2);
        let b = [3.0, 7.0];
        let mut x = [0.0, 0.0];

        let mut gp = GradientProjection::new(
            Dim::Horizontal,
            q,
            TEST_TOLERANCE,
            TEST_MAX_ITERATIONS,
            false,
        );
        let iters = gp.solve(&b, &mut x);

        // Steepest descent on identity converges linearly with rate 0.5.
        // Should need roughly -log2(tolerance / initial_step_sq) iterations.
        assert!(
            iters <= 50,
            "took {} iterations, expected fewer than 50",
            iters
        );
        // Verify it actually converged.
        assert!(
            (x[0] - 3.0).abs() < ASSERT_TOLERANCE,
            "x[0] = {}, expected 3.0",
            x[0]
        );
    }

    // =======================================================================
    // 14. Equality constraint
    // =======================================================================

    #[test]
    fn equality_constraint() {
        // Q = I, b = [0, 10]. Add x[0] + 5 == x[1].
        let q = identity(2);
        let b = [0.0, 10.0];
        let mut x = [0.0, 0.0];

        let mut gp = GradientProjection::new(
            Dim::Horizontal,
            q,
            TEST_TOLERANCE,
            TEST_MAX_ITERATIONS,
            false,
        );
        gp.add_constraint(ConstraintSpec {
            left: 0,
            right: 1,
            gap: 5.0,
            equality: true,
        });
        gp.solve(&b, &mut x);

        // Equality: x[1] = x[0] + 5.
        assert!(
            (x[1] - x[0] - 5.0).abs() < ASSERT_TOLERANCE,
            "equality violated: x[0]={}, x[1]={}, gap={}",
            x[0],
            x[1],
            x[1] - x[0]
        );
    }

    // =======================================================================
    // 15. Solve with sparse Q
    // =======================================================================

    #[test]
    fn solve_with_sparse_q() {
        // Dense Q = 0, sparse Q = I. b = [3, 7]. Solution: x = [3, 7].
        let n = 2;
        let dense_q = vec![0.0; n * n];
        let b = [3.0, 7.0];
        let mut x = [0.0, 0.0];

        let mut smap = SparseMap::new(n);
        smap.set(0, 0, 1.0);
        smap.set(1, 1, 1.0);
        let sparse = SparseMatrix::from_sparse_map(&smap);

        let mut gp = GradientProjection::new(
            Dim::Horizontal,
            dense_q,
            TEST_TOLERANCE,
            TEST_MAX_ITERATIONS,
            false,
        );
        gp.solve_with_sparse(&b, &mut x, Some(&sparse));

        assert!(
            (x[0] - 3.0).abs() < ASSERT_TOLERANCE,
            "x[0] = {}, expected 3.0",
            x[0]
        );
        assert!(
            (x[1] - 7.0).abs() < ASSERT_TOLERANCE,
            "x[1] = {}, expected 7.0",
            x[1]
        );
    }
}
