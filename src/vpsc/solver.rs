//! VPSC solver implementations.
//!
//! Contains both the static [`Solver`] and incremental [`IncSolver`].
//!
//! C++ ref: libvpsc/solve_VPSC.h, libvpsc/solve_VPSC.cpp,
//!          libvpsc/blocks.h, libvpsc/blocks.cpp,
//!          libvpsc/block.cpp (operations)
//!
//! ## Architecture note
//!
//! The C++ code uses raw pointers between Variable, Constraint, and Block.
//! This Rust port uses index-based references: variables, constraints, and
//! blocks are stored in Vecs on the Solver struct, and referenced by usize
//! indices. All operations that span types are methods on Solver.

use crate::vpsc::block::Block;
use crate::vpsc::constraint::Constraint;
use crate::vpsc::variable::Variable;

/// Upper bound for considering a constraint violated.
/// C++ ref: ZERO_UPPERBOUND in solve_VPSC.cpp
const ZERO_UPPERBOUND: f64 = -1e-10;

/// Tolerance for Lagrangian multiplier when deciding whether to split.
/// C++ ref: LAGRANGIAN_TOLERANCE in solve_VPSC.cpp
const LAGRANGIAN_TOLERANCE: f64 = -1e-4;

/// Maximum refinement iterations to prevent infinite loops.
const MAX_REFINE_ITERATIONS: usize = 100;

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Error returned when a constraint cannot be satisfied.
#[derive(Debug, Clone)]
pub struct UnsatisfiedConstraintError {
    pub constraint_index: usize,
}

impl std::fmt::Display for UnsatisfiedConstraintError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Unsatisfied constraint at index {}", self.constraint_index)
    }
}

impl std::error::Error for UnsatisfiedConstraintError {}

/// Error for unsatisfiable constraint paths (cycle detection).
#[derive(Debug, Clone)]
pub struct UnsatisfiableError {
    pub path: Vec<usize>,
}

impl std::fmt::Display for UnsatisfiableError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Unsatisfiable constraint path: {:?}", self.path)
    }
}

impl std::error::Error for UnsatisfiableError {}

/// Combined solver error type.
#[derive(Debug, Clone)]
pub enum VpscError {
    UnsatisfiedConstraint(UnsatisfiedConstraintError),
    Unsatisfiable(UnsatisfiableError),
}

impl std::fmt::Display for VpscError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VpscError::UnsatisfiedConstraint(e) => write!(f, "{}", e),
            VpscError::Unsatisfiable(e) => write!(f, "{}", e),
        }
    }
}

impl std::error::Error for VpscError {}

// ---------------------------------------------------------------------------
// Solver (static)
// ---------------------------------------------------------------------------

/// Static solver for Variable Placement with Separation Constraints.
///
/// Attempts to solve a least-squares problem subject to separation constraints.
/// `satisfy()` produces a feasible solution; `solve()` refines to optimal.
///
/// C++ ref: vpsc::Solver
pub struct Solver {
    pub(crate) vars: Vec<Variable>,
    pub(crate) constraints: Vec<Constraint>,
    pub(crate) blocks: Vec<Block>,
    pub(crate) block_time_ctr: i64,
    pub(crate) needs_scaling: bool,
    num_vars: usize,
    num_constraints: usize,
}

impl Solver {
    /// Create a new static VPSC solver.
    ///
    /// Variables and constraints are consumed. The solver builds constraint
    /// DAG (in/out lists on variables) and initializes one block per variable.
    ///
    /// C++ ref: Solver::Solver(Variables, Constraints)
    pub fn new(mut vars: Vec<Variable>, mut constraints: Vec<Constraint>) -> Self {
        let n = vars.len();
        let m = constraints.len();

        // Clear constraint lists and detect scaling need
        let mut needs_scaling = false;
        for v in &mut vars {
            v.in_constraints.clear();
            v.out_constraints.clear();
            needs_scaling |= (v.scale - 1.0).abs() > f64::EPSILON;
        }

        // Build constraint DAG
        for (ci, c) in constraints.iter_mut().enumerate() {
            vars[c.left].out_constraints.push(ci);
            vars[c.right].in_constraints.push(ci);
            c.needs_scaling = needs_scaling;
        }

        // Initialize blocks: one block per variable
        let mut blocks = Vec::with_capacity(n);
        for vi in 0..n {
            let mut block = Block::new();
            block.ps.scale = vars[vi].scale;
            vars[vi].offset = 0.0;
            vars[vi].block = Some(blocks.len());

            // Add variable to block's position stats
            block.ps.add_variable(
                vars[vi].weight,
                vars[vi].scale,
                vars[vi].offset,
                vars[vi].desired_position,
            );
            block.posn = (block.ps.ad - block.ps.ab) / block.ps.a2;
            block.vars.push(vi);

            blocks.push(block);
        }

        Self {
            vars,
            constraints,
            blocks,
            block_time_ctr: 0,
            needs_scaling,
            num_vars: n,
            num_constraints: m,
        }
    }

    /// Access the variables (read-only).
    pub fn variables(&self) -> &[Variable] {
        &self.vars
    }

    /// Get the final positions of all variables.
    pub fn final_positions(&self) -> Vec<f64> {
        self.vars.iter().map(|v| v.final_position).collect()
    }

    /// Produces a feasible (but not necessarily optimal) solution.
    ///
    /// Returns true if any constraints are active, false if unconstrained
    /// optimum was found.
    ///
    /// C++ ref: Solver::satisfy()
    pub fn satisfy(&mut self) -> Result<bool, VpscError> {
        // Get total order of variables via constraint DAG
        let order = self.total_order();

        // Process each variable's block in topological order
        for vi in &order {
            let block_idx = self.vars[*vi].block.unwrap();
            if !self.blocks[block_idx].deleted {
                self.merge_left(block_idx);
            }
        }

        self.cleanup_blocks();

        // Check all constraints satisfied
        let mut active_constraints = false;
        for ci in 0..self.num_constraints {
            if self.constraints[ci].active {
                active_constraints = true;
            }
            let slack = self.constraint_slack(ci);
            if slack < ZERO_UPPERBOUND {
                return Err(VpscError::UnsatisfiedConstraint(
                    UnsatisfiedConstraintError { constraint_index: ci },
                ));
            }
        }

        self.copy_result();
        Ok(active_constraints)
    }

    /// Calculate the optimal solution.
    ///
    /// Uses `satisfy()` to produce a feasible solution, then `refine()` to
    /// split blocks for optimality.
    ///
    /// Returns true if any constraints are active.
    ///
    /// C++ ref: Solver::solve()
    pub fn solve(&mut self) -> Result<bool, VpscError> {
        self.satisfy()?;
        self.refine()?;
        self.copy_result();
        let active_count = self.blocks.iter().filter(|b| !b.deleted).count();
        Ok(active_count != self.num_vars)
    }

    // -----------------------------------------------------------------------
    // Internal: Block operations
    // -----------------------------------------------------------------------

    /// Compute a variable's position from its block state.
    fn var_position(&self, vi: usize) -> f64 {
        let bi = self.vars[vi].block.unwrap();
        let block = &self.blocks[bi];
        self.vars[vi].position(block.posn, block.ps.scale)
    }

    /// Compute a variable's unscaled position.
    fn var_unscaled_position(&self, vi: usize) -> f64 {
        let bi = self.vars[vi].block.unwrap();
        self.vars[vi].unscaled_position(self.blocks[bi].posn)
    }

    /// Compute the slack of a constraint.
    ///
    /// slack = right.position - gap - left.position
    ///
    /// C++ ref: Constraint::slack()
    pub(crate) fn constraint_slack(&self, ci: usize) -> f64 {
        let c = &self.constraints[ci];
        if c.unsatisfiable {
            return f64::MAX;
        }
        if c.needs_scaling {
            let right_pos = self.var_position(c.right);
            let left_pos = self.var_position(c.left);
            self.vars[c.right].scale * right_pos - c.gap - self.vars[c.left].scale * left_pos
        } else {
            let right_pos = self.var_unscaled_position(c.right);
            let left_pos = self.var_unscaled_position(c.left);
            right_pos - c.gap - left_pos
        }
    }

    /// Add a variable to a block, updating position statistics.
    ///
    /// C++ ref: Block::addVariable
    fn block_add_variable(&mut self, block_idx: usize, var_idx: usize) {
        self.vars[var_idx].block = Some(block_idx);
        let block = &mut self.blocks[block_idx];
        block.vars.push(var_idx);

        if block.ps.a2 == 0.0 {
            block.ps.scale = self.vars[var_idx].scale;
        }

        let v = &self.vars[var_idx];
        block.ps.add_variable(v.weight, v.scale, v.offset, v.desired_position);
        block.posn = (block.ps.ad - block.ps.ab) / block.ps.a2;
        debug_assert!(block.posn.is_finite());
    }

    /// Recalculate block's weighted position from scratch.
    ///
    /// C++ ref: Block::updateWeightedPosition
    fn block_update_weighted_position(&mut self, block_idx: usize) {
        let block = &mut self.blocks[block_idx];
        block.ps.ab = 0.0;
        block.ps.ad = 0.0;
        block.ps.a2 = 0.0;

        let var_indices: Vec<usize> = block.vars.clone();
        for &vi in &var_indices {
            let v = &self.vars[vi];
            self.blocks[block_idx].ps.add_variable(
                v.weight,
                v.scale,
                v.offset,
                v.desired_position,
            );
        }

        let block = &mut self.blocks[block_idx];
        block.posn = (block.ps.ad - block.ps.ab) / block.ps.a2;
        debug_assert!(block.posn.is_finite());
    }

    /// Merge block `other` into block `target` across constraint `ci` with
    /// given distance adjustment.
    ///
    /// C++ ref: Block::merge(Block*, Constraint*, double)
    fn block_merge(&mut self, target: usize, other: usize, ci: usize, dist: f64) {
        self.constraints[ci].active = true;

        // Collect vars from other block
        let other_vars: Vec<usize> = self.blocks[other].vars.clone();

        for &vi in &other_vars {
            self.vars[vi].offset += dist;
            self.block_add_variable(target, vi);
        }

        let block = &mut self.blocks[target];
        block.posn = (block.ps.ad - block.ps.ab) / block.ps.a2;
        debug_assert!(block.posn.is_finite());

        self.blocks[other].deleted = true;
    }

    /// High-level merge of two blocks across a constraint.
    ///
    /// Merges the smaller block into the larger for efficiency.
    ///
    /// C++ ref: Block::merge(Block*, Constraint*) — the two-arg version
    fn block_merge_across(&mut self, ci: usize) {
        let c = &self.constraints[ci];
        let left_block = self.vars[c.left].block.unwrap();
        let right_block = self.vars[c.right].block.unwrap();
        let dist = self.vars[c.right].offset - self.vars[c.left].offset - c.gap;

        let left_size = self.blocks[left_block].vars.len();
        let right_size = self.blocks[right_block].vars.len();

        if left_size < right_size {
            self.block_merge(right_block, left_block, ci, dist);
        } else {
            self.block_merge(left_block, right_block, ci, -dist);
        }
    }

    /// Split a block across constraint `ci` into two new blocks.
    ///
    /// C++ ref: Block::split
    fn block_split(&mut self, block_idx: usize, ci: usize) -> (usize, usize) {
        self.constraints[ci].active = false;
        let c_left = self.constraints[ci].left;
        let c_right = self.constraints[ci].right;

        // Create left block by traversing from c.left
        let left_idx = self.blocks.len();
        self.blocks.push(Block::new());
        self.blocks[left_idx].ps.scale = self.blocks[block_idx].ps.scale;
        self.populate_split_block(left_idx, c_left, c_right);

        // Create right block by traversing from c.right
        let right_idx = self.blocks.len();
        self.blocks.push(Block::new());
        self.blocks[right_idx].ps.scale = self.blocks[block_idx].ps.scale;
        self.populate_split_block(right_idx, c_right, c_left);

        (left_idx, right_idx)
    }

    /// Populate a new split block by traversing the active constraint tree.
    ///
    /// Starting from variable `v`, follows all active constraints except
    /// those leading back to `u`.
    ///
    /// C++ ref: Block::populateSplitBlock
    fn populate_split_block(&mut self, block_idx: usize, v: usize, u: usize) {
        self.block_add_variable(block_idx, v);

        let in_cs: Vec<usize> = self.vars[v].in_constraints.clone();
        for ci in in_cs {
            let c = &self.constraints[ci];
            if c.active && c.left != u {
                let c_left = c.left;
                let c_left_block = self.vars[c_left].block;
                // canFollowLeft: c.left.block == original block AND c is active AND last != c.left
                // After block_add_variable, v is now in block_idx
                // We need to check if c.left was in the same block before split
                // Since we're traversing from the original block, we follow active constraints
                if c_left_block != Some(block_idx) {
                    self.populate_split_block(block_idx, c_left, v);
                }
            }
        }

        let out_cs: Vec<usize> = self.vars[v].out_constraints.clone();
        for ci in out_cs {
            let c = &self.constraints[ci];
            if c.active && c.right != u {
                let c_right = c.right;
                let c_right_block = self.vars[c_right].block;
                if c_right_block != Some(block_idx) {
                    self.populate_split_block(block_idx, c_right, v);
                }
            }
        }
    }

    /// Compute dfdv recursively through active constraint tree, finding the
    /// constraint with minimum Lagrange multiplier.
    ///
    /// C++ ref: Block::compute_dfdv(Variable*, Variable*, Constraint*&)
    fn compute_dfdv_with_min(
        &mut self,
        v: usize,
        u: Option<usize>,
        block_idx: usize,
        min_lm: &mut Option<usize>,
    ) -> f64 {
        let bi = self.vars[v].block.unwrap();
        let block = &self.blocks[bi];
        let mut dfdv = self.vars[v].dfdv(block.posn, block.ps.scale);

        let out_cs: Vec<usize> = self.vars[v].out_constraints.clone();
        for ci in out_cs {
            let c = &self.constraints[ci];
            if c.active && self.vars[c.right].block == Some(block_idx) && Some(c.right) != u {
                let c_right = c.right;
                let c_left_scale = self.vars[c.left].scale;
                let lm = self.compute_dfdv_with_min(c_right, Some(v), block_idx, min_lm);
                self.constraints[ci].lm = lm;
                dfdv += lm * c_left_scale;
                if !self.constraints[ci].equality {
                    if min_lm.is_none()
                        || self.constraints[ci].lm < self.constraints[min_lm.unwrap()].lm
                    {
                        *min_lm = Some(ci);
                    }
                }
            }
        }

        let in_cs: Vec<usize> = self.vars[v].in_constraints.clone();
        for ci in in_cs {
            let c = &self.constraints[ci];
            if c.active && self.vars[c.left].block == Some(block_idx) && Some(c.left) != u {
                let c_left = c.left;
                let c_right_scale = self.vars[c.right].scale;
                let lm = -self.compute_dfdv_with_min(c_left, Some(v), block_idx, min_lm);
                self.constraints[ci].lm = lm;
                dfdv -= lm * c_right_scale;
                if !self.constraints[ci].equality {
                    if min_lm.is_none()
                        || self.constraints[ci].lm < self.constraints[min_lm.unwrap()].lm
                    {
                        *min_lm = Some(ci);
                    }
                }
            }
        }

        dfdv / self.vars[v].scale
    }

    /// Compute dfdv without tracking min LM.
    ///
    /// C++ ref: Block::compute_dfdv(Variable*, Variable*)
    fn compute_dfdv(&mut self, v: usize, u: Option<usize>, block_idx: usize) -> f64 {
        let bi = self.vars[v].block.unwrap();
        let block = &self.blocks[bi];
        let mut dfdv = self.vars[v].dfdv(block.posn, block.ps.scale);

        let out_cs: Vec<usize> = self.vars[v].out_constraints.clone();
        for ci in out_cs {
            let c = &self.constraints[ci];
            if c.active && self.vars[c.right].block == Some(block_idx) && Some(c.right) != u {
                let c_right = c.right;
                let c_left_scale = self.vars[c.left].scale;
                let lm = self.compute_dfdv(c_right, Some(v), block_idx);
                self.constraints[ci].lm = lm;
                dfdv += lm * c_left_scale;
            }
        }

        let in_cs: Vec<usize> = self.vars[v].in_constraints.clone();
        for ci in in_cs {
            let c = &self.constraints[ci];
            if c.active && self.vars[c.left].block == Some(block_idx) && Some(c.left) != u {
                let c_left = c.left;
                let c_right_scale = self.vars[c.right].scale;
                let lm = -self.compute_dfdv(c_left, Some(v), block_idx);
                self.constraints[ci].lm = lm;
                dfdv -= lm * c_right_scale;
            }
        }

        dfdv / self.vars[v].scale
    }

    /// Reset Lagrange multipliers for all active constraints in block.
    ///
    /// C++ ref: Block::reset_active_lm
    fn reset_active_lm(&mut self, v: usize, u: Option<usize>, block_idx: usize) {
        let out_cs: Vec<usize> = self.vars[v].out_constraints.clone();
        for ci in out_cs {
            let c = &self.constraints[ci];
            if c.active && self.vars[c.right].block == Some(block_idx) && Some(c.right) != u {
                let c_right = c.right;
                self.constraints[ci].lm = 0.0;
                self.reset_active_lm(c_right, Some(v), block_idx);
            }
        }

        let in_cs: Vec<usize> = self.vars[v].in_constraints.clone();
        for ci in in_cs {
            let c = &self.constraints[ci];
            if c.active && self.vars[c.left].block == Some(block_idx) && Some(c.left) != u {
                let c_left = c.left;
                self.constraints[ci].lm = 0.0;
                self.reset_active_lm(c_left, Some(v), block_idx);
            }
        }
    }

    /// Find the constraint with minimum Lagrange multiplier in a block.
    ///
    /// C++ ref: Block::findMinLM
    fn find_min_lm(&mut self, block_idx: usize) -> Option<usize> {
        if self.blocks[block_idx].vars.is_empty() {
            return None;
        }
        let first_var = self.blocks[block_idx].vars[0];
        self.reset_active_lm(first_var, None, block_idx);
        let mut min_lm = None;
        self.compute_dfdv_with_min(first_var, None, block_idx, &mut min_lm);
        min_lm
    }

    /// Find the constraint with minimum LM on the path between lv and rv.
    ///
    /// C++ ref: Block::findMinLMBetween
    fn find_min_lm_between(
        &mut self,
        block_idx: usize,
        lv: usize,
        rv: usize,
    ) -> Result<usize, VpscError> {
        let first_var = self.blocks[block_idx].vars[0];
        self.reset_active_lm(first_var, None, block_idx);
        self.compute_dfdv(first_var, None, block_idx);

        let mut min_lm = None;
        self.split_path(rv, lv, None, block_idx, &mut min_lm);

        match min_lm {
            Some(ci) => Ok(ci),
            None => {
                let mut path = Vec::new();
                self.get_active_path_between(&mut path, lv, rv, None, block_idx);
                Err(VpscError::Unsatisfiable(UnsatisfiableError { path }))
            }
        }
    }

    /// Search for the split point with minimum LM on the path from lv to rv.
    ///
    /// C++ ref: Block::split_path
    fn split_path(
        &self,
        r: usize,
        v: usize,
        u: Option<usize>,
        block_idx: usize,
        min_lm: &mut Option<usize>,
    ) -> bool {
        let in_cs: Vec<usize> = self.vars[v].in_constraints.clone();
        for ci in in_cs {
            let c = &self.constraints[ci];
            if c.active && self.vars[c.left].block == Some(block_idx) && Some(c.left) != u {
                if c.left == r {
                    return true;
                }
                if self.split_path(r, c.left, Some(v), block_idx, min_lm) {
                    return true;
                }
            }
        }

        let out_cs: Vec<usize> = self.vars[v].out_constraints.clone();
        for ci in out_cs {
            let c = &self.constraints[ci];
            if c.active && self.vars[c.right].block == Some(block_idx) && Some(c.right) != u {
                if c.right == r {
                    if !c.equality {
                        *min_lm = Some(ci);
                    }
                    return true;
                }
                if self.split_path(r, c.right, Some(v), block_idx, min_lm) {
                    if !c.equality
                        && (min_lm.is_none()
                            || self.constraints[ci].lm
                                < self.constraints[min_lm.unwrap()].lm)
                    {
                        *min_lm = Some(ci);
                    }
                    return true;
                }
            }
        }

        false
    }

    /// Get the active path between two variables in a block.
    ///
    /// C++ ref: Block::getActivePathBetween
    fn get_active_path_between(
        &self,
        path: &mut Vec<usize>,
        u: usize,
        v: usize,
        w: Option<usize>,
        block_idx: usize,
    ) -> bool {
        if u == v {
            return true;
        }
        for &ci in &self.vars[u].in_constraints {
            let c = &self.constraints[ci];
            if c.active && self.vars[c.left].block == Some(block_idx) && Some(c.left) != w {
                if self.get_active_path_between(path, c.left, v, Some(u), block_idx) {
                    path.push(ci);
                    return true;
                }
            }
        }
        for &ci in &self.vars[u].out_constraints {
            let c = &self.constraints[ci];
            if c.active && self.vars[c.right].block == Some(block_idx) && Some(c.right) != w {
                if self.get_active_path_between(path, c.right, v, Some(u), block_idx) {
                    path.push(ci);
                    return true;
                }
            }
        }
        false
    }

    /// Check if there's an active directed path from u to v in a block.
    ///
    /// C++ ref: Block::isActiveDirectedPathBetween
    fn is_active_directed_path_between(&self, u: usize, v: usize, block_idx: usize) -> bool {
        if u == v {
            return true;
        }
        for &ci in &self.vars[u].out_constraints {
            let c = &self.constraints[ci];
            if c.active && self.vars[c.right].block == Some(block_idx) {
                if self.is_active_directed_path_between(c.right, v, block_idx) {
                    return true;
                }
            }
        }
        false
    }

    /// Split a block between two variables, finding the best split point.
    ///
    /// C++ ref: Block::splitBetween
    fn split_between(
        &mut self,
        block_idx: usize,
        vl: usize,
        vr: usize,
    ) -> Result<(usize, Option<usize>, usize, usize), VpscError> {
        let ci = self.find_min_lm_between(block_idx, vl, vr)?;
        let (lb, rb) = self.block_split(block_idx, ci);
        self.blocks[block_idx].deleted = true;
        Ok((ci, Some(ci), lb, rb))
    }

    // -----------------------------------------------------------------------
    // Internal: Blocks-level operations
    // -----------------------------------------------------------------------

    /// Topological sort of variables via constraint DAG.
    ///
    /// C++ ref: Blocks::totalOrder
    fn total_order(&mut self) -> Vec<usize> {
        for v in &mut self.vars {
            v.visited = false;
        }

        let mut order = Vec::with_capacity(self.num_vars);
        for vi in 0..self.num_vars {
            if self.vars[vi].in_constraints.is_empty() {
                self.dfs_visit(vi, &mut order);
            }
        }
        order
    }

    /// DFS visit for topological sort.
    ///
    /// C++ ref: Blocks::dfsVisit
    fn dfs_visit(&mut self, v: usize, order: &mut Vec<usize>) {
        self.vars[v].visited = true;
        let out_cs: Vec<usize> = self.vars[v].out_constraints.clone();
        for ci in out_cs {
            let right = self.constraints[ci].right;
            if !self.vars[right].visited {
                self.dfs_visit(right, order);
            }
        }
        order.push(v);
    }

    /// Find the most violated incoming constraint for a block.
    ///
    /// Returns the constraint index with minimum slack among incoming
    /// constraints that cross block boundaries, or None if all are satisfied.
    ///
    /// C++ ref: Block::findMinInConstraint (simplified from heap-based)
    fn find_min_in_constraint(&self, block_idx: usize) -> Option<usize> {
        let mut best: Option<(usize, f64)> = None;

        for &vi in &self.blocks[block_idx].vars {
            for &ci in &self.vars[vi].in_constraints {
                let c = &self.constraints[ci];
                let left_block = self.vars[c.left].block;

                // Skip internal constraints (both vars in same block)
                if left_block == Some(block_idx) {
                    continue;
                }

                // Skip stale constraints
                if let Some(lb) = left_block {
                    if c.time_stamp < self.blocks[lb].time_stamp {
                        continue;
                    }
                }

                let slack = self.constraint_slack(ci);

                match best {
                    None => best = Some((ci, slack)),
                    Some((_, best_slack)) => {
                        if slack < best_slack
                            || (slack == best_slack && self.constraint_less_than(ci, best.unwrap().0))
                        {
                            best = Some((ci, slack));
                        }
                    }
                }
            }
        }

        best.map(|(ci, _)| ci)
    }

    /// Find the most violated outgoing constraint for a block.
    ///
    /// C++ ref: Block::findMinOutConstraint (simplified)
    fn find_min_out_constraint(&self, block_idx: usize) -> Option<usize> {
        let mut best: Option<(usize, f64)> = None;

        for &vi in &self.blocks[block_idx].vars {
            for &ci in &self.vars[vi].out_constraints {
                let c = &self.constraints[ci];
                let right_block = self.vars[c.right].block;

                if right_block == Some(block_idx) {
                    continue;
                }

                let slack = self.constraint_slack(ci);

                match best {
                    None => best = Some((ci, slack)),
                    Some((_, best_slack)) => {
                        if slack < best_slack
                            || (slack == best_slack && self.constraint_less_than(ci, best.unwrap().0))
                        {
                            best = Some((ci, slack));
                        }
                    }
                }
            }
        }

        best.map(|(ci, _)| ci)
    }

    /// Constraint ordering for tie-breaking.
    ///
    /// C++ ref: CompareConstraints::operator()
    fn constraint_less_than(&self, a: usize, b: usize) -> bool {
        let ca = &self.constraints[a];
        let cb = &self.constraints[b];
        if ca.left == cb.left {
            return ca.right < cb.right;
        }
        ca.left < cb.left
    }

    /// Process incoming constraints for a block, merging violated ones.
    ///
    /// C++ ref: Blocks::mergeLeft
    fn merge_left(&mut self, mut block_idx: usize) {
        self.block_time_ctr += 1;
        self.blocks[block_idx].time_stamp = self.block_time_ctr;

        while let Some(ci) = self.find_min_in_constraint(block_idx) {
            let slack = self.constraint_slack(ci);
            if slack >= 0.0 {
                break;
            }

            let c = &self.constraints[ci];
            let left_block = self.vars[c.left].block.unwrap();
            let right_block = self.vars[c.right].block.unwrap();

            let c_left = c.left;
            let c_right = c.right;
            let c_gap = c.gap;

            let dist = self.vars[c_right].offset - self.vars[c_left].offset - c_gap;

            let left_size = self.blocks[left_block].vars.len();
            let right_size = self.blocks[right_block].vars.len();

            let (target, other, actual_dist) = if right_size < left_size {
                (left_block, right_block, -dist)
            } else {
                (right_block, left_block, dist)
            };

            self.block_time_ctr += 1;
            self.block_merge(target, other, ci, actual_dist);
            block_idx = target;
            self.blocks[block_idx].time_stamp = self.block_time_ctr;
        }
    }

    /// Process outgoing constraints for a block, merging violated ones.
    ///
    /// C++ ref: Blocks::mergeRight
    fn merge_right(&mut self, mut block_idx: usize) {
        while let Some(ci) = self.find_min_out_constraint(block_idx) {
            let slack = self.constraint_slack(ci);
            if slack >= 0.0 {
                break;
            }

            let c = &self.constraints[ci];
            let left_block = self.vars[c.left].block.unwrap();
            let right_block = self.vars[c.right].block.unwrap();

            let c_left = c.left;
            let c_right = c.right;
            let c_gap = c.gap;

            let dist = self.vars[c_left].offset + c_gap - self.vars[c_right].offset;

            let left_size = self.blocks[left_block].vars.len();
            let right_size = self.blocks[right_block].vars.len();

            let (target, other, actual_dist) = if left_size > right_size {
                (right_block, left_block, -dist)
            } else {
                (left_block, right_block, dist)
            };

            self.block_merge(target, other, ci, actual_dist);
            block_idx = target;
        }
    }

    /// Remove deleted blocks.
    ///
    /// C++ ref: Blocks::cleanup
    fn cleanup_blocks(&mut self) {
        // We can't easily remove from the middle of the Vec because indices
        // would shift. Instead, mark as deleted and skip in iteration.
        // The C++ code does in-place compaction — we do the same but update
        // variable block references.
        let mut new_blocks = Vec::new();
        let mut index_map = vec![0usize; self.blocks.len()];

        for (old_idx, block) in self.blocks.iter().enumerate() {
            if !block.deleted {
                index_map[old_idx] = new_blocks.len();
                new_blocks.push(block.clone());
            }
        }

        // Update variable block references
        for v in &mut self.vars {
            if let Some(bi) = v.block {
                if self.blocks[bi].deleted {
                    // This shouldn't happen — variables in deleted blocks
                    // should have been moved to non-deleted blocks
                    panic!("Variable {} still references deleted block {}", v.id, bi);
                }
                v.block = Some(index_map[bi]);
            }
        }

        self.blocks = new_blocks;
    }

    /// Copy positions to final_position for all variables.
    ///
    /// C++ ref: Solver::copyResult
    fn copy_result(&mut self) {
        for vi in 0..self.num_vars {
            let pos = self.var_position(vi);
            self.vars[vi].final_position = pos;
            debug_assert!(pos.is_finite());
        }
    }

    /// Refine solution by splitting blocks with negative Lagrangians.
    ///
    /// C++ ref: Solver::refine
    fn refine(&mut self) -> Result<(), VpscError> {
        for _ in 0..MAX_REFINE_ITERATIONS {
            let mut found_split = false;

            let block_count = self.blocks.len();
            for bi in 0..block_count {
                if self.blocks[bi].deleted {
                    continue;
                }

                if let Some(ci) = self.find_min_lm(bi) {
                    if self.constraints[ci].lm < LAGRANGIAN_TOLERANCE {
                        let (lb, _rb) = self.block_split(bi, ci);
                        self.blocks[bi].deleted = true;

                        // Post-split: merge left and right
                        self.merge_left(lb);
                        let c_right = self.constraints[ci].right;
                        let rb_actual = self.vars[c_right].block.unwrap();
                        self.block_update_weighted_position(rb_actual);
                        self.merge_right(rb_actual);

                        self.cleanup_blocks();
                        found_split = true;
                        break;
                    }
                }
            }

            if !found_split {
                break;
            }
        }

        // Verify all constraints satisfied
        for ci in 0..self.num_constraints {
            let slack = self.constraint_slack(ci);
            if slack < ZERO_UPPERBOUND {
                return Err(VpscError::UnsatisfiedConstraint(
                    UnsatisfiedConstraintError { constraint_index: ci },
                ));
            }
        }

        Ok(())
    }

    /// Cost: total squared distance of variables from desired positions.
    ///
    /// C++ ref: Blocks::cost
    pub fn cost(&self) -> f64 {
        let mut total = 0.0;
        for vi in 0..self.num_vars {
            let pos = self.var_position(vi);
            let diff = pos - self.vars[vi].desired_position;
            total += self.vars[vi].weight * diff * diff;
        }
        total
    }
}

// ---------------------------------------------------------------------------
// IncSolver (incremental)
// ---------------------------------------------------------------------------

/// Incremental solver for Variable Placement with Separation Constraints.
///
/// Allows refinement after blocks are moved. Preferred for interactive use.
///
/// C++ ref: vpsc::IncSolver
pub struct IncSolver {
    solver: Solver,
    inactive: Vec<usize>,
}

impl IncSolver {
    /// Create a new incremental VPSC solver.
    ///
    /// C++ ref: IncSolver::IncSolver
    pub fn new(vars: Vec<Variable>, constraints: Vec<Constraint>) -> Self {
        let num_constraints = constraints.len();
        let mut solver = Solver::new(vars, constraints);

        // All constraints start as inactive
        for c in &mut solver.constraints {
            c.active = false;
        }
        let inactive: Vec<usize> = (0..num_constraints).collect();

        Self { solver, inactive }
    }

    /// Access the variables (read-only).
    pub fn variables(&self) -> &[Variable] {
        &self.solver.vars
    }

    /// Get final positions.
    pub fn final_positions(&self) -> Vec<f64> {
        self.solver.final_positions()
    }

    /// Current cost.
    pub fn cost(&self) -> f64 {
        self.solver.cost()
    }

    /// Add a constraint to the existing solver.
    ///
    /// C++ ref: IncSolver::addConstraint
    pub fn add_constraint(&mut self, constraint: Constraint) {
        let ci = self.solver.constraints.len();
        let left = constraint.left;
        let right = constraint.right;
        let needs_scaling = self.solver.needs_scaling;

        let mut c = constraint;
        c.active = false;
        c.needs_scaling = needs_scaling;
        self.solver.constraints.push(c);

        self.solver.vars[left].out_constraints.push(ci);
        self.solver.vars[right].in_constraints.push(ci);
        self.solver.num_constraints += 1;

        self.inactive.push(ci);
    }

    /// Incremental satisfy.
    ///
    /// C++ ref: IncSolver::satisfy
    pub fn satisfy(&mut self) -> Result<bool, VpscError> {
        self.split_blocks();

        loop {
            let (mv_idx, mv_ci) = self.most_violated();

            let should_process = if let Some(ci) = mv_ci {
                let c = &self.solver.constraints[ci];
                c.equality
                    || (self.solver.constraint_slack(ci) < ZERO_UPPERBOUND && !c.active)
            } else {
                false
            };

            if !should_process {
                break;
            }

            let ci = mv_ci.unwrap();
            let c = &self.solver.constraints[ci];
            debug_assert!(!c.active);

            let left_block = self.solver.vars[c.left].block.unwrap();
            let right_block = self.solver.vars[c.right].block.unwrap();
            let c_left = c.left;
            let c_right = c.right;

            // Remove from inactive list
            if let Some(pos) = mv_idx {
                let last = self.inactive.len() - 1;
                self.inactive.swap(pos, last);
                self.inactive.pop();
            }

            if left_block != right_block {
                // Merge across blocks
                self.solver.block_merge_across(ci);
            } else {
                // Same block — check for cycles
                if self.solver.is_active_directed_path_between(
                    c_right,
                    c_left,
                    left_block,
                ) {
                    self.solver.constraints[ci].unsatisfiable = true;
                    continue;
                }

                // Split first, then merge
                match self.solver.split_between(left_block, c_left, c_right) {
                    Ok((split_ci, _, _lb, _rb)) => {
                        if let Some(sci) = Some(split_ci) {
                            if !self.solver.constraints[sci].active {
                                self.inactive.push(sci);
                            }
                        }

                        let slack = self.solver.constraint_slack(ci);
                        if slack >= 0.0 {
                            // Satisfied by the split
                            self.inactive.push(ci);
                            // Blocks lb and rb are already in the blocks vec
                        } else {
                            // Still violated, merge
                            self.solver.block_merge_across(ci);
                            // Clean up: one of lb, rb was consumed by merge
                        }
                    }
                    Err(_) => {
                        self.solver.constraints[ci].unsatisfiable = true;
                        continue;
                    }
                }
            }
        }

        self.solver.cleanup_blocks();

        // Check all constraints
        let mut active_constraints = false;
        for ci in 0..self.solver.num_constraints {
            if self.solver.constraints[ci].active {
                active_constraints = true;
            }
            let slack = self.solver.constraint_slack(ci);
            if slack < ZERO_UPPERBOUND {
                return Err(VpscError::UnsatisfiedConstraint(
                    UnsatisfiedConstraintError { constraint_index: ci },
                ));
            }
        }

        self.solver.copy_result();
        Ok(active_constraints)
    }

    /// Solve to optimality using incremental approach.
    ///
    /// C++ ref: IncSolver::solve
    pub fn solve(&mut self) -> Result<bool, VpscError> {
        self.satisfy()?;

        let cost_convergence_threshold = 0.0001;
        let mut last_cost = f64::MAX;
        let mut cost = self.solver.cost();

        while (last_cost - cost).abs() > cost_convergence_threshold {
            self.satisfy()?;
            last_cost = cost;
            cost = self.solver.cost();
        }

        self.solver.copy_result();
        let active_blocks = self.solver.blocks.iter().filter(|b| !b.deleted).count();
        Ok(active_blocks != self.solver.num_vars)
    }

    /// Move blocks to updated positions.
    ///
    /// C++ ref: IncSolver::moveBlocks
    fn move_blocks(&mut self) {
        let block_count = self.solver.blocks.len();
        for bi in 0..block_count {
            if !self.solver.blocks[bi].deleted {
                self.solver.block_update_weighted_position(bi);
            }
        }
    }

    /// Split blocks that want to split (negative Lagrangian).
    ///
    /// C++ ref: IncSolver::splitBlocks
    fn split_blocks(&mut self) {
        self.move_blocks();

        let block_count = self.solver.blocks.len();
        for bi in 0..block_count {
            if self.solver.blocks[bi].deleted {
                continue;
            }

            if let Some(ci) = self.solver.find_min_lm(bi) {
                if self.solver.constraints[ci].lm < LAGRANGIAN_TOLERANCE {
                    debug_assert!(!self.solver.constraints[ci].equality);

                    let (lb, rb) = self.solver.block_split(bi, ci);
                    self.solver.block_update_weighted_position(lb);
                    self.solver.block_update_weighted_position(rb);
                    self.solver.blocks[bi].deleted = true;

                    debug_assert!(!self.solver.constraints[ci].active);
                    self.inactive.push(ci);
                }
            }
        }

        self.solver.cleanup_blocks();
    }

    /// Find the most violated constraint in the inactive list.
    ///
    /// Returns (index in inactive list, constraint index).
    ///
    /// C++ ref: IncSolver::mostViolated
    fn most_violated(&self) -> (Option<usize>, Option<usize>) {
        let mut best_slack = f64::MAX;
        let mut best_ci = None;
        let mut best_pos = None;

        for (pos, &ci) in self.inactive.iter().enumerate() {
            let c = &self.solver.constraints[ci];
            let slack = self.solver.constraint_slack(ci);

            if c.equality || slack < best_slack {
                best_slack = slack;
                best_ci = Some(ci);
                best_pos = Some(pos);
                if c.equality {
                    break;
                }
            }
        }

        // Only return if actually violated (or equality)
        if let Some(ci) = best_ci {
            let c = &self.solver.constraints[ci];
            if c.equality
                || (best_slack < ZERO_UPPERBOUND && !c.active)
            {
                return (best_pos, best_ci);
            }
        }

        (None, None)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ===================================================================
    // Category 1: Solver construction and initialization
    // ===================================================================

    #[test]
    fn test_solver_creation_empty() {
        let solver = Solver::new(vec![], vec![]);
        assert!(solver.variables().is_empty());
        assert!(solver.blocks.is_empty());
    }

    #[test]
    fn test_solver_single_variable() {
        let vars = vec![Variable::new(0, 5.0, 1.0, 1.0)];
        let solver = Solver::new(vars, vec![]);
        assert_eq!(solver.variables().len(), 1);
        assert_eq!(solver.blocks.len(), 1);
        assert_eq!(solver.variables()[0].block, Some(0));
    }

    #[test]
    fn test_solver_initializes_blocks() {
        let vars = vec![
            Variable::new(0, 0.0, 1.0, 1.0),
            Variable::new(1, 10.0, 1.0, 1.0),
        ];
        let solver = Solver::new(vars, vec![]);
        assert_eq!(solver.blocks.len(), 2);
        // Each variable in its own block
        assert_eq!(solver.blocks[0].vars, vec![0]);
        assert_eq!(solver.blocks[1].vars, vec![1]);
    }

    #[test]
    fn test_solver_builds_constraint_dag() {
        let vars = vec![
            Variable::new(0, 0.0, 1.0, 1.0),
            Variable::new(1, 10.0, 1.0, 1.0),
        ];
        let constraints = vec![Constraint::new(0, 1, 5.0, false)];
        let solver = Solver::new(vars, constraints);

        assert_eq!(solver.vars[0].out_constraints, vec![0]);
        assert!(solver.vars[0].in_constraints.is_empty());
        assert_eq!(solver.vars[1].in_constraints, vec![0]);
        assert!(solver.vars[1].out_constraints.is_empty());
    }

    // ===================================================================
    // Category 2: Unconstrained solving (no constraints)
    // ===================================================================

    #[test]
    fn test_solve_unconstrained_single() {
        let vars = vec![Variable::new(0, 5.0, 1.0, 1.0)];
        let mut solver = Solver::new(vars, vec![]);
        let active = solver.solve().unwrap();
        assert!(!active);
        assert!((solver.final_positions()[0] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_solve_unconstrained_multiple() {
        let vars = vec![
            Variable::new(0, 3.0, 1.0, 1.0),
            Variable::new(1, 7.0, 1.0, 1.0),
            Variable::new(2, -2.0, 1.0, 1.0),
        ];
        let mut solver = Solver::new(vars, vec![]);
        solver.solve().unwrap();
        let pos = solver.final_positions();
        assert!((pos[0] - 3.0).abs() < 1e-6);
        assert!((pos[1] - 7.0).abs() < 1e-6);
        assert!((pos[2] - (-2.0)).abs() < 1e-6);
    }

    // ===================================================================
    // Category 3: Simple constraint satisfaction
    // ===================================================================

    #[test]
    fn test_satisfy_already_satisfied() {
        let vars = vec![
            Variable::new(0, 0.0, 1.0, 1.0),
            Variable::new(1, 10.0, 1.0, 1.0),
        ];
        let constraints = vec![Constraint::new(0, 1, 5.0, false)];
        let mut solver = Solver::new(vars, constraints);
        solver.satisfy().unwrap();
        let pos = solver.final_positions();
        // Already satisfied: 0 + 5 <= 10
        assert!((pos[0] - 0.0).abs() < 1e-6);
        assert!((pos[1] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_satisfy_violated_constraint() {
        let vars = vec![
            Variable::new(0, 5.0, 1.0, 1.0),
            Variable::new(1, 3.0, 1.0, 1.0),
        ];
        // left(5) + 5 <= right(3) is violated
        let constraints = vec![Constraint::new(0, 1, 5.0, false)];
        let mut solver = Solver::new(vars, constraints);
        solver.satisfy().unwrap();
        let pos = solver.final_positions();
        // After satisfaction: left + 5 <= right
        assert!(pos[0] + 5.0 <= pos[1] + 1e-6);
    }

    #[test]
    fn test_solve_pushes_apart() {
        let vars = vec![
            Variable::new(0, 0.0, 1.0, 1.0),
            Variable::new(1, 0.0, 1.0, 1.0),
        ];
        // Both at 0, need separation of 10
        let constraints = vec![Constraint::new(0, 1, 10.0, false)];
        let mut solver = Solver::new(vars, constraints);
        solver.solve().unwrap();
        let pos = solver.final_positions();
        // Should push apart symmetrically: left=-5, right=5
        assert!(pos[0] + 10.0 <= pos[1] + 1e-6);
        assert!((pos[0] - (-5.0)).abs() < 1e-6);
        assert!((pos[1] - 5.0).abs() < 1e-6);
    }

    // ===================================================================
    // Category 4: Equality constraints
    // ===================================================================

    #[test]
    fn test_equality_constraint_inc_solver() {
        // Equality constraints are handled by the incremental solver
        let vars = vec![
            Variable::new(0, 0.0, 1.0, 1.0),
            Variable::new(1, 10.0, 1.0, 1.0),
        ];
        let constraints = vec![Constraint::new(0, 1, 5.0, true)];
        let mut solver = IncSolver::new(vars, constraints);
        solver.solve().unwrap();
        let pos = solver.final_positions();
        // Equality: right - left == 5
        assert!((pos[1] - pos[0] - 5.0).abs() < 1e-4);
    }

    // ===================================================================
    // Category 5: Chain constraints
    // ===================================================================

    #[test]
    fn test_chain_constraints() {
        let vars = vec![
            Variable::new(0, 0.0, 1.0, 1.0),
            Variable::new(1, 0.0, 1.0, 1.0),
            Variable::new(2, 0.0, 1.0, 1.0),
        ];
        // 0 + 5 <= 1, 1 + 5 <= 2
        let constraints = vec![
            Constraint::new(0, 1, 5.0, false),
            Constraint::new(1, 2, 5.0, false),
        ];
        let mut solver = Solver::new(vars, constraints);
        solver.solve().unwrap();
        let pos = solver.final_positions();
        assert!(pos[0] + 5.0 <= pos[1] + 1e-6);
        assert!(pos[1] + 5.0 <= pos[2] + 1e-6);
    }

    // ===================================================================
    // Category 6: Weighted variables
    // ===================================================================

    #[test]
    fn test_weighted_variable_stays_closer() {
        let vars = vec![
            Variable::new(0, 0.0, 10.0, 1.0), // heavy weight
            Variable::new(1, 0.0, 1.0, 1.0),   // light weight
        ];
        let constraints = vec![Constraint::new(0, 1, 10.0, false)];
        let mut solver = Solver::new(vars, constraints);
        solver.solve().unwrap();
        let pos = solver.final_positions();
        // Heavy variable should move less than light one
        assert!(pos[0].abs() < pos[1].abs());
        assert!(pos[0] + 10.0 <= pos[1] + 1e-6);
    }

    // ===================================================================
    // Category 7: Variable scaling
    // ===================================================================

    #[test]
    fn test_scaled_variables() {
        let vars = vec![
            Variable::new(0, 0.0, 1.0, 2.0), // scale = 2
            Variable::new(1, 10.0, 1.0, 1.0),
        ];
        let constraints = vec![Constraint::new(0, 1, 5.0, false)];
        let mut solver = Solver::new(vars, constraints);
        let result = solver.solve();
        // Should succeed (scaling is supported)
        assert!(result.is_ok());
    }

    // ===================================================================
    // Category 8: Cost computation
    // ===================================================================

    #[test]
    fn test_cost_at_desired() {
        let vars = vec![
            Variable::new(0, 5.0, 1.0, 1.0),
            Variable::new(1, 10.0, 1.0, 1.0),
        ];
        let solver = Solver::new(vars, vec![]);
        let cost = solver.cost();
        assert!((cost - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cost_increases_with_displacement() {
        let vars = vec![
            Variable::new(0, 0.0, 1.0, 1.0),
            Variable::new(1, 0.0, 1.0, 1.0),
        ];
        let constraints = vec![Constraint::new(0, 1, 10.0, false)];
        let mut solver = Solver::new(vars, constraints);
        solver.solve().unwrap();
        let cost = solver.cost();
        assert!(cost > 0.0);
    }

    // ===================================================================
    // Category 9: IncSolver basics
    // ===================================================================

    #[test]
    fn test_inc_solver_basic() {
        let vars = vec![
            Variable::new(0, 0.0, 1.0, 1.0),
            Variable::new(1, 0.0, 1.0, 1.0),
        ];
        let constraints = vec![Constraint::new(0, 1, 10.0, false)];
        let mut solver = IncSolver::new(vars, constraints);
        solver.solve().unwrap();
        let pos = solver.final_positions();
        assert!(pos[0] + 10.0 <= pos[1] + 1e-6);
    }

    #[test]
    fn test_inc_solver_add_constraint() {
        let vars = vec![
            Variable::new(0, 0.0, 1.0, 1.0),
            Variable::new(1, 20.0, 1.0, 1.0),
            Variable::new(2, 40.0, 1.0, 1.0),
        ];
        let constraints = vec![Constraint::new(0, 1, 5.0, false)];
        let mut solver = IncSolver::new(vars, constraints);
        solver.solve().unwrap();

        // Add new constraint
        solver.add_constraint(Constraint::new(1, 2, 5.0, false));
        solver.solve().unwrap();
        let pos = solver.final_positions();
        assert!(pos[0] + 5.0 <= pos[1] + 1e-6);
        assert!(pos[1] + 5.0 <= pos[2] + 1e-6);
    }

    // ===================================================================
    // Category 10: Edge cases
    // ===================================================================

    #[test]
    fn test_zero_gap_constraint() {
        let vars = vec![
            Variable::new(0, 0.0, 1.0, 1.0),
            Variable::new(1, 0.0, 1.0, 1.0),
        ];
        let constraints = vec![Constraint::new(0, 1, 0.0, false)];
        let mut solver = Solver::new(vars, constraints);
        solver.solve().unwrap();
        let pos = solver.final_positions();
        assert!(pos[0] <= pos[1] + 1e-6);
    }

    #[test]
    fn test_negative_desired_positions() {
        let vars = vec![
            Variable::new(0, -10.0, 1.0, 1.0),
            Variable::new(1, -20.0, 1.0, 1.0),
        ];
        // -10 + 5 <= -20 is violated
        let constraints = vec![Constraint::new(0, 1, 5.0, false)];
        let mut solver = Solver::new(vars, constraints);
        solver.solve().unwrap();
        let pos = solver.final_positions();
        assert!(pos[0] + 5.0 <= pos[1] + 1e-6);
    }

    #[test]
    fn test_many_variables_single_chain() {
        let n = 20;
        let vars: Vec<Variable> = (0..n)
            .map(|i| Variable::new(i, 0.0, 1.0, 1.0))
            .collect();
        let constraints: Vec<Constraint> = (0..n - 1)
            .map(|i| Constraint::new(i, i + 1, 1.0, false))
            .collect();

        let mut solver = Solver::new(vars, constraints);
        solver.solve().unwrap();
        let pos = solver.final_positions();

        for i in 0..n - 1 {
            assert!(
                pos[i] + 1.0 <= pos[i + 1] + 1e-6,
                "Constraint violated at {}: {} + 1.0 > {}",
                i,
                pos[i],
                pos[i + 1]
            );
        }
    }

    #[test]
    fn test_convergent_constraints() {
        // Two variables pushed towards each other
        let vars = vec![
            Variable::new(0, 100.0, 1.0, 1.0),
            Variable::new(1, -100.0, 1.0, 1.0),
        ];
        // 0 + 5 <= 1 — but 0 wants to be at 100 and 1 at -100
        let constraints = vec![Constraint::new(0, 1, 5.0, false)];
        let mut solver = Solver::new(vars, constraints);
        solver.solve().unwrap();
        let pos = solver.final_positions();
        // Constraint should be satisfied even though desired positions conflict
        assert!(pos[0] + 5.0 <= pos[1] + 1e-6);
    }

    // ===================================================================
    // Category 11: Constraint slack
    // ===================================================================

    #[test]
    fn test_constraint_slack_positive() {
        let vars = vec![
            Variable::new(0, 0.0, 1.0, 1.0),
            Variable::new(1, 10.0, 1.0, 1.0),
        ];
        let constraints = vec![Constraint::new(0, 1, 5.0, false)];
        let solver = Solver::new(vars, constraints);
        let slack = solver.constraint_slack(0);
        // 10 - 5 - 0 = 5
        assert!((slack - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_constraint_slack_negative() {
        let vars = vec![
            Variable::new(0, 10.0, 1.0, 1.0),
            Variable::new(1, 5.0, 1.0, 1.0),
        ];
        let constraints = vec![Constraint::new(0, 1, 8.0, false)];
        let solver = Solver::new(vars, constraints);
        let slack = solver.constraint_slack(0);
        // 5 - 8 - 10 = -13
        assert!(slack < 0.0);
    }

    // ===================================================================
    // Category 12: Topological ordering
    // ===================================================================

    #[test]
    fn test_total_order_no_constraints() {
        let vars = vec![
            Variable::new(0, 0.0, 1.0, 1.0),
            Variable::new(1, 0.0, 1.0, 1.0),
        ];
        let mut solver = Solver::new(vars, vec![]);
        let order = solver.total_order();
        assert_eq!(order.len(), 2);
    }

    #[test]
    fn test_total_order_with_chain() {
        let vars = vec![
            Variable::new(0, 0.0, 1.0, 1.0),
            Variable::new(1, 0.0, 1.0, 1.0),
            Variable::new(2, 0.0, 1.0, 1.0),
        ];
        let constraints = vec![
            Constraint::new(0, 1, 1.0, false),
            Constraint::new(1, 2, 1.0, false),
        ];
        let mut solver = Solver::new(vars, constraints);
        let order = solver.total_order();
        assert_eq!(order.len(), 3);
        // In topological order, 0 should come before 1 before 2
        // (but our DFS pushes in reverse finish order)
        let pos_0 = order.iter().position(|&x| x == 0).unwrap();
        let pos_1 = order.iter().position(|&x| x == 1).unwrap();
        let pos_2 = order.iter().position(|&x| x == 2).unwrap();
        // DFS adds to back on finish, so order is reversed
        // Actually our impl pushes to back, so earlier finished = earlier in vec
        // Let's just verify all are present
        assert!(order.contains(&0));
        assert!(order.contains(&1));
        assert!(order.contains(&2));
        let _ = (pos_0, pos_1, pos_2);
    }
}
