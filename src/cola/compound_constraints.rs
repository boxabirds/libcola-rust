//! High-level compound constraints for graph layout.
//!
//! Translates diagramming-level constraints (alignment, distribution, separation,
//! page boundaries, etc.) into low-level VPSC separation constraints.
//!
//! C++ ref: libcola/compound_constraints.h, libcola/compound_constraints.cpp

use crate::vpsc::{Dim, Rectangle};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Weight for freely-floating dummy variables (guide lines, etc.).
const FREE_WEIGHT: f64 = 0.0001;

/// Default compound constraint priority.
const DEFAULT_CONSTRAINT_PRIORITY: u32 = 30000;

/// Priority for non-overlap constraints.
const PRIORITY_NONOVERLAP: u32 = DEFAULT_CONSTRAINT_PRIORITY - 2000;

/// Weight assigned to variables with fixed positions.
const FIXED_POSITION_WEIGHT: f64 = 100_000.0;

// ---------------------------------------------------------------------------
// Generated specifications
// ---------------------------------------------------------------------------

/// A generated VPSC variable specification.
#[derive(Debug, Clone)]
pub struct GeneratedVar {
    pub desired_position: f64,
    pub weight: f64,
    pub fixed: bool,
}

/// A generated VPSC separation constraint specification.
/// References variables by index into the combined variable list.
#[derive(Debug, Clone)]
pub struct GeneratedConstraint {
    /// Left variable index.
    pub left: usize,
    /// Right variable index.
    pub right: usize,
    /// Minimum (or exact) gap between variables.
    pub gap: f64,
    /// Whether this is an equality constraint.
    pub equality: bool,
}

/// An instruction to modify an existing variable (e.g., fixing its position).
#[derive(Debug, Clone)]
pub struct VarModification {
    pub var_index: usize,
    pub weight: f64,
    pub fixed: bool,
}

// ---------------------------------------------------------------------------
// VariableIdMap
// ---------------------------------------------------------------------------

/// A bidirectional variable ID mapping for topology addons.
#[derive(Debug, Clone, Default)]
pub struct VariableIdMap {
    mappings: Vec<(usize, usize)>,
}

impl VariableIdMap {
    pub fn new() -> Self {
        Self {
            mappings: Vec::new(),
        }
    }

    /// Adds a mapping from `from` to `to`. Returns `false` if `from` is already
    /// mapped (the mapping is not added in that case).
    pub fn add_mapping(&mut self, from: usize, to: usize) -> bool {
        if self.mappings.iter().any(|&(f, _)| f == from) {
            return false;
        }
        self.mappings.push((from, to));
        true
    }

    /// Look up the mapping for `var`. If `forward` is true, looks up `from -> to`;
    /// if false, looks up `to -> from`. Returns `var` itself if no mapping exists.
    pub fn mapping_for(&self, var: usize, forward: bool) -> usize {
        for &(from, to) in &self.mappings {
            if forward && from == var {
                return to;
            }
            if !forward && to == var {
                return from;
            }
        }
        var
    }

    pub fn clear(&mut self) {
        self.mappings.clear();
    }
}

// ---------------------------------------------------------------------------
// UnsatisfiableConstraintInfo
// ---------------------------------------------------------------------------

/// Info about an unsatisfiable constraint detected during solving.
#[derive(Debug, Clone)]
pub struct UnsatisfiableConstraintInfo {
    pub left_var_index: usize,
    pub right_var_index: usize,
    pub separation: f64,
    pub equality: bool,
}

// ---------------------------------------------------------------------------
// Concrete constraint types
// ---------------------------------------------------------------------------

/// A boundary line that nodes must be to the left/above or right/below of.
///
/// C++ ref: cola::BoundaryConstraint
#[derive(Debug, Clone)]
pub struct BoundaryConstraint {
    pub primary_dim: Dim,
    pub position: f64,
    /// Index of the generated boundary variable (set during `generate_variables`).
    pub variable_index: Option<usize>,
    /// (node_index, offset). Negative offset = node to the left of boundary,
    /// positive offset = node to the right of boundary.
    shapes: Vec<(usize, f64)>,
}

impl BoundaryConstraint {
    pub fn new(dim: Dim) -> Self {
        Self {
            primary_dim: dim,
            position: 0.0,
            variable_index: None,
            shapes: Vec::new(),
        }
    }

    pub fn add_shape(&mut self, index: usize, offset: f64) {
        self.shapes.push((index, offset));
    }
}

/// An alignment line that nodes are constrained to with exact offsets.
///
/// C++ ref: cola::AlignmentConstraint
#[derive(Debug, Clone)]
pub struct AlignmentConstraint {
    pub primary_dim: Dim,
    pub position: f64,
    pub is_fixed: bool,
    /// Index of the generated alignment variable (set during `generate_variables`).
    pub variable_index: Option<usize>,
    /// (node_index, offset from alignment line).
    shapes: Vec<(usize, f64)>,
}

impl AlignmentConstraint {
    pub fn new(dim: Dim, position: f64) -> Self {
        Self {
            primary_dim: dim,
            position,
            is_fixed: false,
            variable_index: None,
            shapes: Vec::new(),
        }
    }

    pub fn add_shape(&mut self, index: usize, offset: f64) {
        self.shapes.push((index, offset));
    }

    /// Fix the alignment line at the given position.
    pub fn fix_pos(&mut self, pos: f64) {
        self.position = pos;
        self.is_fixed = true;
    }

    /// Unfix the alignment line so the solver can move it freely.
    pub fn unfix_pos(&mut self) {
        self.is_fixed = false;
    }
}

/// A separation constraint between two variable indices.
///
/// The caller is responsible for resolving alignment variable indices before
/// constructing this.
///
/// C++ ref: cola::SeparationConstraint
#[derive(Debug, Clone)]
pub struct SeparationConstraint {
    pub primary_dim: Dim,
    /// Left variable index.
    pub left: usize,
    /// Right variable index.
    pub right: usize,
    pub gap: f64,
    pub equality: bool,
}

impl SeparationConstraint {
    pub fn new(dim: Dim, left: usize, right: usize, gap: f64, equality: bool) -> Self {
        Self {
            primary_dim: dim,
            left,
            right,
            gap,
            equality,
        }
    }

    pub fn set_separation(&mut self, gap: f64) {
        self.gap = gap;
    }
}

/// Forces an edge to be orthogonal (endpoints aligned in the perpendicular dimension).
///
/// C++ ref: cola::OrthogonalEdgeConstraint
#[derive(Debug, Clone)]
pub struct OrthogonalEdgeConstraint {
    pub primary_dim: Dim,
    /// Variable index of one endpoint.
    pub left: usize,
    /// Variable index of the other endpoint.
    pub right: usize,
}

/// Equal spacing between pairs of alignment lines.
///
/// C++ ref: cola::MultiSeparationConstraint
#[derive(Debug, Clone)]
pub struct MultiSeparationConstraint {
    pub primary_dim: Dim,
    pub sep: f64,
    pub equality: bool,
    /// Pairs of alignment variable indices.
    pairs: Vec<(usize, usize)>,
}

impl MultiSeparationConstraint {
    pub fn new(dim: Dim, min_sep: f64, equality: bool) -> Self {
        Self {
            primary_dim: dim,
            sep: min_sep,
            equality,
            pairs: Vec::new(),
        }
    }

    pub fn add_alignment_pair(&mut self, left_var: usize, right_var: usize) {
        self.pairs.push((left_var, right_var));
    }

    pub fn set_separation(&mut self, sep: f64) {
        self.sep = sep;
    }
}

/// Fixed equal distribution between alignment lines (always equality constraints).
///
/// C++ ref: cola::DistributionConstraint
#[derive(Debug, Clone)]
pub struct DistributionConstraint {
    pub primary_dim: Dim,
    pub sep: f64,
    /// Pairs of alignment variable indices.
    pairs: Vec<(usize, usize)>,
}

impl DistributionConstraint {
    pub fn new(dim: Dim) -> Self {
        Self {
            primary_dim: dim,
            sep: 0.0,
            pairs: Vec::new(),
        }
    }

    pub fn add_alignment_pair(&mut self, left_var: usize, right_var: usize) {
        self.pairs.push((left_var, right_var));
    }

    pub fn set_separation(&mut self, sep: f64) {
        self.sep = sep;
    }
}

/// Constrains a set of nodes to be fixed relative to each other.
///
/// C++ ref: cola::FixedRelativeConstraint
#[derive(Debug, Clone)]
pub struct FixedRelativeConstraint {
    pub fixed_position: bool,
    /// Sorted, unique node variable indices.
    shape_vars: Vec<usize>,
    /// Pre-computed offsets: (left_var, right_var, dim, offset).
    offsets: Vec<(usize, usize, Dim, f64)>,
}

impl FixedRelativeConstraint {
    /// Create a fixed-relative constraint from current rectangle positions.
    ///
    /// For each consecutive pair of shapes (sorted by index), equality offsets
    /// are computed in both dimensions.
    pub fn new(rects: &[Rectangle], shape_ids: Vec<usize>, fixed_position: bool) -> Self {
        let mut sorted = shape_ids;
        sorted.sort_unstable();
        sorted.dedup();

        let mut offsets = Vec::new();
        for window in sorted.windows(2) {
            let left = window[0];
            let right = window[1];
            for &dim in &[Dim::Horizontal, Dim::Vertical] {
                let offset = rects[right].centre_d(dim) - rects[left].centre_d(dim);
                offsets.push((left, right, dim, offset));
            }
        }

        Self {
            fixed_position,
            shape_vars: sorted,
            offsets,
        }
    }
}

/// Constrains nodes to be within page boundaries.
///
/// C++ ref: cola::PageBoundaryConstraints
#[derive(Debug, Clone)]
pub struct PageBoundaryConstraint {
    /// Left/top margin per dimension `[x, y]`.
    pub left_margin: [f64; 2],
    /// Right/bottom margin per dimension `[x, y]`.
    pub right_margin: [f64; 2],
    /// Weight for the left/top boundary variable per dimension.
    pub left_weight: [f64; 2],
    /// Weight for the right/bottom boundary variable per dimension.
    pub right_weight: [f64; 2],
    /// (node_index, [half_width, half_height]).
    shapes: Vec<(usize, [f64; 2])>,
    /// Variable indices for left/top boundary vars per dimension, set during
    /// `generate_variables`.
    pub left_var: [Option<usize>; 2],
    /// Variable indices for right/bottom boundary vars per dimension.
    pub right_var: [Option<usize>; 2],
}

impl PageBoundaryConstraint {
    pub fn new(
        left_margin: [f64; 2],
        right_margin: [f64; 2],
        left_weight: [f64; 2],
        right_weight: [f64; 2],
    ) -> Self {
        Self {
            left_margin,
            right_margin,
            left_weight,
            right_weight,
            shapes: Vec::new(),
            left_var: [None; 2],
            right_var: [None; 2],
        }
    }

    pub fn add_shape(&mut self, index: usize, half_width: f64, half_height: f64) {
        self.shapes.push((index, [half_width, half_height]));
    }
}

// ---------------------------------------------------------------------------
// CompoundConstraint enum
// ---------------------------------------------------------------------------

/// A high-level compound constraint that generates VPSC variables and constraints.
#[derive(Debug, Clone)]
pub enum CompoundConstraint {
    /// A boundary line that nodes must be to the left or right of.
    Boundary(BoundaryConstraint),
    /// An alignment line that nodes must be aligned to with exact offsets.
    Alignment(AlignmentConstraint),
    /// A separation between two variable indices.
    Separation(SeparationConstraint),
    /// Forces an edge to be orthogonal (endpoints aligned in perpendicular dim).
    OrthogonalEdge(OrthogonalEdgeConstraint),
    /// Equal spacing between pairs of alignment lines.
    MultiSeparation(MultiSeparationConstraint),
    /// Fixed equal distribution between alignment lines.
    Distribution(DistributionConstraint),
    /// Nodes fixed relative to each other.
    FixedRelative(FixedRelativeConstraint),
    /// Nodes contained within page boundaries.
    PageBoundary(PageBoundaryConstraint),
}

impl CompoundConstraint {
    /// The primary dimension this constraint operates on.
    pub fn dimension(&self) -> Dim {
        match self {
            Self::Boundary(c) => c.primary_dim,
            Self::Alignment(c) => c.primary_dim,
            Self::Separation(c) => c.primary_dim,
            Self::OrthogonalEdge(c) => c.primary_dim,
            Self::MultiSeparation(c) => c.primary_dim,
            Self::Distribution(c) => c.primary_dim,
            // FixedRelative applies to both dimensions; default to Horizontal.
            Self::FixedRelative(_) => Dim::Horizontal,
            // PageBoundary applies to both dimensions; default to Horizontal.
            Self::PageBoundary(_) => Dim::Horizontal,
        }
    }

    /// The constraint priority (higher = processed later).
    pub fn priority(&self) -> u32 {
        match self {
            Self::Boundary(_)
            | Self::Alignment(_)
            | Self::Separation(_)
            | Self::OrthogonalEdge(_)
            | Self::MultiSeparation(_)
            | Self::Distribution(_)
            | Self::FixedRelative(_)
            | Self::PageBoundary(_) => DEFAULT_CONSTRAINT_PRIORITY,
        }
    }

    /// Generate any additional variables needed. Returns new variables to
    /// append after the existing node variables. May update internal variable
    /// index bookkeeping.
    pub fn generate_variables(
        &mut self,
        dim: Dim,
        num_existing_vars: usize,
    ) -> Vec<GeneratedVar> {
        match self {
            Self::Boundary(c) => generate_vars_boundary(c, dim, num_existing_vars),
            Self::Alignment(c) => generate_vars_alignment(c, dim, num_existing_vars),
            Self::Separation(_) => Vec::new(),
            Self::OrthogonalEdge(_) => Vec::new(),
            Self::MultiSeparation(_) => Vec::new(),
            Self::Distribution(_) => Vec::new(),
            Self::FixedRelative(_) => Vec::new(),
            Self::PageBoundary(c) => generate_vars_page_boundary(c, dim, num_existing_vars),
        }
    }

    /// Generate separation constraints. `rects` provides current node positions
    /// for computing offsets where needed.
    pub fn generate_separation_constraints(
        &self,
        dim: Dim,
        rects: &[Rectangle],
    ) -> Vec<GeneratedConstraint> {
        match self {
            Self::Boundary(c) => gen_cs_boundary(c, dim),
            Self::Alignment(c) => gen_cs_alignment(c, dim),
            Self::Separation(c) => gen_cs_separation(c, dim),
            Self::OrthogonalEdge(c) => gen_cs_orthogonal_edge(c, dim),
            Self::MultiSeparation(c) => gen_cs_multi_separation(c, dim),
            Self::Distribution(c) => gen_cs_distribution(c, dim),
            Self::FixedRelative(c) => gen_cs_fixed_relative(c, dim, rects),
            Self::PageBoundary(c) => gen_cs_page_boundary(c, dim),
        }
    }

    /// Return modifications that should be applied to existing variables
    /// (e.g., fixing weights for FixedRelativeConstraint).
    pub fn get_var_modifications(&self) -> Vec<VarModification> {
        match self {
            Self::FixedRelative(c) if c.fixed_position => c
                .shape_vars
                .iter()
                .map(|&idx| VarModification {
                    var_index: idx,
                    weight: FIXED_POSITION_WEIGHT,
                    fixed: true,
                })
                .collect(),
            _ => Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Variable generation helpers
// ---------------------------------------------------------------------------

fn generate_vars_boundary(
    c: &mut BoundaryConstraint,
    dim: Dim,
    num_existing_vars: usize,
) -> Vec<GeneratedVar> {
    if dim != c.primary_dim {
        return Vec::new();
    }
    let idx = num_existing_vars;
    c.variable_index = Some(idx);
    vec![GeneratedVar {
        desired_position: c.position,
        weight: FREE_WEIGHT,
        fixed: false,
    }]
}

fn generate_vars_alignment(
    c: &mut AlignmentConstraint,
    dim: Dim,
    num_existing_vars: usize,
) -> Vec<GeneratedVar> {
    if dim != c.primary_dim {
        return Vec::new();
    }
    let idx = num_existing_vars;
    c.variable_index = Some(idx);
    let (weight, fixed) = if c.is_fixed {
        (FIXED_POSITION_WEIGHT, true)
    } else {
        (FREE_WEIGHT, false)
    };
    vec![GeneratedVar {
        desired_position: c.position,
        weight,
        fixed,
    }]
}

fn generate_vars_page_boundary(
    c: &mut PageBoundaryConstraint,
    dim: Dim,
    num_existing_vars: usize,
) -> Vec<GeneratedVar> {
    let d = match dim {
        Dim::Horizontal => 0,
        Dim::Vertical => 1,
        Dim::Depth => 2,
    };

    let left_idx = num_existing_vars;
    let right_idx = num_existing_vars + 1;
    c.left_var[d] = Some(left_idx);
    c.right_var[d] = Some(right_idx);

    vec![
        GeneratedVar {
            desired_position: c.left_margin[d],
            weight: c.left_weight[d],
            fixed: false,
        },
        GeneratedVar {
            desired_position: c.right_margin[d],
            weight: c.right_weight[d],
            fixed: false,
        },
    ]
}

// ---------------------------------------------------------------------------
// Constraint generation helpers
// ---------------------------------------------------------------------------

fn gen_cs_boundary(c: &BoundaryConstraint, dim: Dim) -> Vec<GeneratedConstraint> {
    if dim != c.primary_dim {
        return Vec::new();
    }
    let boundary_var = match c.variable_index {
        Some(v) => v,
        None => return Vec::new(),
    };

    let mut result = Vec::with_capacity(c.shapes.len());
    for &(node_idx, offset) in &c.shapes {
        if offset < 0.0 {
            // Node must be to the left of (or above) the boundary.
            // node + |offset| <= boundary
            result.push(GeneratedConstraint {
                left: node_idx,
                right: boundary_var,
                gap: offset.abs(),
                equality: false,
            });
        } else {
            // Node must be to the right of (or below) the boundary.
            // boundary + offset <= node
            result.push(GeneratedConstraint {
                left: boundary_var,
                right: node_idx,
                gap: offset,
                equality: false,
            });
        }
    }
    result
}

fn gen_cs_alignment(c: &AlignmentConstraint, dim: Dim) -> Vec<GeneratedConstraint> {
    if dim != c.primary_dim {
        return Vec::new();
    }
    let align_var = match c.variable_index {
        Some(v) => v,
        None => return Vec::new(),
    };

    let mut result = Vec::with_capacity(c.shapes.len());
    for &(node_idx, offset) in &c.shapes {
        // alignment_var + offset == node_var
        result.push(GeneratedConstraint {
            left: align_var,
            right: node_idx,
            gap: offset,
            equality: true,
        });
    }
    result
}

fn gen_cs_separation(c: &SeparationConstraint, dim: Dim) -> Vec<GeneratedConstraint> {
    if dim != c.primary_dim {
        return Vec::new();
    }
    vec![GeneratedConstraint {
        left: c.left,
        right: c.right,
        gap: c.gap,
        equality: c.equality,
    }]
}

fn gen_cs_orthogonal_edge(c: &OrthogonalEdgeConstraint, dim: Dim) -> Vec<GeneratedConstraint> {
    if dim != c.primary_dim {
        return Vec::new();
    }
    vec![GeneratedConstraint {
        left: c.left,
        right: c.right,
        gap: 0.0,
        equality: true,
    }]
}

fn gen_cs_multi_separation(
    c: &MultiSeparationConstraint,
    dim: Dim,
) -> Vec<GeneratedConstraint> {
    if dim != c.primary_dim {
        return Vec::new();
    }
    c.pairs
        .iter()
        .map(|&(left, right)| GeneratedConstraint {
            left,
            right,
            gap: c.sep,
            equality: c.equality,
        })
        .collect()
}

fn gen_cs_distribution(c: &DistributionConstraint, dim: Dim) -> Vec<GeneratedConstraint> {
    if dim != c.primary_dim {
        return Vec::new();
    }
    c.pairs
        .iter()
        .map(|&(left, right)| GeneratedConstraint {
            left,
            right,
            gap: c.sep,
            equality: true,
        })
        .collect()
}

fn gen_cs_fixed_relative(
    c: &FixedRelativeConstraint,
    dim: Dim,
    _rects: &[Rectangle],
) -> Vec<GeneratedConstraint> {
    c.offsets
        .iter()
        .filter(|&&(_, _, d, _)| d == dim)
        .map(|&(left, right, _, offset)| GeneratedConstraint {
            left,
            right,
            gap: offset,
            equality: true,
        })
        .collect()
}

fn gen_cs_page_boundary(c: &PageBoundaryConstraint, dim: Dim) -> Vec<GeneratedConstraint> {
    let d = match dim {
        Dim::Horizontal => 0,
        Dim::Vertical => 1,
        Dim::Depth => 2,
    };

    let left_var = match c.left_var[d] {
        Some(v) => v,
        None => return Vec::new(),
    };
    let right_var = match c.right_var[d] {
        Some(v) => v,
        None => return Vec::new(),
    };

    let mut result = Vec::with_capacity(c.shapes.len() * 2);
    for &(node_idx, half_dims) in &c.shapes {
        let half = half_dims[d];
        // left_boundary + half <= node  (node stays right of left boundary)
        result.push(GeneratedConstraint {
            left: left_var,
            right: node_idx,
            gap: half,
            equality: false,
        });
        // node + half <= right_boundary  (node stays left of right boundary)
        result.push(GeneratedConstraint {
            left: node_idx,
            right: right_var,
            gap: half,
            equality: false,
        });
    }
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vpsc::{Dim, Rectangle};

    // ===================================================================
    // Category 1: Constants
    // ===================================================================

    #[test]
    fn test_constant_free_weight() {
        assert!((FREE_WEIGHT - 0.0001).abs() < f64::EPSILON);
    }

    #[test]
    fn test_constant_default_priority() {
        assert_eq!(DEFAULT_CONSTRAINT_PRIORITY, 30000);
    }

    #[test]
    fn test_constant_nonoverlap_priority() {
        assert_eq!(PRIORITY_NONOVERLAP, 28000);
    }

    #[test]
    fn test_constant_fixed_position_weight() {
        assert!((FIXED_POSITION_WEIGHT - 100_000.0).abs() < f64::EPSILON);
    }

    // ===================================================================
    // Category 2: VariableIdMap
    // ===================================================================

    #[test]
    fn test_variable_id_map_new_is_empty() {
        let map = VariableIdMap::new();
        assert_eq!(map.mappings.len(), 0);
    }

    #[test]
    fn test_variable_id_map_add_and_lookup_forward() {
        let mut map = VariableIdMap::new();
        assert!(map.add_mapping(0, 10));
        assert_eq!(map.mapping_for(0, true), 10);
    }

    #[test]
    fn test_variable_id_map_lookup_reverse() {
        let mut map = VariableIdMap::new();
        map.add_mapping(0, 10);
        assert_eq!(map.mapping_for(10, false), 0);
    }

    #[test]
    fn test_variable_id_map_unmapped_returns_identity() {
        let map = VariableIdMap::new();
        assert_eq!(map.mapping_for(42, true), 42);
        assert_eq!(map.mapping_for(42, false), 42);
    }

    #[test]
    fn test_variable_id_map_duplicate_prevention() {
        let mut map = VariableIdMap::new();
        assert!(map.add_mapping(0, 10));
        assert!(!map.add_mapping(0, 20));
        // Original mapping preserved.
        assert_eq!(map.mapping_for(0, true), 10);
    }

    #[test]
    fn test_variable_id_map_clear() {
        let mut map = VariableIdMap::new();
        map.add_mapping(0, 10);
        map.add_mapping(1, 11);
        map.clear();
        assert_eq!(map.mapping_for(0, true), 0);
        assert_eq!(map.mapping_for(1, true), 1);
    }

    #[test]
    fn test_variable_id_map_multiple_mappings() {
        let mut map = VariableIdMap::new();
        map.add_mapping(0, 100);
        map.add_mapping(1, 101);
        map.add_mapping(2, 102);
        assert_eq!(map.mapping_for(0, true), 100);
        assert_eq!(map.mapping_for(1, true), 101);
        assert_eq!(map.mapping_for(2, true), 102);
        assert_eq!(map.mapping_for(100, false), 0);
        assert_eq!(map.mapping_for(101, false), 1);
        assert_eq!(map.mapping_for(102, false), 2);
    }

    #[test]
    fn test_variable_id_map_default() {
        let map = VariableIdMap::default();
        assert_eq!(map.mappings.len(), 0);
    }

    // ===================================================================
    // Category 3: BoundaryConstraint
    // ===================================================================

    #[test]
    fn test_boundary_generate_var() {
        let mut cc = CompoundConstraint::Boundary(BoundaryConstraint::new(Dim::Horizontal));
        if let CompoundConstraint::Boundary(ref mut b) = cc {
            b.position = 50.0;
        }
        let vars = cc.generate_variables(Dim::Horizontal, 5);
        assert_eq!(vars.len(), 1);
        assert!((vars[0].desired_position - 50.0).abs() < f64::EPSILON);
        assert!((vars[0].weight - FREE_WEIGHT).abs() < f64::EPSILON);
        assert!(!vars[0].fixed);
    }

    #[test]
    fn test_boundary_wrong_dimension_skipped() {
        let mut cc = CompoundConstraint::Boundary(BoundaryConstraint::new(Dim::Horizontal));
        let vars = cc.generate_variables(Dim::Vertical, 5);
        assert!(vars.is_empty());

        let rects = vec![Rectangle::new(0.0, 10.0, 0.0, 10.0)];
        let cs = cc.generate_separation_constraints(Dim::Vertical, &rects);
        assert!(cs.is_empty());
    }

    #[test]
    fn test_boundary_negative_offset_node_left_of_boundary() {
        let mut bc = BoundaryConstraint::new(Dim::Horizontal);
        bc.position = 100.0;
        bc.add_shape(0, -5.0); // node must be left of boundary by 5

        let mut cc = CompoundConstraint::Boundary(bc);
        cc.generate_variables(Dim::Horizontal, 3);
        let rects = vec![Rectangle::new(0.0, 10.0, 0.0, 10.0)];
        let cs = cc.generate_separation_constraints(Dim::Horizontal, &rects);

        assert_eq!(cs.len(), 1);
        // node(0) + 5 <= boundary(3)
        assert_eq!(cs[0].left, 0);
        assert_eq!(cs[0].right, 3);
        assert!((cs[0].gap - 5.0).abs() < f64::EPSILON);
        assert!(!cs[0].equality);
    }

    #[test]
    fn test_boundary_positive_offset_node_right_of_boundary() {
        let mut bc = BoundaryConstraint::new(Dim::Horizontal);
        bc.position = 100.0;
        bc.add_shape(1, 10.0); // node must be right of boundary by 10

        let mut cc = CompoundConstraint::Boundary(bc);
        cc.generate_variables(Dim::Horizontal, 4);
        let rects = vec![
            Rectangle::new(0.0, 10.0, 0.0, 10.0),
            Rectangle::new(0.0, 10.0, 0.0, 10.0),
        ];
        let cs = cc.generate_separation_constraints(Dim::Horizontal, &rects);

        assert_eq!(cs.len(), 1);
        // boundary(4) + 10 <= node(1)
        assert_eq!(cs[0].left, 4);
        assert_eq!(cs[0].right, 1);
        assert!((cs[0].gap - 10.0).abs() < f64::EPSILON);
        assert!(!cs[0].equality);
    }

    #[test]
    fn test_boundary_mixed_offsets() {
        let mut bc = BoundaryConstraint::new(Dim::Vertical);
        bc.position = 50.0;
        bc.add_shape(0, -3.0);
        bc.add_shape(1, 7.0);
        bc.add_shape(2, 0.0); // zero offset = right side

        let mut cc = CompoundConstraint::Boundary(bc);
        cc.generate_variables(Dim::Vertical, 10);
        let rects = vec![
            Rectangle::new(0.0, 10.0, 0.0, 10.0),
            Rectangle::new(0.0, 10.0, 0.0, 10.0),
            Rectangle::new(0.0, 10.0, 0.0, 10.0),
        ];
        let cs = cc.generate_separation_constraints(Dim::Vertical, &rects);

        assert_eq!(cs.len(), 3);
        // shape 0: negative offset -> node + 3 <= boundary
        assert_eq!(cs[0].left, 0);
        assert_eq!(cs[0].right, 10);
        // shape 1: positive offset -> boundary + 7 <= node
        assert_eq!(cs[1].left, 10);
        assert_eq!(cs[1].right, 1);
        // shape 2: zero offset -> boundary + 0 <= node
        assert_eq!(cs[2].left, 10);
        assert_eq!(cs[2].right, 2);
        assert!((cs[2].gap).abs() < f64::EPSILON);
    }

    #[test]
    fn test_boundary_no_shapes() {
        let bc = BoundaryConstraint::new(Dim::Horizontal);
        let mut cc = CompoundConstraint::Boundary(bc);
        cc.generate_variables(Dim::Horizontal, 0);
        let cs = cc.generate_separation_constraints(Dim::Horizontal, &[]);
        assert!(cs.is_empty());
    }

    // ===================================================================
    // Category 4: AlignmentConstraint
    // ===================================================================

    #[test]
    fn test_alignment_generate_var_unfixed() {
        let ac = AlignmentConstraint::new(Dim::Horizontal, 25.0);
        let mut cc = CompoundConstraint::Alignment(ac);
        let vars = cc.generate_variables(Dim::Horizontal, 5);
        assert_eq!(vars.len(), 1);
        assert!((vars[0].desired_position - 25.0).abs() < f64::EPSILON);
        assert!((vars[0].weight - FREE_WEIGHT).abs() < f64::EPSILON);
        assert!(!vars[0].fixed);
    }

    #[test]
    fn test_alignment_generate_var_fixed() {
        let mut ac = AlignmentConstraint::new(Dim::Vertical, 42.0);
        ac.fix_pos(42.0);
        let mut cc = CompoundConstraint::Alignment(ac);
        let vars = cc.generate_variables(Dim::Vertical, 3);
        assert_eq!(vars.len(), 1);
        assert!((vars[0].weight - FIXED_POSITION_WEIGHT).abs() < f64::EPSILON);
        assert!(vars[0].fixed);
    }

    #[test]
    fn test_alignment_equality_constraints_with_offsets() {
        let mut ac = AlignmentConstraint::new(Dim::Horizontal, 0.0);
        ac.add_shape(0, 0.0);
        ac.add_shape(1, 5.0);
        ac.add_shape(2, -3.0);

        let mut cc = CompoundConstraint::Alignment(ac);
        cc.generate_variables(Dim::Horizontal, 10);
        let rects = vec![
            Rectangle::new(0.0, 10.0, 0.0, 10.0),
            Rectangle::new(0.0, 10.0, 0.0, 10.0),
            Rectangle::new(0.0, 10.0, 0.0, 10.0),
        ];
        let cs = cc.generate_separation_constraints(Dim::Horizontal, &rects);

        assert_eq!(cs.len(), 3);
        for c in &cs {
            assert!(c.equality);
            assert_eq!(c.left, 10); // alignment var
        }
        assert_eq!(cs[0].right, 0);
        assert!((cs[0].gap - 0.0).abs() < f64::EPSILON);
        assert_eq!(cs[1].right, 1);
        assert!((cs[1].gap - 5.0).abs() < f64::EPSILON);
        assert_eq!(cs[2].right, 2);
        assert!((cs[2].gap - (-3.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_alignment_fix_and_unfix() {
        let mut ac = AlignmentConstraint::new(Dim::Horizontal, 10.0);
        assert!(!ac.is_fixed);

        ac.fix_pos(20.0);
        assert!(ac.is_fixed);
        assert!((ac.position - 20.0).abs() < f64::EPSILON);

        ac.unfix_pos();
        assert!(!ac.is_fixed);
        // Position stays at last fixed value.
        assert!((ac.position - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_alignment_wrong_dimension() {
        let ac = AlignmentConstraint::new(Dim::Horizontal, 10.0);
        let mut cc = CompoundConstraint::Alignment(ac);
        let vars = cc.generate_variables(Dim::Vertical, 5);
        assert!(vars.is_empty());
    }

    #[test]
    fn test_alignment_variable_index_set() {
        let ac = AlignmentConstraint::new(Dim::Horizontal, 10.0);
        let mut cc = CompoundConstraint::Alignment(ac);
        cc.generate_variables(Dim::Horizontal, 7);
        if let CompoundConstraint::Alignment(ref a) = cc {
            assert_eq!(a.variable_index, Some(7));
        } else {
            panic!("Expected Alignment variant");
        }
    }

    // ===================================================================
    // Category 5: SeparationConstraint
    // ===================================================================

    #[test]
    fn test_separation_inequality() {
        let sc = SeparationConstraint::new(Dim::Horizontal, 0, 1, 10.0, false);
        let cc = CompoundConstraint::Separation(sc);
        let cs = cc.generate_separation_constraints(Dim::Horizontal, &[]);
        assert_eq!(cs.len(), 1);
        assert_eq!(cs[0].left, 0);
        assert_eq!(cs[0].right, 1);
        assert!((cs[0].gap - 10.0).abs() < f64::EPSILON);
        assert!(!cs[0].equality);
    }

    #[test]
    fn test_separation_equality() {
        let sc = SeparationConstraint::new(Dim::Vertical, 2, 3, 5.0, true);
        let cc = CompoundConstraint::Separation(sc);
        let cs = cc.generate_separation_constraints(Dim::Vertical, &[]);
        assert_eq!(cs.len(), 1);
        assert!(cs[0].equality);
    }

    #[test]
    fn test_separation_set_separation() {
        let mut sc = SeparationConstraint::new(Dim::Horizontal, 0, 1, 10.0, false);
        sc.set_separation(20.0);
        assert!((sc.gap - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_separation_wrong_dimension() {
        let sc = SeparationConstraint::new(Dim::Horizontal, 0, 1, 10.0, false);
        let cc = CompoundConstraint::Separation(sc);
        let cs = cc.generate_separation_constraints(Dim::Vertical, &[]);
        assert!(cs.is_empty());
    }

    #[test]
    fn test_separation_no_generated_vars() {
        let sc = SeparationConstraint::new(Dim::Horizontal, 0, 1, 10.0, false);
        let mut cc = CompoundConstraint::Separation(sc);
        let vars = cc.generate_variables(Dim::Horizontal, 5);
        assert!(vars.is_empty());
    }

    // ===================================================================
    // Category 6: OrthogonalEdgeConstraint
    // ===================================================================

    #[test]
    fn test_orthogonal_edge_generates_equality_zero_gap() {
        let oc = OrthogonalEdgeConstraint {
            primary_dim: Dim::Vertical,
            left: 3,
            right: 7,
        };
        let cc = CompoundConstraint::OrthogonalEdge(oc);
        let cs = cc.generate_separation_constraints(Dim::Vertical, &[]);
        assert_eq!(cs.len(), 1);
        assert_eq!(cs[0].left, 3);
        assert_eq!(cs[0].right, 7);
        assert!((cs[0].gap).abs() < f64::EPSILON);
        assert!(cs[0].equality);
    }

    #[test]
    fn test_orthogonal_edge_wrong_dimension() {
        let oc = OrthogonalEdgeConstraint {
            primary_dim: Dim::Vertical,
            left: 0,
            right: 1,
        };
        let cc = CompoundConstraint::OrthogonalEdge(oc);
        let cs = cc.generate_separation_constraints(Dim::Horizontal, &[]);
        assert!(cs.is_empty());
    }

    #[test]
    fn test_orthogonal_edge_no_generated_vars() {
        let oc = OrthogonalEdgeConstraint {
            primary_dim: Dim::Horizontal,
            left: 0,
            right: 1,
        };
        let mut cc = CompoundConstraint::OrthogonalEdge(oc);
        let vars = cc.generate_variables(Dim::Horizontal, 5);
        assert!(vars.is_empty());
    }

    // ===================================================================
    // Category 7: MultiSeparationConstraint
    // ===================================================================

    #[test]
    fn test_multi_separation_multiple_pairs_inequality() {
        let mut ms = MultiSeparationConstraint::new(Dim::Horizontal, 15.0, false);
        ms.add_alignment_pair(0, 1);
        ms.add_alignment_pair(2, 3);
        ms.add_alignment_pair(4, 5);

        let cc = CompoundConstraint::MultiSeparation(ms);
        let cs = cc.generate_separation_constraints(Dim::Horizontal, &[]);
        assert_eq!(cs.len(), 3);
        for c in &cs {
            assert!((c.gap - 15.0).abs() < f64::EPSILON);
            assert!(!c.equality);
        }
    }

    #[test]
    fn test_multi_separation_equality_flag() {
        let mut ms = MultiSeparationConstraint::new(Dim::Vertical, 10.0, true);
        ms.add_alignment_pair(0, 1);

        let cc = CompoundConstraint::MultiSeparation(ms);
        let cs = cc.generate_separation_constraints(Dim::Vertical, &[]);
        assert_eq!(cs.len(), 1);
        assert!(cs[0].equality);
    }

    #[test]
    fn test_multi_separation_set_separation() {
        let mut ms = MultiSeparationConstraint::new(Dim::Horizontal, 10.0, false);
        ms.set_separation(25.0);
        assert!((ms.sep - 25.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_multi_separation_wrong_dimension() {
        let mut ms = MultiSeparationConstraint::new(Dim::Horizontal, 10.0, false);
        ms.add_alignment_pair(0, 1);
        let cc = CompoundConstraint::MultiSeparation(ms);
        let cs = cc.generate_separation_constraints(Dim::Vertical, &[]);
        assert!(cs.is_empty());
    }

    #[test]
    fn test_multi_separation_empty_pairs() {
        let ms = MultiSeparationConstraint::new(Dim::Horizontal, 10.0, false);
        let cc = CompoundConstraint::MultiSeparation(ms);
        let cs = cc.generate_separation_constraints(Dim::Horizontal, &[]);
        assert!(cs.is_empty());
    }

    // ===================================================================
    // Category 8: DistributionConstraint
    // ===================================================================

    #[test]
    fn test_distribution_always_equality() {
        let mut dc = DistributionConstraint::new(Dim::Horizontal);
        dc.set_separation(20.0);
        dc.add_alignment_pair(0, 1);
        dc.add_alignment_pair(2, 3);

        let cc = CompoundConstraint::Distribution(dc);
        let cs = cc.generate_separation_constraints(Dim::Horizontal, &[]);
        assert_eq!(cs.len(), 2);
        for c in &cs {
            assert!(c.equality);
            assert!((c.gap - 20.0).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_distribution_set_separation() {
        let mut dc = DistributionConstraint::new(Dim::Vertical);
        dc.set_separation(15.0);
        assert!((dc.sep - 15.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_distribution_wrong_dimension() {
        let mut dc = DistributionConstraint::new(Dim::Horizontal);
        dc.add_alignment_pair(0, 1);
        let cc = CompoundConstraint::Distribution(dc);
        let cs = cc.generate_separation_constraints(Dim::Vertical, &[]);
        assert!(cs.is_empty());
    }

    #[test]
    fn test_distribution_empty_pairs() {
        let dc = DistributionConstraint::new(Dim::Horizontal);
        let cc = CompoundConstraint::Distribution(dc);
        let cs = cc.generate_separation_constraints(Dim::Horizontal, &[]);
        assert!(cs.is_empty());
    }

    // ===================================================================
    // Category 9: FixedRelativeConstraint
    // ===================================================================

    #[test]
    fn test_fixed_relative_computes_offsets_horizontal() {
        let rects = vec![
            Rectangle::new(0.0, 10.0, 0.0, 10.0),   // centre (5, 5)
            Rectangle::new(20.0, 30.0, 0.0, 10.0),   // centre (25, 5)
            Rectangle::new(40.0, 50.0, 10.0, 20.0),  // centre (45, 15)
        ];
        let frc = FixedRelativeConstraint::new(&rects, vec![0, 1, 2], false);

        let cc = CompoundConstraint::FixedRelative(frc);
        let cs_h = cc.generate_separation_constraints(Dim::Horizontal, &rects);
        // Pairs: (0,1) and (1,2) in Horizontal
        assert_eq!(cs_h.len(), 2);
        // offset 0->1 horizontal = 25 - 5 = 20
        assert!((cs_h[0].gap - 20.0).abs() < f64::EPSILON);
        assert!(cs_h[0].equality);
        assert_eq!(cs_h[0].left, 0);
        assert_eq!(cs_h[0].right, 1);
        // offset 1->2 horizontal = 45 - 25 = 20
        assert!((cs_h[1].gap - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_fixed_relative_computes_offsets_vertical() {
        let rects = vec![
            Rectangle::new(0.0, 10.0, 0.0, 10.0),   // centre (5, 5)
            Rectangle::new(20.0, 30.0, 20.0, 30.0),  // centre (25, 25)
        ];
        let frc = FixedRelativeConstraint::new(&rects, vec![0, 1], false);

        let cc = CompoundConstraint::FixedRelative(frc);
        let cs_v = cc.generate_separation_constraints(Dim::Vertical, &rects);
        assert_eq!(cs_v.len(), 1);
        // offset 0->1 vertical = 25 - 5 = 20
        assert!((cs_v[0].gap - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_fixed_relative_deduplicates_and_sorts() {
        let rects = vec![
            Rectangle::new(0.0, 10.0, 0.0, 10.0),
            Rectangle::new(20.0, 30.0, 0.0, 10.0),
        ];
        // Duplicate shape IDs and unsorted.
        let frc = FixedRelativeConstraint::new(&rects, vec![1, 0, 1, 0], false);
        assert_eq!(frc.shape_vars, vec![0, 1]);
    }

    #[test]
    fn test_fixed_relative_fixed_position_var_modifications() {
        let rects = vec![
            Rectangle::new(0.0, 10.0, 0.0, 10.0),
            Rectangle::new(20.0, 30.0, 0.0, 10.0),
        ];
        let frc = FixedRelativeConstraint::new(&rects, vec![0, 1], true);
        let cc = CompoundConstraint::FixedRelative(frc);

        let mods = cc.get_var_modifications();
        assert_eq!(mods.len(), 2);
        for m in &mods {
            assert!((m.weight - FIXED_POSITION_WEIGHT).abs() < f64::EPSILON);
            assert!(m.fixed);
        }
        assert_eq!(mods[0].var_index, 0);
        assert_eq!(mods[1].var_index, 1);
    }

    #[test]
    fn test_fixed_relative_unfixed_no_var_modifications() {
        let rects = vec![
            Rectangle::new(0.0, 10.0, 0.0, 10.0),
            Rectangle::new(20.0, 30.0, 0.0, 10.0),
        ];
        let frc = FixedRelativeConstraint::new(&rects, vec![0, 1], false);
        let cc = CompoundConstraint::FixedRelative(frc);
        let mods = cc.get_var_modifications();
        assert!(mods.is_empty());
    }

    #[test]
    fn test_fixed_relative_single_shape_no_offsets() {
        let rects = vec![Rectangle::new(0.0, 10.0, 0.0, 10.0)];
        let frc = FixedRelativeConstraint::new(&rects, vec![0], false);
        let cc = CompoundConstraint::FixedRelative(frc);
        let cs = cc.generate_separation_constraints(Dim::Horizontal, &rects);
        assert!(cs.is_empty());
    }

    // ===================================================================
    // Category 10: PageBoundaryConstraint
    // ===================================================================

    #[test]
    fn test_page_boundary_generates_vars_horizontal() {
        let mut pbc = PageBoundaryConstraint::new(
            [0.0, 0.0],
            [100.0, 200.0],
            [FREE_WEIGHT, FREE_WEIGHT],
            [FREE_WEIGHT, FREE_WEIGHT],
        );
        pbc.add_shape(0, 5.0, 5.0);

        let mut cc = CompoundConstraint::PageBoundary(pbc);
        let vars = cc.generate_variables(Dim::Horizontal, 10);
        assert_eq!(vars.len(), 2);
        // Left boundary var
        assert!((vars[0].desired_position - 0.0).abs() < f64::EPSILON);
        // Right boundary var
        assert!((vars[1].desired_position - 100.0).abs() < f64::EPSILON);

        if let CompoundConstraint::PageBoundary(ref p) = cc {
            assert_eq!(p.left_var[0], Some(10));
            assert_eq!(p.right_var[0], Some(11));
        }
    }

    #[test]
    fn test_page_boundary_generates_vars_vertical() {
        let mut pbc = PageBoundaryConstraint::new(
            [0.0, 10.0],
            [100.0, 200.0],
            [1.0, 2.0],
            [3.0, 4.0],
        );
        pbc.add_shape(0, 5.0, 5.0);

        let mut cc = CompoundConstraint::PageBoundary(pbc);
        let vars = cc.generate_variables(Dim::Vertical, 6);
        assert_eq!(vars.len(), 2);
        assert!((vars[0].desired_position - 10.0).abs() < f64::EPSILON);
        assert!((vars[0].weight - 2.0).abs() < f64::EPSILON);
        assert!((vars[1].desired_position - 200.0).abs() < f64::EPSILON);
        assert!((vars[1].weight - 4.0).abs() < f64::EPSILON);

        if let CompoundConstraint::PageBoundary(ref p) = cc {
            assert_eq!(p.left_var[1], Some(6));
            assert_eq!(p.right_var[1], Some(7));
        }
    }

    #[test]
    fn test_page_boundary_constraints_per_shape() {
        let mut pbc = PageBoundaryConstraint::new(
            [0.0, 0.0],
            [100.0, 100.0],
            [FREE_WEIGHT; 2],
            [FREE_WEIGHT; 2],
        );
        pbc.add_shape(0, 5.0, 3.0);
        pbc.add_shape(1, 10.0, 8.0);

        let mut cc = CompoundConstraint::PageBoundary(pbc);
        cc.generate_variables(Dim::Horizontal, 4);
        let rects = vec![
            Rectangle::new(0.0, 10.0, 0.0, 6.0),
            Rectangle::new(0.0, 20.0, 0.0, 16.0),
        ];
        let cs = cc.generate_separation_constraints(Dim::Horizontal, &rects);

        // 2 shapes x 2 constraints each (left boundary + right boundary) = 4
        assert_eq!(cs.len(), 4);

        // Shape 0: left_var(4) + 5 <= node(0), node(0) + 5 <= right_var(5)
        assert_eq!(cs[0].left, 4);
        assert_eq!(cs[0].right, 0);
        assert!((cs[0].gap - 5.0).abs() < f64::EPSILON);
        assert!(!cs[0].equality);

        assert_eq!(cs[1].left, 0);
        assert_eq!(cs[1].right, 5);
        assert!((cs[1].gap - 5.0).abs() < f64::EPSILON);

        // Shape 1: left_var(4) + 10 <= node(1), node(1) + 10 <= right_var(5)
        assert_eq!(cs[2].left, 4);
        assert_eq!(cs[2].right, 1);
        assert!((cs[2].gap - 10.0).abs() < f64::EPSILON);

        assert_eq!(cs[3].left, 1);
        assert_eq!(cs[3].right, 5);
        assert!((cs[3].gap - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_page_boundary_both_dims() {
        let mut pbc = PageBoundaryConstraint::new(
            [0.0, 10.0],
            [100.0, 200.0],
            [FREE_WEIGHT; 2],
            [FREE_WEIGHT; 2],
        );
        pbc.add_shape(0, 5.0, 8.0);

        let mut cc = CompoundConstraint::PageBoundary(pbc);

        // Horizontal
        cc.generate_variables(Dim::Horizontal, 2);
        let cs_h = cc.generate_separation_constraints(Dim::Horizontal, &[]);
        assert_eq!(cs_h.len(), 2);
        assert!((cs_h[0].gap - 5.0).abs() < f64::EPSILON);
        assert!((cs_h[1].gap - 5.0).abs() < f64::EPSILON);

        // Vertical (need to generate vars for vertical too)
        cc.generate_variables(Dim::Vertical, 4);
        let cs_v = cc.generate_separation_constraints(Dim::Vertical, &[]);
        assert_eq!(cs_v.len(), 2);
        assert!((cs_v[0].gap - 8.0).abs() < f64::EPSILON);
        assert!((cs_v[1].gap - 8.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_page_boundary_no_shapes() {
        let pbc = PageBoundaryConstraint::new(
            [0.0, 0.0],
            [100.0, 100.0],
            [FREE_WEIGHT; 2],
            [FREE_WEIGHT; 2],
        );
        let mut cc = CompoundConstraint::PageBoundary(pbc);
        cc.generate_variables(Dim::Horizontal, 0);
        let cs = cc.generate_separation_constraints(Dim::Horizontal, &[]);
        assert!(cs.is_empty());
    }

    #[test]
    fn test_page_boundary_no_vars_generated_returns_empty_constraints() {
        let mut pbc = PageBoundaryConstraint::new(
            [0.0, 0.0],
            [100.0, 100.0],
            [FREE_WEIGHT; 2],
            [FREE_WEIGHT; 2],
        );
        pbc.add_shape(0, 5.0, 5.0);
        // Don't call generate_variables, so left_var/right_var are None.
        let cc = CompoundConstraint::PageBoundary(pbc);
        let cs = cc.generate_separation_constraints(Dim::Horizontal, &[]);
        assert!(cs.is_empty());
    }

    // ===================================================================
    // Category 11: CompoundConstraint enum delegation
    // ===================================================================

    #[test]
    fn test_compound_dimension_boundary() {
        let cc = CompoundConstraint::Boundary(BoundaryConstraint::new(Dim::Horizontal));
        assert_eq!(cc.dimension(), Dim::Horizontal);
    }

    #[test]
    fn test_compound_dimension_alignment() {
        let cc = CompoundConstraint::Alignment(AlignmentConstraint::new(Dim::Vertical, 0.0));
        assert_eq!(cc.dimension(), Dim::Vertical);
    }

    #[test]
    fn test_compound_dimension_separation() {
        let cc = CompoundConstraint::Separation(SeparationConstraint::new(
            Dim::Horizontal, 0, 1, 5.0, false,
        ));
        assert_eq!(cc.dimension(), Dim::Horizontal);
    }

    #[test]
    fn test_compound_dimension_orthogonal_edge() {
        let cc = CompoundConstraint::OrthogonalEdge(OrthogonalEdgeConstraint {
            primary_dim: Dim::Vertical,
            left: 0,
            right: 1,
        });
        assert_eq!(cc.dimension(), Dim::Vertical);
    }

    #[test]
    fn test_compound_dimension_multi_separation() {
        let cc = CompoundConstraint::MultiSeparation(MultiSeparationConstraint::new(
            Dim::Horizontal, 10.0, false,
        ));
        assert_eq!(cc.dimension(), Dim::Horizontal);
    }

    #[test]
    fn test_compound_dimension_distribution() {
        let cc = CompoundConstraint::Distribution(DistributionConstraint::new(Dim::Vertical));
        assert_eq!(cc.dimension(), Dim::Vertical);
    }

    #[test]
    fn test_compound_dimension_fixed_relative() {
        let rects = vec![Rectangle::new(0.0, 10.0, 0.0, 10.0)];
        let cc = CompoundConstraint::FixedRelative(FixedRelativeConstraint::new(
            &rects, vec![0], false,
        ));
        // Default to Horizontal for multi-dim constraints.
        assert_eq!(cc.dimension(), Dim::Horizontal);
    }

    #[test]
    fn test_compound_dimension_page_boundary() {
        let cc = CompoundConstraint::PageBoundary(PageBoundaryConstraint::new(
            [0.0; 2], [100.0; 2], [1.0; 2], [1.0; 2],
        ));
        assert_eq!(cc.dimension(), Dim::Horizontal);
    }

    #[test]
    fn test_compound_priority_is_default() {
        let cc = CompoundConstraint::Boundary(BoundaryConstraint::new(Dim::Horizontal));
        assert_eq!(cc.priority(), DEFAULT_CONSTRAINT_PRIORITY);

        let cc2 = CompoundConstraint::Alignment(AlignmentConstraint::new(Dim::Vertical, 0.0));
        assert_eq!(cc2.priority(), DEFAULT_CONSTRAINT_PRIORITY);

        let cc3 = CompoundConstraint::Separation(SeparationConstraint::new(
            Dim::Horizontal, 0, 1, 5.0, false,
        ));
        assert_eq!(cc3.priority(), DEFAULT_CONSTRAINT_PRIORITY);
    }

    // ===================================================================
    // Category 12: UnsatisfiableConstraintInfo
    // ===================================================================

    #[test]
    fn test_unsatisfiable_constraint_info_creation() {
        let info = UnsatisfiableConstraintInfo {
            left_var_index: 3,
            right_var_index: 7,
            separation: 10.0,
            equality: true,
        };
        assert_eq!(info.left_var_index, 3);
        assert_eq!(info.right_var_index, 7);
        assert!((info.separation - 10.0).abs() < f64::EPSILON);
        assert!(info.equality);
    }

    // ===================================================================
    // Category 13: Edge cases and integration
    // ===================================================================

    #[test]
    fn test_boundary_variable_index_none_before_generate() {
        let bc = BoundaryConstraint::new(Dim::Horizontal);
        assert!(bc.variable_index.is_none());
    }

    #[test]
    fn test_alignment_variable_index_none_before_generate() {
        let ac = AlignmentConstraint::new(Dim::Horizontal, 0.0);
        assert!(ac.variable_index.is_none());
    }

    #[test]
    fn test_var_modification_from_non_fixed_relative_is_empty() {
        let cc = CompoundConstraint::Boundary(BoundaryConstraint::new(Dim::Horizontal));
        assert!(cc.get_var_modifications().is_empty());

        let cc2 = CompoundConstraint::Alignment(AlignmentConstraint::new(Dim::Vertical, 0.0));
        assert!(cc2.get_var_modifications().is_empty());
    }

    #[test]
    fn test_generated_var_clone() {
        let v = GeneratedVar {
            desired_position: 42.0,
            weight: 1.0,
            fixed: true,
        };
        let v2 = v.clone();
        assert!((v2.desired_position - 42.0).abs() < f64::EPSILON);
        assert!(v2.fixed);
    }

    #[test]
    fn test_generated_constraint_clone() {
        let c = GeneratedConstraint {
            left: 0,
            right: 1,
            gap: 5.0,
            equality: true,
        };
        let c2 = c.clone();
        assert_eq!(c2.left, 0);
        assert_eq!(c2.right, 1);
        assert!(c2.equality);
    }
}
