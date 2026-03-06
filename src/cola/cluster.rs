//! Hierarchical cluster definitions for constrained layout.
//!
//! Clusters define groups of nodes that should be kept together,
//! with rectangular or convex hull boundaries.
//!
//! C++ ref: libcola/cluster.h, libcola/cluster.cpp

use std::collections::BTreeSet;

use crate::vpsc::{Dim, Rectangle};
use crate::cola::r#box::Box as MarginBox;
use crate::cola::convex_hull::convex_hull;

/// Default weight for cluster boundary variables.
const CLUSTER_VAR_WEIGHT: f64 = 0.0000001;
/// Default internal edge weight factor.
const DEFAULT_INTERNAL_EDGE_WEIGHT_FACTOR: f64 = 1.0;
/// Number of corners per rectangle (for convex hull computation).
const CORNERS_PER_RECT: usize = 4;
/// Number of boundary variables per cluster (min and max).
#[cfg(test)]
const BOUNDARY_VARS_PER_CLUSTER: usize = 2;
/// Index offset for the max boundary variable relative to `cluster_var_id`.
const MAX_VAR_OFFSET: usize = 1;
/// Dimension index for X axis (matches `Dim::Horizontal` discriminant).
const X_DIM: usize = 0;
/// Dimension index for Y axis (matches `Dim::Vertical` discriminant).
const Y_DIM: usize = 1;
/// Number of hull points for a rectangular boundary.
#[cfg(test)]
const RECT_HULL_POINTS: usize = 4;

/// A generated variable specification from cluster boundary creation.
#[derive(Debug, Clone)]
pub struct ClusterVar {
    pub desired_position: f64,
    pub weight: f64,
}

/// Common cluster data shared by all cluster types.
#[derive(Debug, Clone)]
pub struct ClusterData {
    pub nodes: BTreeSet<usize>,
    pub clusters: Vec<Cluster>,
    pub bounds: Rectangle,
    pub hull_x: Vec<f64>,
    pub hull_y: Vec<f64>,
    pub cluster_var_id: usize,
    pub var_weight: f64,
    pub internal_edge_weight_factor: f64,
    desired_bounds: Option<Rectangle>,
}

impl ClusterData {
    /// Creates a new `ClusterData` with default weights and empty bounds.
    pub fn new() -> Self {
        Self {
            nodes: BTreeSet::new(),
            clusters: Vec::new(),
            bounds: Rectangle::invalid(),
            hull_x: Vec::new(),
            hull_y: Vec::new(),
            cluster_var_id: 0,
            var_weight: CLUSTER_VAR_WEIGHT,
            internal_edge_weight_factor: DEFAULT_INTERNAL_EDGE_WEIGHT_FACTOR,
            desired_bounds: None,
        }
    }

    /// Add a child node index to this cluster.
    pub fn add_child_node(&mut self, index: usize) {
        self.nodes.insert(index);
    }

    /// Add a child sub-cluster to this cluster.
    pub fn add_child_cluster(&mut self, cluster: Cluster) {
        self.clusters.push(cluster);
    }

    /// Set desired bounds for this cluster's boundary variables.
    pub fn set_desired_bounds(&mut self, bounds: Rectangle) {
        self.desired_bounds = Some(bounds);
    }

    /// Clear desired bounds (revert to using computed bounds).
    pub fn unset_desired_bounds(&mut self) {
        self.desired_bounds = None;
    }

    /// Returns the desired bounds if set.
    pub fn desired_bounds(&self) -> Option<&Rectangle> {
        self.desired_bounds.as_ref()
    }

    /// Total area of contained nodes' rectangles plus child clusters' areas.
    pub fn area(&self, rects: &[Rectangle]) -> f64 {
        let node_area: f64 = self
            .nodes
            .iter()
            .map(|&i| rects[i].width() * rects[i].height())
            .sum();
        let child_area: f64 = self
            .clusters
            .iter()
            .map(|c| c.data().area(rects))
            .sum();
        node_area + child_area
    }
}

impl Default for ClusterData {
    fn default() -> Self {
        Self::new()
    }
}

/// A cluster in the hierarchy.
#[derive(Debug, Clone)]
pub enum Cluster {
    Root(RootCluster),
    Rectangular(RectangularCluster),
    Convex(ConvexCluster),
}

#[derive(Debug, Clone)]
pub struct RootCluster {
    pub data: ClusterData,
    pub allows_multiple_parents: bool,
}

#[derive(Debug, Clone)]
pub struct RectangularCluster {
    pub data: ClusterData,
    /// `None` means variable-size; `Some(i)` means this cluster is fixed to rectangle `i`.
    pub rectangle_index: Option<usize>,
    pub margin: MarginBox,
    pub padding: MarginBox,
}

#[derive(Debug, Clone)]
pub struct ConvexCluster {
    pub data: ClusterData,
    /// Which rectangle each hull point originated from.
    pub hull_rect_ids: Vec<usize>,
    /// Which corner (0..3) of that rectangle each hull point represents.
    pub hull_corners: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Cluster enum delegation
// ---------------------------------------------------------------------------

impl Cluster {
    /// Immutable access to the common cluster data.
    pub fn data(&self) -> &ClusterData {
        match self {
            Cluster::Root(c) => &c.data,
            Cluster::Rectangular(c) => &c.data,
            Cluster::Convex(c) => &c.data,
        }
    }

    /// Mutable access to the common cluster data.
    pub fn data_mut(&mut self) -> &mut ClusterData {
        match self {
            Cluster::Root(c) => &mut c.data,
            Cluster::Rectangular(c) => &mut c.data,
            Cluster::Convex(c) => &mut c.data,
        }
    }

    /// Compute the boundary hull for this cluster from child node rectangles.
    pub fn compute_boundary(&mut self, rects: &[Rectangle]) {
        match self {
            Cluster::Root(c) => c.compute_boundary(rects),
            Cluster::Rectangular(c) => c.compute_boundary(rects),
            Cluster::Convex(c) => c.compute_boundary(rects),
        }
    }

    /// Compute the bounding rectangle for this cluster from child node rectangles.
    pub fn compute_bounding_rect(&mut self, rects: &[Rectangle]) {
        match self {
            Cluster::Root(c) => c.compute_bounding_rect(rects),
            Cluster::Rectangular(c) => c.compute_bounding_rect(rects),
            Cluster::Convex(c) => c.compute_bounding_rect(rects),
        }
    }

    /// Create boundary variables for this cluster.
    ///
    /// Returns a `Vec<ClusterVar>` (always [`BOUNDARY_VARS_PER_CLUSTER`]: min then max).
    /// Sets `cluster_var_id` on the data so the caller knows which variable
    /// indices correspond to this cluster.
    ///
    /// Child clusters' variables are created first (recursively), so the
    /// returned vec includes children's vars followed by this cluster's vars.
    pub fn create_vars(
        &mut self,
        dim: Dim,
        rects: &[Rectangle],
        next_var_id: usize,
    ) -> Vec<ClusterVar> {
        let mut vars = Vec::new();
        let mut current_id = next_var_id;

        // Recurse into child clusters first.
        let child_count = self.data().clusters.len();
        for i in 0..child_count {
            // Temporarily take the child out so we can recurse without borrow issues.
            let mut child = {
                let children = &mut self.data_mut().clusters;
                std::mem::replace(
                    &mut children[i],
                    Cluster::Root(RootCluster::new()),
                )
            };
            let child_vars = child.create_vars(dim, rects, current_id);
            current_id += child_vars.len();
            vars.extend(child_vars);
            self.data_mut().clusters[i] = child;
        }

        // Now create this cluster's own 2 boundary variables.
        let data = self.data_mut();
        data.cluster_var_id = current_id;

        let (min_pos, max_pos) = match &data.desired_bounds {
            Some(db) => (db.get_min_d(dim), db.get_max_d(dim)),
            None => (data.bounds.get_min_d(dim), data.bounds.get_max_d(dim)),
        };

        vars.push(ClusterVar {
            desired_position: min_pos,
            weight: data.var_weight,
        });
        vars.push(ClusterVar {
            desired_position: max_pos,
            weight: data.var_weight,
        });

        vars
    }

    /// Update bounds from variable final positions.
    ///
    /// `var_positions` is indexed by variable id; each entry is `(min, max)`
    /// for the dimension being updated. The cluster reads positions at
    /// `cluster_var_id` (min) and `cluster_var_id + 1` (max).
    pub fn update_bounds(&mut self, dim: Dim, var_positions: &[f64]) {
        // Recurse into children.
        let child_count = self.data().clusters.len();
        for i in 0..child_count {
            let mut child = {
                let children = &mut self.data_mut().clusters;
                std::mem::replace(
                    &mut children[i],
                    Cluster::Root(RootCluster::new()),
                )
            };
            child.update_bounds(dim, var_positions);
            self.data_mut().clusters[i] = child;
        }

        let data = self.data_mut();
        let min_val = var_positions[data.cluster_var_id];
        let max_val = var_positions[data.cluster_var_id + MAX_VAR_OFFSET];
        data.bounds.set_min_d(dim, min_val);
        data.bounds.set_max_d(dim, max_val);
    }
}

// ---------------------------------------------------------------------------
// RootCluster
// ---------------------------------------------------------------------------

impl RootCluster {
    pub fn new() -> Self {
        Self {
            data: ClusterData::new(),
            allows_multiple_parents: false,
        }
    }

    /// Returns `true` if there are no child sub-clusters (the hierarchy is flat).
    pub fn flat(&self) -> bool {
        self.data.clusters.is_empty()
    }

    /// Compute boundary: delegates to each child cluster.
    pub fn compute_boundary(&mut self, rects: &[Rectangle]) {
        for child in &mut self.data.clusters {
            child.compute_boundary(rects);
        }
    }

    /// Compute bounding rect: delegates to each child cluster.
    pub fn compute_bounding_rect(&mut self, rects: &[Rectangle]) {
        for child in &mut self.data.clusters {
            child.compute_bounding_rect(rects);
        }
    }
}

impl Default for RootCluster {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// RectangularCluster
// ---------------------------------------------------------------------------

impl RectangularCluster {
    /// Create a variable-size rectangular cluster.
    pub fn new() -> Self {
        Self {
            data: ClusterData::new(),
            rectangle_index: None,
            margin: MarginBox::empty_box(),
            padding: MarginBox::empty_box(),
        }
    }

    /// Create a rectangular cluster fixed to a specific rectangle.
    pub fn from_rectangle(rect_index: usize) -> Self {
        Self {
            data: ClusterData::new(),
            rectangle_index: Some(rect_index),
            margin: MarginBox::empty_box(),
            padding: MarginBox::empty_box(),
        }
    }

    /// Returns `true` if this cluster is backed by a fixed rectangle.
    pub fn is_from_fixed_rectangle(&self) -> bool {
        self.rectangle_index.is_some()
    }

    /// Compute boundary: creates a 4-point rectangular hull from min/max of
    /// child node rectangles.
    pub fn compute_boundary(&mut self, rects: &[Rectangle]) {
        if self.data.nodes.is_empty() && self.data.clusters.is_empty() {
            self.data.hull_x.clear();
            self.data.hull_y.clear();
            return;
        }

        // Recurse into child clusters.
        for child in &mut self.data.clusters {
            child.compute_boundary(rects);
        }

        let mut min_x = f64::MAX;
        let mut max_x = f64::MIN;
        let mut min_y = f64::MAX;
        let mut max_y = f64::MIN;

        for &node in &self.data.nodes {
            let r = &rects[node];
            min_x = min_x.min(r.get_min_x());
            max_x = max_x.max(r.get_max_x());
            min_y = min_y.min(r.get_min_y());
            max_y = max_y.max(r.get_max_y());
        }

        // Also include child cluster bounds.
        for child in &self.data.clusters {
            let cb = &child.data().bounds;
            if cb.is_valid() {
                min_x = min_x.min(cb.get_min_x());
                max_x = max_x.max(cb.get_max_x());
                min_y = min_y.min(cb.get_min_y());
                max_y = max_y.max(cb.get_max_y());
            }
        }

        // Apply margin outward.
        min_x -= self.margin.min(X_DIM);
        max_x += self.margin.max(X_DIM);
        min_y -= self.margin.min(Y_DIM);
        max_y += self.margin.max(Y_DIM);

        // Store as 4-point hull (CCW: bottom-left, bottom-right, top-right, top-left).
        self.data.hull_x = vec![min_x, max_x, max_x, min_x];
        self.data.hull_y = vec![min_y, min_y, max_y, max_y];
    }

    /// Compute bounding rectangle.
    ///
    /// If backed by a fixed rectangle, uses that rectangle.
    /// Otherwise, aggregates children and applies padding.
    pub fn compute_bounding_rect(&mut self, rects: &[Rectangle]) {
        if let Some(ri) = self.rectangle_index {
            self.data.bounds = rects[ri].clone();
            return;
        }

        // Recurse into child clusters.
        for child in &mut self.data.clusters {
            child.compute_bounding_rect(rects);
        }

        let mut combined = Rectangle::invalid();

        for &node in &self.data.nodes {
            combined = combined.union_with(&rects[node]);
        }

        for child in &self.data.clusters {
            let cb = &child.data().bounds;
            if cb.is_valid() {
                combined = combined.union_with(cb);
            }
        }

        if combined.is_valid() {
            // Apply padding (expands outward).
            combined = self.padding.apply_to_rectangle(&combined);
        }

        self.data.bounds = combined;
    }
}

impl Default for RectangularCluster {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ConvexCluster
// ---------------------------------------------------------------------------

impl ConvexCluster {
    pub fn new() -> Self {
        Self {
            data: ClusterData::new(),
            hull_rect_ids: Vec::new(),
            hull_corners: Vec::new(),
        }
    }

    /// Compute boundary: takes all 4 corners of each child node's rectangle,
    /// runs convex hull, and stores the result.
    pub fn compute_boundary(&mut self, rects: &[Rectangle]) {
        if self.data.nodes.is_empty() {
            self.data.hull_x.clear();
            self.data.hull_y.clear();
            self.hull_rect_ids.clear();
            self.hull_corners.clear();
            return;
        }

        let num_points = self.data.nodes.len() * CORNERS_PER_RECT;
        let mut all_x = Vec::with_capacity(num_points);
        let mut all_y = Vec::with_capacity(num_points);
        let mut rect_ids = Vec::with_capacity(num_points);
        let mut corners = Vec::with_capacity(num_points);

        for &node in &self.data.nodes {
            let r = &rects[node];
            // Corner 0: min_x, min_y
            all_x.push(r.get_min_x());
            all_y.push(r.get_min_y());
            rect_ids.push(node);
            corners.push(0u8);

            // Corner 1: max_x, min_y
            all_x.push(r.get_max_x());
            all_y.push(r.get_min_y());
            rect_ids.push(node);
            corners.push(1u8);

            // Corner 2: max_x, max_y
            all_x.push(r.get_max_x());
            all_y.push(r.get_max_y());
            rect_ids.push(node);
            corners.push(2u8);

            // Corner 3: min_x, max_y
            all_x.push(r.get_min_x());
            all_y.push(r.get_max_y());
            rect_ids.push(node);
            corners.push(3u8);
        }

        let hull_indices = convex_hull(&all_x, &all_y);

        self.data.hull_x = hull_indices.iter().map(|&i| all_x[i]).collect();
        self.data.hull_y = hull_indices.iter().map(|&i| all_y[i]).collect();
        self.hull_rect_ids = hull_indices.iter().map(|&i| rect_ids[i]).collect();
        self.hull_corners = hull_indices.iter().map(|&i| corners[i]).collect();

        // Also update bounds from hull extents.
        if !self.data.hull_x.is_empty() {
            let min_x = self.data.hull_x.iter().cloned().fold(f64::MAX, f64::min);
            let max_x = self.data.hull_x.iter().cloned().fold(f64::MIN, f64::max);
            let min_y = self.data.hull_y.iter().cloned().fold(f64::MAX, f64::min);
            let max_y = self.data.hull_y.iter().cloned().fold(f64::MIN, f64::max);
            self.data.bounds = Rectangle::new(min_x, max_x, min_y, max_y);
        }
    }

    /// Compute bounding rectangle from child nodes.
    pub fn compute_bounding_rect(&mut self, rects: &[Rectangle]) {
        let mut combined = Rectangle::invalid();
        for &node in &self.data.nodes {
            combined = combined.union_with(&rects[node]);
        }
        self.data.bounds = combined;
    }
}

impl Default for ConvexCluster {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    /// Helper: create a simple rectangle at known coordinates.
    fn make_rect(min_x: f64, max_x: f64, min_y: f64, max_y: f64) -> Rectangle {
        Rectangle::new(min_x, max_x, min_y, max_y)
    }

    /// Helper: create a standard test set of rectangles.
    fn test_rects() -> Vec<Rectangle> {
        vec![
            make_rect(0.0, 10.0, 0.0, 10.0),   // 0: 10x10 at origin
            make_rect(20.0, 30.0, 20.0, 30.0),  // 1: 10x10 offset
            make_rect(5.0, 15.0, 5.0, 15.0),    // 2: 10x10 overlapping 0
        ]
    }

    // ===================================================================
    // Category 1: ClusterData basics
    // ===================================================================

    #[test]
    fn cluster_data_new_defaults() {
        let d = ClusterData::new();
        assert!(d.nodes.is_empty());
        assert!(d.clusters.is_empty());
        assert!(!d.bounds.is_valid());
        assert!(d.hull_x.is_empty());
        assert!(d.hull_y.is_empty());
        assert_eq!(d.cluster_var_id, 0);
        assert!((d.var_weight - CLUSTER_VAR_WEIGHT).abs() < EPSILON);
        assert!(
            (d.internal_edge_weight_factor - DEFAULT_INTERNAL_EDGE_WEIGHT_FACTOR).abs() < EPSILON
        );
        assert!(d.desired_bounds().is_none());
    }

    #[test]
    fn cluster_data_add_child_node() {
        let mut d = ClusterData::new();
        d.add_child_node(5);
        d.add_child_node(3);
        d.add_child_node(5); // duplicate - BTreeSet ignores
        assert_eq!(d.nodes.len(), 2);
        assert!(d.nodes.contains(&3));
        assert!(d.nodes.contains(&5));
    }

    #[test]
    fn cluster_data_add_child_cluster() {
        let mut d = ClusterData::new();
        d.add_child_cluster(Cluster::Rectangular(RectangularCluster::new()));
        d.add_child_cluster(Cluster::Convex(ConvexCluster::new()));
        assert_eq!(d.clusters.len(), 2);
    }

    #[test]
    fn cluster_data_set_unset_desired_bounds() {
        let mut d = ClusterData::new();
        assert!(d.desired_bounds().is_none());

        let bounds = make_rect(1.0, 2.0, 3.0, 4.0);
        d.set_desired_bounds(bounds);
        assert!(d.desired_bounds().is_some());
        assert!((d.desired_bounds().unwrap().get_min_x() - 1.0).abs() < EPSILON);

        d.unset_desired_bounds();
        assert!(d.desired_bounds().is_none());
    }

    #[test]
    fn cluster_data_area_with_nodes() {
        let rects = test_rects();
        let mut d = ClusterData::new();
        d.add_child_node(0); // 10x10 = 100
        d.add_child_node(1); // 10x10 = 100
        assert!((d.area(&rects) - 200.0).abs() < EPSILON);
    }

    #[test]
    fn cluster_data_area_with_child_clusters() {
        let rects = test_rects();
        let mut child_data = ClusterData::new();
        child_data.add_child_node(2); // 10x10 = 100
        let child = Cluster::Rectangular(RectangularCluster {
            data: child_data,
            rectangle_index: None,
            margin: MarginBox::empty_box(),
            padding: MarginBox::empty_box(),
        });

        let mut d = ClusterData::new();
        d.add_child_node(0); // 100
        d.add_child_cluster(child); // child has 100
        assert!((d.area(&rects) - 200.0).abs() < EPSILON);
    }

    #[test]
    fn cluster_data_area_empty() {
        let rects = test_rects();
        let d = ClusterData::new();
        assert!((d.area(&rects) - 0.0).abs() < EPSILON);
    }

    // ===================================================================
    // Category 2: RootCluster
    // ===================================================================

    #[test]
    fn root_cluster_new() {
        let rc = RootCluster::new();
        assert!(!rc.allows_multiple_parents);
        assert!(rc.flat());
    }

    #[test]
    fn root_cluster_flat_when_empty() {
        let rc = RootCluster::new();
        assert!(rc.flat());
    }

    #[test]
    fn root_cluster_not_flat_with_sub_clusters() {
        let mut rc = RootCluster::new();
        rc.data.add_child_cluster(Cluster::Rectangular(RectangularCluster::new()));
        assert!(!rc.flat());
    }

    #[test]
    fn root_cluster_compute_boundary_delegates() {
        let rects = test_rects();
        let mut child = RectangularCluster::new();
        child.data.add_child_node(0);
        child.data.add_child_node(1);

        let mut rc = RootCluster::new();
        rc.data.add_child_cluster(Cluster::Rectangular(child));
        rc.compute_boundary(&rects);

        // After compute_boundary, the child should have its hull set.
        let child_data = rc.data.clusters[0].data();
        assert_eq!(child_data.hull_x.len(), RECT_HULL_POINTS);
        assert_eq!(child_data.hull_y.len(), RECT_HULL_POINTS);
    }

    // ===================================================================
    // Category 3: RectangularCluster construction
    // ===================================================================

    #[test]
    fn rectangular_cluster_new_variable_size() {
        let rc = RectangularCluster::new();
        assert!(rc.rectangle_index.is_none());
        assert!(!rc.is_from_fixed_rectangle());
        assert!(rc.margin.is_empty());
        assert!(rc.padding.is_empty());
    }

    #[test]
    fn rectangular_cluster_from_rectangle() {
        let rc = RectangularCluster::from_rectangle(42);
        assert_eq!(rc.rectangle_index, Some(42));
        assert!(rc.is_from_fixed_rectangle());
    }

    #[test]
    fn rectangular_cluster_margin_padding_settable() {
        let mut rc = RectangularCluster::new();
        rc.margin = MarginBox::uniform(5.0);
        rc.padding = MarginBox::new(1.0, 2.0, 3.0, 4.0);
        assert!(!rc.margin.is_empty());
        assert!(!rc.padding.is_empty());
    }

    // ===================================================================
    // Category 4: RectangularCluster compute_boundary
    // ===================================================================

    #[test]
    fn rectangular_compute_boundary_from_child_nodes() {
        let rects = test_rects();
        let mut rc = RectangularCluster::new();
        rc.data.add_child_node(0); // [0,10] x [0,10]
        rc.data.add_child_node(1); // [20,30] x [20,30]
        rc.compute_boundary(&rects);

        assert_eq!(rc.data.hull_x.len(), RECT_HULL_POINTS);
        assert_eq!(rc.data.hull_y.len(), RECT_HULL_POINTS);

        let min_x = rc.data.hull_x.iter().cloned().fold(f64::MAX, f64::min);
        let max_x = rc.data.hull_x.iter().cloned().fold(f64::MIN, f64::max);
        let min_y = rc.data.hull_y.iter().cloned().fold(f64::MAX, f64::min);
        let max_y = rc.data.hull_y.iter().cloned().fold(f64::MIN, f64::max);

        assert!((min_x - 0.0).abs() < EPSILON);
        assert!((max_x - 30.0).abs() < EPSILON);
        assert!((min_y - 0.0).abs() < EPSILON);
        assert!((max_y - 30.0).abs() < EPSILON);
    }

    #[test]
    fn rectangular_compute_boundary_with_margin() {
        let rects = vec![make_rect(10.0, 20.0, 10.0, 20.0)];
        let mut rc = RectangularCluster::new();
        rc.data.add_child_node(0);
        rc.margin = MarginBox::uniform(5.0);
        rc.compute_boundary(&rects);

        let min_x = rc.data.hull_x.iter().cloned().fold(f64::MAX, f64::min);
        let max_x = rc.data.hull_x.iter().cloned().fold(f64::MIN, f64::max);
        let min_y = rc.data.hull_y.iter().cloned().fold(f64::MAX, f64::min);
        let max_y = rc.data.hull_y.iter().cloned().fold(f64::MIN, f64::max);

        assert!((min_x - 5.0).abs() < EPSILON);
        assert!((max_x - 25.0).abs() < EPSILON);
        assert!((min_y - 5.0).abs() < EPSILON);
        assert!((max_y - 25.0).abs() < EPSILON);
    }

    #[test]
    fn rectangular_compute_boundary_empty_cluster() {
        let rects = test_rects();
        let mut rc = RectangularCluster::new();
        rc.compute_boundary(&rects);
        assert!(rc.data.hull_x.is_empty());
        assert!(rc.data.hull_y.is_empty());
    }

    // ===================================================================
    // Category 5: RectangularCluster compute_bounding_rect
    // ===================================================================

    #[test]
    fn rectangular_bounding_rect_fixed_uses_that_rect() {
        let rects = test_rects();
        let mut rc = RectangularCluster::from_rectangle(1);
        rc.data.add_child_node(0); // should be ignored for bounds
        rc.compute_bounding_rect(&rects);

        assert!((rc.data.bounds.get_min_x() - 20.0).abs() < EPSILON);
        assert!((rc.data.bounds.get_max_x() - 30.0).abs() < EPSILON);
    }

    #[test]
    fn rectangular_bounding_rect_variable_aggregates_children() {
        let rects = test_rects();
        let mut rc = RectangularCluster::new();
        rc.data.add_child_node(0); // [0,10] x [0,10]
        rc.data.add_child_node(1); // [20,30] x [20,30]
        rc.compute_bounding_rect(&rects);

        assert!((rc.data.bounds.get_min_x() - 0.0).abs() < EPSILON);
        assert!((rc.data.bounds.get_max_x() - 30.0).abs() < EPSILON);
        assert!((rc.data.bounds.get_min_y() - 0.0).abs() < EPSILON);
        assert!((rc.data.bounds.get_max_y() - 30.0).abs() < EPSILON);
    }

    #[test]
    fn rectangular_bounding_rect_with_padding() {
        let rects = vec![make_rect(10.0, 20.0, 10.0, 20.0)];
        let mut rc = RectangularCluster::new();
        rc.data.add_child_node(0);
        rc.padding = MarginBox::uniform(3.0);
        rc.compute_bounding_rect(&rects);

        assert!((rc.data.bounds.get_min_x() - 7.0).abs() < EPSILON);
        assert!((rc.data.bounds.get_max_x() - 23.0).abs() < EPSILON);
        assert!((rc.data.bounds.get_min_y() - 7.0).abs() < EPSILON);
        assert!((rc.data.bounds.get_max_y() - 23.0).abs() < EPSILON);
    }

    #[test]
    fn rectangular_bounding_rect_includes_child_cluster_bounds() {
        let rects = test_rects();
        let mut child = RectangularCluster::new();
        child.data.add_child_node(1); // [20,30] x [20,30]
        child.compute_bounding_rect(&rects);

        let mut rc = RectangularCluster::new();
        rc.data.add_child_node(0); // [0,10] x [0,10]
        rc.data.add_child_cluster(Cluster::Rectangular(child));
        rc.compute_bounding_rect(&rects);

        assert!((rc.data.bounds.get_min_x() - 0.0).abs() < EPSILON);
        assert!((rc.data.bounds.get_max_x() - 30.0).abs() < EPSILON);
    }

    // ===================================================================
    // Category 6: ConvexCluster
    // ===================================================================

    #[test]
    fn convex_cluster_new() {
        let cc = ConvexCluster::new();
        assert!(cc.data.nodes.is_empty());
        assert!(cc.hull_rect_ids.is_empty());
        assert!(cc.hull_corners.is_empty());
    }

    #[test]
    fn convex_compute_boundary_single_rect() {
        let rects = vec![make_rect(0.0, 10.0, 0.0, 10.0)];
        let mut cc = ConvexCluster::new();
        cc.data.add_child_node(0);
        cc.compute_boundary(&rects);

        // A single rectangle has 4 corners; convex hull should yield 4 points.
        assert_eq!(cc.data.hull_x.len(), CORNERS_PER_RECT);
        assert_eq!(cc.data.hull_y.len(), CORNERS_PER_RECT);
        assert_eq!(cc.hull_rect_ids.len(), CORNERS_PER_RECT);
        assert_eq!(cc.hull_corners.len(), CORNERS_PER_RECT);

        // All hull_rect_ids should reference rect 0.
        for &id in &cc.hull_rect_ids {
            assert_eq!(id, 0);
        }
    }

    #[test]
    fn convex_compute_boundary_multiple_rects() {
        let rects = vec![
            make_rect(0.0, 10.0, 0.0, 10.0),
            make_rect(20.0, 30.0, 0.0, 10.0),
        ];
        let mut cc = ConvexCluster::new();
        cc.data.add_child_node(0);
        cc.data.add_child_node(1);
        cc.compute_boundary(&rects);

        // The hull should encompass both rectangles.
        let min_x = cc.data.hull_x.iter().cloned().fold(f64::MAX, f64::min);
        let max_x = cc.data.hull_x.iter().cloned().fold(f64::MIN, f64::max);
        assert!((min_x - 0.0).abs() < EPSILON);
        assert!((max_x - 30.0).abs() < EPSILON);
    }

    #[test]
    fn convex_compute_boundary_empty() {
        let rects = test_rects();
        let mut cc = ConvexCluster::new();
        cc.compute_boundary(&rects);
        assert!(cc.data.hull_x.is_empty());
        assert!(cc.hull_rect_ids.is_empty());
    }

    #[test]
    fn convex_compute_boundary_updates_bounds() {
        let rects = vec![make_rect(5.0, 15.0, 10.0, 20.0)];
        let mut cc = ConvexCluster::new();
        cc.data.add_child_node(0);
        cc.compute_boundary(&rects);

        assert!(cc.data.bounds.is_valid());
        assert!((cc.data.bounds.get_min_x() - 5.0).abs() < EPSILON);
        assert!((cc.data.bounds.get_max_x() - 15.0).abs() < EPSILON);
        assert!((cc.data.bounds.get_min_y() - 10.0).abs() < EPSILON);
        assert!((cc.data.bounds.get_max_y() - 20.0).abs() < EPSILON);
    }

    // ===================================================================
    // Category 7: create_vars
    // ===================================================================

    #[test]
    fn create_vars_generates_two_vars() {
        let rects = vec![make_rect(0.0, 10.0, 0.0, 20.0)];
        let mut rc = RectangularCluster::new();
        rc.data.add_child_node(0);
        rc.compute_bounding_rect(&rects);

        let mut cluster = Cluster::Rectangular(rc);
        let vars = cluster.create_vars(Dim::Horizontal, &rects, 0);

        assert_eq!(vars.len(), BOUNDARY_VARS_PER_CLUSTER);
        assert_eq!(cluster.data().cluster_var_id, 0);
    }

    #[test]
    fn create_vars_positions_from_bounds_horizontal() {
        let rects = vec![make_rect(5.0, 15.0, 0.0, 10.0)];
        let mut rc = RectangularCluster::new();
        rc.data.add_child_node(0);
        rc.compute_bounding_rect(&rects);

        let mut cluster = Cluster::Rectangular(rc);
        let vars = cluster.create_vars(Dim::Horizontal, &rects, 0);

        assert!((vars[0].desired_position - 5.0).abs() < EPSILON);
        assert!((vars[1].desired_position - 15.0).abs() < EPSILON);
    }

    #[test]
    fn create_vars_positions_from_bounds_vertical() {
        let rects = vec![make_rect(0.0, 10.0, 3.0, 17.0)];
        let mut rc = RectangularCluster::new();
        rc.data.add_child_node(0);
        rc.compute_bounding_rect(&rects);

        let mut cluster = Cluster::Rectangular(rc);
        let vars = cluster.create_vars(Dim::Vertical, &rects, 0);

        assert!((vars[0].desired_position - 3.0).abs() < EPSILON);
        assert!((vars[1].desired_position - 17.0).abs() < EPSILON);
    }

    #[test]
    fn create_vars_uses_desired_bounds_when_set() {
        let rects = vec![make_rect(0.0, 10.0, 0.0, 10.0)];
        let mut rc = RectangularCluster::new();
        rc.data.add_child_node(0);
        rc.compute_bounding_rect(&rects);
        rc.data.set_desired_bounds(make_rect(100.0, 200.0, 100.0, 200.0));

        let mut cluster = Cluster::Rectangular(rc);
        let vars = cluster.create_vars(Dim::Horizontal, &rects, 0);

        assert!((vars[0].desired_position - 100.0).abs() < EPSILON);
        assert!((vars[1].desired_position - 200.0).abs() < EPSILON);
    }

    #[test]
    fn create_vars_sets_cluster_var_id() {
        let rects = vec![make_rect(0.0, 10.0, 0.0, 10.0)];
        let mut rc = RectangularCluster::new();
        rc.data.add_child_node(0);
        rc.compute_bounding_rect(&rects);

        let start_id = 42;
        let mut cluster = Cluster::Rectangular(rc);
        let vars = cluster.create_vars(Dim::Horizontal, &rects, start_id);

        assert_eq!(cluster.data().cluster_var_id, start_id);
        assert_eq!(vars.len(), BOUNDARY_VARS_PER_CLUSTER);
    }

    #[test]
    fn create_vars_weight_matches_cluster_weight() {
        let rects = vec![make_rect(0.0, 10.0, 0.0, 10.0)];
        let mut rc = RectangularCluster::new();
        rc.data.add_child_node(0);
        rc.data.var_weight = 0.5;
        rc.compute_bounding_rect(&rects);

        let mut cluster = Cluster::Rectangular(rc);
        let vars = cluster.create_vars(Dim::Horizontal, &rects, 0);

        for var in &vars {
            assert!((var.weight - 0.5).abs() < EPSILON);
        }
    }

    #[test]
    fn create_vars_recursive_with_child_clusters() {
        let rects = vec![
            make_rect(0.0, 10.0, 0.0, 10.0),
            make_rect(20.0, 30.0, 20.0, 30.0),
        ];

        let mut child = RectangularCluster::new();
        child.data.add_child_node(1);
        child.compute_bounding_rect(&rects);

        let mut parent = RectangularCluster::new();
        parent.data.add_child_node(0);
        parent.data.add_child_cluster(Cluster::Rectangular(child));
        parent.compute_bounding_rect(&rects);

        let mut cluster = Cluster::Rectangular(parent);
        let vars = cluster.create_vars(Dim::Horizontal, &rects, 0);

        // Child gets 2 vars (ids 0,1), parent gets 2 vars (ids 2,3).
        assert_eq!(vars.len(), 2 * BOUNDARY_VARS_PER_CLUSTER);
        assert_eq!(cluster.data().cluster_var_id, BOUNDARY_VARS_PER_CLUSTER);

        // Check child's var_id was set.
        let child_var_id = cluster.data().clusters[0].data().cluster_var_id;
        assert_eq!(child_var_id, 0);
    }

    // ===================================================================
    // Category 8: Cluster enum dispatch
    // ===================================================================

    #[test]
    fn cluster_enum_data_access() {
        let mut rc = RectangularCluster::new();
        rc.data.add_child_node(7);
        let cluster = Cluster::Rectangular(rc);
        assert!(cluster.data().nodes.contains(&7));
    }

    #[test]
    fn cluster_enum_data_mut_access() {
        let rc = RectangularCluster::new();
        let mut cluster = Cluster::Rectangular(rc);
        cluster.data_mut().add_child_node(42);
        assert!(cluster.data().nodes.contains(&42));
    }

    #[test]
    fn cluster_enum_compute_boundary_dispatch_rectangular() {
        let rects = vec![make_rect(0.0, 10.0, 0.0, 10.0)];
        let mut rc = RectangularCluster::new();
        rc.data.add_child_node(0);
        let mut cluster = Cluster::Rectangular(rc);
        cluster.compute_boundary(&rects);
        assert_eq!(cluster.data().hull_x.len(), RECT_HULL_POINTS);
    }

    #[test]
    fn cluster_enum_compute_boundary_dispatch_convex() {
        let rects = vec![make_rect(0.0, 10.0, 0.0, 10.0)];
        let mut cc = ConvexCluster::new();
        cc.data.add_child_node(0);
        let mut cluster = Cluster::Convex(cc);
        cluster.compute_boundary(&rects);
        assert_eq!(cluster.data().hull_x.len(), CORNERS_PER_RECT);
    }

    #[test]
    fn cluster_enum_compute_boundary_dispatch_root() {
        let rects = vec![make_rect(0.0, 10.0, 0.0, 10.0)];
        let mut child = RectangularCluster::new();
        child.data.add_child_node(0);

        let mut root = RootCluster::new();
        root.data.add_child_cluster(Cluster::Rectangular(child));

        let mut cluster = Cluster::Root(root);
        cluster.compute_boundary(&rects);

        let child_hull_len = cluster.data().clusters[0].data().hull_x.len();
        assert_eq!(child_hull_len, RECT_HULL_POINTS);
    }

    // ===================================================================
    // Category 9: Edge cases
    // ===================================================================

    #[test]
    fn empty_cluster_no_nodes_compute_boundary() {
        let rects = test_rects();
        let mut rc = RectangularCluster::new();
        rc.compute_boundary(&rects);
        assert!(rc.data.hull_x.is_empty());
        assert!(rc.data.hull_y.is_empty());
    }

    #[test]
    fn empty_cluster_no_nodes_compute_bounding_rect() {
        let rects = test_rects();
        let mut rc = RectangularCluster::new();
        rc.compute_bounding_rect(&rects);
        assert!(!rc.data.bounds.is_valid());
    }

    #[test]
    fn cluster_with_only_sub_clusters() {
        let rects = test_rects();
        let mut child = RectangularCluster::new();
        child.data.add_child_node(0);
        child.compute_bounding_rect(&rects);

        let mut parent = RectangularCluster::new();
        // No direct nodes, only a child cluster.
        parent.data.add_child_cluster(Cluster::Rectangular(child));
        parent.compute_bounding_rect(&rects);

        assert!(parent.data.bounds.is_valid());
        assert!((parent.data.bounds.get_min_x() - 0.0).abs() < EPSILON);
        assert!((parent.data.bounds.get_max_x() - 10.0).abs() < EPSILON);
    }

    #[test]
    fn update_bounds_sets_min_max() {
        let rects = vec![make_rect(0.0, 10.0, 0.0, 10.0)];
        let mut rc = RectangularCluster::new();
        rc.data.add_child_node(0);
        rc.compute_bounding_rect(&rects);

        let mut cluster = Cluster::Rectangular(rc);
        let vars = cluster.create_vars(Dim::Horizontal, &rects, 0);
        assert_eq!(vars.len(), BOUNDARY_VARS_PER_CLUSTER);

        // Simulate solver output: positions indexed by var id.
        let var_positions = vec![2.0, 12.0];
        cluster.update_bounds(Dim::Horizontal, &var_positions);

        assert!((cluster.data().bounds.get_min_d(Dim::Horizontal) - 2.0).abs() < EPSILON);
        assert!((cluster.data().bounds.get_max_d(Dim::Horizontal) - 12.0).abs() < EPSILON);
    }

    #[test]
    fn update_bounds_vertical() {
        let rects = vec![make_rect(0.0, 10.0, 0.0, 20.0)];
        let mut rc = RectangularCluster::new();
        rc.data.add_child_node(0);
        rc.compute_bounding_rect(&rects);

        let mut cluster = Cluster::Rectangular(rc);
        let _vars = cluster.create_vars(Dim::Vertical, &rects, 0);

        let var_positions = vec![5.0, 25.0];
        cluster.update_bounds(Dim::Vertical, &var_positions);

        assert!((cluster.data().bounds.get_min_d(Dim::Vertical) - 5.0).abs() < EPSILON);
        assert!((cluster.data().bounds.get_max_d(Dim::Vertical) - 25.0).abs() < EPSILON);
    }

    #[test]
    fn convex_hull_corners_are_valid() {
        let rects = vec![make_rect(0.0, 10.0, 0.0, 10.0)];
        let mut cc = ConvexCluster::new();
        cc.data.add_child_node(0);
        cc.compute_boundary(&rects);

        // All corner indices must be in [0, CORNERS_PER_RECT).
        let max_corner = (CORNERS_PER_RECT - 1) as u8;
        for &c in &cc.hull_corners {
            assert!(c <= max_corner, "corner index {} out of range", c);
        }
    }

    #[test]
    fn rectangular_compute_boundary_single_node() {
        let rects = vec![make_rect(5.0, 15.0, 10.0, 25.0)];
        let mut rc = RectangularCluster::new();
        rc.data.add_child_node(0);
        rc.compute_boundary(&rects);

        assert_eq!(rc.data.hull_x.len(), RECT_HULL_POINTS);
        let min_x = rc.data.hull_x.iter().cloned().fold(f64::MAX, f64::min);
        let max_x = rc.data.hull_x.iter().cloned().fold(f64::MIN, f64::max);
        assert!((min_x - 5.0).abs() < EPSILON);
        assert!((max_x - 15.0).abs() < EPSILON);
    }
}
