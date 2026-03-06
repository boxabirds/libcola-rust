//! WASM bindings for libcola.
//!
//! Exposes the constraint-based graph layout engine to JavaScript via wasm-bindgen.

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "wasm")]
use crate::cola::layout::{
    ConstrainedFDLayout as RustLayout, DesiredPosition as RustDesiredPosition,
    Lock as RustLock,
};
#[cfg(feature = "wasm")]
use crate::vpsc::Rectangle as RustRectangle;

// ---------------------------------------------------------------------------
// LayoutNode - a rectangle for WASM
// ---------------------------------------------------------------------------

/// A node (rectangle) in the graph layout.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct LayoutNode {
    x: f64,
    y: f64,
    width: f64,
    height: f64,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl LayoutNode {
    #[wasm_bindgen(constructor)]
    pub fn new(x: f64, y: f64, width: f64, height: f64) -> Self {
        Self { x, y, width, height }
    }

    #[wasm_bindgen(getter)]
    pub fn x(&self) -> f64 {
        self.x
    }

    #[wasm_bindgen(getter)]
    pub fn y(&self) -> f64 {
        self.y
    }

    #[wasm_bindgen(getter)]
    pub fn width(&self) -> f64 {
        self.width
    }

    #[wasm_bindgen(getter)]
    pub fn height(&self) -> f64 {
        self.height
    }
}

// ---------------------------------------------------------------------------
// LayoutEdge - an edge for WASM
// ---------------------------------------------------------------------------

/// An edge between two nodes.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct LayoutEdge {
    source: usize,
    target: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl LayoutEdge {
    #[wasm_bindgen(constructor)]
    pub fn new(source: usize, target: usize) -> Self {
        Self { source, target }
    }

    #[wasm_bindgen(getter)]
    pub fn source(&self) -> usize {
        self.source
    }

    #[wasm_bindgen(getter)]
    pub fn target(&self) -> usize {
        self.target
    }
}

// ---------------------------------------------------------------------------
// LayoutResult - positions after layout
// ---------------------------------------------------------------------------

/// Result of a layout computation: node positions.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct LayoutResult {
    /// Flat array: [x0, y0, w0, h0, x1, y1, w1, h1, ...]
    data: Vec<f64>,
    node_count: usize,
    stress: f64,
}

/// Number of f64 values per node in the result data.
#[cfg(feature = "wasm")]
const FIELDS_PER_NODE: usize = 4;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl LayoutResult {
    /// Number of nodes.
    #[wasm_bindgen(getter, js_name = "nodeCount")]
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    /// Final stress value.
    #[wasm_bindgen(getter)]
    pub fn stress(&self) -> f64 {
        self.stress
    }

    /// Get x position of node i.
    #[wasm_bindgen(js_name = "getX")]
    pub fn get_x(&self, i: usize) -> f64 {
        self.data[i * FIELDS_PER_NODE]
    }

    /// Get y position of node i.
    #[wasm_bindgen(js_name = "getY")]
    pub fn get_y(&self, i: usize) -> f64 {
        self.data[i * FIELDS_PER_NODE + 1]
    }

    /// Get width of node i.
    #[wasm_bindgen(js_name = "getWidth")]
    pub fn get_width(&self, i: usize) -> f64 {
        self.data[i * FIELDS_PER_NODE + 2]
    }

    /// Get height of node i.
    #[wasm_bindgen(js_name = "getHeight")]
    pub fn get_height(&self, i: usize) -> f64 {
        self.data[i * FIELDS_PER_NODE + 3]
    }

    /// Get all positions as a flat Float64Array [x0,y0,w0,h0,x1,y1,w1,h1,...].
    #[wasm_bindgen(js_name = "getAllPositions")]
    pub fn get_all_positions(&self) -> Vec<f64> {
        self.data.clone()
    }
}

// ---------------------------------------------------------------------------
// SeparationConstraint - for WASM
// ---------------------------------------------------------------------------

/// A separation constraint: left + gap <= right.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct SeparationConstraint {
    left: usize,
    right: usize,
    gap: f64,
    is_equality: bool,
    dim: u8, // 0 = x, 1 = y
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl SeparationConstraint {
    /// Create a separation constraint: node[left] + gap <= node[right].
    /// dim: 0 = horizontal (x), 1 = vertical (y).
    #[wasm_bindgen(constructor)]
    pub fn new(dim: u8, left: usize, right: usize, gap: f64, is_equality: bool) -> Self {
        Self {
            left,
            right,
            gap,
            is_equality,
            dim,
        }
    }
}

// ---------------------------------------------------------------------------
// AlignmentConstraint - for WASM
// ---------------------------------------------------------------------------

/// An alignment constraint: aligns multiple nodes on a guide line.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct AlignmentConstraint {
    dim: u8, // 0 = x, 1 = y
    node_indices: Vec<usize>,
    position: f64,
    is_fixed: bool,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl AlignmentConstraint {
    /// Create an alignment constraint.
    /// dim: 0 = horizontal, 1 = vertical.
    /// position: the guide line position.
    /// is_fixed: if true, the guide position is fixed.
    #[wasm_bindgen(constructor)]
    pub fn new(dim: u8, position: f64, is_fixed: bool) -> Self {
        Self {
            dim,
            node_indices: Vec::new(),
            position,
            is_fixed,
        }
    }

    /// Add a node to this alignment.
    #[wasm_bindgen(js_name = "addNode")]
    pub fn add_node(&mut self, index: usize) {
        self.node_indices.push(index);
    }
}

// ---------------------------------------------------------------------------
// DistributionConstraint - for WASM
// ---------------------------------------------------------------------------

/// A distribution constraint: enforces equal spacing between pairs.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct DistributionConstraint {
    dim: u8,
    pairs: Vec<usize>, // flat: [left0, right0, left1, right1, ...]
    separation: f64,
}

/// Number of indices per pair in distribution constraint pairs array.
#[cfg(feature = "wasm")]
const INDICES_PER_PAIR: usize = 2;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl DistributionConstraint {
    #[wasm_bindgen(constructor)]
    pub fn new(dim: u8, separation: f64) -> Self {
        Self {
            dim,
            pairs: Vec::new(),
            separation,
        }
    }

    /// Add a pair of nodes to distribute.
    #[wasm_bindgen(js_name = "addPair")]
    pub fn add_pair(&mut self, left: usize, right: usize) {
        self.pairs.push(left);
        self.pairs.push(right);
    }
}

// ---------------------------------------------------------------------------
// ColaLayout - main WASM entry point
// ---------------------------------------------------------------------------

/// The main constraint-based layout engine.
///
/// Usage from JavaScript:
/// ```js
/// const layout = new ColaLayout();
/// layout.addNode(0, 0, 40, 30);
/// layout.addNode(100, 0, 40, 30);
/// layout.addEdge(0, 1);
/// layout.setIdealEdgeLength(100);
/// const result = layout.run();
/// ```
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct ColaLayout {
    nodes: Vec<LayoutNode>,
    edges: Vec<(usize, usize)>,
    edge_lengths: Vec<f64>,
    ideal_edge_length: f64,
    avoid_overlaps: bool,
    convergence_tolerance: f64,
    max_iterations: usize,
    use_runge_kutta: bool,
    use_neighbour_stress: bool,
    skip_distant_non_neighbours: bool,
    separation_constraints: Vec<SeparationConstraint>,
    alignment_constraints: Vec<AlignmentConstraint>,
    distribution_constraints: Vec<DistributionConstraint>,
    locks: Vec<RustLock>,
    desired_positions: Vec<RustDesiredPosition>,
}

/// Default ideal edge length for layout.
#[cfg(feature = "wasm")]
const WASM_DEFAULT_IDEAL_EDGE_LENGTH: f64 = 100.0;

/// Default convergence tolerance for WASM layout.
#[cfg(feature = "wasm")]
const WASM_DEFAULT_CONVERGENCE_TOLERANCE: f64 = 1e-4;

/// Default max iterations for WASM layout.
#[cfg(feature = "wasm")]
const WASM_DEFAULT_MAX_ITERATIONS: usize = 100;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl ColaLayout {
    /// Create a new layout engine.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            edge_lengths: Vec::new(),
            ideal_edge_length: WASM_DEFAULT_IDEAL_EDGE_LENGTH,
            avoid_overlaps: false,
            convergence_tolerance: WASM_DEFAULT_CONVERGENCE_TOLERANCE,
            max_iterations: WASM_DEFAULT_MAX_ITERATIONS,
            use_runge_kutta: true,
            use_neighbour_stress: false,
            skip_distant_non_neighbours: true,
            separation_constraints: Vec::new(),
            alignment_constraints: Vec::new(),
            distribution_constraints: Vec::new(),
            locks: Vec::new(),
            desired_positions: Vec::new(),
        }
    }

    /// Add a node at position (x, y) with given width and height.
    /// Returns the node index.
    #[wasm_bindgen(js_name = "addNode")]
    pub fn add_node(&mut self, x: f64, y: f64, width: f64, height: f64) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(LayoutNode::new(x, y, width, height));
        idx
    }

    /// Add an edge between two nodes.
    #[wasm_bindgen(js_name = "addEdge")]
    pub fn add_edge(&mut self, source: usize, target: usize) {
        self.edges.push((source, target));
    }

    /// Add an edge with a custom ideal length.
    #[wasm_bindgen(js_name = "addEdgeWithLength")]
    pub fn add_edge_with_length(&mut self, source: usize, target: usize, length: f64) {
        self.edges.push((source, target));
        // Pad edge_lengths to match edges if needed.
        while self.edge_lengths.len() < self.edges.len() - 1 {
            self.edge_lengths.push(1.0);
        }
        self.edge_lengths.push(length / self.ideal_edge_length);
    }

    /// Set the ideal edge length (default: 100).
    #[wasm_bindgen(js_name = "setIdealEdgeLength")]
    pub fn set_ideal_edge_length(&mut self, length: f64) {
        self.ideal_edge_length = length;
    }

    /// Enable/disable overlap avoidance.
    #[wasm_bindgen(js_name = "setAvoidOverlaps")]
    pub fn set_avoid_overlaps(&mut self, avoid: bool) {
        self.avoid_overlaps = avoid;
    }

    /// Set convergence tolerance (default: 1e-4).
    #[wasm_bindgen(js_name = "setConvergenceTolerance")]
    pub fn set_convergence_tolerance(&mut self, tolerance: f64) {
        self.convergence_tolerance = tolerance;
    }

    /// Set maximum iterations (default: 100).
    #[wasm_bindgen(js_name = "setMaxIterations")]
    pub fn set_max_iterations(&mut self, max: usize) {
        self.max_iterations = max;
    }

    /// Enable/disable Runge-Kutta integration (default: true).
    #[wasm_bindgen(js_name = "setRungeKutta")]
    pub fn set_runge_kutta(&mut self, enabled: bool) {
        self.use_runge_kutta = enabled;
    }

    /// Enable/disable neighbour-only stress (default: false).
    #[wasm_bindgen(js_name = "setNeighbourStress")]
    pub fn set_neighbour_stress(&mut self, enabled: bool) {
        self.use_neighbour_stress = enabled;
    }

    /// Skip forces for non-neighbours already beyond ideal distance.
    /// Matches C++ `if(l>d && p>1) continue;`. Default: true (C++ compat).
    /// Set false for better quality on sparse graphs (cycles, paths).
    #[wasm_bindgen(js_name = "setSkipDistantNonNeighbours")]
    pub fn set_skip_distant_non_neighbours(&mut self, enabled: bool) {
        self.skip_distant_non_neighbours = enabled;
    }

    /// Add a separation constraint.
    #[wasm_bindgen(js_name = "addSeparationConstraint")]
    pub fn add_separation_constraint(&mut self, constraint: SeparationConstraint) {
        self.separation_constraints.push(constraint);
    }

    /// Add an alignment constraint.
    #[wasm_bindgen(js_name = "addAlignmentConstraint")]
    pub fn add_alignment_constraint(&mut self, constraint: AlignmentConstraint) {
        self.alignment_constraints.push(constraint);
    }

    /// Add a distribution constraint.
    #[wasm_bindgen(js_name = "addDistributionConstraint")]
    pub fn add_distribution_constraint(&mut self, constraint: DistributionConstraint) {
        self.distribution_constraints.push(constraint);
    }

    /// Lock a node at a specific position during layout.
    #[wasm_bindgen(js_name = "lockNode")]
    pub fn lock_node(&mut self, id: usize, x: f64, y: f64) {
        self.locks.push(RustLock::new(id, x, y));
    }

    /// Add a desired position (attraction point) for a node.
    #[wasm_bindgen(js_name = "addDesiredPosition")]
    pub fn add_desired_position(&mut self, id: usize, x: f64, y: f64, weight: f64) {
        self.desired_positions.push(RustDesiredPosition {
            id,
            x,
            y,
            z: 0.0,
            weight,
        });
    }

    /// Clear all nodes, edges, and constraints.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.edges.clear();
        self.edge_lengths.clear();
        self.separation_constraints.clear();
        self.alignment_constraints.clear();
        self.distribution_constraints.clear();
        self.locks.clear();
        self.desired_positions.clear();
    }

    /// Run the layout algorithm and return results.
    pub fn run(&self) -> LayoutResult {
        use crate::cola::compound_constraints::{
            CompoundConstraint, SeparationConstraint as CCSep, AlignmentConstraint as CCAlign,
            DistributionConstraint as CCDist,
        };
        use crate::vpsc::Dim;

        let n = self.nodes.len();
        if n == 0 {
            return LayoutResult {
                data: Vec::new(),
                node_count: 0,
                stress: 0.0,
            };
        }

        // Build rectangles from nodes.
        let rects: Vec<RustRectangle> = self
            .nodes
            .iter()
            .map(|node| {
                RustRectangle::new(
                    node.x - node.width / 2.0,
                    node.x + node.width / 2.0,
                    node.y - node.height / 2.0,
                    node.y + node.height / 2.0,
                )
            })
            .collect();

        let el = if self.edge_lengths.is_empty() {
            None
        } else {
            Some(self.edge_lengths.as_slice())
        };

        let mut layout = RustLayout::new(rects, &self.edges, self.ideal_edge_length, el);
        layout.set_convergence(self.convergence_tolerance, self.max_iterations);
        layout.set_avoid_node_overlaps(self.avoid_overlaps);
        layout.set_runge_kutta(self.use_runge_kutta);
        layout.set_use_neighbour_stress(self.use_neighbour_stress);
        layout.set_skip_distant_non_neighbours(self.skip_distant_non_neighbours);

        // Convert separation constraints to compound constraints.
        let mut ccs: Vec<CompoundConstraint> = Vec::new();

        for sc in &self.separation_constraints {
            let dim = if sc.dim == 0 {
                Dim::Horizontal
            } else {
                Dim::Vertical
            };
            ccs.push(CompoundConstraint::Separation(CCSep::new(
                dim, sc.left, sc.right, sc.gap, sc.is_equality,
            )));
        }

        for ac in &self.alignment_constraints {
            let dim = if ac.dim == 0 {
                Dim::Horizontal
            } else {
                Dim::Vertical
            };
            let mut align = CCAlign::new(dim, ac.position);
            if ac.is_fixed {
                align.fix_pos(ac.position);
            }
            for &idx in &ac.node_indices {
                align.add_shape(idx, 0.0);
            }
            ccs.push(CompoundConstraint::Alignment(align));
        }

        for dc in &self.distribution_constraints {
            let dim = if dc.dim == 0 {
                Dim::Horizontal
            } else {
                Dim::Vertical
            };
            let mut dist = CCDist::new(dim);
            dist.set_separation(dc.separation);
            for chunk in dc.pairs.chunks(INDICES_PER_PAIR) {
                if chunk.len() == INDICES_PER_PAIR {
                    dist.add_alignment_pair(chunk[0], chunk[1]);
                }
            }
            ccs.push(CompoundConstraint::Distribution(dist));
        }

        if !ccs.is_empty() {
            layout.set_constraints(ccs);
        }

        if !self.desired_positions.is_empty() {
            layout.set_desired_positions(self.desired_positions.clone());
        }

        // Run layout.
        layout.run();

        // Build result.
        let stress = layout.compute_stress(None);
        let bb = layout.bounding_boxes();
        let mut data = Vec::with_capacity(n * FIELDS_PER_NODE);
        for i in 0..n {
            data.push(bb[i].centre_x());
            data.push(bb[i].centre_y());
            data.push(bb[i].width());
            data.push(bb[i].height());
        }

        LayoutResult {
            data,
            node_count: n,
            stress,
        }
    }

    /// Run a single iteration and return results.
    /// Useful for animation.
    #[wasm_bindgen(js_name = "runOnce")]
    pub fn run_once(&self) -> LayoutResult {
        let n = self.nodes.len();
        if n == 0 {
            return LayoutResult {
                data: Vec::new(),
                node_count: 0,
                stress: 0.0,
            };
        }

        let rects: Vec<RustRectangle> = self
            .nodes
            .iter()
            .map(|node| {
                RustRectangle::new(
                    node.x - node.width / 2.0,
                    node.x + node.width / 2.0,
                    node.y - node.height / 2.0,
                    node.y + node.height / 2.0,
                )
            })
            .collect();

        let el = if self.edge_lengths.is_empty() {
            None
        } else {
            Some(self.edge_lengths.as_slice())
        };

        let mut layout = RustLayout::new(rects, &self.edges, self.ideal_edge_length, el);
        layout.set_avoid_node_overlaps(self.avoid_overlaps);
        layout.set_runge_kutta(self.use_runge_kutta);
        layout.run_once(true, true);

        let stress = layout.compute_stress(None);
        let bb = layout.bounding_boxes();
        let mut data = Vec::with_capacity(n * FIELDS_PER_NODE);
        for i in 0..n {
            data.push(bb[i].centre_x());
            data.push(bb[i].centre_y());
            data.push(bb[i].width());
            data.push(bb[i].height());
        }

        LayoutResult {
            data,
            node_count: n,
            stress,
        }
    }
}

// ---------------------------------------------------------------------------
// LayoutResult3D - positions after 3D layout
// ---------------------------------------------------------------------------

/// Result of a 3D layout: [x, y, z, w, h, d] per node.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct LayoutResult3D {
    data: Vec<f64>,
    node_count: usize,
    stress: f64,
}

#[cfg(feature = "wasm")]
const FIELDS_PER_NODE_3D: usize = 6;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl LayoutResult3D {
    #[wasm_bindgen(getter, js_name = "nodeCount")]
    pub fn node_count(&self) -> usize { self.node_count }

    #[wasm_bindgen(getter)]
    pub fn stress(&self) -> f64 { self.stress }

    #[wasm_bindgen(js_name = "getX")]
    pub fn get_x(&self, i: usize) -> f64 { self.data[i * FIELDS_PER_NODE_3D] }

    #[wasm_bindgen(js_name = "getY")]
    pub fn get_y(&self, i: usize) -> f64 { self.data[i * FIELDS_PER_NODE_3D + 1] }

    #[wasm_bindgen(js_name = "getZ")]
    pub fn get_z(&self, i: usize) -> f64 { self.data[i * FIELDS_PER_NODE_3D + 2] }

    #[wasm_bindgen(js_name = "getWidth")]
    pub fn get_width(&self, i: usize) -> f64 { self.data[i * FIELDS_PER_NODE_3D + 3] }

    #[wasm_bindgen(js_name = "getHeight")]
    pub fn get_height(&self, i: usize) -> f64 { self.data[i * FIELDS_PER_NODE_3D + 4] }

    #[wasm_bindgen(js_name = "getDepth")]
    pub fn get_depth(&self, i: usize) -> f64 { self.data[i * FIELDS_PER_NODE_3D + 5] }

    #[wasm_bindgen(js_name = "getAllPositions")]
    pub fn get_all_positions(&self) -> Vec<f64> { self.data.clone() }
}

// ---------------------------------------------------------------------------
// ColaLayout3D - 3D layout engine for WASM
// ---------------------------------------------------------------------------

#[cfg(feature = "wasm")]
struct LayoutNode3D {
    x: f64,
    y: f64,
    z: f64,
    width: f64,
    height: f64,
    depth: f64,
}

/// 3D constraint-based graph layout engine.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct ColaLayout3D {
    nodes: Vec<LayoutNode3D>,
    edges: Vec<(usize, usize)>,
    edge_lengths: Vec<f64>,
    ideal_edge_length: f64,
    convergence_tolerance: f64,
    max_iterations: usize,
    use_runge_kutta: bool,
    skip_distant_non_neighbours: bool,
    /// Persistent layout for stateful runOnce.
    inner_layout: Option<RustLayout>,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl ColaLayout3D {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            edge_lengths: Vec::new(),
            ideal_edge_length: WASM_DEFAULT_IDEAL_EDGE_LENGTH,
            convergence_tolerance: WASM_DEFAULT_CONVERGENCE_TOLERANCE,
            max_iterations: WASM_DEFAULT_MAX_ITERATIONS,
            use_runge_kutta: true,
            skip_distant_non_neighbours: true,
            inner_layout: None,
        }
    }

    #[wasm_bindgen(js_name = "addNode")]
    pub fn add_node(&mut self, x: f64, y: f64, z: f64, w: f64, h: f64, d: f64) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(LayoutNode3D { x, y, z, width: w, height: h, depth: d });
        self.inner_layout = None;
        idx
    }

    #[wasm_bindgen(js_name = "addEdge")]
    pub fn add_edge(&mut self, source: usize, target: usize) {
        self.edges.push((source, target));
        self.inner_layout = None;
    }

    #[wasm_bindgen(js_name = "setIdealEdgeLength")]
    pub fn set_ideal_edge_length(&mut self, length: f64) {
        self.ideal_edge_length = length;
        self.inner_layout = None;
    }

    #[wasm_bindgen(js_name = "setConvergenceTolerance")]
    pub fn set_convergence_tolerance(&mut self, tolerance: f64) {
        self.convergence_tolerance = tolerance;
    }

    #[wasm_bindgen(js_name = "setMaxIterations")]
    pub fn set_max_iterations(&mut self, max: usize) {
        self.max_iterations = max;
    }

    #[wasm_bindgen(js_name = "setRungeKutta")]
    pub fn set_runge_kutta(&mut self, enabled: bool) {
        self.use_runge_kutta = enabled;
    }

    #[wasm_bindgen(js_name = "setSkipDistantNonNeighbours")]
    pub fn set_skip_distant_non_neighbours(&mut self, enabled: bool) {
        self.skip_distant_non_neighbours = enabled;
    }

    /// Reset for new animation (clears internal layout state).
    pub fn reset(&mut self) {
        self.inner_layout = None;
    }

    fn ensure_layout(&mut self) {
        if self.inner_layout.is_some() {
            return;
        }
        let n = self.nodes.len();
        let rects: Vec<RustRectangle> = self.nodes.iter().map(|nd| {
            RustRectangle::new(
                nd.x - nd.width / 2.0,
                nd.x + nd.width / 2.0,
                nd.y - nd.height / 2.0,
                nd.y + nd.height / 2.0,
            )
        }).collect();
        let z_positions: Vec<f64> = self.nodes.iter().map(|nd| nd.z).collect();
        let el = if self.edge_lengths.is_empty() {
            None
        } else {
            Some(self.edge_lengths.as_slice())
        };

        let mut layout = RustLayout::new_3d(rects, &z_positions, &self.edges, self.ideal_edge_length, el);
        layout.set_convergence(self.convergence_tolerance, self.max_iterations);
        layout.set_runge_kutta(self.use_runge_kutta);
        layout.set_skip_distant_non_neighbours(self.skip_distant_non_neighbours);
        self.inner_layout = Some(layout);
    }

    fn build_result(&self, include_stress: bool) -> LayoutResult3D {
        let layout = self.inner_layout.as_ref().unwrap();
        let n = self.nodes.len();
        let bb = layout.bounding_boxes();
        let mut data = Vec::with_capacity(n * FIELDS_PER_NODE_3D);
        for i in 0..n {
            data.push(layout.x()[i]);
            data.push(layout.y()[i]);
            data.push(layout.z()[i]);
            data.push(bb[i].width());
            data.push(bb[i].height());
            data.push(self.nodes[i].depth);
        }
        let stress = if include_stress {
            layout.compute_stress(None)
        } else {
            0.0
        };
        LayoutResult3D {
            data,
            node_count: n,
            stress,
        }
    }

    /// Run full layout to convergence.
    pub fn run(&mut self) -> LayoutResult3D {
        self.ensure_layout();
        self.inner_layout.as_mut().unwrap().run();
        self.build_result(true)
    }

    /// Run a single iteration (stateful — maintains layout between calls).
    #[wasm_bindgen(js_name = "runOnce")]
    pub fn run_once(&mut self) -> LayoutResult3D {
        self.ensure_layout();
        self.inner_layout.as_mut().unwrap().run_once(true, true);
        self.build_result(false)
    }

    /// Get flat position array [x0,y0,z0, x1,y1,z1, ...] without allocating a full result.
    #[wasm_bindgen(js_name = "getPositions")]
    pub fn get_positions(&self) -> Vec<f64> {
        let layout = self.inner_layout.as_ref().unwrap();
        let n = self.nodes.len();
        let mut pos = Vec::with_capacity(n * 3);
        let x = layout.x();
        let y = layout.y();
        let z = layout.z();
        for i in 0..n {
            pos.push(x[i]);
            pos.push(y[i]);
            pos.push(z[i]);
        }
        pos
    }

    /// Run one iteration without building a result object.
    #[wasm_bindgen(js_name = "stepOnce")]
    pub fn step_once(&mut self) {
        self.ensure_layout();
        self.inner_layout.as_mut().unwrap().run_once(true, true);
    }
}
