//! Connected component decomposition and separation.
//!
//! Finds connected components in a graph and can separate them
//! so their bounding boxes don't overlap.
//!
//! C++ ref: libcola/connected_components.h, libcola/connected_components.cpp

use crate::cola::commondefs::Edge;
use crate::vpsc::{remove_overlaps, Rectangle};

/// A connected component of a graph, holding the subset of nodes,
/// their rectangles (cloned from input), and edges remapped to local indices.
///
/// C++ ref: cola::Component
pub struct Component {
    /// Original (global) node indices belonging to this component.
    pub node_ids: Vec<usize>,
    /// Rectangles for the nodes in this component (same order as `node_ids`).
    pub rects: Vec<Rectangle>,
    /// Edges using local indices within this component.
    pub edges: Vec<Edge>,
}

impl Component {
    /// Translate all rectangles by `(dx, dy)`.
    ///
    /// C++ ref: Component::moveRectangles
    pub fn move_rectangles(&mut self, dx: f64, dy: f64) {
        for rect in &mut self.rects {
            let new_cx = rect.centre_x() + dx;
            let new_cy = rect.centre_y() + dy;
            rect.move_centre(new_cx, new_cy);
        }
    }

    /// Compute the axis-aligned bounding box of all rectangles in this component.
    ///
    /// Returns an invalid rectangle if the component has no rectangles.
    ///
    /// C++ ref: Component::getBoundingBox
    pub fn bounding_box(&self) -> Rectangle {
        let mut bb = Rectangle::invalid();
        for rect in &self.rects {
            bb = bb.union_with(rect);
        }
        bb
    }
}

/// Internal DFS node with an adjacency list.
struct Node {
    /// Indices of neighbour nodes.
    adj: Vec<usize>,
}

/// Find connected components in a graph defined by rectangles and edges.
///
/// Each returned `Component` owns cloned rectangles and edges remapped to
/// local (component-internal) indices. The `node_ids` field preserves the
/// original global indices.
///
/// C++ ref: connectedComponents
pub fn connected_components(rects: &[Rectangle], edges: &[Edge]) -> Vec<Component> {
    let n = rects.len();

    // Build adjacency lists.
    let mut nodes: Vec<Node> = (0..n).map(|_| Node { adj: Vec::new() }).collect();
    for &(u, v) in edges {
        // Skip self-loops and out-of-range edges.
        if u == v || u >= n || v >= n {
            continue;
        }
        nodes[u].adj.push(v);
        nodes[v].adj.push(u);
    }

    // Track which nodes have been assigned to a component.
    let mut visited = vec![false; n];
    let mut components: Vec<Component> = Vec::new();

    // Remaining unvisited nodes, processed in order.
    // Using an explicit remaining list matches the C++ algorithm.
    let mut remaining: Vec<usize> = (0..n).collect();

    while let Some(&start) = remaining.first() {
        // DFS from `start` to collect one component's node set.
        let mut component_nodes: Vec<usize> = Vec::new();
        let mut stack = vec![start];
        visited[start] = true;

        while let Some(node) = stack.pop() {
            component_nodes.push(node);
            for &neighbour in &nodes[node].adj {
                if !visited[neighbour] {
                    visited[neighbour] = true;
                    stack.push(neighbour);
                }
            }
        }

        // Build mapping: global node id -> local index within this component.
        let mut global_to_local = vec![0usize; n];
        for (local_idx, &global_id) in component_nodes.iter().enumerate() {
            global_to_local[global_id] = local_idx;
        }

        // Clone rectangles for this component.
        let comp_rects: Vec<Rectangle> = component_nodes
            .iter()
            .map(|&id| rects[id].clone())
            .collect();

        // Remap edges: only include edges where both endpoints are in this component.
        let mut comp_edges: Vec<Edge> = Vec::new();
        for &(u, v) in edges {
            if u == v || u >= n || v >= n {
                continue;
            }
            if visited_in_component(u, &component_nodes, &global_to_local)
                && visited_in_component(v, &component_nodes, &global_to_local)
            {
                comp_edges.push((global_to_local[u], global_to_local[v]));
            }
        }

        components.push(Component {
            node_ids: component_nodes,
            rects: comp_rects,
            edges: comp_edges,
        });

        // Remove visited nodes from remaining.
        remaining.retain(|&id| !visited[id]);
    }

    components
}

/// Check whether `node` belongs to the component described by `members` and
/// `global_to_local`. We verify by round-tripping: the local index must map
/// back to the same global id.
#[inline]
fn visited_in_component(
    node: usize,
    members: &[usize],
    global_to_local: &[usize],
) -> bool {
    let local = global_to_local[node];
    local < members.len() && members[local] == node
}

/// Separate components so their bounding boxes do not overlap.
///
/// Computes a bounding box for each component, uses VPSC rectangle overlap
/// removal to push bounding boxes apart, then translates each component's
/// rectangles by the resulting displacement.
///
/// C++ ref: separateComponents
pub fn separate_components(components: &mut [Component]) {
    if components.len() <= 1 {
        return;
    }

    // Collect bounding boxes and their original centres.
    let mut bboxes: Vec<Rectangle> = components.iter().map(|c| c.bounding_box()).collect();
    let old_centres: Vec<(f64, f64)> = bboxes
        .iter()
        .map(|bb: &Rectangle| (bb.centre_x(), bb.centre_y()))
        .collect();

    // Use VPSC overlap removal on the bounding boxes.
    remove_overlaps(&mut bboxes);

    // Compute displacement and move each component's rectangles.
    for (i, component) in components.iter_mut().enumerate() {
        let dx = bboxes[i].centre_x() - old_centres[i].0;
        let dy = bboxes[i].centre_y() - old_centres[i].1;
        if dx.abs() > f64::EPSILON || dy.abs() > f64::EPSILON {
            component.move_rectangles(dx, dy);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a unit rectangle at a given position.
    fn rect_at(x: f64, y: f64) -> Rectangle {
        Rectangle::new(x, x + 1.0, y, y + 1.0)
    }

    /// Helper: create a rectangle with explicit bounds.
    fn rect(min_x: f64, max_x: f64, min_y: f64, max_y: f64) -> Rectangle {
        Rectangle::new(min_x, max_x, min_y, max_y)
    }

    // ===================================================================
    // Category 1: Component::move_rectangles
    // ===================================================================

    #[test]
    fn move_rectangles_translates_all() {
        let mut comp = Component {
            node_ids: vec![0, 1],
            rects: vec![rect_at(0.0, 0.0), rect_at(5.0, 5.0)],
            edges: vec![],
        };
        let dx = 10.0;
        let dy = -3.0;
        let old_centres: Vec<(f64, f64)> = comp
            .rects
            .iter()
            .map(|r: &Rectangle| (r.centre_x(), r.centre_y()))
            .collect();

        comp.move_rectangles(dx, dy);

        for (i, r) in comp.rects.iter().enumerate() {
            let expected_x = old_centres[i].0 + dx;
            let expected_y = old_centres[i].1 + dy;
            assert!(
                (r.centre_x() - expected_x).abs() < 1e-9,
                "rect {} X centre mismatch",
                i
            );
            assert!(
                (r.centre_y() - expected_y).abs() < 1e-9,
                "rect {} Y centre mismatch",
                i
            );
        }
    }

    #[test]
    fn move_rectangles_zero_displacement_is_noop() {
        let mut comp = Component {
            node_ids: vec![0],
            rects: vec![rect_at(3.0, 7.0)],
            edges: vec![],
        };
        let cx = comp.rects[0].centre_x();
        let cy = comp.rects[0].centre_y();

        comp.move_rectangles(0.0, 0.0);

        assert!((comp.rects[0].centre_x() - cx).abs() < 1e-12);
        assert!((comp.rects[0].centre_y() - cy).abs() < 1e-12);
    }

    #[test]
    fn move_rectangles_preserves_dimensions() {
        let mut comp = Component {
            node_ids: vec![0],
            rects: vec![rect(0.0, 4.0, 0.0, 6.0)],
            edges: vec![],
        };
        let w = comp.rects[0].width();
        let h = comp.rects[0].height();

        comp.move_rectangles(100.0, 200.0);

        assert!((comp.rects[0].width() - w).abs() < 1e-12);
        assert!((comp.rects[0].height() - h).abs() < 1e-12);
    }

    // ===================================================================
    // Category 2: Component::bounding_box
    // ===================================================================

    #[test]
    fn bounding_box_single_rect() {
        let comp = Component {
            node_ids: vec![0],
            rects: vec![rect(2.0, 8.0, 3.0, 9.0)],
            edges: vec![],
        };
        let bb = comp.bounding_box();
        assert!((bb.get_min_x() - 2.0).abs() < 1e-9);
        assert!((bb.get_max_x() - 8.0).abs() < 1e-9);
        assert!((bb.get_min_y() - 3.0).abs() < 1e-9);
        assert!((bb.get_max_y() - 9.0).abs() < 1e-9);
    }

    #[test]
    fn bounding_box_multiple_rects() {
        let comp = Component {
            node_ids: vec![0, 1, 2],
            rects: vec![
                rect(0.0, 2.0, 0.0, 2.0),
                rect(5.0, 7.0, 5.0, 7.0),
                rect(-1.0, 1.0, 10.0, 12.0),
            ],
            edges: vec![],
        };
        let bb = comp.bounding_box();
        assert!((bb.get_min_x() - (-1.0)).abs() < 1e-9);
        assert!((bb.get_max_x() - 7.0).abs() < 1e-9);
        assert!((bb.get_min_y() - 0.0).abs() < 1e-9);
        assert!((bb.get_max_y() - 12.0).abs() < 1e-9);
    }

    #[test]
    fn bounding_box_empty_component_returns_invalid() {
        let comp = Component {
            node_ids: vec![],
            rects: vec![],
            edges: vec![],
        };
        let bb = comp.bounding_box();
        assert!(!bb.is_valid());
    }

    // ===================================================================
    // Category 3: connected_components - single component (connected graph)
    // ===================================================================

    #[test]
    fn single_connected_component() {
        // Triangle: 0-1, 1-2, 0-2
        let rects = vec![rect_at(0.0, 0.0), rect_at(2.0, 0.0), rect_at(1.0, 2.0)];
        let edges: Vec<Edge> = vec![(0, 1), (1, 2), (0, 2)];

        let comps = connected_components(&rects, &edges);
        assert_eq!(comps.len(), 1);
        assert_eq!(comps[0].node_ids.len(), 3);
        assert_eq!(comps[0].rects.len(), 3);
        assert_eq!(comps[0].edges.len(), 3);
    }

    #[test]
    fn single_component_chain() {
        // Chain: 0-1-2-3
        let rects = vec![
            rect_at(0.0, 0.0),
            rect_at(2.0, 0.0),
            rect_at(4.0, 0.0),
            rect_at(6.0, 0.0),
        ];
        let edges: Vec<Edge> = vec![(0, 1), (1, 2), (2, 3)];

        let comps = connected_components(&rects, &edges);
        assert_eq!(comps.len(), 1);
        assert_eq!(comps[0].node_ids.len(), 4);
        assert_eq!(comps[0].edges.len(), 3);
    }

    // ===================================================================
    // Category 4: connected_components - two components
    // ===================================================================

    #[test]
    fn two_disconnected_components() {
        // Component A: 0-1, Component B: 2-3
        let rects = vec![
            rect_at(0.0, 0.0),
            rect_at(2.0, 0.0),
            rect_at(10.0, 0.0),
            rect_at(12.0, 0.0),
        ];
        let edges: Vec<Edge> = vec![(0, 1), (2, 3)];

        let comps = connected_components(&rects, &edges);
        assert_eq!(comps.len(), 2);

        // Each component should have 2 nodes.
        let mut sizes: Vec<usize> = comps.iter().map(|c| c.node_ids.len()).collect();
        sizes.sort();
        assert_eq!(sizes, vec![2, 2]);

        // Each component should have 1 edge.
        for c in &comps {
            assert_eq!(c.edges.len(), 1);
        }
    }

    #[test]
    fn three_components_various_sizes() {
        // 0-1-2, 3 alone, 4-5
        let rects: Vec<Rectangle> = (0..6).map(|i| rect_at(i as f64 * 3.0, 0.0)).collect();
        let edges: Vec<Edge> = vec![(0, 1), (1, 2), (4, 5)];

        let comps = connected_components(&rects, &edges);
        assert_eq!(comps.len(), 3);

        let mut sizes: Vec<usize> = comps.iter().map(|c| c.node_ids.len()).collect();
        sizes.sort();
        assert_eq!(sizes, vec![1, 2, 3]);
    }

    // ===================================================================
    // Category 5: connected_components - all isolated nodes
    // ===================================================================

    #[test]
    fn all_isolated_nodes() {
        let rects = vec![rect_at(0.0, 0.0), rect_at(5.0, 0.0), rect_at(10.0, 0.0)];
        let edges: Vec<Edge> = vec![];

        let comps = connected_components(&rects, &edges);
        assert_eq!(comps.len(), 3);

        for c in &comps {
            assert_eq!(c.node_ids.len(), 1);
            assert_eq!(c.rects.len(), 1);
            assert!(c.edges.is_empty());
        }
    }

    // ===================================================================
    // Category 6: connected_components - empty graph
    // ===================================================================

    #[test]
    fn empty_graph() {
        let rects: Vec<Rectangle> = vec![];
        let edges: Vec<Edge> = vec![];

        let comps = connected_components(&rects, &edges);
        assert!(comps.is_empty());
    }

    // ===================================================================
    // Category 7: Edge remapping - verify edges use local indices
    // ===================================================================

    #[test]
    fn edge_remapping_to_local_indices() {
        // Nodes 0, 1, 2, 3. Component A: {0, 1}, Component B: {2, 3}
        let rects = vec![
            rect_at(0.0, 0.0),
            rect_at(2.0, 0.0),
            rect_at(10.0, 0.0),
            rect_at(12.0, 0.0),
        ];
        let edges: Vec<Edge> = vec![(0, 1), (2, 3)];

        let comps = connected_components(&rects, &edges);
        assert_eq!(comps.len(), 2);

        for comp in &comps {
            // All edge indices must be < component size (i.e., local).
            for &(u, v) in &comp.edges {
                assert!(
                    u < comp.node_ids.len(),
                    "Edge source {} out of local range (component size {})",
                    u,
                    comp.node_ids.len()
                );
                assert!(
                    v < comp.node_ids.len(),
                    "Edge target {} out of local range (component size {})",
                    v,
                    comp.node_ids.len()
                );
            }
        }
    }

    #[test]
    fn edge_remapping_consistency() {
        // 5 nodes: comp {0,1,2} with edges (0,1),(1,2),(0,2), comp {3,4} with edge (3,4)
        let rects: Vec<Rectangle> = (0..5).map(|i| rect_at(i as f64 * 3.0, 0.0)).collect();
        let edges: Vec<Edge> = vec![(0, 1), (1, 2), (0, 2), (3, 4)];

        let comps = connected_components(&rects, &edges);
        assert_eq!(comps.len(), 2);

        // The component with 3 nodes should have 3 edges.
        let big_comp = comps.iter().find(|c| c.node_ids.len() == 3).unwrap();
        assert_eq!(big_comp.edges.len(), 3);

        // Verify edges map back correctly to global ids.
        for &(lu, lv) in &big_comp.edges {
            let gu = big_comp.node_ids[lu];
            let gv = big_comp.node_ids[lv];
            // The global edge (gu, gv) or (gv, gu) must exist in original edges.
            assert!(
                edges.contains(&(gu, gv)) || edges.contains(&(gv, gu)),
                "Remapped edge ({},{}) -> global ({},{}) not found in original edges",
                lu,
                lv,
                gu,
                gv
            );
        }
    }

    // ===================================================================
    // Category 8: Node ID preservation
    // ===================================================================

    #[test]
    fn node_ids_match_original_indices() {
        let rects = vec![
            rect_at(0.0, 0.0),
            rect_at(2.0, 0.0),
            rect_at(10.0, 0.0),
            rect_at(12.0, 0.0),
        ];
        let edges: Vec<Edge> = vec![(0, 1), (2, 3)];

        let comps = connected_components(&rects, &edges);

        // Collect all node_ids across components.
        let mut all_ids: Vec<usize> = comps.iter().flat_map(|c| c.node_ids.iter().copied()).collect();
        all_ids.sort();
        assert_eq!(all_ids, vec![0, 1, 2, 3], "All original indices must appear exactly once");
    }

    #[test]
    fn node_ids_no_duplicates() {
        let rects: Vec<Rectangle> = (0..6).map(|i| rect_at(i as f64 * 2.0, 0.0)).collect();
        let edges: Vec<Edge> = vec![(0, 1), (2, 3), (4, 5)];

        let comps = connected_components(&rects, &edges);

        let mut all_ids: Vec<usize> = comps.iter().flat_map(|c| c.node_ids.iter().copied()).collect();
        let total = all_ids.len();
        all_ids.sort();
        all_ids.dedup();
        assert_eq!(all_ids.len(), total, "No duplicate node IDs allowed");
    }

    #[test]
    fn rects_correspond_to_node_ids() {
        // Verify that the rectangle at local index i corresponds to global node node_ids[i].
        let rects = vec![
            rect(0.0, 1.0, 0.0, 1.0),   // node 0
            rect(10.0, 11.0, 0.0, 1.0),  // node 1
            rect(20.0, 21.0, 0.0, 1.0),  // node 2
        ];
        let edges: Vec<Edge> = vec![(0, 2)]; // comp {0,2}, comp {1}

        let comps = connected_components(&rects, &edges);
        assert_eq!(comps.len(), 2);

        for comp in &comps {
            for (local_idx, &global_id) in comp.node_ids.iter().enumerate() {
                assert!(
                    (comp.rects[local_idx].centre_x() - rects[global_id].centre_x()).abs() < 1e-9,
                    "Rect at local {} should match global node {}",
                    local_idx,
                    global_id
                );
            }
        }
    }

    // ===================================================================
    // Category 9: separate_components - overlapping components get separated
    // ===================================================================

    #[test]
    fn separate_two_overlapping_components() {
        // Two components whose bounding boxes overlap.
        let rects = vec![
            rect(0.0, 5.0, 0.0, 5.0),
            rect(3.0, 8.0, 3.0, 8.0),
        ];
        let edges: Vec<Edge> = vec![]; // each node is its own component

        let mut comps = connected_components(&rects, &edges);
        assert_eq!(comps.len(), 2);

        separate_components(&mut comps);

        // After separation, bounding boxes should not overlap.
        let bb0 = comps[0].bounding_box();
        let bb1 = comps[1].bounding_box();
        let overlaps_x = bb0.overlap_x(&bb1) > 0.0;
        let overlaps_y = bb0.overlap_y(&bb1) > 0.0;
        assert!(
            !(overlaps_x && overlaps_y),
            "Bounding boxes should not overlap after separation"
        );
    }

    #[test]
    fn separate_components_single_component_is_noop() {
        let rects = vec![rect_at(0.0, 0.0), rect_at(2.0, 0.0)];
        let edges: Vec<Edge> = vec![(0, 1)];

        let mut comps = connected_components(&rects, &edges);
        assert_eq!(comps.len(), 1);

        let cx_before: Vec<f64> = comps[0].rects.iter().map(|r: &Rectangle| r.centre_x()).collect();
        let cy_before: Vec<f64> = comps[0].rects.iter().map(|r: &Rectangle| r.centre_y()).collect();

        separate_components(&mut comps);

        for (i, r) in comps[0].rects.iter().enumerate() {
            assert!((r.centre_x() - cx_before[i]).abs() < 1e-9);
            assert!((r.centre_y() - cy_before[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn separate_non_overlapping_components_minimal_movement() {
        // Two components already far apart.
        let rects = vec![
            rect(0.0, 2.0, 0.0, 2.0),
            rect(100.0, 102.0, 100.0, 102.0),
        ];
        let edges: Vec<Edge> = vec![];

        let mut comps = connected_components(&rects, &edges);
        let centres_before: Vec<(f64, f64)> = comps
            .iter()
            .map(|c| {
                let bb = c.bounding_box();
                (bb.centre_x(), bb.centre_y())
            })
            .collect();

        separate_components(&mut comps);

        // Movement should be negligible since they don't overlap.
        for (i, c) in comps.iter().enumerate() {
            let bb = c.bounding_box();
            assert!(
                (bb.centre_x() - centres_before[i].0).abs() < 1.0,
                "Component {} moved too much in X",
                i
            );
            assert!(
                (bb.centre_y() - centres_before[i].1).abs() < 1.0,
                "Component {} moved too much in Y",
                i
            );
        }
    }

    // ===================================================================
    // Category 10: Edge cases
    // ===================================================================

    #[test]
    fn single_node_no_edges() {
        let rects = vec![rect_at(5.0, 5.0)];
        let edges: Vec<Edge> = vec![];

        let comps = connected_components(&rects, &edges);
        assert_eq!(comps.len(), 1);
        assert_eq!(comps[0].node_ids, vec![0]);
        assert_eq!(comps[0].rects.len(), 1);
        assert!(comps[0].edges.is_empty());
    }

    #[test]
    fn self_loops_are_ignored() {
        let rects = vec![rect_at(0.0, 0.0), rect_at(5.0, 0.0)];
        let edges: Vec<Edge> = vec![(0, 0), (1, 1)]; // self-loops only

        let comps = connected_components(&rects, &edges);
        // Self-loops don't connect anything, so each node is its own component.
        assert_eq!(comps.len(), 2);
        for c in &comps {
            assert!(c.edges.is_empty(), "Self-loops should not appear in component edges");
        }
    }

    #[test]
    fn self_loops_with_real_edges() {
        let rects = vec![rect_at(0.0, 0.0), rect_at(5.0, 0.0), rect_at(10.0, 0.0)];
        let edges: Vec<Edge> = vec![(0, 0), (0, 1), (2, 2)]; // self-loop + real edge + self-loop

        let comps = connected_components(&rects, &edges);
        assert_eq!(comps.len(), 2); // {0,1} and {2}

        let big = comps.iter().find(|c| c.node_ids.len() == 2).unwrap();
        // Only the real edge (0,1) should be remapped; self-loops filtered out.
        assert_eq!(big.edges.len(), 1);
    }

    #[test]
    fn no_edges_means_each_node_is_own_component() {
        let n = 5;
        let rects: Vec<Rectangle> = (0..n).map(|i| rect_at(i as f64 * 3.0, 0.0)).collect();
        let edges: Vec<Edge> = vec![];

        let comps = connected_components(&rects, &edges);
        assert_eq!(comps.len(), n);
    }

    #[test]
    fn duplicate_edges_handled() {
        // Duplicate edges shouldn't create extra components or break anything.
        let rects = vec![rect_at(0.0, 0.0), rect_at(3.0, 0.0)];
        let edges: Vec<Edge> = vec![(0, 1), (0, 1), (1, 0)];

        let comps = connected_components(&rects, &edges);
        assert_eq!(comps.len(), 1);
        assert_eq!(comps[0].node_ids.len(), 2);
        // All three original edges (including duplicates) get remapped.
        assert_eq!(comps[0].edges.len(), 3);
    }

    #[test]
    fn separate_three_overlapping_components() {
        // Three isolated nodes at the same position.
        let rects = vec![
            rect(0.0, 5.0, 0.0, 5.0),
            rect(1.0, 6.0, 1.0, 6.0),
            rect(2.0, 7.0, 2.0, 7.0),
        ];
        let edges: Vec<Edge> = vec![];

        let mut comps = connected_components(&rects, &edges);
        assert_eq!(comps.len(), 3);

        separate_components(&mut comps);

        // All pairwise bounding boxes should be non-overlapping.
        let bbs: Vec<Rectangle> = comps.iter().map(|c| c.bounding_box()).collect();
        for i in 0..bbs.len() {
            for j in (i + 1)..bbs.len() {
                let both_overlap =
                    bbs[i].overlap_x(&bbs[j]) > 0.0 && bbs[i].overlap_y(&bbs[j]) > 0.0;
                assert!(
                    !both_overlap,
                    "Components {} and {} still overlap after separation",
                    i,
                    j
                );
            }
        }
    }
}
