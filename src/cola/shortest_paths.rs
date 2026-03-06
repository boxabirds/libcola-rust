//! Shortest path algorithms for graph layout.
//!
//! C++ ref: libcola/shortest_paths.h

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use super::commondefs::Edge;

/// Default weight assigned to edges when no explicit weights are provided.
const DEFAULT_EDGE_WEIGHT: f64 = 1.0;

/// Sentinel value representing no path between two nodes.
const NO_PATH: f64 = f64::INFINITY;

/// Initial distance from a node to itself.
const SELF_DISTANCE: f64 = 0.0;

/// Initial adjacency value for non-adjacent nodes.
const NOT_ADJACENT: f64 = 0.0;

// ---------------------------------------------------------------------------
// Dijkstra min-heap state
// ---------------------------------------------------------------------------

/// A node state for Dijkstra's priority queue.
///
/// Ordered by cost (ascending) so that [`BinaryHeap`] — which is a max-heap —
/// pops the *smallest* cost first when we reverse the ordering.
#[derive(Debug, PartialEq)]
struct State {
    cost: f64,
    node: usize,
}

impl Eq for State {}

impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering: smaller cost = higher priority.
        // This turns BinaryHeap (max-heap) into a min-heap.
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.node.cmp(&other.node))
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// ---------------------------------------------------------------------------
// Helper: resolve edge weight
// ---------------------------------------------------------------------------

/// Returns the weight for edge at index `i`, falling back to
/// [`DEFAULT_EDGE_WEIGHT`] when no explicit weights are provided.
#[inline]
fn edge_weight(weights: Option<&[f64]>, i: usize) -> f64 {
    weights.map_or(DEFAULT_EDGE_WEIGHT, |w| w[i])
}

// ---------------------------------------------------------------------------
// Adjacency list builder (shared by dijkstra / johnsons)
// ---------------------------------------------------------------------------

/// Adjacency list entry: (neighbour index, edge weight).
type AdjEntry = (usize, f64);

/// Build undirected adjacency lists from an edge list.
fn build_adjacency(n: usize, edges: &[Edge], weights: Option<&[f64]>) -> Vec<Vec<AdjEntry>> {
    let mut adj: Vec<Vec<AdjEntry>> = vec![Vec::new(); n];
    for (i, &(u, v)) in edges.iter().enumerate() {
        let w = edge_weight(weights, i);
        adj[u].push((v, w));
        adj[v].push((u, w));
    }
    adj
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Fill an adjacency matrix with edge weights ([`NOT_ADJACENT`] for
/// non-adjacent pairs).
///
/// Returns an `n x n` matrix where `D[u][v]` is the edge weight if `(u, v)`
/// is an edge, and [`NOT_ADJACENT`] otherwise. All edges are treated as
/// undirected.
pub fn neighbours(n: usize, edges: &[Edge], weights: Option<&[f64]>) -> Vec<Vec<f64>> {
    let mut d = vec![vec![NOT_ADJACENT; n]; n];
    for (i, &(u, v)) in edges.iter().enumerate() {
        let w = edge_weight(weights, i);
        d[u][v] = w;
        d[v][u] = w;
    }
    d
}

/// O(n^3) all-pairs shortest paths via Floyd-Warshall.
///
/// Returns an `n x n` distance matrix. Unreachable pairs have distance
/// [`f64::INFINITY`].
pub fn floyd_warshall(n: usize, edges: &[Edge], weights: Option<&[f64]>) -> Vec<Vec<f64>> {
    let mut d = vec![vec![NO_PATH; n]; n];

    // Self-distances are zero.
    for i in 0..n {
        d[i][i] = SELF_DISTANCE;
    }

    // Seed with direct edge weights (undirected).
    for (i, &(u, v)) in edges.iter().enumerate() {
        let w = edge_weight(weights, i);
        d[u][v] = w;
        d[v][u] = w;
    }

    // Relaxation.
    for k in 0..n {
        for i in 0..n {
            for j in 0..n {
                let through_k = d[i][k] + d[k][j];
                if through_k < d[i][j] {
                    d[i][j] = through_k;
                }
            }
        }
    }

    d
}

/// Single-source shortest paths from node `s` using Dijkstra's algorithm.
///
/// Returns a vector of length `n` with the shortest distance from `s` to
/// every other node. Unreachable nodes have distance [`f64::INFINITY`].
///
/// Uses [`BinaryHeap`] with lazy deletion instead of a decrease-key
/// pairing heap.
pub fn dijkstra(s: usize, n: usize, edges: &[Edge], weights: Option<&[f64]>) -> Vec<f64> {
    let adj = build_adjacency(n, edges, weights);
    dijkstra_on_adj(s, n, &adj)
}

/// Run Dijkstra on a pre-built adjacency list (avoids rebuilding per source).
fn dijkstra_on_adj(s: usize, n: usize, adj: &[Vec<AdjEntry>]) -> Vec<f64> {
    let mut dist = vec![NO_PATH; n];
    dist[s] = SELF_DISTANCE;

    let mut heap = BinaryHeap::new();
    heap.push(State {
        cost: SELF_DISTANCE,
        node: s,
    });

    while let Some(State { cost, node }) = heap.pop() {
        // Lazy deletion: skip stale entries.
        if cost > dist[node] {
            continue;
        }

        for &(next, w) in &adj[node] {
            let new_cost = cost + w;
            if new_cost < dist[next] {
                dist[next] = new_cost;
                heap.push(State {
                    cost: new_cost,
                    node: next,
                });
            }
        }
    }

    dist
}

/// All-pairs shortest paths using Johnson's algorithm (repeated Dijkstra).
///
/// Returns an `n x n` distance matrix. Builds the adjacency list once and
/// runs [`dijkstra`] from every node.
pub fn johnsons(n: usize, edges: &[Edge], weights: Option<&[f64]>) -> Vec<Vec<f64>> {
    let adj = build_adjacency(n, edges, weights);
    (0..n)
        .map(|s| dijkstra_on_adj(s, n, &adj))
        .collect()
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Tolerance for floating-point comparison.
    const EPSILON: f64 = 1e-12;

    /// Assert two f64 values are approximately equal (handles infinities).
    fn assert_approx(a: f64, b: f64, context: &str) {
        if a.is_infinite() && b.is_infinite() && a.signum() == b.signum() {
            return; // both +inf or both -inf
        }
        assert!(
            (a - b).abs() < EPSILON,
            "{context}: expected {b}, got {a}"
        );
    }

    /// Assert two distance matrices are approximately equal.
    fn assert_matrices_approx(a: &[Vec<f64>], b: &[Vec<f64>], context: &str) {
        assert_eq!(a.len(), b.len(), "{context}: row count mismatch");
        for (i, (ra, rb)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(ra.len(), rb.len(), "{context}: col count mismatch in row {i}");
            for (j, (&va, &vb)) in ra.iter().zip(rb.iter()).enumerate() {
                assert_approx(va, vb, &format!("{context} [{i}][{j}]"));
            }
        }
    }

    // ===================================================================
    // 1. neighbours
    // ===================================================================

    #[test]
    fn neighbours_basic_unweighted() {
        // Triangle: 0-1, 1-2, 0-2
        let edges: Vec<Edge> = vec![(0, 1), (1, 2), (0, 2)];
        let d = neighbours(3, &edges, None);
        assert_approx(d[0][1], DEFAULT_EDGE_WEIGHT, "0-1");
        assert_approx(d[1][0], DEFAULT_EDGE_WEIGHT, "1-0");
        assert_approx(d[1][2], DEFAULT_EDGE_WEIGHT, "1-2");
        assert_approx(d[2][1], DEFAULT_EDGE_WEIGHT, "2-1");
        assert_approx(d[0][2], DEFAULT_EDGE_WEIGHT, "0-2");
        assert_approx(d[2][0], DEFAULT_EDGE_WEIGHT, "2-0");
    }

    #[test]
    fn neighbours_weighted() {
        let edges: Vec<Edge> = vec![(0, 1), (1, 2)];
        let weights = [3.0, 7.0];
        let d = neighbours(3, &edges, Some(&weights));
        assert_approx(d[0][1], 3.0, "0-1");
        assert_approx(d[1][0], 3.0, "1-0");
        assert_approx(d[1][2], 7.0, "1-2");
        assert_approx(d[2][1], 7.0, "2-1");
    }

    #[test]
    fn neighbours_empty_graph() {
        let d = neighbours(4, &[], None);
        for i in 0..4 {
            for j in 0..4 {
                assert_approx(d[i][j], NOT_ADJACENT, &format!("{i}-{j}"));
            }
        }
    }

    #[test]
    fn neighbours_self_loop_free_diagonal() {
        // No self-loops in edges, so diagonal should remain 0.
        let edges: Vec<Edge> = vec![(0, 1), (1, 2)];
        let d = neighbours(3, &edges, None);
        for i in 0..3 {
            assert_approx(d[i][i], NOT_ADJACENT, &format!("diag {i}"));
        }
    }

    #[test]
    fn neighbours_non_adjacent_stays_zero() {
        let edges: Vec<Edge> = vec![(0, 1)];
        let d = neighbours(3, &edges, None);
        assert_approx(d[0][2], NOT_ADJACENT, "0-2 not adjacent");
        assert_approx(d[1][2], NOT_ADJACENT, "1-2 not adjacent");
    }

    // ===================================================================
    // 2. floyd_warshall
    // ===================================================================

    #[test]
    fn floyd_warshall_two_nodes() {
        let edges: Vec<Edge> = vec![(0, 1)];
        let d = floyd_warshall(2, &edges, None);
        assert_approx(d[0][0], SELF_DISTANCE, "0-0");
        assert_approx(d[1][1], SELF_DISTANCE, "1-1");
        assert_approx(d[0][1], DEFAULT_EDGE_WEIGHT, "0-1");
        assert_approx(d[1][0], DEFAULT_EDGE_WEIGHT, "1-0");
    }

    #[test]
    fn floyd_warshall_triangle() {
        // 0-1 (1), 1-2 (1), 0-2 (1)
        let edges: Vec<Edge> = vec![(0, 1), (1, 2), (0, 2)];
        let d = floyd_warshall(3, &edges, None);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { SELF_DISTANCE } else { DEFAULT_EDGE_WEIGHT };
                assert_approx(d[i][j], expected, &format!("{i}-{j}"));
            }
        }
    }

    #[test]
    fn floyd_warshall_disconnected_nodes() {
        // 0-1 connected, node 2 isolated
        let edges: Vec<Edge> = vec![(0, 1)];
        let d = floyd_warshall(3, &edges, None);
        assert_approx(d[0][1], DEFAULT_EDGE_WEIGHT, "0-1");
        assert!(d[0][2].is_infinite(), "0-2 should be unreachable");
        assert!(d[1][2].is_infinite(), "1-2 should be unreachable");
        assert!(d[2][0].is_infinite(), "2-0 should be unreachable");
    }

    #[test]
    fn floyd_warshall_weighted() {
        // 0--3.0--1--7.0--2
        // Shortest 0->2 = 10.0
        let edges: Vec<Edge> = vec![(0, 1), (1, 2)];
        let weights = [3.0, 7.0];
        let d = floyd_warshall(3, &edges, Some(&weights));
        assert_approx(d[0][1], 3.0, "0-1");
        assert_approx(d[1][2], 7.0, "1-2");
        assert_approx(d[0][2], 10.0, "0-2 via 1");
    }

    #[test]
    fn floyd_warshall_weighted_shortcut() {
        // Triangle: 0-1 (1), 1-2 (1), 0-2 (5)
        // Direct 0-2 is 5, but via 1 is 2 — FW should pick 2.
        let edges: Vec<Edge> = vec![(0, 1), (1, 2), (0, 2)];
        let weights = [1.0, 1.0, 5.0];
        let d = floyd_warshall(3, &edges, Some(&weights));
        assert_approx(d[0][2], 2.0, "0-2 should go via 1");
        assert_approx(d[2][0], 2.0, "2-0 should go via 1");
    }

    #[test]
    fn floyd_warshall_linear_chain() {
        // 0-1-2-3
        let edges: Vec<Edge> = vec![(0, 1), (1, 2), (2, 3)];
        let d = floyd_warshall(4, &edges, None);
        assert_approx(d[0][3], 3.0, "0-3 hop count");
        assert_approx(d[0][2], 2.0, "0-2 hop count");
        assert_approx(d[1][3], 2.0, "1-3 hop count");
    }

    // ===================================================================
    // 3. dijkstra
    // ===================================================================

    #[test]
    fn dijkstra_single_node() {
        let d = dijkstra(0, 1, &[], None);
        assert_eq!(d.len(), 1);
        assert_approx(d[0], SELF_DISTANCE, "self");
    }

    #[test]
    fn dijkstra_linear_chain() {
        // 0-1-2-3
        let edges: Vec<Edge> = vec![(0, 1), (1, 2), (2, 3)];
        let d = dijkstra(0, 4, &edges, None);
        assert_approx(d[0], 0.0, "source");
        assert_approx(d[1], 1.0, "1 hop");
        assert_approx(d[2], 2.0, "2 hops");
        assert_approx(d[3], 3.0, "3 hops");
    }

    #[test]
    fn dijkstra_triangle() {
        let edges: Vec<Edge> = vec![(0, 1), (1, 2), (0, 2)];
        let d = dijkstra(0, 3, &edges, None);
        assert_approx(d[1], 1.0, "0-1");
        assert_approx(d[2], 1.0, "0-2 direct");
    }

    #[test]
    fn dijkstra_weighted() {
        // 0--2.0--1--3.0--2, also 0--10.0--2
        // Shortest 0->2 = 5.0 via 1
        let edges: Vec<Edge> = vec![(0, 1), (1, 2), (0, 2)];
        let weights = [2.0, 3.0, 10.0];
        let d = dijkstra(0, 3, &edges, Some(&weights));
        assert_approx(d[0], 0.0, "source");
        assert_approx(d[1], 2.0, "0-1");
        assert_approx(d[2], 5.0, "0-2 via 1");
    }

    #[test]
    fn dijkstra_unreachable_node() {
        // 0-1 connected, node 2 isolated
        let edges: Vec<Edge> = vec![(0, 1)];
        let d = dijkstra(0, 3, &edges, None);
        assert_approx(d[0], 0.0, "source");
        assert_approx(d[1], 1.0, "0-1");
        assert!(d[2].is_infinite(), "node 2 unreachable");
    }

    #[test]
    fn dijkstra_from_different_sources() {
        // 0-1-2
        let edges: Vec<Edge> = vec![(0, 1), (1, 2)];
        let d0 = dijkstra(0, 3, &edges, None);
        let d2 = dijkstra(2, 3, &edges, None);
        // Distances should be symmetric for undirected graph.
        assert_approx(d0[2], d2[0], "symmetry 0-2");
    }

    // ===================================================================
    // 4. johnsons
    // ===================================================================

    #[test]
    fn johnsons_matches_floyd_warshall_triangle() {
        let edges: Vec<Edge> = vec![(0, 1), (1, 2), (0, 2)];
        let fw = floyd_warshall(3, &edges, None);
        let jo = johnsons(3, &edges, None);
        assert_matrices_approx(&fw, &jo, "triangle FW vs Johnson");
    }

    #[test]
    fn johnsons_matches_floyd_warshall_weighted() {
        let edges: Vec<Edge> = vec![(0, 1), (1, 2), (0, 2)];
        let weights = [1.0, 1.0, 5.0];
        let fw = floyd_warshall(3, &edges, Some(&weights));
        let jo = johnsons(3, &edges, Some(&weights));
        assert_matrices_approx(&fw, &jo, "weighted FW vs Johnson");
    }

    #[test]
    fn johnsons_larger_graph() {
        // Star: 0 connected to 1,2,3,4
        let edges: Vec<Edge> = vec![(0, 1), (0, 2), (0, 3), (0, 4)];
        let d = johnsons(5, &edges, None);
        // 0 to any leaf = 1, leaf to leaf = 2
        for leaf in 1..5 {
            assert_approx(d[0][leaf], 1.0, &format!("0-{leaf}"));
            assert_approx(d[leaf][0], 1.0, &format!("{leaf}-0"));
        }
        for i in 1..5 {
            for j in (i + 1)..5 {
                assert_approx(d[i][j], 2.0, &format!("{i}-{j} via hub"));
            }
        }
    }

    // ===================================================================
    // 5. Edge cases
    // ===================================================================

    #[test]
    fn zero_nodes() {
        let n = neighbours(0, &[], None);
        assert!(n.is_empty());

        let fw = floyd_warshall(0, &[], None);
        assert!(fw.is_empty());

        let jo = johnsons(0, &[], None);
        assert!(jo.is_empty());
    }

    #[test]
    fn single_node_no_edges() {
        let n = neighbours(1, &[], None);
        assert_eq!(n.len(), 1);
        assert_approx(n[0][0], NOT_ADJACENT, "neighbours diag");

        let fw = floyd_warshall(1, &[], None);
        assert_approx(fw[0][0], SELF_DISTANCE, "fw self");

        let d = dijkstra(0, 1, &[], None);
        assert_approx(d[0], SELF_DISTANCE, "dijkstra self");

        let jo = johnsons(1, &[], None);
        assert_approx(jo[0][0], SELF_DISTANCE, "johnsons self");
    }

    #[test]
    fn single_edge_symmetry() {
        let edges: Vec<Edge> = vec![(0, 1)];
        let d = floyd_warshall(2, &edges, None);
        assert_approx(d[0][1], d[1][0], "symmetry");
    }

    // ===================================================================
    // 6. Cross-validation: floyd_warshall vs johnsons
    // ===================================================================

    #[test]
    fn cross_validate_line_graph() {
        // 0-1-2-3-4
        let edges: Vec<Edge> = vec![(0, 1), (1, 2), (2, 3), (3, 4)];
        let fw = floyd_warshall(5, &edges, None);
        let jo = johnsons(5, &edges, None);
        assert_matrices_approx(&fw, &jo, "line graph");
    }

    #[test]
    fn cross_validate_weighted_diamond() {
        //     1
        //    / \
        //   2   3
        //    \ /
        //     0
        let edges: Vec<Edge> = vec![(0, 2), (0, 3), (2, 1), (3, 1)];
        let weights = [1.0, 4.0, 2.0, 1.0];
        let fw = floyd_warshall(4, &edges, Some(&weights));
        let jo = johnsons(4, &edges, Some(&weights));
        assert_matrices_approx(&fw, &jo, "weighted diamond");
    }

    #[test]
    fn cross_validate_disconnected_components() {
        // Component A: 0-1, Component B: 2-3
        let edges: Vec<Edge> = vec![(0, 1), (2, 3)];
        let fw = floyd_warshall(4, &edges, None);
        let jo = johnsons(4, &edges, None);
        assert_matrices_approx(&fw, &jo, "disconnected");
        // Verify cross-component is infinite.
        assert!(fw[0][2].is_infinite(), "cross-component unreachable");
        assert!(fw[1][3].is_infinite(), "cross-component unreachable");
    }

    #[test]
    fn cross_validate_complete_graph_k4() {
        // K4: all pairs connected
        let edges: Vec<Edge> = vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
        let fw = floyd_warshall(4, &edges, None);
        let jo = johnsons(4, &edges, None);
        assert_matrices_approx(&fw, &jo, "K4");
        // All non-self distances should be 1 (direct edge).
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { SELF_DISTANCE } else { DEFAULT_EDGE_WEIGHT };
                assert_approx(fw[i][j], expected, &format!("K4 [{i}][{j}]"));
            }
        }
    }

    #[test]
    fn cross_validate_varied_weights() {
        // 0 --1.5-- 1 --2.5-- 2 --0.5-- 3
        // 0 --------5.0------------- 3
        let edges: Vec<Edge> = vec![(0, 1), (1, 2), (2, 3), (0, 3)];
        let weights = [1.5, 2.5, 0.5, 5.0];
        let fw = floyd_warshall(4, &edges, Some(&weights));
        let jo = johnsons(4, &edges, Some(&weights));
        assert_matrices_approx(&fw, &jo, "varied weights");
        // 0->3: min(5.0, 1.5+2.5+0.5=4.5) = 4.5
        assert_approx(fw[0][3], 4.5, "0-3 shortest");
    }
}
