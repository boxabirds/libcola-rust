//! Convex hull computation using Graham's scan.
//!
//! Used for cluster boundary computation in the layout engine.
//!
//! C++ ref: libcola/convex_hull.h, libcola/convex_hull.cpp

use std::cmp::Ordering;

/// Cross product of vectors (p0->p1) and (p0->p2).
///
/// Positive when p0->p1->p2 is a counter-clockwise turn,
/// negative for clockwise, zero for collinear.
fn cross_product(x0: f64, y0: f64, x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
}

/// Squared distance from the origin for a displacement vector.
fn dist_sq(dx: f64, dy: f64) -> f64 {
    dx * dx + dy * dy
}

/// Compute the convex hull of a point set using Graham's scan.
///
/// Returns indices into the `x`/`y` arrays forming the hull in counter-clockwise order.
///
/// For degenerate inputs: returns empty for n=0, `[0]` for n=1, `[0, 1]` for n=2.
///
/// # Panics
///
/// Panics if `x.len() != y.len()`.
pub fn convex_hull(x: &[f64], y: &[f64]) -> Vec<usize> {
    assert_eq!(x.len(), y.len(), "x and y must have the same length");

    let n = x.len();

    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0];
    }
    if n == 2 {
        return vec![0, 1];
    }

    // Find the point with minimum Y (leftmost if tie) to use as pivot.
    let mut pivot = 0;
    let mut min_y = f64::MAX;
    let mut min_x = f64::MAX;
    for i in 0..n {
        if (y[i] < min_y) || (y[i] == min_y && x[i] < min_x) {
            pivot = i;
            min_y = y[i];
            min_x = x[i];
        }
    }

    // Collect all point indices except the pivot.
    let mut points: Vec<usize> = (0..n).filter(|&i| i != pivot).collect();

    // Sort by polar angle from the pivot (counter-clockwise order).
    // On tie (collinear), the closer point comes first.
    let px = x[pivot];
    let py = y[pivot];
    points.sort_by(|&i, &j| {
        let xi = x[i] - px;
        let xj = x[j] - px;
        let yi = y[i] - py;
        let yj = y[j] - py;
        let cross = xi * yj - xj * yi;
        if cross != 0.0 {
            if cross > 0.0 {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        } else {
            // Collinear: closer point first.
            let di = dist_sq(xi, yi);
            let dj = dist_sq(xj, yj);
            di.partial_cmp(&dj).unwrap_or(Ordering::Equal)
        }
    });

    // Graham scan.
    let mut hull: Vec<usize> = Vec::with_capacity(n);
    hull.push(pivot);
    hull.push(points[0]);

    for &pi in &points[1..] {
        let o = cross_product(
            x[hull[hull.len() - 2]],
            y[hull[hull.len() - 2]],
            x[hull[hull.len() - 1]],
            y[hull[hull.len() - 1]],
            x[pi],
            y[pi],
        );
        if o == 0.0 {
            // Collinear: replace the last point with this farther one.
            hull.pop();
            hull.push(pi);
        } else if o > 0.0 {
            // Counter-clockwise turn: extend the hull.
            hull.push(pi);
        } else {
            // Clockwise turn: backtrack until we can make a left turn.
            while hull.len() > 2 {
                hull.pop();
                let o2 = cross_product(
                    x[hull[hull.len() - 2]],
                    y[hull[hull.len() - 2]],
                    x[hull[hull.len() - 1]],
                    y[hull[hull.len() - 1]],
                    x[pi],
                    y[pi],
                );
                if o2 > 0.0 {
                    break;
                }
            }
            hull.push(pi);
        }
    }

    hull
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Assert that the hull contains exactly the expected indices (order-independent).
    fn assert_hull_indices(hull: &[usize], expected: &[usize]) {
        assert_eq!(
            hull.len(),
            expected.len(),
            "hull length mismatch: got {:?}, expected {:?}",
            hull,
            expected,
        );
        for &idx in expected {
            assert!(
                hull.contains(&idx),
                "expected index {} in hull {:?}",
                idx,
                hull,
            );
        }
    }

    // ---------------------------------------------------------------
    // 12. Cross product unit tests
    // ---------------------------------------------------------------

    #[test]
    fn cross_product_positive_for_ccw() {
        // (0,0)->(1,0)->(0,1) is a left (CCW) turn.
        let cp = cross_product(0.0, 0.0, 1.0, 0.0, 0.0, 1.0);
        assert!(cp > 0.0, "expected positive, got {}", cp);
    }

    #[test]
    fn cross_product_negative_for_cw() {
        // (0,0)->(0,1)->(1,0) is a right (CW) turn.
        let cp = cross_product(0.0, 0.0, 0.0, 1.0, 1.0, 0.0);
        assert!(cp < 0.0, "expected negative, got {}", cp);
    }

    #[test]
    fn cross_product_zero_for_collinear() {
        let cp = cross_product(0.0, 0.0, 1.0, 1.0, 2.0, 2.0);
        assert!(
            cp.abs() < f64::EPSILON,
            "expected ~0, got {}",
            cp,
        );
    }

    #[test]
    fn cross_product_with_offset_origin() {
        let cp = cross_product(5.0, 5.0, 6.0, 5.0, 5.0, 6.0);
        assert!(cp > 0.0);
    }

    // ---------------------------------------------------------------
    // Edge: empty input
    // ---------------------------------------------------------------

    #[test]
    fn empty_input() {
        let hull = convex_hull(&[], &[]);
        assert!(hull.is_empty());
    }

    // ---------------------------------------------------------------
    // 5. Single point
    // ---------------------------------------------------------------

    #[test]
    fn single_point() {
        let hull = convex_hull(&[3.0], &[4.0]);
        assert_eq!(hull, vec![0]);
    }

    // ---------------------------------------------------------------
    // 6. Two points
    // ---------------------------------------------------------------

    #[test]
    fn two_points() {
        let hull = convex_hull(&[0.0, 1.0], &[0.0, 1.0]);
        assert_eq!(hull.len(), 2);
        assert!(hull.contains(&0));
        assert!(hull.contains(&1));
    }

    // ---------------------------------------------------------------
    // 1. Triangle - all 3 on hull
    // ---------------------------------------------------------------

    #[test]
    fn triangle_all_on_hull() {
        let x = [0.0, 1.0, 0.5];
        let y = [0.0, 0.0, 1.0];
        let hull = convex_hull(&x, &y);
        assert_hull_indices(&hull, &[0, 1, 2]);
    }

    // ---------------------------------------------------------------
    // 2. Square - all 4 on hull
    // ---------------------------------------------------------------

    #[test]
    fn square_all_on_hull() {
        let x = [0.0, 1.0, 1.0, 0.0];
        let y = [0.0, 0.0, 1.0, 1.0];
        let hull = convex_hull(&x, &y);
        assert_hull_indices(&hull, &[0, 1, 2, 3]);
    }

    // ---------------------------------------------------------------
    // 3. Interior point excluded
    // ---------------------------------------------------------------

    #[test]
    fn interior_point_excluded() {
        let x = [0.0, 2.0, 2.0, 0.0, 1.0];
        let y = [0.0, 0.0, 2.0, 2.0, 1.0];
        let hull = convex_hull(&x, &y);
        assert_hull_indices(&hull, &[0, 1, 2, 3]);
        assert!(!hull.contains(&4), "center should not be on hull");
    }

    // ---------------------------------------------------------------
    // 4. Collinear points - middle one excluded
    // ---------------------------------------------------------------

    #[test]
    fn collinear_points_middle_excluded() {
        let x = [0.0, 1.0, 2.0];
        let y = [0.0, 1.0, 2.0];
        let hull = convex_hull(&x, &y);
        assert_eq!(hull.len(), 2, "collinear hull should have 2 points, got {:?}", hull);
        assert!(hull.contains(&0));
        assert!(hull.contains(&2));
        assert!(!hull.contains(&1), "middle collinear point should be excluded");
    }

    // ---------------------------------------------------------------
    // 7. Duplicate points - no panic
    // ---------------------------------------------------------------

    #[test]
    fn duplicate_points_no_panic() {
        let x = [1.0, 1.0, 1.0, 1.0];
        let y = [1.0, 1.0, 1.0, 1.0];
        let hull = convex_hull(&x, &y);
        // Degenerate but must not panic.
        assert!(!hull.is_empty());
    }

    #[test]
    fn duplicates_with_distinct_points() {
        // Two distinct corners each duplicated, plus an interior point.
        let x = [0.0, 1.0, 0.0, 1.0, 0.5];
        let y = [0.0, 0.0, 1.0, 1.0, 0.5];
        let hull = convex_hull(&x, &y);
        assert!(!hull.contains(&4), "interior point should not be on hull");
    }

    // ---------------------------------------------------------------
    // 8. Large convex polygon - octagon
    // ---------------------------------------------------------------

    #[test]
    fn octagon_all_on_hull() {
        const NUM_SIDES: usize = 8;
        let mut x = [0.0; NUM_SIDES];
        let mut y = [0.0; NUM_SIDES];
        for i in 0..NUM_SIDES {
            let angle = 2.0 * PI * (i as f64) / (NUM_SIDES as f64);
            x[i] = angle.cos();
            y[i] = angle.sin();
        }
        let hull = convex_hull(&x, &y);
        let expected: Vec<usize> = (0..NUM_SIDES).collect();
        assert_hull_indices(&hull, &expected);
    }

    #[test]
    fn hexagon_all_on_hull() {
        const NUM_SIDES: usize = 6;
        let mut x = [0.0; NUM_SIDES];
        let mut y = [0.0; NUM_SIDES];
        for i in 0..NUM_SIDES {
            let angle = 2.0 * PI * (i as f64) / (NUM_SIDES as f64);
            x[i] = angle.cos();
            y[i] = angle.sin();
        }
        let hull = convex_hull(&x, &y);
        let expected: Vec<usize> = (0..NUM_SIDES).collect();
        assert_hull_indices(&hull, &expected);
    }

    // ---------------------------------------------------------------
    // 9. Star shape - only outer tips on hull
    // ---------------------------------------------------------------

    #[test]
    fn star_shape_only_tips_on_hull() {
        const NUM_TIPS: usize = 5;
        const OUTER_RADIUS: f64 = 2.0;
        const INNER_RADIUS: f64 = 0.8;
        const TOTAL_POINTS: usize = NUM_TIPS * 2;

        let mut x = [0.0; TOTAL_POINTS];
        let mut y = [0.0; TOTAL_POINTS];
        for i in 0..TOTAL_POINTS {
            let angle = 2.0 * PI * (i as f64) / (TOTAL_POINTS as f64) - PI / 2.0;
            let r = if i % 2 == 0 { OUTER_RADIUS } else { INNER_RADIUS };
            x[i] = r * angle.cos();
            y[i] = r * angle.sin();
        }
        let hull = convex_hull(&x, &y);

        let outer_tips: Vec<usize> = (0..TOTAL_POINTS).filter(|i| i % 2 == 0).collect();
        assert_hull_indices(&hull, &outer_tips);
    }

    // ---------------------------------------------------------------
    // 10. All same Y - horizontal collinear line
    // ---------------------------------------------------------------

    #[test]
    fn all_same_y_horizontal_line() {
        let x = [0.0, 3.0, 1.0, 2.0];
        let y = [5.0, 5.0, 5.0, 5.0];
        let hull = convex_hull(&x, &y);
        // All collinear: only the two endpoints should remain.
        assert_eq!(hull.len(), 2, "horizontal line hull: {:?}", hull);
        assert!(hull.contains(&0));
        assert!(hull.contains(&1));
    }

    // ---------------------------------------------------------------
    // 11. Pivot selection - min-Y leftmost is first
    // ---------------------------------------------------------------

    #[test]
    fn pivot_is_min_y_leftmost() {
        // Index 2 has min Y (0.0) and smallest X (1.0) among min-Y points.
        let x = [5.0, 3.0, 1.0, 4.0];
        let y = [2.0, 0.0, 0.0, 3.0];
        let hull = convex_hull(&x, &y);
        assert_eq!(hull[0], 2, "pivot should be index 2, got {}", hull[0]);
    }

    #[test]
    fn pivot_is_first_when_unique_min_y() {
        let x = [10.0, 0.0, 5.0];
        let y = [0.0, 5.0, 10.0];
        let hull = convex_hull(&x, &y);
        assert_eq!(hull[0], 0, "pivot should be index 0");
    }

    // ---------------------------------------------------------------
    // Additional robustness
    // ---------------------------------------------------------------

    #[test]
    fn many_interior_points_excluded() {
        const BORDER: f64 = 10.0;
        let mut x = vec![0.0, BORDER, BORDER, 0.0];
        let mut y = vec![0.0, 0.0, BORDER, BORDER];

        // Fill interior with a grid.
        const GRID_STEP: f64 = 1.0;
        let mut row = GRID_STEP;
        while row < BORDER {
            let mut col = GRID_STEP;
            while col < BORDER {
                x.push(col);
                y.push(row);
                col += GRID_STEP;
            }
            row += GRID_STEP;
        }

        let hull = convex_hull(&x, &y);
        assert_hull_indices(&hull, &[0, 1, 2, 3]);
    }

    #[test]
    fn hull_is_counter_clockwise() {
        let x = [0.0, 1.0, 1.0, 0.0];
        let y = [0.0, 0.0, 1.0, 1.0];
        let hull = convex_hull(&x, &y);

        // Every consecutive triple on the hull must have non-negative cross product.
        for i in 0..hull.len() {
            let a = hull[i];
            let b = hull[(i + 1) % hull.len()];
            let c = hull[(i + 2) % hull.len()];
            let cp = cross_product(x[a], y[a], x[b], y[b], x[c], y[c]);
            assert!(
                cp >= 0.0,
                "hull not CCW at ({}, {}, {}): cross = {}",
                a, b, c, cp,
            );
        }
    }
}
