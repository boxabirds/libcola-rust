//! Padding and margin box for cluster boundaries.
//!
//! C++ ref: libcola/box.h, libcola/box.cpp

use crate::vpsc::Rectangle;

/// Dimension index for the X (horizontal) axis.
const X_DIM: usize = 0;
/// Dimension index for the Y (vertical) axis.
const Y_DIM: usize = 1;
/// Number of supported dimensions.
const NUM_DIMS: usize = 2;

/// A padding/margin box with four edge values (min/max for each dimension).
///
/// Stores non-negative margin amounts that can be applied to expand a
/// [`Rectangle`] outward.
///
/// C++ ref: `cola::Box`
#[derive(Debug, Clone, Copy)]
pub struct Box {
    min: [f64; NUM_DIMS],
    max: [f64; NUM_DIMS],
}

impl Box {
    /// Creates a box with individual edge margins.
    ///
    /// Negative values are clamped to zero.
    pub fn new(x_min: f64, x_max: f64, y_min: f64, y_max: f64) -> Self {
        Self {
            min: [Self::non_negative(x_min), Self::non_negative(y_min)],
            max: [Self::non_negative(x_max), Self::non_negative(y_max)],
        }
    }

    /// Creates a box with the same margin on all four edges.
    ///
    /// Negative values are clamped to zero.
    pub fn uniform(all: f64) -> Self {
        let val = Self::non_negative(all);
        Self {
            min: [val; NUM_DIMS],
            max: [val; NUM_DIMS],
        }
    }

    /// Creates an empty box with all margins set to zero.
    pub fn empty_box() -> Self {
        Self {
            min: [0.0; NUM_DIMS],
            max: [0.0; NUM_DIMS],
        }
    }

    /// Returns `true` if all margin values are zero.
    pub fn is_empty(&self) -> bool {
        self.min[X_DIM] == 0.0
            && self.min[Y_DIM] == 0.0
            && self.max[X_DIM] == 0.0
            && self.max[Y_DIM] == 0.0
    }

    /// Returns the minimum margin for the given dimension.
    ///
    /// Returns `0.0` if `dim >= 2`.
    pub fn min(&self, dim: usize) -> f64 {
        if dim >= NUM_DIMS {
            return 0.0;
        }
        self.min[dim]
    }

    /// Returns the maximum margin for the given dimension.
    ///
    /// Returns `0.0` if `dim >= 2`.
    pub fn max(&self, dim: usize) -> f64 {
        if dim >= NUM_DIMS {
            return 0.0;
        }
        self.max[dim]
    }

    /// Expands a rectangle outward by the margin amounts in this box.
    ///
    /// The result is a new rectangle where each edge is pushed outward by
    /// the corresponding margin value:
    /// - `min_x = rect.min_x - self.min[X_DIM]`
    /// - `max_x = rect.max_x + self.max[X_DIM]`
    /// - `min_y = rect.min_y - self.min[Y_DIM]`
    /// - `max_y = rect.max_y + self.max[Y_DIM]`
    pub fn apply_to_rectangle(&self, rect: &Rectangle) -> Rectangle {
        Rectangle::new(
            rect.get_min_x() - self.min[X_DIM],
            rect.get_max_x() + self.max[X_DIM],
            rect.get_min_y() - self.min[Y_DIM],
            rect.get_max_y() + self.max[Y_DIM],
        )
    }

    /// Clamps a value to be non-negative.
    fn non_negative(value: f64) -> f64 {
        if value < 0.0 { 0.0 } else { value }
    }
}

impl Default for Box {
    fn default() -> Self {
        Self::empty_box()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    // ===================================================================
    // Category 1: Construction
    // ===================================================================

    #[test]
    fn construction_empty_box_has_all_zeros() {
        let b = Box::empty_box();
        assert_eq!(b.min(X_DIM), 0.0);
        assert_eq!(b.max(X_DIM), 0.0);
        assert_eq!(b.min(Y_DIM), 0.0);
        assert_eq!(b.max(Y_DIM), 0.0);
    }

    #[test]
    fn construction_uniform_sets_all_edges_equal() {
        let margin = 5.0;
        let b = Box::uniform(margin);
        assert_eq!(b.min(X_DIM), margin);
        assert_eq!(b.max(X_DIM), margin);
        assert_eq!(b.min(Y_DIM), margin);
        assert_eq!(b.max(Y_DIM), margin);
    }

    #[test]
    fn construction_per_edge_stores_each_value() {
        let b = Box::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(b.min(X_DIM), 1.0);
        assert_eq!(b.max(X_DIM), 2.0);
        assert_eq!(b.min(Y_DIM), 3.0);
        assert_eq!(b.max(Y_DIM), 4.0);
    }

    #[test]
    fn construction_negative_values_clamped_to_zero() {
        let b = Box::new(-5.0, -3.0, -1.0, -0.001);
        assert_eq!(b.min(X_DIM), 0.0);
        assert_eq!(b.max(X_DIM), 0.0);
        assert_eq!(b.min(Y_DIM), 0.0);
        assert_eq!(b.max(Y_DIM), 0.0);
    }

    #[test]
    fn construction_negative_uniform_clamped_to_zero() {
        let b = Box::uniform(-10.0);
        assert_eq!(b.min(X_DIM), 0.0);
        assert_eq!(b.max(X_DIM), 0.0);
        assert_eq!(b.min(Y_DIM), 0.0);
        assert_eq!(b.max(Y_DIM), 0.0);
    }

    #[test]
    fn construction_mixed_positive_and_negative_values() {
        let b = Box::new(-1.0, 2.0, 3.0, -4.0);
        assert_eq!(b.min(X_DIM), 0.0);
        assert_eq!(b.max(X_DIM), 2.0);
        assert_eq!(b.min(Y_DIM), 3.0);
        assert_eq!(b.max(Y_DIM), 0.0);
    }

    // ===================================================================
    // Category 2: is_empty
    // ===================================================================

    #[test]
    fn is_empty_true_for_zero_box() {
        let b = Box::empty_box();
        assert!(b.is_empty());
    }

    #[test]
    fn is_empty_true_for_default_box() {
        let b = Box::default();
        assert!(b.is_empty());
    }

    #[test]
    fn is_empty_false_when_any_edge_nonzero() {
        assert!(!Box::new(1.0, 0.0, 0.0, 0.0).is_empty());
        assert!(!Box::new(0.0, 1.0, 0.0, 0.0).is_empty());
        assert!(!Box::new(0.0, 0.0, 1.0, 0.0).is_empty());
        assert!(!Box::new(0.0, 0.0, 0.0, 1.0).is_empty());
    }

    #[test]
    fn is_empty_false_for_uniform_nonzero() {
        let b = Box::uniform(0.001);
        assert!(!b.is_empty());
    }

    #[test]
    fn is_empty_true_for_all_negative_inputs() {
        // All negatives get clamped to zero, so box should be empty.
        let b = Box::new(-1.0, -2.0, -3.0, -4.0);
        assert!(b.is_empty());
    }

    // ===================================================================
    // Category 3: min/max accessors
    // ===================================================================

    #[test]
    fn accessor_min_valid_dims() {
        let b = Box::new(10.0, 20.0, 30.0, 40.0);
        assert_eq!(b.min(X_DIM), 10.0);
        assert_eq!(b.min(Y_DIM), 30.0);
    }

    #[test]
    fn accessor_max_valid_dims() {
        let b = Box::new(10.0, 20.0, 30.0, 40.0);
        assert_eq!(b.max(X_DIM), 20.0);
        assert_eq!(b.max(Y_DIM), 40.0);
    }

    #[test]
    fn accessor_min_out_of_range_returns_zero() {
        let b = Box::uniform(99.0);
        assert_eq!(b.min(2), 0.0);
        assert_eq!(b.min(3), 0.0);
        assert_eq!(b.min(usize::MAX), 0.0);
    }

    #[test]
    fn accessor_max_out_of_range_returns_zero() {
        let b = Box::uniform(99.0);
        assert_eq!(b.max(2), 0.0);
        assert_eq!(b.max(100), 0.0);
        assert_eq!(b.max(usize::MAX), 0.0);
    }

    // ===================================================================
    // Category 4: apply_to_rectangle
    // ===================================================================

    #[test]
    fn apply_expands_rectangle_correctly() {
        let b = Box::new(1.0, 2.0, 3.0, 4.0);
        let rect = Rectangle::new(10.0, 20.0, 30.0, 40.0);
        let result = b.apply_to_rectangle(&rect);

        assert!((result.get_min_x() - 9.0).abs() < EPSILON);
        assert!((result.get_max_x() - 22.0).abs() < EPSILON);
        assert!((result.get_min_y() - 27.0).abs() < EPSILON);
        assert!((result.get_max_y() - 44.0).abs() < EPSILON);
    }

    #[test]
    fn apply_empty_box_returns_same_dimensions() {
        let b = Box::empty_box();
        let rect = Rectangle::new(5.0, 15.0, 25.0, 35.0);
        let result = b.apply_to_rectangle(&rect);

        assert!((result.get_min_x() - rect.get_min_x()).abs() < EPSILON);
        assert!((result.get_max_x() - rect.get_max_x()).abs() < EPSILON);
        assert!((result.get_min_y() - rect.get_min_y()).abs() < EPSILON);
        assert!((result.get_max_y() - rect.get_max_y()).abs() < EPSILON);
    }

    #[test]
    fn apply_uniform_margin_expands_symmetrically() {
        let margin = 5.0;
        let b = Box::uniform(margin);
        let rect = Rectangle::new(10.0, 20.0, 30.0, 40.0);
        let result = b.apply_to_rectangle(&rect);

        assert!((result.get_min_x() - 5.0).abs() < EPSILON);
        assert!((result.get_max_x() - 25.0).abs() < EPSILON);
        assert!((result.get_min_y() - 25.0).abs() < EPSILON);
        assert!((result.get_max_y() - 45.0).abs() < EPSILON);

        // Width and height should each grow by 2 * margin
        let original_width = 10.0;
        let original_height = 10.0;
        let expected_width_growth = 2.0 * margin;
        let expected_height_growth = 2.0 * margin;
        assert!((result.width() - (original_width + expected_width_growth)).abs() < EPSILON);
        assert!((result.height() - (original_height + expected_height_growth)).abs() < EPSILON);
    }

    #[test]
    fn apply_result_is_valid_rectangle() {
        let b = Box::new(1.0, 2.0, 3.0, 4.0);
        let rect = Rectangle::new(10.0, 20.0, 30.0, 40.0);
        let result = b.apply_to_rectangle(&rect);
        assert!(result.is_valid());
    }

    #[test]
    fn apply_preserves_centre_with_uniform_margin() {
        let margin = 7.5;
        let b = Box::uniform(margin);
        let rect = Rectangle::new(10.0, 20.0, 30.0, 50.0);
        let result = b.apply_to_rectangle(&rect);

        assert!((result.centre_x() - rect.centre_x()).abs() < EPSILON);
        assert!((result.centre_y() - rect.centre_y()).abs() < EPSILON);
    }

    #[test]
    fn apply_asymmetric_margin_shifts_centre() {
        let b = Box::new(0.0, 10.0, 0.0, 0.0);
        let rect = Rectangle::new(0.0, 10.0, 0.0, 10.0);
        let result = b.apply_to_rectangle(&rect);

        // Only max_x grows, so centre_x should shift right
        assert!(result.centre_x() > rect.centre_x());
        // Y centre unchanged
        assert!((result.centre_y() - rect.centre_y()).abs() < EPSILON);
    }

    // ===================================================================
    // Category 5: Default trait
    // ===================================================================

    #[test]
    fn default_produces_empty_box() {
        let b = Box::default();
        assert!(b.is_empty());
        assert_eq!(b.min(X_DIM), 0.0);
        assert_eq!(b.max(X_DIM), 0.0);
        assert_eq!(b.min(Y_DIM), 0.0);
        assert_eq!(b.max(Y_DIM), 0.0);
    }

    #[test]
    fn default_matches_empty_box() {
        let d = Box::default();
        let e = Box::empty_box();
        assert_eq!(d.min(X_DIM), e.min(X_DIM));
        assert_eq!(d.max(X_DIM), e.max(X_DIM));
        assert_eq!(d.min(Y_DIM), e.min(Y_DIM));
        assert_eq!(d.max(Y_DIM), e.max(Y_DIM));
    }

    // ===================================================================
    // Category 6: non_negative helper
    // ===================================================================

    #[test]
    fn non_negative_clamps_negative() {
        assert_eq!(Box::non_negative(-1.0), 0.0);
        assert_eq!(Box::non_negative(-f64::INFINITY), 0.0);
    }

    #[test]
    fn non_negative_preserves_positive() {
        assert_eq!(Box::non_negative(0.0), 0.0);
        assert_eq!(Box::non_negative(42.0), 42.0);
        assert_eq!(Box::non_negative(f64::INFINITY), f64::INFINITY);
    }

    // ===================================================================
    // Category 7: Clone and Copy semantics
    // ===================================================================

    #[test]
    fn copy_semantics_are_independent() {
        let b1 = Box::new(1.0, 2.0, 3.0, 4.0);
        let b2 = b1; // Copy
        // Both should have identical values (they are independent copies).
        assert_eq!(b1.min(X_DIM), b2.min(X_DIM));
        assert_eq!(b1.max(X_DIM), b2.max(X_DIM));
        assert_eq!(b1.min(Y_DIM), b2.min(Y_DIM));
        assert_eq!(b1.max(Y_DIM), b2.max(Y_DIM));
    }
}
