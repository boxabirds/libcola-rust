//! Common definitions shared across the cola library.
//!
//! C++ ref: libcola/commondefs.h

/// Edge type: (source, target) node index pair.
pub type Edge = (usize, usize);

/// Overlap resolution mode.
///
/// C++ ref: cola::NonOverlapConstraintsMode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NonOverlapConstraintsMode {
    /// No overlap resolution.
    None,
    /// Resolve overlaps only horizontally.
    Horizontal,
    /// Resolve in both dimensions (choosing the direction with less displacement).
    Both,
}

/// A fixed-node tracker: records which nodes are pinned in place.
///
/// C++ ref: cola::FixedList
#[derive(Debug, Clone)]
pub struct FixedList {
    fixed: Vec<bool>,
    all_fixed: bool,
}

impl FixedList {
    pub fn new(n: usize) -> Self {
        Self {
            fixed: vec![false; n],
            all_fixed: false,
        }
    }

    /// Mark node `i` as fixed (or unfixed).
    pub fn set(&mut self, i: usize, value: bool) {
        debug_assert!(i < self.fixed.len());
        self.fixed[i] = value;
    }

    /// Check if node `i` is fixed.
    ///
    /// Returns false if `all_fixed` is true or `i` is out of range
    /// (matching the C++ behaviour where allFixed bypasses per-node checks).
    pub fn check(&self, i: usize) -> bool {
        if self.all_fixed || i >= self.fixed.len() {
            return false;
        }
        self.fixed[i]
    }

    /// Unset all fixed flags.
    pub fn unset_all(&mut self) {
        self.fixed.fill(false);
    }

    /// Set the global "fix all" flag.
    ///
    /// When true, `check()` returns false for all nodes (matching C++ semantics
    /// where `allFixed` causes the check to short-circuit to false).
    pub fn fix_all(&mut self, val: bool) {
        self.all_fixed = val;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===================================================================
    // FixedList
    // ===================================================================

    #[test]
    fn test_fixed_list_initial_state() {
        let fl = FixedList::new(5);
        for i in 0..5 {
            assert!(!fl.check(i), "Node {} should not be fixed initially", i);
        }
    }

    #[test]
    fn test_fixed_list_set_and_check() {
        let mut fl = FixedList::new(5);
        fl.set(2, true);
        assert!(!fl.check(0));
        assert!(!fl.check(1));
        assert!(fl.check(2));
        assert!(!fl.check(3));
        assert!(!fl.check(4));
    }

    #[test]
    fn test_fixed_list_set_then_unset() {
        let mut fl = FixedList::new(3);
        fl.set(0, true);
        fl.set(1, true);
        assert!(fl.check(0));
        assert!(fl.check(1));
        fl.set(0, false);
        assert!(!fl.check(0));
        assert!(fl.check(1));
    }

    #[test]
    fn test_fixed_list_unset_all() {
        let mut fl = FixedList::new(3);
        fl.set(0, true);
        fl.set(1, true);
        fl.set(2, true);
        fl.unset_all();
        for i in 0..3 {
            assert!(!fl.check(i));
        }
    }

    #[test]
    fn test_fixed_list_fix_all_returns_false() {
        // C++ semantics: when allFixed is true, check() returns false
        let mut fl = FixedList::new(3);
        fl.set(1, true);
        fl.fix_all(true);
        assert!(!fl.check(0));
        assert!(!fl.check(1)); // even though set, fix_all overrides
        assert!(!fl.check(2));
    }

    #[test]
    fn test_fixed_list_out_of_range_returns_false() {
        let fl = FixedList::new(3);
        assert!(!fl.check(10)); // out of range
        assert!(!fl.check(usize::MAX));
    }

    #[test]
    fn test_fixed_list_zero_size() {
        let fl = FixedList::new(0);
        assert!(!fl.check(0));
    }

    // ===================================================================
    // NonOverlapConstraintsMode
    // ===================================================================

    #[test]
    fn test_non_overlap_mode_equality() {
        assert_eq!(NonOverlapConstraintsMode::None, NonOverlapConstraintsMode::None);
        assert_ne!(NonOverlapConstraintsMode::None, NonOverlapConstraintsMode::Horizontal);
        assert_ne!(NonOverlapConstraintsMode::Horizontal, NonOverlapConstraintsMode::Both);
    }

    #[test]
    fn test_non_overlap_mode_clone() {
        let mode = NonOverlapConstraintsMode::Both;
        let cloned = mode;
        assert_eq!(mode, cloned);
    }
}
