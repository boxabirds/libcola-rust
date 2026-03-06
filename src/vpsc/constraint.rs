//! Constraint type for the VPSC solver.
//!
//! C++ ref: libvpsc/constraint.h, libvpsc/constraint.cpp

use std::fmt;

/// A separation constraint: left + gap <= right (or left + gap == right).
///
/// C++ ref: vpsc::Constraint
#[derive(Clone)]
pub struct Constraint {
    /// Left variable index.
    pub left: usize,
    /// Right variable index.
    pub right: usize,
    /// Minimum (or exact) distance separating the variables.
    pub gap: f64,
    /// Lagrange multiplier (computed during solving).
    pub lm: f64,
    pub time_stamp: i64,
    pub active: bool,
    /// Whether this is an equality constraint (== vs <=).
    pub equality: bool,
    /// Set to true if this constraint was found to be unsatisfiable.
    pub unsatisfiable: bool,
    /// Whether variable scaling needs to be applied.
    pub needs_scaling: bool,
}

impl Constraint {
    pub fn new(left: usize, right: usize, gap: f64, equality: bool) -> Self {
        Self {
            left,
            right,
            gap,
            lm: 0.0,
            time_stamp: 0,
            active: false,
            equality,
            unsatisfiable: false,
            needs_scaling: true,
        }
    }
}

impl fmt::Debug for Constraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let op = if self.equality { "==" } else { "<=" };
        write!(
            f,
            "Constraint(var({}) + {} {} var({}))",
            self.left, self.gap, op, self.right
        )
    }
}

impl fmt::Display for Constraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let op = if self.equality { "==" } else { "<=" };
        if self.gap < 0.0 {
            write!(f, "var({}) - {} {} var({})", self.left, -self.gap, op, self.right)
        } else {
            write!(f, "var({}) + {} {} var({})", self.left, self.gap, op, self.right)
        }
    }
}

/// Given variables and constraints, returns a filtered set of constraints
/// with redundant equality constraints removed.
///
/// VPSC doesn't work well with redundant equality constraints. This function
/// looks for cycles of equality constraints and removes the redundant ones.
///
/// C++ ref: constraintsRemovingRedundantEqualities
pub fn remove_redundant_equalities(
    num_vars: usize,
    constraints: &[Constraint],
) -> Vec<usize> {
    let mut sets = EqualityConstraintSets::new(num_vars);
    let mut result = Vec::with_capacity(constraints.len());

    for (i, c) in constraints.iter().enumerate() {
        if c.equality {
            if !sets.is_redundant(c.left, c.right, c.gap) {
                sets.merge(c.left, c.right, c.gap);
                result.push(i);
            }
        } else {
            result.push(i);
        }
    }

    result
}

/// Union-Find style structure for detecting redundant equality constraints.
///
/// Each variable starts in its own group with offset 0. When an equality
/// constraint `left + gap == right` is processed, the groups of left and
/// right are merged, adjusting offsets. A constraint is redundant if both
/// variables are already in the same group with consistent offsets.
///
/// C++ ref: EqualityConstraintSet (in constraint.cpp)
struct EqualityConstraintSets {
    /// For each variable, (group_representative, offset_from_representative)
    groups: Vec<Vec<(usize, f64)>>,
}

const REDUNDANCY_TOLERANCE: f64 = 0.0001;

impl EqualityConstraintSets {
    fn new(num_vars: usize) -> Self {
        let groups: Vec<Vec<(usize, f64)>> = (0..num_vars)
            .map(|i| vec![(i, 0.0)])
            .collect();
        Self { groups }
    }

    fn find_group(&self, var: usize) -> Option<(usize, f64)> {
        for (group_idx, group) in self.groups.iter().enumerate() {
            for &(v, offset) in group {
                if v == var {
                    return Some((group_idx, offset));
                }
            }
        }
        None
    }

    fn is_redundant(&self, left: usize, right: usize, gap: f64) -> bool {
        if let (Some((lg, lo)), Some((rg, ro))) = (self.find_group(left), self.find_group(right)) {
            if lg == rg {
                return (lo + gap - ro).abs() < REDUNDANCY_TOLERANCE;
            }
        }
        false
    }

    fn merge(&mut self, left: usize, right: usize, gap: f64) {
        let (lg, lo) = self.find_group(left).unwrap();
        let (rg, ro) = self.find_group(right).unwrap();
        if lg == rg {
            return;
        }

        let new_offset = lo + gap;
        let adjustment = new_offset - ro;

        // Adjust all offsets in rhs group
        for entry in &mut self.groups[rg] {
            entry.1 += adjustment;
        }

        // Merge rhs group into lhs group
        let rhs_group = std::mem::take(&mut self.groups[rg]);
        self.groups[lg].extend(rhs_group);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constraint_creation() {
        let c = Constraint::new(0, 1, 5.0, false);
        assert_eq!(c.left, 0);
        assert_eq!(c.right, 1);
        assert_eq!(c.gap, 5.0);
        assert!(!c.equality);
        assert!(!c.active);
        assert!(!c.unsatisfiable);
    }

    #[test]
    fn test_equality_constraint() {
        let c = Constraint::new(2, 3, 10.0, true);
        assert!(c.equality);
    }

    #[test]
    fn test_remove_redundant_equalities_no_redundancy() {
        let constraints = vec![
            Constraint::new(0, 1, 5.0, true),
            Constraint::new(1, 2, 3.0, true),
        ];
        let result = remove_redundant_equalities(3, &constraints);
        assert_eq!(result, vec![0, 1]);
    }

    #[test]
    fn test_remove_redundant_equalities_with_redundancy() {
        // 0 + 5 == 1, 1 + 3 == 2, 0 + 8 == 2 (redundant: 5+3=8)
        let constraints = vec![
            Constraint::new(0, 1, 5.0, true),
            Constraint::new(1, 2, 3.0, true),
            Constraint::new(0, 2, 8.0, true),
        ];
        let result = remove_redundant_equalities(3, &constraints);
        assert_eq!(result, vec![0, 1]); // third is redundant
    }

    #[test]
    fn test_remove_redundant_equalities_keeps_inequalities() {
        // Inequality between 1-2 doesn't join the equality set, so
        // the equality 0-2 is NOT redundant (0 and 2 are in separate groups).
        let constraints = vec![
            Constraint::new(0, 1, 5.0, true),
            Constraint::new(1, 2, 3.0, false), // inequality always kept
            Constraint::new(0, 2, 8.0, true),  // NOT redundant: 0 and 2 in different groups
        ];
        let result = remove_redundant_equalities(3, &constraints);
        assert_eq!(result, vec![0, 1, 2]);
    }

    #[test]
    fn test_display_positive_gap() {
        let c = Constraint::new(0, 1, 5.0, false);
        assert_eq!(format!("{}", c), "var(0) + 5 <= var(1)");
    }

    #[test]
    fn test_display_negative_gap() {
        let c = Constraint::new(0, 1, -3.0, false);
        assert_eq!(format!("{}", c), "var(0) - 3 <= var(1)");
    }

    #[test]
    fn test_display_equality() {
        let c = Constraint::new(0, 1, 5.0, true);
        assert_eq!(format!("{}", c), "var(0) + 5 == var(1)");
    }
}
