//! Block data structure for the VPSC solver.
//!
//! A block is a group of variables that must be moved together to improve
//! the goal function without violating already active constraints.
//! The variables in a block are spanned by a tree of active constraints.
//!
//! C++ ref: libvpsc/block.h, libvpsc/block.cpp

use std::fmt;

/// Accumulated position statistics for a block.
///
/// Tracks weighted sums needed to compute the optimal block position.
///
/// C++ ref: vpsc::PositionStats
#[derive(Clone, Debug)]
pub struct PositionStats {
    pub scale: f64,
    /// Weighted sum: Σ wi * ai * bi
    pub ab: f64,
    /// Weighted sum: Σ wi * ai * di
    pub ad: f64,
    /// Weighted sum: Σ wi * ai²
    pub a2: f64,
}

impl PositionStats {
    pub fn new() -> Self {
        Self {
            scale: 0.0,
            ab: 0.0,
            ad: 0.0,
            a2: 0.0,
        }
    }

    /// Add a variable's contribution to the position statistics.
    ///
    /// ai = block_scale / var_scale
    /// bi = var_offset / var_scale
    ///
    /// C++ ref: PositionStats::addVariable
    pub fn add_variable(&mut self, var_weight: f64, var_scale: f64, var_offset: f64, var_desired: f64) {
        let ai = self.scale / var_scale;
        let bi = var_offset / var_scale;
        let wi = var_weight;
        self.ab += wi * ai * bi;
        self.ad += wi * ai * var_desired;
        self.a2 += wi * ai * ai;
    }
}

impl Default for PositionStats {
    fn default() -> Self {
        Self::new()
    }
}

/// A block of variables connected by active constraints.
///
/// C++ ref: vpsc::Block
#[derive(Clone)]
pub struct Block {
    /// Variable indices in this block.
    pub vars: Vec<usize>,
    /// Optimal position for this block.
    pub posn: f64,
    /// Position statistics.
    pub ps: PositionStats,
    /// Whether this block has been merged into another and should be cleaned up.
    pub deleted: bool,
    /// Timestamp for constraint staleness detection.
    pub time_stamp: i64,
}

impl Block {
    /// Create a new empty block.
    pub fn new() -> Self {
        Self {
            vars: Vec::new(),
            posn: 0.0,
            ps: PositionStats::new(),
            deleted: false,
            time_stamp: 0,
        }
    }
}

impl Default for Block {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Block(posn={}, vars={:?})", self.posn, self.vars)?;
        if self.deleted {
            write!(f, " DELETED")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_stats_empty() {
        let ps = PositionStats::new();
        assert_eq!(ps.ab, 0.0);
        assert_eq!(ps.ad, 0.0);
        assert_eq!(ps.a2, 0.0);
    }

    #[test]
    fn test_position_stats_add_variable() {
        let mut ps = PositionStats::new();
        ps.scale = 1.0;
        // weight=1, scale=1, offset=0, desired=5
        // ai = 1/1 = 1, bi = 0/1 = 0
        // AB += 1*1*0 = 0, AD += 1*1*5 = 5, A2 += 1*1*1 = 1
        ps.add_variable(1.0, 1.0, 0.0, 5.0);
        assert_eq!(ps.ab, 0.0);
        assert_eq!(ps.ad, 5.0);
        assert_eq!(ps.a2, 1.0);
    }

    #[test]
    fn test_position_stats_multiple_variables() {
        let mut ps = PositionStats::new();
        ps.scale = 1.0;
        // var0: weight=1, scale=1, offset=0, desired=0
        ps.add_variable(1.0, 1.0, 0.0, 0.0);
        // var1: weight=1, scale=1, offset=5, desired=10
        // ai=1, bi=5, AB+=1*1*5=5, AD+=1*1*10=10, A2+=1*1*1=1
        ps.add_variable(1.0, 1.0, 5.0, 10.0);
        assert_eq!(ps.ab, 5.0);
        assert_eq!(ps.ad, 10.0);
        assert_eq!(ps.a2, 2.0);
        // posn should be (AD - AB) / A2 = (10-5)/2 = 2.5
    }

    #[test]
    fn test_block_creation() {
        let b = Block::new();
        assert!(b.vars.is_empty());
        assert_eq!(b.posn, 0.0);
        assert!(!b.deleted);
    }
}
