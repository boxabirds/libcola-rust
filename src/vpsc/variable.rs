//! Variable type for the VPSC solver.
//!
//! C++ ref: libvpsc/variable.h, libvpsc/variable.cpp

use std::fmt;

/// A variable with an ideal position, final position, and weight.
///
/// When creating a variable you specify an ideal value and a weight — how
/// much the variable wants to be at its ideal position. After solving, read
/// back the final position.
///
/// C++ ref: vpsc::Variable
#[derive(Clone)]
pub struct Variable {
    pub id: usize,
    pub desired_position: f64,
    pub final_position: f64,
    pub weight: f64,
    /// Translates variable to another space.
    pub scale: f64,
    pub offset: f64,
    /// Index of the block this variable belongs to.
    pub block: Option<usize>,
    pub visited: bool,
    pub fixed_desired_position: bool,
    /// Incoming constraint indices (constraints where this is the right var).
    pub in_constraints: Vec<usize>,
    /// Outgoing constraint indices (constraints where this is the left var).
    pub out_constraints: Vec<usize>,
}

impl Variable {
    pub fn new(id: usize, desired_position: f64, weight: f64, scale: f64) -> Self {
        Self {
            id,
            desired_position,
            final_position: desired_position,
            weight,
            scale,
            offset: 0.0,
            block: None,
            visited: false,
            fixed_desired_position: false,
            in_constraints: Vec::new(),
            out_constraints: Vec::new(),
        }
    }

    /// Derivative of objective function with respect to this variable.
    ///
    /// dfdv = 2 * weight * (position - desired_position)
    ///
    /// C++ ref: Variable::dfdv()
    pub fn dfdv(&self, block_posn: f64, block_scale: f64) -> f64 {
        let pos = self.position(block_posn, block_scale);
        2.0 * self.weight * (pos - self.desired_position)
    }

    /// Compute position from block state.
    ///
    /// C++ ref: Variable::position()
    pub fn position(&self, block_posn: f64, block_scale: f64) -> f64 {
        (block_scale * block_posn + self.offset) / self.scale
    }

    /// Unscaled position (only valid when scale == 1).
    ///
    /// C++ ref: Variable::unscaledPosition()
    pub fn unscaled_position(&self, block_posn: f64) -> f64 {
        debug_assert!((self.scale - 1.0).abs() < f64::EPSILON);
        block_posn + self.offset
    }
}

impl fmt::Debug for Variable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Var(id={}, desired={})", self.id, self.desired_position)
    }
}

impl fmt::Display for Variable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.block.is_some() {
            write!(f, "({}=<in block>)", self.id)
        } else {
            write!(f, "({}={})", self.id, self.desired_position)
        }
    }
}
