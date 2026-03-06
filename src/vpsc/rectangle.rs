//! Rectangle type and overlap removal functions.
//!
//! C++ ref: libvpsc/rectangle.h, libvpsc/rectangle.cpp

use std::fmt;

use crate::vpsc::constraint::Constraint;
use crate::vpsc::solver::Solver;
use crate::vpsc::variable::Variable;

/// Small extra gap to avoid numerical imprecision in overlap detection.
const EXTRA_GAP: f64 = 1e-3;


/// Dimension indicator.
///
/// C++ ref: vpsc::Dim
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dim {
    Horizontal = 0,
    Vertical = 1,
}

impl Dim {
    /// Returns the conjugate dimension.
    pub fn conjugate(self) -> Dim {
        match self {
            Dim::Horizontal => Dim::Vertical,
            Dim::Vertical => Dim::Horizontal,
        }
    }
}

/// A rectangle representing a fixed-size shape that may be moved to
/// prevent overlaps and satisfy constraints.
///
/// C++ ref: vpsc::Rectangle
#[derive(Clone)]
pub struct Rectangle {
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
    allow_overlap: bool,
    x_border: f64,
    y_border: f64,
}

impl Rectangle {
    /// Create a rectangle from min/max coordinates.
    ///
    /// C++ ref: Rectangle::Rectangle(double, double, double, double, bool)
    pub fn new(min_x: f64, max_x: f64, min_y: f64, max_y: f64) -> Self {
        debug_assert!(min_x < max_x, "min_x ({}) must be < max_x ({})", min_x, max_x);
        debug_assert!(min_y < max_y, "min_y ({}) must be < max_y ({})", min_y, max_y);
        Self {
            min_x,
            max_x,
            min_y,
            max_y,
            allow_overlap: false,
            x_border: 0.0,
            y_border: 0.0,
        }
    }

    /// Create an invalid rectangle (used as a sentinel).
    pub fn invalid() -> Self {
        Self {
            min_x: 1.0,
            max_x: -1.0,
            min_y: 1.0,
            max_y: -1.0,
            allow_overlap: false,
            x_border: 0.0,
            y_border: 0.0,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.min_x <= self.max_x && self.min_y <= self.max_y
    }

    pub fn get_min_x(&self) -> f64 { self.min_x - self.x_border }
    pub fn get_max_x(&self) -> f64 { self.max_x + self.x_border }
    pub fn get_min_y(&self) -> f64 { self.min_y - self.y_border }
    pub fn get_max_y(&self) -> f64 { self.max_y + self.y_border }

    pub fn get_min_d(&self, d: Dim) -> f64 {
        match d {
            Dim::Horizontal => self.get_min_x(),
            Dim::Vertical => self.get_min_y(),
        }
    }

    pub fn get_max_d(&self, d: Dim) -> f64 {
        match d {
            Dim::Horizontal => self.get_max_x(),
            Dim::Vertical => self.get_max_y(),
        }
    }

    pub fn set_min_d(&mut self, d: Dim, val: f64) {
        match d {
            Dim::Horizontal => self.min_x = val,
            Dim::Vertical => self.min_y = val,
        }
    }

    pub fn set_max_d(&mut self, d: Dim, val: f64) {
        match d {
            Dim::Horizontal => self.max_x = val,
            Dim::Vertical => self.max_y = val,
        }
    }

    pub fn width(&self) -> f64 { self.get_max_x() - self.get_min_x() }
    pub fn height(&self) -> f64 { self.get_max_y() - self.get_min_y() }

    pub fn length(&self, d: Dim) -> f64 {
        match d {
            Dim::Horizontal => self.width(),
            Dim::Vertical => self.height(),
        }
    }

    pub fn centre_x(&self) -> f64 { self.get_min_x() + self.width() / 2.0 }
    pub fn centre_y(&self) -> f64 { self.get_min_y() + self.height() / 2.0 }

    pub fn centre_d(&self, d: Dim) -> f64 {
        self.get_min_d(d) + self.length(d) / 2.0
    }

    pub fn set_width(&mut self, w: f64) {
        self.max_x = self.min_x + w - 2.0 * self.x_border;
    }

    pub fn set_height(&mut self, h: f64) {
        self.max_y = self.min_y + h - 2.0 * self.y_border;
    }

    pub fn move_centre_x(&mut self, x: f64) {
        self.move_min_x(x - self.width() / 2.0);
    }

    pub fn move_centre_y(&mut self, y: f64) {
        self.move_min_y(y - self.height() / 2.0);
    }

    pub fn move_centre_d(&mut self, d: Dim, p: f64) {
        match d {
            Dim::Horizontal => self.move_centre_x(p),
            Dim::Vertical => self.move_centre_y(p),
        }
    }

    pub fn move_centre(&mut self, x: f64, y: f64) {
        self.move_centre_x(x);
        self.move_centre_y(y);
    }

    fn move_min_x(&mut self, x: f64) {
        let w = self.width();
        self.min_x = x + self.x_border;
        self.max_x = x + w - self.x_border;
    }

    fn move_min_y(&mut self, y: f64) {
        let h = self.height();
        self.max_y = y + h - self.y_border;
        self.min_y = y + self.y_border;
    }

    /// Amount of overlap in X with another rectangle.
    pub fn overlap_x(&self, other: &Rectangle) -> f64 {
        let ux = self.centre_x();
        let vx = other.centre_x();
        if ux <= vx && other.get_min_x() < self.get_max_x() {
            return self.get_max_x() - other.get_min_x();
        }
        if vx <= ux && self.get_min_x() < other.get_max_x() {
            return other.get_max_x() - self.get_min_x();
        }
        0.0
    }

    /// Amount of overlap in Y with another rectangle.
    pub fn overlap_y(&self, other: &Rectangle) -> f64 {
        let uy = self.centre_y();
        let vy = other.centre_y();
        if uy <= vy && other.get_min_y() < self.get_max_y() {
            return self.get_max_y() - other.get_min_y();
        }
        if vy <= uy && self.get_min_y() < other.get_max_y() {
            return other.get_max_y() - self.get_min_y();
        }
        0.0
    }

    /// Overlap in a given dimension.
    pub fn overlap_d(&self, d: Dim, other: &Rectangle) -> f64 {
        match d {
            Dim::Horizontal => self.overlap_x(other),
            Dim::Vertical => self.overlap_y(other),
        }
    }

    pub fn offset(&mut self, dx: f64, dy: f64) {
        self.min_x += dx;
        self.max_x += dx;
        self.min_y += dy;
        self.max_y += dy;
    }

    pub fn inside(&self, x: f64, y: f64) -> bool {
        x > self.get_min_x() && x < self.get_max_x()
            && y > self.get_min_y() && y < self.get_max_y()
    }

    /// Union of two rectangles.
    pub fn union_with(&self, rhs: &Rectangle) -> Rectangle {
        if !self.is_valid() {
            return rhs.clone();
        }
        if !rhs.is_valid() {
            return self.clone();
        }
        Rectangle::new(
            self.get_min_x().min(rhs.get_min_x()),
            self.get_max_x().max(rhs.get_max_x()),
            self.get_min_y().min(rhs.get_min_y()),
            self.get_max_y().max(rhs.get_max_y()),
        )
    }

    /// Reset dimensions in one axis.
    pub fn reset(&mut self, d: Dim, min: f64, max: f64) {
        match d {
            Dim::Horizontal => {
                self.min_x = min;
                self.max_x = max;
            }
            Dim::Vertical => {
                self.min_y = min;
                self.max_y = max;
            }
        }
    }

    fn set_x_border(&mut self, x: f64) { self.x_border = x; }
    fn set_y_border(&mut self, y: f64) { self.y_border = y; }
}

impl fmt::Debug for Rectangle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Rect({:.2}, {:.2}, {:.2}, {:.2})",
            self.min_x, self.max_x, self.min_y, self.max_y
        )
    }
}

impl fmt::Display for Rectangle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Rectangle[({},{}),({},{})]",
            self.get_min_x(),
            self.get_min_y(),
            self.get_max_x(),
            self.get_max_y()
        )
    }
}

// ---------------------------------------------------------------------------
// Overlap removal constraint generation
// ---------------------------------------------------------------------------

/// Scanline node for overlap removal.
#[derive(Clone)]
struct ScanNode {
    var_idx: usize,
    rect_idx: usize,
    pos: f64,
    first_above: Option<usize>,
    first_below: Option<usize>,
}

/// Scanline event.
#[derive(Clone)]
struct Event {
    is_open: bool,
    node_idx: usize,
    pos: f64,
}

/// Generate X-axis separation constraints for overlap removal.
///
/// Uses a scanline algorithm sweeping in Y to find horizontal neighbours.
///
/// C++ ref: generateXConstraints
pub fn generate_x_constraints(
    rects: &[Rectangle],
    use_neighbour_lists: bool,
) -> (Vec<Variable>, Vec<Constraint>) {
    let n = rects.len();
    let vars: Vec<Variable> = rects
        .iter()
        .enumerate()
        .map(|(i, r)| Variable::new(i, r.centre_x(), 1.0, 1.0))
        .collect();

    let mut constraints = Vec::new();

    if n == 0 {
        return (vars, constraints);
    }

    // Create scan nodes and events
    let mut nodes: Vec<ScanNode> = Vec::with_capacity(n);
    let mut events: Vec<Event> = Vec::with_capacity(2 * n);

    for i in 0..n {
        let node_idx = nodes.len();
        nodes.push(ScanNode {
            var_idx: i,
            rect_idx: i,
            pos: rects[i].centre_x(),
            first_above: None,
            first_below: None,
        });
        events.push(Event {
            is_open: true,
            node_idx,
            pos: rects[i].get_min_y(),
        });
        events.push(Event {
            is_open: false,
            node_idx,
            pos: rects[i].get_max_y(),
        });
    }

    // Sort events by position (opens before closes at same position)
    events.sort_by(|a, b| {
        a.pos
            .partial_cmp(&b.pos)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(if a.is_open && !b.is_open {
                std::cmp::Ordering::Less
            } else if !a.is_open && b.is_open {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            })
    });

    // Scanline: ordered set of active nodes by X position
    // We use a Vec and maintain sorted order
    let mut scanline: Vec<usize> = Vec::new(); // node indices, sorted by pos

    for event in &events {
        let v = event.node_idx;
        if event.is_open {
            // Insert into scanline maintaining sort order
            let insert_pos = scanline
                .partition_point(|&n| nodes[n].pos < nodes[v].pos
                    || (nodes[n].pos == nodes[v].pos && n < v));
            scanline.insert(insert_pos, v);

            if !use_neighbour_lists {
                let scan_pos = scanline.iter().position(|&n| n == v).unwrap();
                if scan_pos > 0 {
                    let above = scanline[scan_pos - 1];
                    nodes[v].first_above = Some(above);
                    nodes[above].first_below = Some(v);
                }
                if scan_pos + 1 < scanline.len() {
                    let below = scanline[scan_pos + 1];
                    nodes[v].first_below = Some(below);
                    nodes[below].first_above = Some(v);
                }
            }
        } else {
            // Close event
            if !use_neighbour_lists {
                let above = nodes[v].first_above;
                let below = nodes[v].first_below;

                if let Some(a) = above {
                    let sep = (rects[nodes[v].rect_idx].width()
                        + rects[nodes[a].rect_idx].width())
                        / 2.0;
                    constraints.push(Constraint::new(nodes[a].var_idx, nodes[v].var_idx, sep, false));
                    nodes[a].first_below = below;
                }
                if let Some(b) = below {
                    let sep = (rects[nodes[v].rect_idx].width()
                        + rects[nodes[b].rect_idx].width())
                        / 2.0;
                    constraints.push(Constraint::new(nodes[v].var_idx, nodes[b].var_idx, sep, false));
                    nodes[b].first_above = above;
                }
            }

            scanline.retain(|&n| n != v);
        }
    }

    (vars, constraints)
}

/// Generate Y-axis separation constraints for overlap removal.
///
/// C++ ref: generateYConstraints
pub fn generate_y_constraints(rects: &[Rectangle]) -> (Vec<Variable>, Vec<Constraint>) {
    let n = rects.len();
    let vars: Vec<Variable> = rects
        .iter()
        .enumerate()
        .map(|(i, r)| Variable::new(i, r.centre_y(), 1.0, 1.0))
        .collect();

    let mut constraints = Vec::new();

    if n == 0 {
        return (vars, constraints);
    }

    let mut nodes: Vec<ScanNode> = Vec::with_capacity(n);
    let mut events: Vec<Event> = Vec::with_capacity(2 * n);

    for i in 0..n {
        let node_idx = nodes.len();
        nodes.push(ScanNode {
            var_idx: i,
            rect_idx: i,
            pos: rects[i].centre_y(),
            first_above: None,
            first_below: None,
        });
        events.push(Event {
            is_open: true,
            node_idx,
            pos: rects[i].get_min_x(),
        });
        events.push(Event {
            is_open: false,
            node_idx,
            pos: rects[i].get_max_x(),
        });
    }

    events.sort_by(|a, b| {
        a.pos
            .partial_cmp(&b.pos)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(if a.is_open && !b.is_open {
                std::cmp::Ordering::Less
            } else if !a.is_open && b.is_open {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            })
    });

    let mut scanline: Vec<usize> = Vec::new();

    for event in &events {
        let v = event.node_idx;
        if event.is_open {
            let insert_pos = scanline
                .partition_point(|&n| nodes[n].pos < nodes[v].pos
                    || (nodes[n].pos == nodes[v].pos && n < v));
            scanline.insert(insert_pos, v);

            let scan_pos = scanline.iter().position(|&n| n == v).unwrap();
            if scan_pos > 0 {
                let above = scanline[scan_pos - 1];
                nodes[v].first_above = Some(above);
                nodes[above].first_below = Some(v);
            }
            if scan_pos + 1 < scanline.len() {
                let below = scanline[scan_pos + 1];
                nodes[v].first_below = Some(below);
                nodes[below].first_above = Some(v);
            }
        } else {
            let above = nodes[v].first_above;
            let below = nodes[v].first_below;

            if let Some(a) = above {
                let sep = (rects[nodes[v].rect_idx].height()
                    + rects[nodes[a].rect_idx].height())
                    / 2.0;
                constraints.push(Constraint::new(nodes[a].var_idx, nodes[v].var_idx, sep, false));
                nodes[a].first_below = below;
            }
            if let Some(b) = below {
                let sep = (rects[nodes[v].rect_idx].height()
                    + rects[nodes[b].rect_idx].height())
                    / 2.0;
                constraints.push(Constraint::new(nodes[v].var_idx, nodes[b].var_idx, sep, false));
                nodes[b].first_above = above;
            }

            scanline.retain(|&n| n != v);
        }
    }

    (vars, constraints)
}

/// Remove overlaps between rectangles.
///
/// Moves rectangles to remove all overlaps using a heuristic that attempts
/// to minimize total movement. Applies VPSC horizontally, then vertically.
///
/// C++ ref: removeoverlaps(Rectangles&)
pub fn remove_overlaps(rects: &mut [Rectangle]) {
    let fixed = std::collections::HashSet::new();
    remove_overlaps_with_fixed(rects, &fixed, true);
}

/// Remove overlaps with some rectangles fixed in place.
///
/// C++ ref: removeoverlaps(Rectangles&, set<unsigned>&, bool)
pub fn remove_overlaps_with_fixed(
    rects: &mut [Rectangle],
    fixed: &std::collections::HashSet<usize>,
    third_pass: bool,
) {
    let n = rects.len();
    if n == 0 {
        return;
    }

    let original_x_border = 0.0;
    let original_y_border = 0.0;

    // Save initial X positions for optional third pass
    let init_x: Vec<f64> = if third_pass {
        rects.iter().map(|r| r.centre_x()).collect()
    } else {
        Vec::new()
    };

    // --- First horizontal pass ---
    // Add extra gap to avoid numerical issues
    for r in rects.iter_mut() {
        r.set_x_border(original_x_border + EXTRA_GAP);
        r.set_y_border(original_y_border + EXTRA_GAP);
    }

    let fixed_weight = 10000.0;
    let default_weight = 1.0;

    {
        let (mut vars, cs) = generate_x_constraints(rects, true);
        for v in &mut vars {
            if fixed.contains(&v.id) {
                v.weight = fixed_weight;
            } else {
                v.weight = default_weight;
            }
        }

        let mut solver = Solver::new(vars, cs);
        if solver.solve().is_ok() {
            let positions = solver.final_positions();
            for (i, r) in rects.iter_mut().enumerate() {
                debug_assert!(positions[i].is_finite());
                r.move_centre_x(positions[i]);
            }
        }
    }

    // --- Vertical pass ---
    // Remove extra gap so adjacent things from X pass aren't considered overlapping
    for r in rects.iter_mut() {
        r.set_x_border(original_x_border);
    }

    {
        let (mut vars, cs) = generate_y_constraints(rects);
        for v in &mut vars {
            if fixed.contains(&v.id) {
                v.weight = fixed_weight;
            } else {
                v.weight = default_weight;
            }
        }

        let mut solver = Solver::new(vars, cs);
        if solver.solve().is_ok() {
            let positions = solver.final_positions();
            for (i, r) in rects.iter_mut().enumerate() {
                debug_assert!(positions[i].is_finite());
                r.move_centre_y(positions[i]);
            }
        }
    }

    for r in rects.iter_mut() {
        r.set_y_border(original_y_border);
    }

    // --- Optional third horizontal pass ---
    if third_pass {
        for r in rects.iter_mut() {
            r.set_x_border(original_x_border + EXTRA_GAP);
        }

        // Reset X positions to original
        for (i, r) in rects.iter_mut().enumerate() {
            r.move_centre_x(init_x[i]);
        }

        let (mut vars, cs) = generate_x_constraints(rects, false);
        for v in &mut vars {
            if fixed.contains(&v.id) {
                v.weight = fixed_weight;
            } else {
                v.weight = default_weight;
            }
        }

        let mut solver = Solver::new(vars, cs);
        if solver.solve().is_ok() {
            let positions = solver.final_positions();
            for (i, r) in rects.iter_mut().enumerate() {
                debug_assert!(positions[i].is_finite());
                r.move_centre_x(positions[i]);
            }
        }

        for r in rects.iter_mut() {
            r.set_x_border(original_x_border);
        }
    }
}

/// Check that no rectangles overlap (useful for assertions).
///
/// C++ ref: noRectangleOverlaps
pub fn no_rectangle_overlaps(rects: &[Rectangle]) -> bool {
    for i in 0..rects.len() {
        for j in (i + 1)..rects.len() {
            if rects[i].overlap_x(&rects[j]) > 0.0 && rects[i].overlap_y(&rects[j]) > 0.0 {
                return false;
            }
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ===================================================================
    // Category 1: Rectangle construction and accessors
    // ===================================================================

    #[test]
    fn test_rectangle_new() {
        let r = Rectangle::new(0.0, 10.0, 0.0, 20.0);
        assert_eq!(r.get_min_x(), 0.0);
        assert_eq!(r.get_max_x(), 10.0);
        assert_eq!(r.get_min_y(), 0.0);
        assert_eq!(r.get_max_y(), 20.0);
    }

    #[test]
    fn test_rectangle_dimensions() {
        let r = Rectangle::new(0.0, 10.0, 0.0, 20.0);
        assert_eq!(r.width(), 10.0);
        assert_eq!(r.height(), 20.0);
    }

    #[test]
    fn test_rectangle_centre() {
        let r = Rectangle::new(0.0, 10.0, 0.0, 20.0);
        assert_eq!(r.centre_x(), 5.0);
        assert_eq!(r.centre_y(), 10.0);
    }

    #[test]
    fn test_rectangle_invalid() {
        let r = Rectangle::invalid();
        assert!(!r.is_valid());
    }

    #[test]
    fn test_rectangle_valid() {
        let r = Rectangle::new(0.0, 10.0, 0.0, 10.0);
        assert!(r.is_valid());
    }

    // ===================================================================
    // Category 2: Rectangle dimension accessors
    // ===================================================================

    #[test]
    fn test_get_min_max_d() {
        let r = Rectangle::new(1.0, 5.0, 2.0, 8.0);
        assert_eq!(r.get_min_d(Dim::Horizontal), 1.0);
        assert_eq!(r.get_max_d(Dim::Horizontal), 5.0);
        assert_eq!(r.get_min_d(Dim::Vertical), 2.0);
        assert_eq!(r.get_max_d(Dim::Vertical), 8.0);
    }

    #[test]
    fn test_length_d() {
        let r = Rectangle::new(0.0, 10.0, 0.0, 20.0);
        assert_eq!(r.length(Dim::Horizontal), 10.0);
        assert_eq!(r.length(Dim::Vertical), 20.0);
    }

    #[test]
    fn test_centre_d() {
        let r = Rectangle::new(2.0, 6.0, 4.0, 10.0);
        assert_eq!(r.centre_d(Dim::Horizontal), 4.0);
        assert_eq!(r.centre_d(Dim::Vertical), 7.0);
    }

    // ===================================================================
    // Category 3: Rectangle movement
    // ===================================================================

    #[test]
    fn test_move_centre_x() {
        let mut r = Rectangle::new(0.0, 10.0, 0.0, 10.0);
        r.move_centre_x(20.0);
        assert!((r.centre_x() - 20.0).abs() < 1e-6);
        assert!((r.width() - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_move_centre_y() {
        let mut r = Rectangle::new(0.0, 10.0, 0.0, 10.0);
        r.move_centre_y(30.0);
        assert!((r.centre_y() - 30.0).abs() < 1e-6);
        assert!((r.height() - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_offset() {
        let mut r = Rectangle::new(0.0, 10.0, 0.0, 10.0);
        r.offset(5.0, -3.0);
        assert_eq!(r.get_min_x(), 5.0);
        assert_eq!(r.get_max_x(), 15.0);
        assert_eq!(r.get_min_y(), -3.0);
        assert_eq!(r.get_max_y(), 7.0);
    }

    // ===================================================================
    // Category 4: Overlap detection
    // ===================================================================

    #[test]
    fn test_overlap_x_overlapping() {
        let a = Rectangle::new(0.0, 10.0, 0.0, 10.0);
        let b = Rectangle::new(5.0, 15.0, 0.0, 10.0);
        assert!(a.overlap_x(&b) > 0.0);
    }

    #[test]
    fn test_overlap_x_no_overlap() {
        let a = Rectangle::new(0.0, 10.0, 0.0, 10.0);
        let b = Rectangle::new(20.0, 30.0, 0.0, 10.0);
        assert_eq!(a.overlap_x(&b), 0.0);
    }

    #[test]
    fn test_overlap_y_overlapping() {
        let a = Rectangle::new(0.0, 10.0, 0.0, 10.0);
        let b = Rectangle::new(0.0, 10.0, 5.0, 15.0);
        assert!(a.overlap_y(&b) > 0.0);
    }

    #[test]
    fn test_overlap_symmetric() {
        let a = Rectangle::new(0.0, 10.0, 0.0, 10.0);
        let b = Rectangle::new(5.0, 15.0, 3.0, 13.0);
        assert_eq!(a.overlap_x(&b), b.overlap_x(&a));
        assert_eq!(a.overlap_y(&b), b.overlap_y(&a));
    }

    // ===================================================================
    // Category 5: Inside test
    // ===================================================================

    #[test]
    fn test_inside_point() {
        let r = Rectangle::new(0.0, 10.0, 0.0, 10.0);
        assert!(r.inside(5.0, 5.0));
        assert!(!r.inside(15.0, 5.0));
        assert!(!r.inside(5.0, 15.0));
    }

    #[test]
    fn test_inside_boundary() {
        let r = Rectangle::new(0.0, 10.0, 0.0, 10.0);
        // Boundary points are NOT inside (strict inequality)
        assert!(!r.inside(0.0, 5.0));
        assert!(!r.inside(10.0, 5.0));
    }

    // ===================================================================
    // Category 6: Union
    // ===================================================================

    #[test]
    fn test_union_with() {
        let a = Rectangle::new(0.0, 5.0, 0.0, 5.0);
        let b = Rectangle::new(3.0, 10.0, 3.0, 10.0);
        let u = a.union_with(&b);
        assert_eq!(u.get_min_x(), 0.0);
        assert_eq!(u.get_max_x(), 10.0);
        assert_eq!(u.get_min_y(), 0.0);
        assert_eq!(u.get_max_y(), 10.0);
    }

    #[test]
    fn test_union_with_invalid() {
        let a = Rectangle::invalid();
        let b = Rectangle::new(3.0, 10.0, 3.0, 10.0);
        let u = a.union_with(&b);
        assert_eq!(u.get_min_x(), 3.0);
        assert_eq!(u.get_max_x(), 10.0);
    }

    // ===================================================================
    // Category 7: Dim operations
    // ===================================================================

    #[test]
    fn test_dim_conjugate() {
        assert_eq!(Dim::Horizontal.conjugate(), Dim::Vertical);
        assert_eq!(Dim::Vertical.conjugate(), Dim::Horizontal);
    }

    // ===================================================================
    // Category 8: No-overlap check
    // ===================================================================

    #[test]
    fn test_no_overlaps_non_overlapping() {
        let rects = vec![
            Rectangle::new(0.0, 5.0, 0.0, 5.0),
            Rectangle::new(10.0, 15.0, 0.0, 5.0),
        ];
        assert!(no_rectangle_overlaps(&rects));
    }

    #[test]
    fn test_no_overlaps_overlapping() {
        let rects = vec![
            Rectangle::new(0.0, 10.0, 0.0, 10.0),
            Rectangle::new(5.0, 15.0, 5.0, 15.0),
        ];
        assert!(!no_rectangle_overlaps(&rects));
    }

    // ===================================================================
    // Category 9: Overlap removal
    // ===================================================================

    #[test]
    fn test_remove_overlaps_non_overlapping() {
        let mut rects = vec![
            Rectangle::new(0.0, 5.0, 0.0, 5.0),
            Rectangle::new(100.0, 105.0, 100.0, 105.0),
        ];
        remove_overlaps(&mut rects);
        assert!(no_rectangle_overlaps(&rects));
    }

    #[test]
    fn test_remove_overlaps_identical_rects() {
        let mut rects = vec![
            Rectangle::new(0.0, 10.0, 0.0, 10.0),
            Rectangle::new(0.0, 10.0, 0.0, 10.0),
        ];
        remove_overlaps(&mut rects);
        assert!(no_rectangle_overlaps(&rects));
    }

    #[test]
    fn test_remove_overlaps_three_overlapping() {
        let mut rects = vec![
            Rectangle::new(0.0, 10.0, 0.0, 10.0),
            Rectangle::new(5.0, 15.0, 5.0, 15.0),
            Rectangle::new(3.0, 13.0, 3.0, 13.0),
        ];
        remove_overlaps(&mut rects);
        assert!(no_rectangle_overlaps(&rects));
    }

    #[test]
    fn test_remove_overlaps_preserves_dimensions() {
        let mut rects = vec![
            Rectangle::new(0.0, 10.0, 0.0, 20.0),
            Rectangle::new(5.0, 15.0, 5.0, 25.0),
        ];
        let orig_widths: Vec<f64> = rects.iter().map(|r| r.width()).collect();
        let orig_heights: Vec<f64> = rects.iter().map(|r| r.height()).collect();

        remove_overlaps(&mut rects);

        for (i, r) in rects.iter().enumerate() {
            assert!(
                (r.width() - orig_widths[i]).abs() < 1e-4,
                "Width changed for rect {}",
                i
            );
            assert!(
                (r.height() - orig_heights[i]).abs() < 1e-4,
                "Height changed for rect {}",
                i
            );
        }
    }

    // ===================================================================
    // Category 10: Constraint generation
    // ===================================================================

    #[test]
    fn test_generate_x_constraints_no_overlap() {
        let rects = vec![
            Rectangle::new(0.0, 5.0, 0.0, 5.0),
            Rectangle::new(100.0, 105.0, 0.0, 5.0),
        ];
        let (vars, cs) = generate_x_constraints(&rects, false);
        assert_eq!(vars.len(), 2);
        // These rects overlap in Y (both [0,5]) so should generate X constraints
        assert!(!cs.is_empty());
    }

    #[test]
    fn test_generate_y_constraints_overlapping() {
        let rects = vec![
            Rectangle::new(0.0, 10.0, 0.0, 10.0),
            Rectangle::new(5.0, 15.0, 5.0, 15.0),
        ];
        let (vars, cs) = generate_y_constraints(&rects);
        assert_eq!(vars.len(), 2);
        // Rects overlap in X ([0,10] and [5,15]) so should generate Y constraints
        assert!(!cs.is_empty());
    }

    #[test]
    fn test_generate_constraints_empty() {
        let rects: Vec<Rectangle> = vec![];
        let (vars, constraints) = generate_x_constraints(&rects, false);
        assert!(vars.is_empty());
        assert!(constraints.is_empty());
    }
}
