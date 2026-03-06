//! Yale/CSR sparse matrix for Laplacian storage.
//!
//! C++ ref: libcola/sparse_matrix.h

use std::collections::BTreeMap;

/// Map-based sparse matrix builder.
///
/// Entries are stored in a `BTreeMap` keyed by `(row, col)`, which preserves
/// insertion-order-independent sorted iteration — matching the C++ `std::map`
/// behaviour that the CSR construction relies on.
pub struct SparseMap {
    pub n: usize,
    lookup: BTreeMap<(usize, usize), f64>,
}

impl SparseMap {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            lookup: BTreeMap::new(),
        }
    }

    pub fn set(&mut self, i: usize, j: usize, val: f64) {
        self.lookup.insert((i, j), val);
    }

    /// Returns the value at `(i, j)`, or `0.0` if absent.
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.lookup.get(&(i, j)).copied().unwrap_or(0.0)
    }

    pub fn non_zero_count(&self) -> usize {
        self.lookup.len()
    }

    pub fn resize(&mut self, n: usize) {
        self.n = n;
    }

    pub fn clear(&mut self) {
        self.lookup.clear();
    }
}

/// Yale / Compressed Sparse Row (CSR) matrix.
///
/// Built once from a [`SparseMap`] and then used for repeated sparse
/// matrix-vector products (`right_multiply`).
pub struct SparseMatrix {
    n: usize,
    /// Non-zero values (length == number of stored entries).
    a: Vec<f64>,
    /// Row pointers — `ia[i]..ia[i+1]` indexes into `a` and `ja` for row `i`.
    /// Length is `n + 1`.
    ia: Vec<usize>,
    /// Column indices corresponding to each entry in `a`.
    ja: Vec<usize>,
    /// Copy of the original map for O(log n) `get` lookups.
    lookup: BTreeMap<(usize, usize), f64>,
}

impl SparseMatrix {
    /// Construct a CSR matrix from a [`SparseMap`].
    ///
    /// The BTreeMap iteration order is `(row, col)` ascending, which is
    /// exactly what CSR construction needs.
    pub fn from_sparse_map(map: &SparseMap) -> Self {
        let n = map.n;
        let nz = map.non_zero_count();

        let mut a = Vec::with_capacity(nz);
        let mut ia = vec![0usize; n + 1];
        let mut ja = Vec::with_capacity(nz);

        // Sentinel: no row seen yet.
        const NO_ROW: usize = usize::MAX;
        let mut last_row: usize = NO_ROW;

        for (&(row, col), &val) in &map.lookup {
            // When the row advances, fill ia entries for every row from
            // (last_row+1) through `row` (inclusive).
            if last_row == NO_ROW || row != last_row {
                let start = if last_row == NO_ROW { 0 } else { last_row + 1 };
                for r in start..=row {
                    ia[r] = a.len();
                }
                last_row = row;
            }
            a.push(val);
            ja.push(col);
        }

        // Fill remaining ia entries with nz (total number of stored values).
        let fill_start = if last_row == NO_ROW { 0 } else { last_row + 1 };
        for r in fill_start..=n {
            ia[r] = a.len();
        }

        Self {
            n,
            a,
            ia,
            ja,
            lookup: map.lookup.clone(),
        }
    }

    /// Sparse matrix-vector product: `result = self * v`.
    ///
    /// Both `v` and `result` must have length `>= self.n`.
    pub fn right_multiply(&self, v: &[f64], result: &mut [f64]) {
        for i in 0..self.n {
            let mut sum = 0.0;
            for idx in self.ia[i]..self.ia[i + 1] {
                sum += self.a[idx] * v[self.ja[idx]];
            }
            result[i] = sum;
        }
    }

    /// Returns the value at `(i, j)`, or `0.0` if absent.
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.lookup.get(&(i, j)).copied().unwrap_or(0.0)
    }

    pub fn row_size(&self) -> usize {
        self.n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------
    // 1. SparseMap basics
    // ---------------------------------------------------------------

    #[test]
    fn sparse_map_new_is_empty() {
        let m = SparseMap::new(5);
        assert_eq!(m.n, 5);
        assert_eq!(m.non_zero_count(), 0);
    }

    #[test]
    fn sparse_map_set_and_get() {
        let mut m = SparseMap::new(3);
        m.set(0, 1, 4.5);
        m.set(2, 2, -1.0);
        assert_eq!(m.get(0, 1), 4.5);
        assert_eq!(m.get(2, 2), -1.0);
        assert_eq!(m.non_zero_count(), 2);
    }

    #[test]
    fn sparse_map_overwrite() {
        let mut m = SparseMap::new(3);
        m.set(1, 1, 10.0);
        m.set(1, 1, 20.0);
        assert_eq!(m.get(1, 1), 20.0);
        assert_eq!(m.non_zero_count(), 1);
    }

    #[test]
    fn sparse_map_missing_returns_zero() {
        let m = SparseMap::new(4);
        assert_eq!(m.get(3, 3), 0.0);
    }

    #[test]
    fn sparse_map_clear() {
        let mut m = SparseMap::new(2);
        m.set(0, 0, 1.0);
        m.set(1, 1, 2.0);
        m.clear();
        assert_eq!(m.non_zero_count(), 0);
        assert_eq!(m.get(0, 0), 0.0);
    }

    #[test]
    fn sparse_map_resize() {
        let mut m = SparseMap::new(2);
        assert_eq!(m.n, 2);
        m.resize(10);
        assert_eq!(m.n, 10);
    }

    // ---------------------------------------------------------------
    // 2. SparseMatrix construction
    // ---------------------------------------------------------------

    #[test]
    fn from_empty_map() {
        let m = SparseMap::new(3);
        let sm = SparseMatrix::from_sparse_map(&m);
        assert_eq!(sm.row_size(), 3);
        assert_eq!(sm.a.len(), 0);
        assert_eq!(sm.ja.len(), 0);
        // All row pointers should be 0.
        assert!(sm.ia.iter().all(|&v| v == 0));
    }

    #[test]
    fn from_diagonal_map() {
        let n = 4;
        let mut m = SparseMap::new(n);
        for i in 0..n {
            m.set(i, i, (i + 1) as f64);
        }
        let sm = SparseMatrix::from_sparse_map(&m);
        assert_eq!(sm.a.len(), n);
        // Each row has exactly one entry.
        for i in 0..n {
            assert_eq!(sm.ia[i + 1] - sm.ia[i], 1);
        }
    }

    #[test]
    fn from_full_row() {
        // Row 0 has entries in every column.
        let n = 3;
        let mut m = SparseMap::new(n);
        for j in 0..n {
            m.set(0, j, (j + 1) as f64);
        }
        let sm = SparseMatrix::from_sparse_map(&m);
        assert_eq!(sm.ia[0], 0);
        assert_eq!(sm.ia[1], n); // all 3 entries in row 0
        assert_eq!(sm.ia[2], n); // rows 1 and 2 empty
        assert_eq!(sm.ia[3], n);
    }

    // ---------------------------------------------------------------
    // 3. right_multiply
    // ---------------------------------------------------------------

    #[test]
    fn right_multiply_identity() {
        // Build a 3x3 identity matrix.
        let n = 3;
        let mut m = SparseMap::new(n);
        for i in 0..n {
            m.set(i, i, 1.0);
        }
        let sm = SparseMatrix::from_sparse_map(&m);

        let v = vec![10.0, 20.0, 30.0];
        let mut result = vec![0.0; n];
        sm.right_multiply(&v, &mut result);
        assert_eq!(result, v);
    }

    #[test]
    fn right_multiply_diagonal_scaling() {
        let n = 3;
        let mut m = SparseMap::new(n);
        m.set(0, 0, 2.0);
        m.set(1, 1, 3.0);
        m.set(2, 2, 4.0);
        let sm = SparseMatrix::from_sparse_map(&m);

        let v = vec![1.0, 2.0, 3.0];
        let mut result = vec![0.0; n];
        sm.right_multiply(&v, &mut result);
        assert_eq!(result, vec![2.0, 6.0, 12.0]);
    }

    #[test]
    fn right_multiply_general() {
        // Matrix:
        //  [1 2 0]
        //  [0 3 4]
        //  [5 0 6]
        let n = 3;
        let mut m = SparseMap::new(n);
        m.set(0, 0, 1.0);
        m.set(0, 1, 2.0);
        m.set(1, 1, 3.0);
        m.set(1, 2, 4.0);
        m.set(2, 0, 5.0);
        m.set(2, 2, 6.0);
        let sm = SparseMatrix::from_sparse_map(&m);

        let v = vec![1.0, 2.0, 3.0];
        let mut result = vec![0.0; n];
        sm.right_multiply(&v, &mut result);
        // row 0: 1*1 + 2*2 = 5
        // row 1: 3*2 + 4*3 = 18
        // row 2: 5*1 + 6*3 = 23
        assert_eq!(result, vec![5.0, 18.0, 23.0]);
    }

    // ---------------------------------------------------------------
    // 4. SparseMatrix::get
    // ---------------------------------------------------------------

    #[test]
    fn get_existing_entry() {
        let mut m = SparseMap::new(3);
        m.set(1, 2, 7.5);
        let sm = SparseMatrix::from_sparse_map(&m);
        assert_eq!(sm.get(1, 2), 7.5);
    }

    #[test]
    fn get_missing_entry() {
        let mut m = SparseMap::new(3);
        m.set(0, 0, 1.0);
        let sm = SparseMatrix::from_sparse_map(&m);
        assert_eq!(sm.get(1, 2), 0.0);
    }

    #[test]
    fn get_matches_sparse_map() {
        let n = 4;
        let mut m = SparseMap::new(n);
        m.set(0, 1, 3.0);
        m.set(2, 3, 9.0);
        m.set(3, 0, -1.0);
        let sm = SparseMatrix::from_sparse_map(&m);
        for i in 0..n {
            for j in 0..n {
                assert_eq!(sm.get(i, j), m.get(i, j),
                    "mismatch at ({}, {})", i, j);
            }
        }
    }

    // ---------------------------------------------------------------
    // 5. Edge cases
    // ---------------------------------------------------------------

    #[test]
    fn zero_dimension_matrix() {
        let m = SparseMap::new(0);
        let sm = SparseMatrix::from_sparse_map(&m);
        assert_eq!(sm.row_size(), 0);
        assert_eq!(sm.a.len(), 0);
        // ia should have length 1 (n+1 = 0+1).
        assert_eq!(sm.ia.len(), 1);
        assert_eq!(sm.ia[0], 0);
        // right_multiply on empty should be a no-op.
        let v: Vec<f64> = vec![];
        let mut result: Vec<f64> = vec![];
        sm.right_multiply(&v, &mut result);
    }

    #[test]
    fn single_element_matrix() {
        let mut m = SparseMap::new(1);
        m.set(0, 0, 42.0);
        let sm = SparseMatrix::from_sparse_map(&m);
        assert_eq!(sm.row_size(), 1);
        assert_eq!(sm.get(0, 0), 42.0);

        let v = vec![2.0];
        let mut result = vec![0.0];
        sm.right_multiply(&v, &mut result);
        assert_eq!(result, vec![84.0]);
    }

    // ---------------------------------------------------------------
    // 6. Cross-validation: sparse multiply vs dense multiply
    // ---------------------------------------------------------------

    #[test]
    fn right_multiply_matches_dense() {
        // Build an arbitrary 4x4 sparse matrix, perform multiplication
        // both via CSR and via a naive dense loop, and compare.
        let n = 4;
        let entries: &[(usize, usize, f64)] = &[
            (0, 0, 2.0), (0, 2, -1.0),
            (1, 1, 5.0), (1, 3, 3.0),
            (2, 0, 1.0), (2, 2, 4.0),
            (3, 1, -2.0), (3, 3, 7.0),
        ];

        let mut m = SparseMap::new(n);
        for &(i, j, val) in entries {
            m.set(i, j, val);
        }
        let sm = SparseMatrix::from_sparse_map(&m);

        let v = vec![1.0, -1.0, 2.0, 0.5];
        let mut sparse_result = vec![0.0; n];
        sm.right_multiply(&v, &mut sparse_result);

        // Dense multiply from the map.
        let mut dense_result = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                dense_result[i] += m.get(i, j) * v[j];
            }
        }

        for i in 0..n {
            assert!(
                (sparse_result[i] - dense_result[i]).abs() < 1e-12,
                "row {}: sparse={} dense={}", i, sparse_result[i], dense_result[i],
            );
        }
    }

    #[test]
    fn sparse_rows_with_gaps() {
        // Matrix where some interior rows are entirely empty.
        // Row 0: has entries. Row 1: empty. Row 2: empty. Row 3: has entries.
        let n = 4;
        let mut m = SparseMap::new(n);
        m.set(0, 0, 1.0);
        m.set(3, 3, 2.0);
        let sm = SparseMatrix::from_sparse_map(&m);

        let v = vec![3.0, 4.0, 5.0, 6.0];
        let mut result = vec![0.0; n];
        sm.right_multiply(&v, &mut result);
        assert_eq!(result, vec![3.0, 0.0, 0.0, 12.0]);

        // Verify ia correctly bridges the empty rows.
        assert_eq!(sm.ia[0], 0); // row 0 starts at 0
        assert_eq!(sm.ia[1], 1); // row 0 had 1 entry
        assert_eq!(sm.ia[2], 1); // row 1 empty
        assert_eq!(sm.ia[3], 1); // row 2 empty
        assert_eq!(sm.ia[4], 2); // row 3 had 1 entry
    }
}
