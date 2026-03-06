//! Conjugate gradient solver for dense symmetric positive-definite systems.
//!
//! Solves Ax = b where A is a dense n×n matrix stored in row-major order
//! as a flat Vec of length n².
//!
//! C++ ref: libcola/conjugate_gradient.h, libcola/conjugate_gradient.cpp

/// Dot product of two vectors.
///
/// C++ ref: inner()
pub fn inner(x: &[f64], y: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), y.len());
    x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
}

/// Dense matrix-vector multiply: result = A * vec.
///
/// A is stored row-major as a flat slice of length m*n.
///
/// C++ ref: matrix_times_vector()
fn matrix_times_vector(a: &[f64], vec: &[f64], result: &mut [f64]) {
    let n = vec.len();
    let m = result.len();
    debug_assert_eq!(a.len(), m * n);

    for i in 0..m {
        let row_start = i * n;
        let mut res = 0.0;
        for j in 0..n {
            res += a[row_start + j] * vec[j];
        }
        result[i] = res;
    }
}

/// Compute cost = 2bx - xAx.
///
/// C++ ref: compute_cost()
pub fn compute_cost(a: &[f64], b: &[f64], x: &[f64], n: usize) -> f64 {
    let cost = 2.0 * inner(b, x);
    let mut ax = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            ax[i] += a[i * n + j] * x[j];
        }
    }
    cost - inner(x, &ax)
}

/// Conjugate gradient solver for Ax = b.
///
/// Solves the symmetric positive-definite system Ax = b iteratively.
/// `x` should be initialized with an initial guess (can be zero).
/// `a` is a dense n×n matrix in row-major flat layout.
///
/// C++ ref: conjugate_gradient()
pub fn conjugate_gradient(a: &[f64], x: &mut [f64], b: &[f64], n: usize, tol: f64, max_iterations: usize) {
    debug_assert_eq!(a.len(), n * n);
    debug_assert_eq!(x.len(), n);
    debug_assert_eq!(b.len(), n);

    let mut ap = vec![0.0; n];
    let mut p = vec![0.0; n];
    let mut r = vec![0.0; n];

    // r = b - A*x
    matrix_times_vector(a, x, &mut ap);
    for i in 0..n {
        r[i] = b[i] - ap[i];
    }

    let mut r_r = inner(&r, &r);
    let tol_squared = tol * tol;
    let mut k = 0u32;

    while (k as usize) < max_iterations && r_r > tol_squared {
        k += 1;
        let r_r_new;

        if k == 1 {
            p.copy_from_slice(&r);
            r_r_new = r_r;
        } else {
            r_r_new = inner(&r, &r);
            if r_r_new < tol_squared {
                break;
            }
            let beta = r_r_new / r_r;
            for i in 0..n {
                p[i] = r[i] + beta * p[i];
            }
        }

        matrix_times_vector(a, &p, &mut ap);
        let alpha = r_r_new / inner(&p, &ap);

        for i in 0..n {
            x[i] += alpha * p[i];
        }
        for i in 0..n {
            r[i] -= alpha * ap[i];
        }

        r_r = r_r_new;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===================================================================
    // Category 1: inner product
    // ===================================================================

    #[test]
    fn test_inner_zero_vectors() {
        assert_eq!(inner(&[0.0, 0.0], &[0.0, 0.0]), 0.0);
    }

    #[test]
    fn test_inner_unit_vectors() {
        assert_eq!(inner(&[1.0, 0.0], &[0.0, 1.0]), 0.0); // orthogonal
        assert_eq!(inner(&[1.0, 0.0], &[1.0, 0.0]), 1.0); // parallel
    }

    #[test]
    fn test_inner_known_value() {
        // [1,2,3] . [4,5,6] = 4+10+18 = 32
        assert_eq!(inner(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
    }

    #[test]
    fn test_inner_negative_values() {
        assert_eq!(inner(&[-1.0, 2.0], &[3.0, -4.0]), -3.0 + (-8.0));
    }

    #[test]
    fn test_inner_single_element() {
        assert_eq!(inner(&[5.0], &[3.0]), 15.0);
    }

    #[test]
    fn test_inner_commutative() {
        let a = &[1.0, 2.0, 3.0];
        let b = &[4.0, 5.0, 6.0];
        assert_eq!(inner(a, b), inner(b, a));
    }

    // ===================================================================
    // Category 2: matrix_times_vector
    // ===================================================================

    #[test]
    fn test_mtv_identity() {
        // I * [3, 7] = [3, 7]
        let a = [1.0, 0.0, 0.0, 1.0];
        let v = [3.0, 7.0];
        let mut r = [0.0, 0.0];
        matrix_times_vector(&a, &v, &mut r);
        assert_eq!(r, [3.0, 7.0]);
    }

    #[test]
    fn test_mtv_scaling() {
        // diag(2, 3) * [4, 5] = [8, 15]
        let a = [2.0, 0.0, 0.0, 3.0];
        let v = [4.0, 5.0];
        let mut r = [0.0, 0.0];
        matrix_times_vector(&a, &v, &mut r);
        assert_eq!(r, [8.0, 15.0]);
    }

    #[test]
    fn test_mtv_full_matrix() {
        // [[1,2],[3,4]] * [5,6] = [17, 39]
        let a = [1.0, 2.0, 3.0, 4.0];
        let v = [5.0, 6.0];
        let mut r = [0.0, 0.0];
        matrix_times_vector(&a, &v, &mut r);
        assert_eq!(r, [17.0, 39.0]);
    }

    #[test]
    fn test_mtv_zero_matrix() {
        let a = [0.0; 4];
        let v = [1.0, 2.0];
        let mut r = [99.0, 99.0];
        matrix_times_vector(&a, &v, &mut r);
        assert_eq!(r, [0.0, 0.0]);
    }

    // ===================================================================
    // Category 3: compute_cost
    // ===================================================================

    #[test]
    fn test_compute_cost_identity_at_solution() {
        // A=I, b=[1,1], x=[1,1]: cost = 2*<b,x> - <x,Ax> = 2*2 - 2 = 2
        let a = [1.0, 0.0, 0.0, 1.0];
        let b = [1.0, 1.0];
        let x = [1.0, 1.0];
        let cost = compute_cost(&a, &b, &x, 2);
        assert!((cost - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_cost_zero_x() {
        let a = [1.0, 0.0, 0.0, 1.0];
        let b = [1.0, 1.0];
        let x = [0.0, 0.0];
        let cost = compute_cost(&a, &b, &x, 2);
        assert!((cost - 0.0).abs() < 1e-10);
    }

    // ===================================================================
    // Category 4: CG solver - identity matrix
    // ===================================================================

    #[test]
    fn test_cg_identity_matrix() {
        // Ix = b => x = b
        let a = [1.0, 0.0, 0.0, 1.0];
        let b = [3.0, 7.0];
        let mut x = [0.0, 0.0];
        conjugate_gradient(&a, &mut x, &b, 2, 1e-10, 100);
        assert!((x[0] - 3.0).abs() < 1e-6, "x[0] = {}", x[0]);
        assert!((x[1] - 7.0).abs() < 1e-6, "x[1] = {}", x[1]);
    }

    // ===================================================================
    // Category 5: CG solver - diagonal matrix
    // ===================================================================

    #[test]
    fn test_cg_diagonal_matrix() {
        // diag(2, 5) * x = [4, 10] => x = [2, 2]
        let a = [2.0, 0.0, 0.0, 5.0];
        let b = [4.0, 10.0];
        let mut x = [0.0, 0.0];
        conjugate_gradient(&a, &mut x, &b, 2, 1e-10, 100);
        assert!((x[0] - 2.0).abs() < 1e-6, "x[0] = {}", x[0]);
        assert!((x[1] - 2.0).abs() < 1e-6, "x[1] = {}", x[1]);
    }

    // ===================================================================
    // Category 6: CG solver - symmetric positive-definite matrix
    // ===================================================================

    #[test]
    fn test_cg_spd_matrix() {
        // [[4, 1], [1, 3]] * x = [1, 2]
        // Solution: x = [1/11, 7/11] ≈ [0.0909, 0.6364]
        let a = [4.0, 1.0, 1.0, 3.0];
        let b = [1.0, 2.0];
        let mut x = [0.0, 0.0];
        conjugate_gradient(&a, &mut x, &b, 2, 1e-10, 100);
        assert!((x[0] - 1.0 / 11.0).abs() < 1e-6, "x[0] = {}", x[0]);
        assert!((x[1] - 7.0 / 11.0).abs() < 1e-6, "x[1] = {}", x[1]);
    }

    #[test]
    fn test_cg_3x3_spd() {
        // [[2, -1, 0], [-1, 2, -1], [0, -1, 2]] * x = [1, 0, 1]
        // This is a tridiagonal SPD matrix (1D Laplacian)
        // Solution: x = [1, 1, 1]
        let a = [
            2.0, -1.0, 0.0,
            -1.0, 2.0, -1.0,
            0.0, -1.0, 2.0,
        ];
        let b = [1.0, 0.0, 1.0];
        let mut x = [0.0, 0.0, 0.0];
        conjugate_gradient(&a, &mut x, &b, 3, 1e-10, 100);
        for i in 0..3 {
            assert!((x[i] - 1.0).abs() < 1e-6, "x[{}] = {}", i, x[i]);
        }
    }

    // ===================================================================
    // Category 7: CG solver - convergence
    // ===================================================================

    #[test]
    fn test_cg_already_at_solution() {
        let a = [1.0, 0.0, 0.0, 1.0];
        let b = [3.0, 7.0];
        let mut x = [3.0, 7.0]; // already at solution
        conjugate_gradient(&a, &mut x, &b, 2, 1e-10, 100);
        assert!((x[0] - 3.0).abs() < 1e-10);
        assert!((x[1] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_cg_respects_max_iterations() {
        // With max_iterations=0, x should not change
        let a = [1.0, 0.0, 0.0, 1.0];
        let b = [3.0, 7.0];
        let mut x = [0.0, 0.0];
        conjugate_gradient(&a, &mut x, &b, 2, 1e-10, 0);
        assert_eq!(x, [0.0, 0.0]);
    }

    #[test]
    fn test_cg_loose_tolerance() {
        let a = [1.0, 0.0, 0.0, 1.0];
        let b = [3.0, 7.0];
        let mut x = [0.0, 0.0];
        // Very loose tolerance - should converge quickly
        conjugate_gradient(&a, &mut x, &b, 2, 10.0, 100);
        // May not be accurate but shouldn't crash
    }

    // ===================================================================
    // Category 8: CG solver - larger system
    // ===================================================================

    #[test]
    fn test_cg_5x5_diagonal() {
        let n = 5;
        let mut a = vec![0.0; n * n];
        let mut b = vec![0.0; n];
        for i in 0..n {
            a[i * n + i] = (i + 1) as f64; // diag = [1,2,3,4,5]
            b[i] = (i + 1) as f64 * 2.0;   // b = [2,4,6,8,10]
        }
        // Solution: x = [2, 2, 2, 2, 2]
        let mut x = vec![0.0; n];
        conjugate_gradient(&a, &mut x, &b, n, 1e-10, 100);
        for i in 0..n {
            assert!((x[i] - 2.0).abs() < 1e-6, "x[{}] = {}", i, x[i]);
        }
    }

    // ===================================================================
    // Category 9: CG solver - residual verification
    // ===================================================================

    #[test]
    fn test_cg_residual_small() {
        let a = [4.0, 1.0, 1.0, 3.0];
        let b = [1.0, 2.0];
        let mut x = [0.0, 0.0];
        conjugate_gradient(&a, &mut x, &b, 2, 1e-12, 100);

        // Verify: Ax ≈ b
        let mut ax = [0.0, 0.0];
        matrix_times_vector(&a, &x, &mut ax);
        for i in 0..2 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-6,
                "Residual too large at {}: Ax={}, b={}", i, ax[i], b[i]
            );
        }
    }
}
