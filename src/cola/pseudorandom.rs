//! Deterministic pseudo-random number generator.
//!
//! Linear congruential generator matching the C++ implementation exactly
//! for reproducible layouts.
//!
//! C++ ref: libcola/pseudorandom.h, libcola/pseudorandom.cpp

/// LCG multiplier.
const LCG_A: u64 = 214013;
/// LCG increment.
const LCG_C: u64 = 2531011;
/// LCG modulus (2^31).
const LCG_M: u64 = 2147483648;
/// Range divisor for normalizing to [0, ~1].
const LCG_RANGE: f64 = 32767.0;
/// Right shift to extract high bits.
const LCG_SHIFT: u32 = 16;

/// Deterministic pseudo-random number generator.
///
/// Produces the same sequence for the same seed, ensuring reproducible layouts.
///
/// C++ ref: cola::PseudoRandom
#[derive(Debug, Clone)]
pub struct PseudoRandom {
    seed: u64,
}

impl PseudoRandom {
    /// Create a new PRNG with the given seed.
    pub fn new(seed: f64) -> Self {
        Self { seed: seed as u64 }
    }

    /// Get the next random value in [0, ~1].
    ///
    /// C++ ref: PseudoRandom::getNext()
    pub fn get_next(&mut self) -> f64 {
        self.seed = (self.seed.wrapping_mul(LCG_A).wrapping_add(LCG_C)) % LCG_M;
        (self.seed >> LCG_SHIFT) as f64 / LCG_RANGE
    }

    /// Get the next random value in [min, max].
    ///
    /// C++ ref: PseudoRandom::getNextBetween()
    pub fn get_next_between(&mut self, min: f64, max: f64) -> f64 {
        min + self.get_next() * (max - min)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===================================================================
    // Category 1: Determinism
    // ===================================================================

    #[test]
    fn test_same_seed_same_sequence() {
        let mut a = PseudoRandom::new(42.0);
        let mut b = PseudoRandom::new(42.0);
        for _ in 0..100 {
            assert_eq!(a.get_next(), b.get_next());
        }
    }

    #[test]
    fn test_different_seed_different_sequence() {
        let mut a = PseudoRandom::new(1.0);
        let mut b = PseudoRandom::new(2.0);
        // First values should differ (not guaranteed but overwhelmingly likely)
        let va = a.get_next();
        let vb = b.get_next();
        assert_ne!(va, vb);
    }

    // ===================================================================
    // Category 2: Range
    // ===================================================================

    #[test]
    fn test_get_next_range() {
        let mut rng = PseudoRandom::new(1.0);
        for _ in 0..1000 {
            let v = rng.get_next();
            assert!(v >= 0.0, "Value {} is below 0", v);
            // Maximum possible: (LCG_M-1) >> 16 / 32767 ≈ 1.0
            assert!(v <= 1.001, "Value {} is above ~1", v);
        }
    }

    #[test]
    fn test_get_next_between_range() {
        let mut rng = PseudoRandom::new(7.0);
        let min = -5.0;
        let max = 15.0;
        for _ in 0..1000 {
            let v = rng.get_next_between(min, max);
            assert!(v >= min, "Value {} < min {}", v, min);
            assert!(v <= max + 0.001, "Value {} > max {}", v, max);
        }
    }

    #[test]
    fn test_get_next_between_equal_bounds() {
        let mut rng = PseudoRandom::new(1.0);
        let v = rng.get_next_between(5.0, 5.0);
        assert!((v - 5.0).abs() < 1e-10);
    }

    // ===================================================================
    // Category 3: Known values (regression)
    // ===================================================================

    #[test]
    fn test_seed_1_first_values() {
        // Pin the first few outputs from seed=1 for regression
        let mut rng = PseudoRandom::new(1.0);
        let v1 = rng.get_next();
        let v2 = rng.get_next();
        let v3 = rng.get_next();

        // These are deterministic - snapshot them
        // seed=1: (1*214013+2531011)%2147483648 = 2745024
        // v1 = (2745024 >> 16) / 32767 = 41/32767
        assert!((v1 - 41.0 / 32767.0).abs() < 1e-10);

        // Verify subsequent values are deterministic
        let mut rng2 = PseudoRandom::new(1.0);
        assert_eq!(rng2.get_next(), v1);
        assert_eq!(rng2.get_next(), v2);
        assert_eq!(rng2.get_next(), v3);
    }

    // ===================================================================
    // Category 4: Seed edge cases
    // ===================================================================

    #[test]
    fn test_seed_zero() {
        let mut rng = PseudoRandom::new(0.0);
        // seed=0: (0*214013+2531011)%2147483648 = 2531011
        // v = (2531011 >> 16) / 32767 = 38/32767
        let v = rng.get_next();
        assert!((v - 38.0 / 32767.0).abs() < 1e-10);
    }

    #[test]
    fn test_clone_continues_independently() {
        let mut rng = PseudoRandom::new(42.0);
        rng.get_next();
        rng.get_next();
        let mut clone = rng.clone();
        assert_eq!(rng.get_next(), clone.get_next());
        assert_eq!(rng.get_next(), clone.get_next());
    }

    // ===================================================================
    // Category 5: Distribution (basic statistical sanity)
    // ===================================================================

    #[test]
    fn test_distribution_not_constant() {
        let mut rng = PseudoRandom::new(1.0);
        let first = rng.get_next();
        let mut all_same = true;
        for _ in 0..100 {
            if (rng.get_next() - first).abs() > 1e-10 {
                all_same = false;
                break;
            }
        }
        assert!(!all_same, "Generator produces constant output");
    }

    #[test]
    fn test_distribution_roughly_uniform() {
        let mut rng = PseudoRandom::new(1.0);
        let n = 10000;
        let mut below_half = 0;
        for _ in 0..n {
            if rng.get_next() < 0.5 {
                below_half += 1;
            }
        }
        // Should be roughly 50% (allow wide margin for LCG)
        let ratio = below_half as f64 / n as f64;
        assert!(ratio > 0.3 && ratio < 0.7,
            "Distribution heavily skewed: {:.2}% below 0.5", ratio * 100.0);
    }
}
