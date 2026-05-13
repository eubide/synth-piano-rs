//! Tiny PRNG for white-noise excitation. Deliberately not cryptographic.
//!
//! Xorshift32 (Marsaglia 2003): 32-bit state, period 2³² − 1, one branch.
//! Each voice owns one so plucks differ from each other without locking.

#[derive(Debug, Clone)]
pub struct Xorshift32 {
    state: u32,
}

impl Xorshift32 {
    /// Seed must be non-zero; we clamp to 1 to guarantee a valid generator.
    pub fn new(seed: u32) -> Self {
        Self { state: seed.max(1) }
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        x
    }

    /// Uniform `[-1.0, 1.0)`.
    #[inline]
    pub fn next_signed_unit(&mut self) -> f32 {
        // Reinterpret as signed so we get a centred range.
        let u = self.next_u32() as i32;
        (u as f32) / (i32::MAX as f32 + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_unit_stays_in_bounds() {
        let mut rng = Xorshift32::new(42);
        for _ in 0..10_000 {
            let x = rng.next_signed_unit();
            assert!((-1.0..1.0).contains(&x), "out of range: {x}");
        }
    }

    #[test]
    fn signed_unit_has_zero_mean_approximately() {
        let mut rng = Xorshift32::new(42);
        let n = 50_000;
        let mut sum = 0.0;
        for _ in 0..n {
            sum += rng.next_signed_unit();
        }
        let mean = sum / n as f32;
        // 1/√n ≈ 0.0045; pad generously to avoid flakiness.
        assert!(mean.abs() < 0.02, "mean drift: {mean}");
    }

    #[test]
    fn zero_seed_is_clamped() {
        let mut rng = Xorshift32::new(0);
        // Should still produce non-zero output.
        let x = rng.next_u32();
        assert_ne!(x, 0);
    }
}
