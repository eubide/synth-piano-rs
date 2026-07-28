//! Power-of-two ring-buffer delay line with linear-interpolated fractional reads.
//!
//! Capacity is rounded up to the next power of two so `index & mask` replaces
//! `index % capacity` — fast, branchless wraparound.
//!
//! ## Why linear interpolation, not allpass
//! - Allpass fractional delays preserve magnitude response exactly (no high-freq
//!   loss) but require state and care about transients.
//! - Linear interpolation is `(1-α)·s₀ + α·s₁`. It's a 2-tap FIR with
//!   |H(ω)| = √(1 − 4α(1-α)·sin²(ω/2)) — i.e. a mild lowpass that worsens
//!   when α=0.5. For piano (highest fundamental ≈ 4.2 kHz, well below Nyquist),
//!   the extra HF loss is <1 dB. Cheap, allocation-free, good enough.
//! - Phase 4 introduces an allpass tuner only when the dispersion cascade
//!   already lives in the loop and the extra state is essentially free.

#[derive(Debug)]
pub struct DelayLine {
    buffer: Vec<f32>,
    mask: usize,
    write_pos: usize,
}

impl DelayLine {
    /// Allocates a ring buffer of capacity rounded up to the next power of two.
    pub fn new(min_capacity: usize) -> Self {
        let cap = min_capacity.max(2).next_power_of_two();
        Self {
            buffer: vec![0.0; cap],
            mask: cap - 1,
            write_pos: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }

    /// Multiply every stored sample by `g`, leaving the write head where it
    /// is. The delay line is linear, so this scales the whole future
    /// zero-input response by `g` without introducing a discontinuity —
    /// used to fold an output-side gain back into the loop state.
    pub fn scale(&mut self, g: f32) {
        for s in self.buffer.iter_mut() {
            *s *= g;
        }
    }

    /// Zero the buffer and rewind the write head.
    pub fn clear(&mut self) {
        for s in self.buffer.iter_mut() {
            *s = 0.0;
        }
        self.write_pos = 0;
    }

    /// Push a sample at the head, advance by one.
    #[inline]
    pub fn write(&mut self, x: f32) {
        self.buffer[self.write_pos] = x;
        self.write_pos = (self.write_pos + 1) & self.mask;
    }

    /// Read at integer delay `d` (1 ≤ d < capacity).
    #[inline]
    pub fn read_int(&self, d: usize) -> f32 {
        let idx = self.write_pos.wrapping_sub(d) & self.mask;
        self.buffer[idx]
    }

    /// Read at fractional delay `d_int + frac`, `frac ∈ [0, 1)`.
    /// Linear interpolation between the two adjacent integer taps.
    #[inline]
    pub fn read_frac(&self, d_int: usize, frac: f32) -> f32 {
        let s0 = self.read_int(d_int);
        let s1 = self.read_int(d_int + 1);
        s0 * (1.0 - frac) + s1 * frac
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_rounds_to_power_of_two() {
        assert_eq!(DelayLine::new(100).capacity(), 128);
        assert_eq!(DelayLine::new(128).capacity(), 128);
        assert_eq!(DelayLine::new(129).capacity(), 256);
    }

    #[test]
    fn write_then_read_int_returns_recent_sample() {
        let mut d = DelayLine::new(16);
        d.write(0.5);
        // After one write, the head is at index 1. Reading at delay 1 hits index 0.
        assert_eq!(d.read_int(1), 0.5);
    }

    #[test]
    fn integer_delay_returns_correct_history() {
        let mut d = DelayLine::new(16);
        for i in 0..8 {
            d.write(i as f32);
        }
        // Most recently written sample = 7.0 → read_int(1)
        assert_eq!(d.read_int(1), 7.0);
        assert_eq!(d.read_int(2), 6.0);
        assert_eq!(d.read_int(8), 0.0);
    }

    #[test]
    fn fractional_read_interpolates_linearly() {
        let mut d = DelayLine::new(16);
        for v in [10.0_f32, 20.0, 30.0, 40.0] {
            d.write(v);
        }
        // read_int(1)=40, read_int(2)=30 → frac 0.25 → 0.75*40 + 0.25*30 = 37.5
        assert!((d.read_frac(1, 0.25) - 37.5).abs() < 1e-6);
    }

    #[test]
    fn ring_wraps_around() {
        let mut d = DelayLine::new(4); // capacity 4
        for v in 0..6 {
            d.write(v as f32);
        }
        // Wrote 0,1,2,3,4,5. Buffer now holds 4,5,2,3 (in some order); last
        // four samples are 2,3,4,5. So read_int(1)=5, read_int(4)=2.
        assert_eq!(d.read_int(1), 5.0);
        assert_eq!(d.read_int(4), 2.0);
    }
}
