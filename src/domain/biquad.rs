//! Second-order IIR section (biquad), Transposed Direct Form II.
//!
//! ## Why TDF-II
//! Two state variables (`z1`, `z2`), four multiplications and four
//! additions per sample. Numerically more robust than Direct Form I for
//! time-varying coefficients (which we don't actually need here, but
//! costs nothing to choose the safer topology).
//!
//! ## Coefficient cookbook
//! [`Biquad::set_bandpass`] uses Robert Bristow-Johnson's *Audio EQ
//! Cookbook* formulas. The bandpass with constant 0 dB peak gain (CPG)
//! puts a complex-conjugate pole pair at radius `r ≈ exp(−π·f₀/(Q·Fs))`
//! and angle `θ ≈ 2π·f₀/Fs`. Closer to the unit circle = sharper
//! resonance = longer impulse-response tail.
//!
//! ## Stability
//! Coefficients are bounded by construction: `a₀ = 1 + α > 0` always,
//! and `|a₂| < 1` because `|1 − α| < 1 + α` for `α > 0`. So both poles
//! stay inside the unit disk. No runtime clamping needed.

use std::f32::consts::TAU;

#[derive(Debug, Clone, Copy)]
pub struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl Biquad {
    /// Identity biquad — passes input unchanged. Call `set_*` before use.
    pub fn new() -> Self {
        Self {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    /// Configure as a bandpass with constant 0 dB peak gain (RBJ).
    ///
    /// At `freq_hz` the magnitude response peaks at 1.0. Bandwidth shrinks
    /// with increasing `q`; sensible values are 5–60 for audible body
    /// resonances. `q < 0.01` is clamped to avoid division blow-up.
    pub fn set_bandpass(&mut self, freq_hz: f32, q: f32, sample_rate: f32) {
        let q = q.max(0.01);
        let omega = TAU * freq_hz / sample_rate;
        let sin_o = omega.sin();
        let cos_o = omega.cos();
        let alpha = sin_o / (2.0 * q);

        let a0 = 1.0 + alpha;
        self.b0 = alpha / a0;
        self.b1 = 0.0;
        self.b2 = -alpha / a0;
        self.a1 = -2.0 * cos_o / a0;
        self.a2 = (1.0 - alpha) / a0;
    }

    pub fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }

    /// Transposed Direct Form II tick.
    #[inline]
    pub fn tick(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }
}

impl Default for Biquad {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rms_gain_at(filter: &mut Biquad, freq_hz: f32, sample_rate: f32) -> f32 {
        let omega = TAU * freq_hz / sample_rate;
        let n = 8_192usize;
        let mut x_sq = 0.0;
        let mut y_sq = 0.0;
        for i in 0..n {
            let x = (omega * i as f32).sin();
            let y = filter.tick(x);
            if i > 500 {
                x_sq += x * x;
                y_sq += y * y;
            }
        }
        (y_sq / x_sq).sqrt()
    }

    #[test]
    fn default_biquad_is_identity() {
        let mut b = Biquad::new();
        for x in [0.5, -0.3, 1.0, 0.0] {
            assert_eq!(b.tick(x), x);
        }
    }

    #[test]
    fn bandpass_kills_dc() {
        let mut b = Biquad::new();
        b.set_bandpass(1_000.0, 10.0, 48_000.0);
        let mut y = 0.0;
        // Feed a DC step long enough for transients to die.
        for _ in 0..10_000 {
            y = b.tick(1.0);
        }
        assert!(y.abs() < 1e-3, "DC leaked through: {y}");
    }

    #[test]
    fn bandpass_passes_centre_at_unity_peak_gain() {
        // RBJ "constant 0 dB peak" — at f = f₀ the magnitude is exactly 1.
        let mut b = Biquad::new();
        b.set_bandpass(1_000.0, 10.0, 48_000.0);
        let g = rms_gain_at(&mut b, 1_000.0, 48_000.0);
        assert!((g - 1.0).abs() < 0.02, "centre gain {g}, expected ~1");
    }

    #[test]
    fn bandpass_attenuates_far_from_centre() {
        let mut b = Biquad::new();
        b.set_bandpass(1_000.0, 20.0, 48_000.0);
        let g = rms_gain_at(&mut b, 100.0, 48_000.0); // 1 decade below
        assert!(g < 0.1, "expected strong attenuation, got {g}");
    }

    #[test]
    fn higher_q_is_sharper() {
        // Same centre, two Q values. The higher-Q filter should attenuate
        // a sideband more than the low-Q one.
        let off_centre = 1_500.0;
        let mut b_lowq = Biquad::new();
        b_lowq.set_bandpass(1_000.0, 5.0, 48_000.0);
        let g_lowq = rms_gain_at(&mut b_lowq, off_centre, 48_000.0);

        let mut b_highq = Biquad::new();
        b_highq.set_bandpass(1_000.0, 30.0, 48_000.0);
        let g_highq = rms_gain_at(&mut b_highq, off_centre, 48_000.0);

        assert!(
            g_highq < g_lowq,
            "expected higher Q to be sharper: lowQ={g_lowq} highQ={g_highq}"
        );
    }

    #[test]
    fn impulse_response_decays() {
        // Resonant biquad has an exponentially decaying ringing impulse
        // response — energy must drop over time.
        let mut b = Biquad::new();
        b.set_bandpass(500.0, 30.0, 48_000.0);
        let mut early_energy = 0.0;
        let mut late_energy = 0.0;
        let n = 8_192;
        let half = n / 2;
        let mut x = 1.0;
        for i in 0..n {
            let y = b.tick(x);
            x = 0.0;
            if i < half {
                early_energy += y * y;
            } else {
                late_energy += y * y;
            }
        }
        assert!(
            late_energy < early_energy * 0.5,
            "ringing did not decay: early={early_energy} late={late_energy}"
        );
    }

    #[test]
    fn reset_clears_internal_state() {
        let mut b = Biquad::new();
        b.set_bandpass(500.0, 30.0, 48_000.0);
        // Pump some energy into the state.
        for _ in 0..1_000 {
            b.tick(1.0);
        }
        b.reset();
        // First tick with zero input should be exactly zero (state is wiped).
        assert_eq!(b.tick(0.0), 0.0);
    }
}
