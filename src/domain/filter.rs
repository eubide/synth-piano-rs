//! Filter primitives used inside the Karplus-Strong loop and elsewhere.
//!
//! ## TwoTapLowpass
//! `y[n] = ½·(x[n] + x[n-1])`. Frequency response
//! `H(ω) = cos(ω/2)·exp(-jω/2)`. Two important properties for piano synth:
//! 1. **Constant phase delay of exactly ½ sample**, independent of frequency.
//!    Lets us compensate the loop length precisely so the perceived pitch
//!    matches `Fs/N` without fancy allpass tuning (Smith, "Physical Audio
//!    Signal Processing", §6.10).
//! 2. **Magnitude rolls off as `cos(ω/2)`**, zero at Nyquist. High partials
//!    lose energy each loop pass → natural string-like decay, with HF dying
//!    faster than LF. This is exactly the behaviour Karplus & Strong (1983)
//!    chose for their original "plucked string" algorithm.

/// Tunable one-pole lowpass: `y[n] = (1−α)·x[n] + α·y[n−1]`.
/// Pole at `z = α`; DC gain = 1; rolls off at `−6 dB/octave` past cutoff.
///
/// We map cutoff frequency → α with the "leaky integrator" rule
/// `α = exp(−2π·f_c / Fs)`. Exact only in the small-cutoff limit, but
/// good enough as a *perceptual* brightness knob (which is how the hammer
/// uses it). For sharp cutoffs near Nyquist use a biquad instead.
#[derive(Debug, Clone, Copy)]
pub struct OnePoleLowpass {
    a: f32,
    state: f32,
}

impl OnePoleLowpass {
    pub fn new() -> Self {
        Self {
            a: 0.0,
            state: 0.0,
        }
    }

    pub fn set_cutoff(&mut self, cutoff_hz: f32, sample_rate: f32) {
        // Clamp to a safe range. 1 Hz lower bound avoids α ≈ 1 (infinite memory).
        // Half-Nyquist upper bound stops α going negative.
        let c = cutoff_hz.clamp(1.0, sample_rate * 0.45);
        self.a = (-std::f32::consts::TAU * c / sample_rate).exp();
    }

    pub fn reset(&mut self) {
        self.state = 0.0;
    }

    #[inline]
    pub fn tick(&mut self, x: f32) -> f32 {
        let y = (1.0 - self.a) * x + self.a * self.state;
        self.state = y;
        y
    }
}

impl Default for OnePoleLowpass {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct TwoTapLowpass {
    prev: f32,
}

impl TwoTapLowpass {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        self.prev = 0.0;
    }

    #[inline]
    pub fn tick(&mut self, x: f32) -> f32 {
        let y = 0.5 * (x + self.prev);
        self.prev = x;
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    #[test]
    fn zero_in_zero_out() {
        let mut f = TwoTapLowpass::new();
        for _ in 0..10 {
            assert_eq!(f.tick(0.0), 0.0);
        }
    }

    #[test]
    fn dc_passes_unchanged_after_priming() {
        let mut f = TwoTapLowpass::new();
        f.tick(1.0); // primes prev = 1
        for _ in 0..10 {
            assert!((f.tick(1.0) - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn nyquist_is_attenuated_to_zero() {
        // Feeding alternating +1/-1 (Nyquist frequency) should average to 0.
        let mut f = TwoTapLowpass::new();
        f.tick(1.0);
        f.tick(-1.0);
        f.tick(1.0);
        let y = f.tick(-1.0);
        assert!(y.abs() < 1e-6, "nyquist not attenuated: {y}");
    }

    #[test]
    fn reset_clears_state() {
        let mut f = TwoTapLowpass::new();
        f.tick(1.0);
        f.reset();
        // After reset, prev = 0, so y = 0.5*(1+0) = 0.5
        assert!((f.tick(1.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn one_pole_zero_in_zero_out() {
        let mut f = OnePoleLowpass::new();
        f.set_cutoff(1_000.0, 48_000.0);
        for _ in 0..32 {
            assert_eq!(f.tick(0.0), 0.0);
        }
    }

    #[test]
    fn one_pole_dc_settles_to_input() {
        let mut f = OnePoleLowpass::new();
        f.set_cutoff(100.0, 48_000.0);
        let mut y = 0.0;
        for _ in 0..10_000 {
            y = f.tick(1.0);
        }
        assert!((y - 1.0).abs() < 1e-3, "settled at {y}, expected ~1.0");
    }

    #[test]
    fn one_pole_low_cutoff_attenuates_high_freq() {
        // Feed a sinusoid at 4 kHz through an LPF cutting off at 200 Hz.
        // Expect heavy attenuation (well below 0.5).
        let mut f = OnePoleLowpass::new();
        f.set_cutoff(200.0, 48_000.0);
        let omega = TAU * 4_000.0 / 48_000.0;
        let mut x_sq = 0.0;
        let mut y_sq = 0.0;
        for i in 0..8_192 {
            let x = (omega * i as f32).sin();
            let y = f.tick(x);
            if i > 500 {
                x_sq += x * x;
                y_sq += y * y;
            }
        }
        let gain = (y_sq / x_sq).sqrt();
        assert!(gain < 0.1, "expected heavy attenuation, got gain {gain}");
    }

    #[test]
    fn one_pole_high_cutoff_passes_high_freq() {
        // Feed a sinusoid at 1 kHz through an LPF cutting off at 12 kHz.
        // Expect near-unity passing.
        let mut f = OnePoleLowpass::new();
        f.set_cutoff(12_000.0, 48_000.0);
        let omega = TAU * 1_000.0 / 48_000.0;
        let mut x_sq = 0.0;
        let mut y_sq = 0.0;
        for i in 0..8_192 {
            let x = (omega * i as f32).sin();
            let y = f.tick(x);
            if i > 200 {
                x_sq += x * x;
                y_sq += y * y;
            }
        }
        let gain = (y_sq / x_sq).sqrt();
        assert!(gain > 0.85, "expected near-unity, got gain {gain}");
    }

    #[test]
    fn sinusoid_rms_gain_matches_cos_half_omega() {
        // For a sinusoid input at ω, the LPF output has magnitude cos(ω/2).
        // Peak-comparison is unreliable because the discrete samples need not
        // align with the continuous output's peak. RMS is exact:
        //   x[n] = sin(ωn)        → RMS = 1/√2
        //   y[n] = cos(ω/2)·sin(ωn − ω/2) → RMS = cos(ω/2)/√2
        // Ratio = cos(ω/2).
        let omega = std::f32::consts::FRAC_PI_2;
        let mut f = TwoTapLowpass::new();
        let n = 4_096usize;
        let mut x_sq_sum = 0.0;
        let mut y_sq_sum = 0.0;
        for i in 0..n {
            let x = (omega * i as f32).sin();
            let y = f.tick(x);
            if i > 100 {
                x_sq_sum += x * x;
                y_sq_sum += y * y;
            }
        }
        let gain = (y_sq_sum / x_sq_sum).sqrt();
        let expected = (omega / 2.0).cos();
        let _ = TAU; // silence unused-import warning
        assert!(
            (gain - expected).abs() < 0.01,
            "gain {gain} expected ~{expected}"
        );
    }
}
