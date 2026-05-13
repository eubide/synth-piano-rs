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

/// First-order allpass: `H(z) = (a + z⁻¹) / (1 + a·z⁻¹)`.
///
/// - Magnitude is exactly 1 at all frequencies (defining property).
/// - Phase varies with frequency, so group delay does too — the building
///   block of the dispersion cascade that produces piano inharmonicity.
///
/// ## Group delay
/// `τ_g(ω) = (1 − a²) / (1 + 2a·cos(ω) + a²)` samples.
///
/// For `a < 0`: τ_g(0) > τ_g(π) — high-frequency components traverse the
/// filter faster than low-frequency ones, which is what we want to stretch
/// the partials of a stiff string upwards.
///
/// For `a > 0`: τ_g(0) < τ_g(π) — flattens partials (not useful for piano).
///
/// `a = 0` collapses the filter to `y[n] = x[n-1]` (a 1-sample delay, no
/// dispersion). Coefficient must lie strictly inside the unit circle
/// (`|a| < 1`) for stability; we clamp to ±0.99.
#[derive(Debug, Clone, Copy)]
pub struct AllpassFirstOrder {
    a: f32,
    x_prev: f32,
    y_prev: f32,
}

impl AllpassFirstOrder {
    pub fn new() -> Self {
        Self {
            a: 0.0,
            x_prev: 0.0,
            y_prev: 0.0,
        }
    }

    pub fn set_coefficient(&mut self, a: f32) {
        self.a = a.clamp(-0.99, 0.99);
    }

    pub fn coefficient(&self) -> f32 {
        self.a
    }

    pub fn reset(&mut self) {
        self.x_prev = 0.0;
        self.y_prev = 0.0;
    }

    /// Direct-form I implementation:
    ///   `y[n] = a·x[n] + x[n-1] − a·y[n-1]`
    /// Two multiplications, one addition, one subtraction.
    #[inline]
    pub fn tick(&mut self, x: f32) -> f32 {
        let y = self.a * x + self.x_prev - self.a * self.y_prev;
        self.x_prev = x;
        self.y_prev = y;
        y
    }

    /// Closed-form group delay at angular frequency `omega ∈ [0, π]`.
    /// Used to tune the loop length so the fundamental still lands on `f₀`.
    pub fn group_delay_at(&self, omega: f32) -> f32 {
        let a = self.a;
        (1.0 - a * a) / (1.0 + 2.0 * a * omega.cos() + a * a)
    }
}

impl Default for AllpassFirstOrder {
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
    fn allpass_dc_step_settles_to_input() {
        let mut ap = AllpassFirstOrder::new();
        ap.set_coefficient(-0.5);
        let mut y = 0.0;
        for _ in 0..1_000 {
            y = ap.tick(1.0);
        }
        assert!((y - 1.0).abs() < 1e-4, "settled at {y}");
    }

    #[test]
    fn allpass_has_unity_magnitude_at_arbitrary_frequency() {
        // Drive with a sinusoid; steady-state RMS must match the input RMS.
        let mut ap = AllpassFirstOrder::new();
        ap.set_coefficient(-0.5);
        let omega = TAU * 1_000.0 / 48_000.0;
        let mut x_sq = 0.0;
        let mut y_sq = 0.0;
        for i in 0..8_192 {
            let x = (omega * i as f32).sin();
            let y = ap.tick(x);
            if i > 500 {
                x_sq += x * x;
                y_sq += y * y;
            }
        }
        let gain = (y_sq / x_sq).sqrt();
        assert!((gain - 1.0).abs() < 1e-3, "expected unity magnitude, got {gain}");
    }

    #[test]
    fn allpass_group_delay_at_dc_matches_formula() {
        // For a=-0.5 at ω=0: (1 - 0.25) / (1 - 1 + 0.25) = 3.
        let mut ap = AllpassFirstOrder::new();
        ap.set_coefficient(-0.5);
        let td = ap.group_delay_at(0.0);
        assert!((td - 3.0).abs() < 1e-5, "got {td}, expected 3");
    }

    #[test]
    fn allpass_group_delay_at_nyquist_matches_formula() {
        // For a=-0.5 at ω=π: (1 - 0.25) / (1 + 1 + 0.25) = 0.75/2.25 = 1/3.
        let mut ap = AllpassFirstOrder::new();
        ap.set_coefficient(-0.5);
        let td = ap.group_delay_at(std::f32::consts::PI);
        assert!(
            (td - 1.0 / 3.0).abs() < 1e-5,
            "got {td}, expected 1/3"
        );
    }

    #[test]
    fn allpass_ramp_delay_matches_dc_group_delay() {
        // Steady-state response to a ramp x[n] = n is y[n] = n − τ_g(0).
        let mut ap = AllpassFirstOrder::new();
        ap.set_coefficient(-0.5);
        let mut last_y = 0.0;
        for n in 0..2_000 {
            last_y = ap.tick(n as f32);
        }
        let delay_observed = (1999.0 - last_y).abs();
        assert!(
            (delay_observed - 3.0).abs() < 1e-2,
            "observed delay {delay_observed}, expected 3"
        );
    }

    #[test]
    fn allpass_zero_coefficient_is_pure_sample_delay() {
        // a=0 collapses to y[n] = x[n-1].
        let mut ap = AllpassFirstOrder::new();
        ap.set_coefficient(0.0);
        assert_eq!(ap.tick(1.0), 0.0);
        assert_eq!(ap.tick(2.0), 1.0);
        assert_eq!(ap.tick(3.0), 2.0);
    }

    #[test]
    fn allpass_coefficient_is_clamped_to_unit_circle() {
        let mut ap = AllpassFirstOrder::new();
        ap.set_coefficient(5.0);
        assert!(ap.coefficient() < 1.0);
        ap.set_coefficient(-5.0);
        assert!(ap.coefficient() > -1.0);
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
