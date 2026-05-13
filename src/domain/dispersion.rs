//! Allpass-cascade dispersion model for piano string stiffness.
//!
//! ## Why
//! A perfectly flexible string has partials at exact integer multiples of
//! the fundamental: `f_k = k·f₀`. Real piano strings have stiffness, which
//! stretches the partials upward: `f_k ≈ k·f₀·√(1 + B·k²)` where `B` is
//! the inharmonicity coefficient (Fletcher & Rossing, "The Physics of
//! Musical Instruments", §12.5).
//!
//! ## How
//! Inside the loop we cascade `N` first-order allpasses with `a < 0`. Each
//! allpass has unity magnitude (doesn't change loudness) but a group delay
//! that is largest at DC and smallest at Nyquist. So high-frequency
//! components circulate through the loop "faster" than low-frequency ones,
//! and the loop resonates at progressively higher frequencies for higher
//! partials — exactly the stretching the physics predicts.
//!
//! ## Coefficient strategy
//! Stronger dispersion (larger `|a|`) means more inharmonicity, but also a
//! larger DC group delay we have to subtract from the delay line. For
//! short loops (high notes) the cascade can eat the whole budget. We adapt
//! `|a|` so the cascade contributes a target *fraction* of the loop
//! period (currently 5 %), clamped so per-stage delay stays in `[1, ∞)`
//! and overall `|a| ≤ 0.5`. Notes whose loops are too short for any
//! dispersion fall back to `a = 0` (the cascade degenerates to `N` pure
//! sample delays — no inharmonicity, just length).

use crate::domain::filter::AllpassFirstOrder;

/// Number of first-order allpasses in the cascade. More stages allow
/// stronger dispersion at a given per-stage `|a|`, at modest CPU cost.
pub const CASCADE_STAGES: usize = 4;

/// Target cascade DC group delay as a fraction of the loop period.
const TARGET_FRACTION_OF_LOOP: f32 = 0.05;

/// Hard cap on `|a|` so the cascade never dominates the loop dynamics.
const MAX_ABS_COEFFICIENT: f32 = 0.5;

#[derive(Debug)]
pub struct DispersionCascade {
    stages: [AllpassFirstOrder; CASCADE_STAGES],
}

impl DispersionCascade {
    pub fn new() -> Self {
        Self {
            stages: [AllpassFirstOrder::new(); CASCADE_STAGES],
        }
    }

    /// Configure all stages to share a coefficient. Negative `a` stretches
    /// partials upward (piano-like); positive `a` flattens them (unused).
    pub fn set_coefficient(&mut self, a: f32) {
        for s in &mut self.stages {
            s.set_coefficient(a);
        }
    }

    pub fn coefficient(&self) -> f32 {
        self.stages[0].coefficient()
    }

    pub fn reset(&mut self) {
        for s in &mut self.stages {
            s.reset();
        }
    }

    #[inline]
    pub fn tick(&mut self, x: f32) -> f32 {
        // Allpasses cascade by sequential application; their phase
        // responses add and so do their group delays.
        let mut y = x;
        for s in &mut self.stages {
            y = s.tick(y);
        }
        y
    }

    /// Sum of stage group delays at `omega`.
    pub fn group_delay_at(&self, omega: f32) -> f32 {
        self.stages.iter().map(|s| s.group_delay_at(omega)).sum()
    }

    /// Pick an `a` for a given target fundamental frequency, sample rate,
    /// and available loop budget. Returns the coefficient applied to all
    /// stages. Falls back to `0.0` when no dispersion fits.
    pub fn fit_to_frequency(&mut self, freq: f32, sample_rate: f32) {
        let loop_period = sample_rate / freq;
        let target_total_delay = loop_period * TARGET_FRACTION_OF_LOOP;
        let target_per_stage = target_total_delay / CASCADE_STAGES as f32;
        let a = if target_per_stage <= 1.0 {
            // Degenerate: the loop is too short for any dispersion. Use a=0,
            // which makes each stage a 1-sample delay. The cascade contributes
            // exactly N samples of constant delay (no inharmonicity).
            0.0
        } else {
            // Solve τ_g(0) = (1+|a|)/(1−|a|) = target_per_stage for |a|.
            let mag = (target_per_stage - 1.0) / (target_per_stage + 1.0);
            -mag.min(MAX_ABS_COEFFICIENT)
        };
        self.set_coefficient(a);
    }
}

impl Default for DispersionCascade {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    #[test]
    fn fresh_cascade_has_zero_coefficient() {
        let c = DispersionCascade::new();
        assert_eq!(c.coefficient(), 0.0);
    }

    #[test]
    fn group_delay_sums_across_stages() {
        let mut c = DispersionCascade::new();
        c.set_coefficient(-0.5);
        // Single stage at DC: τ_g = 3 (verified in filter tests).
        // Cascade of 4: 12.
        let td = c.group_delay_at(0.0);
        assert!((td - 12.0).abs() < 1e-4, "got {td}, expected 12");
    }

    #[test]
    fn cascade_tick_chains_stages_sequentially() {
        // With a=0, each stage is a 1-sample delay, so the cascade should
        // delay by N=4 samples total.
        let mut c = DispersionCascade::new();
        c.set_coefficient(0.0);
        for i in 0..4 {
            assert_eq!(c.tick(i as f32 + 1.0), 0.0);
        }
        // The 5th input should appear at the output now.
        assert_eq!(c.tick(99.0), 1.0);
        assert_eq!(c.tick(99.0), 2.0);
    }

    #[test]
    fn cascade_has_unity_magnitude() {
        // Sum-of-allpasses-in-series remains allpass — magnitude 1.
        let mut c = DispersionCascade::new();
        c.set_coefficient(-0.5);
        let omega = TAU * 1_000.0 / 48_000.0;
        let mut x_sq = 0.0;
        let mut y_sq = 0.0;
        for i in 0..8_192 {
            let x = (omega * i as f32).sin();
            let y = c.tick(x);
            if i > 500 {
                x_sq += x * x;
                y_sq += y * y;
            }
        }
        let gain = (y_sq / x_sq).sqrt();
        assert!((gain - 1.0).abs() < 1e-3, "got gain {gain}");
    }

    #[test]
    fn fit_falls_back_to_zero_for_high_notes() {
        // C8 (4186 Hz) at 48 kHz: loop period 11.5; 5% = 0.575 samples;
        // per stage = 0.144 — well below the τ_g(0) ≥ 1 floor.
        let mut c = DispersionCascade::new();
        c.fit_to_frequency(4_186.0, 48_000.0);
        assert_eq!(c.coefficient(), 0.0);
    }

    #[test]
    fn fit_picks_negative_coefficient_for_low_notes() {
        // A4 (440 Hz) at 48 kHz: loop period 109; 5% = 5.45 samples;
        // per stage = 1.36 → a < 0 chosen.
        let mut c = DispersionCascade::new();
        c.fit_to_frequency(440.0, 48_000.0);
        let a = c.coefficient();
        assert!(a < 0.0 && a >= -MAX_ABS_COEFFICIENT, "got a={a}");
    }

    #[test]
    fn fit_caps_coefficient_for_very_long_loops() {
        // A0 (27.5 Hz): loop period 1745; 5% = 87; per stage = 21.8 →
        // ideal |a| ≈ 0.91, but we cap at MAX_ABS_COEFFICIENT.
        let mut c = DispersionCascade::new();
        c.fit_to_frequency(27.5, 48_000.0);
        assert!((c.coefficient() + MAX_ABS_COEFFICIENT).abs() < 1e-4);
    }
}
