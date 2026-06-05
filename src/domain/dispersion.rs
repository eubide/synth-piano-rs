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
//! larger DC group delay we have to subtract from the delay line. We aim for
//! a *fixed* strong coefficient (`|a| = 0.5`) across the whole keyboard and
//! back it off only in the extreme treble, where the loop is too short to
//! hold the cascade's group delay. Holding `|a|` constant while the loop
//! period shrinks makes the cascade's group delay a growing *fraction* of
//! the loop toward the treble, so the relative partial stretch — the audible
//! inharmonicity — rises with pitch, matching real strings (B grows with
//! note number). Only the very top (≳ C8) tapers `|a|` down; nothing in the
//! played range collapses to a flat, inharmonicity-free `a = 0`.

use crate::domain::filter::AllpassFirstOrder;

/// Number of first-order allpasses in the cascade. More stages allow
/// stronger dispersion at a given per-stage `|a|`, at modest CPU cost.
pub const CASCADE_STAGES: usize = 4;

/// Desired per-stage DC group delay, in samples. `(1+|a|)/(1−|a|) = 3`
/// gives `|a| = 0.5` — a strong, fixed dispersion strength. Keeping the
/// *coefficient* roughly constant across the keyboard (rather than a fixed
/// fraction of each loop) is what makes inharmonicity grow with pitch: the
/// cascade's group delay stays ~constant in samples while the loop period
/// shrinks, so the *relative* partial stretch rises toward the treble —
/// exactly the physics (B increases with note number; Fletcher & Rossing).
const TARGET_PER_STAGE_DELAY: f32 = 3.0;

/// Fraction of the (usable) loop period the cascade's group delay may
/// consume. The remainder feeds the delay line, which must stay ≥ 2 samples.
/// Only binds in the extreme treble, where the loop is too short to hold the
/// full target — there `|a|` tapers down gracefully instead of collapsing to
/// zero (the old behaviour, which left the treble with no inharmonicity).
const MAX_CASCADE_FRACTION: f32 = 0.6;

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
        // Group delay the cascade may consume, leaving the delay line its
        // ≥ 2 samples plus the LPF's ½. Only the extreme treble is tight.
        let budget = ((loop_period - 2.5) * MAX_CASCADE_FRACTION).max(0.0);
        let max_per_stage = budget / CASCADE_STAGES as f32;
        // Aim for the fixed target strength, backing off only when the loop
        // cannot hold it. Constant target → inharmonicity grows with pitch.
        let per_stage = max_per_stage.min(TARGET_PER_STAGE_DELAY);
        let a = if per_stage <= 1.0 {
            // Loop too short for even a 1-sample stage delay: no dispersion.
            0.0
        } else {
            // Solve τ_g(0) = (1+|a|)/(1−|a|) = per_stage for |a|.
            let mag = (per_stage - 1.0) / (per_stage + 1.0);
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
    fn fit_keeps_treble_dispersive_but_bounded() {
        // C8 (4186 Hz) at 48 kHz: loop period ≈ 11.5. The cascade can no
        // longer hold the full target, so |a| tapers down — but it must NOT
        // collapse to 0 (the old behaviour that left the treble harmonic).
        let mut c = DispersionCascade::new();
        c.fit_to_frequency(4_186.0, 48_000.0);
        let a = c.coefficient();
        assert!(
            a < 0.0 && a > -MAX_ABS_COEFFICIENT,
            "treble should stay dispersive but below the cap: a={a}"
        );
    }

    #[test]
    fn fit_picks_full_strength_for_mid_notes() {
        // A4 (440 Hz): the loop has ample room, so |a| reaches the target
        // strength (the cap MAX_ABS_COEFFICIENT).
        let mut c = DispersionCascade::new();
        c.fit_to_frequency(440.0, 48_000.0);
        let a = c.coefficient();
        assert!(
            (a + MAX_ABS_COEFFICIENT).abs() < 1e-4,
            "mid note should hit the target strength: a={a}"
        );
    }

    #[test]
    fn fit_caps_coefficient_for_very_long_loops() {
        // A0 (27.5 Hz): loop period 1745, far more room than the target
        // needs → |a| pinned at the cap MAX_ABS_COEFFICIENT.
        let mut c = DispersionCascade::new();
        c.fit_to_frequency(27.5, 48_000.0);
        assert!((c.coefficient() + MAX_ABS_COEFFICIENT).abs() < 1e-4);
    }
}
