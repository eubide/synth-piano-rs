//! Strike-position comb filter for the hammer excitation.
//!
//! ## Why
//! A hammer striking a string at fraction `p` of its speaking length cannot
//! excite any partial with a node at the strike point: partial `k` receives
//! energy weighted by `sin(k·π·p)` (Fletcher & Rossing §2.9). Grand pianos
//! place the hammer line near `p ≈ 1/7`, so the 7th-partial family is
//! suppressed and energy tilts toward the mid partials. Without this
//! weighting the
//! excitation drives *all* partials equally, and the result — most audible
//! in the bass, where dozens of partials fit under Nyquist — is a buzzy,
//! organ-like tone.
//!
//! ## How
//! In a waveguide the strike-position weighting is exactly a feedforward
//! comb applied to the excitation before it enters the loop:
//! `y[n] = x[n] − g·x[n − p·N]`, with `N = Fs/f₀` the loop period and `g`
//! the comb depth. At `g = 1` the magnitude at partial `k` is the ideal
//! point-hammer weighting `2·sin(k·π·p)`. The comb sits *outside* the
//! loop, so it cannot affect tuning or stability; it only shapes what
//! spectrum gets injected.
//!
//! ## Why the comb is only partial (`g < 1`)
//! A real hammer is not a point: the felt contacts a segment of the
//! speaking length, so the reflection returning from the agraffe is smeared
//! and never cancels the direct wave exactly. A full-depth comb (`g = 1`)
//! produces total nulls and a 2× boost of the mid partials — audibly a
//! hollow, flanger-like "electronic" coloration. With `g = 0.7` the nulls
//! become ≈ −10 dB dips and the mid-partial boost tames to ≈ 1.66×, which
//! keeps the piano-like partial tilt without the metallic sheen.

use crate::domain::delay_line::DelayLine;

/// Strike point as a fraction of string length. Real grands use ≈ 1/7–1/9
/// over most of the compass (Fletcher & Rossing §12.4). 1/7 favours the
/// fundamental slightly more than 1/8 (warmer) and puts the comb dips on
/// the 7th-partial family.
pub const STRIKE_POSITION: f32 = 1.0 / 7.0;

/// Comb depth (see module docs). 1.0 = ideal point hammer (hard nulls);
/// lower values model the felt's finite contact width.
pub const COMB_DEPTH: f32 = 0.7;

#[derive(Debug)]
pub struct StrikeComb {
    delay: DelayLine,
    d_int: usize,
    d_frac: f32,
}

impl StrikeComb {
    /// `max_period_samples` is the longest loop period (`Fs` / lowest `f₀`)
    /// the comb must support; the internal line only needs `p` times that.
    pub fn new(max_period_samples: usize) -> Self {
        let cap = (max_period_samples as f32 * STRIKE_POSITION).ceil() as usize + 2;
        Self {
            delay: DelayLine::new(cap),
            d_int: 1,
            d_frac: 0.0,
        }
    }

    /// Retune for a note whose loop period is `period_samples` (= `Fs/f₀`)
    /// and wipe stale state. The delay clamps to ≥ 1 sample (extreme treble,
    /// where the first comb null sits far above the audible spectrum anyway)
    /// and to the line's capacity (sub-audio safety).
    pub fn set_period(&mut self, period_samples: f32) {
        let d = (period_samples * STRIKE_POSITION).clamp(1.0, (self.delay.capacity() - 2) as f32);
        self.d_int = d.floor() as usize;
        self.d_frac = d - self.d_int as f32;
        self.delay.clear();
    }

    #[inline]
    pub fn tick(&mut self, x: f32) -> f32 {
        // Read before write: the most recent stored sample sits at integer
        // delay 1, so `read_frac(d_int, d_frac)` is x[n − d] exactly.
        let delayed = self.delay.read_frac(self.d_int, self.d_frac);
        self.delay.write(x);
        x - COMB_DEPTH * delayed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    const SR: f32 = 48_000.0;

    /// Steady-state RMS gain of the comb for a sinusoid at `freq`.
    fn sine_gain(comb: &mut StrikeComb, freq: f32) -> f32 {
        let omega = TAU * freq / SR;
        let n = 8_192usize;
        let mut x_sq = 0.0;
        let mut y_sq = 0.0;
        for i in 0..n {
            let x = (omega * i as f32).sin();
            let y = comb.tick(x);
            if i > 500 {
                x_sq += x * x;
                y_sq += y * y;
            }
        }
        (y_sq / x_sq).sqrt()
    }

    #[test]
    fn zero_in_zero_out() {
        let mut c = StrikeComb::new(2_400);
        c.set_period(160.0);
        for _ in 0..256 {
            assert_eq!(c.tick(0.0), 0.0);
        }
    }

    #[test]
    fn partial_at_strike_node_is_dipped() {
        // f0 = 300 Hz → period 160, d = 160/7. The 7th partial (2100 Hz)
        // has a node at the strike point: with COMB_DEPTH = 0.7 it must dip
        // to |1 − 0.7| = 0.3 (≈ −10 dB), not to a hard null — the felt's
        // finite width never cancels it completely.
        let mut c = StrikeComb::new(2_400);
        c.set_period(160.0);
        let g = sine_gain(&mut c, 7.0 * 300.0);
        assert!(
            (g - (1.0 - COMB_DEPTH)).abs() < 0.05,
            "7th partial should dip to ~0.3, gain={g}"
        );
    }

    #[test]
    fn mid_partials_are_emphasised_over_fundamental() {
        // |H(k·f0)| = |1 − g·e^(−j·2πk/7)|: partial 4 sits near the comb
        // crest at ≈ 1.66 while the fundamental stays at ≈ 0.79 — the
        // strike-position tilt real piano spectra show, without the 2×
        // boost a full-depth comb would add.
        let mut c4 = StrikeComb::new(2_400);
        c4.set_period(160.0);
        let g4 = sine_gain(&mut c4, 4.0 * 300.0);
        let mut c1 = StrikeComb::new(2_400);
        c1.set_period(160.0);
        let g1 = sine_gain(&mut c1, 300.0);
        assert!(
            (g4 - 1.66).abs() < 0.05,
            "partial 4 should peak near 1.66, got {g4}"
        );
        assert!(
            (g1 - 0.785).abs() < 0.05,
            "fundamental should be ~0.79, got {g1}"
        );
    }

    #[test]
    fn extreme_periods_clamp_without_panicking() {
        let mut c = StrikeComb::new(2_400);
        c.set_period(0.5); // shorter than 1 sample → clamps to 1
        for _ in 0..64 {
            c.tick(1.0);
        }
        c.set_period(1.0e6); // far beyond capacity → clamps to the line
        for _ in 0..64 {
            c.tick(1.0);
        }
    }

    #[test]
    fn set_period_clears_stale_state() {
        let mut c = StrikeComb::new(2_400);
        c.set_period(160.0);
        for _ in 0..64 {
            c.tick(1.0);
        }
        c.set_period(160.0);
        // First output after a re-arm sees an empty line: y = x − 0.
        assert_eq!(c.tick(0.25), 0.25);
    }
}
