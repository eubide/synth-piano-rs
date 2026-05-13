//! Modal soundboard model.
//!
//! ## Physical picture
//! A grand piano soundboard is a thin spruce plate (~ 1 m² × 8 mm)
//! anchored by ribs and the bridge. It has *hundreds* of mechanical modes
//! between 50 Hz and 5 kHz; each mode is a damped second-order oscillator
//! with its own natural frequency, Q, and radiation efficiency. The
//! ensemble shapes the radiated spectrum, gives the instrument its
//! characteristic "body", and adds a brief reverb-like tail.
//!
//! ## Model
//! We approximate the plate with **16 bandpass biquads in parallel**,
//! covering 89 Hz – 5.5 kHz with logarithmic-ish spacing. Each biquad is
//! a second-order resonator — exactly the DSP equivalent of a mechanical
//! mass-spring-damper. The 16 modes are not real measurements; they're a
//! plausible distribution that produces audibly "wooden" coloration.
//!
//! ## Wet / dry mix
//! In a real piano the radiated sound IS the soundboard output — there's
//! no "dry" path. Our model keeps a dry component because the strings'
//! direct output is already a usable signal; the modes add coloration on
//! top. Dry 0.7, wet 0.5 yields a balanced result where mid-range bands
//! are emphasised without the soundboard dominating the timbre.
//!
//! ## Limitations
//! - Modes are not coupled. Real soundboards have inter-mode coupling
//!   through the bridge and the radiation impedance.
//! - Q is frequency-independent within each mode — a real plate has
//!   slightly varying loss factors.
//! - No directional radiation. Output is monaural.

use crate::domain::biquad::Biquad;

/// Number of modes. 16 is the practical sweet-spot: enough to give
/// audible "wood" coloration, few enough to stay below 200 ops/sample.
pub const N_MODES: usize = 16;

/// Mode table: `(frequency_hz, Q, mode_gain)`. Frequencies are roughly
/// log-spaced with slight irregularity (real modal densities have no
/// equal-tempered structure). Q values taper from 32 in the low body
/// (where ringing is desired) to 12 at the top (where modes should
/// blend into a smooth spectral shape).
const MODES: [(f32, f32, f32); N_MODES] = [
    (89.0, 20.0, 0.70),
    (146.0, 22.0, 0.80),
    (200.0, 26.0, 0.90),
    (270.0, 30.0, 1.00),
    (360.0, 32.0, 0.95),
    (480.0, 30.0, 0.90),
    (640.0, 30.0, 0.85),
    (820.0, 30.0, 0.80),
    (1_050.0, 28.0, 0.75),
    (1_300.0, 26.0, 0.70),
    (1_650.0, 24.0, 0.65),
    (2_100.0, 22.0, 0.60),
    (2_700.0, 20.0, 0.50),
    (3_400.0, 18.0, 0.40),
    (4_300.0, 15.0, 0.30),
    (5_500.0, 12.0, 0.20),
];

/// Dry path: direct string signal scaled. Together with `WET_GAIN` it
/// produces ≈ unity peak at modal centres and a small dip elsewhere.
const DRY_GAIN: f32 = 0.7;

/// Wet path: scaling applied to the sum of modal outputs.
const WET_GAIN: f32 = 0.5;

#[derive(Debug)]
pub struct Soundboard {
    modes: [Biquad; N_MODES],
    mode_gains: [f32; N_MODES],
}

impl Soundboard {
    pub fn new(sample_rate: f32) -> Self {
        let mut modes: [Biquad; N_MODES] = [Biquad::new(); N_MODES];
        let mut mode_gains = [0.0f32; N_MODES];
        for i in 0..N_MODES {
            let (freq, q, gain) = MODES[i];
            modes[i].set_bandpass(freq, q, sample_rate);
            mode_gains[i] = gain;
        }
        Self { modes, mode_gains }
    }

    /// Process one sample. The output is `dry · x + wet · Σ mode_i(x)`.
    #[inline]
    pub fn tick(&mut self, x: f32) -> f32 {
        let mut wet = 0.0f32;
        for i in 0..N_MODES {
            wet += self.modes[i].tick(x) * self.mode_gains[i];
        }
        x * DRY_GAIN + wet * WET_GAIN
    }

    /// Wipe internal state. Useful for tests and panic-reset semantics.
    pub fn reset(&mut self) {
        for m in &mut self.modes {
            m.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silent_input_yields_silent_output() {
        let mut sb = Soundboard::new(48_000.0);
        for _ in 0..1_024 {
            assert_eq!(sb.tick(0.0), 0.0);
        }
    }

    #[test]
    fn non_zero_input_produces_signal() {
        let mut sb = Soundboard::new(48_000.0);
        let mut any_nonzero = false;
        for _ in 0..512 {
            let y = sb.tick(1.0);
            if y.abs() > 0.0 {
                any_nonzero = true;
            }
        }
        assert!(any_nonzero);
    }

    #[test]
    fn dc_input_settles_to_dry_gain() {
        // The bandpass modes all reject DC, so a held DC input passes
        // through only the dry path → output should settle to DRY_GAIN.
        let mut sb = Soundboard::new(48_000.0);
        let mut y = 0.0;
        for _ in 0..50_000 {
            y = sb.tick(1.0);
        }
        assert!((y - DRY_GAIN).abs() < 1e-2, "settled to {y}, expected ~{DRY_GAIN}");
    }

    #[test]
    fn impulse_response_decays_within_two_seconds() {
        // After roughly 100k samples of silence following a unit impulse,
        // the slowest mode should be below ~−60 dB.
        let mut sb = Soundboard::new(48_000.0);
        let mut peak_tail = 0.0f32;
        let mut x = 1.0f32;
        let n = 100_000;
        for i in 0..n {
            let y = sb.tick(x);
            x = 0.0;
            // Skip the first 0.5 s — we measure the asymptotic tail.
            if i > 80_000 {
                peak_tail = peak_tail.max(y.abs());
            }
        }
        assert!(peak_tail < 1e-3, "tail did not decay: peak={peak_tail}");
    }

    #[test]
    fn modal_centres_are_emphasised_relative_to_off_band() {
        use std::f32::consts::TAU;

        fn steady_rms(sb: &mut Soundboard, freq_hz: f32) -> f32 {
            let omega = TAU * freq_hz / 48_000.0;
            let n = 8_192usize;
            let mut sq = 0.0;
            for i in 0..n {
                let x = (omega * i as f32).sin();
                let y = sb.tick(x);
                if i > 1_000 {
                    sq += y * y;
                }
            }
            (sq / (n - 1_000) as f32).sqrt()
        }
        let mut sb_at_mode = Soundboard::new(48_000.0);
        let on_mode = steady_rms(&mut sb_at_mode, 270.0); // matches MODES[3]
        let mut sb_off_mode = Soundboard::new(48_000.0);
        // 7_000 Hz is well above every mode centre.
        let off_mode = steady_rms(&mut sb_off_mode, 7_000.0);
        assert!(
            on_mode > off_mode * 1.2,
            "on-mode RMS {on_mode} should exceed off-mode {off_mode}"
        );
    }
}
