//! Sympathetic-resonance bank.
//!
//! ## Why two octaves of strings cover the whole keyboard
//! A Karplus-Strong string with fundamental `f₀` resonates at every
//! integer multiple `k·f₀` — its loop transfer function has gain peaks at
//! all positive harmonics. A single chromatic octave already tiles the
//! frequency axis through those harmonics, but harmonic coupling is weak
//! and treble-biased: a low note whose fundamental sits *below* the bank
//! only drives the bank's strings at its upper partials, so the deep,
//! enveloping "bass bloom" of a real pedalled piano is missing.
//!
//! We therefore span **two chromatic octaves (C2..B3, 24 strings)**. The
//! lower octave gives bass and tenor notes a sympathetic string at (or an
//! octave from) their own fundamental — the strongest coupling there is —
//! restoring the bass bloom, while the union of all 24 harmonic series
//! still blankets the treble. 24 KS instances is far cheaper than the 88
//! a literal "all undamped strings" simulation would need, with no audible
//! gap in coverage.
//!
//! ## How the pedal couples in
//! - **Pedal up (default)**: every string's loop gain is set just below 1.0
//!   so the bank dies off in ~50 ms — modelling the felt damper resting
//!   on the strings.
//! - **Pedal down**: loop gain becomes 1.0 — only the loop LPF removes
//!   energy now, so the strings can ring for several seconds. Notes
//!   played during this window inject a fraction of their output into
//!   every sympathetic string; energy accumulates at frequencies that
//!   match a sympathetic's harmonics, giving the audible "halo".
//!
//! ## Signal flow at the engine level
//! ```text
//!   voice_sum ──┬──────────────────────┬── soundboard L/R ── master
//!               │                      │
//!               └── × SEND ── sympathetic ─┘ (centred)
//! ```
//! The sympathetic output is summed back into the bus before *both*
//! soundboard plates so the bank's contribution gets the same modal
//! coloration as the played strings — which is what physically happens. It
//! is fed the unpanned voice sum and returns to centre, because the whole
//! undamped bank rings as one body rather than tracking the struck note's
//! position along the bridge.

use crate::domain::string::KarplusStrong;
use crate::domain::voice::stretched_midi_to_hz;

/// Lowest sympathetic note (C2 = MIDI 36). The bank spans up from here.
const BASE_NOTE: u8 = 36;

/// Two chromatic octaves of strings (C2..B3) — the lower octave restores
/// sympathetic coupling for the bass/tenor register; harmonics of all 24
/// cover everything above.
pub const N_STRINGS: usize = 24;

/// Loop gain when the damper is on. The dampers of a real piano are
/// felts that *strongly* mute the strings — typical decay times when a
/// damper engages are 30–100 ms. With `0.85` an A4 sympathetic loses
/// ≈ 14 % per cycle, hitting −60 dB after roughly 60 ms.
const PEDAL_UP_LOOP_GAIN: f32 = 0.85;

/// Loop gain when the pedal is depressed — purely the LPF's frequency-
/// dependent loss remains, so the strings ring for seconds.
const PEDAL_DOWN_LOOP_GAIN: f32 = 1.0;

/// Send level applied to the voice-bus signal that excites this bank.
/// Modest by design — a real piano's sympathetic coupling is small, and
/// the listener should hear "halo" rather than a parallel copy.
pub const EXCITATION_SEND: f32 = 0.08;

/// Output trim. The 24 strings sum and we want the bank to sit clearly
/// below the played voices; trimmed down from the 12-string value to keep
/// the halo at the same modest level now that twice as many strings sum.
const OUTPUT_GAIN: f32 = 0.035;

#[derive(Debug)]
pub struct Sympathetic {
    strings: [KarplusStrong; N_STRINGS],
}

impl Sympathetic {
    pub fn new(sample_rate: f32) -> Self {
        // Size the delay line to the lowest sympathetic note — `BASE_NOTE`,
        // the longest delay this bank ever needs (every string is ≥ it). The
        // 1.5× factor over the raw period leaves headroom for the cascade
        // group delay; DelayLine then rounds the capacity up to the next
        // power of two. Deriving `lowest_freq` from `BASE_NOTE` keeps this
        // self-consistent if the bank's range is ever changed.
        let lowest_freq = stretched_midi_to_hz(BASE_NOTE);
        let max_delay = (sample_rate * 1.5 / lowest_freq).ceil() as usize;
        let mut strings: [KarplusStrong; N_STRINGS] =
            std::array::from_fn(|_| KarplusStrong::new(sample_rate, max_delay));
        // Arm each string at its assigned note (stretch-tuned, so it lines
        // up with the played strings) and start with damper engaged.
        for (i, s) in strings.iter_mut().enumerate() {
            let note = BASE_NOTE + i as u8;
            s.pluck(stretched_midi_to_hz(note));
            s.set_loop_gain(PEDAL_UP_LOOP_GAIN);
        }
        Self { strings }
    }

    /// Mirror the sustain-pedal state.
    pub fn set_pedal(&mut self, down: bool) {
        let gain = if down {
            PEDAL_DOWN_LOOP_GAIN
        } else {
            PEDAL_UP_LOOP_GAIN
        };
        for s in &mut self.strings {
            s.set_loop_gain(gain);
        }
    }

    /// Drive the bank with the (small) sympathetic send and return the
    /// summed, trimmed output.
    #[inline]
    pub fn tick(&mut self, voice_bus: f32) -> f32 {
        let exc = voice_bus * EXCITATION_SEND;
        let mut sum = 0.0f32;
        for s in &mut self.strings {
            sum += s.tick(exc);
        }
        sum * OUTPUT_GAIN
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rms(samples: &[f32]) -> f32 {
        let s: f32 = samples.iter().map(|x| x * x).sum();
        (s / samples.len() as f32).sqrt()
    }

    #[test]
    fn fresh_sympathetic_is_silent_with_zero_input() {
        let mut s = Sympathetic::new(48_000.0);
        for _ in 0..1_024 {
            assert_eq!(s.tick(0.0), 0.0);
        }
    }

    #[test]
    fn excitation_produces_audible_ring() {
        let mut s = Sympathetic::new(48_000.0);
        s.set_pedal(true);
        // Drive an impulse, then let the bank ring.
        let mut buf = vec![0.0; 4_096];
        buf[0] = s.tick(1.0);
        for v in buf.iter_mut().skip(1) {
            *v = s.tick(0.0);
        }
        // The bank should be ringing well past the first sample.
        let tail = &buf[1_024..];
        assert!(rms(tail) > 1e-5, "tail silent: {}", rms(tail));
    }

    #[test]
    fn pedal_down_sustains_longer_than_pedal_up() {
        fn tail_rms_after_impulse(pedal_down: bool) -> f32 {
            let mut s = Sympathetic::new(48_000.0);
            s.set_pedal(pedal_down);
            let mut buf = vec![0.0; 24_000]; // 500 ms
            buf[0] = s.tick(1.0);
            for v in buf.iter_mut().skip(1) {
                *v = s.tick(0.0);
            }
            let tail = &buf[buf.len() - 2_048..];
            rms(tail)
        }
        let down = tail_rms_after_impulse(true);
        let up = tail_rms_after_impulse(false);
        assert!(
            down > up * 3.0,
            "pedal-down tail ({down}) should sustain longer than pedal-up ({up})"
        );
    }

    #[test]
    fn pedal_up_eventually_silences_ring() {
        // After flipping the pedal back up, the bank must drain.
        let mut s = Sympathetic::new(48_000.0);
        s.set_pedal(true);
        // Pump energy in for 100 ms.
        for _ in 0..4_800 {
            s.tick(0.5);
        }
        s.set_pedal(false);
        // Drain for 1 s with no input.
        let mut buf = vec![0.0; 48_000];
        for v in buf.iter_mut() {
            *v = s.tick(0.0);
        }
        let tail = &buf[buf.len() - 1_024..];
        assert!(rms(tail) < 1e-3, "bank still ringing: {}", rms(tail));
    }
}
