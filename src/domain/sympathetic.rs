//! Sympathetic-resonance bank.
//!
//! ## Why 12 strings cover the whole keyboard
//! A Karplus-Strong string with fundamental `f₀` resonates at every
//! integer multiple `k·f₀` — its loop transfer function has gain peaks at
//! all positive harmonics. If we instantiate 12 strings whose
//! fundamentals are the 12 chromatic notes of a single octave, the union
//! of their harmonic series tiles the entire frequency axis with at most
//! ~5.95 % spacing (one semitone). Every note in equal temperament lands
//! on or very close to a harmonic of some sympathetic string, so it can
//! drive that string sympathetically.
//!
//! Twelve KS instances is two orders of magnitude cheaper than the 88
//! that a literal "all undamped strings" simulation would require, with
//! essentially no audible loss because the union of their resonances
//! already covers everything an audible note could excite.
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
//!   voice_sum ──┬──────────────────────┬── soundboard ── master
//!               │                      │
//!               └── × SEND ── sympathetic ─┘
//! ```
//! The sympathetic output is summed back into the bus before the
//! soundboard so the bank's contribution gets the same modal coloration
//! as the played strings — which is what physically happens.

use crate::domain::string::KarplusStrong;
use crate::domain::voice::midi_to_hz;

/// Lowest sympathetic note (C4 = MIDI 60).
const BASE_NOTE: u8 = 60;

/// One octave of chromatic strings — covers all equal-temperament
/// pitches through harmonic relationships.
pub const N_STRINGS: usize = 12;

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

/// Output trim. The 12 strings sum and we want the bank to sit clearly
/// below the played voices.
const OUTPUT_GAIN: f32 = 0.05;

#[derive(Debug)]
pub struct Sympathetic {
    strings: [KarplusStrong; N_STRINGS],
}

impl Sympathetic {
    pub fn new(sample_rate: f32) -> Self {
        // Size the delay line to the lowest sympathetic note (C4) with a
        // 50 % margin for cascade group delay and short-rate variation.
        // 8× smaller than Voice's bass-friendly 20 Hz floor.
        let lowest_freq = midi_to_hz(BASE_NOTE);
        let max_delay = (sample_rate * 1.5 / lowest_freq).ceil() as usize;
        let mut strings: [KarplusStrong; N_STRINGS] = std::array::from_fn(|_| {
            KarplusStrong::new(sample_rate, max_delay)
        });
        // Arm each string at its assigned note and start with damper engaged.
        for (i, s) in strings.iter_mut().enumerate() {
            let note = BASE_NOTE + i as u8;
            s.pluck(midi_to_hz(note));
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
