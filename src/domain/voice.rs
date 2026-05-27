//! Phase 5 voice: hammer-excited [`StringGroup`] (1–3 detuned strings) with
//! a damper envelope.
//!
//! ## Signal path
//! `Hammer.tick() → StringGroup.tick(exc) → · damper_gain → output`
//!
//! - Number of strings depends on the MIDI note range (see
//!   [`strings_for_note`]).
//! - Damper applies to the summed string output, so a release fades the
//!   whole group together (matching how a real damper bar engages every
//!   string for a given key).
//!
//! ## What the voice exposes for the allocator
//! [`Voice::is_active`] and [`Voice::is_released`] together let the engine
//! distinguish *holding* (key down, full sustain) from *releasing* (key
//! up, damper decaying) — the allocator prefers to steal releasing voices
//! because they are perceptually closer to silent.

use crate::domain::hammer::Hammer;
use crate::domain::string_group::{strings_for_note, StringGroup};

/// MIDI note → frequency in Hz (A4 = 440 Hz, MIDI 69).
pub fn midi_to_hz(note: u8) -> f32 {
    440.0 * ((note as f32 - 69.0) / 12.0).exp2()
}

const DAMPER_GATE_THRESHOLD: f32 = 1.0e-4;
const MIN_FREQUENCY_HZ: f32 = 20.0;

/// Per-sample damper-gain decay factor for `note`. Real piano dampers are
/// felts whose engagement time depends on the string they meet: bass
/// strings carry much more energy and the felt takes ≈ 100 ms to mute
/// them; treble strings stop almost instantly under their tiny dampers.
/// We linearly taper τ across the playable range A0..C8.
fn damper_decay_per_sample(note: u8, sample_rate: f32) -> f32 {
    let t = (note.min(108).saturating_sub(21) as f32 / 87.0).clamp(0.0, 1.0);
    let tau_secs = 0.12 - t * 0.08;
    (-1.0 / (tau_secs * sample_rate)).exp()
}

#[derive(Debug)]
pub struct Voice {
    sample_rate: f32,
    strings: StringGroup,
    hammer: Hammer,
    note: u8,
    damper_gain: f32,
    damper_decay: f32,
    released: bool,
}

impl Voice {
    pub fn new(sample_rate: f32) -> Self {
        let max_delay = (sample_rate / MIN_FREQUENCY_HZ).ceil() as usize;
        Self {
            sample_rate,
            strings: StringGroup::new(sample_rate, max_delay),
            hammer: Hammer::new(sample_rate),
            note: 0,
            damper_gain: 0.0,
            // Overwritten on every note_on with a note-dependent value.
            damper_decay: 1.0,
            released: false,
        }
    }

    pub fn note(&self) -> u8 {
        self.note
    }

    pub fn is_active(&self) -> bool {
        self.strings.is_active()
    }

    /// True while the damper is decaying after a note-off (and the
    /// strings still ring). The allocator uses this to prefer stealing
    /// fading voices over held ones.
    pub fn is_released(&self) -> bool {
        self.released
    }

    pub fn note_on(&mut self, note: u8, velocity: u8) {
        self.note = note;
        let freq = midi_to_hz(note);
        let n_strings = strings_for_note(note);
        let v = (velocity as f32 / 127.0).clamp(0.0, 1.0);
        self.strings.pluck(freq, n_strings);
        self.hammer.fire(v);
        self.damper_decay = damper_decay_per_sample(note, self.sample_rate);
        self.damper_gain = 1.0;
        self.released = false;
    }

    pub fn note_off(&mut self) {
        if self.strings.is_active() {
            self.released = true;
        }
    }

    /// Lift the damper off a voice that was already in release decay.
    /// `damper_gain` keeps its current (partially decayed) value — the
    /// felt had been engaging before the pedal pulled it back, so the
    /// string continues to ring at the amplitude it had reached.
    pub fn cancel_release(&mut self) {
        self.released = false;
    }

    #[inline]
    pub fn tick(&mut self) -> f32 {
        if !self.strings.is_active() {
            return 0.0;
        }
        let exc = self.hammer.tick();
        let s = self.strings.tick(exc);
        let out = s * self.damper_gain;
        if self.released {
            self.damper_gain *= self.damper_decay;
            if self.damper_gain < DAMPER_GATE_THRESHOLD {
                self.strings.deactivate();
                self.damper_gain = 0.0;
                self.released = false;
            }
        }
        out
    }

    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a4_resolves_to_440_hz() {
        assert!((midi_to_hz(69) - 440.0).abs() < 1e-3);
    }

    #[test]
    fn c4_resolves_to_about_261_63() {
        assert!((midi_to_hz(60) - 261.625_56).abs() < 1e-2);
    }

    #[test]
    fn fresh_voice_is_idle_and_silent() {
        let mut v = Voice::new(48_000.0);
        assert!(!v.is_active());
        assert!(!v.is_released());
        assert_eq!(v.tick(), 0.0);
    }

    #[test]
    fn note_on_activates_voice_and_produces_signal() {
        let mut v = Voice::new(48_000.0);
        v.note_on(69, 100);
        assert!(v.is_active());
        assert!(!v.is_released());
        let mut any_nonzero = false;
        for _ in 0..1_024 {
            if v.tick().abs() > 0.0 {
                any_nonzero = true;
                break;
            }
        }
        assert!(any_nonzero);
    }

    #[test]
    fn note_off_sets_released_then_drains_to_idle() {
        let mut v = Voice::new(48_000.0);
        v.note_on(60, 127);
        for _ in 0..1_000 {
            v.tick();
        }
        v.note_off();
        assert!(v.is_released());
        for _ in 0..60_000 {
            v.tick();
        }
        assert!(!v.is_active());
        assert!(!v.is_released());
    }

    #[test]
    fn velocity_zero_produces_silence() {
        let mut v = Voice::new(48_000.0);
        v.note_on(60, 0);
        for _ in 0..4_096 {
            assert_eq!(v.tick(), 0.0);
        }
    }

    #[test]
    fn bass_note_uses_single_string() {
        // Note 30 (F#1) falls into the singles range.
        let mut v = Voice::new(48_000.0);
        v.note_on(30, 100);
        // We don't expose string count directly from the voice; verify
        // indirectly via the underlying group.
        let count = v.strings.active_count();
        assert_eq!(count, 1, "expected bass to be single-string, got {count}");
    }

    #[test]
    fn treble_note_uses_three_strings() {
        let mut v = Voice::new(48_000.0);
        v.note_on(72, 100); // C5
        assert_eq!(v.strings.active_count(), 3);
    }

    #[test]
    fn hard_strike_has_more_high_frequency_content_than_soft() {
        use crate::domain::filter::OnePoleLowpass;

        fn early_hp_rms(velocity: u8) -> f32 {
            let mut v = Voice::new(48_000.0);
            v.note_on(60, velocity);
            let mut buf = vec![0.0; 1_024];
            for s in buf.iter_mut() {
                *s = v.tick();
            }
            let s = &buf[100..];
            let mut lp = OnePoleLowpass::new();
            lp.set_cutoff(2_000.0, 48_000.0);
            let mut hp_sq = 0.0;
            for &x in s {
                let lp_out = lp.tick(x);
                let hp = x - lp_out;
                hp_sq += hp * hp;
            }
            (hp_sq / s.len() as f32).sqrt()
        }
        let hard = early_hp_rms(120);
        let soft = early_hp_rms(30);
        // 3× is the qualitative-correctness threshold: it confirms the
        // felt-compression LPF actually injects HF differentially with
        // velocity. The absolute ratio depends on `AMPLITUDE_WARP` and
        // is fine-tuned by ear, not by this test.
        assert!(
            hard > soft * 3.0,
            "expected harder strike to inject more HF energy: hard={hard} soft={soft}"
        );
    }
}
