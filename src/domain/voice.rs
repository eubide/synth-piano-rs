//! Phase 3 voice: hammer-excited Karplus-Strong string with damper envelope.
//!
//! ## Signal path
//! `Hammer.tick()  →  KarplusStrong.tick(exc)  →  ·damper_gain  →  output`
//!
//! - Hammer produces the velocity-shaped excitation pulse over its
//!   contact time and then goes silent.
//! - String resonates indefinitely (modulo loop-filter losses) until the
//!   damper gain envelope falls below the deactivation threshold after a
//!   `note_off`.
//!
//! ## Why the seed argument is gone
//! Phase 2 needed a per-voice RNG seed for the noise burst. Phase 3 has no
//! random component (deterministic raised-cosine pulse), so the seed is no
//! longer needed. We keep `Voice::new(sample_rate)` simple.

use crate::domain::hammer::Hammer;
use crate::domain::string::KarplusStrong;

/// MIDI note → frequency in Hz (A4 = 440 Hz, MIDI 69).
pub fn midi_to_hz(note: u8) -> f32 {
    440.0 * ((note as f32 - 69.0) / 12.0).exp2()
}

/// Damper decay time constant when the key is released, in seconds.
const DAMPER_TAU_SECS: f32 = 0.08;

/// Output multiplier below which the voice is considered silent (-80 dB).
const DAMPER_GATE_THRESHOLD: f32 = 1.0e-4;

/// Lowest frequency the string delay must accommodate. 20 Hz covers below
/// A0 (27.5 Hz) with slack for later detuning.
const MIN_FREQUENCY_HZ: f32 = 20.0;

#[derive(Debug)]
pub struct Voice {
    sample_rate: f32,
    string: KarplusStrong,
    hammer: Hammer,
    note: u8,
    damper_gain: f32,
    damper_decay: f32,
    released: bool,
}

impl Voice {
    pub fn new(sample_rate: f32) -> Self {
        let max_delay = (sample_rate / MIN_FREQUENCY_HZ).ceil() as usize;
        let damper_decay = (-1.0 / (DAMPER_TAU_SECS * sample_rate)).exp();
        Self {
            sample_rate,
            string: KarplusStrong::new(sample_rate, max_delay),
            hammer: Hammer::new(sample_rate),
            note: 0,
            damper_gain: 0.0,
            damper_decay,
            released: false,
        }
    }

    pub fn note(&self) -> u8 {
        self.note
    }

    pub fn is_active(&self) -> bool {
        self.string.is_active()
    }

    pub fn note_on(&mut self, note: u8, velocity: u8) {
        self.note = note;
        let freq = midi_to_hz(note);
        let v = (velocity as f32 / 127.0).clamp(0.0, 1.0);
        self.string.pluck(freq);
        self.hammer.fire(v);
        self.damper_gain = 1.0;
        self.released = false;
    }

    pub fn note_off(&mut self) {
        if self.string.is_active() {
            self.released = true;
        }
    }

    #[inline]
    pub fn tick(&mut self) -> f32 {
        if !self.string.is_active() {
            return 0.0;
        }
        let exc = self.hammer.tick();
        let s = self.string.tick(exc);
        let out = s * self.damper_gain;
        if self.released {
            self.damper_gain *= self.damper_decay;
            if self.damper_gain < DAMPER_GATE_THRESHOLD {
                self.string.deactivate();
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
        assert_eq!(v.tick(), 0.0);
    }

    #[test]
    fn note_on_activates_voice_and_produces_signal() {
        let mut v = Voice::new(48_000.0);
        v.note_on(69, 100);
        assert!(v.is_active());
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
    fn note_off_eventually_returns_to_idle() {
        let mut v = Voice::new(48_000.0);
        v.note_on(60, 127);
        for _ in 0..1_000 {
            v.tick();
        }
        v.note_off();
        for _ in 0..60_000 {
            v.tick();
        }
        assert!(!v.is_active());
    }

    #[test]
    fn velocity_zero_produces_silence() {
        // Hammer is inert at v=0, so the string never receives energy.
        let mut v = Voice::new(48_000.0);
        v.note_on(60, 0);
        for _ in 0..4_096 {
            assert_eq!(v.tick(), 0.0);
        }
    }

    /// Brightness validation: hard strikes inject more energy above 2 kHz
    /// into the string. We use an *absolute* HP-RMS measurement over a short
    /// post-attack window because the loop filter normalises spectral
    /// *shape* to the string's harmonic resonances within tens of ms — only
    /// the early post-attack tail still carries the hammer's spectral
    /// imprint.
    #[test]
    fn hard_strike_has_more_high_frequency_content_than_soft() {
        use crate::domain::filter::OnePoleLowpass;

        fn early_hp_rms(velocity: u8) -> f32 {
            let mut v = Voice::new(48_000.0);
            v.note_on(60, velocity); // C4
            let mut buf = vec![0.0; 1_024]; // 21 ms
            for s in buf.iter_mut() {
                *s = v.tick();
            }
            // Skip the 1.5 ms hammer pulse itself.
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
        // v² scaling alone yields ≈ 16× amplitude difference. With the
        // additional felt-compression filtering the HF gap is much wider —
        // a 5× threshold is conservative and unlikely to false-positive on
        // refactors that quietly remove the velocity-dependent cutoff.
        assert!(
            hard > soft * 5.0,
            "expected harder strike to inject more HF energy: hard={hard} soft={soft}"
        );
    }
}
