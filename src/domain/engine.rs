//! Pure DSP engine. No I/O, no allocation after construction.
//!
//! ## Phase 2 voicing
//! The engine owns a fixed pool of [`Voice`]s but Phase 2 plays **mono** —
//! every note-on lands on slot 0, replacing the previous note. The other
//! slots stay idle so the polyphony machinery exists for Phase 6 to flip
//! a single boolean and switch to full 32-voice allocation.

use crate::domain::midi_event::MidiEvent;
use crate::domain::voice::Voice;

/// Maximum simultaneous voices once polyphony is on (Phase 6).
pub const MAX_VOICES: usize = 32;

pub struct Engine {
    sample_rate: f32,
    voices: [Voice; MAX_VOICES],
    /// Phase 2: only voice 0 is used.
    mono: bool,
    /// Master gain applied after summing voices. A single KS string can
    /// hit ±1 on the strongest plucks; 0.5 leaves headroom for the
    /// soundboard / sympathetic-resonance bus added in later phases.
    master_gain: f32,
}

impl Engine {
    pub fn new(sample_rate: f32) -> Self {
        let voices = std::array::from_fn(|_| Voice::new(sample_rate));
        Self {
            sample_rate,
            voices,
            mono: true,
            master_gain: 0.5,
        }
    }

    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Toggle monophonic mode. Phase 6 sets this to `false`.
    pub fn set_mono(&mut self, mono: bool) {
        self.mono = mono;
    }

    pub fn is_mono(&self) -> bool {
        self.mono
    }

    /// Apply one MIDI event. Cheap, allocation-free.
    pub fn handle_event(&mut self, ev: MidiEvent) {
        match ev {
            MidiEvent::NoteOn { note, velocity } => self.alloc_voice(note, velocity),
            MidiEvent::NoteOff { note } => self.release_voice(note),
            MidiEvent::AllNotesOff => {
                for v in &mut self.voices {
                    v.note_off();
                }
            }
        }
    }

    /// Fill `buf` with one mono sample per slot.
    pub fn render(&mut self, buf: &mut [f32]) {
        for s in buf.iter_mut() {
            let mut acc = 0.0f32;
            if self.mono {
                acc += self.voices[0].tick();
            } else {
                for v in &mut self.voices {
                    acc += v.tick();
                }
            }
            *s = acc * self.master_gain;
        }
    }

    fn alloc_voice(&mut self, note: u8, velocity: u8) {
        if self.mono {
            self.voices[0].note_on(note, velocity);
            return;
        }
        // Same-note retrigger steals the previous voice for that note.
        for v in &mut self.voices {
            if v.is_active() && v.note() == note {
                v.note_on(note, velocity);
                return;
            }
        }
        for v in &mut self.voices {
            if !v.is_active() {
                v.note_on(note, velocity);
                return;
            }
        }
        // No idle slot: steal voice 0. Phase 6 will use a proper LRU.
        self.voices[0].note_on(note, velocity);
    }

    fn release_voice(&mut self, note: u8) {
        if self.mono {
            if self.voices[0].is_active() && self.voices[0].note() == note {
                self.voices[0].note_off();
            }
            return;
        }
        for v in &mut self.voices {
            if v.is_active() && v.note() == note {
                v.note_off();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_into_silent_buffer_stays_silent() {
        let mut eng = Engine::new(48_000.0);
        let mut buf = [0.0; 64];
        eng.render(&mut buf);
        assert!(buf.iter().all(|s| *s == 0.0));
    }

    #[test]
    fn note_on_produces_non_zero_output() {
        let mut eng = Engine::new(48_000.0);
        eng.handle_event(MidiEvent::note_on(69, 100));
        let mut buf = [0.0; 4_096];
        eng.render(&mut buf);
        assert!(buf.iter().any(|s| s.abs() > 0.0));
    }

    #[test]
    fn note_off_eventually_silences_output() {
        let mut eng = Engine::new(48_000.0);
        eng.handle_event(MidiEvent::note_on(60, 100));
        let mut buf = [0.0; 512];
        eng.render(&mut buf);
        eng.handle_event(MidiEvent::note_off(60));
        // Render past the damper tail (τ=80 ms, gate 1e-4 → ~0.74 s).
        let mut tail = vec![0.0; 60_000];
        eng.render(&mut tail);
        let last_window = &tail[tail.len() - 256..];
        assert!(last_window.iter().all(|s| *s == 0.0));
    }

    #[test]
    fn all_notes_off_silences_everything() {
        let mut eng = Engine::new(48_000.0);
        for n in 60..68 {
            eng.handle_event(MidiEvent::note_on(n, 100));
        }
        eng.handle_event(MidiEvent::AllNotesOff);
        let mut tail = vec![0.0; 60_000];
        eng.render(&mut tail);
        let last_window = &tail[tail.len() - 256..];
        assert!(last_window.iter().all(|s| *s == 0.0));
    }

    #[test]
    fn mono_mode_only_uses_voice_zero() {
        let mut eng = Engine::new(48_000.0);
        assert!(eng.is_mono());
        // Trigger several notes; only voice 0 should ever be active.
        eng.handle_event(MidiEvent::note_on(60, 100));
        eng.handle_event(MidiEvent::note_on(64, 100));
        eng.handle_event(MidiEvent::note_on(67, 100));
        for i in 1..MAX_VOICES {
            assert!(!eng.voices[i].is_active(), "voice {i} should be idle");
        }
        assert!(eng.voices[0].is_active());
        assert_eq!(eng.voices[0].note(), 67);
    }

    #[test]
    fn poly_mode_uses_multiple_voices() {
        let mut eng = Engine::new(48_000.0);
        eng.set_mono(false);
        eng.handle_event(MidiEvent::note_on(60, 100));
        eng.handle_event(MidiEvent::note_on(64, 100));
        eng.handle_event(MidiEvent::note_on(67, 100));
        let active = eng.voices.iter().filter(|v| v.is_active()).count();
        assert_eq!(active, 3);
    }
}
