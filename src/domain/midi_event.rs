/// Domain-level MIDI event. Adapters translate raw bytes into these.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MidiEvent {
    /// Note on with MIDI note number (0..127) and velocity (1..127).
    /// Velocity 0 is normalised to `NoteOff` at the adapter boundary.
    NoteOn { note: u8, velocity: u8 },
    /// Note off — velocity ignored for the moment.
    NoteOff { note: u8 },
    /// All Notes Off (CC 123). Silences every active voice immediately.
    AllNotesOff,
    /// Sustain pedal (CC 64). `down` follows the MIDI convention:
    /// values ≥ 64 → `true`, < 64 → `false`. While `down`, note-offs are
    /// deferred until the pedal goes up; sympathetic resonance is freed
    /// to ring.
    SustainPedal { down: bool },
}

impl MidiEvent {
    /// Convenience used by tests / offline rendering.
    pub fn note_on(note: u8, velocity: u8) -> Self {
        Self::NoteOn { note, velocity }
    }

    pub fn note_off(note: u8) -> Self {
        Self::NoteOff { note }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equality_holds_for_same_payload() {
        assert_eq!(
            MidiEvent::note_on(60, 100),
            MidiEvent::NoteOn { note: 60, velocity: 100 }
        );
        assert_eq!(MidiEvent::note_off(60), MidiEvent::NoteOff { note: 60 });
    }
}
