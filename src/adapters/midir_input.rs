//! midir input adapter.
//!
//! Connects to the first available MIDI input, parses bytes into the
//! engine's [`MidiEvent`](crate::domain::MidiEvent) vocabulary, and pushes
//! them through the SPSC ring buffer that the audio thread drains.
//!
//! Understands Note On, Note Off, sustain pedal (CC 64) and `All Notes
//! Off` (CC 123). Other CCs, SysEx and pitch bend are dropped silently.

use anyhow::{anyhow, Context, Result};
use midir::{MidiInput as MidirMidiInput, MidiInputConnection};
use ringbuf::traits::Producer;
use ringbuf::HeapProd;

use crate::domain::MidiEvent;
use crate::ports::MidiInput;

pub struct MidirInput {
    _connection: MidiInputConnection<HeapProd<MidiEvent>>,
    port_name: String,
}

impl MidirInput {
    pub fn connect_first(midi_tx: HeapProd<MidiEvent>) -> Result<Self> {
        let input = MidirMidiInput::new("synth-piano-rs").context("midir init")?;
        let ports = input.ports();
        let port = ports
            .first()
            .ok_or_else(|| anyhow!("no MIDI input devices available"))?;
        let port_name = input.port_name(port).unwrap_or_else(|_| "<unknown>".into());

        log::info!("midir: connecting to '{port_name}'");

        let connection = input
            .connect(
                port,
                "synth-piano-rs",
                |_timestamp, message, tx| {
                    if let Some(ev) = parse_message(message) {
                        // try_push returns Err if the consumer (audio thread)
                        // is falling behind. We drop the event rather than
                        // block — preferable to a stuck callback.
                        if tx.try_push(ev).is_err() {
                            log::warn!("midi ring buffer full, dropping event");
                        }
                    }
                },
                midi_tx,
            )
            .map_err(|e| anyhow!("midir connect failed: {e}"))?;

        Ok(Self {
            _connection: connection,
            port_name,
        })
    }
}

impl MidiInput for MidirInput {
    fn port_name(&self) -> Option<&str> {
        Some(&self.port_name)
    }
}

/// Translate a raw MIDI message into a domain event. Returns `None` for
/// messages we don't yet care about (CCs other than All Notes Off, SysEx,
/// pitch bend, etc.). Kept as a free function so it is unit-testable.
pub(crate) fn parse_message(msg: &[u8]) -> Option<MidiEvent> {
    if msg.len() < 2 {
        return None;
    }
    let status = msg[0] & 0xF0;
    match status {
        // Note On: velocity 0 is conventionally Note Off.
        0x90 if msg.len() >= 3 => {
            let note = msg[1];
            let velocity = msg[2];
            if velocity == 0 {
                Some(MidiEvent::NoteOff { note })
            } else {
                Some(MidiEvent::NoteOn { note, velocity })
            }
        }
        0x80 if msg.len() >= 3 => Some(MidiEvent::NoteOff { note: msg[1] }),
        // Control Change: CC 64 (sustain pedal) and CC 123 (All Notes Off).
        0xB0 if msg.len() >= 3 => match msg[1] {
            64 => Some(MidiEvent::SustainPedal { down: msg[2] >= 64 }),
            123 => Some(MidiEvent::AllNotesOff),
            _ => None,
        },
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn note_on_with_velocity_parses() {
        assert_eq!(
            parse_message(&[0x90, 60, 100]),
            Some(MidiEvent::NoteOn {
                note: 60,
                velocity: 100
            })
        );
    }

    #[test]
    fn note_on_with_velocity_zero_is_note_off() {
        assert_eq!(
            parse_message(&[0x90, 60, 0]),
            Some(MidiEvent::NoteOff { note: 60 })
        );
    }

    #[test]
    fn note_off_parses() {
        assert_eq!(
            parse_message(&[0x80, 60, 0]),
            Some(MidiEvent::NoteOff { note: 60 })
        );
    }

    #[test]
    fn all_notes_off_parses() {
        assert_eq!(parse_message(&[0xB0, 123, 0]), Some(MidiEvent::AllNotesOff));
    }

    #[test]
    fn unknown_cc_is_ignored() {
        assert_eq!(parse_message(&[0xB0, 7, 100]), None);
    }

    #[test]
    fn sustain_pedal_down_parses_at_value_above_64() {
        assert_eq!(
            parse_message(&[0xB0, 64, 127]),
            Some(MidiEvent::SustainPedal { down: true })
        );
        assert_eq!(
            parse_message(&[0xB0, 64, 64]),
            Some(MidiEvent::SustainPedal { down: true })
        );
    }

    #[test]
    fn sustain_pedal_up_parses_at_value_below_64() {
        assert_eq!(
            parse_message(&[0xB0, 64, 0]),
            Some(MidiEvent::SustainPedal { down: false })
        );
        assert_eq!(
            parse_message(&[0xB0, 64, 63]),
            Some(MidiEvent::SustainPedal { down: false })
        );
    }

    #[test]
    fn channel_byte_is_stripped() {
        // Note On on channel 5 (status 0x94).
        assert_eq!(
            parse_message(&[0x94, 60, 100]),
            Some(MidiEvent::NoteOn {
                note: 60,
                velocity: 100
            })
        );
    }

    #[test]
    fn truncated_message_is_ignored() {
        assert_eq!(parse_message(&[0x90, 60]), None);
        assert_eq!(parse_message(&[]), None);
    }
}
