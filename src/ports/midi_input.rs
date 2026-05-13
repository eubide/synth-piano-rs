//! MIDI input port.
//!
//! Adapters parse raw bytes and forward `MidiEvent`s through an SPSC
//! lock-free ring buffer that the audio thread drains at the start of
//! every callback. The port is a "started connection handle" — drop it
//! to stop receiving MIDI.

pub trait MidiInput {
    /// Human-readable name of the active input port, if any.
    fn port_name(&self) -> Option<&str>;
}
