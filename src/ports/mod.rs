//! Driving / driven ports for the engine (hexagonal architecture).
//!
//! Adapters in `crate::adapters` implement these so the engine never
//! references `cpal` or `midir` directly. Tests can use trivial fakes.

pub mod audio_output;
pub mod midi_input;

pub use audio_output::AudioOutput;
pub use midi_input::MidiInput;
