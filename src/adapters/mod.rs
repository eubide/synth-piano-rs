//! Driving / driven adapters. Translate the outside world (cpal, midir)
//! into the engine's [`MidiEvent`](crate::domain::MidiEvent) language and
//! consume its rendered mono samples.

pub mod cpal_output;
pub mod midir_input;

pub use cpal_output::CpalOutput;
pub use midir_input::MidirInput;
