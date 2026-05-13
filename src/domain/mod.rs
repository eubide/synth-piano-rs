//! Pure DSP / domain layer. No `cpal`, no `midir`, no allocation after
//! construction. Everything here is unit-testable offline.

pub mod delay_line;
pub mod dispersion;
pub mod engine;
pub mod filter;
pub mod hammer;
pub mod midi_event;
pub mod rng;
pub mod string;
pub mod string_group;
pub mod voice;

pub use engine::{Engine, MAX_VOICES};
pub use midi_event::MidiEvent;
