//! Hexagonal physical-modelling piano synthesizer.
//!
//! - `domain`: pure DSP, sample-rate parametric, no I/O.
//! - `ports`: traits the engine consumes (driving + driven).
//! - `adapters`: concrete I/O implementations (`cpal`, `midir`).

pub mod adapters;
pub mod domain;
pub mod ports;
