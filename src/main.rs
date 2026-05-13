//! Phase 1 entry point.
//!
//! 1. Pick the default audio device, infer its sample rate.
//! 2. Build the [`Engine`] at that sample rate.
//! 3. Spin up an SPSC ring buffer for MIDI events.
//! 4. Start the cpal output adapter (owns the engine on the RT thread).
//! 5. Connect the midir input adapter to the first available port.
//! 6. Park the main thread until Ctrl-C kills the process.

use anyhow::{anyhow, Result};
use cpal::traits::{DeviceTrait, HostTrait};
use ringbuf::traits::Split;
use ringbuf::HeapRb;

use synth_piano_rs::adapters::{CpalOutput, MidirInput};
use synth_piano_rs::domain::Engine;
use synth_piano_rs::ports::{AudioOutput, MidiInput};

/// SPSC capacity. MIDI peaks at ~3 KB/s; 1024 events covers many seconds
/// of frantic playing even if the audio thread stalls for a buffer.
const MIDI_QUEUE_CAPACITY: usize = 1024;

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    log::info!("synth-piano-rs Phase 1 starting…");

    // Sample rate has to match the device — query it before constructing
    // the engine so all DSP coefficients are right the first time.
    let sample_rate = probe_default_sample_rate()?;
    log::info!("device sample rate: {sample_rate} Hz");

    let engine = Engine::new(sample_rate);

    let rb = HeapRb::<synth_piano_rs::domain::MidiEvent>::new(MIDI_QUEUE_CAPACITY);
    let (midi_tx, midi_rx) = rb.split();

    let audio = CpalOutput::start(engine, midi_rx)?;
    log::info!(
        "audio: device='{}' sr={} buf={:?}",
        audio.device_name(),
        audio.sample_rate(),
        audio.buffer_frames()
    );

    let midi = match MidirInput::connect_first(midi_tx) {
        Ok(m) => {
            log::info!("midi: connected to {:?}", m.port_name());
            Some(m)
        }
        Err(e) => {
            log::warn!("MIDI input unavailable ({e}); running silent");
            None
        }
    };

    log::info!("ready — press Ctrl-C to quit");
    // Block the main thread; the audio + midi callbacks run on their own.
    // Process exit drops both adapters and stops their streams.
    std::thread::park();

    // park() can theoretically return spuriously; explicit cleanup is fine.
    drop(midi);
    drop(audio);
    Ok(())
}

fn probe_default_sample_rate() -> Result<f32> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| anyhow!("no default output device"))?;
    let config = device.default_output_config()?;
    // cpal 0.17: SupportedStreamConfig::sample_rate() returns a SampleRate
    // type alias for u32 — cast directly.
    Ok(config.sample_rate() as f32)
}
