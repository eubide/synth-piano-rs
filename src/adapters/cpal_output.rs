//! cpal output adapter.
//!
//! Targets the consigna spec: 48 kHz / 256 samples. If the default device
//! does not advertise that exact pair we fall back to the closest available
//! and warn; the engine adapts to whatever sample rate it ends up with.
//!
//! ## Real-time discipline
//! - No heap allocations inside the callback. A stereo pair of scratch
//!   buffers is allocated at construction (`MAX_BUFFER_FRAMES` each) and
//!   reused every block.
//! - The consumer end of the MIDI ring buffer is drained at the start of
//!   each block. Engine state never crosses thread boundaries.
//! - Errors during stream callback are logged at most every N occurrences
//!   to avoid log-storms hurting latency.

use anyhow::{anyhow, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ringbuf::traits::Consumer;
use ringbuf::HeapCons;

use crate::domain::{Engine, MidiEvent};
use crate::ports::AudioOutput;

/// Hard upper bound on the cpal buffer size we will service. The consigna
/// asks for 256, but some hosts deliver larger blocks (especially when the
/// device is shared). We size each scratch buffer to cover the worst case.
const MAX_BUFFER_FRAMES: usize = 8_192;

/// Target sample rate from the spec.
pub const TARGET_SAMPLE_RATE: u32 = 48_000;
/// Target buffer size from the spec.
pub const TARGET_BUFFER_FRAMES: u32 = 256;

pub struct CpalOutput {
    _stream: cpal::Stream,
    sample_rate: f32,
    buffer_frames: Option<u32>,
    device_name: String,
}

impl CpalOutput {
    pub fn start(engine: Engine, midi_rx: HeapCons<MidiEvent>) -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| anyhow!("no default output device"))?;
        // `description()` replaces deprecated `name()` and returns a richer label.
        let device_name = device
            .description()
            .map(|d| d.name().to_string())
            .unwrap_or_else(|_| "<unknown>".into());

        let (config, sample_format) = pick_config(&device)?;
        let sample_rate = config.sample_rate as f32;
        let buffer_frames = match config.buffer_size {
            cpal::BufferSize::Fixed(n) => Some(n),
            cpal::BufferSize::Default => None,
        };

        log::info!(
            "cpal: device='{device_name}' sample_rate={sample_rate} buffer={:?} channels={} format={:?}",
            config.buffer_size,
            config.channels,
            sample_format,
        );

        // Sanity-check: the engine was built for the device's sample rate.
        // If the caller already constructed `Engine::new(sr)` with a stale
        // value we still play, but at the wrong pitch. Make this loud.
        if (engine.sample_rate() - sample_rate).abs() > 0.5 {
            log::warn!(
                "engine sample rate {} != device {}; pitch will be off",
                engine.sample_rate(),
                sample_rate
            );
        }

        let stream = match sample_format {
            cpal::SampleFormat::F32 => build_stream::<f32>(&device, &config, engine, midi_rx)?,
            cpal::SampleFormat::I16 => build_stream::<i16>(&device, &config, engine, midi_rx)?,
            cpal::SampleFormat::U16 => build_stream::<u16>(&device, &config, engine, midi_rx)?,
            other => {
                return Err(anyhow!("unsupported sample format: {other:?}"));
            }
        };

        stream.play().context("failed to start audio stream")?;

        Ok(Self {
            _stream: stream,
            sample_rate,
            buffer_frames,
            device_name,
        })
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }
}

impl AudioOutput for CpalOutput {
    fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    fn buffer_frames(&self) -> Option<u32> {
        self.buffer_frames
    }
}

/// Build a stream-config matching the consigna spec when the device can
/// honour it, falling back to the device default otherwise.
fn pick_config(device: &cpal::Device) -> Result<(cpal::StreamConfig, cpal::SampleFormat)> {
    let target_sr: cpal::SampleRate = TARGET_SAMPLE_RATE;

    let supported_iter = device
        .supported_output_configs()
        .context("device does not expose supported configs")?;

    // Prefer a config that includes 48 kHz with f32 samples.
    let mut chosen = None;
    for cfg in supported_iter {
        let includes_target =
            cfg.min_sample_rate() <= target_sr && cfg.max_sample_rate() >= target_sr;
        if includes_target {
            chosen = Some(cfg.with_sample_rate(target_sr));
            if chosen.as_ref().unwrap().sample_format() == cpal::SampleFormat::F32 {
                break;
            }
        }
    }

    let supported = match chosen {
        Some(c) => c,
        None => {
            log::warn!("48 kHz not supported, falling back to device default");
            device
                .default_output_config()
                .context("device has no default output config")?
        }
    };

    let sample_format = supported.sample_format();
    let mut config: cpal::StreamConfig = supported.into();
    // Phase 1 spec: 256-frame buffer for ~5.3 ms latency at 48 kHz.
    config.buffer_size = cpal::BufferSize::Fixed(TARGET_BUFFER_FRAMES);
    Ok((config, sample_format))
}

fn build_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    mut engine: Engine,
    mut midi_rx: HeapCons<MidiEvent>,
) -> Result<cpal::Stream>
where
    T: cpal::Sample + cpal::SizedSample + cpal::FromSample<f32>,
{
    let channels = config.channels as usize;
    let mono_device = channels == 1;
    // Pre-allocated stereo scratch buffers, reused every callback.
    let mut left = vec![0.0f32; MAX_BUFFER_FRAMES];
    let mut right = vec![0.0f32; MAX_BUFFER_FRAMES];

    let stream = device
        .build_output_stream(
            config,
            move |data: &mut [T], _info: &cpal::OutputCallbackInfo| {
                // 1. Drain incoming MIDI for this block.
                while let Some(ev) = midi_rx.try_pop() {
                    engine.handle_event(ev);
                }

                // 2. Render stereo into the scratch buffers (render_stereo
                //    assigns every sample; no pre-zeroing needed).
                let frames = data.len() / channels.max(1);
                let frames = frames.min(MAX_BUFFER_FRAMES);
                engine.render_stereo(&mut left[..frames], &mut right[..frames]);

                // 3. Write L/R into the device frame. A mono device gets the
                //    exact downmix; anything else gets L in channel 0, R in
                //    channel 1, silence elsewhere. We deliberately do *not*
                //    fan out to the remaining channels: in standard WAVE
                //    order (FL, FR, FC, LFE, BL, BR) channel 3 is the LFE
                //    feed, and sending it a full-bandwidth piano signal is
                //    both wrong and potentially damaging to a subwoofer.
                //    The channel count is fixed for the stream's lifetime, so
                //    the mono/stereo decision is hoisted out of the per-frame
                //    loop rather than re-tested on every sample.
                for (frame_idx, frame) in data.chunks_mut(channels).enumerate() {
                    let (l, r) = if frame_idx < frames {
                        (left[frame_idx], right[frame_idx])
                    } else {
                        // Block bigger than our scratch — pad silence rather than panic.
                        (0.0, 0.0)
                    };
                    if mono_device {
                        frame[0] = T::from_sample(0.5 * (l + r));
                    } else {
                        frame[0] = T::from_sample(l);
                        frame[1] = T::from_sample(r);
                        for out in frame.iter_mut().skip(2) {
                            *out = T::from_sample(0.0f32);
                        }
                    }
                }
            },
            |err| log::error!("cpal stream error: {err}"),
            None,
        )
        .context("failed to build cpal output stream")?;
    Ok(stream)
}
