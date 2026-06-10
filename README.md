# synth-piano-rs

Real-time physically-modelled acoustic piano synthesizer written in Rust.

The DSP core is an Extended Karplus-Strong digital waveguide with velocity-sensitive hammer excitation, string-stiffness dispersion, a modal soundboard, and a sympathetic-resonance bank gated by the sustain pedal. Audio output goes through `cpal`; MIDI input comes from `midir`.

## Requirements

- Rust toolchain (stable, 2021 edition or later)
- A system audio output device supported by `cpal` (CoreAudio on macOS, WASAPI on Windows, ALSA/PipeWire on Linux)
- A MIDI input device (keyboard, controller) — optional; the synth starts silently if none is found

## Build and run

```bash
# Run in release mode — always use this for real playing
cargo run --release

# Build without running
cargo build --release

# Run with verbose DSP logging
RUST_LOG=debug cargo run --release

# Run all unit tests
cargo test
```

> Never run in debug mode for audio: the DSP is far too slow and will produce dropouts.

## Signal path

```
                                    ┌── × master_gain ── soft-clip ── output
                                    │
voices.sum() ──┬──────────────────┬─┴── soundboard ───┘
               │                  │
               └── × send ── sympathetic ─┘
```

Each voice: `Hammer → StrikeComb → StringGroup → × damper_gain`

## Architecture

Hexagonal (ports & adapters), three strict layers:

| Layer | Path | Role |
|---|---|---|
| Domain | `src/domain/` | Pure DSP — no I/O, no allocation after construction |
| Ports | `src/ports/` | `AudioOutput` and `MidiInput` traits |
| Adapters | `src/adapters/` | `CpalOutput` and `MidirInput` concrete implementations |

### Threads

- **Main** — builds the engine, parks after handing it to the audio adapter.
- **Audio (RT)** — owned by cpal; drains MIDI events from an SPSC ring buffer, calls `Engine::render`. No locks, no allocations, no logging inside the render loop.
- **MIDI** — owned by midir; parses raw bytes, pushes `MidiEvent`s into the ring buffer with non-blocking `try_push`.

## DSP modules

### Hammer (`domain/hammer.rs`)
A raised-cosine pulse whose duration tapers geometrically from ~3.5 ms at A0 down to ~0.4 ms at C8, matching Askenfelt & Jansson's measurements on real grands. A one-pole lowpass shaped by MIDI velocity approximates the felt compression: soft hits are muffled, hard hits are bright.

### Strike comb (`domain/strike_comb.rs`)
Feedforward comb filter applied to the excitation before it enters the loop. Grand piano hammers strike near 1/7 of the string length, suppressing the 7th-partial family and tilting energy toward the mid partials. Without this the excitation drives all partials equally, producing a buzzy, organ-like tone especially audible in the bass.

### Karplus-Strong string (`domain/string.rs`)
Extended KS loop: integer delay + first-order allpass fractional tuner (lossless, unlike linear interpolation which over-damps treble fundamentals inside the resonant loop) + tunable two-tap loop filter + allpass-cascade dispersion. The loop filter's smoothing weight adapts per note so the filter's own T60 at the fundamental never drops below 4 s, keeping the top octaves ringing past their attack.

### Dispersion (`domain/dispersion.rs`)
Allpass cascade that stretches partials upward to model string stiffness (Fletcher & Rossing §12.5). The inharmonicity coefficient follows a U-shaped Railsback curve: high in both the bass (thick wound strings) and treble (short stiff strings), minimum in the middle register.

### String group (`domain/string_group.rs`)
Manages the 1–3 detuned KS strings per voice (bass = 1, tenor = 2, mid/treble = 3). Single-string bass notes arm a second loop detuned by +1.4 cents to stand in for the horizontal string polarization (Weinreich 1977), giving the bass its characteristic slow shimmer rather than a flat exponential decay.

### Soundboard (`domain/soundboard.rs`)
18 bandpass biquad resonators in parallel, covering 50 Hz – 5.5 kHz with log-spaced irregular spacing. Approximates a grand piano plate's modal response: adds body coloration, shapes the spectral envelope of every voice, and contributes a short reverb-like tail. The two lowest modes (50 Hz, 66 Hz) are grounded in Suzuki (1986)'s measurements of the lowest grand soundboard modes and are essential for bass bloom.

### Sympathetic resonance bank (`domain/sympathetic.rs`)
24 KS strings spanning C2–B3 (two chromatic octaves). Pedal up: loop gain just below 1.0, bank damps in ~50 ms. Pedal down: loop gain = 1.0, strings ring for several seconds accumulating energy at their harmonic frequencies, producing the audible "halo" of a real sustained pedal. The bank's output re-enters the signal bus before the soundboard so it receives the same modal coloration as played notes.

### Voice allocator (`domain/engine.rs`)
32-voice polyphony. Stealing priority: same-note retrigger → idle slot → oldest releasing voice → oldest holding voice. Ages tracked with a monotonic `u64` counter.

## MIDI

Handled CCs:

| Event | Action |
|---|---|
| Note On (vel > 0) | Trigger voice |
| Note On (vel = 0) | Release (same as Note Off) |
| Note Off | Release voice or queue if pedal is down |
| CC 64 | Sustain pedal |
| CC 123 | All notes off |

All other CCs and SysEx are silently dropped.

## License

MIT
