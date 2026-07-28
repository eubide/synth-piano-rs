# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Real-time, physically-modelled acoustic piano synthesizer in Rust. The DSP core is an Extended Karplus-Strong digital waveguide with hammer excitation, dispersion (string stiffness), a modal soundboard, and a sympathetic-resonance bank gated by the sustain pedal. Output goes through `cpal`; MIDI input comes from `midir`.

## Commands

```bash
cargo run --release        # Run the synth (RT-critical — never use debug for real playing)
cargo build --release      # Release build (LTO thin, codegen-units 1)
cargo test                 # All unit tests (DSP modules + MIDI parser)
cargo test -p synth-piano-rs <name>   # Run a single test by name fragment
cargo test domain::engine  # Run tests in one module
RUST_LOG=debug cargo run --release    # Verbose logging (default is info)
```

There are no integration tests, no benches, no fmt/clippy configuration files — `cargo fmt` and `cargo clippy` use defaults.

## Architecture

Hexagonal (ports & adapters). The three layers are strictly enforced:

- **`src/domain/`** — pure DSP. **No `cpal`, no `midir`, no allocation after construction, no I/O.** Sample-rate parametric: every component takes `sample_rate: f32` in `::new`. Everything is unit-testable offline by feeding samples through `tick()` / `render()`.
- **`src/ports/`** — traits the application code depends on (`AudioOutput`, `MidiInput`). Intentionally minimal: the cpal stream owns the engine; the port is just a "started handle" you drop to stop.
- **`src/adapters/`** — concrete I/O (`CpalOutput`, `MidirInput`). Translate raw bytes ↔ `MidiEvent`, drive `Engine::render` from the cpal callback.

### Threading model

Three threads:
1. **Main thread** — sets up the engine, hands it to the cpal adapter, then `thread::park()`s.
2. **Audio (RT) thread** — owned by cpal. Drains MIDI events from the SPSC ring buffer at the top of every callback, then calls `Engine::render_stereo(&mut [f32], &mut [f32])`. **No allocations, no locks, no logging inside the render loop.** A pre-allocated stereo pair of scratch buffers (`MAX_BUFFER_FRAMES = 8192` each) is reused every callback; even device channels get L, odd get R, and a mono device gets the exact `(L+R)/2` downmix (also available as `Engine::render`).
3. **MIDI thread** — owned by midir. Parses bytes, pushes `MidiEvent`s into the SPSC ring buffer with `try_push` (drops on overflow rather than blocking).

The engine itself is single-threaded and lives on the audio thread. `cpal::Stream` is `!Send` on macOS — that is why `AudioOutput` is not `Send`.

### DSP signal path (engine.rs)

```
voices[i] ── × pan_i ──┐
     │                 ├─ Σ ──┬── soundboard L/R ── × master_gain ── clip ── L/R
     └─(unpanned Σ)    │      │
          └── × send ── sympathetic ─┘ (centred)
```

Stereo: each voice is panned by note (constant power, A0 left → C8 right, `PAN_WIDTH = 0.55`), and the two channels are coloured by *two* soundboards with slightly skewed mode sets (`SOUNDBOARD_SKEW`) for interchannel decorrelation. `Engine::render` (mono) is the exact `(L+R)/2` downmix of `render_stereo` — tests mostly use it.

A "voice" is `Hammer → StringGroup (1–3 detuned KarplusStrong strings) → damper envelope → (+ HammerKnock attack noise)`. The string count per voice depends on the MIDI note: bass (≤31) = 1 string, tenor (32–43) = 2, mid/treble (≥44) = 3. Strings are stretch-tuned (Railsback cubic in voice.rs — the dispersion model makes ET octaves beat otherwise). Sympathetic output is summed *before* the soundboards so it shares the same plate coloration as played notes — physically correct (real sympathetic strings also drive the bridge).

Each `KarplusStrong` loop is: `delay → loop-LPF → dispersion-allpass-cascade → + excitation → write-back`. Excitation enters undispersed/unfiltered so the player hears the natural strike transient on the first pass.

### Voice allocator (`Engine::pick_slot`)

Polyphony cap is `MAX_VOICES = 32`. Stealing priority: same-note-retrigger (holding > releasing) → idle slot → oldest releasing voice → oldest holding voice. Ages are tracked with a monotonic `u64` counter (wraparound is not a practical concern).

### Sustain pedal semantics

Pedal down → incoming `NoteOff` events go into `pending_note_offs[note]` and the voice keeps ringing. Pedal up → flush all pending releases. A retrigger while pedal-down clears its pending mark (player wants the fresh hit, not the deferred release). The sympathetic bank's loop gain is driven directly by the pedal state.

### Sample-rate discipline

The engine and every DSP component are built for one specific sample rate. `main.rs` queries the device first and then constructs `Engine::new(sample_rate)`. The cpal adapter warns loudly if `engine.sample_rate()` does not match what the device actually opened — a mismatch silently detunes everything.

## Conventions

- Module docs (`//!`) explain *why* a chosen DSP model exists and its physical justification (Fletcher & Rossing, Jaffe & Smith, etc.). Treat these as load-bearing — they encode tuning decisions that aren't obvious from code alone.
- Tests inside `#[cfg(test)] mod tests` are physically-motivated assertions (e.g. "pedal down leaves more residual ring", "hard strike has more HF energy than soft"), not just type-level checks. When changing DSP behavior, expect to update the thresholds in these tests.
- New DSP modules live in `src/domain/` and must stay allocation-free past `::new`. Add them to `src/domain/mod.rs`.
- The MIDI parser (`adapters/midir_input.rs::parse_message`) currently handles Note On (velocity 0 = NoteOff), Note Off, CC 64 (sustain), and CC 123 (All Notes Off). Other CCs and SysEx are dropped silently.
