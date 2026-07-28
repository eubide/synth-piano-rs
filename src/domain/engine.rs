//! Pure DSP engine — polyphonic with voice stealing, stereo soundboard
//! coloration, and sympathetic-resonance bank gated by a sustain pedal.
//!
//! ## Signal path (per stereo channel)
//! ```text
//!   voices[i] ── × pan_i ──┐
//!        │                 ├─ Σ ──┬── soundboard L/R ── × master ── clip ── L/R
//!        └─(unpanned Σ)    │      │
//!             │            │      │
//!             └ × send ── sympathetic ─┘ (centred)
//! ```
//! - **Pan**: each voice sits at a constant-power pan position derived from
//!   its note — bass keys left, treble keys right, as heard from the
//!   player's seat along the bridge. See [`note_pan_gains`].
//! - **Soundboard**: TWO modal plates with slightly skewed mode sets (see
//!   [`Soundboard::new_skewed`]) colour the two channels. The interchannel
//!   decorrelation they introduce is what widens the image beyond simple
//!   amplitude panning — a real plate radiates a different modal mix in
//!   every direction.
//! - **Sympathetic** is fed the *unpanned* voice sum and returns a centred
//!   halo, injected before both plates so it shares their coloration
//!   (physically: sympathetic strings also drive the bridge).
//! - `render` (mono) is the exact `(L+R)/2` downmix of `render_stereo`.
//!   It is *not*, however, equivalent to the pre-stereo single-plate bus:
//!   the two channels are coloured by differently skewed plates and are
//!   soft-clipped independently before the sum, so the downmix carries the
//!   average of two skewed mode sets rather than the nominal one. Panning
//!   also leaves a mild register tilt — a compass-end note sums to
//!   `(0.938 + 0.346)/2 = 0.642` against a centred note's 0.707, so the
//!   keyboard extremes sit ≈ 0.9 dB below the middle on a mono device.
//!   Both are accepted consequences of colouring the channels separately.
//!
//! ## Allocation strategy (`pick_slot`)
//! 1. **Retrigger**: a voice already holding the requested note is
//!    reused. Falls back to a releasing voice of the same note if the
//!    holding instance does not exist.
//! 2. **Idle slot**.
//! 3. **Oldest releasing**: steal the voice that has been in damper
//!    decay longest — perceptually closest to silent.
//! 4. **Oldest holding**: last resort, audible artefact.
//!
//! Ages are tracked with a monotonic `u64` counter. Wraparound is not a
//! practical concern (10⁹ years at 10 notes/second).
//!
//! ## Sustain pedal semantics
//! - Pedal down: incoming `NoteOff` events are recorded in
//!   `pending_note_offs[note]`; the voice keeps ringing.
//! - Pedal up: every pending note is released for real, and the
//!   sympathetic bank is damped.

use std::f32::consts::{FRAC_1_SQRT_2, FRAC_PI_4};

use crate::domain::midi_event::MidiEvent;
use crate::domain::soundboard::Soundboard;
use crate::domain::sympathetic::Sympathetic;
use crate::domain::voice::{Voice, HIGHEST_KEY, LOWEST_KEY};

pub const MAX_VOICES: usize = 32;

/// Fractional mode-frequency skew of the stereo soundboard pair: the left
/// plate's modes shift by −this, the right's by +this (alternating per
/// mode inside each plate). 2.5 % decorrelates the channels audibly while
/// keeping both on the same modal skeleton.
const SOUNDBOARD_SKEW: f32 = 0.025;

/// Maximum pan excursion in [0, 1]. 1.0 would put A0/C8 hard left/right;
/// 0.55 keeps the extremes clearly lateral but still anchored to the case,
/// the way a listener a few metres from a grand hears the bridge spread.
const PAN_WIDTH: f32 = 0.55;

/// Constant-power pan gains for a note: A0 left, C8 right, A4 ≈ centre.
/// `θ` sweeps `[(1−W)·π/4, (1+W)·π/4]`, so `gl² + gr² = 1` everywhere —
/// equal perceived loudness at every position.
fn note_pan_gains(note: u8) -> (f32, f32) {
    // A0..C8; notes outside the compass pin to the ends.
    let span = (HIGHEST_KEY - LOWEST_KEY) as f32;
    let t = (note.saturating_sub(LOWEST_KEY) as f32 / span).clamp(0.0, 1.0);
    let theta = (1.0 + (2.0 * t - 1.0) * PAN_WIDTH) * FRAC_PI_4;
    (theta.cos(), theta.sin())
}

pub struct Engine {
    sample_rate: f32,
    voices: [Voice; MAX_VOICES],
    /// Constant-power pan gains per voice slot, set from the note on
    /// allocation.
    pans: [(f32, f32); MAX_VOICES],
    ages: [u64; MAX_VOICES],
    age_counter: u64,
    mono: bool,
    /// Master gain applied per channel after the soundboard.
    master_gain: f32,

    soundboard_l: Soundboard,
    soundboard_r: Soundboard,
    sympathetic: Sympathetic,

    sustain_pedal_down: bool,
    /// `pending_note_offs[note]` is true when a NoteOff for that pitch
    /// arrived while the pedal was held down. They all release when the
    /// pedal goes back up.
    pending_note_offs: [bool; 128],
}

impl Engine {
    pub fn new(sample_rate: f32) -> Self {
        let voices = std::array::from_fn(|_| Voice::new(sample_rate));
        Self {
            sample_rate,
            voices,
            pans: [(FRAC_1_SQRT_2, FRAC_1_SQRT_2); MAX_VOICES],
            ages: [0; MAX_VOICES],
            age_counter: 0,
            mono: false,
            // Calibrated so that ordinary playing is perfectly linear and
            // only genuinely dense fortissimo clusters reach the soft-clip
            // knee at SOFT_CLIP_THRESHOLD — a real soundboard does not
            // distort at mf.
            //
            // Why 0.405 and not 0.38·√2 ≈ 0.54: the "√2 restores the
            // pre-stereo level" argument only holds at pan centre. Constant
            // -power panning puts up to cos((1−PAN_WIDTH)·π/4) = 0.938 into
            // a note's near channel, not 1/√2 = 0.707, so a compass-end note
            // runs 2.4 dB hotter than that derivation assumes. Dividing by
            // that worst-case pan gain (0.54/0.938 ≈ 0.405) makes the hottest
            // channel match the pre-stereo bus instead of the coldest one.
            //
            // Measured per-channel peaks at 48 kHz, 1 s render, velocity 127
            // unless noted (see the tests below):
            //   single ff, centred (60)     0.452   linear
            //   single ff, bass (24)        0.410   linear
            //   3-note mf, centred (vel 64) 0.363   linear
            //   2-note ff, centred (60,64)  0.839   linear, just under the knee
            //   2-note ff, bass (24,31)     0.787   linear
            //   3-note ff, bass (24,28,31)  0.999   1.20 pre-clip → deep in the knee
            //
            // The last row is the accepted limit: a fortissimo three-note
            // bass cluster is exactly the "dense ff" case the saturator
            // exists for. Pulling it under the knee too would need ≈ 0.30,
            // i.e. 5.2 dB off the whole instrument, which is not worth it.
            // Raising PAN_WIDTH or this gain trades directly against all of
            // the above.
            master_gain: 0.405,
            soundboard_l: Soundboard::new_skewed(sample_rate, -SOUNDBOARD_SKEW),
            soundboard_r: Soundboard::new_skewed(sample_rate, SOUNDBOARD_SKEW),
            sympathetic: Sympathetic::new(sample_rate),
            sustain_pedal_down: false,
            pending_note_offs: [false; 128],
        }
    }

    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    pub fn set_mono(&mut self, mono: bool) {
        self.mono = mono;
    }

    pub fn is_mono(&self) -> bool {
        self.mono
    }

    pub fn is_sustain_pedal_down(&self) -> bool {
        self.sustain_pedal_down
    }

    pub fn handle_event(&mut self, ev: MidiEvent) {
        match ev {
            MidiEvent::NoteOn { note, velocity } => {
                // A retrigger while the pedal is down also clears the
                // "pending release" mark for that note — the player
                // wants the fresh hit, not the deferred release.
                self.pending_note_offs[note as usize] = false;
                self.alloc_voice(note, velocity);
            }
            MidiEvent::NoteOff { note } => {
                if self.sustain_pedal_down {
                    self.pending_note_offs[note as usize] = true;
                } else {
                    self.release_voice(note);
                }
            }
            MidiEvent::AllNotesOff => {
                for v in &mut self.voices {
                    v.note_off();
                }
                for b in &mut self.pending_note_offs {
                    *b = false;
                }
            }
            MidiEvent::SustainPedal { down } => {
                if self.sustain_pedal_down == down {
                    return;
                }
                self.sustain_pedal_down = down;
                self.sympathetic.set_pedal(down);
                if down {
                    // Pedal pressed: lift the damper off any voice mid-release.
                    // Real piano — pressing the pedal physically retracts the
                    // damper bar, so a string that was being muted by the felt
                    // keeps ringing at whatever amplitude it had reached. We
                    // re-mark the note as pending so it resumes releasing once
                    // the pedal lifts again.
                    for v in &mut self.voices {
                        if v.is_active() && v.is_released() {
                            self.pending_note_offs[v.note() as usize] = true;
                            v.cancel_release();
                        }
                    }
                } else {
                    // Pedal lifted: flush every pending note-off in one pass.
                    for note in 0..128u8 {
                        if self.pending_note_offs[note as usize] {
                            self.release_voice(note);
                            self.pending_note_offs[note as usize] = false;
                        }
                    }
                }
            }
        }
    }

    /// One stereo output frame: pan and sum the voices, drive the
    /// sympathetic bank with the unpanned sum, colour each channel with its
    /// own plate, then gain + soft-clip.
    #[inline]
    fn tick_frame(&mut self) -> (f32, f32) {
        let mut l = 0.0f32;
        let mut r = 0.0f32;
        let mut voice_sum = 0.0f32;
        if self.mono {
            let s = self.voices[0].tick();
            let (gl, gr) = self.pans[0];
            voice_sum = s;
            l = s * gl;
            r = s * gr;
        } else {
            for (v, &(gl, gr)) in self.voices.iter_mut().zip(&self.pans) {
                let s = v.tick();
                voice_sum += s;
                l += s * gl;
                r += s * gr;
            }
        }
        // Centred at constant power: 1/√2 into each channel.
        let sympa = self.sympathetic.tick(voice_sum) * FRAC_1_SQRT_2;
        let bl = self.soundboard_l.tick(l + sympa);
        let br = self.soundboard_r.tick(r + sympa);
        (clip(bl * self.master_gain), clip(br * self.master_gain))
    }

    /// Stereo render. `left` and `right` must be the same length — a
    /// mismatch renders only the shorter and leaves stale samples in the
    /// longer, which the cpal adapter would play as a glitch (it relies on
    /// this filling every frame and does no pre-zeroing).
    pub fn render_stereo(&mut self, left: &mut [f32], right: &mut [f32]) {
        debug_assert_eq!(left.len(), right.len(), "stereo buffers must match");
        for (l, r) in left.iter_mut().zip(right.iter_mut()) {
            let (a, b) = self.tick_frame();
            *l = a;
            *r = b;
        }
    }

    /// Mono render: the exact `(L+R)/2` downmix of [`Engine::render_stereo`].
    pub fn render(&mut self, buf: &mut [f32]) {
        for s in buf.iter_mut() {
            let (l, r) = self.tick_frame();
            *s = 0.5 * (l + r);
        }
    }

    fn alloc_voice(&mut self, note: u8, velocity: u8) {
        if self.mono {
            self.voices[0].note_on(note, velocity);
            self.pans[0] = note_pan_gains(note);
            self.bump_age(0);
            return;
        }
        let idx = self.pick_slot(note);
        self.voices[idx].note_on(note, velocity);
        self.pans[idx] = note_pan_gains(note);
        self.bump_age(idx);
    }

    fn release_voice(&mut self, note: u8) {
        if self.mono {
            if self.voices[0].is_active() && self.voices[0].note() == note {
                self.voices[0].note_off();
            }
            return;
        }
        for v in &mut self.voices {
            if v.is_active() && !v.is_released() && v.note() == note {
                v.note_off();
            }
        }
    }

    fn pick_slot(&self, note: u8) -> usize {
        // 1. Same-note retrigger — prefer a HOLDING instance.
        for i in 0..MAX_VOICES {
            if self.voices[i].is_active()
                && !self.voices[i].is_released()
                && self.voices[i].note() == note
            {
                return i;
            }
        }
        for i in 0..MAX_VOICES {
            if self.voices[i].is_active() && self.voices[i].note() == note {
                return i;
            }
        }
        // 2. Idle.
        for i in 0..MAX_VOICES {
            if !self.voices[i].is_active() {
                return i;
            }
        }
        // 3. Oldest releasing.
        let mut best: Option<usize> = None;
        let mut best_age = u64::MAX;
        for i in 0..MAX_VOICES {
            if self.voices[i].is_released() && self.ages[i] < best_age {
                best_age = self.ages[i];
                best = Some(i);
            }
        }
        if let Some(i) = best {
            return i;
        }
        // 4. Oldest holding.
        let mut oldest = 0;
        let mut oldest_age = u64::MAX;
        for i in 0..MAX_VOICES {
            if self.ages[i] < oldest_age {
                oldest_age = self.ages[i];
                oldest = i;
            }
        }
        oldest
    }

    fn bump_age(&mut self, idx: usize) {
        self.age_counter = self.age_counter.wrapping_add(1);
        self.ages[idx] = self.age_counter;
    }
}

/// Amplitude below which each output channel is passed through untouched.
/// With `master_gain = 0.405` (see its comment for the full measured table)
/// a single fortissimo note peaks ≈ 0.45 per channel, a mezzo-forte 3-note
/// chord ≈ 0.36 and even a two-note ff chord ≈ 0.84, so normal playing is
/// perfectly linear; the saturator only engages on genuinely dense
/// fortissimo clusters. 0.88 leaves 0.12 of range for the soft knee to
/// curve through before reaching ±1.
const SOFT_CLIP_THRESHOLD: f32 = 0.88;

/// Soft-clip the output bus to ±1 with a tanh knee above
/// [`SOFT_CLIP_THRESHOLD`].
///
/// A hard clip (`clamp`) recovers from polyphonic overshoot by slicing the
/// waveform flat at ±1. Those sharp corners inject broadband high-frequency
/// harmonics — the audible "buzz"/"fart" a player hears when a dense
/// fortissimo cluster sums well past the rail. Replacing the corner with a
/// smooth tanh knee removes the harsh harmonics: the bus still cannot exceed
/// ±1, but it *approaches* the rail gradually, the way a real soundboard
/// saturates under a fortissimo chord.
///
/// Below the threshold the signal is untouched (unity gain, no colour). Above
/// it, the excess is compressed through `tanh`, which is C¹-continuous at the
/// knee (matching slope) so the transition itself adds no distortion.
#[inline]
fn clip(x: f32) -> f32 {
    let a = x.abs();
    if a <= SOFT_CLIP_THRESHOLD {
        return x;
    }
    let t = SOFT_CLIP_THRESHOLD;
    let knee = ((a - t) / (1.0 - t)).tanh();
    x.signum() * (t + (1.0 - t) * knee)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drain length used by "eventually silences" assertions. Needs to
    /// cover the longest tail in the engine: the soundboard's lowest
    /// mode at Q=15, f=50 Hz has τ ≈ 95 ms. We allow ~20 τ for full
    /// quietness — call it 2 seconds at 48 kHz.
    const SILENCE_TAIL_SAMPLES: usize = 96_000;
    /// −80 dB amplitude — well below anything audible.
    const SILENCE_EPSILON: f32 = 1.0e-4;

    fn tail_max_abs(samples: &[f32]) -> f32 {
        samples
            .iter()
            .map(|s| s.abs())
            .fold(0.0_f32, |acc, x| acc.max(x))
    }

    #[test]
    fn engine_defaults_to_polyphonic() {
        let eng = Engine::new(48_000.0);
        assert!(!eng.is_mono());
    }

    #[test]
    fn engine_defaults_to_pedal_up() {
        let eng = Engine::new(48_000.0);
        assert!(!eng.is_sustain_pedal_down());
    }

    #[test]
    fn render_into_silent_buffer_stays_silent() {
        let mut eng = Engine::new(48_000.0);
        let mut buf = [0.0; 64];
        eng.render(&mut buf);
        assert!(buf.iter().all(|s| *s == 0.0));
    }

    #[test]
    fn note_on_produces_non_zero_output() {
        let mut eng = Engine::new(48_000.0);
        eng.handle_event(MidiEvent::note_on(69, 100));
        let mut buf = [0.0; 4_096];
        eng.render(&mut buf);
        assert!(buf.iter().any(|s| s.abs() > 0.0));
    }

    #[test]
    fn note_off_eventually_silences_output() {
        let mut eng = Engine::new(48_000.0);
        eng.handle_event(MidiEvent::note_on(60, 100));
        let mut buf = [0.0; 512];
        eng.render(&mut buf);
        eng.handle_event(MidiEvent::note_off(60));
        let mut tail = vec![0.0; SILENCE_TAIL_SAMPLES];
        eng.render(&mut tail);
        let max_abs = tail_max_abs(&tail[tail.len() - 1024..]);
        assert!(max_abs < SILENCE_EPSILON, "tail not silent: {max_abs}");
    }

    #[test]
    fn all_notes_off_silences_everything() {
        let mut eng = Engine::new(48_000.0);
        for n in 60..68 {
            eng.handle_event(MidiEvent::note_on(n, 100));
        }
        eng.handle_event(MidiEvent::AllNotesOff);
        let mut tail = vec![0.0; SILENCE_TAIL_SAMPLES];
        eng.render(&mut tail);
        let max_abs = tail_max_abs(&tail[tail.len() - 1024..]);
        assert!(max_abs < SILENCE_EPSILON, "tail not silent: {max_abs}");
    }

    #[test]
    fn poly_uses_one_voice_per_distinct_note() {
        let mut eng = Engine::new(48_000.0);
        eng.handle_event(MidiEvent::note_on(60, 100));
        eng.handle_event(MidiEvent::note_on(64, 100));
        eng.handle_event(MidiEvent::note_on(67, 100));
        let active = eng.voices.iter().filter(|v| v.is_active()).count();
        assert_eq!(active, 3);
    }

    #[test]
    fn same_note_retrigger_reuses_one_voice() {
        let mut eng = Engine::new(48_000.0);
        eng.handle_event(MidiEvent::note_on(60, 100));
        eng.handle_event(MidiEvent::note_on(60, 120));
        let count_60 = eng
            .voices
            .iter()
            .filter(|v| v.is_active() && v.note() == 60)
            .count();
        assert_eq!(count_60, 1);
    }

    #[test]
    fn stealing_prefers_released_over_holding() {
        let mut eng = Engine::new(48_000.0);
        for i in 0..MAX_VOICES {
            eng.handle_event(MidiEvent::note_on(40 + i as u8, 100));
        }
        eng.handle_event(MidiEvent::note_off(40));
        let mut buf = [0.0; 32];
        eng.render(&mut buf);
        eng.handle_event(MidiEvent::note_on(100, 100));
        let still_have_40 = eng.voices.iter().any(|v| v.is_active() && v.note() == 40);
        let held_count = eng
            .voices
            .iter()
            .filter(|v| v.is_active() && !v.is_released())
            .count();
        assert!(!still_have_40);
        assert_eq!(held_count, MAX_VOICES);
    }

    #[test]
    fn stealing_falls_back_to_oldest_holding_when_no_released() {
        let mut eng = Engine::new(48_000.0);
        for i in 0..MAX_VOICES {
            eng.handle_event(MidiEvent::note_on(40 + i as u8, 100));
        }
        eng.handle_event(MidiEvent::note_on(100, 100));
        assert!(!eng.voices.iter().any(|v| v.is_active() && v.note() == 40));
        assert!(eng.voices.iter().any(|v| v.is_active() && v.note() == 100));
    }

    #[test]
    fn full_polyphony_keeps_all_voices_alive_simultaneously() {
        let mut eng = Engine::new(48_000.0);
        for i in 0..MAX_VOICES {
            eng.handle_event(MidiEvent::note_on(40 + i as u8, 100));
        }
        let active = eng.voices.iter().filter(|v| v.is_active()).count();
        assert_eq!(active, MAX_VOICES);
    }

    #[test]
    fn mono_mode_only_uses_voice_zero() {
        let mut eng = Engine::new(48_000.0);
        eng.set_mono(true);
        eng.handle_event(MidiEvent::note_on(60, 100));
        eng.handle_event(MidiEvent::note_on(64, 100));
        eng.handle_event(MidiEvent::note_on(67, 100));
        for i in 1..MAX_VOICES {
            assert!(!eng.voices[i].is_active(), "voice {i} should be idle");
        }
        assert!(eng.voices[0].is_active());
        assert_eq!(eng.voices[0].note(), 67);
    }

    // ─── Sustain pedal tests ──────────────────────────────────────────

    #[test]
    fn pedal_down_defers_note_off() {
        let mut eng = Engine::new(48_000.0);
        eng.handle_event(MidiEvent::SustainPedal { down: true });
        eng.handle_event(MidiEvent::note_on(60, 100));
        eng.handle_event(MidiEvent::note_off(60));
        // The voice should still be active and NOT released.
        let v = eng.voices.iter().find(|v| v.note() == 60).unwrap();
        assert!(v.is_active());
        assert!(!v.is_released());
    }

    #[test]
    fn pedal_up_flushes_pending_note_offs() {
        let mut eng = Engine::new(48_000.0);
        eng.handle_event(MidiEvent::SustainPedal { down: true });
        eng.handle_event(MidiEvent::note_on(60, 100));
        eng.handle_event(MidiEvent::note_off(60));
        eng.handle_event(MidiEvent::SustainPedal { down: false });
        let v = eng.voices.iter().find(|v| v.note() == 60).unwrap();
        assert!(v.is_released(), "voice should have entered release phase");
    }

    #[test]
    fn note_on_during_pedal_clears_its_pending_release() {
        // pedal down → A on → A off (pending) → A on (should clear pending)
        // → pedal up → A should still be holding (NOT released).
        let mut eng = Engine::new(48_000.0);
        eng.handle_event(MidiEvent::SustainPedal { down: true });
        eng.handle_event(MidiEvent::note_on(60, 100));
        eng.handle_event(MidiEvent::note_off(60));
        eng.handle_event(MidiEvent::note_on(60, 110));
        eng.handle_event(MidiEvent::SustainPedal { down: false });
        let v = eng.voices.iter().find(|v| v.note() == 60).unwrap();
        assert!(v.is_active() && !v.is_released());
    }

    #[test]
    fn pedal_down_lets_sympathetic_ring_longer() {
        // Hammer the engine briefly with the pedal up vs down, then
        // measure the post-note tail RMS. With the pedal held down the
        // sympathetic bank should sustain audibly longer.
        fn tail_rms(pedal_down: bool) -> f32 {
            let mut eng = Engine::new(48_000.0);
            if pedal_down {
                eng.handle_event(MidiEvent::SustainPedal { down: true });
            }
            eng.handle_event(MidiEvent::note_on(60, 110));
            let mut attack = vec![0.0; 4_800]; // 100 ms
            eng.render(&mut attack);
            eng.handle_event(MidiEvent::note_off(60));
            let mut tail = vec![0.0; 24_000]; // 500 ms
            eng.render(&mut tail);
            // Look at the very end (after the voice damper has gated).
            let end = &tail[tail.len() - 2_048..];
            let sq: f32 = end.iter().map(|x| x * x).sum();
            (sq / end.len() as f32).sqrt()
        }
        let down = tail_rms(true);
        let up = tail_rms(false);
        assert!(
            down > up * 2.0,
            "pedal down should leave more residual ring: down={down} up={up}"
        );
    }

    // ─── Soft-clip / saturation tests ─────────────────────────────────

    #[test]
    fn soft_clip_is_transparent_below_threshold() {
        // A lone fortissimo note peaks at ≈ 0.76; everything up to the
        // knee must pass through bit-for-bit so single notes keep their
        // exact timbre.
        for &x in &[0.0, 0.1, 0.5, SOFT_CLIP_THRESHOLD] {
            assert_eq!(clip(x), x);
            assert_eq!(clip(-x), -x);
        }
    }

    #[test]
    fn soft_clip_never_exceeds_unity() {
        // No matter how dense the chord (a 5-note bass ff cluster peaks
        // at ≈ 3.3), the bus must stay inside ±1 — the safety-net role.
        for i in 0..2_000 {
            let x = i as f32 * 0.01; // 0 .. 20
            assert!(clip(x) <= 1.0, "clip({x}) = {} exceeded 1.0", clip(x));
            assert!(clip(-x) >= -1.0, "clip({}) underflowed -1.0", -x);
        }
    }

    #[test]
    fn soft_clip_is_monotonic_and_odd() {
        // Monotonic → no fold-back artefacts; odd → no DC bias added.
        let mut prev = clip(0.0);
        for i in 1..2_000 {
            let x = i as f32 * 0.005;
            let y = clip(x);
            assert!(y >= prev, "non-monotonic at x={x}: {y} < {prev}");
            assert!((clip(-x) + y).abs() < 1e-6, "not odd at x={x}");
            prev = y;
        }
    }

    // ─── Stereo tests ─────────────────────────────────────────────────

    fn channel_rms_for_note(note: u8) -> (f32, f32) {
        let mut eng = Engine::new(48_000.0);
        eng.handle_event(MidiEvent::note_on(note, 100));
        let n = 9_600; // 200 ms
        let mut l = vec![0.0; n];
        let mut r = vec![0.0; n];
        eng.render_stereo(&mut l, &mut r);
        let rms = |b: &[f32]| (b.iter().map(|x| x * x).sum::<f32>() / b.len() as f32).sqrt();
        (rms(&l), rms(&r))
    }

    #[test]
    fn bass_pans_left_treble_pans_right() {
        let (bl, br) = channel_rms_for_note(24); // C1
        assert!(bl > br * 1.3, "bass should favour left: l={bl} r={br}");
        let (tl, tr) = channel_rms_for_note(105); // A7
        assert!(tr > tl * 1.3, "treble should favour right: l={tl} r={tr}");
    }

    #[test]
    fn stereo_channels_are_decorrelated_but_coherent() {
        // A centre-register chord reaches both channels at similar level,
        // but the skewed plates must decorrelate the fine structure: the
        // normalised cross-correlation stays clearly below 1 (mono would be
        // exactly 1) and above 0 (the channels are the same notes, not two
        // different signals).
        let mut eng = Engine::new(48_000.0);
        for n in [60, 64, 67] {
            eng.handle_event(MidiEvent::note_on(n, 100));
        }
        let n = 48_000;
        let mut l = vec![0.0; n];
        let mut r = vec![0.0; n];
        eng.render_stereo(&mut l, &mut r);
        let (mut lr, mut ll, mut rr) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..n {
            lr += (l[i] * r[i]) as f64;
            ll += (l[i] * l[i]) as f64;
            rr += (r[i] * r[i]) as f64;
        }
        let corr = lr / (ll * rr).sqrt().max(1e-30);
        assert!(
            (0.2..0.995).contains(&corr),
            "stereo correlation out of range: {corr}"
        );
    }

    #[test]
    fn mono_render_is_exact_downmix_of_stereo() {
        let events = [
            MidiEvent::note_on(36, 110),
            MidiEvent::note_on(60, 90),
            MidiEvent::note_on(96, 70),
        ];
        let mut eng_mono = Engine::new(48_000.0);
        let mut eng_stereo = Engine::new(48_000.0);
        for ev in events {
            eng_mono.handle_event(ev);
            eng_stereo.handle_event(ev);
        }
        let n = 4_096;
        let mut mono = vec![0.0; n];
        let mut l = vec![0.0; n];
        let mut r = vec![0.0; n];
        eng_mono.render(&mut mono);
        eng_stereo.render_stereo(&mut l, &mut r);
        for i in 0..n {
            assert_eq!(
                mono[i],
                0.5 * (l[i] + r[i]),
                "downmix mismatch at sample {i}"
            );
        }
    }

    /// Worst per-channel peak over a 1 s render of `notes` struck together.
    ///
    /// Headroom must be measured per channel, not on the mono downmix: the
    /// `(L+R)/2` average cancels exactly the pan boost that eats the
    /// headroom, so a downmix test would pass no matter how hot the panned
    /// channels ran.
    fn peak_per_channel(notes: &[u8], velocity: u8) -> f32 {
        let mut eng = Engine::new(48_000.0);
        for &n in notes {
            eng.handle_event(MidiEvent::note_on(n, velocity));
        }
        let mut l = vec![0.0; 48_000];
        let mut r = vec![0.0; 48_000];
        eng.render_stereo(&mut l, &mut r);
        l.iter()
            .chain(r.iter())
            .map(|s| s.abs())
            .fold(0.0_f32, f32::max)
    }

    #[test]
    fn ordinary_playing_stays_below_the_soft_knee_on_every_channel() {
        // The calibration promise of `master_gain`: everything short of a
        // dense ff cluster is passed through untouched. Bass notes are the
        // demanding case — panning puts 0.938 of them into one channel — so
        // they are covered explicitly rather than only through the downmix.
        for (label, notes, vel) in [
            ("single ff centred", &[60u8][..], 127),
            ("single ff bass", &[24u8][..], 127),
            ("3-note mf centred", &[60u8, 64, 67][..], 64),
            ("2-note ff centred", &[60u8, 64][..], 127),
            ("2-note ff bass", &[24u8, 31][..], 127),
        ] {
            let peak = peak_per_channel(notes, vel);
            assert!(
                peak < SOFT_CLIP_THRESHOLD,
                "{label} must stay linear on both channels: peak {peak} \
                 reached the {SOFT_CLIP_THRESHOLD} knee"
            );
        }
    }

    #[test]
    fn dense_ff_bass_cluster_uses_the_knee_without_hitting_the_rail() {
        // The accepted limit case. It *should* engage the saturator — that
        // is what the saturator is for — but must approach ±1 gradually
        // through the tanh knee rather than being pinned flat by a hard
        // clip. Measured 0.9988 (≈ 1.20 before the knee).
        let peak = peak_per_channel(&[24, 28, 31], 127);
        assert!(
            peak > SOFT_CLIP_THRESHOLD,
            "dense ff bass cluster should reach the knee: {peak}"
        );
        assert!(peak < 1.0, "soft knee must stay below the rail: {peak}");
    }
}
