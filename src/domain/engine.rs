//! Pure DSP engine — polyphonic with voice stealing, soundboard
//! coloration, and sympathetic-resonance bank gated by a sustain pedal.
//!
//! ## Signal path
//! ```text
//!                                  ┌── × master_gain ── soft-clip ── output
//!                                  │
//!   voices.sum() ──┬──────────────┬┴─── soundboard ───┘
//!                  │              │
//!                  │              │
//!                  └── × send ── sympathetic ─┘
//! ```
//! - **Soundboard** applies modal coloration to the combined string +
//!   sympathetic bus. It is the radiating element of the piano — *every*
//!   pitched signal passes through it.
//! - **Sympathetic** is fed a small fraction of the voice bus. It rings
//!   when the pedal is down and dies quickly when up. Its output joins
//!   the voice bus *before* the soundboard, so the sympathetic ring is
//!   coloured by the same plate modes as the played notes (physically:
//!   sympathetic strings also drive the bridge).
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

use crate::domain::midi_event::MidiEvent;
use crate::domain::soundboard::Soundboard;
use crate::domain::sympathetic::Sympathetic;
use crate::domain::voice::Voice;

pub const MAX_VOICES: usize = 32;

pub struct Engine {
    sample_rate: f32,
    voices: [Voice; MAX_VOICES],
    ages: [u64; MAX_VOICES],
    age_counter: u64,
    mono: bool,
    /// Master gain applied after summing voices, sympathetic and the
    /// soundboard's wet/dry mix.
    master_gain: f32,

    soundboard: Soundboard,
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
            ages: [0; MAX_VOICES],
            age_counter: 0,
            mono: false,
            // Lowered from 0.5 to leave headroom for polyphony: a single ff
            // note now peaks ≈ 0.5 (not 0.76), so ordinary 2–3 note chords
            // stay below the soft-clip knee and only genuinely dense ff
            // clusters saturate — a real soundboard does not distort at mf.
            // The lower nominal level is recovered downstream by system gain.
            master_gain: 0.34,
            soundboard: Soundboard::new(sample_rate),
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

    pub fn render(&mut self, buf: &mut [f32]) {
        if self.mono {
            for s in buf.iter_mut() {
                let voice_sum = self.voices[0].tick();
                let sympa = self.sympathetic.tick(voice_sum);
                let body = self.soundboard.tick(voice_sum + sympa);
                *s = clip(body * self.master_gain);
            }
            return;
        }
        for s in buf.iter_mut() {
            let mut voice_sum = 0.0f32;
            for v in &mut self.voices {
                voice_sum += v.tick();
            }
            let sympa = self.sympathetic.tick(voice_sum);
            let body = self.soundboard.tick(voice_sum + sympa);
            *s = clip(body * self.master_gain);
        }
    }

    fn alloc_voice(&mut self, note: u8, velocity: u8) {
        if self.mono {
            self.voices[0].note_on(note, velocity);
            self.bump_age(0);
            return;
        }
        let idx = self.pick_slot(note);
        self.voices[idx].note_on(note, velocity);
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

/// Amplitude below which the output bus is passed through untouched. With
/// `master_gain = 0.34` a single fortissimo note peaks ≈ 0.5 and even an
/// ordinary mezzo-forte 3-note chord stays below this knee, so normal
/// playing is perfectly linear; the saturator only engages on genuinely
/// dense fortissimo clusters. 0.88 leaves 0.12 of range for the soft knee
/// to curve through before reaching ±1.
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
    /// mode at Q=20, f=89 Hz has τ ≈ 71 ms. We allow ~30 τ for full
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

    #[test]
    fn ff_chord_enters_soft_knee_below_rail() {
        // After the lowered master_gain, a two-note fortissimo chord just
        // reaches into the soft knee (peak ≈ 0.94): above the 0.88 threshold
        // yet strictly below the ±1 rail. This proves the bus *approaches*
        // the rail gradually through the tanh knee instead of being pinned
        // flat by a hard clip — while ordinary mf playing stays fully linear
        // thanks to the new headroom.
        let mut eng = Engine::new(48_000.0);
        eng.handle_event(MidiEvent::note_on(60, 127));
        eng.handle_event(MidiEvent::note_on(64, 127));
        let mut buf = vec![0.0; 9_600]; // 200 ms
        eng.render(&mut buf);
        let peak = buf.iter().map(|s| s.abs()).fold(0.0_f32, f32::max);
        assert!(peak > 0.88, "ff chord should reach into the knee: {peak}");
        assert!(peak < 1.0, "soft knee should stay below the rail: {peak}");
    }
}
