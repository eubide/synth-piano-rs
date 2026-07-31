//! Phase 5 voice: hammer-excited [`StringGroup`] (1–3 detuned strings) with
//! a damper envelope.
//!
//! ## Signal path
//! `Hammer.tick() → StrikeComb.tick() → StringGroup.tick(exc) → · damper_gain → (+ knock) → output`
//!
//! - The strike comb imposes the hammer's strike-position weighting
//!   (dips on the 7th-partial family) on the excitation before it reaches
//!   the strings — see [`crate::domain::strike_comb`].
//! - The [`HammerKnock`] burst (mechanical attack noise) sums in *after*
//!   the damper product: structural noise is not muted by the damper felt.
//! - Number of strings depends on the MIDI note range (see
//!   [`strings_for_note`]).
//! - Damper applies to the summed string output, so a release fades the
//!   whole group together (matching how a real damper bar engages every
//!   string for a given key).
//!
//! ## What the voice exposes for the allocator
//! [`Voice::is_active`] and [`Voice::is_released`] together let the engine
//! distinguish *holding* (key down, full sustain) from *releasing* (key
//! up, damper decaying) — the allocator prefers to steal releasing voices
//! because they are perceptually closer to silent.

use crate::domain::hammer::Hammer;
use crate::domain::knock::HammerKnock;
use crate::domain::strike_comb::StrikeComb;
use crate::domain::string_group::{strings_for_note, StringGroup};

/// MIDI note → equal-tempered frequency in Hz (A4 = 440 Hz, MIDI 69).
pub fn midi_to_hz(note: u8) -> f32 {
    440.0 * ((note as f32 - 69.0) / 12.0).exp2()
}

/// Reference note for the stretch curve: A4 stays at exactly 440 Hz.
const STRETCH_ANCHOR_NOTE: f32 = 69.0;
/// Railsback deviation at the ends of the compass, in cents. Measured
/// pianos deviate ≈ −30 cents at A0 and ≈ +30 cents at C8.
const STRETCH_BASS_CENTS: f32 = -30.0;
const STRETCH_TREBLE_CENTS: f32 = 30.0;

/// Railsback stretch offset for `note`, in cents.
///
/// The dispersion cascade makes every string inharmonic: partial 2 sits
/// *above* 2·f₀. Tuning the keyboard to pure equal temperament therefore
/// makes octaves beat — the lower note's stretched 2nd partial clashes with
/// the upper note's fundamental. Real tuners resolve this by stretching the
/// scale (Railsback curve): bass flat, treble sharp, flat middle. A cubic
/// anchored at A4 matches the measured curve's shape — near zero across the
/// two middle octaves, climbing steeply only toward the extremes.
fn railsback_cents(note: u8) -> f32 {
    let n = note as f32;
    if n < STRETCH_ANCHOR_NOTE {
        let t = ((STRETCH_ANCHOR_NOTE - n) / (STRETCH_ANCHOR_NOTE - LOWEST_KEY as f32)).min(1.0);
        STRETCH_BASS_CENTS * t * t * t
    } else {
        let t = ((n - STRETCH_ANCHOR_NOTE) / (HIGHEST_KEY as f32 - STRETCH_ANCHOR_NOTE)).min(1.0);
        STRETCH_TREBLE_CENTS * t * t * t
    }
}

/// MIDI note → stretch-tuned frequency in Hz (equal temperament plus the
/// [`railsback_cents`] offset). This is what struck and sympathetic strings
/// are tuned to; [`midi_to_hz`] remains the unstretched reference.
pub fn stretched_midi_to_hz(note: u8) -> f32 {
    midi_to_hz(note) * (railsback_cents(note) / 1200.0).exp2()
}

const DAMPER_GATE_THRESHOLD: f32 = 1.0e-4;
const MIN_FREQUENCY_HZ: f32 = 20.0;

/// MIDI note numbers bounding the 88-key piano: A0 and C8. Every
/// note-indexed taper (damper τ, string T60, stretch, stereo pan) is defined
/// across this range; notes outside it pin to the ends.
pub const LOWEST_KEY: u8 = 21;
pub const HIGHEST_KEY: u8 = 108;

/// Highest note that still carries a damper. On a real grand the top ~1.5
/// octaves have no dampers at all — those strings are short and quiet enough
/// that the makers leave them to ring and decay naturally. Notes above this
/// are not muted on key-up; they fade out at their own [`string_loop_gain`]
/// rate (and are reclaimed by the envelope auto-gate once inaudible).
const HIGHEST_DAMPED_NOTE: u8 = 95; // B6 — notes 96 (C7)…108 ring free

/// Damper time constant at the bass end (A0). This is *one* time constant
/// (−8.7 dB); the felt fully mutes (the −80 dB gate) in ≈ τ·ln(10⁴) ≈ 1.1 s
/// for the heavy bass strings, faster as τ shrinks toward the treble.
const DAMPER_TAU_BASS_SECS: f32 = 0.12;
/// How much shorter τ gets at the treble end (C8): 0.12 − 0.08 = 0.04 s,
/// a near-instant mute under the tiny treble dampers. Must stay below
/// [`DAMPER_TAU_BASS_SECS`] so τ never reaches zero (see the
/// `damper_decay_never_reaches_unity` test).
const DAMPER_TAU_RANGE_SECS: f32 = 0.08;

/// Envelope auto-gate. A struck string only stops via its damper today; but
/// undamped treble notes (and any held note whose loop gain has decayed it
/// to silence) would otherwise keep an active voice slot forever. We track a
/// fast-attack / slow-release envelope of the voice output and free the slot
/// once it has stayed below [`AUTO_GATE_LEVEL`] for [`AUTO_GATE_HOLD_SECS`].
const AUTO_GATE_LEVEL: f32 = 1.0e-4; // −80 dB, below audibility
const AUTO_GATE_HOLD_SECS: f32 = 0.2;
/// Envelope follower release time (peak-hold decay).
const ENV_RELEASE_SECS: f32 = 0.02;

/// Target T60 (time to decay −60 dB) of the *struck* string's fundamental,
/// in seconds, at the two ends of the keyboard. A real grand sustains the
/// bass for tens of seconds and the treble for a few; we taper geometrically
/// between these so each register decays at a physically plausible rate.
///
/// Without this the strings run at `loop_gain = 1.0` and the only loss is
/// the loop LPF, which barely touches a low fundamental — so
/// a held bass/mid note would ring almost forever (organ-like), the single
/// biggest realism defect. The LPF still shapes the *spectral* decay (highs
/// die first); this just sets the overall envelope length per pitch.
///
/// These targets are *nominal*: [`StringGroup::set_loop_gain`] spreads each
/// unison string's T60 around them (and gives the bass polarization a much
/// longer one) to produce the two-stage decay, so the audible tail of a
/// note outlives the nominal figure by design.
const STRING_T60_BASS_SECS: f32 = 20.0;
const STRING_T60_TREBLE_SECS: f32 = 4.0;

/// Per-cycle loop gain that yields [`STRING_T60_BASS_SECS`]..
/// [`STRING_T60_TREBLE_SECS`] for `note`'s fundamental. A string circulates
/// its loop `f₀` times per second; after `t` seconds the amplitude is
/// `gain^(f₀·t)`, so `gain = exp(ln(10⁻³) / (f₀·T60))` hits −60 dB at `T60`.
/// The loop LPF adds extra loss on top (mostly in the treble), so the real
/// envelope is a touch shorter than the target — intended.
fn string_loop_gain(note: u8) -> f32 {
    let f0 = midi_to_hz(note).max(MIN_FREQUENCY_HZ);
    let span = (HIGHEST_KEY - LOWEST_KEY) as f32;
    let t = (note.saturating_sub(LOWEST_KEY) as f32 / span).clamp(0.0, 1.0);
    // Geometric taper: T60 = bass · (treble/bass)^t.
    let t60 = STRING_T60_BASS_SECS * (STRING_T60_TREBLE_SECS / STRING_T60_BASS_SECS).powf(t);
    let gain = (-6.907_755_3 / (f0 * t60)).exp(); // ln(1e-3) = −6.907_755
                                                  // Stay strictly below 1 (decay guaranteed) and clear of instability.
    gain.clamp(0.0, 0.999_99)
}

/// Per-sample damper-gain decay factor for `note`. Real piano dampers are
/// felts whose engagement time depends on the string they meet: bass
/// strings carry much more energy and the felt takes longer to mute them;
/// treble strings stop almost instantly under their tiny dampers. We
/// linearly taper τ across the playable range A0..C8. The `clamp` alone
/// bounds out-of-range notes, so no separate `min`/`max` is needed.
fn damper_decay_per_sample(note: u8, sample_rate: f32) -> f32 {
    let span = (HIGHEST_KEY - LOWEST_KEY) as f32;
    let t = (note.saturating_sub(LOWEST_KEY) as f32 / span).clamp(0.0, 1.0);
    let tau_secs = DAMPER_TAU_BASS_SECS - t * DAMPER_TAU_RANGE_SECS;
    (-1.0 / (tau_secs * sample_rate)).exp()
}

#[derive(Debug)]
pub struct Voice {
    sample_rate: f32,
    strings: StringGroup,
    hammer: Hammer,
    strike_comb: StrikeComb,
    knock: HammerKnock,
    note: u8,
    damper_gain: f32,
    damper_decay: f32,
    released: bool,
    /// Peak-hold envelope of the output, used by the auto-gate.
    env: f32,
    /// Per-sample release coefficient for `env`.
    env_release: f32,
    /// Consecutive samples `env` has stayed below [`AUTO_GATE_LEVEL`].
    quiet_samples: u32,
    /// Auto-gate trips once `quiet_samples` reaches this.
    auto_gate_hold: u32,
}

impl Voice {
    pub fn new(sample_rate: f32) -> Self {
        let max_delay = (sample_rate / MIN_FREQUENCY_HZ).ceil() as usize;
        Self {
            sample_rate,
            strings: StringGroup::new(sample_rate, max_delay),
            hammer: Hammer::new(sample_rate),
            strike_comb: StrikeComb::new(max_delay),
            knock: HammerKnock::new(sample_rate),
            note: 0,
            damper_gain: 0.0,
            // Overwritten on every note_on with a note-dependent value.
            // 0.0 (instant gate) is the safe placeholder: were a voice ever
            // to enter release without a preceding note_on, it would gate to
            // silence at once rather than ring forever — which a 1.0 "no
            // decay" default would cause.
            damper_decay: 0.0,
            released: false,
            env: 0.0,
            env_release: (-1.0 / (ENV_RELEASE_SECS * sample_rate)).exp(),
            quiet_samples: 0,
            auto_gate_hold: (AUTO_GATE_HOLD_SECS * sample_rate) as u32,
        }
    }

    pub fn note(&self) -> u8 {
        self.note
    }

    pub fn is_active(&self) -> bool {
        self.strings.is_active()
    }

    /// True while the damper is decaying after a note-off (and the
    /// strings still ring). The allocator uses this to prefer stealing
    /// fading voices over held ones.
    pub fn is_released(&self) -> bool {
        self.released
    }

    pub fn note_on(&mut self, note: u8, velocity: u8) {
        let freq = stretched_midi_to_hz(note);
        let v = (velocity as f32 / 127.0).clamp(0.0, 1.0);
        // Same-note re-strike while the strings still ring: keep the delay
        // lines intact. A real hammer hits an already-vibrating string and
        // the old motion carries through the new attack; wiping the loop
        // (the fresh-pluck path below) would cut the residue dead at the
        // instant of the strike — audible on fast repetitions and tremolo.
        // Tuning, loop gain and comb period are unchanged for the same note.
        let retrigger = self.note == note && self.strings.is_active();
        if retrigger {
            // The key lifts the damper before the hammer reaches the string,
            // so the fresh strike must sound at full level. `damper_gain` is
            // an output-side scaler, so leaving it engaged (or ramping it up
            // over milliseconds) would mute the new attack — a re-strike
            // ~0.5 s after key-up came out 30 dB down. Fold the engaged
            // damper into the string state instead: the residue keeps
            // exactly the amplitude it had (no step, no click) and the gain
            // returns to unity for everything that arrives afterwards.
            self.lift_damper();
        } else {
            self.note = note;
            self.strings.pluck(freq, strings_for_note(note));
            self.strings.set_loop_gain(string_loop_gain(note));
            self.strike_comb
                .set_period(self.sample_rate / freq.max(MIN_FREQUENCY_HZ));
            self.damper_gain = 1.0;
        }
        self.hammer.fire(v, freq);
        self.knock.fire(v, freq);
        self.damper_decay = damper_decay_per_sample(note, self.sample_rate);
        self.released = false;
        self.env = 0.0;
        self.quiet_samples = 0;
    }

    pub fn note_off(&mut self) {
        // The top of the keyboard has no dampers: those notes ignore key-up
        // and simply ring down at their natural loop-gain rate, reclaimed by
        // the envelope auto-gate once inaudible.
        if self.note > HIGHEST_DAMPED_NOTE {
            return;
        }
        if self.strings.is_active() {
            self.released = true;
        }
    }

    /// Lift the damper off a voice that was already in release decay: the
    /// felt had been engaging, so the string keeps exactly the amplitude it
    /// had reached — but it must then ring on at its own loop rate rather
    /// than stay pinned under a partially engaged damper.
    ///
    /// Folding the engaged `damper_gain` into the string state and
    /// restoring the gain to 1.0 does both at once: the output is
    /// continuous across the lift, and the voice is no longer scaled by a
    /// stale value for the rest of its life.
    fn lift_damper(&mut self) {
        if self.damper_gain < 1.0 {
            self.strings.scale_state(self.damper_gain);
            self.damper_gain = 1.0;
        }
    }

    /// Pedal pressed while this voice was in release decay — the damper bar
    /// retracts, so stop the release and lift the felt (see
    /// [`Voice::lift_damper`]).
    pub fn cancel_release(&mut self) {
        self.released = false;
        self.lift_damper();
    }

    #[inline]
    pub fn tick(&mut self) -> f32 {
        if !self.strings.is_active() {
            return 0.0;
        }
        let exc = self.strike_comb.tick(self.hammer.tick());
        let s = self.strings.tick(exc);
        // The knock adds outside the damper product: mechanical noise is not
        // muted by the damper felt (it never touches the string loop).
        let out = s * self.damper_gain + self.knock.tick();
        if self.released {
            self.damper_gain *= self.damper_decay;
            if self.damper_gain < DAMPER_GATE_THRESHOLD {
                self.strings.deactivate();
                self.damper_gain = 0.0;
                self.released = false;
            }
        }
        // Envelope auto-gate: peak-hold follower with slow release. Once the
        // voice has been inaudible long enough, free the slot — covers
        // undamped treble notes and held notes whose loop gain decayed them
        // to silence (neither crosses the damper gate above).
        let a = out.abs();
        self.env = if a > self.env {
            a
        } else {
            self.env * self.env_release
        };
        if self.env < AUTO_GATE_LEVEL {
            self.quiet_samples += 1;
            if self.quiet_samples >= self.auto_gate_hold {
                self.strings.deactivate();
                self.damper_gain = 0.0;
                self.released = false;
                self.env = 0.0;
                self.quiet_samples = 0;
            }
        } else {
            self.quiet_samples = 0;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a4_resolves_to_440_hz() {
        assert!((midi_to_hz(69) - 440.0).abs() < 1e-3);
    }

    #[test]
    fn c4_resolves_to_about_261_63() {
        assert!((midi_to_hz(60) - 261.625_56).abs() < 1e-2);
    }

    /// Deviation of the stretched scale from equal temperament, in cents.
    fn stretch_cents(note: u8) -> f32 {
        1200.0 * (stretched_midi_to_hz(note) / midi_to_hz(note)).log2()
    }

    #[test]
    fn stretch_keeps_a4_at_440() {
        assert!((stretched_midi_to_hz(69) - 440.0).abs() < 1e-3);
    }

    #[test]
    fn stretch_is_flat_in_bass_and_sharp_in_treble() {
        // Railsback: A0 lands tens of cents flat, C8 tens of cents sharp.
        let a0 = stretch_cents(21);
        let c8 = stretch_cents(108);
        assert!((-40.0..=-15.0).contains(&a0), "A0 should be flat: {a0}");
        assert!((15.0..=40.0).contains(&c8), "C8 should be sharp: {c8}");
    }

    #[test]
    fn stretch_is_gentle_in_the_middle_octaves() {
        // The cubic must leave the middle of the keyboard near ET — the
        // curve's steepness lives at the extremes, not around middle C.
        for note in 57..=81 {
            // A3..A5
            let c = stretch_cents(note);
            assert!(c.abs() < 5.0, "note {note} deviates {c} cents");
        }
    }

    #[test]
    fn stretch_preserves_pitch_ordering() {
        // ±30 cents of stretch is far below a semitone: frequency must stay
        // strictly increasing across the whole keyboard.
        for note in 21..108u8 {
            assert!(
                stretched_midi_to_hz(note + 1) > stretched_midi_to_hz(note),
                "ordering inverted at note {note}"
            );
        }
    }

    #[test]
    fn retrigger_keeps_residual_string_energy() {
        // Strike hard, let it ring 200 ms, then re-strike the same note at
        // velocity 1 (a near-silent hammer). The old vibration must survive
        // the re-strike: the output right after has to stay far above what a
        // fresh velocity-1 note produces. Before the retrigger path, pluck()
        // wiped the delay lines and the residue cut dead at the new attack.
        fn rms_after_quiet_strike(prime: bool) -> f32 {
            let mut v = Voice::new(48_000.0);
            if prime {
                v.note_on(60, 120);
                for _ in 0..9_600 {
                    v.tick();
                }
            }
            v.note_on(60, 1);
            let mut sq = 0.0f32;
            let n = 4_800;
            for _ in 0..n {
                let y = v.tick();
                sq += y * y;
            }
            (sq / n as f32).sqrt()
        }
        let retriggered = rms_after_quiet_strike(true);
        let fresh = rms_after_quiet_strike(false);
        assert!(
            retriggered > fresh * 5.0,
            "residue should survive the re-strike: retriggered={retriggered} fresh={fresh}"
        );
    }

    fn rms(v: &mut Voice, n: usize) -> f32 {
        let mut sq = 0.0f32;
        for _ in 0..n {
            let y = v.tick();
            sq += y * y;
        }
        (sq / n as f32).sqrt()
    }

    #[test]
    fn retrigger_mid_release_lifts_damper_without_discontinuity() {
        // Re-striking a releasing note lifts the damper. The engaged gain is
        // folded into the string state, so the surviving residue keeps its
        // level (no step, no click) while `damper_gain` returns to unity —
        // simply snapping the gain to 1.0 would multiply the residue up by
        // 1/gain, an audible click.
        let mut v = Voice::new(48_000.0);
        v.note_on(60, 100);
        for _ in 0..4_800 {
            v.tick();
        }
        v.note_off();
        for _ in 0..2_400 {
            v.tick(); // 50 ms of damper decay
        }
        assert!(
            v.damper_gain < 1.0,
            "damper should have engaged: {}",
            v.damper_gain
        );
        let before = rms(&mut v, 256);
        // Silent re-strike: isolates the damper lift from the new hammer.
        v.note_on(60, 0);
        assert_eq!(v.damper_gain, 1.0, "damper must be fully lifted");
        assert!(!v.is_released());
        let after = rms(&mut v, 256);
        let ratio = after / before.max(1e-9);
        assert!(
            (0.8..1.25).contains(&ratio),
            "residue level jumped across the re-strike: {before} → {after}"
        );
    }

    #[test]
    fn retrigger_deep_in_release_sounds_at_full_level() {
        // Regression: `damper_gain` scales the voice *output*, so a re-strike
        // while the felt is nearly closed used to be multiplied down to
        // near-silence — a note repeated ~0.5 s after key-up came out ~30 dB
        // quiet. The damper must be off the string by the time the hammer
        // lands, at every point in the release.
        fn attack_peak(release_secs: f32) -> f32 {
            let mut v = Voice::new(48_000.0);
            if release_secs >= 0.0 {
                v.note_on(60, 120);
                for _ in 0..4_800 {
                    v.tick();
                }
                v.note_off();
                for _ in 0..(48_000.0 * release_secs) as usize {
                    v.tick();
                }
            }
            v.note_on(60, 120);
            let mut peak = 0.0f32;
            for _ in 0..240 {
                // first 5 ms after the strike
                peak = peak.max(v.tick().abs());
            }
            peak
        }
        let fresh = attack_peak(-1.0); // virgin voice, nothing to lift
        for &secs in &[0.05, 0.2, 0.35, 0.5, 0.65, 0.8] {
            let p = attack_peak(secs);
            assert!(
                p > fresh * 0.5,
                "re-strike {secs}s into the release is muted: {p} vs fresh {fresh}"
            );
        }
    }

    #[test]
    fn pedal_lift_mid_release_restores_unity_damper_gain() {
        // `cancel_release` models the pedal retracting the damper bar. The
        // voice must not stay scaled by the stale, partially decayed gain for
        // the rest of its life (it used to sit ~30 dB down forever).
        let mut v = Voice::new(48_000.0);
        v.note_on(60, 110);
        for _ in 0..4_800 {
            v.tick();
        }
        v.note_off();
        for _ in 0..(48_000 * 3 / 10) {
            v.tick(); // 300 ms of damper decay
        }
        assert!(v.damper_gain < 0.1, "damper should be well engaged");
        let before = rms(&mut v, 256);
        v.cancel_release();
        assert_eq!(v.damper_gain, 1.0, "pedal must lift the damper fully");
        let after = rms(&mut v, 256);
        let ratio = after / before.max(1e-9);
        assert!(
            (0.8..1.25).contains(&ratio),
            "level jumped when the pedal lifted the damper: {before} → {after}"
        );
    }

    #[test]
    fn knock_reaches_the_output_past_the_damper() {
        // The knock is summed *outside* the damper product: structural noise
        // never touches the string loop, so the felt cannot mute it. Hold a
        // voice with the damper fully closed (gain driven to ~0 by a long
        // release) and re-strike: with the knock wired up the attack still
        // carries energy, and it decays away on the knock's own envelope.
        // Without the `+ knock.tick()` term this window would be silent.
        let mut v = Voice::new(48_000.0);
        v.note_on(60, 120);
        // Force the damper hard shut without gating the strings off, so the
        // string term contributes exactly zero and only the knock is left.
        v.damper_gain = 0.0;
        let knock_rms = rms(&mut v, 480); // 10 ms
        assert!(
            knock_rms > 1e-4,
            "knock should survive a closed damper: {knock_rms}"
        );
        // …and it is a short burst, not a sustained tone.
        let later = rms(&mut v, 480);
        assert!(
            later < knock_rms,
            "knock should decay: {knock_rms} → {later}"
        );
    }

    #[test]
    fn damper_decay_never_reaches_unity() {
        // The decay factor must stay strictly inside (0, 1) across the whole
        // MIDI range. A factor ≥ 1.0 would make a released voice's
        // damper_gain hold or grow, so it would never cross the gate — a
        // runaway/stuck voice on the RT thread. This locks the invariant
        // τ > 0, i.e. DAMPER_TAU_RANGE_SECS < DAMPER_TAU_BASS_SECS.
        for sr in [44_100.0, 48_000.0, 96_000.0] {
            for note in 0..=127u8 {
                let d = damper_decay_per_sample(note, sr);
                assert!(
                    d > 0.0 && d < 1.0,
                    "damper decay out of (0,1) at note {note}, sr {sr}: {d}"
                );
            }
        }
    }

    /// Measure the time a held note (no note-off) takes to fall −60 dB below
    /// its attack peak. Returns `max_secs` if it never crosses (i.e. it would
    /// "ring forever") so the caller can assert a finite decay.
    #[cfg(test)]
    fn held_note_t60_secs(note: u8, max_secs: f32) -> f32 {
        let sr = 48_000.0;
        let mut v = Voice::new(sr);
        v.note_on(note, 110);
        let n = (sr * max_secs) as usize;
        let warm = (sr * 0.1) as usize;
        let mut buf = vec![0.0f32; n];
        let mut peak = 0.0f32;
        for (i, s) in buf.iter_mut().enumerate() {
            *s = v.tick();
            if i < warm {
                peak = peak.max(s.abs());
            }
        }
        let thresh = peak * 1e-3; // −60 dB
        let win = 2_048usize;
        let mut last = 0.0f32;
        let mut i = warm;
        while i + win < n {
            let rms = (buf[i..i + win].iter().map(|x| x * x).sum::<f32>() / win as f32).sqrt();
            if rms > thresh {
                last = (i + win) as f32 / sr;
            }
            i += win;
        }
        last
    }

    #[test]
    fn struck_string_decays_in_realistic_time_per_register() {
        // With per-note loop gain a held note must decay to −60 dB in a
        // physically plausible time (seconds — not forever, not instantly),
        // and the bass must sustain audibly longer than the treble. Before
        // the per-note loop gain the fundamental never decayed (organ-like).
        // The bass tail is carried by the slow polarization loop (T60 factor
        // 1.9 at −9 dB weight), so the audible T60 lands well past the
        // nominal 15 s target — that aftersound is the two-stage decay
        // working as intended, matching the tens of seconds a real grand's
        // bass quietly rings.
        let bass = held_note_t60_secs(36, 28.0); // C2
        let treble = held_note_t60_secs(96, 10.0); // C7
        assert!(
            bass > treble + 1.0,
            "bass should sustain longer than treble: bass={bass} treble={treble}"
        );
        // Finite decay — would equal the cap if it rang forever.
        assert!(
            bass < 27.0,
            "bass should still decay (not ring forever): {bass}"
        );
        assert!(
            (2.0..9.5).contains(&treble),
            "treble T60 out of plausible range: {treble}"
        );
        assert!(
            (10.0..27.0).contains(&bass),
            "bass T60 out of plausible range: {bass}"
        );
    }

    #[test]
    fn fresh_voice_is_idle_and_silent() {
        let mut v = Voice::new(48_000.0);
        assert!(!v.is_active());
        assert!(!v.is_released());
        assert_eq!(v.tick(), 0.0);
    }

    #[test]
    fn note_on_activates_voice_and_produces_signal() {
        let mut v = Voice::new(48_000.0);
        v.note_on(69, 100);
        assert!(v.is_active());
        assert!(!v.is_released());
        let mut any_nonzero = false;
        for _ in 0..1_024 {
            if v.tick().abs() > 0.0 {
                any_nonzero = true;
                break;
            }
        }
        assert!(any_nonzero);
    }

    #[test]
    fn note_off_sets_released_then_drains_to_idle() {
        let mut v = Voice::new(48_000.0);
        v.note_on(60, 127);
        for _ in 0..1_000 {
            v.tick();
        }
        v.note_off();
        assert!(v.is_released());
        for _ in 0..60_000 {
            v.tick();
        }
        assert!(!v.is_active());
        assert!(!v.is_released());
    }

    #[test]
    fn undamped_treble_ignores_note_off_then_auto_gates() {
        // Note 100 is above HIGHEST_DAMPED_NOTE, so key-up must NOT mute it —
        // a damped note would be silent ~0.6 s after note_off, this one keeps
        // ringing at its natural rate. Eventually the envelope auto-gate
        // reclaims the slot so it does not hang forever.
        let mut v = Voice::new(48_000.0);
        v.note_on(100, 110);
        for _ in 0..4_800 {
            v.tick();
        }
        v.note_off();
        let mut still_ringing = false;
        for _ in 0..48_000 {
            if v.tick().abs() > 1.0e-3 {
                still_ringing = true;
            }
        }
        assert!(
            still_ringing,
            "undamped treble should ring on after note_off"
        );
        assert!(
            v.is_active(),
            "undamped treble still active 1 s after key-up"
        );
        // Let it decay fully; the auto-gate must free the slot.
        for _ in 0..48_000 * 9 {
            v.tick();
        }
        assert!(!v.is_active(), "auto-gate should reclaim the silent voice");
    }

    #[test]
    fn damped_note_still_silences_after_note_off() {
        // Regression guard: a normal (damped) note must still stop on key-up.
        let mut v = Voice::new(48_000.0);
        v.note_on(60, 110);
        for _ in 0..4_800 {
            v.tick();
        }
        v.note_off();
        assert!(v.is_released());
        for _ in 0..96_000 {
            v.tick();
        }
        assert!(
            !v.is_active(),
            "damped note should drain to idle after key-up"
        );
    }

    #[test]
    fn velocity_zero_produces_silence() {
        let mut v = Voice::new(48_000.0);
        v.note_on(60, 0);
        for _ in 0..4_096 {
            assert_eq!(v.tick(), 0.0);
        }
    }

    #[test]
    fn bass_note_uses_single_string() {
        // Note 30 (F#1) falls into the singles range.
        let mut v = Voice::new(48_000.0);
        v.note_on(30, 100);
        // We don't expose string count directly from the voice; verify
        // indirectly via the underlying group.
        let count = v.strings.active_count();
        assert_eq!(count, 1, "expected bass to be single-string, got {count}");
    }

    #[test]
    fn treble_note_uses_three_strings() {
        let mut v = Voice::new(48_000.0);
        v.note_on(72, 100); // C5
        assert_eq!(v.strings.active_count(), 3);
    }

    #[test]
    fn hard_strike_has_more_high_frequency_content_than_soft() {
        use crate::domain::filter::OnePoleLowpass;

        fn early_hp_rms(velocity: u8) -> f32 {
            let mut v = Voice::new(48_000.0);
            v.note_on(60, velocity);
            let mut buf = vec![0.0; 1_024];
            for s in buf.iter_mut() {
                *s = v.tick();
            }
            let s = &buf[100..];
            let mut lp = OnePoleLowpass::new();
            lp.set_cutoff(2_000.0, 48_000.0);
            let mut hp_sq = 0.0;
            for &x in s {
                let lp_out = lp.tick(x);
                let hp = x - lp_out;
                hp_sq += hp * hp;
            }
            (hp_sq / s.len() as f32).sqrt()
        }
        let hard = early_hp_rms(120);
        let soft = early_hp_rms(30);
        // 3× is the qualitative-correctness threshold: it confirms the
        // felt-compression LPF actually injects HF differentially with
        // velocity. The absolute ratio depends on `AMPLITUDE_WARP` and
        // is fine-tuned by ear, not by this test.
        assert!(
            hard > soft * 3.0,
            "expected harder strike to inject more HF energy: hard={hard} soft={soft}"
        );
    }
}
