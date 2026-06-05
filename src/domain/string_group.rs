//! Group of up to 3 detuned strings sharing one excitation.
//!
//! ## Why multiple strings
//! A grand piano uses 1, 2, or 3 strings per note depending on the register:
//! - **Bass (≈ A0–G2)**: a single wound string, low-mass copper wrap on a
//!   steel core. The long, heavy string already produces enough fundamental
//!   energy.
//! - **Tenor (≈ G#2–A#2)**: two strings, often a narrow-spread transition
//!   between bass and triple registers.
//! - **Mid/Treble (≈ B2 upward)**: three strings, tuned slightly apart.
//!
//! Sub-cent detuning between strings is what gives the piano its
//! characteristic warmth — the slow beating between partials at `f` and
//! `f+δ` produces an amplitude modulation at rate `δ`. Listeners hear this
//! as the "body" of the tone and usually attribute it to the soundboard.
//!
//! ## What this module does NOT model
//! Real strings are mechanically coupled through the bridge, which causes
//! a "two-stage decay": an initial fast decay from the in-phase mode whose
//! energy leaves through the bridge, then a long slow tail from the
//! out-of-phase mode whose net force on the bridge is small (Weinreich
//! 1977). Implementing that needs a coupled-waveguide network with a
//! shared termination filter — useful enough to be a candidate for a
//! Phase 5b refinement, but out of scope here.

use crate::domain::string::KarplusStrong;

/// Hardcoded maximum number of strings per note. Three is the most a real
/// piano uses; allocating for it up-front lets every voice share the same
/// fixed layout.
pub const MAX_STRINGS_PER_NOTE: usize = 3;

/// Detune offsets in cents, indexed by `[active_count][string_index]`.
/// `active_count == 0` is unused; we keep the row to make the lookup
/// total. Detune values were picked by ear for warmth without sounding
/// detuned: ±0.7 cent for doubles, ±1 cent for triples.
const DETUNE_CENTS: [[f32; MAX_STRINGS_PER_NOTE]; MAX_STRINGS_PER_NOTE + 1] = [
    [0.0, 0.0, 0.0],  // n=0 (safe default, never used in practice)
    [0.0, 0.0, 0.0],  // n=1: single string, no detune
    [-0.7, 0.7, 0.0], // n=2: doubles, symmetric ±0.7 cent
    [-1.0, 0.0, 1.0], // n=3: triples, centre + ±1 cent
];

/// `1/√N` normalisation factors, indexed by `active_count`. Pre-computed
/// to keep the audio path free of square roots.
const STRING_NORM_FACTOR: [f32; MAX_STRINGS_PER_NOTE + 1] = [
    0.0,          // n=0 (silent)
    1.0,          // 1/√1
    0.707_106_77, // 1/√2
    0.577_350_30, // 1/√3
];

/// Map a MIDI note to its string count. Crossovers chosen to match the
/// general layout of a grand piano without being pedantic about specific
/// manufacturer transitions.
pub fn strings_for_note(note: u8) -> usize {
    if note <= 31 {
        1
    } else if note <= 43 {
        2
    } else {
        3
    }
}

#[derive(Debug)]
pub struct StringGroup {
    strings: [KarplusStrong; MAX_STRINGS_PER_NOTE],
    /// How many of the strings are currently driven by the loop. The
    /// remaining slots stay inactive — `tick(_)` returns 0 for them.
    active_count: usize,
}

impl StringGroup {
    pub fn new(sample_rate: f32, max_delay: usize) -> Self {
        Self {
            strings: std::array::from_fn(|_| KarplusStrong::new(sample_rate, max_delay)),
            active_count: 0,
        }
    }

    pub fn active_count(&self) -> usize {
        self.active_count
    }

    /// Set the per-cycle loop loss on every string in the group. The voice
    /// uses this to give each note a pitch-dependent decay rate: without it
    /// the only loss is the loop LPF, which barely touches the fundamental
    /// of bass/mid notes (they would ring almost forever). See
    /// [`crate::domain::voice`] for the T60 → loop-gain mapping.
    pub fn set_loop_gain(&mut self, gain: f32) {
        for s in &mut self.strings {
            s.set_loop_gain(gain);
        }
    }

    /// Any string still ringing → the group is active.
    pub fn is_active(&self) -> bool {
        self.strings.iter().any(|s| s.is_active())
    }

    pub fn deactivate(&mut self) {
        for s in &mut self.strings {
            s.deactivate();
        }
    }

    /// Pluck `n_strings` (clamped to [1, MAX]) tuned around `center_hz`
    /// with the per-count detune profile. Strings beyond `n_strings`
    /// are deactivated.
    pub fn pluck(&mut self, center_hz: f32, n_strings: usize) {
        let n = n_strings.clamp(1, MAX_STRINGS_PER_NOTE);
        self.active_count = n;
        let detunes = DETUNE_CENTS[n];
        for i in 0..n {
            let f = center_hz * 2.0f32.powf(detunes[i] / 1200.0);
            self.strings[i].pluck(f);
        }
        for i in n..MAX_STRINGS_PER_NOTE {
            self.strings[i].deactivate();
        }
    }

    /// Excite each active string with the same input, return the
    /// amplitude-normalised sum.
    #[inline]
    pub fn tick(&mut self, excitation: f32) -> f32 {
        if self.active_count == 0 {
            return 0.0;
        }
        let mut sum = 0.0f32;
        for i in 0..self.active_count {
            sum += self.strings[i].tick(excitation);
        }
        // Normalise by √N rather than N. Detuned strings drift out of
        // phase very quickly: the peak grows ≈ N at the attack but the
        // RMS grows ≈ √N once partials decohere. Dividing by √N keeps
        // *perceived loudness* roughly even across registers — singles
        // (bass) at one string and triples (treble) at three sum to
        // comparable subjective levels, instead of triples being −10 dB
        // quieter than singles.
        sum * STRING_NORM_FACTOR[self.active_count]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strings_for_note_matches_register_layout() {
        assert_eq!(strings_for_note(21), 1); // A0 — bass
        assert_eq!(strings_for_note(31), 1); // G2 — bass
        assert_eq!(strings_for_note(32), 2); // G#2 — doubles
        assert_eq!(strings_for_note(43), 2); // G3 — doubles
        assert_eq!(strings_for_note(44), 3); // G#3 — triples
        assert_eq!(strings_for_note(108), 3); // C8 — triples
    }

    #[test]
    fn fresh_group_is_inactive() {
        let g = StringGroup::new(48_000.0, 4096);
        assert!(!g.is_active());
        assert_eq!(g.active_count(), 0);
    }

    #[test]
    fn pluck_arms_the_requested_number_of_strings() {
        let mut g = StringGroup::new(48_000.0, 4096);
        g.pluck(440.0, 3);
        assert_eq!(g.active_count(), 3);
        assert!(g.is_active());
    }

    #[test]
    fn pluck_clamps_count_into_valid_range() {
        let mut g = StringGroup::new(48_000.0, 4096);
        g.pluck(440.0, 0);
        assert_eq!(g.active_count(), 1);
        g.pluck(440.0, 99);
        assert_eq!(g.active_count(), 3);
    }

    #[test]
    fn ticked_group_produces_non_zero_signal() {
        let mut g = StringGroup::new(48_000.0, 4096);
        g.pluck(440.0, 3);
        let mut any_nonzero = false;
        // Excite once, then let the group ring.
        g.tick(1.0);
        for _ in 0..1_024 {
            if g.tick(0.0).abs() > 0.0 {
                any_nonzero = true;
                break;
            }
        }
        assert!(any_nonzero);
    }

    #[test]
    fn deactivate_silences_all_strings() {
        let mut g = StringGroup::new(48_000.0, 4096);
        g.pluck(440.0, 3);
        g.deactivate();
        for _ in 0..512 {
            assert_eq!(g.tick(0.0), 0.0);
        }
    }

    /// Detune breaks perfect periodicity. With 3 strings tuned slightly
    /// apart, the summed signal is *quasi*-periodic — locally it looks
    /// like a clean note, globally the partials drift out of phase. The
    /// autocorrelation at the fundamental period drops accordingly.
    ///
    /// This is a deterministic way to check that the detune values are
    /// actually being applied. (Beat *frequency* at piano detune ranges
    /// is sub-Hz, so a unit-test buffer is too short to see the envelope
    /// modulation directly — that's a listening test.)
    #[test]
    fn detune_reduces_periodicity_vs_single_string() {
        fn normalized_acf(samples: &[f32], lag: usize) -> f32 {
            let mut acf = 0.0;
            let mut energy = 0.0;
            for i in 0..(samples.len() - lag) {
                acf += samples[i] * samples[i + lag];
            }
            for &x in samples {
                energy += x * x;
            }
            acf / energy.max(1e-12)
        }
        fn render_n_strings(n: usize, freq: f32, samples: usize) -> Vec<f32> {
            let mut g = StringGroup::new(48_000.0, 4096);
            g.pluck(freq, n);
            let mut buf = vec![0.0; samples];
            buf[0] = g.tick(1.0);
            for v in buf.iter_mut().skip(1) {
                *v = g.tick(0.0);
            }
            buf
        }
        // A4 at 440 Hz → period 109 samples; 0.4 s buffer is enough for
        // detune-induced phase drift to be visible in the ACF.
        let buf_single = render_n_strings(1, 440.0, 19_200);
        let buf_triple = render_n_strings(3, 440.0, 19_200);
        let lag = 109;
        let acf_s = normalized_acf(&buf_single[400..], lag);
        let acf_t = normalized_acf(&buf_triple[400..], lag);
        assert!(
            acf_t < acf_s,
            "triple-string periodicity ({acf_t}) should be lower than single-string ({acf_s})"
        );
    }
}
