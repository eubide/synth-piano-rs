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
//! ## Bass: two polarizations instead of two strings
//! A single wound bass string still beats: it vibrates in two transverse
//! polarizations (vertical and horizontal) whose effective terminations at
//! the bridge differ slightly, detuning them by a cent or so (Weinreich
//! 1977). The partials of the two polarizations drift in and out of phase —
//! partial `k` beats at `k` times the fundamental detune — producing the
//! slow shimmer a real bass note has. A lone KS loop decays as a featureless
//! exponential instead, which reads as "electronic". For single-string notes
//! we arm a second, quieter loop detuned by ~+1.4 cents to stand in for the
//! horizontal polarization. It reuses one of the pre-allocated string slots,
//! so it costs nothing extra at construction time.
//!
//! ## Two-stage decay (approximated)
//! Real strings are mechanically coupled through the bridge, which causes
//! a "two-stage decay": an initial fast decay from the in-phase mode whose
//! energy leaves through the bridge, then a long slow tail ("aftersound")
//! from the out-of-phase mode whose net force on the bridge is small
//! (Weinreich 1977). The full model needs a coupled-waveguide network with
//! a shared termination filter; we approximate its *audible signature*
//! instead: each unison string gets a different T60 (see
//! [`DECAY_T60_FACTORS`]), so the summed envelope starts at the average
//! decay rate and flattens as the slowest string takes over — a convex
//! dB-envelope with a distinct knee, where a single loop gain gives the
//! straight-line exponential that reads as "electronic". Single-string
//! bass notes get the same treatment through their polarization pair: the
//! horizontal polarization decays much more slowly
//! ([`BASS_POLARIZATION_T60_FACTOR`]), which is precisely Weinreich's
//! vertical→horizontal aftersound.

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

/// Detune (in cents) of the second-polarization loop armed for
/// single-string bass notes, and its output weight. +1.4 cents puts the
/// 10th partial of C2 at a ~0.5 Hz beat — slow shimmer, not chorus; the
/// fundamental itself beats over tens of seconds and stays solid. The
/// weight keeps the horizontal polarization clearly subordinate (the
/// hammer strikes vertically; the second polarization is only leaked
/// into via the bridge).
const BASS_POLARIZATION_DETUNE_CENTS: f32 = 1.4;
const BASS_POLARIZATION_WEIGHT: f32 = 0.35;

/// Per-string T60 multipliers, indexed by `[active_count][string_index]`.
/// The spread around the nominal per-note T60 is what produces the
/// two-stage decay (see module docs): the fastest string dominates the
/// early slope, the slowest owns the tail. Values keep the *geometric
/// mean* close to 1 so the overall note length stays near the nominal
/// target the voice installs.
const DECAY_T60_FACTORS: [[f32; MAX_STRINGS_PER_NOTE]; MAX_STRINGS_PER_NOTE + 1] = [
    [1.0, 1.0, 1.0],  // n=0 (unused)
    [1.0, 1.0, 1.0],  // n=1: single string (polarization handled below)
    [0.75, 1.3, 1.0], // n=2
    [0.7, 1.0, 1.35], // n=3
];

/// T60 multiplier for the bass polarization loop. The horizontal
/// polarization couples weakly to the bridge, so it outlives the struck
/// vertical polarization by a wide margin — Weinreich's "aftersound". At
/// its 0.35 output weight the tail sits ≈ 9 dB below the note's body and
/// emerges as the main string fades.
const BASS_POLARIZATION_T60_FACTOR: f32 = 1.9;

/// Half-width of the per-note detune jitter, as a fraction of the nominal
/// detune. A technician never leaves every unison at the *same* offset:
/// each note carries its own micro-tuning, and that per-note variation of
/// beat rates is part of why 88 keys read as one organic instrument rather
/// than one sample transposed. Jitter is derived deterministically from the
/// note's frequency bits, so a given note always beats the same way (its
/// "fingerprint") and renders stay reproducible.
const DETUNE_JITTER: f32 = 0.35;

/// Deterministic multiplier in `[1 − DETUNE_JITTER, 1 + DETUNE_JITTER]`
/// derived from `seed` via one xorshift32 round.
fn detune_jitter_factor(seed: u32) -> f32 {
    let mut x = seed | 1;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    let unit = x as f32 / 4_294_967_296.0; // [0, 1)
    1.0 + DETUNE_JITTER * (2.0 * unit - 1.0)
}

/// `1/√N` normalisation factors, indexed by `active_count`. Pre-computed
/// to keep the audio path free of square roots.
const STRING_NORM_FACTOR: [f32; MAX_STRINGS_PER_NOTE + 1] = [
    0.0,          // n=0 (silent)
    1.0,          // 1/√1
    0.707_106_77, // 1/√2
    0.577_350_3,  // 1/√3
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
    /// How many *physical* strings are currently driven by the loop. The
    /// remaining slots stay inactive — `tick(_)` returns 0 for them.
    active_count: usize,
    /// Single-string bass note: slot 1 holds the second-polarization loop
    /// (see module docs), mixed in at [`BASS_POLARIZATION_WEIGHT`].
    bass_polarization: bool,
}

impl StringGroup {
    pub fn new(sample_rate: f32, max_delay: usize) -> Self {
        Self {
            strings: std::array::from_fn(|_| KarplusStrong::new(sample_rate, max_delay)),
            active_count: 0,
            bass_polarization: false,
        }
    }

    pub fn active_count(&self) -> usize {
        self.active_count
    }

    /// Set the per-cycle loop loss for the group from the note's *nominal*
    /// loop gain (see [`crate::domain::voice`] for the T60 → gain mapping).
    /// Each string receives `gain^(1/factor)` — a per-cycle gain whose T60
    /// is the nominal times its [`DECAY_T60_FACTORS`] entry — so the unison
    /// decays at spread rates and the summed envelope shows the two-stage
    /// knee (module docs). Call after [`StringGroup::pluck`], which sets
    /// the string count the factor lookup depends on.
    pub fn set_loop_gain(&mut self, gain: f32) {
        let factors = &DECAY_T60_FACTORS[self.active_count];
        for (s, &factor) in self.strings.iter_mut().zip(factors).take(self.active_count) {
            s.set_loop_gain(gain.powf(1.0 / factor));
        }
        if self.bass_polarization {
            self.strings[1].set_loop_gain(gain.powf(1.0 / BASS_POLARIZATION_T60_FACTOR));
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

    /// Scale every string's loop state by `g` — see
    /// [`KarplusStrong::scale_state`]. The group's output drops to `g` times
    /// what it was, continuously, while any excitation fed in afterwards
    /// still passes at full level.
    pub fn scale_state(&mut self, g: f32) {
        for s in &mut self.strings {
            s.scale_state(g);
        }
    }

    /// Pluck `n_strings` (clamped to [1, MAX]) tuned around `center_hz`
    /// with the per-count detune profile, each offset scaled by the note's
    /// deterministic jitter fingerprint. Strings beyond `n_strings` are
    /// deactivated.
    pub fn pluck(&mut self, center_hz: f32, n_strings: usize) {
        let n = n_strings.clamp(1, MAX_STRINGS_PER_NOTE);
        self.active_count = n;
        let detunes = DETUNE_CENTS[n];
        let seed = center_hz.to_bits();
        for (i, &cents) in detunes.iter().enumerate().take(n) {
            let jittered = cents
                * detune_jitter_factor(seed.wrapping_add((i as u32).wrapping_mul(0x9E37_79B9)));
            let f = center_hz * 2.0f32.powf(jittered / 1200.0);
            self.strings[i].pluck(f);
        }
        for i in n..MAX_STRINGS_PER_NOTE {
            self.strings[i].deactivate();
        }
        // Single wound bass string: arm a second loop in the spare slot as
        // its horizontal polarization (see module docs).
        self.bass_polarization = n == 1;
        if self.bass_polarization {
            let jittered = BASS_POLARIZATION_DETUNE_CENTS
                * detune_jitter_factor(seed.wrapping_add(3u32.wrapping_mul(0x9E37_79B9)));
            let f = center_hz * 2.0f32.powf(jittered / 1200.0);
            self.strings[1].pluck(f);
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
        let mut out = sum * STRING_NORM_FACTOR[self.active_count];
        // The bass polarization pair is a perturbation on top of the single
        // string, not a second unison string — mixed at its fixed weight,
        // outside the √N normalisation (the energy it adds is ~6 %).
        if self.bass_polarization {
            out += self.strings[1].tick(excitation) * BASS_POLARIZATION_WEIGHT;
        }
        out
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
    fn single_string_bass_arms_second_polarization() {
        // A single-string note must still report one *physical* string but
        // keep a second (detuned, quieter) loop ringing as its horizontal
        // polarization — the source of the slow bass shimmer.
        let mut g = StringGroup::new(48_000.0, 4096);
        g.pluck(65.4, 1);
        assert_eq!(g.active_count(), 1);
        assert!(g.bass_polarization);
        assert!(g.strings[1].is_active(), "polarization loop should ring");
        // Multi-string notes must NOT arm it — their unison detune already
        // provides the beating.
        g.pluck(440.0, 3);
        assert!(!g.bass_polarization);
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

    #[test]
    fn detune_jitter_is_bounded_and_note_specific() {
        // Every factor stays inside [1−J, 1+J]; the same seed always yields
        // the same factor (reproducible renders); and across the keyboard the
        // factors actually vary (no uniform beat fingerprint).
        let mut distinct = std::collections::HashSet::new();
        for note in 21..=108u32 {
            let freq = 440.0f32 * 2.0f32.powf((note as f32 - 69.0) / 12.0);
            let f = detune_jitter_factor(freq.to_bits());
            assert!(
                (1.0 - DETUNE_JITTER..=1.0 + DETUNE_JITTER).contains(&f),
                "factor out of bounds at note {note}: {f}"
            );
            assert_eq!(f, detune_jitter_factor(freq.to_bits()), "not deterministic");
            distinct.insert(f.to_bits());
        }
        assert!(
            distinct.len() > 60,
            "jitter should vary across notes: {} distinct",
            distinct.len()
        );
    }

    /// Two-stage decay: the tail of a group whose unison T60s are spread
    /// must far outlive the tail of the same group forced to a uniform
    /// nominal gain (the pre-spread behaviour). Comparing against the
    /// uniform twin isolates the spread's contribution — the loop LPF's
    /// spectral decay affects both renders identically.
    fn late_tail_rms(n_strings: usize, freq: f32, t60: f32, uniform: bool) -> f32 {
        const SR: f32 = 48_000.0;
        let mut g = StringGroup::new(SR, 4096);
        g.pluck(freq, n_strings);
        // Nominal per-cycle gain for the requested T60 (same formula the
        // voice uses): gain = exp(ln(10⁻³) / (f₀·T60)).
        let gain = (-6.907_755_3 / (freq * t60)).exp();
        if uniform {
            // Bypass the group's factor table: every loop (including the
            // bass polarization slot) decays at the nominal rate.
            for s in &mut g.strings {
                s.set_loop_gain(gain);
            }
        } else {
            g.set_loop_gain(gain);
        }
        let n = (4.0 * SR) as usize;
        let mut buf = vec![0.0f32; n];
        buf[0] = g.tick(1.0);
        for v in buf.iter_mut().skip(1) {
            *v = g.tick(0.0);
        }
        let tail = &buf[(3.5 * SR) as usize..];
        (tail.iter().map(|x| x * x).sum::<f32>() / tail.len() as f32).sqrt()
    }

    #[test]
    fn unison_spread_produces_two_stage_decay() {
        let spread = late_tail_rms(3, 440.0, 1.5, false);
        let uniform = late_tail_rms(3, 440.0, 1.5, true);
        assert!(
            spread > uniform * 4.0,
            "spread T60s should leave a much longer tail: spread={spread} uniform={uniform}"
        );
    }

    #[test]
    fn bass_polarization_produces_aftersound() {
        // Single wound string: the long-lived horizontal polarization must
        // carry the tail once the struck vertical polarization has faded.
        let spread = late_tail_rms(1, 65.4, 2.0, false);
        let uniform = late_tail_rms(1, 65.4, 2.0, true);
        assert!(
            spread > uniform * 4.0,
            "polarization aftersound missing: spread={spread} uniform={uniform}"
        );
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
