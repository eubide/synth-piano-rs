//! Velocity-dependent hammer excitation.
//!
//! ## Model
//! A raised-cosine pulse whose duration depends on the struck note (long in
//! the bass, sub-millisecond in the treble), shaped by a one-pole lowpass
//! whose cutoff frequency rises with velocity. The pulse is the *force
//! profile* of the hammer striking the string; the LPF approximates the
//! *felt compression* that smooths that force on a soft hit and sharpens it
//! on a hard hit.
//!
//! ## Why velocity → cutoff
//! Real piano felt is non-linear:
//! - Soft strike: felt deforms gradually, contact lasts longer, force pulse
//!   is "rounder", spectrum is biased to low frequencies → dull tone.
//! - Hard strike: felt compresses, contact is brief and sharp, force pulse
//!   contains higher frequencies → bright tone.
//!
//! Mapping that to a single first-order LPF cutoff `f_c(v)` is a deliberate
//! simplification (Bank & Sujbert 2003 use a full non-linear contact model)
//! but it captures the perceptual dimension we care about with very little
//! state.
//!
//! ## Mapping
//! `f_c(v) = lerp(300 Hz, 7000 Hz, v^1.5)` for `v ∈ [0, 1]`.
//! The `^1.5` warp clusters the cutoff toward 300 Hz at low velocity,
//! mirroring real felt that stays soft and damps high frequencies for `pp`
//! strikes.

use std::f32::consts::TAU;

use crate::domain::filter::OnePoleLowpass;

/// Hammer-string contact duration bounds, in seconds. Askenfelt & Jansson
/// measured roughly 4 ms (bass) down to under 1 ms (treble) on real grands:
/// the heavy, soft bass hammers press into thick wound strings far longer
/// than the light, hard treble hammers. The duration sets the excitation
/// bandwidth (raised-cosine main lobe ends at `2/T`), so this taper is what
/// gives the bass its round "thump" *and* lets the treble actually energise
/// its own fundamental — a fixed 1.5 ms pulse has almost no spectrum left
/// at C7's 2.1 kHz, leaving the top octaves attack-only.
const CONTACT_SECS_BASS: f32 = 0.0035; // at A0, 27.5 Hz
const CONTACT_SECS_TREBLE: f32 = 0.0004; // at C8, 4186 Hz

/// Velocity scaling of the contact duration. Felt hardens under fast
/// impacts, so a hard strike's contact is briefer than a soft one's —
/// Askenfelt & Jansson measured roughly a 2× spread pp → ff. The scale is
/// 1.0 at mf (v = 0.5): the [`contact_secs`] anchors describe a mf strike,
/// pp stretches to 1.3× and ff compresses to 0.7×. This widens the pulse
/// *bandwidth* with velocity (main lobe ends at `2/T`), complementing the
/// felt-compression LPF's brightness tilt with a physically distinct
/// mechanism: the pulse's own spectrum changes shape, not just its filter.
///
/// Side effect on level: the transferred impulse is `peak × T / 2`, so
/// shortening `T` with velocity trades away part of the dynamic span that
/// [`AMPLITUDE_WARP`] sets — see that constant's docs for the numbers. The
/// trade is deliberate (it buys the brightness tilt) but it is a real cost,
/// not a free win, and the two constants must be retuned together.
const CONTACT_SCALE_SOFT: f32 = 1.3; // at v = 0
const CONTACT_SCALE_HARD: f32 = 0.7; // at v = 1

/// Linear pp→ff interpolation of the contact-time scale.
fn contact_scale(velocity: f32) -> f32 {
    CONTACT_SCALE_SOFT + (CONTACT_SCALE_HARD - CONTACT_SCALE_SOFT) * velocity
}
/// Frequency anchors for register tapers (A0 and C8 fundamentals).
pub(crate) const REGISTER_REF_BASS_HZ: f32 = 27.5;
pub(crate) const REGISTER_REF_TREBLE_HZ: f32 = 4_186.0;

/// Interpolate a parameter geometrically between its A0 and C8 anchors,
/// linearly in log-frequency — string scaling is itself roughly geometric
/// across the compass, so every "bass value → treble value" taper in the
/// strike model (contact time, and the knock's cutoff / decay / gain)
/// follows this same curve. `freq` outside the compass pins to the ends.
pub(crate) fn register_taper(freq: f32, bass_value: f32, treble_value: f32) -> f32 {
    let t = ((freq / REGISTER_REF_BASS_HZ).ln()
        / (REGISTER_REF_TREBLE_HZ / REGISTER_REF_BASS_HZ).ln())
    .clamp(0.0, 1.0);
    bass_value * (treble_value / bass_value).powf(t)
}

/// Contact duration for a note with fundamental `freq`.
fn contact_secs(freq: f32) -> f32 {
    register_taper(freq, CONTACT_SECS_BASS, CONTACT_SECS_TREBLE)
}

/// Cutoff bounds for the felt-compression LPF, in Hz.
///
/// Why these numbers: the raised-cosine pulse has its main spectral lobe up
/// to ~1/contact_secs (≈ 285 Hz in the bass, ≈ 2.5 kHz in the treble). The
/// LPF only contributes audible filtering when the cutoff sits *below* the
/// pulse bandwidth. We pick:
/// - 300 Hz at v=0 → cutoff well below pulse bandwidth → noticeably muffled.
/// - 7 kHz at v=1 → cutoff well above pulse bandwidth → nearly transparent.
const CUTOFF_MIN_HZ: f32 = 300.0;
const CUTOFF_MAX_HZ: f32 = 7_000.0;
/// Velocity-to-cutoff warp exponent. Concave (>1) so that low velocities
/// cluster the cutoff toward `CUTOFF_MIN_HZ`, matching the way real felt
/// stays soft and damps high frequencies for `pp` strikes.
const VELOCITY_WARP: f32 = 1.5;

/// Exponent applied to velocity for *amplitude* scaling.
///
/// Real hammer kinetic energy scales as `v²`, giving a ~40 dB pp→ff range —
/// the dynamic span that makes a piano expressive. A purely linear curve
/// (`v^1.0`) collapses that to ~20 dB and the instrument feels dynamically
/// flat. We compromise at `1.6`: ~32 dB of amplitude range (plus the
/// felt-compression LPF's ≈ 6–8 dB of brightness-driven loudness ≈ a
/// realistic ~38 dB span) while keeping the local slope gentle enough
/// (~0.9 dB per 5-velocity step around mf) to avoid the "twitchy" feel
/// that the steeper `v²` produces in the MIDI 60–100 zone where players
/// spend most time.
///
/// ## Amplitude span is not the whole story
/// The ~32 dB above is the span of the pulse's *peak*, which this exponent
/// controls on its own. What the string actually integrates is the pulse's
/// *area* — `peak × pulse_len / 2` — and [`CONTACT_SCALE_SOFT`] shortens
/// `pulse_len` as velocity rises. Over v ∈ [0.1, 1.0] (the range the 32 dB
/// figure refers to) the area therefore spans
/// `1·0.7 / (0.1^1.6·1.24)` ≈ 22.5×, i.e. **27 dB, not 32** — the contact
/// taper gives back ≈ 5 dB of low-frequency drive range.
///
/// This is not a bug and not a sign error: the area still rises
/// monotonically with velocity (`d/dv [v^1.6·(1.3−0.6v)] > 0` on all of
/// `[0, 1]`, = +0.52 at v = 1), matching the physical fact that a harder
/// strike transfers more momentum despite its briefer contact. The
/// perceptual span also sits above 27 dB, because the shorter ff pulse puts
/// more of its energy into the partials rather than the fundamental. But if
/// you retune this exponent for a target *loudness* range, 27 dB is the
/// number to reason from. `impulse_span_matches_documented_range` pins it.
const AMPLITUDE_WARP: f32 = 1.6;

#[derive(Debug)]
pub struct Hammer {
    sample_rate: f32,
    lpf: OnePoleLowpass,
    pulse_len: usize,
    pulse_pos: usize,
    /// Velocity in [0, 1]. 0 produces silence.
    velocity: f32,
    /// Cached cutoff for inspection / tests. The LPF doesn't expose it.
    cutoff_hz: f32,
    active: bool,
}

impl Hammer {
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            lpf: OnePoleLowpass::new(),
            // Placeholder; `fire` recomputes it from the struck note.
            pulse_len: (CONTACT_SECS_TREBLE * sample_rate) as usize,
            pulse_pos: 0,
            velocity: 0.0,
            cutoff_hz: CUTOFF_MIN_HZ,
            active: false,
        }
    }

    /// Cutoff currently programmed into the felt-compression LPF, in Hz.
    /// Exposed mostly for tests and metering.
    pub fn cutoff_hz(&self) -> f32 {
        self.cutoff_hz
    }

    /// Pulse length currently programmed, in samples. For tests and metering.
    pub fn pulse_len(&self) -> usize {
        self.pulse_len
    }

    /// Trigger a strike. `velocity` is the normalised MIDI velocity in
    /// [0, 1]; `freq` is the struck note's fundamental in Hz, which sets
    /// the register-dependent contact duration.
    pub fn fire(&mut self, velocity: f32, freq: f32) {
        let v = velocity.clamp(0.0, 1.0);
        self.velocity = v;
        self.pulse_pos = 0;
        self.pulse_len =
            ((contact_secs(freq) * contact_scale(v) * self.sample_rate) as usize).max(2);
        self.lpf.reset();
        let warp = v.powf(VELOCITY_WARP);
        let cutoff = CUTOFF_MIN_HZ + (CUTOFF_MAX_HZ - CUTOFF_MIN_HZ) * warp;
        self.cutoff_hz = cutoff;
        self.lpf.set_cutoff(cutoff, self.sample_rate);
        self.active = v > 0.0;
    }

    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Emit one excitation sample. Returns 0 once the pulse plus a short
    /// LPF tail have elapsed.
    #[inline]
    pub fn tick(&mut self) -> f32 {
        if !self.active {
            return 0.0;
        }

        // Raised-cosine ("Hann") force profile: 0 → 1 → 0 over `pulse_len`
        // samples. Smooth onset / offset means no audible click.
        let raw = if self.pulse_pos < self.pulse_len {
            let phase = self.pulse_pos as f32 / self.pulse_len as f32;
            0.5 * (1.0 - (TAU * phase).cos())
        } else {
            0.0
        };

        // Amplitude warp: `v^AMPLITUDE_WARP`. See the constant's docs for
        // the trade-off between physical realism and perceived loudness.
        let scaled = raw * self.velocity.powf(AMPLITUDE_WARP);

        // Felt-compression filter.
        let filtered = self.lpf.tick(scaled);

        self.pulse_pos += 1;
        // After ~6× the pulse length the LPF tail is below −60 dB.
        // Disable to save CPU and let the string take over.
        if self.pulse_pos > self.pulse_len * 6 {
            self.active = false;
        }
        filtered
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rms(samples: &[f32]) -> f32 {
        let s: f32 = samples.iter().map(|x| x * x).sum();
        (s / samples.len() as f32).sqrt()
    }

    #[test]
    fn fresh_hammer_is_silent() {
        let mut h = Hammer::new(48_000.0);
        for _ in 0..100 {
            assert_eq!(h.tick(), 0.0);
        }
    }

    #[test]
    fn velocity_zero_produces_silence() {
        let mut h = Hammer::new(48_000.0);
        h.fire(0.0, 261.6);
        for _ in 0..1_000 {
            assert_eq!(h.tick(), 0.0);
        }
    }

    #[test]
    fn fired_hammer_produces_signal_then_settles() {
        let mut h = Hammer::new(48_000.0);
        h.fire(1.0, 261.6);
        let mut buf = vec![0.0; 2_048];
        for v in buf.iter_mut() {
            *v = h.tick();
        }
        // Early window has audible energy.
        let early = rms(&buf[0..256]);
        assert!(early > 0.01, "no energy in early window: {early}");
        // Tail is silent.
        let tail = rms(&buf[1_500..]);
        assert!(tail < 1e-4, "tail not silent: {tail}");
    }

    #[test]
    fn higher_velocity_has_more_total_energy() {
        // AMPLITUDE_WARP = 1.6: v=1 vs v=0.5 is a 0.5^1.6 ≈ 3× amplitude
        // ratio, widened further by the brighter felt LPF at high velocity.
        let mut hard = Hammer::new(48_000.0);
        hard.fire(1.0, 261.6);
        let mut soft = Hammer::new(48_000.0);
        soft.fire(0.5, 261.6);
        let mut buf_h = vec![0.0; 1_024];
        let mut buf_s = vec![0.0; 1_024];
        for v in buf_h.iter_mut() {
            *v = hard.tick();
        }
        for v in buf_s.iter_mut() {
            *v = soft.tick();
        }
        assert!(rms(&buf_h) > rms(&buf_s) * 2.0);
    }

    #[test]
    fn contact_time_tapers_from_bass_to_treble() {
        // A bass hammer presses into its string several times longer than a
        // treble hammer — that taper is what rounds the bass attack and
        // keeps the treble excitation wideband enough to reach its own
        // fundamental.
        let mut h = Hammer::new(48_000.0);
        h.fire(0.5, 27.5); // A0 at mf, where contact_scale = 1.0
        let bass_len = h.pulse_len();
        h.fire(0.5, 4_186.0); // C8 at mf
        let treble_len = h.pulse_len();
        assert!(
            bass_len > treble_len * 4,
            "bass contact should be much longer: bass={bass_len} treble={treble_len}"
        );
        // Sanity: at mf the anchors land where the constants say (±1 sample
        // of float rounding through the velocity scale).
        let bass_expect = (CONTACT_SECS_BASS * 48_000.0) as usize;
        let treble_expect = (CONTACT_SECS_TREBLE * 48_000.0) as usize;
        assert!(
            bass_len.abs_diff(bass_expect) <= 1,
            "bass anchor off: {bass_len}"
        );
        assert!(
            treble_len.abs_diff(treble_expect) <= 1,
            "treble anchor off: {treble_len}"
        );
    }

    #[test]
    fn harder_strike_has_briefer_contact() {
        // Felt hardening: the ff contact must be markedly shorter than the
        // pp contact for the same note (~2× across the full range).
        let mut h = Hammer::new(48_000.0);
        h.fire(0.05, 261.6);
        let soft_len = h.pulse_len();
        h.fire(1.0, 261.6);
        let hard_len = h.pulse_len();
        assert!(
            (hard_len as f32) < soft_len as f32 * 0.65,
            "hard contact should be ~half the soft one: soft={soft_len} hard={hard_len}"
        );
    }

    /// Total transferred impulse: the area under the (filtered) force pulse.
    /// The one-pole LPF has unity DC gain, so summing until the hammer
    /// deactivates recovers the area to within its truncated tail.
    fn impulse(h: &mut Hammer, velocity: f32, freq: f32) -> f32 {
        h.fire(velocity, freq);
        let mut sum = 0.0;
        while h.is_active() {
            sum += h.tick();
        }
        sum
    }

    #[test]
    fn impulse_span_matches_documented_range() {
        // AMPLITUDE_WARP alone would give v=0.1 → v=1.0 a 0.1^-1.6 ≈ 39.8×
        // (32 dB) span. CONTACT_SCALE shortens the ff pulse to 0.7× and
        // stretches the pp one to 1.24×, so the *area* span is only
        // ≈ 22.5× (27 dB). Pin that, so retuning either constant without
        // re-reading the other trips here rather than silently shifting the
        // instrument's dynamic range.
        let mut h = Hammer::new(48_000.0);
        let pp = impulse(&mut h, 0.1, 261.6);
        let ff = impulse(&mut h, 1.0, 261.6);
        let ratio = ff / pp;
        assert!(
            (20.0..25.0).contains(&ratio),
            "impulse span should be ≈22.5× (27 dB), got {ratio:.1}×"
        );
        assert!(
            ratio < 35.0,
            "span reached {ratio:.1}×, i.e. the contact taper is no longer \
             costing level — AMPLITUDE_WARP's docs need updating"
        );
    }

    #[test]
    fn impulse_rises_monotonically_with_velocity() {
        // A harder strike must transfer more momentum despite its briefer
        // contact. If CONTACT_SCALE's spread ever outgrows AMPLITUDE_WARP
        // this inverts, and loud playing would get *weaker* fundamentals.
        let mut h = Hammer::new(48_000.0);
        let mut prev = 0.0;
        for step in 1..=20 {
            let v = step as f32 / 20.0;
            let i = impulse(&mut h, v, 261.6);
            assert!(
                i > prev,
                "impulse must increase with velocity: v={v:.2} gave {i:.5}, previous {prev:.5}"
            );
            prev = i;
        }
    }

    #[test]
    fn cutoff_scales_monotonically_with_velocity() {
        // The hammer pulse alone has too narrow a spectrum (~700 Hz main
        // lobe) for spectral-shape tests to be sensitive to LPF cutoff —
        // there's little HF content to filter. The audible "brightness"
        // shows up once the pulse drives the string and selects which
        // harmonics resonate (validated in the voice-level integration test).
        //
        // Here we just verify the LPF cutoff is programmed correctly.
        let mut h = Hammer::new(48_000.0);

        h.fire(0.0, 261.6);
        assert!((h.cutoff_hz() - CUTOFF_MIN_HZ).abs() < 1.0);

        h.fire(1.0, 261.6);
        assert!((h.cutoff_hz() - CUTOFF_MAX_HZ).abs() < 1.0);

        h.fire(0.25, 261.6);
        let c_quarter = h.cutoff_hz();
        h.fire(0.75, 261.6);
        let c_three_quarter = h.cutoff_hz();
        assert!(
            c_three_quarter > c_quarter * 2.0,
            "expected concave growth: 0.25 → {c_quarter} Hz, 0.75 → {c_three_quarter} Hz"
        );
        // 0.25^1.5 = 0.125; 0.75^1.5 = 0.6495 → ratio ≈ 5.2, well > 2.
    }
}
