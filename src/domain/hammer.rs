//! Velocity-dependent hammer excitation.
//!
//! ## Model
//! A raised-cosine pulse (1.5 ms) shaped by a one-pole lowpass whose cutoff
//! frequency rises with velocity. The pulse is the *force profile* of the
//! hammer striking the string; the LPF approximates the *felt compression*
//! that smooths that force on a soft hit and sharpens it on a hard hit.
//!
//! ## Why velocity → cutoff
//! Real piano felt is non-linear:
//! - Soft strike: felt deforms gradually, contact lasts longer, force pulse
//!   is "rounder", spectrum is biased to low frequencies → dull tone.
//! - Hard strike: felt compresses, contact is brief and sharp, force pulse
//!   contains higher frequencies → bright tone.
//! Mapping that to a single first-order LPF cutoff `f_c(v)` is a deliberate
//! simplification (Bank & Sujbert 2003 use a full non-linear contact model)
//! but it captures the perceptual dimension we care about with very little
//! state.
//!
//! ## Mapping
//! `f_c(v) = lerp(800 Hz, 6000 Hz, v^0.7)` for `v ∈ [0, 1]`.
//! `^0.7` flattens the curve at high velocity, mirroring real felt that
//! "saturates" once heavily compressed.

use std::f32::consts::TAU;

use crate::domain::filter::OnePoleLowpass;

/// Hammer-string contact duration in seconds. Real grand-piano contact time
/// ranges 1–4 ms. We pick a midpoint and keep it velocity-independent — the
/// LPF supplies the velocity-dependent spectrum tilt.
const PULSE_SECS: f32 = 0.0015;

/// Cutoff bounds for the felt-compression LPF, in Hz.
///
/// Why these numbers: the raised-cosine pulse has its main spectral lobe up
/// to ~1/PULSE_SECS ≈ 670 Hz. The LPF only contributes audible filtering
/// when the cutoff sits *below* that. We pick:
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
            pulse_len: (PULSE_SECS * sample_rate) as usize,
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

    /// Trigger a strike. `velocity` is the normalised MIDI velocity in [0, 1].
    pub fn fire(&mut self, velocity: f32) {
        let v = velocity.clamp(0.0, 1.0);
        self.velocity = v;
        self.pulse_pos = 0;
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
        h.fire(0.0);
        for _ in 0..1_000 {
            assert_eq!(h.tick(), 0.0);
        }
    }

    #[test]
    fn fired_hammer_produces_signal_then_settles() {
        let mut h = Hammer::new(48_000.0);
        h.fire(1.0);
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
        hard.fire(1.0);
        let mut soft = Hammer::new(48_000.0);
        soft.fire(0.5);
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
    fn cutoff_scales_monotonically_with_velocity() {
        // The hammer pulse alone has too narrow a spectrum (~700 Hz main
        // lobe) for spectral-shape tests to be sensitive to LPF cutoff —
        // there's little HF content to filter. The audible "brightness"
        // shows up once the pulse drives the string and selects which
        // harmonics resonate (validated in the voice-level integration test).
        //
        // Here we just verify the LPF cutoff is programmed correctly.
        let mut h = Hammer::new(48_000.0);

        h.fire(0.0);
        assert!((h.cutoff_hz() - CUTOFF_MIN_HZ).abs() < 1.0);

        h.fire(1.0);
        assert!((h.cutoff_hz() - CUTOFF_MAX_HZ).abs() < 1.0);

        h.fire(0.25);
        let c_quarter = h.cutoff_hz();
        h.fire(0.75);
        let c_three_quarter = h.cutoff_hz();
        assert!(
            c_three_quarter > c_quarter * 2.0,
            "expected concave growth: 0.25 → {c_quarter} Hz, 0.75 → {c_three_quarter} Hz"
        );
        // 0.25^1.5 = 0.125; 0.75^1.5 = 0.6495 → ratio ≈ 5.2, well > 2.
    }
}
