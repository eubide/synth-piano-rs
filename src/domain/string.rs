//! Single Karplus-Strong string with a dispersion-allpass cascade in the loop.
//!
//! ## Algorithm (Extended Karplus-Strong, Jaffe & Smith 1983 + dispersion)
//!
//! On every audio sample:
//!   `read    = delay.read_int(N_int)`
//!   `tuned   = tuning.tick(read)`        ← lossless fractional delay
//!   `lpf_out = lpf.tick(tuned)`
//!   `disp    = dispersion.tick(lpf_out)`
//!   `y       = disp + excitation`        ← hammer enters here, undispersed
//!   `delay.write(y)`
//!   `output  = y`
//!
//! ## Why an allpass tuner, not linear interpolation
//! Linear interpolation of the fractional delay is a 2-tap FIR whose loss
//! peaks at `frac = 0.5` — harmless for a single traversal, but inside a
//! resonant loop it compounds `f₀` times per second. At C7 it alone kills
//! the fundamental in ≈ 0.35 s, faster than the loop filter. A first-order
//! allpass has unity magnitude at every frequency, so tuning costs no
//! decay time anywhere on the keyboard (Jaffe & Smith 1983, §"Tuning").
//!
//! ## Tuning compensation
//! The loop has three delay-affecting components: the integer + fractional
//! delay line, the two-tap LPF (phase delay = its smoothing weight `s`),
//! and the dispersion cascade (group delay varies with frequency). To make
//! the fundamental land on the requested `f₀`, we set:
//!   `D + s + N·τ_g(ω₀) = Fs / f₀`
//! and recover `D` from there. The cascade group delay is computed
//! analytically at `ω₀` (not at DC) — important for high notes where the
//! difference matters.
//!
//! ## Where the dispersion lives
//! After the loop LPF, before the excitation injection. The excitation is
//! kept undispersed for the same reason it's kept un-LPF'd in Phase 3: the
//! player hears the strike's natural transient on the first pass, and only
//! subsequent loop circulations get filtered + dispersed.

use std::f32::consts::TAU;

use crate::domain::delay_line::DelayLine;
use crate::domain::dispersion::DispersionCascade;
use crate::domain::filter::{AllpassFirstOrder, TwoTapLowpass};

/// Floor on the loop filter's own T60 at the fundamental, in seconds.
///
/// The loop filter's job is *spectral* shaping (high partials die first);
/// the overall envelope is owned by the per-note loop gain the voice
/// installs. But at the classic `s = 0.5` the filter alone kills a C7
/// fundamental in ≈ 0.35 s — far shorter than any plausible treble
/// envelope, leaving the top octaves attack-only. (This went unnoticed for
/// a while because the unipolar hammer pulse parked a DC pedestal in the
/// loop, which the filter passes losslessly; once the strike comb removed
/// the DC, the real decay surfaced.) Above ≈ G5 we shrink the smoothing so
/// the filter's own fundamental T60 never drops below this floor. Per-pass
/// loss at the fundamental is ≈ `s(1−s)(1−cos ω₀)` nepers, hence
/// `T60 = ln(10³) / (f₀ · s(1−s) · (1−cos ω₀))`.
///
/// 4 s (down from an initial 6 s) keeps the top octaves ringing audibly
/// while letting their upper partials die noticeably faster — at 6 s the
/// sustained inharmonic highs read as "metallic" / bell-like.
const LOOP_FILTER_T60_FLOOR_SECS: f32 = 4.0;

/// Smoothing weight for a note at `freq`: the largest `s ≤ 0.5` whose
/// fundamental T60 stays at or above [`LOOP_FILTER_T60_FLOOR_SECS`].
/// Saturates at 0.5 (no change) below ≈ 800 Hz.
fn loop_filter_smoothing(freq: f32, sample_rate: f32) -> f32 {
    let omega_0 = TAU * freq / sample_rate;
    // p = s(1−s), capped at its s = 0.5 maximum of 0.25. The division is
    // safe: callers pluck at freq > 0, and a vanishing (1−cos ω₀) just
    // sends the budget to +inf, where `min` saturates.
    let p = (6.907_755_3 / (LOOP_FILTER_T60_FLOOR_SECS * freq * (1.0 - omega_0.cos()))).min(0.25);
    0.5 - (0.25 - p).sqrt()
}

#[derive(Debug)]
pub struct KarplusStrong {
    sample_rate: f32,
    delay: DelayLine,
    /// Lossless fractional-delay tuner (see module docs).
    tuning: AllpassFirstOrder,
    lpf: TwoTapLowpass,
    dispersion: DispersionCascade,
    delay_int: usize,
    active: bool,
    /// Multiplicative loss applied to the loop output before it is written
    /// back into the delay line. `1.0` = no extra loss (the LPF still
    /// damps HF), values below 1 model felt damping that engages with the
    /// string mid-vibration. Used by the sympathetic bank to simulate
    /// damper-on / damper-off behaviour.
    loop_gain: f32,
}

impl KarplusStrong {
    pub fn new(sample_rate: f32, max_delay: usize) -> Self {
        Self {
            sample_rate,
            delay: DelayLine::new(max_delay),
            tuning: AllpassFirstOrder::new(),
            lpf: TwoTapLowpass::new(),
            dispersion: DispersionCascade::new(),
            delay_int: 1,
            active: false,
            loop_gain: 1.0,
        }
    }

    /// Set per-cycle multiplicative loop loss. `1.0` = transparent (the
    /// default); values like `0.985` damp the entire spectrum by ~1.5 %
    /// per loop cycle. Frequency-independent — for piano-correct damping
    /// the LPF inside the loop already shapes the loss by frequency.
    pub fn set_loop_gain(&mut self, gain: f32) {
        self.loop_gain = gain.clamp(0.0, 1.0);
    }

    pub fn loop_gain(&self) -> f32 {
        self.loop_gain
    }

    pub fn is_active(&self) -> bool {
        self.active
    }

    pub fn deactivate(&mut self) {
        self.active = false;
    }

    /// Arm the string at a given fundamental frequency. Clears delay,
    /// loop filter and dispersion state, picks a per-note dispersion
    /// coefficient, and tunes the delay length so `f₀` still lands on
    /// target despite the cascade's contribution.
    pub fn pluck(&mut self, freq: f32) {
        // 1. Pick dispersion strength and loop-filter smoothing for this note.
        self.dispersion.fit_to_frequency(freq, self.sample_rate);
        let s = loop_filter_smoothing(freq, self.sample_rate);
        self.lpf.set_smoothing(s);

        // 2. Tune the delay line. Subtract the LPF phase delay (= s) and the
        //    cascade's group delay *at the fundamental*. Using ω₀ rather
        //    than DC keeps high notes in tune.
        let omega_0 = TAU * freq / self.sample_rate;
        let cascade_delay = self.dispersion.group_delay_at(omega_0);
        let total = self.sample_rate / freq - s - cascade_delay;

        // 3. Clamp to the buffer-imposed window. Notes that need more than
        //    the buffer holds get clipped at the bottom; notes whose total
        //    would go below 2 samples (extreme treble) get clipped at the
        //    top — pitch will be slightly off but the loop stays stable.
        let max_total = (self.delay.capacity() - 2) as f32;
        let total = total.clamp(2.0, max_total);

        // 4. Split into integer delay + tuning allpass. Keeping the
        //    fractional part in [0.3, 1.3) keeps the allpass coefficient
        //    well inside the unit circle (a ∈ (−0.13, 0.54]). The DC
        //    phase-delay formula `a = (1−frac)/(1+frac)` is exact in the
        //    low-frequency limit; the treble-end error is under a cent.
        // `total` is clamped to ≥ 2.0 above, so `total − 0.3 ≥ 1.7` and the
        // floor is always ≥ 1 — no separate lower guard on `d_int` is needed.
        let d_int = (total - 0.3).floor();
        let frac = total - d_int;
        self.delay_int = d_int as usize;
        self.tuning.set_coefficient((1.0 - frac) / (1.0 + frac));

        // 5. Wipe state. Energy enters only via tick()'s `excitation`.
        self.delay.clear();
        self.tuning.reset();
        self.lpf.reset();
        self.dispersion.reset();
        self.active = true;
    }

    #[inline]
    pub fn tick(&mut self, excitation: f32) -> f32 {
        if !self.active {
            return 0.0;
        }
        let read = self.delay.read_int(self.delay_int);
        let tuned = self.tuning.tick(read);
        let lpf_out = self.lpf.tick(tuned);
        let disp = self.dispersion.tick(lpf_out);
        let y = disp + excitation;
        // Apply loop loss to the *recirculated* portion only. The voice
        // hears `y` (with the excitation intact), while the delay line
        // stores `y · loop_gain` so subsequent passes lose energy. With
        // loop_gain = 1.0 this is a no-op multiply.
        self.delay.write(y * self.loop_gain);
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detect_period_samples(samples: &[f32], min_lag: usize, max_lag: usize) -> usize {
        let mut best_lag = min_lag;
        let mut best_score = f32::NEG_INFINITY;
        for lag in min_lag..=max_lag {
            let mut sum = 0.0f32;
            let n = samples.len() - lag;
            for i in 0..n {
                sum += samples[i + lag] * samples[i];
            }
            let score = sum / n as f32;
            if score > best_score {
                best_score = score;
                best_lag = lag;
            }
        }
        best_lag
    }

    fn pitch_from_period(period_samples: usize, sample_rate: f32) -> f32 {
        sample_rate / period_samples as f32
    }

    fn render_impulse_response(s: &mut KarplusStrong, n: usize) -> Vec<f32> {
        let mut buf = vec![0.0; n];
        buf[0] = s.tick(1.0);
        for v in buf.iter_mut().skip(1) {
            *v = s.tick(0.0);
        }
        buf
    }

    #[test]
    fn fresh_string_is_silent() {
        let mut s = KarplusStrong::new(48_000.0, 4096);
        for _ in 0..100 {
            assert_eq!(s.tick(0.0), 0.0);
        }
    }

    #[test]
    fn unplucked_string_ignores_excitation() {
        let mut s = KarplusStrong::new(48_000.0, 4096);
        for _ in 0..16 {
            assert_eq!(s.tick(1.0), 0.0);
        }
    }

    #[test]
    fn impulse_response_a4_has_period_near_109_samples() {
        // With dispersion the fundamental can sit a sample or two off from
        // a no-dispersion baseline — widen the autocorrelation search and
        // the tolerance slightly to account for that.
        let mut s = KarplusStrong::new(48_000.0, 4096);
        s.pluck(440.0);
        let buf = render_impulse_response(&mut s, 8_192);
        let lag = detect_period_samples(&buf[400..], 80, 140);
        let f = pitch_from_period(lag, 48_000.0);
        assert!(
            (f - 440.0).abs() / 440.0 < 0.03,
            "got {f} Hz at lag {lag}, expected ~440"
        );
    }

    #[test]
    fn impulse_response_c4_has_period_near_183_samples() {
        let mut s = KarplusStrong::new(48_000.0, 4096);
        s.pluck(261.625_56);
        let buf = render_impulse_response(&mut s, 8_192);
        let lag = detect_period_samples(&buf[400..], 140, 220);
        let f = pitch_from_period(lag, 48_000.0);
        assert!(
            (f - 261.625_56).abs() / 261.625_56 < 0.03,
            "got {f} Hz at lag {lag}, expected ~261.63"
        );
    }

    #[test]
    fn loop_decays_over_time_due_to_lpf() {
        let mut s = KarplusStrong::new(48_000.0, 4096);
        s.pluck(440.0);
        let buf = render_impulse_response(&mut s, 48_000);
        let early_rms: f32 = buf[200..1_224].iter().map(|x| x * x).sum::<f32>().sqrt();
        let late_rms: f32 = buf[buf.len() - 1_024..]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        assert!(
            late_rms < early_rms,
            "expected decay: early={early_rms} late={late_rms}"
        );
    }

    #[test]
    fn extreme_frequency_does_not_panic() {
        let mut s = KarplusStrong::new(48_000.0, 4096);
        s.pluck(0.5);
        for _ in 0..1_000 {
            s.tick(0.0);
        }
        s.pluck(40_000.0);
        for _ in 0..1_000 {
            s.tick(0.0);
        }
    }

    #[test]
    fn loop_stays_bounded_over_a_second() {
        // Smoke test: with dispersion + LPF + delay all in the loop, the
        // total gain must be ≤ 1 at every frequency. Any latent instability
        // shows up as exponentially growing samples within a few seconds.
        let mut s = KarplusStrong::new(48_000.0, 4096);
        s.pluck(220.0);
        let mut peak = 0.0f32;
        let mut sample = s.tick(1.0);
        peak = peak.max(sample.abs());
        for _ in 1..48_000 {
            sample = s.tick(0.0);
            peak = peak.max(sample.abs());
        }
        // Initial impulse is 1.0; after passing through the loop a few
        // times the peak should be ≤ 1.0 (LPF removes energy each pass).
        assert!(peak <= 1.0 + 1e-3, "loop blew up, peak={peak}");
    }

    #[test]
    fn lower_loop_gain_speeds_up_decay() {
        // Two strings, same pluck, different loop_gain. The one with
        // smaller loop_gain must have lower RMS in the long-term tail.
        fn render_with_loop_gain(g: f32) -> f32 {
            let mut s = KarplusStrong::new(48_000.0, 4096);
            s.pluck(440.0);
            s.set_loop_gain(g);
            let mut buf = vec![0.0; 24_000];
            buf[0] = s.tick(1.0);
            for v in buf.iter_mut().skip(1) {
                *v = s.tick(0.0);
            }
            let tail = &buf[buf.len() - 2_048..];
            let energy: f32 = tail.iter().map(|x| x * x).sum();
            (energy / tail.len() as f32).sqrt()
        }
        let rms_full = render_with_loop_gain(1.0);
        let rms_damped = render_with_loop_gain(0.985);
        assert!(
            rms_damped < rms_full * 0.5,
            "expected loop_gain 0.985 to damp faster: full={rms_full} damped={rms_damped}"
        );
    }

    #[test]
    fn high_note_stays_stable_with_tapered_dispersion() {
        // C8 — its short loop can only hold reduced dispersion, so |a| tapers
        // down (but stays non-zero, keeping the treble inharmonic). The KS
        // loop must still produce a stable, audible, finite signal.
        let mut s = KarplusStrong::new(48_000.0, 4096);
        s.pluck(4_186.0);
        let buf = render_impulse_response(&mut s, 2_048);
        let energy: f32 = buf.iter().map(|x| x * x).sum();
        assert!(energy > 0.0, "no signal at C8");
        assert!(energy.is_finite(), "C8 loop diverged");
    }

    #[test]
    fn bass_partials_are_stretched_not_harmonic() {
        // The bass must be inharmonic: a perfectly harmonic bass sounds
        // synthetic/organ-like. Measure the 12th partial of C2 — it should
        // sit clearly above 12·f0 (real grands stretch it tens of cents).
        // Goertzel magnitude at f over the buffer.
        fn goertzel(buf: &[f32], f: f32, sr: f32) -> f32 {
            let w = std::f32::consts::TAU * f / sr;
            let coeff = 2.0 * w.cos();
            let (mut q1, mut q2) = (0.0f32, 0.0f32);
            for &x in buf {
                let q0 = coeff * q1 - q2 + x;
                q2 = q1;
                q1 = q0;
            }
            (q1 * q1 + q2 * q2 - coeff * q1 * q2).sqrt()
        }
        let sr = 48_000.0;
        let f0 = 65.406; // C2
        let mut s = KarplusStrong::new(sr, 4096);
        s.pluck(f0);
        let buf = render_impulse_response(&mut s, 32_768);
        // Scan ±4% around the 12th harmonic for the actual partial peak.
        let center = 12.0 * f0;
        let (mut best_f, mut best_m) = (center, 0.0f32);
        for i in 0..=240 {
            let f = center * (0.97 + 0.06 * i as f32 / 240.0);
            let m = goertzel(&buf, f, sr);
            if m > best_m {
                best_m = m;
                best_f = f;
            }
        }
        let cents = 1200.0 * (best_f / center).log2();
        assert!(
            cents > 5.0,
            "C2 12th partial should be stretched sharp (inharmonic), got {cents:.1} cents"
        );
    }
}
