//! Single Karplus-Strong string with a dispersion-allpass cascade in the loop.
//!
//! ## Algorithm (Extended Karplus-Strong, Jaffe & Smith 1983 + dispersion)
//!
//! On every audio sample:
//!   `read    = delay.read_frac(N_int, frac)`
//!   `lpf_out = lpf.tick(read)`
//!   `disp    = dispersion.tick(lpf_out)`
//!   `y       = disp + excitation`        ← hammer enters here, undispersed
//!   `delay.write(y)`
//!   `output  = y`
//!
//! ## Tuning compensation
//! The loop has three delay-affecting components: the integer + fractional
//! delay line, the two-tap LPF (constant ½-sample phase delay), and the
//! dispersion cascade (group delay varies with frequency). To make the
//! fundamental land on the requested `f₀`, we set:
//!   `D + 0.5 + N·τ_g(ω₀) = Fs / f₀`
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
use crate::domain::filter::TwoTapLowpass;

#[derive(Debug)]
pub struct KarplusStrong {
    sample_rate: f32,
    delay: DelayLine,
    lpf: TwoTapLowpass,
    dispersion: DispersionCascade,
    delay_int: usize,
    delay_frac: f32,
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
            lpf: TwoTapLowpass::new(),
            dispersion: DispersionCascade::new(),
            delay_int: 1,
            delay_frac: 0.0,
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
        // 1. Pick dispersion strength for this note.
        self.dispersion.fit_to_frequency(freq, self.sample_rate);

        // 2. Tune the delay line. Subtract LPF phase delay (½) and the
        //    cascade's group delay *at the fundamental*. Using ω₀ rather
        //    than DC keeps high notes in tune.
        let omega_0 = TAU * freq / self.sample_rate;
        let cascade_delay = self.dispersion.group_delay_at(omega_0);
        let total = self.sample_rate / freq - 0.5 - cascade_delay;

        // 3. Clamp to the buffer-imposed window. Notes that need more than
        //    the buffer holds get clipped at the bottom; notes whose total
        //    would go below 2 samples (extreme treble) get clipped at the
        //    top — pitch will be slightly off but the loop stays stable.
        let max_total = (self.delay.capacity() - 2) as f32;
        let total = total.clamp(2.0, max_total);
        self.delay_int = total.floor() as usize;
        self.delay_frac = total - self.delay_int as f32;

        // 4. Wipe state. Energy enters only via tick()'s `excitation`.
        self.delay.clear();
        self.lpf.reset();
        self.dispersion.reset();
        self.active = true;
    }

    #[inline]
    pub fn tick(&mut self, excitation: f32) -> f32 {
        if !self.active {
            return 0.0;
        }
        let read = self.delay.read_frac(self.delay_int, self.delay_frac);
        let lpf_out = self.lpf.tick(read);
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
    fn high_note_falls_back_to_zero_dispersion() {
        // C8 — its short loop can't accommodate any dispersion, so the
        // cascade collapses to N pure sample delays. The KS algorithm
        // should still produce a stable, audible signal.
        let mut s = KarplusStrong::new(48_000.0, 4096);
        s.pluck(4_186.0);
        let buf = render_impulse_response(&mut s, 2_048);
        let energy: f32 = buf.iter().map(|x| x * x).sum();
        assert!(energy > 0.0, "no signal at C8");
        assert!(energy.is_finite(), "C8 loop diverged");
    }
}
