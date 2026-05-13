//! Single Karplus-Strong string driven by an external excitation signal.
//!
//! ## Algorithm (Extended Karplus-Strong, Jaffe & Smith 1983)
//!
//! On every audio sample:
//!   `read    = delay.read_frac(N_int, frac)`
//!   `lpf_out = lpf.tick(read)`
//!   `y       = lpf_out + excitation`     ← hammer / pluck enters here
//!   `delay.write(y)`
//!   `output  = y`
//!
//! ## Why the excitation is added *after* the loop filter
//! Jaffe & Smith's formulation keeps the excitation outside the loop's
//! feedback path on its first pass: the player hears the unfiltered strike
//! immediately, and subsequent circulations get filtered like any other
//! energy. Adding it before the LPF instead would double-filter the strike
//! and dull the transient.
//!
//! ## Tuning compensation
//! The two-tap loop filter has exactly ½-sample phase delay, so we tune by
//! splitting `total_delay = Fs/f` as `N_int + frac + 0.5`. The integer part
//! is realised by the ring-buffer index; the fraction by linear interp in
//! [`DelayLine::read_frac`].

use crate::domain::delay_line::DelayLine;
use crate::domain::filter::TwoTapLowpass;

#[derive(Debug)]
pub struct KarplusStrong {
    sample_rate: f32,
    delay: DelayLine,
    lpf: TwoTapLowpass,
    /// Integer part of the loop delay, in samples.
    delay_int: usize,
    /// Fractional remainder, [0, 1).
    delay_frac: f32,
    active: bool,
}

impl KarplusStrong {
    /// `max_delay` is the longest delay the line must support, in samples.
    /// At 48 kHz, A0 (27.5 Hz) needs ≈ 1745 samples — so a `max_delay` of
    /// 2400 (corresponding to 20 Hz minimum) covers the full piano range
    /// with slack for future detuning.
    pub fn new(sample_rate: f32, max_delay: usize) -> Self {
        Self {
            sample_rate,
            delay: DelayLine::new(max_delay),
            lpf: TwoTapLowpass::new(),
            delay_int: 1,
            delay_frac: 0.0,
            active: false,
        }
    }

    pub fn is_active(&self) -> bool {
        self.active
    }

    pub fn deactivate(&mut self) {
        self.active = false;
    }

    /// Arm the string at a given fundamental frequency. Clears the delay
    /// line and the loop-filter state; does NOT inject any energy. Energy
    /// is supplied per-sample via [`Self::tick`]'s `excitation` argument.
    pub fn pluck(&mut self, freq: f32) {
        let total = (self.sample_rate / freq) - 0.5;
        // Clamp to a safe range: at least 2 samples (read taps at d_int and
        // d_int+1 must both lie inside the buffer), at most capacity-2 to
        // keep the second tap in bounds.
        let max_total = (self.delay.capacity() - 2) as f32;
        let total = total.clamp(2.0, max_total);
        self.delay_int = total.floor() as usize;
        self.delay_frac = total - self.delay_int as f32;

        self.delay.clear();
        self.lpf.reset();
        self.active = true;
    }

    /// Render one sample. `excitation` is added to the loop output and the
    /// new value written back into the delay line. Pass `0.0` when no
    /// external force is acting.
    #[inline]
    pub fn tick(&mut self, excitation: f32) -> f32 {
        if !self.active {
            return 0.0;
        }
        let read = self.delay.read_frac(self.delay_int, self.delay_frac);
        let lpf_out = self.lpf.tick(read);
        let y = lpf_out + excitation;
        self.delay.write(y);
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Autocorrelation-based period detection. Returns the integer lag at
    /// which the (biased) autocorrelation is maximised, restricted to
    /// `[min_lag, max_lag]`.
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

    /// Test helper: render a string given a single impulse of unit amplitude
    /// on the first tick, then zeros. Used to characterise loop behaviour.
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
        // The string is inert until `pluck()`; excitation should be discarded.
        let mut s = KarplusStrong::new(48_000.0, 4096);
        for _ in 0..16 {
            assert_eq!(s.tick(1.0), 0.0);
        }
    }

    #[test]
    fn impulse_response_a4_has_period_near_109_samples() {
        let mut s = KarplusStrong::new(48_000.0, 4096);
        s.pluck(440.0);
        let buf = render_impulse_response(&mut s, 8_192);
        let lag = detect_period_samples(&buf[200..], 90, 130);
        let f = pitch_from_period(lag, 48_000.0);
        assert!(
            (f - 440.0).abs() / 440.0 < 0.02,
            "got {f} Hz at lag {lag}, expected ~440"
        );
    }

    #[test]
    fn impulse_response_c4_has_period_near_183_samples() {
        let mut s = KarplusStrong::new(48_000.0, 4096);
        s.pluck(261.625_56);
        let buf = render_impulse_response(&mut s, 8_192);
        let lag = detect_period_samples(&buf[200..], 150, 220);
        let f = pitch_from_period(lag, 48_000.0);
        assert!(
            (f - 261.625_56).abs() / 261.625_56 < 0.02,
            "got {f} Hz at lag {lag}, expected ~261.63"
        );
    }

    #[test]
    fn loop_decays_over_time_due_to_lpf() {
        let mut s = KarplusStrong::new(48_000.0, 4096);
        s.pluck(440.0);
        let buf = render_impulse_response(&mut s, 48_000);
        // Compare RMS shortly after the impulse vs near the end.
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
        s.pluck(0.5); // below buffer-imposed minimum — clamps
        for _ in 0..1_000 {
            s.tick(0.0);
        }
        s.pluck(40_000.0); // above Nyquist — clamps
        for _ in 0..1_000 {
            s.tick(0.0);
        }
    }
}
