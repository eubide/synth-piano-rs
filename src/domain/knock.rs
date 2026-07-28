//! Hammer/key knock — the mechanical attack noise of a strike.
//!
//! ## Why
//! A piano attack is not only the string: the hammer shank vibrates, the
//! key bottoms out on the front rail, and both impulses reach the
//! soundboard through the action and the case. Askenfelt & Jansson's
//! attack-transient measurements show this broadband "thump" arriving with
//! (and slightly before) the tonal build-up — dark and heavy in the bass,
//! a light "tick" in the treble, and growing faster than the tone with
//! velocity. Synthesis that omits it reads instantly as "electronic": the
//! ear misses the percussive floor under the note.
//!
//! ## Model
//! A short burst of white noise → exponential decay envelope → two cascaded
//! one-pole lowpasses (−12 dB/oct — a single pole leaves the burst hissy;
//! the steeper slope reads as a structural thud). All three parameters
//! follow the register (through the same log-frequency taper the hammer's
//! contact time uses — [`crate::domain::hammer::register_taper`]):
//! - **Cutoff** rises bass → treble: the bass knock is a dark thud (shank
//!   and keybed resonances sit low), the treble a brighter tick.
//! - **Decay τ** shrinks bass → treble: heavy bass parts ring longer.
//! - **Gain** tapers bass → treble, and scales with `v²` — knock energy
//!   grows faster than tone energy, which is why fortissimo playing gets
//!   percussive.
//!
//! The noise source is a per-voice xorshift32 PRNG (allocation-free,
//! RT-safe). The seed mixes the note frequency with a running strike
//! counter, so repeated strikes of the same note vary — no two knocks are
//! identical, but the sequence is fully deterministic for tests.
//!
//! The knock is summed at the *voice output* (outside the string loop and
//! the damper): it reaches the listener through the soundboard like every
//! other structural noise, is not pitch-filtered by the string, and is not
//! muted by the damper felt.

use crate::domain::filter::OnePoleLowpass;
use crate::domain::hammer::register_taper;

/// Lowpass cutoff of the knock at the register extremes, in Hz.
const KNOCK_CUTOFF_BASS_HZ: f32 = 180.0;
const KNOCK_CUTOFF_TREBLE_HZ: f32 = 1_600.0;

/// Envelope decay time constant at the register extremes, in seconds.
const KNOCK_TAU_BASS_SECS: f32 = 0.02;
const KNOCK_TAU_TREBLE_SECS: f32 = 0.006;

/// Register gain taper: the bass thud is prominent, the treble tick subtle.
const KNOCK_GAIN_BASS: f32 = 1.0;
const KNOCK_GAIN_TREBLE: f32 = 0.45;

/// Overall knock level. Chosen so the ff bass knock sits ≈ 10 % of the
/// early note RMS — clearly audible as a percussive floor without reading
/// as a separate drum hit. A listening-test knob, like `master_gain`.
const KNOCK_GAIN: f32 = 0.35;

/// Velocity exponent. Steeper than the tone's `AMPLITUDE_WARP` (1.6): the
/// knock all but vanishes at pp and dominates the attack floor at ff.
const KNOCK_VELOCITY_WARP: f32 = 2.0;

/// Envelope gate: below this the burst is inaudible; free the state.
const KNOCK_GATE: f32 = 1.0e-3;

#[derive(Debug)]
pub struct HammerKnock {
    sample_rate: f32,
    lpf: OnePoleLowpass,
    lpf2: OnePoleLowpass,
    env: f32,
    env_decay: f32,
    gain: f32,
    /// xorshift32 state; never zero while active.
    rng: u32,
    /// Running strike counter mixed into the seed so consecutive strikes
    /// of the same note produce different (but deterministic) noise.
    strikes: u32,
    active: bool,
}

impl HammerKnock {
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            lpf: OnePoleLowpass::new(),
            lpf2: OnePoleLowpass::new(),
            env: 0.0,
            env_decay: 0.0,
            gain: 0.0,
            rng: 1,
            strikes: 0,
            active: false,
        }
    }

    /// Trigger the knock for a strike at normalised `velocity` on a note
    /// with fundamental `freq`.
    pub fn fire(&mut self, velocity: f32, freq: f32) {
        let v = velocity.clamp(0.0, 1.0);
        self.strikes = self.strikes.wrapping_add(1);
        self.active = v > 0.0;
        if !self.active {
            return;
        }
        let cutoff = register_taper(freq, KNOCK_CUTOFF_BASS_HZ, KNOCK_CUTOFF_TREBLE_HZ);
        let tau = register_taper(freq, KNOCK_TAU_BASS_SECS, KNOCK_TAU_TREBLE_SECS);
        let register_gain = register_taper(freq, KNOCK_GAIN_BASS, KNOCK_GAIN_TREBLE);
        self.gain = KNOCK_GAIN * register_gain * v.powf(KNOCK_VELOCITY_WARP);
        self.env = 1.0;
        self.env_decay = (-1.0 / (tau * self.sample_rate)).exp();
        self.lpf.set_cutoff(cutoff, self.sample_rate);
        self.lpf.reset();
        self.lpf2.set_cutoff(cutoff, self.sample_rate);
        self.lpf2.reset();
        // Golden-ratio hash of the strike counter mixed with the note's
        // frequency bits; `| 1` keeps xorshift out of its zero fixpoint.
        self.rng = (self
            .strikes
            .wrapping_mul(0x9E37_79B9)
            .wrapping_add(freq.to_bits()))
            | 1;
    }

    /// One sample of white noise in [−1, 1).
    #[inline]
    fn next_noise(&mut self) -> f32 {
        let mut x = self.rng;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.rng = x;
        x as f32 * (2.0 / 4_294_967_296.0) - 1.0
    }

    #[inline]
    pub fn tick(&mut self) -> f32 {
        if !self.active {
            return 0.0;
        }
        let n = self.next_noise();
        let y = self.lpf2.tick(self.lpf.tick(n * self.gain * self.env));
        self.env *= self.env_decay;
        if self.env < KNOCK_GATE {
            self.active = false;
        }
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rms(samples: &[f32]) -> f32 {
        let s: f32 = samples.iter().map(|x| x * x).sum();
        (s / samples.len() as f32).sqrt()
    }

    fn render(k: &mut HammerKnock, n: usize) -> Vec<f32> {
        let mut buf = vec![0.0; n];
        for v in buf.iter_mut() {
            *v = k.tick();
        }
        buf
    }

    #[test]
    fn fresh_knock_is_silent() {
        let mut k = HammerKnock::new(48_000.0);
        for _ in 0..256 {
            assert_eq!(k.tick(), 0.0);
        }
    }

    #[test]
    fn velocity_zero_produces_silence() {
        let mut k = HammerKnock::new(48_000.0);
        k.fire(0.0, 65.4);
        for _ in 0..1_024 {
            assert_eq!(k.tick(), 0.0);
        }
    }

    #[test]
    fn fired_knock_produces_burst_then_settles() {
        let mut k = HammerKnock::new(48_000.0);
        k.fire(1.0, 65.4);
        let buf = render(&mut k, 24_000); // 500 ms
        assert!(rms(&buf[..512]) > 1e-3, "no burst energy");
        // Bass τ = 20 ms; the gate trips at env < 1e-3 (≈ 140 ms). The tail
        // window must be fully silent.
        assert!(rms(&buf[12_000..]) == 0.0, "knock did not gate off");
    }

    #[test]
    fn harder_strike_is_superlinearly_louder() {
        // v² warp: doubling velocity quadruples the amplitude.
        fn burst_rms(v: f32) -> f32 {
            let mut k = HammerKnock::new(48_000.0);
            k.fire(v, 220.0);
            rms(&render(&mut k, 2_048))
        }
        let hard = burst_rms(1.0);
        let soft = burst_rms(0.5);
        assert!(
            hard > soft * 3.0,
            "knock should grow ~v²: hard={hard} soft={soft}"
        );
    }

    #[test]
    fn bass_knock_is_darker_than_treble_knock() {
        // Compare the fraction of energy above 1 kHz: the bass thud (180 Hz
        // cutoff) must be much darker than the treble tick (1.6 kHz cutoff).
        fn hf_fraction(freq: f32) -> f32 {
            let mut k = HammerKnock::new(48_000.0);
            k.fire(1.0, freq);
            let buf = render(&mut k, 4_096);
            let mut lp = OnePoleLowpass::new();
            lp.set_cutoff(1_000.0, 48_000.0);
            let mut hp_sq = 0.0;
            let mut total_sq = 0.0;
            for &x in &buf {
                let hp = x - lp.tick(x);
                hp_sq += hp * hp;
                total_sq += x * x;
            }
            hp_sq / total_sq.max(1e-12)
        }
        let bass = hf_fraction(27.5);
        let treble = hf_fraction(4_186.0);
        assert!(
            treble > bass * 3.0,
            "treble knock should carry far more HF: bass={bass} treble={treble}"
        );
    }

    #[test]
    fn consecutive_strikes_vary_but_deterministically() {
        // Two strikes of the same note must not produce the identical
        // waveform (the strike counter re-seeds the noise) — but the whole
        // sequence must replay exactly on a fresh instance.
        fn two_strikes() -> (Vec<f32>, Vec<f32>) {
            let mut k = HammerKnock::new(48_000.0);
            k.fire(0.8, 220.0);
            let a = render(&mut k, 1_024);
            k.fire(0.8, 220.0);
            let b = render(&mut k, 1_024);
            (a, b)
        }
        let (a1, b1) = two_strikes();
        let (a2, b2) = two_strikes();
        assert_ne!(a1, b1, "strikes should vary");
        assert_eq!(a1, a2, "sequence should be deterministic");
        assert_eq!(b1, b2, "sequence should be deterministic");
    }
}
