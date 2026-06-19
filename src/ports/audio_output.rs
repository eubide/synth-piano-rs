//! Audio output port.
//!
//! Currently very thin — the cpal adapter takes ownership of the engine
//! and drives `Engine::render` from its real-time callback. We model the
//! port as a "started stream handle" so the application code can drop it
//! to stop audio. The handle is intentionally neither `Send` nor `Sync`
//! (see the note below).

// `Send` is intentionally NOT required: `cpal::Stream` is `!Send` on macOS
// (CoreAudio thread affinity), and the adapter only needs to live on the
// main thread until shutdown.
pub trait AudioOutput {
    /// Audio sample rate reported by the device.
    fn sample_rate(&self) -> f32;

    /// Buffer size in frames, when the host honours our request.
    fn buffer_frames(&self) -> Option<u32>;
}
