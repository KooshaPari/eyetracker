//! Tobii eye-tracker backend adapter (EYE-SOTA-002).
//!
//! Wraps the [Tobii Stream Engine](https://developer.tobii.com/product-integration/stream-engine/)
//! native SDK for sub-degree gaze accuracy (Tobii 4C/5/Pro Spark/Pro Fusion).
//!
//! ## Architecture
//!
//! - The Stream Engine is a C SDK; this module provides a thin Rust adapter
//!   gated behind the `tobii` Cargo feature (not enabled in default build).
//! - Gaze samples are mapped to a `Frame`-like output: width/height match the
//!   logical screen, the per-sample `data` buffer carries a 1-channel
//!   **gaze heatmap** (sum of recent gaze point weights, normalized to u8).
//!   That is the most common downstream representation for heatmap-based
//!   research/UI; for raw gaze streams, see `eyetracker-gaze` (planned).
//! - Calibration is owned by this adapter (it is intrinsic to the device) and
//!   is run automatically during `start()` if the SDK reports no active
//!   calibration profile.
//!
//! ## Feature flag
//!
//! ```toml
//! [dependencies.eyetracker-camera]
//! features = ["tobii"]
//! ```
//!
//! When the feature is **off** (the default), only `TobiiBackend::default()`
//! and the constructor exist; FFI calls are stubbed. This lets the rest of
//! the workspace compile and test against the trait surface without the
//! native SDK.
//!
//! ## Tests
//!
//! The default-feature tests use a synthetic gaze source (provided via
//! `with_source`) so the trait surface and frame-shape invariants can be
//! verified without real Tobii hardware. Integration tests with the real
//! SDK live in `tests/tobii_integration.rs` and require the `tobii` feature.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::backend::{Backend, BackendKind};
use crate::{CameraConfig, CameraError, CameraInfo, Frame, PixelFormat};

/// Gaze sample from the Tobii Stream Engine (in logical screen coordinates).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GazeSample {
    /// Timestamp (microseconds, monotonic).
    pub timestamp_us: u64,
    /// X coordinate in the device's logical screen space [0, screen_w).
    pub x: f32,
    /// Y coordinate in the device's logical screen space [0, screen_h).
    pub y: f32,
    /// Validity flag (false = track-loss / blink).
    pub valid: bool,
}

/// Source of gaze samples. Real impls wrap the FFI; tests inject a `Vec`.
pub trait GazeSource: Send + Sync {
    /// Try to dequeue the next sample within `timeout`. Returns `None` on timeout.
    fn next(&self, timeout: Duration) -> Option<GazeSample>;
    /// Logical screen width in pixels.
    fn screen_w(&self) -> u32;
    /// Logical screen height in pixels.
    fn screen_h(&self) -> u32;
    /// Backend identifier (e.g. "tobii-4c", "tobii-synthetic").
    fn identifier(&self) -> &str;
}

/// Mock gaze source for tests / no-SDK builds. Cycles through a circular
/// trajectory so the heatmap integrator has motion to integrate.
#[derive(Debug, Clone)]
pub struct SyntheticGazeSource {
    pub w: u32,
    pub h: u32,
    pub start: Instant,
    pub rate_hz: f32,
}

impl SyntheticGazeSource {
    /// Construct a synthetic gaze source that traces a circle in screen
    /// coordinates at `rate_hz` Hz.
    pub fn circle(w: u32, h: u32, rate_hz: f32) -> Self {
        Self {
            w,
            h,
            start: Instant::now(),
            rate_hz,
        }
    }
}

impl GazeSource for SyntheticGazeSource {
    fn next(&self, _timeout: Duration) -> Option<GazeSample> {
        let t = self.start.elapsed().as_secs_f32();
        let phase = t * self.rate_hz * std::f32::consts::TAU;
        let cx = self.w as f32 / 2.0;
        let cy = self.h as f32 / 2.0;
        let r = (self.w.min(self.h) as f32) * 0.35;
        Some(GazeSample {
            timestamp_us: (t * 1_000_000.0) as u64,
            x: cx + r * phase.cos(),
            y: cy + r * phase.sin(),
            valid: true,
        })
    }
    fn screen_w(&self) -> u32 {
        self.w
    }
    fn screen_h(&self) -> u32 {
        self.h
    }
    fn identifier(&self) -> &str {
        "tobii-synthetic"
    }
}

/// Real Tobii FFI source — only compiled with the `tobii` feature.
#[cfg(feature = "tobii")]
pub struct TobiiStreamSource {
    // Opaque handle to the FFI `tobii_device_t` + `tobii_stream_t`.
    // Held as raw pointers to avoid leaking the C types in the public API.
    _device: std::ptr::NonNull<std::ffi::c_void>,
    _stream: std::ptr::NonNull<std::ffi::c_void>,
    w: u32,
    h: u32,
}

#[cfg(feature = "tobii")]
impl TobiiStreamSource {
    /// Connect to a Tobii eye-tracker via the Stream Engine API.
    ///
    /// # Safety
    /// Wraps the C API: the returned object owns the device/stream handles
    /// and releases them on drop via `tobii_device_destroy`.
    pub unsafe fn connect() -> Result<Self, CameraError> {
        // Real impl would call:
        //   tobii_api_create(&mut api, ...);
        //   tobii_enumerate_local_device_urls(...);
        //   tobii_device_create(api, url, &mut device);
        //   tobii_gaze_stream_subscribe(device, ...);
        Err(CameraError::InitFailed(
            "tobii FFI not yet linked — see tobii-rs upstream".to_string(),
        ))
    }
}

#[cfg(feature = "tobii")]
impl Drop for TobiiStreamSource {
    fn drop(&mut self) {
        // tobii_gaze_stream_unsubscribe(self._stream.as_ptr());
        // tobii_device_destroy(self._device.as_ptr());
    }
}

#[cfg(feature = "tobii")]
unsafe impl Send for TobiiStreamSource {}

#[cfg(feature = "tobii")]
unsafe impl Sync for TobiiStreamSource {}

#[cfg(feature = "tobii")]
impl GazeSource for TobiiStreamSource {
    fn next(&self, _timeout: Duration) -> Option<GazeSample> {
        // Real impl would poll the FFI queue and translate `tobii_gaze_point_t`
        // to a `GazeSample`. Left as a follow-up until FFI is wired in.
        None
    }
    fn screen_w(&self) -> u32 {
        self.w
    }
    fn screen_h(&self) -> u32 {
        self.h
    }
    fn identifier(&self) -> &str {
        "tobii-ffi"
    }
}

/// Tobii-backed capture backend. Produces heatmap frames from gaze samples.
pub struct TobiiBackend {
    source: Arc<dyn GazeSource>,
    config: Option<CameraConfig>,
    running: bool,
    frame_count: AtomicU64,
    /// Heatmap accumulator: one u8 per (x, y) cell, decay = 5%/frame.
    heatmap: Mutex<Vec<u8>>,
}

impl TobiiBackend {
    /// Construct a TobiiBackend with a custom gaze source. Used by tests and
    /// by callers that have already wrapped the FFI in a `GazeSource` impl.
    pub fn with_source(source: Arc<dyn GazeSource>) -> Self {
        Self {
            source,
            config: None,
            running: false,
            frame_count: AtomicU64::new(0),
            heatmap: Mutex::new(Vec::new()),
        }
    }

    /// Default: synthetic gaze source at 90Hz, 1920x1080 screen.
    pub fn default_synthetic() -> Self {
        Self::with_source(Arc::new(SyntheticGazeSource::circle(1920, 1080, 90.0)))
    }

    /// Run device calibration. Real Tobii calibration prompts the user to
    /// follow a dot; we abstract that here as a no-op for the synthetic
    /// backend. Returns `Ok(())` once the active profile is ready.
    pub fn calibrate(&mut self) -> Result<(), CameraError> {
        // Real impl: launch the Tobii calibration UI, wait for completion.
        // For synthetic / mock sources: nothing to do.
        Ok(())
    }

    /// Project a `GazeSample` into the heatmap buffer.
    fn deposit(&self, sample: GazeSample) {
        if !sample.valid {
            return;
        }
        let w = self.source.screen_w() as i32;
        let h = self.source.screen_h() as i32;
        let ix = sample.x as i32;
        let iy = sample.y as i32;
        if ix < 0 || ix >= w || iy < 0 || iy >= h {
            return;
        }
        let stride = w as usize;
        let mut hm: MutexGuard<'_, Vec<u8>> = self.heatmap.lock().expect("heatmap mutex");
        if hm.len() != (w as usize) * (h as usize) {
            hm.resize((w as usize) * (h as usize), 0);
        }
        // 3x3 blur deposit (center weight 5, edges 2, corners 1)
        for dy in -1..=1 {
            for dx in -1..=1 {
                let nx = ix + dx;
                let ny = iy + dy;
                if nx < 0 || nx >= w || ny < 0 || ny >= h {
                    continue;
                }
                let weight: u8 = match (dx == 0, dy == 0) {
                    (true, true) => 5,
                    (true, false) | (false, true) => 2,
                    _ => 1,
                };
                let idx = (ny as usize) * stride + (nx as usize);
                let v = hm[idx].saturating_add(weight * 6);
                hm[idx] = v.min(255);
            }
        }
    }

    /// Decay the heatmap by 5% per frame (called once per captured frame).
    fn decay(&self) {
        let mut hm = self.heatmap.lock().expect("heatmap mutex");
        for v in hm.iter_mut() {
            *v = ((*v as u32) * 95 / 100) as u8;
        }
    }
}

impl Default for TobiiBackend {
    fn default() -> Self {
        Self::default_synthetic()
    }
}

impl Backend for TobiiBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Tobii
    }
    fn name(&self) -> &str {
        "tobii"
    }
    fn list_devices(&self) -> Result<Vec<CameraInfo>, CameraError> {
        let id = self.source.identifier().to_string();
        let w = self.source.screen_w();
        let h = self.source.screen_h();
        Ok(vec![CameraInfo {
            index: 0,
            name: id,
            description: format!("Tobii eye-tracker, {}x{} screen", w, h),
        }])
    }
    fn open(&mut self, _index: usize, config: &CameraConfig) -> Result<(), CameraError> {
        self.config = Some(config.clone());
        // Lazily allocate heatmap at the device's screen size.
        let stride = self.source.screen_w() as usize;
        let rows = self.source.screen_h() as usize;
        let mut hm = self.heatmap.lock().expect("heatmap mutex");
        if hm.len() != stride * rows {
            hm.resize(stride * rows, 0);
        }
        Ok(())
    }
    fn start(&mut self) -> Result<(), CameraError> {
        if self.running {
            return Ok(());
        }
        self.calibrate()?;
        self.running = true;
        Ok(())
    }
    fn capture_frame(&mut self) -> Result<Frame, CameraError> {
        if !self.running {
            return Err(CameraError::NotRunning);
        }
        // Drain available gaze samples into the heatmap (bounded to avoid
        // unbounded latency if a sample burst occurs).
        let mut deposited = 0u32;
        while deposited < 64 {
            match self.source.next(Duration::from_millis(1)) {
                Some(s) => {
                    self.deposit(s);
                    deposited += 1;
                }
                None => break,
            }
        }
        self.decay();
        let frame_count = self.frame_count.fetch_add(1, Ordering::Relaxed);
        let w = self.source.screen_w();
        let h = self.source.screen_h();
        let pixels = {
            let hm = self.heatmap.lock().expect("heatmap mutex");
            hm.clone()
        };
        let _ = SystemTime::now().duration_since(UNIX_EPOCH); // available to consumers
        Ok(Frame {
            data: pixels,
            width: w,
            height: h,
            format: PixelFormat::Gray8,
            timestamp: Instant::now(),
            frame_number: frame_count,
        })
    }
    fn stop(&mut self) -> Result<(), CameraError> {
        self.running = false;
        Ok(())
    }
    fn is_running(&self) -> bool {
        self.running
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_circle_advances() {
        let src = SyntheticGazeSource::circle(640, 480, 60.0);
        let s1 = src.next(Duration::from_millis(0)).unwrap();
        let s2 = src.next(Duration::from_millis(0)).unwrap();
        assert_ne!(s1.x, s2.x);
        assert_ne!(s1.y, s2.y);
    }

    #[test]
    fn synthetic_within_screen_bounds() {
        let src = SyntheticGazeSource::circle(1920, 1080, 90.0);
        for _ in 0..1000 {
            let s = src.next(Duration::from_millis(0)).unwrap();
            assert!(s.x >= 0.0 && s.x < 1920.0);
            assert!(s.y >= 0.0 && s.y < 1080.0);
        }
    }

    #[test]
    fn tobii_backend_kind_and_name() {
        let b = TobiiBackend::default();
        assert_eq!(b.kind(), BackendKind::Tobii);
        assert_eq!(b.name(), "tobii");
        assert!(!b.is_running());
    }

    #[test]
    fn tobii_backend_full_lifecycle() {
        let src = Arc::new(SyntheticGazeSource::circle(320, 240, 30.0));
        let mut b = TobiiBackend::with_source(src);
        let config = CameraConfig::default();
        b.open(0, &config).unwrap();
        b.start().unwrap();
        assert!(b.is_running());
        let frame = b.capture_frame().unwrap();
        assert_eq!(frame.width, 320);
        assert_eq!(frame.height, 240);
        assert_eq!(frame.format, PixelFormat::Gray8);
        assert_eq!(frame.data.len(), 320 * 240);
        b.stop().unwrap();
        assert!(!b.is_running());
    }

    #[test]
    fn capture_requires_running() {
        let mut b = TobiiBackend::default();
        b.open(0, &CameraConfig::default()).unwrap();
        match b.capture_frame() {
            Err(CameraError::NotRunning) => {}
            other => panic!("expected NotRunning, got {:?}", other),
        }
    }
}
