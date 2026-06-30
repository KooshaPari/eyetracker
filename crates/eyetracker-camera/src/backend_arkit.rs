//! ARKit / Vision-based eye-tracker backend (EYE-SOTA-004).
//!
//! Provides gaze + face tracking for Apple iOS / iPadOS / visionOS apps that
//! ship ARKit. Mirrors the `backend_tobii.rs` pattern: a `GazeSource` trait
//! abstracts the producer so the FFI to ARKit + the Vision framework can be
//! feature-gated and replaced with a synthetic source in tests / desktop CI.
//!
//! ARKit is the only eye-tracker API that ships on commodity Apple devices
//! (iPhone XS+, iPad Pro 2018+ with TrueDepth, visionOS). Other
//! mobile/desktop paths (Android CameraX + ML Kit Face Detection, macOS
//! Vision via VNDetectFaceLandmarksRequest) are intentionally NOT here —
//! they belong to follow-up units EYE-SOTA-006 / EYE-SOTA-007 / EYE-SOTA-008
//! to keep this adapter's surface area focused.
//!
//! ## Architecture
//!
//! ```text
//!   ArcKitBackend (impl backend::Backend)
//!        │
//!        │   Arc<dyn GazeSource>
//!        ▼
//!   ┌────────────────────────────────────────────┐
//!   │  GazeSource trait                          │
//!   │   fn next_gaze(&mut self) -> Option<Gaze>  │
//!   │   fn device_info(&self) -> DeviceInfo      │
//!   └────────────────────────────────────────────┘
//!        ▲
//!        │ runtime-selected
//!   ┌────┴───────────────────────────────┐
//!   │                                    │
//!   SyntheticArkItSource          ArkItFfiSource
//!   (deterministic                   (gated behind `arkit` feature;
//!    circle at 60Hz,                   unsafe FFI to ARKit session +
//!    used by tests                     Vision VNDetectFaceLandmarks)
//!   ```
//!
//! ## Heatmap output
//!
//! The ARKit adapter mirrors the Tobii adapter: gaze samples are
//! accumulated into a heatmap buffer (1u8/cell, 3x3 blur deposit,
//! 5%/frame decay). The buffer is rendered as an RGB Frame so the rest
//! of the inference pipeline is backend-agnostic.
//!
//! ## Feature flags
//!
//! - `arkit = []`: gate for the unsafe FFI source. Off by default so the
//!   crate builds on non-Apple platforms and CI without the iOS SDK.
//!   The `Backend` impl, `SyntheticArkItSource`, and the
//!   `ArkItBackend::with_source(...)` constructor are always available.
//!
//! ## Why iOS first
//!
//! ARKit + the TrueDepth front camera is the only eye-tracker source that
//! ships on commodity Apple hardware. It's also the only one with a
//! real device launch + validation path (visionOS App Store requires
//! declared eye-tracking usage strings). Tobii/Pupil/HTC Vive Pro Eye
//! cover the dedicated-hardware tier; the ARKit adapter covers the
//! consumer mobile tier.
//!
//! ## References
//!
//! - ARKit `ARFaceAnchor` (`lookAtPoint`, `blendShapes`, `transform`).
//! - Apple Vision framework: `VNDetectFaceLandmarksRequest`,
//!   `VNFaceLandmarkRegion2D` (eye corners, pupils).
//! - [`arkit-rs`](https://crates.io/crates/arkit-rs) (community Rust
//!   bindings — used as a model for the FFI shape once feature is wired).

use std::sync::Arc;
use std::time::Instant;

use super::backend::{Backend, BackendKind, CameraConfig, CameraError, CameraInfo, Frame};
use super::PixelFormat;

/// A single gaze sample produced by the GazeSource.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GazeSample {
    /// Normalized x in [0.0, 1.0] (0 = left edge, 1 = right edge).
    pub x: f32,
    /// Normalized y in [0.0, 1.0] (0 = top edge, 1 = bottom edge).
    pub y: f32,
    /// Confidence in [0.0, 1.0]. 0 means "no face detected this frame".
    pub confidence: f32,
    /// Wall-clock instant when the source produced the sample.
    pub timestamp: Instant,
}

impl GazeSample {
    /// Construct a gaze sample, clamping to unit square and confidence in [0, 1].
    pub fn new(x: f32, y: f32, confidence: f32, timestamp: Instant) -> Self {
        Self {
            x: x.clamp(0.0, 1.0),
            y: y.clamp(0.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
            timestamp,
        }
    }
}

/// Identifier for the underlying ARKit device / face anchor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArkItDeviceInfo {
    /// ARKit `ARFaceAnchor` UUID, or "synthetic" for the test source.
    pub anchor_id: String,
    /// ARKit `ARFaceGeometry` vertex count, when known.
    pub vertex_count: Option<usize>,
    /// Backend display name (e.g. "iPhone 15 Pro TrueDepth", "SyntheticARKitSource").
    pub display_name: String,
}

impl ArkItDeviceInfo {
    /// Format the device info as a single string for logs / errors.
    pub fn as_str(&self) -> String {
        match self.vertex_count {
            Some(v) => format!("{} ({}, {} vertices)", self.display_name, self.anchor_id, v),
            None => format!("{} ({})", self.display_name, self.anchor_id),
        }
    }
}

/// Producer abstraction for ARKit gaze samples. Mirrors the Tobii adapter.
pub trait GazeSource: Send {
    /// Produce the next gaze sample, or `None` if the face is lost.
    fn next_gaze(&mut self) -> Option<GazeSample>;

    /// Device / face-anchor metadata.
    fn device_info(&self) -> ArkItDeviceInfo;
}

// -----------------------------------------------------------------------------
// Synthetic source (no SDK, used by tests + desktop CI)
// -----------------------------------------------------------------------------

/// Deterministic source: traces a slow circle trajectory at 60Hz and
/// reports full confidence. The `arkit` feature gates only the FFI
/// source; this one is always available so the `Backend` impl can be
/// tested in CI without iOS hardware.
pub struct SyntheticArkItSource {
    width: u32,
    height: u32,
    device: ArkItDeviceInfo,
    started: Option<Instant>,
    sample_count: u64,
}

impl SyntheticArkItSource {
    /// Create a synthetic source. Resolution is only used to scale the
    /// gaze trajectory; the heatmap is rendered at the Backend layer.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            device: ArkItDeviceInfo {
                anchor_id: "synthetic".to_string(),
                vertex_count: Some(1220), // ARFaceGeometry default
                display_name: "SyntheticARKitSource".to_string(),
            },
            started: None,
            sample_count: 0,
        }
    }
}

impl GazeSource for SyntheticArkItSource {
    fn next_gaze(&mut self) -> Option<GazeSample> {
        let started = self.started.unwrap_or_else(Instant::now);
        if self.started.is_none() {
            self.started = Some(started);
        }
        // 60Hz cadence: ~16.67ms per sample
        let elapsed = started.elapsed();
        let expected = self.sample_count * 16;
        if elapsed.as_millis() < expected as u128 {
            // Not yet time for the next sample — return the last position.
            return self.last_synthetic_sample();
        }
        self.sample_count += 1;
        // Slow circle (radius 0.3, period 6s)
        let t = elapsed.as_secs_f32() * std::f32::consts::TAU / 6.0;
        let x = 0.5 + 0.3 * t.cos();
        let y = 0.5 + 0.3 * t.sin();
        Some(GazeSample::new(x, y, 1.0, Instant::now()))
    }

    fn device_info(&self) -> ArkItDeviceInfo {
        self.device.clone()
    }
}

impl SyntheticArkItSource {
    fn last_synthetic_sample(&self) -> Option<GazeSample> {
        if self.sample_count == 0 {
            return None;
        }
        let started = self.started.unwrap_or_else(Instant::now);
        let t = started.elapsed().as_secs_f32() * std::f32::consts::TAU / 6.0;
        let x = 0.5 + 0.3 * t.cos();
        let y = 0.5 + 0.3 * t.sin();
        Some(GazeSample::new(x, y, 1.0, Instant::now()))
    }
}

// -----------------------------------------------------------------------------
// Real ARKit FFI source (gated behind `arkit` feature; unsafe stub for now)
// -----------------------------------------------------------------------------

/// ARKit FFI source. Currently a stub that returns `None` (face-lost)
/// — the actual FFI to `ARSession` + `ARFaceAnchor` is wired in a
/// follow-up commit once the iOS toolchain + bridging header are
/// added to the repo. The unsafe contract is documented so the
/// eventual binding can be slotted in without changing the public
/// `GazeSource` trait.
pub struct ArkItFfiSource {
    device: ArkItDeviceInfo,
    /// `Some(())` once `ARSession` is running; `None` means the
    /// feature is wired but the unsafe FFI handle is not yet bound.
    session_active: bool,
}

impl ArkItFfiSource {
    /// Attempt to open an `ARSession` and start face tracking. The
    /// actual `objc2` FFI call is unsafe and is gated behind the
    /// `arkit` feature; this constructor returns an error in the
    /// default build (no iOS toolchain) and the unsafe binding is
    /// a follow-up commit.
    pub fn try_new() -> Result<Self, CameraError> {
        #[cfg(feature = "arkit")]
        {
            // SAFETY: the eventual FFI boundary must document the
            // invariants (single-thread access, retain/release on the
            // ARSession handle, dispatch queue pinning). The stub
            // is replaced with the real unsafe block in EYE-SOTA-004-ffi.
            Ok(Self {
                device: ArkItDeviceInfo {
                    anchor_id: "arkit-ffi-stub".to_string(),
                    vertex_count: None,
                    display_name: "ARKit FFI (unbound)".to_string(),
                },
                session_active: false,
            })
        }
        #[cfg(not(feature = "arkit"))]
        {
            Err(CameraError::InitFailed(
                "arkit feature not enabled (set `features = [\"arkit\"]` in Cargo.toml)".to_string(),
            ))
        }
    }
}

impl GazeSource for ArkItFfiSource {
    fn next_gaze(&mut self) -> Option<GazeSample> {
        if !self.session_active {
            return None;
        }
        // When the FFI is wired, the `objc2` runtime will marshal
        // `ARFaceAnchor` updates through a per-frame callback. The
        // callback path is the cleanest because ARKit's session runs
        // on its own dispatch queue.
        None
    }

    fn device_info(&self) -> ArkItDeviceInfo {
        self.device.clone()
    }
}

// -----------------------------------------------------------------------------
// Backend impl
// -----------------------------------------------------------------------------

/// ARKit-based `Backend` implementation. Holds an `Arc<dyn GazeSource>`
/// (so the same source can be shared with the inference pipeline), a
/// heatmap accumulator, and the lifecycle state machine.
pub struct ArkItBackend {
    source: Arc<Mutex<dyn GazeSource>>,
    config: Option<CameraConfig>,
    running: bool,
    frame_count: u64,
    /// Heatmap buffer (RGB) with 1u8/cell, 3x3 blur deposit, 5%/frame decay.
    heatmap: Vec<u8>,
    width: u32,
    height: u32,
}

impl ArkItBackend {
    /// Create an ARKit backend driven by a caller-provided `GazeSource`.
    /// Use this with `SyntheticArkItSource` in tests and CI; use
    /// `ArkItFfiSource::try_new()?` on iOS hardware.
    pub fn with_source(source: Arc<Mutex<dyn GazeSource>>, config: CameraConfig) -> Self {
        Self {
            source,
            config: None,
            running: false,
            frame_count: 0,
            heatmap: Vec::new(),
            width: 0,
            height: 0,
        }
    }

    /// Decay the heatmap by 5% per frame (drainage toward black).
    fn decay(&mut self) {
        for v in self.heatmap.iter_mut() {
            *v = ((*v as f32) * 0.95) as u8;
        }
    }

    /// Deposit a single gaze point with a 3x3 blur.
    fn deposit(&mut self, x: f32, y: f32) {
        let w = self.width as i32;
        let h = self.height as i32;
        let cx = (x.clamp(0.0, 1.0) * (w - 1) as f32) as i32;
        let cy = (y.clamp(0.0, 1.0) * (h - 1) as f32) as i32;
        for dy in -1..=1 {
            for dx in -1..=1 {
                let nx = cx + dx;
                let ny = cy + dy;
                if nx < 0 || ny < 0 || nx >= w || ny >= h {
                    continue;
                }
                let idx = ((ny as usize) * (w as usize) + (nx as usize)) * 3;
                // Brighten red channel; gaze hot spot.
                self.heatmap[idx] = self.heatmap[idx].saturating_add(64);
                self.heatmap[idx + 1] = self.heatmap[idx + 1].saturating_add(16);
                self.heatmap[idx + 2] = self.heatmap[idx + 2].saturating_add(16);
            }
        }
    }
}

use std::sync::Mutex;

impl Backend for ArkItBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::ARKit
    }

    fn name(&self) -> &str {
        "ArkItBackend"
    }

    fn list_devices(&self) -> Result<Vec<CameraInfo>, CameraError> {
        let info = self.source.lock().map_err(|e| CameraError::InitFailed(format!("source poisoned: {e}")))?.device_info();
        Ok(vec![CameraInfo {
            index: 0,
            name: info.display_name.clone(),
            description: format!(
                "ARKit face anchor ({} vertices, anchor={})",
                info.vertex_count.unwrap_or(0),
                info.anchor_id
            ),
        }])
    }

    fn open(&mut self, _index: usize, config: &CameraConfig) -> Result<(), CameraError> {
        self.width = config.width;
        self.height = config.height;
        let buf = (config.width as usize) * (config.height as usize) * 3;
        self.heatmap = vec![0u8; buf];
        self.config = Some(config.clone());
        self.frame_count = 0;
        self.running = false;
        Ok(())
    }

    fn start(&mut self) -> Result<(), CameraError> {
        if self.config.is_none() {
            return Err(CameraError::NotRunning);
        }
        self.running = true;
        Ok(())
    }

    fn capture_frame(&mut self) -> Result<Frame, CameraError> {
        if !self.running {
            return Err(CameraError::NotRunning);
        }
        self.frame_count += 1;

        // Drain up to 8 gaze samples per frame (ARKit runs at 60Hz, the
        // consumer might be at 30Hz; coalesce in batches of 8).
        let mut count = 0;
        loop {
            if count >= 8 {
                break;
            }
            let sample = {
                let mut src = self
                    .source
                    .lock()
                    .map_err(|e| CameraError::CaptureFailed(format!("source poisoned: {e}")))?;
                src.next_gaze()
            };
            match sample {
                Some(s) if s.confidence > 0.05 => {
                    self.deposit(s.x, s.y);
                    count += 1;
                }
                _ => break,
            }
        }

        // Decay after deposit so the next frame starts from a slightly
        // dimmed heatmap.
        self.decay();

        let config = self.config.as_ref().ok_or(CameraError::NotRunning)?;
        Ok(Frame {
            data: self.heatmap.clone(),
            width: config.width,
            height: config.height,
            format: PixelFormat::Rgb8,
            timestamp: Instant::now(),
            frame_number: self.frame_count,
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
    use std::time::Duration;

    #[test]
    fn synthetic_circle_advances() {
        let mut src = SyntheticArkItSource::new(640, 480);
        let s1 = src.next_gaze();
        std::thread::sleep(Duration::from_millis(20));
        let s2 = src.next_gaze();
        assert!(s1.is_some());
        assert!(s2.is_some());
    }

    #[test]
    fn synthetic_within_screen_bounds() {
        let mut src = SyntheticArkItSource::new(640, 480);
        for _ in 0..32 {
            let s = src.next_gaze().expect("sample");
            assert!((0.0..=1.0).contains(&s.x), "x out of bounds: {}", s.x);
            assert!((0.0..=1.0).contains(&s.y), "y out of bounds: {}", s.y);
        }
    }

    #[test]
    fn arkit_backend_kind_and_name() {
        let src: Arc<Mutex<dyn GazeSource>> = Arc::new(Mutex::new(SyntheticArkItSource::new(640, 480)));
        let backend = ArkItBackend::with_source(src, CameraConfig::default());
        assert_eq!(backend.kind(), BackendKind::ARKit);
        assert_eq!(backend.name(), "ArkItBackend");
    }

    #[test]
    fn arkit_backend_full_lifecycle() {
        let src: Arc<Mutex<dyn GazeSource>> = Arc::new(Mutex::new(SyntheticArkItSource::new(640, 480)));
        let mut backend = ArkItBackend::with_source(src, CameraConfig::default());
        backend.open(0, &CameraConfig::default()).expect("open");
        backend.start().expect("start");
        let frame = backend.capture_frame().expect("capture");
        assert_eq!(frame.width, 640);
        assert_eq!(frame.height, 480);
        assert_eq!(frame.format, PixelFormat::Rgb8);
        backend.stop().expect("stop");
        assert!(!backend.is_running());
    }

    #[test]
    fn capture_requires_running() {
        let src: Arc<Mutex<dyn GazeSource>> = Arc::new(Mutex::new(SyntheticArkItSource::new(640, 480)));
        let mut backend = ArkItBackend::with_source(src, CameraConfig::default());
        backend.open(0, &CameraConfig::default()).expect("open");
        let err = backend.capture_frame().expect_err("should not capture before start");
        assert!(matches!(err, CameraError::NotRunning));
    }

    #[test]
    fn ffi_source_disabled_without_feature() {
        // Without the `arkit` feature, the FFI source is intentionally
        // unavailable so non-Apple builds don't link CoreFoundation.
        let result = ArkItFfiSource::try_new();
        assert!(result.is_err(), "expected InitFailed when arkit feature is off");
    }
}
