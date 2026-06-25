//! eyetracker-camera: Webcam capture abstraction
//!
//! Provides camera enumeration, frame capture, and configuration
//! via nokhwa (AVFoundation on macOS).

use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{anyhow, Result};
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{
    ApiBackend, CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType,
    Resolution,
};
use nokhwa::Camera as NokhwaCamera;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Camera-related errors
#[derive(Error, Debug)]
pub enum CameraError {
    #[error("No cameras available")]
    NoCameras,
    #[error("Camera initialization failed: {0}")]
    InitFailed(String),
    #[error("Camera capture failed: {0}")]
    CaptureFailed(String),
    #[error("Camera not running")]
    NotRunning,
}

/// Pixel format of a captured frame
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PixelFormat {
    Rgb8,
    Bgr8,
    Gray8,
}

/// A single captured frame from the camera
#[derive(Debug, Clone)]
pub struct Frame {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
    pub timestamp: Instant,
    pub frame_number: u64,
}

impl Frame {
    pub fn new(data: Vec<u8>, width: u32, height: u32, format: PixelFormat) -> Self {
        Self {
            data,
            width,
            height,
            format,
            timestamp: Instant::now(),
            frame_number: 0,
        }
    }

    /// Total pixel count
    pub fn pixel_count(&self) -> u64 {
        self.width as u64 * self.height as u64
    }

    /// Bytes per pixel based on format
    pub fn bytes_per_pixel(&self) -> u8 {
        match self.format {
            PixelFormat::Rgb8 | PixelFormat::Bgr8 => 3,
            PixelFormat::Gray8 => 1,
        }
    }
}

/// Camera configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraConfig {
    /// Target frame rate
    pub target_fps: u32,
    /// Frame width in pixels
    pub width: u32,
    /// Frame height in pixels
    pub height: u32,
    /// Camera device index
    pub camera_index: usize,
    /// Enable low-light enhancement
    pub low_light_mode: bool,
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            target_fps: 60,
            width: 640,
            height: 480,
            camera_index: 0,
            low_light_mode: false,
        }
    }
}

impl CameraConfig {
    /// Configuration optimized for eye tracking (640x480@60fps)
    pub fn eye_tracking() -> Self {
        Self {
            target_fps: 60,
            width: 640,
            height: 480,
            camera_index: 0,
            low_light_mode: true,
        }
    }

    /// Low-resolution config for faster processing (320x240@120fps)
    pub fn low_res() -> Self {
        Self {
            target_fps: 120,
            width: 320,
            height: 240,
            camera_index: 0,
            low_light_mode: false,
        }
    }

    /// High-resolution config (1280x720@30fps)
    pub fn high_res() -> Self {
        Self {
            target_fps: 30,
            width: 1280,
            height: 720,
            camera_index: 0,
            low_light_mode: false,
        }
    }
}

/// Camera device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraInfo {
    pub index: usize,
    pub name: String,
    pub description: String,
}

/// List available camera devices
pub fn list_cameras() -> Vec<CameraInfo> {
    let mut cameras = Vec::new();
    match nokhwa::query(ApiBackend::Auto) {
        Ok(list) => {
            for (i, info) in list.iter().enumerate() {
                cameras.push(CameraInfo {
                    index: i,
                    name: info.human_name().to_string(),
                    description: info.description().to_string(),
                });
            }
        }
        Err(e) => {
            tracing::warn!("Failed to query cameras: {e}");
        }
    }
    cameras
}

/// Camera capture handle wrapping nokhwa
pub struct Camera {
    inner: Option<NokhwaCamera>,
    config: CameraConfig,
    running: bool,
    frame_count: u64,
    start_time: Option<Instant>,
    backend: Option<Arc<Mutex<dyn backend::Backend>>>,
}

impl Camera {
    /// Create a new camera with the given configuration
    pub fn new(config: CameraConfig) -> Result<Self> {
        let index = CameraIndex::Index(config.camera_index as u32);
        let resolution = Resolution::new(config.width, config.height);
        let camera_format = CameraFormat::new(resolution, FrameFormat::RAWRGB, config.target_fps);
        let requested =
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::Exact(camera_format));

        let camera = NokhwaCamera::new(index, requested)
            .map_err(|e| anyhow!("Failed to create camera: {e}"))?;

        tracing::info!(
            "Camera created: idx={} res={}x{} fps={}",
            config.camera_index,
            config.width,
            config.height,
            config.target_fps,
        );

        Ok(Self {
            inner: Some(camera),
            config,
            running: false,
            frame_count: 0,
            start_time: None,
            backend: None,
        })
    }

    /// Open the default camera with eye-tracking optimized settings
    pub fn open_default() -> Result<Self> {
        let config = CameraConfig::eye_tracking();
        let mut cam = Self::new(config)?;
        cam.start()?;
        Ok(cam)
    }

    /// Start the camera stream
    pub fn start(&mut self) -> Result<()> {
        if self.running {
            return Ok(());
        }
        let camera = self.inner.as_mut().ok_or_else(|| anyhow!("No camera"))?;
        camera
            .open_stream()
            .map_err(|e| anyhow!("Failed to open stream: {e}"))?;
        self.running = true;
        self.start_time = Some(Instant::now());
        self.frame_count = 0;
        tracing::info!("Camera stream started");
        Ok(())
    }

    /// Stop the camera stream
    pub fn stop(&mut self) -> Result<()> {
        if !self.running {
            return Ok(());
        }
        if let Some(ref mut camera) = self.inner {
            let _ = camera.stop_stream();
        }
        self.running = false;
        tracing::info!("Camera stream stopped");
        Ok(())
    }

    /// Capture a single frame (blocking). Returns RGB8 pixel data.
    pub fn capture_frame(&mut self) -> Result<Frame> {
        let camera = self.inner.as_mut().ok_or_else(|| CameraError::NotRunning)?;
        if !self.running {
            return Err(CameraError::NotRunning.into());
        }

        let buffer = camera
            .frame()
            .map_err(|e| anyhow!("Frame capture failed: {e}"))?;

        self.frame_count += 1;

        // Decode to ImageBuffer<Rgb<u8>, Vec<u8>>
        let decoded = buffer
            .decode_image::<RgbFormat>()
            .map_err(|e| anyhow!("Frame decode failed: {e}"))?;

        let data = decoded.into_raw();

        Ok(Frame {
            data,
            width: self.config.width,
            height: self.config.height,
            format: PixelFormat::Rgb8,
            timestamp: Instant::now(),
            frame_number: self.frame_count,
        })
    }

    /// Capture a frame with async timeout
    pub async fn capture_frame_async(&mut self, timeout: std::time::Duration) -> Result<Frame> {
        tokio::time::timeout(timeout, async { self.capture_frame() })
            .await
            .map_err(|_| anyhow!("Frame capture timed out"))?
    }

    /// Get current frames-per-second
    pub fn fps(&self) -> f64 {
        self.current_fps()
    }

    /// Get current frames-per-second
    pub fn current_fps(&self) -> f64 {
        if let Some(start) = self.start_time {
            let elapsed = start.elapsed().as_secs_f64();
            if elapsed > 0.0 {
                return self.frame_count as f64 / elapsed;
            }
        }
        0.0
    }

    /// Check if camera is running
    pub fn is_running(&self) -> bool {
        self.running
    }

    /// Get camera configuration
    pub fn config(&self) -> &CameraConfig {
        &self.config
    }

    /// Total frames captured
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Camera human-readable name
    pub fn name(&self) -> String {
        self.inner
            .as_ref()
            .map(|c| c.info().human_name().to_string())
            .unwrap_or_else(|| "Unknown".to_string())
    }
}

impl Drop for Camera {
    fn drop(&mut self) {
        if self.running {
            let _ = self.stop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_camera_config_defaults() {
        let c = CameraConfig::default();
        assert_eq!(c.width, 640);
        assert_eq!(c.height, 480);
        assert_eq!(c.target_fps, 60);
    }

    #[test]
    fn test_frame_creation() {
        let data = vec![0u8; 640 * 480 * 3];
        let f = Frame::new(data, 640, 480, PixelFormat::Rgb8);
        assert_eq!(f.pixel_count(), 640 * 480);
        assert_eq!(f.bytes_per_pixel(), 3);
    }

    #[test]
    fn test_frame_format_bytes() {
        assert_eq!(PixelFormat::Rgb8.bytes_per_pixel(), 3);
        assert_eq!(PixelFormat::Gray8.bytes_per_pixel(), 1);
    }
}

// Helper on PixelFormat (in this module, not conflicting)
impl PixelFormat {
    #[allow(dead_code)]
    fn bytes_per_pixel(&self) -> u8 {
        match self {
            PixelFormat::Rgb8 | PixelFormat::Bgr8 => 3,
            PixelFormat::Gray8 => 1,
        }
    }
}

// ============================================================================
// Pluggable Backend trait (EYE-SOTA-001)
// ============================================================================
//
// Decouples `Camera` from a single capture implementation. The default
// `WebcamBackend` (nokhwa/AVFoundation) remains the implicit backend so
// existing callers compile unchanged.
//
// New backends implement `Backend` and are constructed via `Camera::with_backend`:
//
//     use std::sync::Arc;
//     use eyetracker_camera::{Camera, CameraConfig};
//     use eyetracker_camera::backend::{Backend, WebcamBackend, SyntheticBackend};
//
//     let backend = Arc::new(WebcamBackend::new());
//     let camera = Camera::with_backend(backend, CameraConfig::eye_tracking())?;
//
// Future EYE-SOTA units will add:
//   - TobiiBackend (Tobii Stream Engine)
//   - PupilBackend (ZMQ bridge to pupil-core)
//   - ARKitBackend (UniFFI Swift bridge)
//   - V4l2Backend (Linux)

pub mod backend {
    //! Pluggable capture backends. See `Backend` trait docs.

    use super::{CameraConfig, CameraInfo, CameraError, Frame};
    use std::sync::Arc;

    /// Identifier for a capture backend kind.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum BackendKind {
        Webcam,
        Synthetic,
        Tobii,
        Pupil,
        ARKit,
        V4l2,
        MediaFoundation,
    }

    impl BackendKind {
        pub fn as_str(&self) -> &'static str {
            match self {
                BackendKind::Webcam => "webcam",
                BackendKind::Synthetic => "synthetic",
                BackendKind::Tobii => "tobii",
                BackendKind::Pupil => "pupil",
                BackendKind::ARKit => "arkit",
                BackendKind::V4l2 => "v4l2",
                BackendKind::MediaFoundation => "media_foundation",
            }
        }
    }

    /// Pluggable capture backend. Implementations produce `Frame`s on demand.
    ///
    /// The default implementation is `WebcamBackend` (nokhwa). All
    /// implementations must be `Send + Sync` so `Camera` can wrap them
    /// in `Arc<dyn Backend>`.
    pub trait Backend: Send + Sync {
        /// Identify the backend kind.
        fn kind(&self) -> BackendKind;

        /// Human-readable backend name (for logs / errors).
        fn name(&self) -> &str;

        /// Enumerate available devices.
        fn list_devices(&self) -> Result<Vec<CameraInfo>, CameraError>;

        /// Open the device at `index` with the given config.
        fn open(&mut self, index: usize, config: &CameraConfig) -> Result<(), CameraError>;

        /// Start streaming frames.
        fn start(&mut self) -> Result<(), CameraError>;

        /// Capture a single frame (blocking).
        fn capture_frame(&mut self) -> Result<Frame, CameraError>;

        /// Stop streaming.
        fn stop(&mut self) -> Result<(), CameraError>;

        /// Whether the backend is currently streaming.
        fn is_running(&self) -> bool;
    }

    /// Default backend: nokhwa / AVFoundation webcam. Always available where
    /// nokhwa is built.
    pub struct WebcamBackend {
        inner: Option<nokhwa::Camera>,
        config: Option<CameraConfig>,
        running: bool,
        frame_count: u64,
    }

    impl WebcamBackend {
        pub fn new() -> Self {
            Self {
                inner: None,
                config: None,
                running: false,
                frame_count: 0,
            }
        }
    }

    impl Default for WebcamBackend {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Backend for WebcamBackend {
        fn kind(&self) -> BackendKind { BackendKind::Webcam }
        fn name(&self) -> &str { "WebcamBackend (nokhwa)" }

        fn list_devices(&self) -> Result<Vec<CameraInfo>, CameraError> {
            let mut cameras = Vec::new();
            match nokhwa::utils::query(nokhwa::utils::ApiBackend::Auto) {
                Ok(list) => {
                    for (i, info) in list.iter().enumerate() {
                        cameras.push(CameraInfo {
                            index: i,
                            name: info.human_name().to_string(),
                            description: info.description().to_string(),
                        });
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to query cameras: {e}");
                }
            }
            Ok(cameras)
        }

        fn open(&mut self, index: usize, config: &CameraConfig) -> Result<(), CameraError> {
            use nokhwa::pixel_format::RgbFormat;
            use nokhwa::utils::*;
            let cam_index = CameraIndex::Index(index as u32);
            let resolution = Resolution::new(config.width, config.height);
            let camera_format = CameraFormat::new(resolution, FrameFormat::RAWRGB, config.target_fps);
            let requested = RequestedFormat::new::<RgbFormat>(RequestedFormatType::Exact(camera_format));
            let camera = nokhwa::Camera::new(cam_index, requested)
                .map_err(|e| CameraError::InitFailed(e.to_string()))?;
            self.inner = Some(camera);
            self.config = Some(config.clone());
            self.frame_count = 0;
            Ok(())
        }

        fn start(&mut self) -> Result<(), CameraError> {
            if self.running { return Ok(()); }
            let camera = self.inner.as_mut().ok_or(CameraError::NotRunning)?;
            camera.open_stream().map_err(|e| CameraError::CaptureFailed(e.to_string()))?;
            self.running = true;
            Ok(())
        }

        fn capture_frame(&mut self) -> Result<Frame, CameraError> {
            use nokhwa::pixel_format::RgbFormat;
            let camera = self.inner.as_mut().ok_or(CameraError::NotRunning)?;
            if !self.running { return Err(CameraError::NotRunning); }
            let buffer = camera.frame().map_err(|e| CameraError::CaptureFailed(e.to_string()))?;
            let decoded = buffer.decode_image::<RgbFormat>()
                .map_err(|e| CameraError::CaptureFailed(e.to_string()))?;
            let config = self.config.as_ref().ok_or(CameraError::NotRunning)?;
            self.frame_count += 1;
            Ok(Frame {
                data: decoded.into_raw(),
                width: config.width,
                height: config.height,
                format: super::PixelFormat::Rgb8,
                timestamp: std::time::Instant::now(),
                frame_number: self.frame_count,
            })
        }

        fn stop(&mut self) -> Result<(), CameraError> {
            if !self.running { return Ok(()); }
            if let Some(ref mut camera) = self.inner {
                let _ = camera.stop_stream();
            }
            self.running = false;
            Ok(())
        }

        fn is_running(&self) -> bool { self.running }
    }

    /// Synthetic backend: produces deterministic frames from an embedded
    /// gradient. Used for tests and CI environments without camera hardware.
    /// See EYE-SOTA-005.
    pub struct SyntheticBackend {
        config: Option<CameraConfig>,
        running: bool,
        frame_count: u64,
        width: u32,
        height: u32,
    }

    impl SyntheticBackend {
        pub fn new() -> Self {
            Self { config: None, running: false, frame_count: 0, width: 0, height: 0 }
        }
    }

    impl Default for SyntheticBackend {
        fn default() -> Self { Self::new() }
    }

    impl Backend for SyntheticBackend {
        fn kind(&self) -> BackendKind { BackendKind::Synthetic }
        fn name(&self) -> &str { "SyntheticBackend" }

        fn list_devices(&self) -> Result<Vec<CameraInfo>, CameraError> {
            Ok(vec![CameraInfo {
                index: 0,
                name: "synthetic-0".to_string(),
                description: "Deterministic gradient generator (no hardware)".to_string(),
            }])
        }

        fn open(&mut self, _index: usize, config: &CameraConfig) -> Result<(), CameraError> {
            self.width = config.width;
            self.height = config.height;
            self.config = Some(config.clone());
            self.frame_count = 0;
            Ok(())
        }

        fn start(&mut self) -> Result<(), CameraError> {
            self.running = true;
            Ok(())
        }

        fn capture_frame(&mut self) -> Result<Frame, CameraError> {
            if !self.running { return Err(CameraError::NotRunning); }
            self.frame_count += 1;
            let w = self.width as usize;
            let h = self.height as usize;
            let mut data = vec![0u8; w * h * 3];
            // Animated radial gradient. Deterministic given frame_count.
            let phase = (self.frame_count % 256) as u8;
            for y in 0..h {
                for x in 0..w {
                    let dx = (x as i32 - w as i32 / 2).unsigned_abs() as u8;
                    let dy = (y as i32 - h as i32 / 2).unsigned_abs() as u8;
                    let r = dx.wrapping_add(phase);
                    let g = dy.wrapping_add(phase);
                    let b = phase;
                    let idx = (y * w + x) * 3;
                    data[idx] = r;
                    data[idx + 1] = g;
                    data[idx + 2] = b;
                }
            }
            Ok(Frame {
                data,
                width: self.width,
                height: self.height,
                format: super::PixelFormat::Rgb8,
                timestamp: std::time::Instant::now(),
                frame_number: self.frame_count,
            })
        }

        fn stop(&mut self) -> Result<(), CameraError> {
            self.running = false;
            Ok(())
        }

        fn is_running(&self) -> bool { self.running }
    }
}

use backend::{Backend, BackendKind};

impl Camera {
    /// Create a camera backed by an explicit backend implementation.
    ///
    /// Use this to plug in non-webcam backends (Tobii, Pupil, ARKit, synthetic).
    /// The default webcam path is `Camera::new(config)` (unchanged).
    pub fn with_backend(backend: Arc<dyn Backend>, config: CameraConfig) -> Result<Self> {
        // Lock once to call open; the Mutex<...> wrapper enforces interior
        // mutability so the trait's &mut self methods remain callable through
        // an Arc (which is otherwise only Deref, not DerefMut).
        {
            let mut guard = backend
                .as_ref()
                .lock()
                .map_err(|e| anyhow!("backend mutex poisoned: {e}"))?;
            guard
                .open(0, &config)
                .map_err(|e| anyhow!("backend.open failed: {e}"))?;
        }
        tracing::info!(
            "Camera created with backend: {} res={}x{} fps={}",
            backend.name(),
            config.width,
            config.height,
            config.target_fps,
        );
        Ok(Self {
            inner: None, // legacy nokhwa inner is unused with explicit backend
            config,
            running: false,
            frame_count: 0,
            start_time: None,
            backend: Some(Arc::new(Mutex::new(backend.as_ref().lock().map_err(
                |e| anyhow!("backend mutex poisoned: {e}"),
            )?))),
        })
    }

    /// Identify the active backend kind (None when using legacy path).
    pub fn backend_kind(&self) -> Option<BackendKind> {
        self.backend
            .as_ref()
            .and_then(|arc| arc.lock().ok().map(|g| g.kind()))
    }
} // end of pub mod backend

// Tobii eye-tracker backend (EYE-SOTA-002). Gated behind the `tobii` feature
// at the GazeSource-FFI level; the Backend impl + synthetic source are always
// available so downstream callers (e.g. eyetracker-inference) can switch
// between webcam and Tobii via Camera::with_backend without conditional
// compilation in their own code.
pub mod backend_tobii;

// Pupil Labs eye-tracker backend (EYE-SOTA-003). Same pattern as Tobii:
// the Backend impl + SyntheticPupilSource compile in the default build;
// the real ZMQ stream is gated behind the `pupil` feature.
pub mod backend_pupil;

