//! eyetracker-camera: Webcam capture abstraction
//!
//! Provides camera enumeration, frame capture, and configuration
//! via nokhwa (AVFoundation on macOS).

use anyhow::{anyhow, Result};
use nokhwa::{
    pixel_format::RgbFormat,
    utils::{
        ApiBackend, CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType,
        Resolution,
    },
    Camera as NokhwaCamera,
};
use serde::{Deserialize, Serialize};
use std::time::Instant;
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
        let camera = self.inner.as_mut().ok_or(CameraError::NotRunning)?;
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
