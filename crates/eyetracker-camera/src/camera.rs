//! Camera capture module
//!
//! Handles webcam enumeration, initialization, and frame streaming
//! using cross-platform video capture via ccap-rs.

use crate::Frame;
use anyhow::{anyhow, Result};
use ccap::{PixelFormat, Provider};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use thiserror::Error;

/// Camera errors
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

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

/// Camera configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraConfig {
    /// Target frame rate
    pub target_fps: u32,
    /// Frame resolution width
    pub width: u32,
    /// Frame resolution height
    pub height: u32,
    /// Camera index (if using index-based selection)
    pub camera_index: Option<usize>,
    /// Request low-light mode if available
    pub low_light_mode: bool,
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            target_fps: 60,
            width: 640,
            height: 480,
            camera_index: None,
            low_light_mode: false,
        }
    }
}

/// Camera resolution presets optimized for eye tracking
impl CameraConfig {
    /// Low resolution - faster processing
    pub fn low_res() -> Self {
        Self {
            target_fps: 120,
            width: 320,
            height: 240,
            ..Default::default()
        }
    }

    /// Medium resolution - balanced
    pub fn medium_res() -> Self {
        Self {
            target_fps: 60,
            width: 640,
            height: 480,
            ..Default::default()
        }
    }

    /// High resolution - more detail for pupil detection
    pub fn high_res() -> Self {
        Self {
            target_fps: 30,
            width: 1280,
            height: 720,
            ..Default::default()
        }
    }

    /// Preferred settings for eye tracking
    pub fn eye_tracking() -> Self {
        Self {
            target_fps: 60,
            width: 640,
            height: 480,
            low_light_mode: true,
            ..Default::default()
        }
    }
}

/// Available camera information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraInfo {
    pub index: usize,
    pub name: String,
    pub supported_formats: Vec<String>,
}

impl std::fmt::Display for CameraInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}", self.index, self.name)
    }
}

/// List available cameras
pub fn list_cameras() -> Result<Vec<CameraInfo>> {
    let mut cameras = Vec::new();

    // Query available devices by trying indices
    for index in 0..10 {
        match Provider::with_device(index) {
            Ok(provider) => {
                if let Ok(info) = provider.device_info() {
                    cameras.push(CameraInfo {
                        index: index as usize,
                        name: info.name,
                        supported_formats: info
                            .supported_pixel_formats
                            .iter()
                            .map(|f| f.as_str().to_string())
                            .collect(),
                    });
                }
            }
            Err(_) => continue,
        }
    }

    Ok(cameras)
}

/// Camera capture handle
pub struct Camera {
    provider: Arc<Mutex<Provider>>,
    config: CameraConfig,
    running: bool,
    frame_count: u64,
    start_time: Option<Instant>,
}

impl Camera {
    /// Create a new camera with the given configuration
    pub fn new(config: CameraConfig) -> Result<Self> {
        let camera_index = config.camera_index.unwrap_or(0);

        let mut provider = Provider::with_device(camera_index as i32)
            .map_err(|e| anyhow!("Failed to open camera: {}", e))?;

        // Set resolution
        provider
            .set_resolution(config.width, config.height)
            .map_err(|e| anyhow!("Failed to set resolution: {}", e))?;

        // Set frame rate
        provider
            .set_frame_rate(config.target_fps as f64)
            .map_err(|e| anyhow!("Failed to set frame rate: {}", e))?;

        // Set pixel format to RGB
        provider
            .set_pixel_format(PixelFormat::Rgb24)
            .map_err(|e| anyhow!("Failed to set pixel format: {}", e))?;

        tracing::info!(
            "Camera opened at index {} with resolution {}x{} @ {}fps",
            camera_index,
            config.width,
            config.height,
            config.target_fps
        );

        Ok(Self {
            provider: Arc::new(Mutex::new(provider)),
            config,
            running: false,
            frame_count: 0,
            start_time: None,
        })
    }

    /// Open the default camera with recommended settings for eye tracking
    pub fn open_default() -> Result<Self> {
        Self::new(CameraConfig::eye_tracking())
    }

    /// Start capturing frames
    pub fn start(&mut self) -> Result<()> {
        if self.running {
            return Ok(());
        }

        {
            let mut provider = self.provider.lock().unwrap();
            provider
                .start()
                .map_err(|e| anyhow!("Failed to start camera: {}", e))?;
        }

        self.running = true;
        self.start_time = Some(Instant::now());
        self.frame_count = 0;

        tracing::info!("Camera stream started");
        Ok(())
    }

    /// Stop capturing frames
    pub fn stop(&mut self) -> Result<()> {
        if !self.running {
            return Ok(());
        }

        {
            let mut provider = self.provider.lock().unwrap();
            provider
                .stop()
                .map_err(|e| anyhow!("Failed to stop camera: {}", e))?;
        }

        self.running = false;
        tracing::info!("Camera stream stopped");
        Ok(())
    }

    /// Capture a single frame (blocking)
    pub fn capture_frame(&mut self) -> Result<Frame> {
        if !self.running {
            return Err(CameraError::NotRunning.into());
        }

        let frame = {
            let mut provider = self.provider.lock().unwrap();
            provider
                .grab_frame(1000)
                .map_err(|e| anyhow!("Failed to capture frame: {}", e))?
        };

        let frame = match frame {
            Some(f) => f,
            None => return Err(CameraError::CaptureFailed("No frame available".into()).into()),
        };

        self.frame_count += 1;

        let width = frame.width();
        let height = frame.height();

        // Get frame data
        let data = frame
            .data()
            .map_err(|e| anyhow!("Failed to get frame data: {}", e))?
            .to_vec();

        let metadata = crate::FrameMetadata {
            timestamp: std::time::SystemTime::now(),
            frame_number: self.frame_count,
            fps: self.calculate_fps(),
            width,
            height,
        };

        Ok(Frame {
            data,
            width,
            height,
            format: crate::FrameFormat::Rgb8,
            metadata,
        })
    }

    /// Capture a frame with timeout
    pub async fn capture_frame_async(&mut self, timeout: Duration) -> Result<Frame> {
        if !self.running {
            return Err(CameraError::NotRunning.into());
        }

        tokio::time::timeout(timeout, async { self.capture_frame() }).await?
    }

    fn calculate_fps(&self) -> f64 {
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

    /// Get total frames captured
    pub fn frame_count(&self) -> u64 {
        self.frame_count
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
        let config = CameraConfig::default();
        assert_eq!(config.width, 640);
        assert_eq!(config.height, 480);
        assert_eq!(config.target_fps, 60);
    }

    #[test]
    fn test_camera_config_eye_tracking() {
        let config = CameraConfig::eye_tracking();
        assert_eq!(config.target_fps, 60);
        assert!(config.low_light_mode);
    }
}
