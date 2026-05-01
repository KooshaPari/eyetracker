//! Camera capture module
//!
//! Handles webcam enumeration, initialization, and frame streaming.

use anyhow::{anyhow, Result};
use nokhwa::prelude::*;
use nokhwa::utils::*;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::sync::mpsc;

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
    /// Camera ID (if using ID-based selection)
    pub camera_id: Option<String>,
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
            camera_id: None,
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
    pub id: String,
    pub supported_formats: Vec<Resolution>,
}

impl std::fmt::Display for CameraInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {} ({})", self.index, self.name, self.id)
    }
}

/// List available cameras
pub fn list_cameras() -> Result<Vec<CameraInfo>> {
    let mut cameras = Vec::new();

    // Query cameras
    match nokhwa::utils::Camera::camera_list() {
        Ok(list) => {
            for (index, cam_spec) in list.enumerate() {
                cameras.push(CameraInfo {
                    index,
                    name: cam_spec.human_name().to_string(),
                    id: cam_spec.id().to_string(),
                    supported_formats: cam_spec
                        .supported_resolution()
                        .iter()
                        .map(|r| Resolution(r.0, r.1))
                        .collect(),
                });
            }
        }
        Err(e) => {
            tracing::warn!("Failed to query cameras: {}", e);
        }
    }

    Ok(cameras)
}

/// Camera capture handle
pub struct Camera {
    inner: Option<nokhwa::Camera>,
    config: CameraConfig,
    running: bool,
    frame_count: u64,
    start_time: Option<Instant>,
}

impl Camera {
    /// Create a new camera with the given configuration
    pub fn new(config: CameraConfig) -> Result<Self> {
        let camera_index = config.camera_index.unwrap_or(0);

        let frame_spec = FrameSpec::from_res_and_fps(
            Resolution(config.width, config.height),
            FrameRate::from_fps(config.target_fps as u16),
        );

        let camera = nokhwa::Camera::new(
            CameraIndex::Index(camera_index as u32),
            Some(frame_spec),
        )
        .map_err(|e| anyhow!("Failed to open camera: {}", e))?;

        tracing::info!(
            "Camera opened at index {} with resolution {:?}",
            camera_index,
            frame_spec
        );

        Ok(Self {
            inner: Some(camera),
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

        if let Some(ref mut cam) = self.inner {
            cam.open_stream()
                .map_err(|e| anyhow!("Failed to start stream: {}", e))?;
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

        if let Some(ref mut cam) = self.inner {
            cam.stop_stream()
                .map_err(|e| anyhow!("Failed to stop stream: {}", e))?;
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

        let camera = self.inner.as_mut().ok_or_else(|| CameraError::NotRunning)?;

        let frame = camera
            .frame()
            .map_err(|e| anyhow!("Failed to capture frame: {}", e))?;

        self.frame_count += 1;

        let width = self.config.width;
        let height = self.config.height;

        // Convert buffer to raw bytes - use bytes() method
        let data = frame.bytes().to_vec();

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

        tokio::time::timeout(timeout, async {
            self.capture_frame()
        }).await?
    }

    /// Get a channel for frame streaming
    pub fn frame_channel(&mut self, buffer_size: usize) -> Result<mpsc::Receiver<Frame>> {
        let (tx, rx) = mpsc::channel(buffer_size);
        self.start()?;

        let width = self.config.width;
        let height = self.config.height;

        tokio::spawn(async move {
            let camera = nokhwa::Camera::new(
                CameraIndex::Index(0),
                Some(FrameSpec::from_res_and_fps(
                    Resolution(width, height),
                    FrameRate::from_fps(60),
                )),
            );

            if let Ok(mut cam) = camera {
                let _ = cam.open_stream();
                loop {
                    match cam.frame() {
                        Ok(frame) => {
                            let frame_data = frame.bytes().to_vec();
                            let frame = Frame {
                                data: frame_data,
                                width,
                                height,
                                format: crate::FrameFormat::Rgb8,
                                metadata: crate::FrameMetadata {
                                    timestamp: std::time::SystemTime::now(),
                                    frame_number: 0,
                                    fps: 0.0,
                                    width,
                                    height,
                                },
                            };

                            if tx.send(frame).await.is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            tracing::warn!("Frame capture error: {}", e);
                        }
                    }
                }
            }
        });

        Ok(rx)
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

    /// Get human-readable camera name
    pub fn name(&self) -> String {
        self.inner
            .as_ref()
            .map(|c| c.human_name())
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
