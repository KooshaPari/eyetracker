//! Camera capture module
//!
//! Handles webcam enumeration, initialization, and frame streaming.

use anyhow::{anyhow, Result};
use nokhwa::prelude::*;
use nokhwa::utils::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
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
            target_fps: 60, // High FPS for eye tracking
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
    /// Low resolution - faster processing, less detail
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
    let query = Query {
        // Query all camera backends
        ..Default::default()
    };

    let mut cameras = Vec::new();

    // Query cameras
    match query.format_list() {
        Ok(formats) => {
            for (index, format) in formats.iter().enumerate() {
                cameras.push(CameraInfo {
                    index,
                    name: format.human_name().to_string(),
                    id: format.id().to_string(),
                    supported_formats: format
                        .supported_resolutions()
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
    inner: Arc<nokhwa::Camera>,
    config: CameraConfig,
    running: bool,
    frame_count: u64,
    start_time: Option<Instant>,
}

impl Camera {
    /// Create a new camera with the given configuration
    pub fn new(config: CameraConfig) -> Result<Self> {
        // Determine which camera to open
        let camera_spec = if let Some(index) = config.camera_index {
            CameraIndex::Index(index.into())
        } else if let Some(ref id) = config.camera_id {
            CameraIndex::Index(id.parse().unwrap_or(0).into())
        } else {
            CameraIndex::Index(0.into())
        };

        // Create frame specification
        let frame_spec = FrameSpec::from_res_and_fps(
            Resolution(config.width, config.height),
            FrameRate::from_fps(config.target_fps as u16),
        );

        // Open camera
        let camera = nokhwa::Camera::new(camera_spec, Some(frame_spec))
            .map_err(|e| anyhow!("Failed to open camera: {}", e))?;

        tracing::info!(
            "Camera opened: {} at {:?}",
            camera.human_name(),
            frame_spec
        );

        Ok(Self {
            inner: Arc::new(camera),
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

        self.inner
            .start_streaming()
            .map_err(|e| anyhow!("Failed to start stream: {}", e))?;

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

        self.inner
            .stop_streaming()
            .map_err(|e| anyhow!("Failed to stop stream: {}", e))?;

        self.running = false;
        tracing::info!("Camera stream stopped");
        Ok(())
    }

    /// Capture a single frame (blocking)
    pub fn capture_frame(&mut self) -> Result<crate::Frame> {
        if !self.running {
            return Err(CameraError::NotRunning.into());
        }

        let frame = self.inner.frame()
            .map_err(|e| anyhow!("Failed to capture frame: {}", e))?;

        self.frame_count += 1;

        // Calculate metadata
        let metadata = crate::FrameMetadata {
            timestamp: std::time::SystemTime::now(),
            frame_number: self.frame_count,
            fps: self.calculate_fps(),
            width: frame.width(),
            height: frame.height(),
        };

        Ok(crate::Frame {
            data: frame.into_raw().to_vec(),
            width: frame.width() as u32,
            height: frame.height() as u32,
            format: crate::FrameFormat::Rgb8, // Most webcams output RGB
            metadata,
        })
    }

    /// Capture a frame with timeout
    pub async fn capture_frame_async(&mut self, timeout: Duration) -> Result<crate::Frame> {
        if !self.running {
            return Err(CameraError::NotRunning.into());
        }

        tokio::time::timeout(timeout, async {
            let frame = self.inner.frame()
                .map_err(|e| anyhow!("Failed to capture frame: {}", e))?;
            self.frame_count += 1;

            Ok(crate::Frame {
                data: frame.into_raw().to_vec(),
                width: frame.width() as u32,
                height: frame.height() as u32,
                format: crate::FrameFormat::Rgb8,
                metadata: crate::FrameMetadata {
                    timestamp: std::time::SystemTime::now(),
                    frame_number: self.frame_count,
                    fps: self.calculate_fps(),
                    width: frame.width() as u32,
                    height: frame.height() as u32,
                },
            })
        }).await?
    }

    /// Get a channel for frame streaming
    pub fn frame_channel(&mut self, buffer_size: usize) -> Result<mpsc::Receiver<crate::Frame>> {
        let (tx, rx) = mpsc::channel(buffer_size);

        // Start streaming in background
        self.start()?;

        let inner = Arc::clone(&self.inner);

        // Spawn frame capture task
        tokio::spawn(async move {
            loop {
                match inner.frame() {
                    Ok(frame) => {
                        let frame_data = crate::Frame {
                            data: frame.into_raw().to_vec(),
                            width: frame.width() as u32,
                            height: frame.height() as u32,
                            format: crate::FrameFormat::Rgb8,
                            metadata: crate::FrameMetadata {
                                timestamp: std::time::SystemTime::now(),
                                frame_number: 0,
                                fps: 0.0,
                                width: frame.width() as u32,
                                height: frame.height() as u32,
                            },
                        };

                        if tx.send(frame_data).await.is_err() {
                            break; // Receiver dropped
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Frame capture error: {}", e);
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
        self.inner.human_name()
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

    #[test]
    fn test_list_cameras() {
        let cameras = list_cameras().unwrap_or_default();
        println!("Found {} cameras", cameras.len());
        for cam in &cameras {
            println!("{}", cam);
        }
    }
}
