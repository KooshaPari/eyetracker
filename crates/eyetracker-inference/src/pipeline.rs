//! Inference pipeline
//!
//! Orchestrates the complete eye tracking pipeline from frame to gaze.

use crate::face_mesh::{FaceMesh, FaceMeshResult};
use crate::gaze_estimator::{GazeEstimation, GazeEstimationResult, ScreenCalibration};
use crate::model::ModelConfig;
use crate::processing::PreprocessOptions;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Pipeline configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Face mesh model config
    pub face_mesh_config: ModelConfig,
    /// Preprocessing options
    pub preprocess_options: PreprocessOptions,
    /// Enable face mesh detection
    pub detect_face_mesh: bool,
    /// Enable gaze estimation
    pub estimate_gaze: bool,
    /// Minimum face confidence threshold
    pub min_face_confidence: f32,
    /// Enable head pose estimation
    pub estimate_head_pose: bool,
    /// Smoothing window size (0 = no smoothing)
    pub smoothing_window: usize,
    /// Output frame rate limit (0 = unlimited)
    pub max_fps: u32,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            face_mesh_config: ModelConfig::default(),
            preprocess_options: PreprocessOptions::default(),
            detect_face_mesh: true,
            estimate_gaze: true,
            min_face_confidence: 0.5,
            estimate_head_pose: true,
            smoothing_window: 5,
            max_fps: 0,
        }
    }
}

impl PipelineConfig {
    /// Create configuration optimized for real-time tracking
    pub fn real_time() -> Self {
        Self {
            face_mesh_config: ModelConfig {
                input_shape: vec![1, 192, 192, 3], // Smaller for speed
                timeout_ms: 50,                    // 50ms max inference
                ..Default::default()
            },
            smoothing_window: 3,
            max_fps: 30,
            ..Default::default()
        }
    }

    /// Create configuration optimized for accuracy
    pub fn high_accuracy() -> Self {
        Self {
            face_mesh_config: ModelConfig {
                input_shape: vec![1, 256, 256, 3],
                timeout_ms: 100,
                ..Default::default()
            },
            smoothing_window: 7,
            max_fps: 15,
            ..Default::default()
        }
    }
}

/// Complete inference result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResult {
    /// Face mesh detection result
    pub face_mesh: Option<FaceMeshResult>,
    /// Gaze estimation result
    pub gaze: Option<GazeEstimationResult>,
    /// Total pipeline latency
    pub total_latency: Duration,
    /// Timestamp
    pub timestamp: std::time::SystemTime,
    /// Whether face was detected
    pub face_detected: bool,
}

impl InferenceResult {
    /// Get screen position if available
    pub fn screen_position(&self) -> Option<(f32, f32)> {
        self.gaze.as_ref().map(|g| g.screen_position)
    }

    /// Get combined gaze direction if available
    pub fn combined_gaze(&self) -> Option<crate::gaze_estimator::GazeDirection> {
        self.gaze.as_ref().map(|g| g.combined)
    }
}

/// Inference pipeline
pub struct InferencePipeline {
    config: PipelineConfig,
    face_mesh: Arc<FaceMesh>,
    gaze_estimator: GazeEstimation,
    /// Gaze smoothing buffer
    smoothing_buffer: Vec<(f32, f32)>,
    last_frame_time: Option<Instant>,
}

impl InferencePipeline {
    /// Create a new inference pipeline
    pub fn new(config: PipelineConfig) -> Self {
        let face_mesh = FaceMesh::new();
        let gaze_estimator = GazeEstimation::default_config();

        Self {
            config,
            face_mesh: Arc::new(face_mesh),
            gaze_estimator,
            smoothing_buffer: Vec::new(),
            last_frame_time: None,
        }
    }

    /// Create with default configuration
    pub fn default_pipeline() -> Self {
        Self::new(PipelineConfig::default())
    }

    /// Create for real-time tracking
    pub fn real_time_pipeline() -> Self {
        Self::new(PipelineConfig::real_time())
    }

    /// Create for high accuracy
    pub fn high_accuracy_pipeline() -> Self {
        Self::new(PipelineConfig::high_accuracy())
    }

    /// Set screen calibration
    pub fn set_calibration(&mut self, calibration: ScreenCalibration) {
        self.gaze_estimator.set_calibration(calibration);
    }

    /// Update configuration
    pub fn update_config(&mut self, config: PipelineConfig) {
        self.config = config;
        self.smoothing_buffer.clear();
    }

    /// Process a frame and return inference result
    pub fn process_frame(&mut self, pixels: &[f32], width: u32, height: u32) -> InferenceResult {
        let start = Instant::now();
        let timestamp = std::time::SystemTime::now();

        // Rate limiting
        if let Some(last) = self.last_frame_time {
            if self.config.max_fps > 0 {
                let min_interval = Duration::from_secs_f64(1.0 / self.config.max_fps as f64);
                if last.elapsed() < min_interval {
                    return InferenceResult {
                        face_mesh: None,
                        gaze: None,
                        total_latency: start.elapsed(),
                        timestamp,
                        face_detected: false,
                    };
                }
            }
        }
        self.last_frame_time = Some(Instant::now());

        // Detect face mesh
        let face_mesh = if self.config.detect_face_mesh {
            match self.face_mesh.detect(pixels, width, height) {
                Ok(result) => Some(result),
                Err(e) => {
                    tracing::warn!("Face mesh detection failed: {}", e);
                    None
                }
            }
        } else {
            None
        };

        // Check face confidence
        let face_detected = face_mesh
            .as_ref()
            .map(|r| r.confidence >= self.config.min_face_confidence)
            .unwrap_or(false);

        // Estimate gaze
        let gaze = if face_detected && self.config.estimate_gaze {
            face_mesh.as_ref().map(|fm| self.gaze_estimator.estimate(fm))
        } else {
            None
        };

        // Apply smoothing
        let gaze = gaze.map(|mut g| {
            if let Some((x, y)) = self.smooth_gaze(g.screen_position) {
                g.screen_position = (x, y);
            }
            g
        });

        InferenceResult {
            face_mesh,
            gaze,
            total_latency: start.elapsed(),
            timestamp,
            face_detected,
        }
    }

    /// Smooth gaze using moving average
    fn smooth_gaze(&mut self, position: (f32, f32)) -> Option<(f32, f32)> {
        let window = self.config.smoothing_window;
        if window == 0 {
            return Some(position);
        }

        // Add new position
        self.smoothing_buffer.push(position);

        // Keep only recent positions
        while self.smoothing_buffer.len() > window {
            self.smoothing_buffer.remove(0);
        }

        // Calculate average
        if self.smoothing_buffer.is_empty() {
            return Some(position);
        }

        let sum_x: f32 = self.smoothing_buffer.iter().map(|(x, _)| x).sum();
        let sum_y: f32 = self.smoothing_buffer.iter().map(|(_, y)| y).sum();
        let count = self.smoothing_buffer.len() as f32;

        Some((sum_x / count, sum_y / count))
    }

    /// Reset smoothing buffer
    pub fn reset_smoothing(&mut self) {
        self.smoothing_buffer.clear();
    }

    /// Get current configuration
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }

    /// Check if pipeline is ready for processing
    pub fn is_ready(&self) -> bool {
        true // Would check model loading in real implementation
    }

    /// Get statistics
    pub fn stats(&self) -> PipelineStats {
        PipelineStats {
            smoothing_buffer_size: self.smoothing_buffer.len(),
            smoothing_window: self.config.smoothing_window,
            last_latency: self.last_frame_time.map(|t| t.elapsed()),
        }
    }
}

/// Pipeline statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStats {
    pub smoothing_buffer_size: usize,
    pub smoothing_window: usize,
    pub last_latency: Option<Duration>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_config_defaults() {
        let config = PipelineConfig::default();
        assert!(config.detect_face_mesh);
        assert!(config.estimate_gaze);
        assert_eq!(config.smoothing_window, 5);
    }

    #[test]
    fn test_pipeline_config_real_time() {
        let config = PipelineConfig::real_time();
        assert_eq!(config.smoothing_window, 3);
        assert_eq!(config.max_fps, 30);
    }

    #[test]
    fn test_inference_result() {
        let result = InferenceResult {
            face_mesh: None,
            gaze: None,
            total_latency: Duration::from_millis(10),
            timestamp: std::time::SystemTime::now(),
            face_detected: false,
        };

        assert!(result.screen_position().is_none());
        assert!(!result.face_detected);
    }

    #[test]
    fn test_pipeline_smoothing() {
        let mut pipeline = InferencePipeline::default_pipeline();

        // Add several positions
        pipeline.smooth_gaze((0.1, 0.2));
        pipeline.smooth_gaze((0.2, 0.3));
        pipeline.smooth_gaze((0.3, 0.4));

        // Should average
        let smoothed = pipeline.smooth_gaze((0.4, 0.5));
        assert!(smoothed.is_some());

        let (x, y) = smoothed.unwrap();
        assert!(x > 0.2 && x < 0.4);
    }

    #[test]
    fn test_pipeline_reset() {
        let mut pipeline = InferencePipeline::default_pipeline();
        pipeline.smooth_gaze((0.5, 0.5));
        pipeline.reset_smoothing();
        assert_eq!(pipeline.stats().smoothing_buffer_size, 0);
    }
}

// Re-export processing for convenience
pub use crate::processing::preprocess_frame;
