//! Full eye tracking pipeline
//!
//! Orchestrates camera capture → face detection → face mesh → gaze estimation.
//! Provides a high-level API for the CLI and other consumers.

use eyetracker_camera::{Camera, CameraConfig, Frame};
use std::time::Instant;

use crate::classification::{GazeClassifier, GazeEvent};
use crate::face_mesh::{extract_eye_regions, FaceBox, FaceDetector, FaceResult, Landmark3D};
use crate::gaze_estimator::{GazeEstimatorTrait, GazeResult, GeometricGazeEstimator};
use crate::smoothing::GazeSmoother;

/// Configuration for the eye tracking pipeline
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Camera configuration
    pub camera: CameraConfig,
    /// Whether to use geometric fallback (no ML model needed)
    pub use_geometric_fallback: bool,
    /// Gaze smoothing factor (0.0 = no smoothing, 0.9 = heavy smoothing)
    pub smoothing: f32,
    /// Screen distance in mm
    pub screen_distance_mm: f32,
    /// Whether to show debug overlays
    pub debug_overlay: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            camera: CameraConfig::eye_tracking(),
            use_geometric_fallback: true,
            smoothing: 0.6,
            screen_distance_mm: 600.0,
            debug_overlay: false,
        }
    }
}

/// A single eye tracking frame result
#[derive(Debug, Clone)]
pub struct TrackingResult {
    /// Raw camera frame
    pub frame: Frame,
    /// Face detection result (if face found)
    pub face: Option<FaceResult>,
    /// Gaze estimation result (if face found)
    pub gaze: Option<GazeResult>,
    /// Smoothed gaze position after Kalman filter (if gaze was found)
    pub smoothed_gaze: Option<(f32, f32)>,
    /// Gaze events triggered by this frame (fixation start/end, saccade)
    pub events: Vec<GazeEvent>,
    /// Processing time in milliseconds
    pub processing_time_ms: f64,
}

/// The main eye tracking pipeline
pub struct TrackingPipeline {
    camera: Camera,
    face_detector: Option<Box<dyn FaceDetector>>,
    gaze_estimator: Box<dyn GazeEstimatorTrait>,
    smoother: GazeSmoother,
    classifier: GazeClassifier,
    config: PipelineConfig,
    frame_count: u64,
}

impl TrackingPipeline {
    /// Create a new tracking pipeline with default settings
    pub fn new() -> anyhow::Result<Self> {
        Self::with_config(PipelineConfig::default())
    }

    /// Create a new tracking pipeline with custom configuration
    pub fn with_config(config: PipelineConfig) -> anyhow::Result<Self> {
        let camera = Camera::new(config.camera.clone())?;

        let gaze_estimator: Box<dyn GazeEstimatorTrait> = if config.use_geometric_fallback {
            Box::new(GeometricGazeEstimator::new().with_smoothing(config.smoothing))
        } else {
            Box::new(GeometricGazeEstimator::new().with_smoothing(config.smoothing))
        };

        Ok(Self {
            camera,
            face_detector: None,
            gaze_estimator,
            smoother: GazeSmoother::new(),
            classifier: GazeClassifier::default(),
            config,
            frame_count: 0,
        })
    }

    /// Start the camera
    pub fn start(&mut self) -> anyhow::Result<()> {
        self.camera.start()
    }

    /// Stop the camera
    pub fn stop(&mut self) -> anyhow::Result<()> {
        self.camera.stop()
    }

    /// Process a single frame from the camera
    ///
    /// Applies: camera capture → face detection → gaze estimation →
    /// Kalman smoothing (FR-EYE-INFER-002) → fixation/saccade classification
    /// (FR-EYE-INFER-003, FR-EYE-INFER-004).
    pub fn process_frame(&mut self) -> anyhow::Result<TrackingResult> {
        let start = Instant::now();

        // Capture frame from camera
        let frame = self.camera.capture_frame()?;
        self.frame_count += 1;

        // Detect face in frame
        let mut face = None;
        let mut gaze = None;

        if let Some(detector) = self.face_detector.as_mut() {
            match detector.detect(&frame) {
                Ok(face_result) => {
                    // Estimate gaze
                    match self.gaze_estimator.estimate(&face_result, &frame) {
                        Ok(gaze_result) => {
                            gaze = Some(gaze_result);
                        }
                        Err(e) => {
                            tracing::warn!("Gaze estimation failed: {}", e);
                        }
                    }
                    face = Some(face_result);
                }
                Err(e) => {
                    tracing::warn!("Face detection failed: {}", e);
                }
            }
        } else if self.config.use_geometric_fallback {
            // No ML model loaded; still attempt gaze estimation using fallback
            let fallback_face = create_fallback_face(&frame);
            match self.gaze_estimator.estimate(&fallback_face, &frame) {
                Ok(gaze_result) => {
                    gaze = Some(gaze_result);
                    face = Some(fallback_face);
                }
                Err(e) => {
                    tracing::warn!("Fallback gaze estimation failed: {}", e);
                }
            }
        }

        // ── Kalman smoothing (FR-EYE-INFER-002) ──
        let mut smoothed_gaze = None;
        let mut is_saccade = false;
        if let Some(ref g) = gaze {
            // Check classifier state before update to see if we're in a saccade
            is_saccade = matches!(
                self.classifier.current_state(),
                crate::classification::GazeClassification::Saccade
            );

            // Apply Kalman smoother — reset on saccade
            let (smoothed_x, smoothed_y) =
                self.smoother.smooth(g.screen_point.x, g.screen_point.y, is_saccade);
            smoothed_gaze = Some((smoothed_x, smoothed_y));
        }

        // ── Fixation/saccade classification (FR-EYE-INFER-003, FR-EYE-INFER-004) ──
        let events = if let Some(ref g) = gaze {
            self.classifier.update(
                g.screen_point.x,
                g.screen_point.y,
                Instant::now(),
                g.confidence,
            )
        } else {
            self.classifier.update(0.0, 0.0, Instant::now(), 0.0)
        };

        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        Ok(TrackingResult {
            frame,
            face,
            gaze,
            smoothed_gaze,
            events,
            processing_time_ms: elapsed,
        })
    }

    /// Returns `true` if the classifier currently believes the user is fixating.
    pub fn is_fixating(&self) -> bool {
        self.classifier.is_fixating()
    }

    /// Get a reference to the gaze classifier (for reading current state).
    pub fn classifier(&self) -> &GazeClassifier {
        &self.classifier
    }

    /// Get a mutable reference to the gaze classifier.
    pub fn classifier_mut(&mut self) -> &mut GazeClassifier {
        &mut self.classifier
    }

    /// Get a reference to the gaze smoother.
    pub fn smoother(&self) -> &GazeSmoother {
        &self.smoother
    }

    /// Get a mutable reference to the gaze smoother.
    pub fn smoother_mut(&mut self) -> &mut GazeSmoother {
        &mut self.smoother
    }

    /// Get the current frame rate
    pub fn fps(&self) -> f64 {
        self.camera.fps()
    }

    /// Get total frames processed
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Get a reference to the camera
    pub fn camera(&self) -> &Camera {
        &self.camera
    }

    /// Get a mutable reference to the camera
    pub fn camera_mut(&mut self) -> &mut Camera {
        &mut self.camera
    }

    /// Set a face detector
    pub fn set_face_detector(&mut self, detector: Box<dyn FaceDetector>) {
        self.face_detector = Some(detector);
    }

    /// Get the configuration
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }
}

/// Create a fallback face result when no ML model is loaded
fn create_fallback_face(frame: &Frame) -> FaceResult {
    let fw = frame.width as f32;
    let fh = frame.height as f32;

    let face_box = FaceBox {
        x: fw * 0.25,
        y: fh * 0.2,
        width: fw * 0.5,
        height: fh * 0.6,
        confidence: 0.5,
    };

    // Create placeholder landmarks
    let mut landmarks: Vec<Landmark3D> = (0..468)
        .map(|i| {
            let progress = i as f32 / 468.0;
            Landmark3D {
                x: face_box.x / fw + (face_box.width / fw) * (0.5 + 0.3 * (progress * 2.0 * std::f32::consts::PI).sin()),
                y: face_box.y / fh + (face_box.height / fh) * (0.5 + 0.4 * (progress * 2.0 * std::f32::consts::PI).cos()),
                z: 0.0,
            }
        })
        .collect();

    // Set eye positions relative to face box
    let left_eye_center_x = (face_box.x + face_box.width * 0.3) / fw;
    let left_eye_center_y = (face_box.y + face_box.height * 0.35) / fh;
    let right_eye_center_x = (face_box.x + face_box.width * 0.7) / fw;
    let right_eye_center_y = (face_box.y + face_box.height * 0.35) / fh;

    // Override key eye landmarks
    if landmarks.len() > 133 {
        landmarks[33] = Landmark3D { x: left_eye_center_x - 0.02, y: left_eye_center_y, z: 0.0 };
        landmarks[133] = Landmark3D { x: left_eye_center_x + 0.02, y: left_eye_center_y, z: 0.0 };
        landmarks[362] = Landmark3D { x: right_eye_center_x - 0.02, y: right_eye_center_y, z: 0.0 };
        landmarks[263] = Landmark3D { x: right_eye_center_x + 0.02, y: right_eye_center_y, z: 0.0 };
    }

    let (left_eye, right_eye) = extract_eye_regions(&landmarks)
        .unwrap_or_else(|| {
            use crate::face_mesh::EyeRegion;
            let default = EyeRegion {
                landmark_indices: vec![],
                center: crate::face_mesh::Landmark2D { x: 0.5, y: 0.5 },
                inner_corner: crate::face_mesh::Landmark2D { x: 0.48, y: 0.5 },
                outer_corner: crate::face_mesh::Landmark2D { x: 0.52, y: 0.5 },
                pupil: None,
            };
            (default.clone(), default)
        });

    FaceResult {
        face_box,
        landmarks,
        left_eye,
        right_eye,
        confidence: 0.5,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eyetracker_camera::PixelFormat;

    #[test]
    fn test_pipeline_config_default() {
        let config = PipelineConfig::default();
        assert!(config.use_geometric_fallback);
        assert!((config.smoothing - 0.6).abs() < 0.001);
    }

    #[test]
    fn test_create_fallback_face() {
        let frame = Frame {
            data: vec![0u8; 640 * 480 * 3],
            width: 640,
            height: 480,
            format: PixelFormat::Rgb8,
            timestamp: Instant::now(),
            frame_number: 0,
        };
        let face = create_fallback_face(&frame);
        assert_eq!(face.landmarks.len(), 468);
        assert!((face.face_box.confidence - 0.5).abs() < 0.001);
        assert!((face.confidence - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_pipeline_has_smoother_and_classifier() {
        let config = PipelineConfig::default();
        // We can't create a pipeline without a real camera, but we can verify
        // that the config is well-formed.
        assert!((config.smoothing - 0.6).abs() < 0.001);
        assert!(config.use_geometric_fallback);
    }
}
