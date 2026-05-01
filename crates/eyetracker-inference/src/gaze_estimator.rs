//! Gaze estimation module
//!
//! Estimates gaze direction and screen position from face landmarks.

use crate::face_mesh::{FaceMeshResult, FaceBox, Landmark, LandmarkType};
use crate::model::{ModelConfig, ModelType};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Gaze direction in normalized screen coordinates
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GazeDirection {
    /// Horizontal gaze offset (-1 to 1, where 0 is center)
    pub x: f32,
    /// Vertical gaze offset (-1 to 1, where 0 is center)
    pub y: f32,
    /// Gaze confidence (0-1)
    pub confidence: f32,
    /// Whether gaze is estimated for left eye
    pub is_left_eye: bool,
}

impl GazeDirection {
    /// Create a new gaze direction
    pub fn new(x: f32, y: f32, confidence: f32, is_left_eye: bool) -> Self {
        Self {
            x: x.clamp(-1.0, 1.0),
            y: y.clamp(-1.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
            is_left_eye,
        }
    }

    /// Get the magnitude of gaze offset
    pub fn magnitude(&self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    /// Check if gaze is looking straight ahead
    pub fn is_center(&self, threshold: f32) -> bool {
        self.magnitude() < threshold
    }

    /// Blend two gaze directions
    pub fn blend(&self, other: &GazeDirection, weight: f32) -> GazeDirection {
        GazeDirection::new(
            self.x * (1.0 - weight) + other.x * weight,
            self.y * (1.0 - weight) + other.y * weight,
            (self.confidence + other.confidence) / 2.0,
            false, // Blended result isn't from a specific eye
        )
    }
}

/// Gaze estimation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GazeEstimationResult {
    /// Left eye gaze direction
    pub left_gaze: GazeDirection,
    /// Right eye gaze direction
    pub right_gaze: GazeDirection,
    /// Combined/interpolated gaze direction
    pub combined: GazeDirection,
    /// Estimated screen position (normalized 0-1)
    pub screen_position: (f32, f32),
    /// Face box at time of estimation
    pub face_box: FaceBox,
    /// Interpupillary distance
    pub ipd: f32,
    /// Head pose (yaw, pitch, roll)
    pub head_pose: (f32, f32, f32),
    /// Inference latency
    pub latency: Duration,
    /// Timestamp
    pub timestamp: std::time::SystemTime,
}

impl GazeEstimationResult {
    /// Get the average gaze direction
    pub fn average_gaze(&self) -> GazeDirection {
        self.left_gaze.blend(&self.right_gaze, 0.5)
    }

    /// Get weighted gaze based on confidence
    pub fn weighted_gaze(&self) -> GazeDirection {
        let total_conf = self.left_gaze.confidence + self.right_gaze.confidence;
        if total_conf < 0.01 {
            return self.combined;
        }

        let left_weight = self.left_gaze.confidence / total_conf;
        GazeDirection::new(
            self.left_gaze.x * left_weight + self.right_gaze.x * (1.0 - left_weight),
            self.left_gaze.y * left_weight + self.right_gaze.y * (1.0 - left_weight),
            (self.left_gaze.confidence + self.right_gaze.confidence) / 2.0,
            false,
        )
    }

    /// Convert screen position to pixels
    pub fn to_screen_pixels(&self, screen_width: u32, screen_height: u32) -> (f32, f32) {
        (
            self.screen_position.0 * screen_width as f32,
            self.screen_position.1 * screen_height as f32,
        )
    }

    /// Check if estimation is valid (good confidence)
    pub fn is_valid(&self, min_confidence: f32) -> bool {
        self.left_gaze.confidence >= min_confidence
            && self.right_gaze.confidence >= min_confidence
    }
}

/// Gaze estimator using face mesh landmarks
pub struct GazeEstimation {
    config: ModelConfig,
    /// Screen/display calibration
    screen_calibration: Option<ScreenCalibration>,
}

impl GazeEstimation {
    /// Create a new gaze estimator
    pub fn new(config: ModelConfig) -> Self {
        Self {
            config,
            screen_calibration: None,
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(ModelConfig::default())
    }

    /// Set screen calibration for accurate gaze mapping
    pub fn set_calibration(&mut self, calibration: ScreenCalibration) {
        self.screen_calibration = Some(calibration);
    }

    /// Clear screen calibration
    pub fn clear_calibration(&mut self) {
        self.screen_calibration = None;
    }

    /// Estimate gaze from face mesh result
    pub fn estimate(&self, face_mesh: &FaceMeshResult) -> GazeEstimationResult {
        let start = Instant::now();

        // Extract eye landmarks
        let left_gaze = self.estimate_eye_gaze(face_mesh, true);
        let right_gaze = self.estimate_eye_gaze(face_mesh, false);

        // Combine gaze directions
        let combined = self.combine_gaze(&left_gaze, &right_gaze);

        // Calculate screen position
        let screen_position = self.gaze_to_screen(&combined, face_mesh);

        let result = GazeEstimationResult {
            left_gaze,
            right_gaze,
            combined,
            screen_position,
            face_box: face_mesh.face_box,
            ipd: face_mesh.ipd(),
            head_pose: face_mesh.face_pose(),
            latency: start.elapsed(),
            timestamp: face_mesh.timestamp,
        };

        tracing::debug!(
            "Gaze estimated: ({:.2}, {:.2}) conf={:.2} latency={:?}",
            result.screen_position.0,
            result.screen_position.1,
            result.combined.confidence,
            result.latency
        );

        result
    }

    /// Estimate gaze for a single eye
    fn estimate_eye_gaze(&self, face_mesh: &FaceMeshResult, is_left: bool) -> GazeDirection {
        // Get relevant landmarks for eye
        let eye_landmarks = if is_left {
            face_mesh.left_eye()
        } else {
            face_mesh.right_eye()
        };

        if eye_landmarks.len() < 3 {
            return GazeDirection::new(0.0, 0.0, 0.0, is_left);
        }

        // Estimate pupil center from iris landmarks
        let (pupil_x, pupil_y) = if is_left {
            face_mesh
                .get(LandmarkType::LeftPupil)
                .map(|l| (l.x, l.y))
                .unwrap_or((0.42, 0.40))
        } else {
            face_mesh
                .get(LandmarkType::RightPupil)
                .map(|l| (l.x, l.y))
                .unwrap_or((0.58, 0.40))
        };

        // Get eye corner reference points
        let (inner_corner, outer_corner) = if is_left {
            (
                face_mesh.get(LandmarkType::LeftEyeInner),
                face_mesh.get(LandmarkType::LeftEyeOuter),
            )
        } else {
            (
                face_mesh.get(LandmarkType::RightEyeInner),
                face_mesh.get(LandmarkType::RightEyeOuter),
            )
        };

        let (inner_x, outer_x) = match (inner_corner, outer_corner) {
            (Some(inner), Some(outer)) => {
                if is_left {
                    (inner.x, outer.x)
                } else {
                    (outer.x, inner.x) // Swap for right eye
                }
            }
            _ => {
                if is_left {
                    (0.40, 0.36)
                } else {
                    (0.64, 0.60)
                }
            }
        };

        // Calculate gaze offset
        // Map pupil position relative to eye corners
        let eye_width = (outer_x - inner_x).abs();
        let pupil_center = (inner_x + outer_x) / 2.0;
        let gaze_offset = if eye_width > 0.001 {
            (pupil_x - pupil_center) / eye_width
        } else {
            0.0
        };

        // Vertical gaze (from upper/lower lid)
        let (top_lm, bottom_lm) = if is_left {
            (
                face_mesh.get(LandmarkType::LeftEyeTop),
                face_mesh.get(LandmarkType::LeftEyeBottom),
            )
        } else {
            (
                face_mesh.get(LandmarkType::RightEyeTop),
                face_mesh.get(LandmarkType::RightEyeBottom),
            )
        };

        let gaze_y = match (top_lm, bottom_lm) {
            (Some(top), Some(bottom)) => {
                let eye_height = bottom.y - top.y;
                let pupil_offset = pupil_y - (top.y + bottom.y) / 2.0;
                if eye_height > 0.001 {
                    pupil_offset / (eye_height / 2.0)
                } else {
                    0.0
                }
            }
            _ => 0.0,
        };

        // Adjust for head pose
        let (yaw, pitch, _) = face_mesh.face_pose();
        let adjusted_x = gaze_offset - yaw * 0.5;
        let adjusted_y = gaze_y - pitch * 0.3;

        GazeDirection::new(
            adjusted_x.clamp(-1.0, 1.0),
            adjusted_y.clamp(-1.0, 1.0),
            face_mesh.confidence,
            is_left,
        )
    }

    /// Combine left and right eye gaze
    fn combine_gaze(&self, left: &GazeDirection, right: &GazeDirection) -> GazeDirection {
        // Weight by confidence
        let total_conf = left.confidence + right.confidence;
        if total_conf < 0.01 {
            return GazeDirection::new(0.0, 0.0, 0.0, false);
        }

        let left_weight = left.confidence / total_conf;
        let right_weight = right.confidence / total_conf;

        GazeDirection::new(
            left.x * left_weight + right.x * right_weight,
            left.y * left_weight + right.y * right_weight,
            (left.confidence + right.confidence) / 2.0,
            false,
        )
    }

    /// Convert gaze direction to screen position
    fn gaze_to_screen(
        &self,
        gaze: &GazeDirection,
        face_mesh: &FaceMeshResult,
    ) -> (f32, f32) {
        // Base position (face center)
        let (face_cx, face_cy) = face_mesh.face_box.center();

        // Scale factors based on calibration
        let (scale_x, scale_y) = if let Some(ref cal) = self.screen_calibration {
            (cal.horizontal_scale, cal.vertical_scale)
        } else {
            (0.8, 0.6) // Default scaling
        };

        // Offset from gaze direction
        let offset_x = gaze.x * scale_x;
        let offset_y = gaze.y * scale_y;

        // Combine face position with gaze offset
        let screen_x = (face_cx + offset_x * face_mesh.ipd() * 5.0).clamp(0.0, 1.0);
        let screen_y = (face_cy - offset_y * face_mesh.ipd() * 3.0).clamp(0.0, 1.0);

        (screen_x, screen_y)
    }
}

impl Default for GazeEstimation {
    fn default() -> Self {
        Self::default_config()
    }
}

/// Screen calibration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScreenCalibration {
    /// Horizontal scale factor
    pub horizontal_scale: f32,
    /// Vertical scale factor
    pub vertical_scale: f32,
    /// Horizontal offset
    pub horizontal_offset: f32,
    /// Vertical offset
    pub vertical_offset: f32,
    /// Reference IPD (interpupillary distance)
    pub reference_ipd: f32,
    /// Screen distance (meters)
    pub screen_distance: f32,
    /// Screen width (meters)
    pub screen_width: f32,
    /// Screen height (meters)
    pub screen_height: f32,
}

impl Default for ScreenCalibration {
    fn default() -> Self {
        Self {
            horizontal_scale: 0.8,
            vertical_scale: 0.6,
            horizontal_offset: 0.0,
            vertical_offset: 0.0,
            reference_ipd: 0.063, // Average IPD: 63mm
            screen_distance: 0.6,   // 60cm from screen
            screen_width: 0.6,     // 60cm wide (27" monitor)
            screen_height: 0.34,   // 34cm tall
        }
    }
}

impl ScreenCalibration {
    /// Create for a specific screen configuration
    pub fn for_screen(width_m: f32, height_m: f32, distance_m: f32) -> Self {
        let aspect = width_m / height_m;
        Self {
            horizontal_scale: 1.0 / aspect,
            vertical_scale: 1.0,
            screen_width: width_m,
            screen_height: height_m,
            screen_distance: distance_m,
            ..Default::default()
        }
    }

    /// Create for a standard 27" monitor at 60cm
    pub fn standard_27_inch() -> Self {
        Self::for_screen(0.597, 0.336, 0.6)
    }

    /// Create for a laptop screen (15")
    pub fn laptop_15_inch() -> Self {
        Self::for_screen(0.344, 0.215, 0.5)
    }

    /// Apply calibration to gaze
    pub fn transform(&self, gaze: (f32, f32)) -> (f32, f32) {
        let x = (gaze.0 - 0.5) * self.horizontal_scale + 0.5 + self.horizontal_offset;
        let y = (gaze.1 - 0.5) * self.vertical_scale + 0.5 + self.vertical_offset;
        (x.clamp(0.0, 1.0), y.clamp(0.0, 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaze_direction() {
        let gaze = GazeDirection::new(0.5, -0.3, 0.9, true);
        assert_eq!(gaze.x, 0.5);
        assert_eq!(gaze.y, -0.3);
        assert!(gaze.is_left_eye);
    }

    #[test]
    fn test_gaze_magnitude() {
        let gaze = GazeDirection::new(0.6, 0.8, 0.9, false);
        let mag = gaze.magnitude();
        assert!((mag - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_gaze_blend() {
        let a = GazeDirection::new(0.0, 0.0, 1.0, true);
        let b = GazeDirection::new(1.0, 1.0, 1.0, false);
        let blended = a.blend(&b, 0.5);
        assert_eq!(blended.x, 0.5);
        assert_eq!(blended.y, 0.5);
    }

    #[test]
    fn test_screen_calibration() {
        let cal = ScreenCalibration::standard_27_inch();
        assert_eq!(cal.screen_width, 0.597);

        let transformed = cal.transform((0.5, 0.5));
        assert_eq!(transformed.0, 0.5);
        assert_eq!(transformed.1, 0.5);
    }

    #[test]
    fn test_gaze_estimation_default() {
        let estimator = GazeEstimation::default_config();
        assert!(estimator.screen_calibration.is_none());
    }

    #[test]
    fn test_gaze_estimation_with_calibration() {
        let mut estimator = GazeEstimation::default_config();
        let cal = ScreenCalibration::standard_27_inch();
        estimator.set_calibration(cal);
        assert!(estimator.screen_calibration.is_some());
    }
}
