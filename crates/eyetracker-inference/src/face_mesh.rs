//! Face mesh detection and landmark extraction
//!
//! Detects facial landmarks using ML models.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// A single facial landmark
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Landmark {
    /// X coordinate (normalized 0-1)
    pub x: f32,
    /// Y coordinate (normalized 0-1)
    pub y: f32,
    /// Z coordinate (depth, normalized 0-1)
    pub z: f32,
    /// Landmark type (MediaPipe landmark index)
    pub landmark_type: LandmarkType,
}

impl Landmark {
    /// Create a new landmark
    pub fn new(x: f32, y: f32, z: f32, landmark_type: LandmarkType) -> Self {
        Self {
            x,
            y,
            z,
            landmark_type,
        }
    }

    /// Convert to screen coordinates
    pub fn to_screen(&self, width: u32, height: u32) -> (f32, f32) {
        (self.x * width as f32, self.y * height as f32)
    }

    /// Distance to another landmark
    pub fn distance(&self, other: &Landmark) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

/// Types of facial landmarks (MediaPipe Face Mesh indices)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LandmarkType {
    // Face contour (0-16)
    Chin = 152,
    LeftJaw = 234,
    RightJaw = 454,
    // Eyes (17-26)
    LeftEyeOuter = 33,
    LeftEyeInner = 133,
    RightEyeOuter = 362,
    RightEyeInner = 263,
    LeftEyeTop = 159,
    LeftEyeBottom = 145,
    RightEyeTop = 386,
    RightEyeBottom = 374,
    // Eyebrows
    LeftEyebrowOuter = 70,
    LeftEyebrowInner = 107,
    LeftEyebrowTop = 65,
    RightEyebrowOuter = 300,
    RightEyebrowInner = 336,
    RightEyebrowTop = 293,
    // Nose
    NoseTip = 1,
    NoseBottom = 2,
    NoseLeft = 49,
    NoseRight = 275,
    // Mouth
    MouthLeft = 61,
    MouthRight = 291,
    MouthTop = 13,
    MouthBottom = 14,
    UpperLipTop = 12,
    UpperLipBottom = 15,
    LowerLipTop = 17,
    LowerLipBottom = 18,
    // Iris/Pupil (468-477)
    LeftPupil = 468,
    LeftIrisLeft = 469,
    LeftIrisTop = 470,
    LeftIrisRight = 471,
    LeftIrisBottom = 472,
    RightPupil = 473,
    RightIrisLeft = 474,
    RightIrisTop = 475,
    RightIrisRight = 476,
    RightIrisBottom = 477,
    // Center of face
    Forehead = 10,
    Unknown,
}

impl LandmarkType {
    /// Get MediaPipe index
    pub fn index(&self) -> usize {
        *self as usize
    }

    /// From MediaPipe landmark index
    pub fn from_index(index: usize) -> Option<Self> {
        match index {
            1 => Some(LandmarkType::NoseTip),
            10 => Some(LandmarkType::Forehead),
            33 => Some(LandmarkType::LeftEyeOuter),
            133 => Some(LandmarkType::LeftEyeInner),
            145 => Some(LandmarkType::LeftEyeBottom),
            152 => Some(LandmarkType::Chin),
            159 => Some(LandmarkType::LeftEyeTop),
            234 => Some(LandmarkType::LeftJaw),
            263 => Some(LandmarkType::RightEyeInner),
            275 => Some(LandmarkType::NoseRight),
            362 => Some(LandmarkType::RightEyeOuter),
            374 => Some(LandmarkType::RightEyeBottom),
            454 => Some(LandmarkType::RightJaw),
            468 => Some(LandmarkType::LeftPupil),
            469 => Some(LandmarkType::LeftIrisLeft),
            470 => Some(LandmarkType::LeftIrisTop),
            471 => Some(LandmarkType::LeftIrisRight),
            472 => Some(LandmarkType::LeftIrisBottom),
            473 => Some(LandmarkType::RightPupil),
            474 => Some(LandmarkType::RightIrisLeft),
            475 => Some(LandmarkType::RightIrisTop),
            476 => Some(LandmarkType::RightIrisRight),
            477 => Some(LandmarkType::RightIrisBottom),
            _ => Some(LandmarkType::Unknown),
        }
    }
}

/// Face mesh detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceMeshResult {
    /// All detected landmarks (MediaPipe: 478 points)
    pub landmarks: Vec<Landmark>,
    /// Detection confidence (0-1)
    pub confidence: f32,
    /// Face bounding box (normalized coordinates)
    pub face_box: FaceBox,
    /// Detection timestamp
    pub timestamp: std::time::SystemTime,
    /// Inference latency
    pub latency: Duration,
}

impl FaceMeshResult {
    /// Get landmark by type
    pub fn get(&self, landmark_type: LandmarkType) -> Option<&Landmark> {
        self.landmarks
            .get(landmark_type.index())
            .filter(|l| l.landmark_type == landmark_type)
    }

    /// Get left eye landmarks
    pub fn left_eye(&self) -> Vec<&Landmark> {
        let indices = [
            LandmarkType::LeftEyeOuter,
            LandmarkType::LeftEyeInner,
            LandmarkType::LeftEyeTop,
            LandmarkType::LeftEyeBottom,
        ];
        indices
            .iter()
            .filter_map(|t| self.get(*t))
            .collect()
    }

    /// Get right eye landmarks
    pub fn right_eye(&self) -> Vec<&Landmark> {
        let indices = [
            LandmarkType::RightEyeOuter,
            LandmarkType::RightEyeInner,
            LandmarkType::RightEyeTop,
            LandmarkType::RightEyeBottom,
        ];
        indices
            .iter()
            .filter_map(|t| self.get(*t))
            .collect()
    }

    /// Get eye centers (pupil positions)
    pub fn eye_centers(&self) -> Option<(Landmark, Landmark)> {
        let left = self.get(LandmarkType::LeftPupil)?;
        let right = self.get(LandmarkType::RightPupil)?;
        Some((*left, *right))
    }

    /// Calculate face yaw (left-right rotation)
    pub fn face_yaw(&self) -> f32 {
        // Use nose tip and eye centers to estimate yaw
        let Some(left_eye) = self.get(LandmarkType::LeftEyeOuter) else {
            return 0.0;
        };
        let Some(right_eye) = self.get(LandmarkType::RightEyeOuter) else {
            return 0.0;
        };
        let Some(nose) = self.get(LandmarkType::NoseTip) else {
            return 0.0;
        };

        // Horizontal distance between eyes
        let eye_dist = left_eye.distance(right_eye);
        if eye_dist < 0.01 {
            return 0.0;
        }

        // Offset of nose from eye midpoint
        let eye_mid_x = (left_eye.x + right_eye.x) / 2.0;
        let nose_offset = nose.x - eye_mid_x;

        // Convert to approximate yaw angle (simplified)
        nose_offset / (eye_dist / 2.0)
    }

    /// Calculate face pitch (up-down rotation)
    pub fn face_pitch(&self) -> f32 {
        let Some(nose_tip) = self.get(LandmarkType::NoseTip) else {
            return 0.0;
        };
        let Some(nose_bottom) = self.get(LandmarkType::NoseBottom) else {
            return 0.0;
        };
        let Some(forehead) = self.get(LandmarkType::Forehead) else {
            return 0.0;
        };

        // Vertical offset from expected position
        let expected_nose_y = (forehead.y + nose_bottom.y) / 2.0;
        let pitch_offset = nose_tip.y - expected_nose_y;

        // Normalize
        pitch_offset / 0.1 // Approximate
    }

    /// Calculate face roll (tilt)
    pub fn face_roll(&self) -> f32 {
        let Some(left_eye) = self.get(LandmarkType::LeftEyeOuter) else {
            return 0.0;
        };
        let Some(right_eye) = self.get(LandmarkType::RightEyeOuter) else {
            return 0.0;
        };

        // Angle between eyes
        let dy = right_eye.y - left_eye.y;
        let dx = right_eye.x - left_eye.x;

        dy.atan2(dx)
    }

    /// Get face pose as (yaw, pitch, roll) in radians
    pub fn face_pose(&self) -> (f32, f32, f32) {
        (self.face_yaw(), self.face_pitch(), self.face_roll())
    }

    /// Estimate interpupillary distance (IPD) in normalized coordinates
    pub fn ipd(&self) -> f32 {
        if let Some((left, right)) = self.eye_centers() {
            left.distance(&right)
        } else {
            0.06 // Default average IPD
        }
    }
}

/// Face bounding box
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FaceBox {
    /// Top-left X (normalized)
    pub x_min: f32,
    /// Top-left Y (normalized)
    pub y_min: f32,
    /// Bottom-right X (normalized)
    pub x_max: f32,
    /// Bottom-right Y (normalized)
    pub y_max: f32,
}

impl FaceBox {
    /// Create from min/max coordinates
    pub fn new(x_min: f32, y_min: f32, x_max: f32, y_max: f32) -> Self {
        Self {
            x_min,
            y_min,
            x_max,
            y_max,
        }
    }

    /// Create from center and size
    pub fn from_center(cx: f32, cy: f32, width: f32, height: f32) -> Self {
        Self {
            x_min: cx - width / 2.0,
            y_min: cy - height / 2.0,
            x_max: cx + width / 2.0,
            y_max: cy + height / 2.0,
        }
    }

    /// Get width
    pub fn width(&self) -> f32 {
        self.x_max - self.x_min
    }

    /// Get height
    pub fn height(&self) -> f32 {
        self.y_max - self.y_min
    }

    /// Get center
    pub fn center(&self) -> (f32, f32) {
        (
            (self.x_min + self.x_max) / 2.0,
            (self.y_min + self.y_max) / 2.0,
        )
    }

    /// Convert to pixel coordinates
    pub fn to_pixels(&self, width: u32, height: u32) -> (u32, u32, u32, u32) {
        (
            (self.x_min * width as f32) as u32,
            (self.y_min * height as f32) as u32,
            (self.x_max * width as f32) as u32,
            (self.y_max * height as f32) as u32,
        )
    }

    /// Check if a point is inside the box
    pub fn contains(&self, x: f32, y: f32) -> bool {
        x >= self.x_min && x <= self.x_max && y >= self.y_min && y <= self.y_max
    }
}

/// Face mesh detector
pub struct FaceMesh {
    _private: (),
}

impl FaceMesh {
    /// Create a new face mesh detector
    pub fn new() -> Self {
        Self { _private: () }
    }

    /// Detect face mesh from image pixels
    pub fn detect(&self, pixels: &[f32], width: u32, height: u32) -> Result<FaceMeshResult> {
        let start = Instant::now();

        // This is a placeholder - real implementation would:
        // 1. Preprocess image for the model
        // 2. Run ONNX inference
        // 3. Postprocess landmarks

        // For now, return a synthetic result for testing
        let landmarks = Self::generate_synthetic_landmarks(width, height);
        let face_box = FaceBox::from_center(0.5, 0.45, 0.4, 0.4);

        Ok(FaceMeshResult {
            landmarks,
            confidence: 0.95,
            face_box,
            timestamp: std::time::SystemTime::now(),
            latency: start.elapsed(),
        })
    }

    /// Generate synthetic landmarks for testing
    fn generate_synthetic_landmarks(width: u32, height: u32) -> Vec<Landmark> {
        let mut landmarks = Vec::with_capacity(478);

        for i in 0..478 {
            let lm_type = LandmarkType::from_index(i).unwrap_or(LandmarkType::Unknown);

            // Generate approximate positions based on landmark type
            let (x, y, z) = match lm_type {
                LandmarkType::NoseTip => (0.5, 0.45, 0.05),
                LandmarkType::LeftEyeInner => (0.42, 0.40, 0.02),
                LandmarkType::LeftEyeOuter => (0.38, 0.40, 0.02),
                LandmarkType::RightEyeInner => (0.58, 0.40, 0.02),
                LandmarkType::RightEyeOuter => (0.62, 0.40, 0.02),
                LandmarkType::LeftPupil => (0.42, 0.40, 0.03),
                LandmarkType::RightPupil => (0.58, 0.40, 0.03),
                _ => {
                    // Random but plausible position
                    let angle = i as f32 * 0.1;
                    let radius = if i < 17 { 0.2 } else if i < 70 { 0.15 } else { 0.25 };
                    (
                        0.5 + radius * angle.cos(),
                        0.45 + radius * angle.sin() * 0.5,
                        0.0,
                    )
                }
            };

            landmarks.push(Landmark::new(x, y, z, lm_type));
        }

        landmarks
    }
}

impl Default for FaceMesh {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_landmark_creation() {
        let lm = Landmark::new(0.5, 0.5, 0.0, LandmarkType::NoseTip);
        assert_eq!(lm.x, 0.5);
        assert_eq!(lm.landmark_type, LandmarkType::NoseTip);
    }

    #[test]
    fn test_landmark_to_screen() {
        let lm = Landmark::new(0.5, 0.5, 0.0, LandmarkType::NoseTip);
        let (x, y) = lm.to_screen(1920, 1080);
        assert_eq!(x, 960.0);
        assert_eq!(y, 540.0);
    }

    #[test]
    fn test_landmark_distance() {
        let a = Landmark::new(0.0, 0.0, 0.0, LandmarkType::NoseTip);
        let b = Landmark::new(0.3, 0.4, 0.0, LandmarkType::Forehead);
        let dist = a.distance(&b);
        assert!((dist - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_face_box() {
        let box_ = FaceBox::from_center(0.5, 0.5, 0.2, 0.3);
        assert_eq!(box_.width(), 0.2);
        assert_eq!(box_.height(), 0.3);
        assert!(box_.contains(0.5, 0.5));
        assert!(!box_.contains(0.9, 0.5));
    }

    #[test]
    fn test_face_mesh_detection() {
        let detector = FaceMesh::new();
        let pixels = vec![0.0f32; 256 * 256 * 3];
        let result = detector.detect(&pixels, 256, 256).unwrap();

        assert!(!result.landmarks.is_empty());
        assert!(result.confidence > 0.0);
    }

    #[test]
    fn test_eye_centers() {
        let detector = FaceMesh::new();
        let pixels = vec![0.0f32; 256 * 256 * 3];
        let result = detector.detect(&pixels, 256, 256).unwrap();

        let (left, right) = result.eye_centers().unwrap();
        assert!(left.x < right.x); // Left eye should be on left
    }
}
