//! Face mesh detection module
//!
//! Detects facial landmarks (468-point face mesh) from a camera frame.
//! Supports ONNX-based models and provides a trait abstraction.

use eyetracker_camera::Frame;
use serde::{Deserialize, Serialize};

/// A single 2D landmark point
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Landmark2D {
    pub x: f32,
    pub y: f32,
}

/// A single 3D landmark point
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Landmark3D {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// Face bounding box
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FaceBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub confidence: f32,
}

impl FaceBox {
    pub fn center(&self) -> (f32, f32) {
        (self.x + self.width / 2.0, self.y + self.height / 2.0)
    }

    pub fn area(&self) -> f32 {
        self.width * self.height
    }

    /// Check if a point is inside this bounding box
    pub fn contains(&self, px: f32, py: f32) -> bool {
        px >= self.x && px <= self.x + self.width && py >= self.y && py <= self.y + self.height
    }
}

/// Full face detection result with landmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceResult {
    /// Bounding box of the face
    pub face_box: FaceBox,
    /// 468 facial landmarks (MediaPipe convention)
    pub landmarks: Vec<Landmark3D>,
    /// Eye landmarks (indices into the full landmarks vec)
    pub left_eye: EyeRegion,
    pub right_eye: EyeRegion,
    /// Overall detection confidence
    pub confidence: f32,
}

/// Eye region with key landmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EyeRegion {
    /// Indices into the full landmarks vec
    pub landmark_indices: Vec<usize>,
    /// Eye center (computed from landmarks)
    pub center: Landmark2D,
    /// Eye corners
    pub inner_corner: Landmark2D,
    pub outer_corner: Landmark2D,
    /// Pupil position (relative within eye region)
    pub pupil: Option<Landmark2D>,
}

/// Trait for face detection backends
pub trait FaceDetector: Send + Sync {
    /// Detect faces in a frame, returning the primary face result
    fn detect(&mut self, frame: &Frame) -> anyhow::Result<FaceResult>;

    /// Detect all faces in a frame
    fn detect_all(&mut self, frame: &Frame) -> anyhow::Result<Vec<FaceResult>>;

    /// Get the name of this detector implementation
    fn name(&self) -> &str;
}

/// MediaPipe-convention landmark indices for eyes
pub mod eye_indices {
    // Left eye landmarks (MediaPipe face mesh)
    pub const LEFT_EYE: &[usize] = &[
        33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
    ];
    pub const LEFT_EYE_INNER: usize = 133;
    pub const LEFT_EYE_OUTER: usize = 33;

    // Right eye landmarks
    pub const RIGHT_EYE: &[usize] = &[
        362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
    ];
    pub const RIGHT_EYE_INNER: usize = 362;
    pub const RIGHT_EYE_OUTER: usize = 263;

    // Iris landmarks (MediaPipe 468-point model with iris refinement)
    pub const LEFT_IRIS: usize = 468; // first iris landmark
    #[allow(dead_code)]
    pub const RIGHT_IRIS: usize = 473;
}

// Standard 468-point template for fallback face mesh creation
// These are simplified reference points for the face outline.
// Used by FR-EYE-CAL-005 multi-monitor calibration to anchor face region.
#[allow(dead_code)]
const FACE_OVAL_INDICES: &[usize] = &[
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
    148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
];

/// Create a placeholder face mesh (useful when no face detected)
pub fn create_placeholder_mesh() -> Vec<Landmark3D> {
    (0..468)
        .map(|i| {
            let angle = i as f32 * 2.0 * std::f32::consts::PI / 468.0;
            Landmark3D {
                x: 0.5 + 0.3 * angle.cos(),
                y: 0.5 + 0.4 * angle.sin(),
                z: 0.0,
            }
        })
        .collect()
}

/// Normalize landmarks relative to face bounding box
pub fn normalize_landmarks(
    landmarks: &[Landmark3D],
    face_box: &FaceBox,
    frame_width: f32,
    frame_height: f32,
) -> Vec<Landmark3D> {
    landmarks
        .iter()
        .map(|l| Landmark3D {
            x: (l.x * frame_width - face_box.x) / face_box.width,
            y: (l.y * frame_height - face_box.y) / face_box.height,
            z: l.z,
        })
        .collect()
}

/// Estimate eye region from face landmarks (MediaPipe 468 convention)
pub fn extract_eye_regions(landmarks: &[Landmark3D]) -> Option<(EyeRegion, EyeRegion)> {
    if landmarks.len() < 468 {
        return None;
    }

    let to_2d = |l: &Landmark3D| Landmark2D { x: l.x, y: l.y };

    // Left eye
    let left_center = Landmark2D {
        x: landmarks[eye_indices::LEFT_EYE_INNER].x
            + (landmarks[eye_indices::LEFT_EYE_OUTER].x - landmarks[eye_indices::LEFT_EYE_INNER].x)
                / 2.0,
        y: landmarks[eye_indices::LEFT_EYE_INNER].y
            + (landmarks[eye_indices::LEFT_EYE_OUTER].y - landmarks[eye_indices::LEFT_EYE_INNER].y)
                / 2.0,
    };
    let left = EyeRegion {
        landmark_indices: eye_indices::LEFT_EYE.to_vec(),
        center: left_center,
        inner_corner: to_2d(&landmarks[eye_indices::LEFT_EYE_INNER]),
        outer_corner: to_2d(&landmarks[eye_indices::LEFT_EYE_OUTER]),
        pupil: None,
    };

    // Right eye
    let right_center = Landmark2D {
        x: landmarks[eye_indices::RIGHT_EYE_INNER].x
            + (landmarks[eye_indices::RIGHT_EYE_OUTER].x
                - landmarks[eye_indices::RIGHT_EYE_INNER].x)
                / 2.0,
        y: landmarks[eye_indices::RIGHT_EYE_INNER].y
            + (landmarks[eye_indices::RIGHT_EYE_OUTER].y
                - landmarks[eye_indices::RIGHT_EYE_INNER].y)
                / 2.0,
    };
    let right = EyeRegion {
        landmark_indices: eye_indices::RIGHT_EYE.to_vec(),
        center: right_center,
        inner_corner: to_2d(&landmarks[eye_indices::RIGHT_EYE_INNER]),
        outer_corner: to_2d(&landmarks[eye_indices::RIGHT_EYE_OUTER]),
        pupil: None,
    };

    Some((left, right))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_face_box_center() {
        let fb = FaceBox {
            x: 0.2,
            y: 0.3,
            width: 0.5,
            height: 0.4,
            confidence: 0.95,
        };
        let (cx, cy) = fb.center();
        assert!((cx - 0.45).abs() < 0.001);
        assert!((cy - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_face_box_contains() {
        let fb = FaceBox {
            x: 10.0,
            y: 10.0,
            width: 100.0,
            height: 100.0,
            confidence: 1.0,
        };
        assert!(fb.contains(50.0, 50.0));
        assert!(!fb.contains(0.0, 0.0));
        assert!(!fb.contains(200.0, 200.0));
    }

    #[test]
    fn test_placeholder_mesh_length() {
        let mesh = create_placeholder_mesh();
        assert_eq!(mesh.len(), 468);
    }

    #[test]
    fn test_extract_eye_regions() {
        let mut landmarks = create_placeholder_mesh();
        // Position eyes approximately
        landmarks[33] = Landmark3D {
            x: 0.3,
            y: 0.4,
            z: 0.0,
        }; // left outer
        landmarks[133] = Landmark3D {
            x: 0.4,
            y: 0.4,
            z: 0.0,
        }; // left inner
        landmarks[362] = Landmark3D {
            x: 0.6,
            y: 0.4,
            z: 0.0,
        }; // right inner
        landmarks[263] = Landmark3D {
            x: 0.7,
            y: 0.4,
            z: 0.0,
        }; // right outer

        let result = extract_eye_regions(&landmarks);
        assert!(result.is_some());
        let (left, right) = result.unwrap();
        assert!((left.center.x - 0.35).abs() < 0.01);
        assert!((right.center.x - 0.65).abs() < 0.01);
    }

    #[test]
    fn test_landmark_normalization() {
        let landmarks = vec![
            Landmark3D {
                x: 0.5,
                y: 0.5,
                z: 0.0,
            },
            Landmark3D {
                x: 0.6,
                y: 0.6,
                z: 0.0,
            },
        ];
        let face_box = FaceBox {
            x: 100.0,
            y: 100.0,
            width: 200.0,
            height: 200.0,
            confidence: 1.0,
        };
        let normalized = normalize_landmarks(&landmarks, &face_box, 640.0, 480.0);
        assert!((normalized[0].x - 1.1).abs() < 0.01);
        assert!((normalized[0].y - 0.7).abs() < 0.01);
    }
}
