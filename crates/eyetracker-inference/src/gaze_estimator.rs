//! Gaze direction estimation module
//!
//! Estimates gaze direction from eye landmarks and eye region images.
//! Uses geometric methods (pupil center cornea reflection) with ONNX fallback.

use eyetracker_camera::Frame;
use serde::{Deserialize, Serialize};

use crate::face_mesh::{FaceResult, Landmark2D};

/// 2D point with f32 precision
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Point2D {
    pub x: f32,
    pub y: f32,
}

impl Point2D {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

/// 3D vector with f32 precision (for gaze direction)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Vector3D {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vector3D {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
}

/// Gaze direction vector in 3D space
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GazeVector {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl GazeVector {
    /// Create a normalized gaze vector
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        let mut v = Self { x, y, z };
        v.normalize();
        v
    }

    /// Normalize the vector to unit length
    pub fn normalize(&mut self) {
        let len = (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        if len > 0.0 {
            self.x /= len;
            self.y /= len;
            self.z /= len;
        }
    }

    /// Get the normalized version
    pub fn normalized(&self) -> Self {
        let mut v = *self;
        v.normalize();
        v
    }

    /// Convert to 2D screen position (intersection with z=0 plane at given distance)
    pub fn screen_position(&self, distance: f32) -> Point2D {
        if self.z.abs() < 0.001 {
            return Point2D::new(0.0, 0.0);
        }
        let scale = distance / self.z;
        Point2D::new(self.x * scale, self.y * scale)
    }

    /// Angle in radians from forward (z) axis
    pub fn angle_from_forward(&self) -> f32 {
        let horizontal = (self.x * self.x + self.z * self.z).sqrt();
        self.y.atan2(horizontal)
    }

    /// Horizontal angle in radians (positive = right)
    pub fn horizontal_angle(&self) -> f32 {
        self.x.atan2(self.z)
    }

    /// Vertical angle in radians (positive = down)
    pub fn vertical_angle(&self) -> f32 {
        self.y.atan2(self.z)
    }

    /// Magnitude (length) of the gaze vector
    pub fn magnitude(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
}

impl From<GazeVector> for Vector3D {
    fn from(g: GazeVector) -> Self {
        Vector3D::new(g.x, g.y, g.z)
    }
}

/// Combined gaze result for both eyes
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GazeResult {
    /// Left eye gaze vector
    pub left: GazeVector,
    /// Right eye gaze vector
    pub right: GazeVector,
    /// Combined (average) gaze vector
    pub combined: GazeVector,
    /// Confidence of estimation (0.0 - 1.0)
    pub confidence: f32,
    /// Screen-space gaze point (pixels, relative to frame center)
    pub screen_point: Point2D,
}

/// Trait for gaze estimation backends
pub trait GazeEstimatorTrait: Send + Sync {
    /// Estimate gaze from a face result
    fn estimate(&mut self, face: &FaceResult, frame: &Frame) -> anyhow::Result<GazeResult>;

    /// Get the name of this estimator
    fn name(&self) -> &str;
}

/// Geometric gaze estimator using eye landmarks
///
/// Estimates gaze direction based on the position of the pupil relative
/// to the eye corners. This is a simplified geometric approach that works
/// without a dedicated ML model.
pub struct GeometricGazeEstimator {
    /// Smoothing factor (0.0 - 1.0, higher = smoother)
    smoothing: f32,
    /// Previous gaze result for smoothing
    previous: Option<GazeResult>,
    /// Screen distance in mm (for converting to screen coordinates)
    screen_distance_mm: f32,
}

impl GeometricGazeEstimator {
    pub fn new() -> Self {
        Self {
            smoothing: 0.6,
            previous: None,
            screen_distance_mm: 600.0,
        }
    }

    pub fn with_smoothing(mut self, smoothing: f32) -> Self {
        self.smoothing = smoothing.clamp(0.0, 0.95);
        self
    }

    fn estimate_eye_gaze(
        &self,
        eye_center: &Landmark2D,
        eye_inner: &Landmark2D,
        eye_outer: &Landmark2D,
        pupil: &Option<Landmark2D>,
        eye_width: f32,
    ) -> GazeVector {
        let eye_cx = eye_center.x;
        let eye_cy = eye_center.y;

        let (pupil_x, pupil_y) = if let Some(p) = pupil {
            (p.x, p.y)
        } else {
            (
                eye_cx + (eye_outer.x - eye_inner.x) * 0.02,
                eye_cy + eye_width * 0.01,
            )
        };

        let gaze_x = (pupil_x - eye_cx) / eye_width.max(1.0);
        let gaze_y = (pupil_y - eye_cy) / eye_width.max(1.0);

        GazeVector::new(gaze_x * 0.5, gaze_y * 0.5, 1.0)
    }
}

impl Default for GeometricGazeEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl GazeEstimatorTrait for GeometricGazeEstimator {
    fn estimate(&mut self, face: &FaceResult, frame: &Frame) -> anyhow::Result<GazeResult> {
        let eye_width = face.face_box.width * 0.1;

        let left = self.estimate_eye_gaze(
            &face.left_eye.center,
            &face.left_eye.inner_corner,
            &face.left_eye.outer_corner,
            &face.left_eye.pupil,
            eye_width,
        );

        let right = self.estimate_eye_gaze(
            &face.right_eye.center,
            &face.right_eye.inner_corner,
            &face.right_eye.outer_corner,
            &face.right_eye.pupil,
            eye_width,
        );

        let combined = GazeVector::new(
            (left.x + right.x) / 2.0,
            (left.y + right.y) / 2.0,
            (left.z + right.z) / 2.0,
        );

        let confidence = face.confidence * 0.8;

        let screen_center_x = frame.width as f32 / 2.0;
        let screen_center_y = frame.height as f32 / 2.0;
        let screen_dist = self.screen_distance_mm * 0.5;
        let screen_point = Point2D::new(
            screen_center_x + combined.x * screen_dist,
            screen_center_y + combined.y * screen_dist,
        );

        let result = GazeResult {
            left,
            right,
            combined,
            confidence,
            screen_point,
        };

        let result = if let Some(prev) = self.previous {
            GazeResult {
                left: smooth_gaze(&prev.left, &result.left, self.smoothing),
                right: smooth_gaze(&prev.right, &result.right, self.smoothing),
                combined: smooth_gaze(&prev.combined, &result.combined, self.smoothing),
                confidence: result.confidence,
                screen_point: Point2D::new(
                    prev.screen_point.x * self.smoothing
                        + result.screen_point.x * (1.0 - self.smoothing),
                    prev.screen_point.y * self.smoothing
                        + result.screen_point.y * (1.0 - self.smoothing),
                ),
            }
        } else {
            result
        };

        self.previous = Some(result);
        Ok(result)
    }

    fn name(&self) -> &str {
        "geometric"
    }
}

fn smooth_gaze(prev: &GazeVector, current: &GazeVector, factor: f32) -> GazeVector {
    GazeVector {
        x: prev.x * factor + current.x * (1.0 - factor),
        y: prev.y * factor + current.y * (1.0 - factor),
        z: prev.z * factor + current.z * (1.0 - factor),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaze_vector_normalization() {
        let mut gv = GazeVector::new(3.0, 4.0, 0.0);
        gv.normalize();
        assert!((gv.magnitude() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_gaze_vector_to_screen() {
        let gv = GazeVector::new(0.1, 0.05, 1.0);
        let screen = gv.screen_position(600.0);
        assert!((screen.x - 60.0).abs() < 0.1);
        assert!((screen.y - 30.0).abs() < 0.1);
    }

    #[test]
    fn test_geometric_estimator_creation() {
        let estimator = GeometricGazeEstimator::new();
        assert!((estimator.smoothing - 0.6).abs() < 0.001);
    }
}
