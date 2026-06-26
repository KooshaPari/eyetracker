//! Face detector trait + shared types.
//!
//! This module provides the `FaceDetector` trait and supporting data types
//! (`FaceBox`, `GazeVector`, `DetectionError`) that other detector backends
//! (candle, onnx, v4l2, etc.) implement.
//!
//! It is intentionally separate from `face_mesh` (which provides landmark
//! detection on top of a face box). A detector implementing this trait only
//! needs to find the box; gaze estimation is optional.

use crate::Frame;

/// A bounding box around a detected face, in image pixel coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct FaceBox {
    /// Top-left x coordinate (inclusive, px).
    pub x: u32,
    /// Top-left y coordinate (inclusive, px).
    pub y: u32,
    /// Box width (px).
    pub width: u32,
    /// Box height (px).
    pub height: u32,
    /// Detection confidence in [0.0, 1.0].
    pub confidence: f32,
}

impl FaceBox {
    pub fn new(x: u32, y: u32, width: u32, height: u32, confidence: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
            confidence,
        }
    }

    pub fn right(&self) -> u32 {
        self.x.saturating_add(self.width)
    }

    pub fn bottom(&self) -> u32 {
        self.y.saturating_add(self.height)
    }

    pub fn area(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    pub fn iou(&self, other: &Self) -> f32 {
        let ix1 = self.x.max(other.x);
        let iy1 = self.y.max(other.y);
        let ix2 = self.right().min(other.right());
        let iy2 = self.bottom().min(other.bottom());
        let iw = ix2.saturating_sub(ix1);
        let ih = iy2.saturating_sub(iy1);
        let inter = u64::from(iw) * u64::from(ih);
        let union = self.area() + other.area() - inter;
        if union == 0 {
            0.0
        } else {
            inter as f32 / union as f32
        }
    }
}

/// Estimated gaze direction for a face, expressed as a unit vector plus
/// a screen-space point. The unit vector is in the camera frame (x right,
/// y down, z forward). The screen point is normalized [0.0, 1.0].
#[derive(Debug, Clone, PartialEq)]
pub struct GazeVector {
    /// Camera-frame unit vector x component.
    pub x: f32,
    /// Camera-frame unit vector y component.
    pub y: f32,
    /// Camera-frame unit vector z component.
    pub z: f32,
    /// Normalized screen-space gaze point x in [0.0, 1.0].
    pub screen_x: f32,
    /// Normalized screen-space gaze point y in [0.0, 1.0].
    pub screen_y: f32,
}

impl GazeVector {
    pub fn new(x: f32, y: f32, z: f32, screen_x: f32, screen_y: f32) -> Self {
        Self {
            x,
            y,
            z,
            screen_x,
            screen_y,
        }
    }

    pub fn zero() -> Self {
        Self::new(0.0, 0.0, 1.0, 0.5, 0.5)
    }
}

/// Errors returned by `FaceDetector` implementations.
#[derive(Debug, Clone, PartialEq)]
pub enum DetectionError {
    /// The input frame has zero width or height.
    InvalidFrameDimensions,
    /// The detector backend failed to initialize (e.g. model not found).
    InitializationFailed(String),
    /// The detector ran but no faces were found.
    NoFacesFound,
    /// The detector backend encountered an internal error during inference.
    InferenceFailed(String),
}

impl std::fmt::Display for DetectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidFrameDimensions => write!(f, "invalid frame dimensions"),
            Self::InitializationFailed(msg) => write!(f, "initialization failed: {msg}"),
            Self::NoFacesFound => write!(f, "no faces found"),
            Self::InferenceFailed(msg) => write!(f, "inference failed: {msg}"),
        }
    }
}

impl std::error::Error for DetectionError {}

/// A face detector: takes a frame, returns bounding boxes + optional gaze.
///
/// Backends (candle, onnx, v4l2, arkit, etc.) implement this trait to
/// provide interchangeable face-detection capabilities.
pub trait FaceDetector {
    /// Run face detection on a single frame. Returns bounding boxes for
    /// every face found, sorted by descending confidence.
    fn detect_faces(&mut self, frame: &Frame) -> Result<Vec<FaceBox>, DetectionError>;

    /// Run gaze estimation for the largest face in the frame. Returns
    /// `None` if no face is found. Default impl calls `detect_faces` then
    /// `estimate_gaze` on the highest-confidence box; backends with
    /// fused face-detection + gaze-estimation should override.
    fn detect_with_gaze(&mut self, frame: &Frame) -> Result<Option<GazeVector>, DetectionError> {
        let boxes = self.detect_faces(frame)?;
        let largest = boxes.into_iter().max_by(|a, b| {
            a.area()
                .partial_cmp(&b.area())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        match largest {
            None => Ok(None),
            Some(b) => Ok(Some(self.estimate_gaze(frame, &b)?)),
        }
    }

    /// Estimate gaze for a specific face box within a frame. Backends
    /// without gaze-estimation capability return `DetectionError::InferenceFailed`.
    fn estimate_gaze(
        &mut self,
        _frame: &Frame,
        _box: &FaceBox,
    ) -> Result<GazeVector, DetectionError> {
        Err(DetectionError::InferenceFailed(
            "gaze estimation not supported by this backend".into(),
        ))
    }

    /// Backend name (e.g. "candle-mlp", "onnx-yolov8", "v4l2-ir"). Used
    /// for telemetry + grading labels.
    fn backend_name(&self) -> &'static str;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn face_box_right_bottom() {
        let b = FaceBox::new(10, 20, 30, 40, 0.9);
        assert_eq!(b.right(), 40);
        assert_eq!(b.bottom(), 60);
        assert_eq!(b.area(), 1200);
    }

    #[test]
    fn face_box_iou_full_overlap() {
        let a = FaceBox::new(0, 0, 100, 100, 0.9);
        let b = FaceBox::new(0, 0, 100, 100, 0.8);
        assert!((a.iou(&b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn face_box_iou_no_overlap() {
        let a = FaceBox::new(0, 0, 10, 10, 0.9);
        let b = FaceBox::new(20, 20, 10, 10, 0.9);
        assert_eq!(a.iou(&b), 0.0);
    }

    #[test]
    fn face_box_iou_partial() {
        let a = FaceBox::new(0, 0, 10, 10, 0.9);
        let b = FaceBox::new(5, 5, 10, 10, 0.9);
        // overlap = 5*5 = 25, union = 100 + 100 - 25 = 175
        assert!((a.iou(&b) - (25.0 / 175.0)).abs() < 1e-6);
    }

    #[test]
    fn gaze_vector_zero() {
        let g = GazeVector::zero();
        assert_eq!(g.screen_x, 0.5);
        assert_eq!(g.screen_y, 0.5);
    }

    #[test]
    fn detection_error_display() {
        assert_eq!(
            DetectionError::InvalidFrameDimensions.to_string(),
            "invalid frame dimensions"
        );
        assert_eq!(DetectionError::NoFacesFound.to_string(), "no faces found");
    }

    /// A stub detector used for trait-level testing.
    struct StubDetector;
    impl FaceDetector for StubDetector {
        fn detect_faces(&mut self, _frame: &Frame) -> Result<Vec<FaceBox>, DetectionError> {
            Ok(vec![FaceBox::new(0, 0, 10, 10, 1.0)])
        }
        fn backend_name(&self) -> &'static str {
            "stub"
        }
    }

    #[test]
    fn stub_detector_default_gaze_unsupported() {
        let mut d = StubDetector;
        let err = d.estimate_gaze(&Frame::new(10, 10)).unwrap_err();
        assert!(matches!(err, DetectionError::InferenceFailed(_)));
    }
}
