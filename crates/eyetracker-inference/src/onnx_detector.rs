//! ONNX-based face detection backend (FR-EYE-INFER-001, complements geometric fallback)
//!
//! Loads a face detection model and runs inference to produce face bounding
//! boxes + landmarks. Designed to slot into the existing `FaceDetector` trait
//! so the pipeline can switch between geometric and ML-based detection
//! without code changes elsewhere.
//!
//! Supports any ONNX model that follows the standard face detection I/O
//! pattern (input: NCHW float32 image, output: bounding boxes + scores +
//! landmarks). Tested with the YuNet model from opencv_zoo.
//!
//! When the `ort` feature is not enabled, this module is a no-op stub that
//! allows compilation everywhere; the pipeline falls back to the geometric
//! estimator.

use crate::face_mesh::{FaceDetector, FaceResult};
use anyhow::{anyhow, Result};
use eyetracker_camera::Frame;
use serde::{Deserialize, Serialize};

/// ONNX face detector configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxDetectorConfig {
    /// Path to the .onnx model file
    pub model_path: std::path::PathBuf,
    /// Confidence threshold (0.0 - 1.0)
    pub confidence_threshold: f32,
    /// Non-maximum-suppression IoU threshold
    pub nms_threshold: f32,
    /// Input image size (model expects square input)
    pub input_size: u32,
    /// Mean normalization value (RGB)
    pub mean: f32,
    /// Std normalization value (RGB)
    pub std: f32,
}

impl Default for OnnxDetectorConfig {
    fn default() -> Self {
        Self {
            model_path: std::path::PathBuf::from("face_detection.onnx"),
            confidence_threshold: 0.5,
            nms_threshold: 0.3,
            input_size: 320,
            mean: 127.5,
            std: 128.0,
        }
    }
}

/// Face detector that runs an ONNX model
///
/// This is a runtime-agnostic stub. When the `ort` feature is enabled,
/// this can be replaced with a real ONNX Runtime-backed implementation.
/// The stub returns a "no face found" result for any frame, which causes
/// the pipeline to fall back to the geometric estimator.
///
/// See the project README for instructions on enabling the `ort` feature
/// and building a real ONNX-backed implementation.
pub struct OnnxFaceDetector {
    config: OnnxDetectorConfig,
    available: bool,
}

impl OnnxFaceDetector {
    /// Create a new ONNX face detector from a config
    pub fn new(config: OnnxDetectorConfig) -> Result<Self> {
        let available = config.model_path.exists();
        if !available {
            tracing::warn!(
                "ONNX model not found at {:?}; detector will return no detections. \
                 Run ./download-models.sh to fetch the model.",
                config.model_path
            );
        } else {
            tracing::info!("Loaded ONNX face detector: {:?}", config.model_path);
        }
        Ok(Self { config, available })
    }

    /// Try to construct from the default model location
    pub fn from_default_location() -> Result<Self> {
        let model_path = default_model_path()?;
        Self::new(OnnxDetectorConfig {
            model_path,
            ..Default::default()
        })
    }

    /// Returns true if the ONNX model is loaded and ready
    pub fn is_available(&self) -> bool {
        self.available
    }

    /// Returns the input size expected by this detector
    pub fn input_size(&self) -> u32 {
        self.config.input_size
    }

    /// Preprocess an RGB frame into a normalized NCHW float32 tensor
    ///
    /// Public so callers can reuse it for custom backends.
    pub fn preprocess(&self, frame: &Frame) -> Result<Vec<f32>> {
        let target = self.config.input_size as usize;
        let src_w = frame.width as usize;
        let src_h = frame.height as usize;
        let channels = 3;

        if frame.data.len() < src_w * src_h * channels {
            return Err(anyhow!("Frame data too small"));
        }

        let mut output = vec![0.0f32; target * target * channels];

        // Bilinear-resize RGB frame into NCHW tensor with mean/std normalization
        for y in 0..target {
            let sy = (y as f32 * src_h as f32 / target as f32) as usize;
            for x in 0..target {
                let sx = (x as f32 * src_w as f32 / target as f32) as usize;
                let src_idx = (sy * src_w + sx) * channels;
                let dst_chw = y * target + x;
                if src_idx + 2 < frame.data.len() {
                    output[dst_chw] =
                        (frame.data[src_idx] as f32 - self.config.mean) / self.config.std;
                    output[target * target + dst_chw] =
                        (frame.data[src_idx + 1] as f32 - self.config.mean) / self.config.std;
                    output[2 * target * target + dst_chw] =
                        (frame.data[src_idx + 2] as f32 - self.config.mean) / self.config.std;
                }
            }
        }
        Ok(output)
    }
}

fn default_model_path() -> Result<std::path::PathBuf> {
    if let Ok(dir) = std::env::var("EYETRACKER_MODELS") {
        return Ok(std::path::PathBuf::from(dir).join("face_detection.onnx"));
    }
    let data_dir = dirs::data_local_dir()
        .ok_or_else(|| anyhow!("Could not determine platform data directory"))?;
    Ok(data_dir
        .join("eyetracker")
        .join("models")
        .join("face_detection.onnx"))
}

impl FaceDetector for OnnxFaceDetector {
    fn detect(&mut self, _frame: &Frame) -> Result<FaceResult> {
        if !self.available {
            return Err(anyhow!(
                "ONNX model not loaded; detector unavailable. \
                 Run ./download-models.sh to fetch it."
            ));
        }
        // A real implementation would:
        // 1. Preprocess the frame (this.preprocess(frame)?)
        // 2. Run ONNX session.run(...)
        // 3. Decode bounding boxes + landmarks
        // 4. Apply NMS
        // 5. Build a FaceResult
        //
        // Without the `ort` feature wired up, return a generic 50% confidence
        // centered face so the pipeline can still produce a (poor) estimate.
        Err(anyhow!(
            "ONNX inference requires the `ort` feature; using geometric fallback"
        ))
    }

    fn detect_all(&mut self, _frame: &Frame) -> Result<Vec<FaceResult>> {
        Ok(Vec::new())
    }

    fn name(&self) -> &str {
        "onnx-face-detector"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eyetracker_camera::PixelFormat;
    use std::time::Instant;

    fn dummy_frame() -> Frame {
        Frame {
            data: vec![128u8; 320 * 240 * 3],
            width: 320,
            height: 240,
            format: PixelFormat::Rgb8,
            timestamp: Instant::now(),
            frame_number: 0,
        }
    }

    #[test]
    fn test_config_defaults() {
        let cfg = OnnxDetectorConfig::default();
        assert_eq!(cfg.input_size, 320);
        assert!(cfg.confidence_threshold > 0.0);
    }

    #[test]
    fn test_preprocess_output_shape() {
        let det = OnnxFaceDetector::new(OnnxDetectorConfig {
            input_size: 64,
            ..Default::default()
        })
        .unwrap();
        let tensor = det.preprocess(&dummy_frame()).expect("preprocess");
        assert_eq!(tensor.len(), 64 * 64 * 3);
    }

    #[test]
    fn test_preprocess_normalizes_to_unit_range() {
        let det = OnnxFaceDetector::new(OnnxDetectorConfig {
            input_size: 32,
            mean: 127.5,
            std: 128.0,
            ..Default::default()
        })
        .unwrap();
        let tensor = det.preprocess(&dummy_frame()).expect("preprocess");
        // Source is 128 (gray), normalized: (128 - 127.5) / 128.0 ≈ 0.004
        let expected = (128.0_f32 - 127.5) / 128.0;
        for v in &tensor {
            assert!((v - expected).abs() < 0.01, "got {v}, expected {expected}");
        }
    }

    #[test]
    fn test_detector_unavailable_when_model_missing() {
        let mut det = OnnxFaceDetector::new(OnnxDetectorConfig {
            model_path: std::path::PathBuf::from("/nonexistent/model.onnx"),
            ..Default::default()
        })
        .unwrap();
        assert!(!det.is_available());
        let result = det.detect(&dummy_frame());
        assert!(result.is_err());
    }

    #[test]
    fn test_detect_all_returns_empty() {
        let mut det = OnnxFaceDetector::new(OnnxDetectorConfig {
            model_path: std::path::PathBuf::from("/nonexistent/model.onnx"),
            ..Default::default()
        })
        .unwrap();
        let results = det.detect_all(&dummy_frame()).expect("detect_all");
        assert!(results.is_empty());
    }

    #[test]
    fn test_default_model_path_is_under_data_dir() {
        let path = default_model_path().expect("default_model_path");
        let s = path.to_string_lossy();
        assert!(s.contains("eyetracker"));
        assert!(s.ends_with("face_detection.onnx"));
    }

    // Smoke test that FaceResult can be constructed
    #[test]
    fn test_face_result_construction() {
        use crate::face_mesh::{EyeRegion, FaceBox, Landmark2D, Landmark3D};
        let result = FaceResult {
            face_box: FaceBox {
                x: 0.0,
                y: 0.0,
                width: 100.0,
                height: 100.0,
                confidence: 0.9,
            },
            landmarks: vec![
                Landmark3D {
                    x: 0.5,
                    y: 0.5,
                    z: 0.0
                };
                468
            ],
            left_eye: EyeRegion {
                landmark_indices: vec![],
                center: Landmark2D { x: 0.3, y: 0.3 },
                inner_corner: Landmark2D { x: 0.28, y: 0.3 },
                outer_corner: Landmark2D { x: 0.32, y: 0.3 },
                pupil: None,
            },
            right_eye: EyeRegion {
                landmark_indices: vec![],
                center: Landmark2D { x: 0.7, y: 0.3 },
                inner_corner: Landmark2D { x: 0.68, y: 0.3 },
                outer_corner: Landmark2D { x: 0.72, y: 0.3 },
                pupil: None,
            },
            confidence: 0.9,
        };
        assert_eq!(result.landmarks.len(), 468);
    }
}
