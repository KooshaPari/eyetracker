//! Candle inference backend for gaze detection.
//!
//! Provides a Rust-native ML backend using [`candle-core`](https://github.com/huggingface/candle).
//! Same `FaceDetector` contract as [`crate::onnx_detector::OnnxFaceDetector`],
//! but no external runtime / dynamic library required (candle is pure Rust).
//!
//! ## Two-stage pipeline
//!
//! 1. **Detection** — produce a list of `FaceBox { x, y, w, h, confidence }`
//!    from an input frame.
//! 2. **Regression** — produce a `GazeVector { pitch, yaw }` for each face
//!    from the cropped face region.
//!
//! For stage 1 the Candle backend currently uses the same heuristic as the
//! ONNX backend's mock fallback (skin-tone histogram + grid scan) so that
//! tests pass without a trained model file. For stage 2 a tiny single-layer
//! MLP head (12 -> 64 -> 2 with ReLU) takes the mean-pooled normalized
//! face patch and regresses pitch/yaw.
//!
//! ## Feature gate
//!
//! Gated behind `feature = "candle"`. Default builds remain lightweight
//! (no `candle-core` dependency). The `SyntheticCandleBackend` and the
//! `CandleDetector` struct compile + test in the default build; the
//! real-model load path is stubbed behind the feature flag.

#![cfg_attr(
    feature = "candle",
    doc = "Candle backend (feature = \"candle\") — pure-Rust ML runtime."
)]

use std::path::Path;

use crate::face_detector::{DetectionError, FaceBox, FaceDetector, GazeVector};
use crate::preprocess::preprocess_face;

// ============================================================================
// Public API
// ============================================================================

/// Candle-based gaze detector.
///
/// Mirrors [`crate::onnx_detector::OnnxFaceDetector`] so callers can swap
/// backends without changing higher-level pipeline code.
#[derive(Debug)]
pub struct CandleDetector {
    /// Backend name for diagnostics / logs.
    name: String,
    /// Mock-detection threshold (skin-tone pixel ratio required to claim a face).
    threshold: f32,
    /// Backbone variant: "mock-mlp", "vit-tiny" (placeholder).
    backbone: CandleBackbone,
    /// Cached model bytes (placeholder; real impl loads safetensors).
    #[allow(dead_code)]
    weights: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandleBackbone {
    /// Single-layer MLP head — used for tests + CI.
    MockMlp,
    /// ViT-tiny — placeholder for future landed model.
    VitTiny,
}

impl CandleDetector {
    /// Construct a new Candle detector with a fresh in-memory model.
    ///
    /// In the default build this uses [`CandleBackbone::MockMlp`] so tests pass
    /// without a model file. With the `candle` feature enabled + a model path,
    /// call [`CandleDetector::from_pretrained`] instead.
    pub fn new() -> Self {
        Self {
            name: "candle-detector".to_string(),
            threshold: 0.15,
            backbone: CandleBackbone::MockMlp,
            weights: Vec::new(),
        }
    }

    /// Load a Candle-compatible model from a safetensors file.
    ///
    /// # Errors
    ///
    /// Returns [`DetectionError::ModelLoad`] if the file cannot be read or
    /// parsed. With `feature = "candle"` enabled, real parsing is performed;
    /// otherwise this returns an error indicating the feature is required.
    pub fn from_pretrained(_path: impl AsRef<Path>) -> Result<Self, DetectionError> {
        #[cfg(feature = "candle")]
        {
            // Real impl would:
            //   1. Read bytes from path
            //   2. candle::safetensors::load(path)
            //   3. Build VarMap / VarBuilder from tensors
            //   4. Construct the chosen backbone
            //
            // For now we return a stubbed error; the contract is documented.
            let _ = _path.as_ref();
            Err(DetectionError::ModelLoad(
                "candle feature enabled but no real model shipped; use CandleDetector::new() \
                 for tests or wire a ViT-tiny in EYE-SOTA-007b"
                    .to_string(),
            ))
        }
        #[cfg(not(feature = "candle"))]
        {
            Err(DetectionError::ModelLoad(
                "candle feature not enabled; rebuild with --features candle".to_string(),
            ))
        }
    }

    /// Set the mock-detection threshold.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set the backbone variant.
    pub fn with_backbone(mut self, backbone: CandleBackbone) -> Self {
        self.backbone = backbone;
        self
    }

    /// Get the backend name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the current backbone.
    pub fn backbone(&self) -> CandleBackbone {
        self.backbone
    }

    /// Synthetic 2-layer MLP regressor — used for tests + as a placeholder
    /// until a real gaze regression head lands.
    ///
    /// Input: 12-dim mean-pooled normalized face patch (4 channels x 3 RGB).
    /// Output: 2-dim (pitch, yaw) in radians.
    fn synthetic_mlp_forward(&self, features: &[f32; 12]) -> (f32, f32) {
        // Hidden layer: 12 -> 64 with ReLU
        // (Deterministic weights baked in for test reproducibility.)
        let mut hidden = [0.0_f32; 64];
        for (i, h) in hidden.iter_mut().enumerate() {
            let mut acc = 0.0_f32;
            for (j, &x) in features.iter().enumerate() {
                // Deterministic weight matrix: w[i][j] = sin(i*0.31 + j*0.17) * 0.25
                acc += ((((i * 31 + j * 17) % 100) as f32) / 100.0 - 0.5) * x * 0.25;
            }
            *h = acc.max(0.0); // ReLU
        }
        // Output layer: 64 -> 2
        let mut out = [0.0_f32; 2];
        for (i, o) in out.iter_mut().enumerate() {
            let mut acc = 0.0_f32;
            for (j, &h) in hidden.iter().enumerate() {
                acc += ((((i * 13 + j * 7) % 100) as f32) / 100.0 - 0.5) * h * 0.1;
            }
            *o = acc.tanh(); // bounded in [-1, 1]
        }
        (out[0], out[1])
    }
}

impl Default for CandleDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl FaceDetector for CandleDetector {
    fn detect(
        &self,
        frame: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Vec<FaceBox>, DetectionError> {
        if frame.is_empty() {
            return Err(DetectionError::InvalidInput("empty frame".to_string()));
        }
        if width == 0 || height == 0 {
            return Err(DetectionError::InvalidInput(format!(
                "zero dimension: {}x{}",
                width, height
            )));
        }
        if (frame.len() as u32) < width * height * 3 {
            return Err(DetectionError::InvalidInput(format!(
                "frame too small: {} bytes for {}x{} RGB",
                frame.len(),
                width,
                height
            )));
        }

        // Stage 1: heuristic face detection — same as OnnxFaceDetector::detect.
        // Scans a 4x4 grid of candidate boxes; keeps those with enough skin-tone pixels.
        let mut boxes = Vec::new();
        let grid_x = 4;
        let grid_y = 4;
        let cell_w = width / grid_x;
        let cell_h = height / grid_y;

        for gy in 0..grid_y {
            for gx in 0..grid_x {
                let x = gx * cell_w;
                let y = gy * cell_h;
                let skin_ratio = skin_tone_ratio(frame, width, height, x, y, cell_w, cell_h);
                if skin_ratio > self.threshold {
                    boxes.push(FaceBox {
                        x,
                        y,
                        w: cell_w,
                        h: cell_h,
                        confidence: skin_ratio.min(1.0),
                    });
                }
            }
        }

        if boxes.is_empty() {
            // Fallback: center box with low confidence.
            boxes.push(FaceBox {
                x: width / 4,
                y: height / 4,
                w: width / 2,
                h: height / 2,
                confidence: 0.1,
            });
        }

        // Sort by confidence descending.
        boxes.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Trim to top-3 detections.
        boxes.truncate(3);

        // Sanity check: dim ratio.
        for b in &boxes {
            if b.w == 0 || b.h == 0 {
                return Err(DetectionError::Inference(
                    "zero-dim face box produced".to_string(),
                ));
            }
        }
        Ok(boxes)
    }

    fn estimate_gaze(
        &self,
        frame: &[u8],
        width: u32,
        height: u32,
        face: &FaceBox,
    ) -> Result<GazeVector, DetectionError> {
        let patch = preprocess_face(frame, width, height, face, 12)?;
        let (pitch, yaw) = self.synthetic_mlp_forward(&patch);
        Ok(GazeVector { pitch, yaw })
    }

    fn backend_name(&self) -> &str {
        "candle"
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Compute the skin-tone pixel ratio for a sub-region.
///
/// Heuristic: a pixel is "skin-toned" if R > G > B and R > 95, G > 40, B > 20.
fn skin_tone_ratio(frame: &[u8], width: u32, height: u32, x: u32, y: u32, w: u32, h: u32) -> f32 {
    let mut skin = 0_u32;
    let mut total = 0_u32;
    let max_x = (x + w).min(width);
    let max_y = (y + h).min(height);

    for py in y..max_y {
        for px in x..max_x {
            let idx = ((py * width + px) * 3) as usize;
            if idx + 2 >= frame.len() {
                return 0.0;
            }
            let r = frame[idx];
            let g = frame[idx + 1];
            let b = frame[idx + 2];
            total += 1;
            if r > g && g > b && r > 95 && g > 40 && b > 20 {
                skin += 1;
            }
        }
    }
    if total == 0 {
        0.0
    } else {
        skin as f32 / total as f32
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_face_frame(width: u32, height: u32) -> Vec<u8> {
        // Solid skin-tone background so the heuristic detector finds a face.
        let mut buf = vec![0_u8; (width * height * 3) as usize];
        for chunk in buf.chunks_exact_mut(3) {
            chunk[0] = 220; // R
            chunk[1] = 180; // G
            chunk[2] = 140; // B
        }
        buf
    }

    #[test]
    fn synthetic_mlp_is_deterministic() {
        let d = CandleDetector::new();
        let f = [0.5_f32; 12];
        let (p1, y1) = d.synthetic_mlp_forward(&f);
        let (p2, y2) = d.synthetic_mlp_forward(&f);
        assert_eq!(p1, p2);
        assert_eq!(y1, y2);
        assert!(p1.abs() <= 1.0);
        assert!(y1.abs() <= 1.0);
    }

    #[test]
    fn synthetic_mlp_outputs_bounded() {
        let d = CandleDetector::new();
        for seed in 0..16_u32 {
            let mut f = [0.0_f32; 12];
            for (i, x) in f.iter_mut().enumerate() {
                *x = ((seed as f32 + i as f32) * 0.31).sin();
            }
            let (p, y) = d.synthetic_mlp_forward(&f);
            assert!(
                p >= -1.0 && p <= 1.0,
                "pitch {} out of bounds for seed {}",
                p,
                seed
            );
            assert!(
                y >= -1.0 && y <= 1.0,
                "yaw {} out of bounds for seed {}",
                y,
                seed
            );
        }
    }

    #[test]
    fn detect_skin_tone_frame_yields_boxes() {
        let d = CandleDetector::new().with_threshold(0.05);
        let frame = synth_face_frame(640, 480);
        let boxes = d.detect(&frame, 640, 480).unwrap();
        assert!(!boxes.is_empty(), "expected at least one face box");
        for b in &boxes {
            assert!(b.confidence > 0.0);
            assert!(b.w > 0 && b.h > 0);
        }
    }

    #[test]
    fn detect_empty_frame_errors() {
        let d = CandleDetector::new();
        let err = d.detect(&[], 640, 480).unwrap_err();
        assert!(matches!(err, DetectionError::InvalidInput(_)));
    }

    #[test]
    fn detect_zero_dimension_errors() {
        let d = CandleDetector::new();
        let frame = synth_face_frame(640, 480);
        let err = d.detect(&frame, 0, 480).unwrap_err();
        assert!(matches!(err, DetectionError::InvalidInput(_)));
    }

    #[test]
    fn detect_undersized_frame_errors() {
        let d = CandleDetector::new();
        let frame = vec![0_u8; 100];
        let err = d.detect(&frame, 640, 480).unwrap_err();
        assert!(matches!(err, DetectionError::InvalidInput(_)));
    }

    #[test]
    fn estimate_gaze_returns_bounded_vector() {
        let d = CandleDetector::new();
        let frame = synth_face_frame(640, 480);
        let face = FaceBox {
            x: 100,
            y: 100,
            w: 200,
            h: 200,
            confidence: 0.9,
        };
        let g = d.estimate_gaze(&frame, 640, 480, &face).unwrap();
        assert!(g.pitch >= -1.0 && g.pitch <= 1.0);
        assert!(g.yaw >= -1.0 && g.yaw <= 1.0);
    }

    #[test]
    fn estimate_gaze_is_deterministic() {
        let d = CandleDetector::new();
        let frame = synth_face_frame(640, 480);
        let face = FaceBox {
            x: 100,
            y: 100,
            w: 200,
            h: 200,
            confidence: 0.9,
        };
        let g1 = d.estimate_gaze(&frame, 640, 480, &face).unwrap();
        let g2 = d.estimate_gaze(&frame, 640, 480, &face).unwrap();
        assert_eq!(g1.pitch, g2.pitch);
        assert_eq!(g1.yaw, g2.yaw);
    }

    #[test]
    fn backend_name_is_candle() {
        let d = CandleDetector::new();
        assert_eq!(d.backend_name(), "candle");
        assert_eq!(FaceDetector::backend_name(&d), "candle");
    }

    #[test]
    fn from_pretrained_without_feature_errors() {
        #[cfg(not(feature = "candle"))]
        {
            let r = CandleDetector::from_pretrained("/nonexistent/model.safetensors");
            assert!(r.is_err());
            assert!(matches!(r.unwrap_err(), DetectionError::ModelLoad(_)));
        }
        #[cfg(feature = "candle")]
        {
            // Even with feature enabled, real model loading is stubbed.
            let r = CandleDetector::from_pretrained("/nonexistent/model.safetensors");
            assert!(r.is_err());
        }
    }

    #[test]
    fn backbone_variant_round_trips() {
        let d = CandleDetector::new().with_backbone(CandleBackbone::VitTiny);
        assert_eq!(d.backbone(), CandleBackbone::VitTiny);
    }

    #[test]
    fn threshold_round_trips() {
        let d = CandleDetector::new().with_threshold(0.42);
        assert!((d.threshold - 0.42).abs() < 1e-6);
    }

    #[test]
    fn detect_sorts_boxes_by_confidence_descending() {
        let d = CandleDetector::new().with_threshold(0.01);
        let frame = synth_face_frame(640, 480);
        let boxes = d.detect(&frame, 640, 480).unwrap();
        for pair in boxes.windows(2) {
            assert!(
                pair[0].confidence >= pair[1].confidence,
                "boxes not sorted by confidence: {:?} then {:?}",
                pair[0],
                pair[1]
            );
        }
    }

    #[test]
    fn detect_truncates_to_top_3() {
        let d = CandleDetector::new().with_threshold(0.01);
        let frame = synth_face_frame(640, 480);
        let boxes = d.detect(&frame, 640, 480).unwrap();
        assert!(boxes.len() <= 3);
    }

    #[test]
    fn default_matches_new() {
        let a = CandleDetector::default();
        let b = CandleDetector::new();
        assert_eq!(a.name, b.name);
        assert_eq!(a.threshold, b.threshold);
        assert_eq!(a.backbone, b.backbone);
    }
}
