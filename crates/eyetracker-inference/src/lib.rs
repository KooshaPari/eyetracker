//! ML inference pipeline for eye tracking
//!
//! This crate provides face detection, facial landmark estimation (face mesh),
//! and gaze direction estimation. It is designed to work with ONNX models
//! (MediaPipe-style) but also provides fallback implementations for testing.

pub mod classification;
pub mod face_mesh;
pub mod gaze_estimator;
pub mod pipeline;
pub mod smoothing;

// Re-export core types
pub use face_mesh::*;
pub use gaze_estimator::*;
pub use pipeline::*;
