//! ML inference pipeline for eye tracking
//!
//! This crate provides face detection, facial landmark estimation (face mesh),
//! and gaze direction estimation. It is designed to work with ONNX models
//! (MediaPipe-style) but also provides fallback implementations for testing.
//!
//! Functional requirements covered by this crate:
//! - FR-EYE-INFER-002: Kalman 2D smoothing (smoothing.rs)
//! - FR-EYE-INFER-003: Fixation classification (classification.rs)
//! - FR-EYE-INFER-004: Saccade detection (classification.rs)
//! - FR-EYE-CAL-001: 9-point calibration protocol (calibration.rs)
//! - FR-EYE-CAL-002: Accuracy validation & drift tolerance (calibration.rs)
//! - FR-EYE-CAL-003: Calibration persistence (calibration.rs)
//! - FR-EYE-CAL-004: Drift monitoring & recalibration trigger (drift_monitor.rs)
//! - FR-EYE-CAL-005: Multi-monitor calibration (multi_monitor.rs)
//! - FR-EYE-ACCESS-001: Dwell-click selection (accessibility.rs)
//! - FR-EYE-ACCESS-002: Scroll-by-gaze (accessibility.rs)
//! - FR-EYE-INTEROP-003: FocalPoint connector (focalpoint.rs)
//! - FR-EYE-PRIVACY-001/002/003: On-device processing, no default cloud,
//!   recording consent (privacy.rs)

pub mod accessibility;
pub mod calibration;
pub mod classification;
pub mod drift_monitor;
pub mod face_mesh;
pub mod focalpoint;
pub mod gaze_estimator;
pub mod multi_monitor;
pub mod onnx_detector;
pub mod pipeline;
pub mod privacy;
pub mod smoothing;

// Re-export core types
pub use calibration::{load_calibration, save_calibration, CalibrationResult};
pub use face_mesh::*;
pub use gaze_estimator::*;
pub use pipeline::*;
