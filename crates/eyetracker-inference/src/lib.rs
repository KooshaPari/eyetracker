//! eyetracker-inference: ML inference for eye tracking
//!
//! Provides gaze estimation using ONNX models with hardware acceleration.
//!
//! ## Architecture
//!
//! ```text
//! Frame ──► Face Detection ──► Landmark Extraction ──► Gaze Estimation ──► Gaze Point
//!              (ONNX)              (ONNX)                  (ONNX)
//! ```

mod face_mesh;
mod gaze_estimator;
mod model;
mod pipeline;
pub mod processing;

pub use face_mesh::{FaceMesh, FaceMeshResult, Landmark, LandmarkType, FaceBox};
pub use gaze_estimator::{GazeEstimation, GazeEstimationResult, GazeDirection, ScreenCalibration};
pub use model::{ModelConfig, ModelLoader, ModelType, InferenceProvider};
pub use pipeline::{InferencePipeline, PipelineConfig};
pub use processing::{preprocess_frame, PreprocessOptions, detect_eye_region, crop_eye_region};
