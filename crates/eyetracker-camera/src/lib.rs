//! eyetracker-camera: High-performance webcam capture for eye tracking
//!
//! Provides camera enumeration, frame capture, and processing utilities
//! optimized for real-time eye tracking workloads.

mod camera;
mod frame;
mod processing;

pub use camera::{Camera, CameraConfig, CameraError};
pub use frame::{Frame, FrameFormat, FrameMetadata};
pub use processing::{preprocess_frame, detect_eye_region, crop_eye_region};
