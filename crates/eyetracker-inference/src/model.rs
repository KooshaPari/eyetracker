//! ONNX model loading and configuration
//!
//! Handles model discovery, caching, and initialization.
//! 
//! Note: Full ONNX runtime integration requires adding the `ort` crate.
//! This module provides the configuration and model management without
//! direct ONNX session handling.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Type of inference model
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelType {
    /// Face detection model (e.g., BlazeFace)
    FaceDetection,
    /// Face mesh / landmark model (e.g., MediaPipe Face Mesh)
    FaceMesh,
    /// Eye segmentation model
    EyeSegmentation,
    /// Gaze estimation model
    GazeEstimation,
    /// Combined model (face + eyes + gaze in one)
    Combined,
}

impl ModelType {
    pub fn extension(&self) -> &'static str {
        "onnx"
    }

    pub fn default_name(&self) -> &'static str {
        match self {
            ModelType::FaceDetection => "face_detection.onnx",
            ModelType::FaceMesh => "face_mesh.onnx",
            ModelType::EyeSegmentation => "eye_segmentation.onnx",
            ModelType::GazeEstimation => "gaze_estimation.onnx",
            ModelType::Combined => "gaze_tracking.onnx",
        }
    }
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model type
    pub model_type: ModelType,
    /// Path to model file (None for default/model hub)
    pub path: Option<PathBuf>,
    /// Input tensor name
    pub input_name: String,
    /// Output tensor names
    pub output_names: Vec<String>,
    /// Expected input shape [batch, height, width, channels]
    pub input_shape: Vec<i64>,
    /// Hardware acceleration provider
    pub provider: InferenceProvider,
    /// Inference timeout in milliseconds
    pub timeout_ms: u64,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_type: ModelType::FaceMesh,
            path: None,
            input_name: "input".to_string(),
            output_names: vec!["output".to_string()],
            input_shape: vec![1, 256, 256, 3],
            provider: InferenceProvider::Auto,
            timeout_ms: 100,
        }
    }
}

/// Inference hardware provider
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InferenceProvider {
    /// Automatically select best available
    Auto,
    /// CPU inference
    Cpu,
    /// NVIDIA CUDA GPU
    Cuda,
    /// Apple Metal GPU
    Metal,
    /// CoreML (iOS/macOS)
    CoreML,
}

impl InferenceProvider {
    /// Get the name for this provider
    pub fn name(&self) -> &'static str {
        match self {
            InferenceProvider::Auto => "auto",
            InferenceProvider::Cpu => "CPU",
            InferenceProvider::Cuda => "CUDA",
            InferenceProvider::Metal => "Metal",
            InferenceProvider::CoreML => "CoreML",
        }
    }

    /// Check if this provider requires GPU
    pub fn requires_gpu(&self) -> bool {
        matches!(
            self,
            InferenceProvider::Cuda | InferenceProvider::Metal | InferenceProvider::CoreML
        )
    }

    /// Check if this provider is available on the current system
    #[cfg(target_os = "macos")]
    pub fn is_available(&self) -> bool {
        match self {
            InferenceProvider::Metal | InferenceProvider::CoreML => true,
            InferenceProvider::Auto | InferenceProvider::Cpu => true,
            InferenceProvider::Cuda => false,
        }
    }

    #[cfg(not(target_os = "macos"))]
    pub fn is_available(&self) -> bool {
        true
    }
}

/// Model loader and manager
pub struct ModelLoader {
    models_dir: PathBuf,
    cache_dir: PathBuf,
}

impl ModelLoader {
    /// Create a new model loader with standard directories
    pub fn new() -> Result<Self> {
        let models_dir = Self::default_models_dir()?;
        let cache_dir = Self::default_cache_dir()?;

        std::fs::create_dir_all(&models_dir)?;
        std::fs::create_dir_all(&cache_dir)?;

        Ok(Self {
            models_dir,
            cache_dir,
        })
    }

    fn default_models_dir() -> Result<PathBuf> {
        if let Ok(dir) = std::env::var("EYETRACKER_MODELS") {
            return Ok(PathBuf::from(dir));
        }

        #[cfg(target_os = "macos")]
        let base = dirs::home_dir()
            .map(|h| h.join("Library/Application Support/eyetracker/models"))
            .unwrap_or_else(|| PathBuf::from("./models"));

        #[cfg(target_os = "linux")]
        let base = dirs::home_dir()
            .map(|h| h.join(".local/share/eyetracker/models"))
            .unwrap_or_else(|| PathBuf::from("./models"));

        #[cfg(target_os = "windows")]
        let base = std::env::var("LOCALAPPDATA")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("C:\\ProgramData\\eyetracker\\models"))
            .join("models");

        #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
        let base = PathBuf::from("./models");

        Ok(base)
    }

    fn default_cache_dir() -> Result<PathBuf> {
        if let Ok(dir) = std::env::var("EYETRACKER_CACHE") {
            return Ok(PathBuf::from(dir));
        }

        let base = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("./.cache"))
            .join("eyetracker");

        Ok(base)
    }

    pub fn models_dir(&self) -> &Path {
        &self.models_dir
    }

    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    pub fn find_model(&self, model_type: ModelType) -> Option<PathBuf> {
        let full_path = self.models_dir.join(model_type.default_name());
        if full_path.exists() {
            return Some(full_path);
        }
        None
    }

    pub fn has_model(&self, model_type: ModelType) -> bool {
        self.find_model(model_type).is_some()
    }

    pub fn list_models(&self) -> Vec<(ModelType, PathBuf)> {
        let mut models = Vec::new();
        for model_type in [
            ModelType::FaceDetection,
            ModelType::FaceMesh,
            ModelType::EyeSegmentation,
            ModelType::GazeEstimation,
            ModelType::Combined,
        ] {
            if let Some(path) = self.find_model(model_type) {
                models.push((model_type, path));
            }
        }
        models
    }

    pub fn load_config(&self, model_type: ModelType) -> Result<ModelConfig> {
        let mut config = ModelConfig::default();
        config.model_type = model_type;

        match model_type {
            ModelType::FaceDetection => {
                config.input_name = "input".to_string();
                config.output_names = vec!["boxes".to_string(), "scores".to_string()];
                config.input_shape = vec![1, 128, 128, 3];
            }
            ModelType::FaceMesh => {
                config.input_name = "input".to_string();
                config.output_names = vec!["landmarks".to_string()];
                config.input_shape = vec![1, 256, 256, 3];
            }
            ModelType::GazeEstimation => {
                config.input_name = "input".to_string();
                config.output_names = vec!["gaze".to_string()];
                config.input_shape = vec![1, 224, 224, 3];
            }
            ModelType::Combined => {
                config.input_name = "frames".to_string();
                config.output_names = vec![
                    "face_box".to_string(),
                    "landmarks".to_string(),
                    "gaze".to_string(),
                ];
                config.input_shape = vec![1, 256, 256, 3];
            }
            _ => {}
        }

        Ok(config)
    }
}

impl Default for ModelLoader {
    fn default() -> Self {
        Self::new().expect("Failed to create model loader")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_type_extension() {
        assert_eq!(ModelType::FaceMesh.extension(), "onnx");
        assert_eq!(ModelType::GazeEstimation.extension(), "onnx");
    }

    #[test]
    fn test_inference_provider() {
        assert!(InferenceProvider::Metal.requires_gpu());
        assert!(InferenceProvider::Cuda.requires_gpu());
        assert!(!InferenceProvider::Cpu.requires_gpu());
    }

    #[test]
    fn test_model_loader_creation() {
        let loader = ModelLoader::new();
        assert!(loader.is_ok());
    }
}
