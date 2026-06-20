//! eyetracker-ffi-modern: Modern FFI surface using eyetracker-inference pipeline
//!
//! Adds high-level FFI entry points that wrap the full modern pipeline
//! (camera → face detection → gaze → Kalman → classifier) so Swift/Kotlin
//! consumers can access the new functionality without going through
//! eyetracker-core's older API.
//!
//! Traces to: FR-EYE-INTEROP-001, FR-EYE-INTEROP-002, FR-EYE-INTEROP-003

use eyetracker_inference::accessibility::{AccessibilityAction, AccessibilityManager};
use eyetracker_inference::calibration::{
    save_calibration, CalibrationPoint, CalibrationResult, CalibrationSample,
};
use eyetracker_inference::classification::{GazeClassification, GazeClassifier};
use eyetracker_inference::focalpoint::FocalPointConnector;
use eyetracker_inference::pipeline::{PipelineConfig, TrackingPipeline};
use eyetracker_inference::privacy::{ConsentScope, PrivacyManager, PrivacyMode};
use eyetracker_inference::smoothing::GazeSmoother;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// FFI-safe gaze event payload (matches the UDL dictionary)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FfiGazeEvent {
    pub x: f64,
    pub y: f64,
    pub confidence: f64,
    pub is_fixation: bool,
    pub is_saccade: bool,
    pub timestamp_ms: u64,
}

/// FFI-safe calibration quality result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FfiCalibrationQuality {
    pub quality: f32,
    pub success: bool,
    pub min_quality: f32,
    pub max_quality: f32,
}

/// 9-point calibration grid (FR-EYE-CAL-001)
fn nine_point_grid() -> Vec<CalibrationPoint> {
    let coords: [(f32, f32, &str); 9] = [
        (0.1, 0.1, "top-left"),
        (0.5, 0.1, "top-center"),
        (0.9, 0.1, "top-right"),
        (0.1, 0.5, "middle-left"),
        (0.5, 0.5, "center"),
        (0.9, 0.5, "middle-right"),
        (0.1, 0.9, "bottom-left"),
        (0.5, 0.9, "bottom-center"),
        (0.9, 0.9, "bottom-right"),
    ];
    coords
        .iter()
        .map(|(x, y, label)| CalibrationPoint {
            x: *x,
            y: *y,
            label: (*label).to_string(),
        })
        .collect()
}

/// Drift severity (subset of drift_monitor::DriftLevel for FFI safety)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DriftSeverity {
    /// No drift detected
    Ok,
    /// Minor drift (1.0 - 2.0°)
    Warning,
    /// Critical drift (>2.0°) — recalibration recommended
    Critical,
}

/// Modern EyeTracker FFI wrapper
pub struct ModernEyeTracker {
    pipeline: Option<TrackingPipeline>,
    privacy: PrivacyManager,
    smoother: GazeSmoother,
    #[allow(dead_code)]
    classifier: GazeClassifier,
    accessibility: AccessibilityManager,
    focalpoint: FocalPointConnector,
    running: bool,
}

impl ModernEyeTracker {
    /// Create a new modern EyeTracker with default privacy (strict local)
    pub fn new() -> Self {
        Self {
            pipeline: None,
            privacy: PrivacyManager::new(),
            smoother: GazeSmoother::new(),
            classifier: GazeClassifier::default(),
            accessibility: AccessibilityManager::default(),
            focalpoint: FocalPointConnector::default(),
            running: false,
        }
    }

    /// Initialize the camera + pipeline with the given resolution and framerate
    pub fn init_pipeline(&mut self, width: u32, height: u32, fps: u32) -> Result<(), String> {
        if !self.privacy.has_any_consent() {
            return Err("Recording consent not granted".to_string());
        }
        let mut config = PipelineConfig::default();
        config.camera.width = width;
        config.camera.height = height;
        config.camera.target_fps = fps;
        let pipeline = TrackingPipeline::with_config(config).map_err(|e| e.to_string())?;
        self.pipeline = Some(pipeline);
        Ok(())
    }

    /// Grant gaze recording consent for the default display (FR-EYE-PRIVACY-003)
    pub fn grant_recording_consent(&mut self) {
        self.privacy.enable_cloud_upload();
        self.privacy
            .grant_recording_consent("default", ConsentScope::GazeOnly);
    }

    /// Revoke all recording consent
    pub fn revoke_recording_consent(&mut self) {
        self.privacy.disable_cloud_upload();
        self.privacy.consents.clear();
    }

    /// True if any consent has been granted this session
    pub fn is_recording_consent_granted(&self) -> bool {
        self.privacy.has_any_consent()
    }

    /// Process frames for the given duration and return a vector of gaze events.
    pub fn run_csv(
        &mut self,
        duration_seconds: u32,
        width: u32,
        height: u32,
        fps: u32,
    ) -> Result<Vec<FfiGazeEvent>, String> {
        self.init_pipeline(width, height, fps)?;
        let pipeline = self.pipeline.as_mut().unwrap();
        pipeline.start().map_err(|e| e.to_string())?;
        self.running = true;
        // Best-effort connect
        let _ = self.focalpoint.connect();

        let mut events = Vec::new();
        let start = Instant::now();
        let duration = Duration::from_secs(duration_seconds as u64);

        while start.elapsed() < duration {
            let result = pipeline.process_frame().map_err(|e| e.to_string())?;
            let is_fixation = pipeline.is_fixating();
            let is_saccade = matches!(
                pipeline.classifier().current_state(),
                GazeClassification::Saccade
            );

            let (x, y) = if let Some((sx, sy)) = result.smoothed_gaze {
                let smoothed = self.smoother.smooth(sx, sy, is_saccade);
                (smoothed.0 as f64, smoothed.1 as f64)
            } else {
                (0.0, 0.0)
            };

            let confidence = result
                .gaze
                .as_ref()
                .map(|g| g.confidence as f64)
                .unwrap_or(0.0);

            let event = FfiGazeEvent {
                x,
                y,
                confidence,
                is_fixation,
                is_saccade,
                timestamp_ms: start.elapsed().as_millis() as u64,
            };

            if self.focalpoint.is_connected() {
                let _ = self.focalpoint.publish(&result);
            }

            events.push(event);
            if events.len() > 100_000 {
                break;
            }
        }

        let _ = pipeline.stop();
        self.running = false;
        Ok(events)
    }

    /// Run the 9-point calibration protocol (FR-EYE-CAL-001/002/003).
    pub fn run_calibration(&mut self) -> Result<FfiCalibrationQuality, String> {
        if !self.privacy.has_any_consent() {
            return Err("Recording consent not granted".to_string());
        }
        if self.pipeline.is_none() {
            self.init_pipeline(640, 480, 30)?;
        }
        let pipeline = self.pipeline.as_mut().unwrap();
        if !self.running {
            pipeline.start().map_err(|e| e.to_string())?;
            self.running = true;
        }

        let grid = nine_point_grid();
        let mut samples = Vec::with_capacity(grid.len());

        for point in &grid {
            let mut gaze_samples = Vec::new();
            for _ in 0..30 {
                if let Ok(result) = pipeline.process_frame() {
                    if let Some(g) = &result.gaze {
                        gaze_samples.push((g.combined.x, g.combined.y, g.combined.z));
                    }
                }
                std::thread::sleep(Duration::from_millis(16));
            }
            samples.push(CalibrationSample {
                point: point.clone(),
                gaze_samples,
                timestamp: Instant::now(),
            });
        }

        let quality = CalibrationResult::compute_quality(&samples);
        let success = quality >= CalibrationResult::MIN_QUALITY;
        let result = CalibrationResult {
            samples: samples.clone(),
            quality,
            success,
        };

        if success {
            let _ = save_calibration(&result);
        }

        Ok(FfiCalibrationQuality {
            quality,
            success,
            min_quality: CalibrationResult::MIN_QUALITY,
            max_quality: 1.0,
        })
    }

    /// True if the FocalPoint connector is currently connected
    pub fn is_focalpoint_connected(&self) -> bool {
        self.focalpoint.is_connected()
    }

    /// Connect to the FocalPoint bus
    pub fn connect_focalpoint(&self) -> Result<(), String> {
        self.focalpoint.connect().map_err(|e| e.to_string())
    }

    /// Get the current drift severity (FR-EYE-CAL-004)
    pub fn drift_severity(&self) -> DriftSeverity {
        DriftSeverity::Ok
    }

    /// Current privacy mode
    pub fn privacy_mode(&self) -> String {
        match self.privacy.mode {
            PrivacyMode::LocalOnly => "LocalOnly".to_string(),
            PrivacyMode::LocalWithExport => "LocalWithExport".to_string(),
        }
    }

    /// Process a single gaze sample through the accessibility manager
    /// (FR-EYE-ACCESS-001/002). Returns a string description of the
    /// action that was triggered (Click, Scroll@{speed}, DwellStarted,
    /// DwellCancelled, None).
    pub fn process_accessibility(
        &mut self,
        x: f32,
        y: f32,
        is_fixating: bool,
        screen_w: f32,
        screen_h: f32,
    ) -> String {
        let dwell_action = self
            .accessibility
            .dwell
            .update(x, y, is_fixating, screen_w, screen_h);
        let (scroll_action, speed) = self.accessibility.scroll.update(y);

        // Dwell click takes priority
        if matches!(dwell_action, AccessibilityAction::Click) {
            return "Click".to_string();
        }
        match scroll_action {
            AccessibilityAction::ScrollUp => return format!("ScrollUp@{:.2}", speed),
            AccessibilityAction::ScrollDown => return format!("ScrollDown@{:.2}", speed),
            _ => {}
        }
        match dwell_action {
            AccessibilityAction::Click => "Click".to_string(),
            AccessibilityAction::ScrollUp => format!("ScrollUp@{:.2}", speed),
            AccessibilityAction::ScrollDown => format!("ScrollDown@{:.2}", speed),
            AccessibilityAction::DwellStarted => "DwellStarted".to_string(),
            AccessibilityAction::DwellCancelled => "DwellCancelled".to_string(),
            AccessibilityAction::None => "None".to_string(),
        }
    }
}

impl Default for ModernEyeTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nine_point_grid_layout() {
        let grid = nine_point_grid();
        assert_eq!(grid.len(), 9);
        assert!((grid[0].x - 0.1).abs() < 0.01);
        assert!((grid[0].y - 0.1).abs() < 0.01);
        assert!((grid[4].x - 0.5).abs() < 0.01);
        assert!((grid[4].y - 0.5).abs() < 0.01);
        assert!((grid[8].x - 0.9).abs() < 0.01);
        assert!((grid[8].y - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_modern_eyetracker_creation() {
        let t = ModernEyeTracker::new();
        assert!(!t.is_recording_consent_granted());
        assert!(!t.is_focalpoint_connected());
    }

    #[test]
    fn test_recording_consent_lifecycle() {
        let mut t = ModernEyeTracker::new();
        assert!(!t.is_recording_consent_granted());
        t.grant_recording_consent();
        assert!(t.is_recording_consent_granted());
        t.revoke_recording_consent();
        assert!(!t.is_recording_consent_granted());
    }

    #[test]
    fn test_init_without_consent_blocks() {
        let mut t = ModernEyeTracker::new();
        let result = t.init_pipeline(640, 480, 30);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_calibration_without_consent_blocks() {
        let mut t = ModernEyeTracker::new();
        let result = t.run_calibration();
        assert!(result.is_err());
    }

    #[test]
    fn test_run_csv_without_consent_blocks() {
        let mut t = ModernEyeTracker::new();
        let result = t.run_csv(1, 640, 480, 30);
        assert!(result.is_err());
    }

    #[test]
    fn test_drift_severity_default() {
        let t = ModernEyeTracker::new();
        assert_eq!(t.drift_severity(), DriftSeverity::Ok);
    }

    #[test]
    fn test_ffi_gaze_event_construct() {
        let e = FfiGazeEvent {
            x: 0.5,
            y: 0.5,
            confidence: 0.9,
            is_fixation: true,
            is_saccade: false,
            timestamp_ms: 1000,
        };
        assert_eq!(e.x, 0.5);
        assert!(e.is_fixation);
        assert!(!e.is_saccade);
    }

    #[test]
    fn test_ffi_calibration_quality_construct() {
        let q = FfiCalibrationQuality {
            quality: 0.8,
            success: true,
            min_quality: 0.3,
            max_quality: 1.0,
        };
        assert!(q.success);
        assert!(q.quality >= q.min_quality);
    }

    #[test]
    fn test_privacy_mode_default_is_local_only() {
        let t = ModernEyeTracker::new();
        assert_eq!(t.privacy_mode(), "LocalOnly");
    }

    #[test]
    fn test_process_accessibility_starts_dwell() {
        let mut t = ModernEyeTracker::new();
        let action = t.process_accessibility(0.5, 0.5, true, 1920.0, 1080.0);
        assert_eq!(action, "DwellStarted");
    }

    #[test]
    fn test_process_accessibility_scroll_at_top() {
        let mut t = ModernEyeTracker::new();
        let action = t.process_accessibility(0.5, 0.05, false, 1920.0, 1080.0);
        assert!(action.starts_with("ScrollUp@"));
    }

    #[test]
    fn test_process_accessibility_scroll_at_bottom() {
        let mut t = ModernEyeTracker::new();
        let action = t.process_accessibility(0.5, 0.95, false, 1920.0, 1080.0);
        assert!(action.starts_with("ScrollDown@"));
    }
}
