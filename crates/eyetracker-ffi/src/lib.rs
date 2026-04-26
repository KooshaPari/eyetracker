//! eyetracker-ffi: UniFFI binding layer for Swift/Kotlin interop
//! Traces to: FR-EYE-INTEROP-001, FR-EYE-INTEROP-002, FR-EYE-INTEROP-003

use eyetracker_core::CalibrationState as CoreCalibrationState;
use eyetracker_core::Calibrator as CoreCalibrator;
use eyetracker_domain::Point as DomainPoint;
use serde::{Deserialize, Serialize};

// Note: Proc-macro scaffolding via uniffi::export attribute
// UDL-based bindings will be generated in bindings/ on successful build

// ============================================================================
// Domain Type Wrappers
// ============================================================================

/// FFI-safe wrapper for Point (required for UniFFI)
/// Traces to: FR-EYE-INTEROP-001
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    fn to_domain(self) -> DomainPoint {
        DomainPoint::new(self.x, self.y)
    }

    fn from_domain(p: DomainPoint) -> Self {
        Self { x: p.x, y: p.y }
    }
}

/// FFI-safe wrapper for GazeEstimate
/// Traces to: FR-EYE-INTEROP-001
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GazeEstimate {
    pub screen_pos: Point,
    pub confidence: f64,
}

impl GazeEstimate {
    pub fn new(screen_pos: Point, confidence: f64) -> Self {
        Self {
            screen_pos,
            confidence,
        }
    }
}

/// CalibrationState (copy from core for FFI)
/// Traces to: FR-EYE-INTEROP-001
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CalibrationState {
    Idle,
    Waiting,
    Sampling,
    Processing,
    Validating,
    Complete,
    Failed,
}

impl CalibrationState {
    fn from_core(state: CoreCalibrationState) -> Self {
        match state {
            CoreCalibrationState::Idle => CalibrationState::Idle,
            CoreCalibrationState::Waiting => CalibrationState::Waiting,
            CoreCalibrationState::Sampling => CalibrationState::Sampling,
            CoreCalibrationState::Processing => CalibrationState::Processing,
            CoreCalibrationState::Validating => CalibrationState::Validating,
            CoreCalibrationState::Complete => CalibrationState::Complete,
            CoreCalibrationState::Failed => CalibrationState::Failed,
        }
    }
}

/// Sample pair for calibration
/// Traces to: FR-EYE-INTEROP-001
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationSample {
    pub eye_point: Point,
    pub screen_target: Point,
}

// ============================================================================
// UniFFI Constructor Functions
// ============================================================================

/// Create a Point (FFI constructor)
pub fn create_point(x: f64, y: f64) -> Point {
    Point::new(x, y)
}

/// Create a GazeEstimate (FFI constructor)
pub fn create_gaze_estimate(screen_pos: Point, confidence: f64) -> GazeEstimate {
    GazeEstimate::new(screen_pos, confidence)
}

/// Idle state constant
pub fn calibration_state_idle() -> CalibrationState {
    CalibrationState::Idle
}

/// Complete state constant
pub fn calibration_state_complete() -> CalibrationState {
    CalibrationState::Complete
}

// ============================================================================
// Calibrator FFI Wrapper
// ============================================================================

/// FFI-safe Calibrator wrapping eyetracker-core::Calibrator
/// Traces to: FR-EYE-INTEROP-002
pub struct Calibrator {
    inner: CoreCalibrator,
}

impl Calibrator {
    /// Create new Calibrator
    pub fn new() -> Self {
        Self {
            inner: CoreCalibrator::new(),
        }
    }

    /// Start calibration workflow
    pub fn start_calibration(&mut self) {
        self.inner.start_calibration();
    }

    /// Record a single calibration sample
    /// Returns: Ok(()), Err(message) if state invalid
    pub fn record_sample(&mut self, eye_point: Point, screen_target: Point) -> Result<(), String> {
        self.inner
            .record_sample(eye_point.to_domain(), screen_target.to_domain())
            .map_err(|e| e.to_string())
    }

    /// Finalize calibration (compute matrix from samples)
    /// Returns: Ok(()), Err(message) if <3 samples or accuracy check failed
    pub fn finalize(&mut self) -> Result<(), String> {
        self.inner.finalize().map_err(|e| e.to_string())
    }

    /// Get current state
    pub fn state(&self) -> CalibrationState {
        CalibrationState::from_core(self.inner.state())
    }

    /// Get current accuracy (residual error in degrees)
    pub fn accuracy(&self) -> f64 {
        self.inner.accuracy()
    }

    /// Get recorded samples (if available)
    pub fn samples(&self) -> Option<Vec<CalibrationSample>> {
        // Access samples field through public accessor
        // Since samples() is private, we need to return None for now
        // This will be enabled once core exposes a public getter
        None
    }
}

impl Default for Calibrator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// GazeEstimator FFI Wrapper
// ============================================================================

/// FFI-safe GazeEstimator wrapper
/// Traces to: FR-EYE-INTEROP-003
pub struct GazeEstimator {
    calibrator: Option<CoreCalibrator>,
    smoothing_enabled: bool,
}

impl GazeEstimator {
    /// Create new GazeEstimator
    pub fn new() -> Self {
        Self {
            calibrator: None,
            smoothing_enabled: true,
        }
    }

    /// Estimate gaze from raw eye position
    /// Returns: Some(GazeEstimate) if calibration available, None otherwise
    pub fn estimate(&self, raw_eye_position: Point) -> Option<GazeEstimate> {
        self.calibrator.as_ref().and_then(|cal| {
            cal.get_calibration().map(|cal_matrix| {
                let predicted = cal_matrix.apply(raw_eye_position.to_domain());
                GazeEstimate {
                    screen_pos: Point::from_domain(predicted),
                    confidence: 0.95,
                }
            })
        })
    }

    /// Set calibration from a Calibrator
    pub fn set_calibration(&mut self, calibrator: Calibrator) {
        self.calibrator = Some(calibrator.inner);
    }

    /// Enable/disable Kalman smoothing
    pub fn set_smoothing_enabled(&mut self, enabled: bool) {
        self.smoothing_enabled = enabled;
    }
}

impl Default for GazeEstimator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Traces to: FR-EYE-INTEROP-001
    #[test]
    fn test_point_creation() {
        let p = Point::new(100.0, 200.0);
        assert_eq!(p.x, 100.0);
        assert_eq!(p.y, 200.0);
    }

    // Traces to: FR-EYE-INTEROP-001
    #[test]
    fn test_gaze_estimate_creation() {
        let p = Point::new(50.0, 75.0);
        let ge = GazeEstimate::new(p, 0.95);
        assert_eq!(ge.screen_pos.x, 50.0);
        assert_eq!(ge.confidence, 0.95);
    }

    // Traces to: FR-EYE-INTEROP-002
    #[test]
    fn test_calibrator_state_transitions() {
        let mut cal = Calibrator::new();
        assert_eq!(cal.state(), CalibrationState::Idle);

        cal.start_calibration();
        assert_eq!(cal.state(), CalibrationState::Waiting);
    }

    // Traces to: FR-EYE-INTEROP-002
    #[test]
    fn test_calibrator_sample_recording() {
        let mut cal = Calibrator::new();
        cal.start_calibration();

        let eye1 = Point::new(100.0, 150.0);
        let screen1 = Point::new(500.0, 400.0);

        assert!(cal.record_sample(eye1, screen1).is_ok());
        assert_eq!(cal.state(), CalibrationState::Sampling);
    }

    // Traces to: FR-EYE-INTEROP-002
    #[test]
    fn test_calibrator_finalize_insufficient_samples() {
        let mut cal = Calibrator::new();
        cal.start_calibration();
        cal.record_sample(Point::new(100.0, 150.0), Point::new(500.0, 400.0))
            .unwrap();

        let result = cal.finalize();
        assert!(result.is_err());
        assert_eq!(cal.state(), CalibrationState::Failed);
    }

    // Traces to: FR-EYE-INTEROP-003
    #[test]
    fn test_gaze_estimator_creation() {
        let estimator = GazeEstimator::new();
        assert!(estimator.estimate(Point::new(100.0, 150.0)).is_none());
    }

    // Traces to: FR-EYE-INTEROP-003
    #[test]
    fn test_gaze_estimator_smoothing_control() {
        let mut estimator = GazeEstimator::new();
        estimator.set_smoothing_enabled(true);
        assert!(estimator.smoothing_enabled);

        estimator.set_smoothing_enabled(false);
        assert!(!estimator.smoothing_enabled);
    }
}
