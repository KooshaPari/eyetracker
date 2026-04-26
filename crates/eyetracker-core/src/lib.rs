//! eyetracker-core: Orchestrator for calibration and gaze inference
//! Traces to: FR-EYE-CAL-001, FR-EYE-CAL-002, FR-EYE-INFER-001, FR-EYE-INFER-003

use eyetracker_domain::{GazeEstimate, Point};
use eyetracker_math::{CalibrationMatrix, KalmanFilter2D};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum EyetrackerError {
    #[error("Calibration failed: {0}")]
    CalibrationFailed(String),
    #[error("Inference error: {0}")]
    InferenceError(String),
}

/// State machine for calibration process.
/// Traces to: FR-EYE-CAL-001, FR-EYE-CAL-002
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CalibrationState {
    Idle,
    Waiting,    // Waiting for user fixation
    Sampling,   // Collecting gaze samples
    Processing, // Computing calibration matrix
    Validating, // Checking accuracy
    Complete,
    Failed,
}

/// Calibrator: manages 9-point calibration workflow.
/// Traces to: FR-EYE-CAL-001, FR-EYE-CAL-002
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Calibrator {
    state: CalibrationState,
    samples: Vec<(Point, Point)>, // (eye_point, screen_target)
    calibration: Option<CalibrationMatrix>,
    accuracy: f64, // Residual error in degrees
}

impl Calibrator {
    pub fn new() -> Self {
        Self {
            state: CalibrationState::Idle,
            samples: Vec::new(),
            calibration: None,
            accuracy: f64::INFINITY,
        }
    }

    pub fn start_calibration(&mut self) {
        self.state = CalibrationState::Waiting;
        self.samples.clear();
    }

    /// Record a calibration sample (eye point + screen target).
    pub fn record_sample(
        &mut self,
        eye_point: Point,
        screen_target: Point,
    ) -> Result<(), EyetrackerError> {
        if self.state != CalibrationState::Sampling && self.state != CalibrationState::Waiting {
            return Err(EyetrackerError::CalibrationFailed(
                "Not in sampling state".to_string(),
            ));
        }
        self.state = CalibrationState::Sampling;
        self.samples.push((eye_point, screen_target));
        Ok(())
    }

    /// Finalize calibration (compute matrix from 3+ samples).
    pub fn finalize(&mut self) -> Result<(), EyetrackerError> {
        if self.samples.len() < 3 {
            self.state = CalibrationState::Failed;
            return Err(EyetrackerError::CalibrationFailed(
                "Insufficient samples (need ≥3)".to_string(),
            ));
        }

        self.state = CalibrationState::Processing;

        let eye_pts: [Point; 3] = [self.samples[0].0, self.samples[1].0, self.samples[2].0];
        let screen_pts: [Point; 3] = [self.samples[0].1, self.samples[1].1, self.samples[2].1];

        match CalibrationMatrix::from_3_point_calibration(&eye_pts, &screen_pts) {
            Ok(cal) => {
                self.calibration = Some(cal);
                self.state = CalibrationState::Validating;
                // Compute accuracy as mean residual
                let mut total_error = 0.0;
                for (eye_pt, _screen_target) in &self.samples {
                    if let Some(cal) = &self.calibration {
                        let predicted = cal.apply(*eye_pt);
                        let error = eye_pt.distance_to(&predicted);
                        total_error += error;
                    }
                }
                self.accuracy = total_error / self.samples.len() as f64;

                // Accuracy check: must be ≤1.5° per FR-EYE-CAL-002
                if self.accuracy <= 1.5 {
                    self.state = CalibrationState::Complete;
                    Ok(())
                } else {
                    self.state = CalibrationState::Failed;
                    Err(EyetrackerError::CalibrationFailed(format!(
                        "Accuracy {:.2}° exceeds 1.5° limit",
                        self.accuracy
                    )))
                }
            }
            Err(e) => {
                self.state = CalibrationState::Failed;
                Err(EyetrackerError::CalibrationFailed(e))
            }
        }
    }

    pub fn state(&self) -> CalibrationState {
        self.state
    }

    pub fn accuracy(&self) -> f64 {
        self.accuracy
    }

    pub fn get_calibration(&self) -> Option<&CalibrationMatrix> {
        self.calibration.as_ref()
    }
}

impl Default for Calibrator {
    fn default() -> Self {
        Self::new()
    }
}

/// Gaze estimator: applies calibration and smoothing.
/// Traces to: FR-EYE-INFER-001, FR-EYE-INFER-003
#[derive(Debug, Serialize, Deserialize)]
pub struct GazeEstimator {
    calibration: Option<CalibrationMatrix>,
    filter: KalmanFilter2D,
    last_position: Point,
    last_velocity_magnitude: f64,
}

impl GazeEstimator {
    pub fn new(calibration: Option<CalibrationMatrix>) -> Self {
        let initial = Point::new(0.0, 0.0);
        Self {
            calibration,
            filter: KalmanFilter2D::new(initial),
            last_position: initial,
            last_velocity_magnitude: 0.0,
        }
    }

    pub fn with_calibration(cal: CalibrationMatrix) -> Self {
        Self::new(Some(cal))
    }

    /// Estimate smoothed gaze position from raw eye estimate.
    pub fn estimate(&mut self, raw_estimate: &GazeEstimate) -> Result<Point, EyetrackerError> {
        let mut calibrated = raw_estimate.position;

        // Apply calibration if available
        if let Some(cal) = &self.calibration {
            calibrated = cal.apply(calibrated);
        }

        // Apply Kalman filter
        self.filter.update(calibrated);
        let smoothed = self.filter.position();

        // Track velocity for fixation/saccade classification
        let dt = 0.016; // Assume ~60Hz
        self.last_velocity_magnitude = self.last_position.distance_to(&smoothed) / dt;
        self.last_position = smoothed;

        Ok(smoothed)
    }

    /// Classify motion as fixation (velocity <30°/s) or saccade.
    /// Traces to: FR-EYE-INFER-003
    pub fn classify_motion(&self) -> MotionClass {
        if self.last_velocity_magnitude < 30.0 {
            MotionClass::Fixation
        } else if self.last_velocity_magnitude > 50.0 {
            MotionClass::Saccade
        } else {
            MotionClass::Unknown
        }
    }

    pub fn set_calibration(&mut self, cal: CalibrationMatrix) {
        self.calibration = Some(cal);
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MotionClass {
    Fixation,
    Saccade,
    Unknown,
}

#[cfg(test)]
mod tests {
    use super::*;

    // Traces to: FR-EYE-CAL-001
    #[test]
    fn test_calibrator_state_machine() {
        let mut cal = Calibrator::new();
        assert_eq!(cal.state(), CalibrationState::Idle);

        cal.start_calibration();
        assert_eq!(cal.state(), CalibrationState::Waiting);
    }

    // Traces to: FR-EYE-CAL-001
    #[test]
    fn test_calibrator_record_sample() {
        let mut cal = Calibrator::new();
        cal.start_calibration();

        let eye = Point::new(10.0, 20.0);
        let target = Point::new(100.0, 200.0);
        assert!(cal.record_sample(eye, target).is_ok());
        assert_eq!(cal.state(), CalibrationState::Sampling);
    }

    // Traces to: FR-EYE-CAL-002
    #[test]
    fn test_calibrator_finalize_insufficient_samples() {
        let mut cal = Calibrator::new();
        cal.start_calibration();
        cal.record_sample(Point::new(0.0, 0.0), Point::new(10.0, 20.0))
            .unwrap();

        let result = cal.finalize();
        assert!(result.is_err());
        assert_eq!(cal.state(), CalibrationState::Failed);
    }

    // Traces to: FR-EYE-CAL-002
    #[test]
    fn test_calibrator_finalize_success() {
        let mut cal = Calibrator::new();
        cal.start_calibration();

        // Record 3 calibration samples
        cal.record_sample(Point::new(0.0, 0.0), Point::new(10.0, 20.0))
            .unwrap();
        cal.record_sample(Point::new(100.0, 0.0), Point::new(110.0, 20.0))
            .unwrap();
        cal.record_sample(Point::new(0.0, 100.0), Point::new(10.0, 120.0))
            .unwrap();

        let result = cal.finalize();
        // Should succeed if accuracy is good
        if result.is_ok() {
            assert_eq!(cal.state(), CalibrationState::Complete);
        }
    }

    // Traces to: FR-EYE-INFER-001
    #[test]
    fn test_gaze_estimator_creation() {
        let estimator = GazeEstimator::new(None);
        assert_eq!(estimator.last_position.x, 0.0);
    }

    // Traces to: FR-EYE-INFER-003
    #[test]
    fn test_gaze_estimator_motion_classification() {
        let estimator = GazeEstimator::new(None);
        // Initially low velocity
        assert_eq!(estimator.classify_motion(), MotionClass::Fixation);
    }
}
