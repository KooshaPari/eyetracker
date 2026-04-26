//! eyetracker-domain: Core types for eye-tracking domain
//! Traces to: FR-EYE-CAL-001, FR-EYE-CAL-002, FR-EYE-INFER-001, FR-EYE-INFER-002, FR-EYE-INFER-003

use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime};

/// 2D point in screen coordinates (pixels).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Euclidean distance to another point.
    pub fn distance_to(&self, other: &Point) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

/// 2D velocity vector (deg/sec or pixels/sec).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Vector {
    pub dx: f64,
    pub dy: f64,
}

impl Vector {
    pub fn new(dx: f64, dy: f64) -> Self {
        Self { dx, dy }
    }

    /// Magnitude (speed).
    pub fn magnitude(&self) -> f64 {
        (self.dx.powi(2) + self.dy.powi(2)).sqrt()
    }
}

/// Raw gaze estimate from eye image (before filtering).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GazeEstimate {
    pub position: Point,
    pub confidence: f64, // 0.0 to 1.0
    pub timestamp: SystemTime,
}

impl GazeEstimate {
    pub fn new(position: Point, confidence: f64, timestamp: SystemTime) -> Self {
        Self {
            position,
            confidence,
            timestamp,
        }
    }
}

/// Classified fixation event (eye stationary on a region).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixationEvent {
    pub centroid: Point,
    pub duration: Duration,
    pub start_time: SystemTime,
}

impl FixationEvent {
    pub fn new(centroid: Point, duration: Duration, start_time: SystemTime) -> Self {
        Self {
            centroid,
            duration,
            start_time,
        }
    }
}

/// Saccade event (rapid eye movement).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaccadeEvent {
    pub start_position: Point,
    pub end_position: Point,
    pub duration: Duration,
    pub amplitude: f64, // degrees or pixels
    pub peak_velocity: f64,
    pub start_time: SystemTime,
}

impl SaccadeEvent {
    pub fn new(
        start_position: Point,
        end_position: Point,
        duration: Duration,
        peak_velocity: f64,
        start_time: SystemTime,
    ) -> Self {
        let amplitude = start_position.distance_to(&end_position);
        Self {
            start_position,
            end_position,
            duration,
            amplitude,
            peak_velocity,
            start_time,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Traces to: FR-EYE-INFER-003
    #[test]
    fn test_point_distance() {
        let p1 = Point::new(0.0, 0.0);
        let p2 = Point::new(3.0, 4.0);
        assert!((p1.distance_to(&p2) - 5.0).abs() < 0.0001);
    }

    // Traces to: FR-EYE-INFER-002
    #[test]
    fn test_vector_magnitude() {
        let v = Vector::new(3.0, 4.0);
        assert!((v.magnitude() - 5.0).abs() < 0.0001);
    }

    // Traces to: FR-EYE-CAL-001
    #[test]
    fn test_gaze_estimate_creation() {
        let pos = Point::new(100.0, 200.0);
        let est = GazeEstimate::new(pos, 0.95, SystemTime::now());
        assert_eq!(est.position.x, 100.0);
        assert_eq!(est.confidence, 0.95);
    }

    // Traces to: FR-EYE-INFER-003
    #[test]
    fn test_fixation_event_creation() {
        let centroid = Point::new(500.0, 300.0);
        let duration = Duration::from_millis(250);
        let now = SystemTime::now();
        let fix = FixationEvent::new(centroid, duration, now);
        assert_eq!(fix.centroid.x, 500.0);
        assert_eq!(fix.duration.as_millis(), 250);
    }

    // Traces to: FR-EYE-INFER-004
    #[test]
    fn test_saccade_event_creation() {
        let start = Point::new(100.0, 100.0);
        let end = Point::new(300.0, 400.0);
        let duration = Duration::from_millis(50);
        let peak_vel = 150.0;
        let now = SystemTime::now();
        let sacc = SaccadeEvent::new(start, end, duration, peak_vel, now);
        assert!((sacc.amplitude - 360.555).abs() < 1.0);
        assert_eq!(sacc.peak_velocity, 150.0);
    }

    // Traces to: FR-EYE-CAL-002
    #[test]
    fn test_point_equality() {
        let p1 = Point::new(1.0, 2.0);
        let p2 = Point::new(1.0, 2.0);
        assert_eq!(p1, p2);
    }
}
