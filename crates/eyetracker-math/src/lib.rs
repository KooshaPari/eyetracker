//! eyetracker-math: Kalman filtering and calibration matrix solving
//! Traces to: FR-EYE-INFER-002, FR-EYE-CAL-001

use eyetracker_domain::{Point, Vector};
use serde::{Deserialize, Serialize};

/// 2D Kalman filter for gaze smoothing (velocity model).
/// Traces to: FR-EYE-INFER-002
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KalmanFilter2D {
    // State: [x, y, vx, vy]
    state: [f64; 4],
    // Covariance matrix (4x4, stored as flat array)
    covariance: [f64; 16],
    // Process noise
    q: f64,
    // Measurement noise
    r: f64,
}

impl KalmanFilter2D {
    /// Create a new Kalman filter with initial position.
    pub fn new(initial_pos: Point) -> Self {
        let mut covariance = [0.0; 16];
        // Initialize with high uncertainty
        for i in 0..4 {
            covariance[i * 4 + i] = 1.0;
        }

        Self {
            state: [initial_pos.x, initial_pos.y, 0.0, 0.0],
            covariance,
            q: 0.01,
            r: 1.0,
        }
    }

    /// Update filter with a new measurement (gaze point).
    pub fn update(&mut self, measurement: Point) {
        // Simplified Kalman update: blend measurement with prediction
        let alpha = 0.5; // Smoothing factor
        self.state[0] = alpha * measurement.x + (1.0 - alpha) * self.state[0];
        self.state[1] = alpha * measurement.y + (1.0 - alpha) * self.state[1];
    }

    /// Predict next state (simple velocity extrapolation).
    pub fn predict(&mut self, dt: f64) {
        self.state[0] += self.state[2] * dt;
        self.state[1] += self.state[3] * dt;
    }

    /// Get smoothed gaze position.
    pub fn position(&self) -> Point {
        Point::new(self.state[0], self.state[1])
    }

    /// Get estimated velocity.
    pub fn velocity(&self) -> Vector {
        Vector::new(self.state[2], self.state[3])
    }

    /// Reset filter state (e.g., on saccade).
    pub fn reset(&mut self, pos: Point) {
        self.state = [pos.x, pos.y, 0.0, 0.0];
    }
}

/// 3-point calibration matrix solver.
/// Solves for affine transform: screen_coords = matrix * eye_coords
/// Traces to: FR-EYE-CAL-001
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationMatrix {
    // 2x3 affine matrix (row-major): [[a, b, tx], [c, d, ty]]
    matrix: [f64; 6],
}

impl CalibrationMatrix {
    /// Create identity calibration (no transform).
    pub fn identity() -> Self {
        Self {
            matrix: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        }
    }

    /// Solve 3-point calibration from eye→screen correspondences.
    /// Takes 3 pairs of (eye_point, screen_point).
    pub fn from_3_point_calibration(
        eye_points: &[Point; 3],
        screen_points: &[Point; 3],
    ) -> Result<Self, String> {
        // Simple least-squares solve for affine matrix
        // eye = [x, y, 1], screen = [sx, sy]
        // We solve: [a b tx] @ [x, y, 1]^T = [sx]
        //           [c d ty]                   [sy]

        let mut a_matrix = [[0.0; 3]; 3];
        let mut b_vec = [0.0; 3];

        for i in 0..3 {
            a_matrix[i] = [eye_points[i].x, eye_points[i].y, 1.0];
            b_vec[i] = screen_points[i].x;
        }

        let det = determinant_3x3(&a_matrix);
        if det.abs() < 1e-6 {
            return Err("Singular calibration matrix".to_string());
        }

        // Simple Cramer's rule for x-component
        let x_coeff = cramer_3x3(&a_matrix, &b_vec, det);

        // Now solve for y-component
        let mut b_vec_y = [0.0; 3];
        for i in 0..3 {
            b_vec_y[i] = screen_points[i].y;
        }
        let y_coeff = cramer_3x3(&a_matrix, &b_vec_y, det);

        Ok(Self {
            matrix: [
                x_coeff[0], x_coeff[1], x_coeff[2], y_coeff[0], y_coeff[1], y_coeff[2],
            ],
        })
    }

    /// Apply calibration: transform eye coords to screen coords.
    pub fn apply(&self, eye_point: Point) -> Point {
        let x = self.matrix[0] * eye_point.x + self.matrix[1] * eye_point.y + self.matrix[2];
        let y = self.matrix[3] * eye_point.x + self.matrix[4] * eye_point.y + self.matrix[5];
        Point::new(x, y)
    }

    /// Get raw matrix coefficients.
    pub fn coefficients(&self) -> [f64; 6] {
        self.matrix
    }
}

fn determinant_3x3(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

fn cramer_3x3(a: &[[f64; 3]; 3], b: &[f64; 3], det: f64) -> [f64; 3] {
    // Solve Ax = b using Cramer's rule
    let mut x = [0.0; 3];

    for j in 0..3 {
        let mut a_j = *a;
        for i in 0..3 {
            a_j[i][j] = b[i];
        }
        x[j] = determinant_3x3(&a_j) / det;
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    // Traces to: FR-EYE-INFER-002
    #[test]
    fn test_kalman_filter_initialization() {
        let pos = Point::new(100.0, 200.0);
        let kf = KalmanFilter2D::new(pos);
        assert_eq!(kf.position().x, 100.0);
        assert_eq!(kf.position().y, 200.0);
    }

    // Traces to: FR-EYE-INFER-002
    #[test]
    fn test_kalman_filter_update() {
        let pos = Point::new(100.0, 200.0);
        let mut kf = KalmanFilter2D::new(pos);
        let new_meas = Point::new(110.0, 210.0);
        kf.update(new_meas);
        let smoothed = kf.position();
        // Smoothed should be between original and measurement
        assert!(smoothed.x > 100.0 && smoothed.x < 110.0);
    }

    // Traces to: FR-EYE-CAL-001
    #[test]
    fn test_calibration_matrix_identity() {
        let cal = CalibrationMatrix::identity();
        let point = Point::new(10.0, 20.0);
        let result = cal.apply(point);
        assert!((result.x - 10.0).abs() < 0.0001);
        assert!((result.y - 20.0).abs() < 0.0001);
    }

    // Traces to: FR-EYE-CAL-001
    #[test]
    fn test_calibration_matrix_3point() {
        let eye_points = [
            Point::new(0.0, 0.0),
            Point::new(100.0, 0.0),
            Point::new(0.0, 100.0),
        ];
        let screen_points = [
            Point::new(10.0, 20.0),
            Point::new(110.0, 20.0),
            Point::new(10.0, 120.0),
        ];

        let cal = CalibrationMatrix::from_3_point_calibration(&eye_points, &screen_points)
            .expect("Failed to create calibration");
        let result = cal.apply(eye_points[0]);
        assert!((result.x - screen_points[0].x).abs() < 0.1);
        assert!((result.y - screen_points[0].y).abs() < 0.1);
    }

    // Traces to: FR-EYE-INFER-002
    #[test]
    fn test_kalman_filter_reset() {
        let mut kf = KalmanFilter2D::new(Point::new(100.0, 200.0));
        kf.reset(Point::new(500.0, 600.0));
        assert_eq!(kf.position().x, 500.0);
        assert_eq!(kf.position().y, 600.0);
    }
}
