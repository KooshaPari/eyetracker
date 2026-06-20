//! 2D Kalman filter with constant-velocity model for gaze smoothing.
//!
//! Traces to: FR-EYE-INFER-002
//!
//! The filter maintains a 4-element state vector:
//!   [x, y, vx, vy]ᵀ
//!
//! where (x, y) is the smoothed gaze position and (vx, vy) is the estimated
//! velocity.  A standard predict–update cycle is run on every call to
//! [`KalmanState2D::update`].  During saccades consumers should call
//! [`KalmanState2D::reset`] (or use [`GazeSmoother`] which does this
//! automatically) so the filter does not lag behind the rapid jump.

use nalgebra::SMatrix;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default process-noise variance for position (pixels² per frame).
const DEFAULT_PROCESS_NOISE_POS: f32 = 1.0;

/// Default process-noise variance for velocity (pixels² per frame).
const DEFAULT_PROCESS_NOISE_VEL: f32 = 10.0;

/// Default measurement-noise variance (pixels²).
const DEFAULT_MEASUREMENT_NOISE: f32 = 25.0;

/// Default dt assumed per call when no timing information is available.
const DEFAULT_DT: f32 = 1.0;

// ---------------------------------------------------------------------------
// KalmanState2D
// ---------------------------------------------------------------------------

/// A 2-D Kalman filter with a constant-velocity model.
///
/// State vector (4×1):  `[x, y, vx, vy]ᵀ`
///
/// | Matrix | Dims | Purpose                        |
/// |--------|------|--------------------------------|
/// | `F`    | 4×4  | State transition               |
/// | `H`    | 2×4  | Measurement model               |
/// | `P`    | 4×4  | Estimate covariance             |
/// | `Q`    | 4×4  | Process noise covariance        |
/// | `R`    | 2×2  | Measurement noise covariance    |
#[derive(Debug, Clone)]
pub struct KalmanState2D {
    /// State estimate: [x, y, vx, vy]ᵀ
    pub state: nalgebra::SVector<f32, 4>,

    /// State-transition matrix (4×4)
    pub f: SMatrix<f32, 4, 4>,

    /// Measurement matrix (2×4)
    pub h: SMatrix<f32, 2, 4>,

    /// Estimate covariance (4×4)
    pub p: SMatrix<f32, 4, 4>,

    /// Process-noise covariance (4×4) — reproduced here so users can inspect it.
    pub q: SMatrix<f32, 4, 4>,

    /// Measurement-noise covariance (2×2).
    pub r: SMatrix<f32, 2, 2>,
}

impl KalmanState2D {
    /// Create a new filter with default noise levels.
    pub fn new() -> Self {
        Self::with_noise(DEFAULT_PROCESS_NOISE_POS, DEFAULT_PROCESS_NOISE_VEL, DEFAULT_MEASUREMENT_NOISE)
    }

    /// Create a filter with explicit noise variances.
    ///
    /// * `process_noise_pos` — variance for position components of Q
    /// * `process_noise_vel` — variance for velocity components of Q
    /// * `measurement_noise` — variance for both axes of R
    pub fn with_noise(process_noise_pos: f32, process_noise_vel: f32, measurement_noise: f32) -> Self {
        let dt = DEFAULT_DT;

        // F = [1 0 dt 0]
        //     [0 1 0 dt]
        //     [0 0 1  0]
        //     [0 0 0  1]
        let f = SMatrix::<f32, 4, 4>::new(
            1.0, 0.0, dt,  0.0,
            0.0, 1.0, 0.0, dt,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        );

        // H = [1 0 0 0]
        //     [0 1 0 0]
        let h = SMatrix::<f32, 2, 4>::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
        );

        // Initial covariance: moderate uncertainty
        let p = SMatrix::<f32, 4, 4>::identity() * 100.0;

        // Q — diagonal process noise (position and velocity components)
        let q = SMatrix::<f32, 4, 4>::from_diagonal(&nalgebra::SVector::<f32, 4>::new(
            process_noise_pos,
            process_noise_pos,
            process_noise_vel,
            process_noise_vel,
        ));

        // R — diagonal measurement noise
        let r = SMatrix::<f32, 2, 2>::from_diagonal(&nalgebra::SVector::<f32, 2>::new(
            measurement_noise,
            measurement_noise,
        ));

        Self {
            state: nalgebra::SVector::<f32, 4>::zeros(),
            f,
            h,
            p,
            q,
            r,
        }
    }

    /// Run one predict–update cycle given a noisy measurement `(x, y)`.
    ///
    /// Returns the **filtered** (smoothed) position `(x, y)`.
    pub fn update(&mut self, measurement_x: f32, measurement_y: f32) -> (f32, f32) {
        // ---- predict ----
        self.state = self.f * self.state;
        self.p = self.f * self.p * self.f.transpose() + self.q;

        // ---- update ----
        // Innovation: y = z - H * x
        let z = nalgebra::SVector::<f32, 2>::new(measurement_x, measurement_y);
        let hx = self.h * self.state;
        let innovation = z - hx;

        // Innovation covariance: S = H * P * Hᵀ + R
        let s = self.h * self.p * self.h.transpose() + self.r;

        // Kalman gain:  K = P * Hᵀ * S⁻¹
        // Use the stable (P * Hᵀ) * S⁻¹ order rather than computing S⁻¹ * H * P.
        let ph_t = self.p * self.h.transpose();
        // nalgebra provides `spsolve` or we can use `s.try_inverse()`.
        // S is 2×2 — a direct inverse is fine.
        let k = match s.try_inverse() {
            Some(s_inv) => ph_t * s_inv,
            None => {
                // If S is singular fall back to zero gain.
                tracing::warn!("Kalman filter innovation covariance is singular; skipping update");
                return (self.state[0], self.state[1]);
            }
        };

        // State update: x = x + K * y
        self.state = self.state + k * innovation;

        // Covariance update: P = (I - K * H) * P
        let kh = k * self.h;
        self.p = (SMatrix::<f32, 4, 4>::identity() - kh) * self.p;

        (self.state[0], self.state[1])
    }

    /// Reset the filter to its initial (zero) state with default covariance.
    pub fn reset(&mut self) {
        self.state = nalgebra::SVector::<f32, 4>::zeros();
        self.p = SMatrix::<f32, 4, 4>::identity() * 100.0;
    }

    /// Return the current estimated velocity `(vx, vy)`.
    pub fn velocity(&self) -> (f32, f32) {
        (self.state[2], self.state[3])
    }
}

impl Default for KalmanState2D {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// GazeSmoother
// ---------------------------------------------------------------------------

/// A higher-level wrapper around [`KalmanState2D`] that handles saccade-triggered
/// resets and keeps noise parameters configurable.
#[derive(Debug, Clone)]
pub struct GazeSmoother {
    filter: KalmanState2D,
    process_noise_pos: f32,
    process_noise_vel: f32,
    measurement_noise: f32,
}

impl GazeSmoother {
    /// Create a new smoother with default noise levels.
    pub fn new() -> Self {
        Self::with_noise(
            DEFAULT_PROCESS_NOISE_POS,
            DEFAULT_PROCESS_NOISE_VEL,
            DEFAULT_MEASUREMENT_NOISE,
        )
    }

    /// Create a smoother with explicit noise variances.
    ///
    /// * `process_noise_pos` — position-component process noise (higher = filter trusts
    ///   measurements more / smooths less).
    /// * `process_noise_vel` — velocity-component process noise.
    /// * `measurement_noise` — measurement variance (higher = filter smooths more).
    pub fn with_noise(process_noise_pos: f32, process_noise_vel: f32, measurement_noise: f32) -> Self {
        Self {
            filter: KalmanState2D::with_noise(process_noise_pos, process_noise_vel, measurement_noise),
            process_noise_pos,
            process_noise_vel,
            measurement_noise,
        }
    }

    /// Feed a raw gaze measurement and retrieve the smoothed position.
    ///
    /// When `is_saccade` is `true` the internal filter is **reset** before
    /// processing the measurement so that the rapid eye movement does not
    /// create a long lag tail.  The measurement itself is still used as the
    /// post-reset state.
    pub fn smooth(&mut self, x: f32, y: f32, is_saccade: bool) -> (f32, f32) {
        if is_saccade {
            self.filter.reset();
        }
        self.filter.update(x, y)
    }

    /// Access the underlying Kalman filter (e.g. to read velocity).
    pub fn filter(&self) -> &KalmanState2D {
        &self.filter
    }

    /// Mutable access to the underlying Kalman filter.
    pub fn filter_mut(&mut self) -> &mut KalmanState2D {
        &mut self.filter
    }

    /// Rebuild the inner filter with current noise settings (useful after tuning).
    pub fn reconfigure(&mut self) {
        self.filter = KalmanState2D::with_noise(
            self.process_noise_pos,
            self.process_noise_vel,
            self.measurement_noise,
        );
    }
}

impl Default for GazeSmoother {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// After feeding the same stationary point multiple times the filter
    /// should converge toward that point with very low error.
    #[test]
    fn test_convergence() {
        let mut kf = KalmanState2D::new();

        // Feed (100, 200) twenty times.
        let target_x = 100.0;
        let target_y = 200.0;
        let (mut sx, mut sy) = (0.0, 0.0);
        for _ in 0..20 {
            (sx, sy) = kf.update(target_x, target_y);
        }

        // After 20 iterations the error should be well under 1 pixel.
        assert!(
            (sx - target_x).abs() < 1.0,
            "x error too large: {} vs {}",
            sx,
            target_x,
        );
        assert!(
            (sy - target_y).abs() < 1.0,
            "y error too large: {} vs {}",
            sy,
            target_y,
        );

        // Velocity should have decayed toward zero.
        let (vx, vy) = kf.velocity();
        assert!(
            vx.abs() < 0.5,
            "vx should be near zero after convergence, got {}",
            vx,
        );
        assert!(
            vy.abs() < 0.5,
            "vy should be near zero after convergence, got {}",
            vy,
        );
    }

    /// When fed consistently moving measurements the velocity estimate should
    /// be non-zero and point in the correct direction.
    #[test]
    fn test_velocity() {
        let mut kf = KalmanState2D::new();

        // Feed a point that moves right 2 px per iteration.
        for i in 0..30 {
            kf.update(i as f32 * 2.0, 0.0);
        }

        let (vx, vy) = kf.velocity();

        // After 30 steps the velocity should be positive in x and near zero in y.
        assert!(
            vx > 0.5,
            "vx should be positive for rightward movement, got {}",
            vx,
        );
        assert!(
            vy.abs() < 1.0,
            "vy should be near zero, got {}",
            vy,
        );

        // Now feed leftward movement.
        let mut kf2 = KalmanState2D::new();
        for i in 0..30 {
            kf2.update(-i as f32 * 2.0, 0.0);
        }

        let (vx2, _) = kf2.velocity();
        assert!(
            vx2 < -0.5,
            "vx should be negative for leftward movement, got {}",
            vx2,
        );
    }

    /// After `reset()` the filter should behave as if newly created:
    /// state is zeroed, covariance is large, and it converges to measurements
    /// over subsequent updates.
    #[test]
    fn test_reset() {
        let mut kf = KalmanState2D::new();

        // Move the filter away from zero so it has a confident estimate.
        for _ in 0..15 {
            kf.update(100.0, 200.0);
        }

        // Reset.
        kf.reset();

        // State should be zero.
        assert_eq!(kf.state[0], 0.0);
        assert_eq!(kf.state[1], 0.0);
        assert_eq!(kf.state[2], 0.0);
        assert_eq!(kf.state[3], 0.0);

        // Covariance should have been restored to the large initial value.
        assert!((kf.p[(0, 0)] - 100.0).abs() < 1.0, "P[0,0] should be ~100 after reset, got {}", kf.p[(0, 0)]);

        // The first post-reset update: the Kalman gain is high (~0.89 given
        // default noise) but not 1.0, so the estimate is a weighted blend
        // that favours the measurement.
        let (sx, sy) = kf.update(50.0, 75.0);
        assert!(
            sx > 30.0 && sx < 50.0,
            "first post-reset x should be between initial (0) and measurement (50), got {}",
            sx,
        );
        assert!(
            sy > 45.0 && sy < 75.0,
            "first post-reset y should be between initial (0) and measurement (75), got {}",
            sy,
        );

        // Subsequent updates should converge toward the measurement.
        let (sx2, sy2) = kf.update(50.0, 75.0);
        assert!(
            (sx2 - 50.0).abs() < (sx - 50.0).abs(),
            "x should get closer to measurement on second update ({} → {})",
            sx,
            sx2,
        );
        assert!(
            (sy2 - 75.0).abs() < (sy - 75.0).abs(),
            "y should get closer to measurement on second update ({} → {})",
            sy,
            sy2,
        );
    }

    /// The `GazeSmoother` should reset its internal filter when `is_saccade`
    /// is `true`, preventing large lag.
    ///
    /// After a saccade reset the Kalman gain is high, so the output jumps
    /// much closer to the new measurement than without reset.
    #[test]
    fn test_smoother_saccade_reset() {
        let mut smoother = GazeSmoother::new();

        // Converge on a point.
        for _ in 0..10 {
            smoother.smooth(200.0, 300.0, false);
        }

        // Now issue a saccade to a far-away point WITH reset.
        let (sx_reset, sy_reset) = smoother.smooth(800.0, 100.0, true);

        // Build the same filter WITHOUT saccade reset.
        let mut laggy = GazeSmoother::new();
        for _ in 0..10 {
            laggy.smooth(200.0, 300.0, false);
        }
        let (sx_no_reset, sy_no_reset) = laggy.smooth(800.0, 100.0, false);

        // The non-reset output should still be very close to the old
        // convergent point (the filter smooths aggressively).
        let no_reset_dist_from_measurement =
            ((sx_no_reset - 800.0).powi(2) + (sy_no_reset - 100.0).powi(2)).sqrt();
        let reset_dist_from_measurement =
            ((sx_reset - 800.0).powi(2) + (sy_reset - 100.0).powi(2)).sqrt();

        // The reset output should be significantly closer to the raw
        // measurement than the non-reset output (a ~2x difference is
        // expected given default noise parameters).
        assert!(
            no_reset_dist_from_measurement > reset_dist_from_measurement * 1.5,
            "saccade reset should bring output much closer to measurement \
             (no_reset dist={:.2}, reset dist={:.2})",
            no_reset_dist_from_measurement,
            reset_dist_from_measurement,
        );

        // Also verify that reset output actually moved a substantial
        // portion of the distance toward the target (gain ~0.89, so at
        // least 50% of the way there).
        assert!(
            sx_reset > 400.0,
            "saccade-reset x should jump past midpoint, got {}",
            sx_reset,
        );
        assert!(
            sy_reset < 200.0,
            "saccade-reset y should jump past midpoint, got {}",
            sy_reset,
        );
    }
}
