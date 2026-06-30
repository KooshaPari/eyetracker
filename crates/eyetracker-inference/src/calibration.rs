//! Calibration result types and quality scoring (FR-EYE-CAL-002)
//!
//! Defines the canonical calibration result types used across the workspace.
//! Includes 9-point grid constants and quality scoring used to decide whether
//! a calibration should be accepted.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Calibration point on screen
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationPoint {
    /// Normalized x (0.0 - 1.0)
    pub x: f32,
    /// Normalized y (0.0 - 1.0)
    pub y: f32,
    /// Label for display
    pub label: String,
}

/// Calibration sample collected at a target point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationSample {
    /// Which calibration point
    pub point: CalibrationPoint,
    /// Collected gaze vectors during the sample period
    pub gaze_samples: Vec<(f32, f32, f32)>,
    /// Timestamp (used for drift monitoring per FR-EYE-CAL-004)
    #[serde(skip, default = "Instant::now")]
    #[allow(dead_code)]
    pub timestamp: Instant,
}

/// Calibration result mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationResult {
    /// Per-point samples
    pub samples: Vec<CalibrationSample>,
    /// Mapping quality score (0.0 - 1.0)
    pub quality: f32,
    /// Whether calibration succeeded
    pub success: bool,
}

impl CalibrationResult {
    /// Minimum quality threshold for a successful calibration
    pub const MIN_QUALITY: f32 = 0.3;

    /// Number of points in the standard 9-point grid
    pub const GRID_POINTS: usize = 9;

    /// Required fixation duration per point (FR-EYE-CAL-001)
    pub const REQUIRED_FIXATION_MS: u64 = 500;

    /// Compute the centroid of gaze samples at a target point
    pub fn point_centroid(sample: &CalibrationSample) -> Option<(f32, f32, f32)> {
        if sample.gaze_samples.is_empty() {
            return None;
        }
        let n = sample.gaze_samples.len() as f32;
        let x = sample.gaze_samples.iter().map(|s| s.0).sum::<f32>() / n;
        let y = sample.gaze_samples.iter().map(|s| s.1).sum::<f32>() / n;
        let z = sample.gaze_samples.iter().map(|s| s.2).sum::<f32>() / n;
        Some((x, y, z))
    }

    /// Compute calibration quality from collected samples (0.0 - 1.0)
    ///
    /// Combines stability (low variance per point) and coverage (>= 5 samples
    /// per point) into a single score.
    pub fn compute_quality(samples: &[CalibrationSample]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let mut total = 0.0;
        for sample in samples {
            let count = sample.gaze_samples.len();
            if count < 5 {
                continue;
            }
            let n = count as f32;
            let mx = sample.gaze_samples.iter().map(|s| s.0).sum::<f32>() / n;
            let my = sample.gaze_samples.iter().map(|s| s.1).sum::<f32>() / n;
            let mz = sample.gaze_samples.iter().map(|s| s.2).sum::<f32>() / n;
            let variance: f32 = sample
                .gaze_samples
                .iter()
                .map(|s| (s.0 - mx).powi(2) + (s.1 - my).powi(2) + (s.2 - mz).powi(2))
                .sum::<f32>()
                / n;
            let stability = 1.0 / (1.0 + variance * 10.0);
            let coverage = (count as f32 / 30.0).min(1.0);
            total += stability * coverage;
        }
        total / samples.len() as f32
    }

    /// Build a result from samples, computing quality + success
    pub fn from_samples(samples: Vec<CalibrationSample>) -> Self {
        let quality = Self::compute_quality(&samples);
        let success = quality >= Self::MIN_QUALITY && samples.len() == Self::GRID_POINTS;
        Self {
            samples,
            quality,
            success,
        }
    }

    /// Compute residual error (FR-EYE-CAL-002: drift tolerance check).
    /// Returns mean angular drift in degrees, or None if no samples.
    pub fn residual_drift_degrees(&self) -> Option<f32> {
        if self.samples.is_empty() {
            return None;
        }
        let mut total_err = 0.0;
        let mut count = 0;
        for sample in &self.samples {
            if let Some((cx, cy, _)) = Self::point_centroid(sample) {
                let dx = cx - sample.point.x;
                let dy = cy - sample.point.y;
                let dist = (dx * dx + dy * dy).sqrt();
                // Approximate 1.0 normalized = 30° of gaze angle
                total_err += dist * 30.0;
                count += 1;
            }
        }
        if count == 0 {
            None
        } else {
            Some(total_err / count as f32)
        }
    }

    /// Check if calibration passes FR-EYE-CAL-002 (≤1.5° drift tolerance)
    pub fn is_within_tolerance(&self) -> bool {
        self.residual_drift_degrees()
            .map(|d| d <= 1.5)
            .unwrap_or(false)
    }
}

/// Standard 9-point calibration grid positions (3x3)
pub fn default_grid_points() -> Vec<CalibrationPoint> {
    vec![
        CalibrationPoint {
            x: 0.1,
            y: 0.1,
            label: "Top-left".into(),
        },
        CalibrationPoint {
            x: 0.5,
            y: 0.1,
            label: "Top-center".into(),
        },
        CalibrationPoint {
            x: 0.9,
            y: 0.1,
            label: "Top-right".into(),
        },
        CalibrationPoint {
            x: 0.1,
            y: 0.5,
            label: "Mid-left".into(),
        },
        CalibrationPoint {
            x: 0.5,
            y: 0.5,
            label: "Center".into(),
        },
        CalibrationPoint {
            x: 0.9,
            y: 0.5,
            label: "Mid-right".into(),
        },
        CalibrationPoint {
            x: 0.1,
            y: 0.9,
            label: "Bottom-left".into(),
        },
        CalibrationPoint {
            x: 0.5,
            y: 0.9,
            label: "Bottom-center".into(),
        },
        CalibrationPoint {
            x: 0.9,
            y: 0.9,
            label: "Bottom-right".into(),
        },
    ]
}

/// Path to the calibration file on disk
pub fn calibration_path() -> Result<std::path::PathBuf> {
    let data_dir = dirs::data_local_dir()
        .ok_or_else(|| anyhow::anyhow!("Could not determine platform data directory"))?;
    Ok(data_dir.join("eyetracker").join("cal.bin"))
}

/// Save calibration result to disk using bincode serialization
pub fn save_calibration(result: &CalibrationResult) -> Result<()> {
    let path = calibration_path()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let encoded = bincode::serialize(result)
        .map_err(|e| anyhow::anyhow!("Failed to serialize calibration: {e}"))?;
    std::fs::write(&path, encoded)?;
    Ok(())
}

/// Load calibration result from disk
pub fn load_calibration() -> Result<Option<CalibrationResult>> {
    let path = calibration_path()?;
    if !path.exists() {
        return Ok(None);
    }
    let encoded = std::fs::read(&path)?;
    let result: CalibrationResult = bincode::deserialize(&encoded)
        .map_err(|e| anyhow::anyhow!("Failed to deserialize calibration: {e}"))?;
    Ok(Some(result))
}

/// FR-EYE-CAL-001: tolerance (normalized screen coords) within which a sample
/// counts as "stable" — i.e. the user's gaze is on-target. 0.05 ≈ 5% of
/// screen, which at typical head distance is roughly the foveal radius (~1.5°).
pub const FIXATION_TOLERANCE: f32 = 0.05;

/// FR-EYE-CAL-001: outcome of evaluating a single calibration point.
#[derive(Debug, Clone, PartialEq)]
pub enum PointOutcome {
    /// User held fixation for at least the required duration.
    Stable {
        /// Total duration the user held fixation (ms).
        fixation_ms: u64,
        /// Number of stable samples collected.
        stable_count: usize,
    },
    /// User did not hold fixation (gaze drifted away from target).
    NoFixation {
        /// Maximum gaze distance from target observed (normalized units).
        max_drift: f32,
    },
    /// Not enough samples were collected during the calibration window.
    InsufficientSamples {
        /// Total samples actually collected.
        collected: usize,
    },
}

/// FR-EYE-CAL-001: minimum sample count to even attempt classification.
/// Below this we always report `InsufficientSamples` regardless of stability.
pub const MIN_SAMPLES_FOR_EVAL: usize = 5;

/// FR-EYE-CAL-001: classify a calibration point's samples.
///
/// Counts how many samples fall within `FIXATION_TOLERANCE` of the target
/// and computes the implied fixation duration by multiplying by the
/// `frame_duration_ms`. The point passes (`Stable`) if the resulting
/// fixation duration is at least `CalibrationResult::REQUIRED_FIXATION_MS`.
pub fn classify_point(sample: &CalibrationSample, frame_duration_ms: u64) -> PointOutcome {
    let n = sample.gaze_samples.len();
    if n < MIN_SAMPLES_FOR_EVAL {
        return PointOutcome::InsufficientSamples { collected: n };
    }
    let mut stable_count = 0usize;
    let mut max_drift: f32 = 0.0;
    for g in &sample.gaze_samples {
        let dx = g.0 - sample.point.x;
        let dy = g.1 - sample.point.y;
        let dist = (dx * dx + dy * dy).sqrt();
        if dist <= FIXATION_TOLERANCE {
            stable_count += 1;
        } else if dist > max_drift {
            max_drift = dist;
        }
    }
    let fixation_ms = stable_count as u64 * frame_duration_ms;
    if fixation_ms >= CalibrationResult::REQUIRED_FIXATION_MS {
        PointOutcome::Stable {
            fixation_ms,
            stable_count,
        }
    } else {
        PointOutcome::NoFixation { max_drift }
    }
}

/// FR-EYE-CAL-001: retry wrapper. Calls `collect` until a `Stable` outcome
/// is achieved or `max_retries` is exhausted. Returns the final outcome and
/// the number of attempts made.
pub fn classify_point_with_retry<F>(
    mut collect: F,
    frame_duration_ms: u64,
    max_retries: u32,
) -> (PointOutcome, u32)
where
    F: FnMut() -> CalibrationSample,
{
    let mut attempts = 0u32;
    loop {
        attempts += 1;
        let sample = collect();
        let outcome = classify_point(&sample, frame_duration_ms);
        match &outcome {
            PointOutcome::Stable { .. } => return (outcome, attempts),
            _ if attempts >= max_retries => return (outcome, attempts),
            _ => continue,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sample(x: f32, y: f32, gaze: Vec<(f32, f32, f32)>) -> CalibrationSample {
        CalibrationSample {
            point: CalibrationPoint {
                x,
                y,
                label: "p".into(),
            },
            gaze_samples: gaze,
            timestamp: Instant::now(),
        }
    }

    #[test]
    fn test_default_grid_has_nine_points() {
        let grid = default_grid_points();
        assert_eq!(grid.len(), 9);
    }

    #[test]
    fn test_compute_quality_empty() {
        assert_eq!(CalibrationResult::compute_quality(&[]), 0.0);
    }

    #[test]
    fn test_compute_quality_stable_high() {
        let samples = vec![
            make_sample(0.5, 0.5, vec![(0.5, 0.5, 0.0); 30]),
            make_sample(0.1, 0.1, vec![(0.1, 0.1, 0.0); 30]),
        ];
        let q = CalibrationResult::compute_quality(&samples);
        assert!(q > 0.9, "expected high quality, got {q}");
    }

    #[test]
    fn test_compute_quality_noisy_low() {
        // Cover a wide area with stride > 0.1 — high variance
        let noisy: Vec<(f32, f32, f32)> = (0..30)
            .map(|i| {
                let t = i as f32;
                (
                    0.05 + (t * 0.31).sin().abs() * 0.9,
                    0.05 + (t * 0.47).cos().abs() * 0.9,
                    0.0,
                )
            })
            .collect();
        let samples = vec![make_sample(0.5, 0.5, noisy)];
        let q = CalibrationResult::compute_quality(&samples);
        assert!(q < 0.7, "expected low quality for noisy data, got {q}");
    }

    #[test]
    fn test_from_samples_9_points_passes() {
        let mut samples = Vec::new();
        for (i, pt) in default_grid_points().into_iter().enumerate() {
            let gaze = vec![(pt.x, pt.y, 0.0); 30];
            let mut s = make_sample(pt.x, pt.y, gaze);
            s.point = pt;
            samples.push(s);
            // touch i to silence warning
            let _ = i;
        }
        let r = CalibrationResult::from_samples(samples);
        assert!(r.success);
        assert!(r.quality > 0.9);
    }

    #[test]
    fn test_from_samples_5_points_fails() {
        let samples: Vec<_> = default_grid_points()
            .into_iter()
            .take(5)
            .map(|pt| {
                let gaze = vec![(pt.x, pt.y, 0.0); 30];
                make_sample(pt.x, pt.y, gaze)
            })
            .collect();
        let r = CalibrationResult::from_samples(samples);
        assert!(!r.success);
    }

    #[test]
    fn test_residual_drift_perfect() {
        let samples: Vec<_> = default_grid_points()
            .into_iter()
            .map(|pt| {
                let gaze = vec![(pt.x, pt.y, 0.0); 30];
                make_sample(pt.x, pt.y, gaze)
            })
            .collect();
        let r = CalibrationResult::from_samples(samples);
        let drift = r.residual_drift_degrees().unwrap();
        assert!(drift < 0.1, "perfect calibration drift = {drift}");
    }

    #[test]
    fn test_residual_drift_off_target() {
        // Gaze is offset by (0.05, 0.05) normalized → ~2.1° drift
        let samples: Vec<_> = default_grid_points()
            .into_iter()
            .map(|pt| {
                let gaze = vec![(pt.x + 0.05, pt.y + 0.05, 0.0); 30];
                make_sample(pt.x, pt.y, gaze)
            })
            .collect();
        let r = CalibrationResult::from_samples(samples);
        let drift = r.residual_drift_degrees().unwrap();
        assert!(
            drift > 1.5 && drift < 3.0,
            "expected ~2.1° drift, got {drift}"
        );
        assert!(!r.is_within_tolerance(), "should fail tolerance check");
    }

    #[test]
    fn test_persistence_round_trip() {
        let tmp =
            std::env::temp_dir().join(format!("eyetracker-cal-test-{}.bin", std::process::id()));
        let _ = std::fs::remove_file(&tmp);

        let samples: Vec<_> = default_grid_points()
            .into_iter()
            .map(|pt| {
                let gaze = vec![(pt.x, pt.y, 0.0); 30];
                make_sample(pt.x, pt.y, gaze)
            })
            .collect();
        let original = CalibrationResult::from_samples(samples);
        save_calibration_at(&original, &tmp).unwrap();

        let decoded = load_calibration_at(&tmp).unwrap().unwrap();
        assert_eq!(decoded.samples.len(), 9);
        assert!((decoded.quality - original.quality).abs() < 0.001);
        assert_eq!(decoded.success, original.success);

        let _ = std::fs::remove_file(&tmp);
    }

    fn save_calibration_at(r: &CalibrationResult, p: &std::path::Path) -> Result<()> {
        let encoded = bincode::serialize(r).map_err(|e| anyhow::anyhow!("Failed: {e}"))?;
        std::fs::write(p, encoded)?;
        Ok(())
    }
    fn load_calibration_at(p: &std::path::Path) -> Result<Option<CalibrationResult>> {
        let encoded = std::fs::read(p)?;
        let r: CalibrationResult =
            bincode::deserialize(&encoded).map_err(|e| anyhow::anyhow!("Failed: {e}"))?;
        Ok(Some(r))
    }

    // ---------- FR-EYE-CAL-001: classify_point tests ----------

    /// Helper: build a sample where every gaze point is exactly at target.
    fn make_perfect_sample(target_x: f32, target_y: f32, n: usize) -> CalibrationSample {
        let gaze = vec![(target_x, target_y, 0.0); n];
        CalibrationSample {
            point: CalibrationPoint {
                x: target_x,
                y: target_y,
                label: "p".into(),
            },
            gaze_samples: gaze,
            timestamp: Instant::now(),
        }
    }

    #[test]
    fn fr_eye_cal_001_classify_stable() {
        // 30 samples perfectly at target. 30ms frame = 900ms fixation > 500ms required.
        let s = make_perfect_sample(0.5, 0.5, 30);
        let outcome = classify_point(&s, 30);
        match outcome {
            PointOutcome::Stable {
                fixation_ms,
                stable_count,
            } => {
                assert_eq!(stable_count, 30);
                assert_eq!(fixation_ms, 900);
            }
            other => panic!("expected Stable, got {other:?}"),
        }
    }

    #[test]
    fn fr_eye_cal_001_classify_no_fixation() {
        // 30 wildly-drifting samples: gaze covers a large area, none stable.
        let pt = CalibrationPoint {
            x: 0.5,
            y: 0.5,
            label: "p".into(),
        };
        let gaze: Vec<(f32, f32, f32)> = (0..30)
            .map(|i| {
                let t = i as f32 * 0.31;
                (0.5 + t.sin() * 0.4, 0.5 + t.cos() * 0.4, 0.0)
            })
            .collect();
        let s = CalibrationSample {
            point: pt,
            gaze_samples: gaze,
            timestamp: Instant::now(),
        };
        let outcome = classify_point(&s, 30);
        assert!(
            matches!(outcome, PointOutcome::NoFixation { .. }),
            "expected NoFixation, got {outcome:?}"
        );
    }

    #[test]
    fn fr_eye_cal_001_classify_insufficient_samples() {
        // Only 3 samples — below MIN_SAMPLES_FOR_EVAL (5).
        let s = make_perfect_sample(0.5, 0.5, 3);
        let outcome = classify_point(&s, 30);
        assert!(
            matches!(outcome, PointOutcome::InsufficientSamples { collected: 3 }),
            "got {outcome:?}"
        );
    }

    #[test]
    fn fr_eye_cal_001_partial_fixation_below_threshold() {
        // 10 stable samples + 20 drifting samples. 10 * 30ms = 300ms < 500ms.
        let pt = CalibrationPoint {
            x: 0.5,
            y: 0.5,
            label: "p".into(),
        };
        let mut gaze = vec![(0.5, 0.5, 0.0); 10];
        for i in 0..20 {
            let t = i as f32 * 0.31;
            gaze.push((0.5 + t.sin() * 0.4, 0.5 + t.cos() * 0.4, 0.0));
        }
        let s = CalibrationSample {
            point: pt,
            gaze_samples: gaze,
            timestamp: Instant::now(),
        };
        let outcome = classify_point(&s, 30);
        assert!(
            matches!(outcome, PointOutcome::NoFixation { .. }),
            "expected NoFixation (300ms < 500ms threshold), got {outcome:?}"
        );
    }

    #[test]
    fn fr_eye_cal_001_retry_eventually_succeeds() {
        // First call: NoFixation (drift). Second call: Stable. Retry should succeed.
        let attempt = std::cell::Cell::new(0u32);
        let stable = make_perfect_sample(0.5, 0.5, 30);
        let unstable_pt = CalibrationPoint {
            x: 0.5,
            y: 0.5,
            label: "p".into(),
        };
        let unstable_gaze: Vec<(f32, f32, f32)> = (0..30)
            .map(|i| {
                let t = i as f32 * 0.31;
                (0.5 + t.sin() * 0.4, 0.5 + t.cos() * 0.4, 0.0)
            })
            .collect();

        let (outcome, attempts) = classify_point_with_retry(
            || {
                let n = attempt.get();
                attempt.set(n + 1);
                if n == 0 {
                    CalibrationSample {
                        point: unstable_pt.clone(),
                        gaze_samples: unstable_gaze.clone(),
                        timestamp: Instant::now(),
                    }
                } else {
                    stable.clone()
                }
            },
            30,
            3,
        );
        assert_eq!(attempts, 2, "should succeed on the 2nd attempt");
        assert!(matches!(outcome, PointOutcome::Stable { .. }));
    }

    #[test]
    fn fr_eye_cal_001_retry_exhausts() {
        // Always returns NoFixation; retry should give up after max_retries.
        let attempt = std::cell::Cell::new(0u32);
        let pt = CalibrationPoint {
            x: 0.5,
            y: 0.5,
            label: "p".into(),
        };
        let unstable_gaze: Vec<(f32, f32, f32)> = (0..30)
            .map(|i| {
                let t = i as f32 * 0.31;
                (0.5 + t.sin() * 0.4, 0.5 + t.cos() * 0.4, 0.0)
            })
            .collect();

        let (outcome, attempts) = classify_point_with_retry(
            || {
                let n = attempt.get();
                attempt.set(n + 1);
                CalibrationSample {
                    point: pt.clone(),
                    gaze_samples: unstable_gaze.clone(),
                    timestamp: Instant::now(),
                }
            },
            30,
            3,
        );
        assert_eq!(attempts, 3, "should exhaust max_retries");
        assert!(matches!(outcome, PointOutcome::NoFixation { .. }));
    }
}
