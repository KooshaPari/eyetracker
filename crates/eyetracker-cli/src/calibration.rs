//! Eye tracker calibration module
//!
//! Implements a 9-point calibration routine where the user looks at
//! target points on screen and gaze samples are collected to build
//! a calibration mapping.

use anyhow::Result;
use eyetracker_inference::{PipelineConfig, TrackingPipeline};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// Calibration point on screen
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationPoint {
    /// Normalized x (0.0 - 1.0)
    pub x: f32,
    /// Normalized y (0.0 - 1.0)
    pub y: f32,
    /// Label for display
    pub label: Cow<'static, str>,
}

/// 9-point calibration grid positions (3x3)
const CALIBRATION_POINTS: &[CalibrationPoint] = &[
    CalibrationPoint {
        x: 0.1,
        y: 0.1,
        label: Cow::Borrowed("Top-left"),
    },
    CalibrationPoint {
        x: 0.5,
        y: 0.1,
        label: Cow::Borrowed("Top-center"),
    },
    CalibrationPoint {
        x: 0.9,
        y: 0.1,
        label: Cow::Borrowed("Top-right"),
    },
    CalibrationPoint {
        x: 0.1,
        y: 0.5,
        label: Cow::Borrowed("Mid-left"),
    },
    CalibrationPoint {
        x: 0.5,
        y: 0.5,
        label: Cow::Borrowed("Center"),
    },
    CalibrationPoint {
        x: 0.9,
        y: 0.5,
        label: Cow::Borrowed("Mid-right"),
    },
    CalibrationPoint {
        x: 0.1,
        y: 0.9,
        label: Cow::Borrowed("Bottom-left"),
    },
    CalibrationPoint {
        x: 0.5,
        y: 0.9,
        label: Cow::Borrowed("Bottom-center"),
    },
    CalibrationPoint {
        x: 0.9,
        y: 0.9,
        label: Cow::Borrowed("Bottom-right"),
    },
];

/// Calibration sample collected at a target point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationSample {
    /// Which calibration point
    pub point: CalibrationPoint,
    /// Collected gaze vectors during the sample period
    pub gaze_samples: Vec<(f32, f32, f32)>, // (x, y, z) gaze vectors
    /// Timestamp of collection (used for drift monitoring per FR-EYE-CAL-004)
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

/// Run the calibration routine
pub fn run_calibration(config: &PipelineConfig) -> Result<CalibrationResult> {
    let mut pipeline = TrackingPipeline::with_config(config.clone())?;
    pipeline.start()?;

    println!("=== Eye Tracker Calibration ===");
    println!("Look at each target point on screen for 3 seconds.");
    println!("Press Enter to start...");

    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;

    let mut samples = Vec::new();

    // FR-EYE-CAL-001: each target point must be retried (up to
    // MAX_RETRIES_PER_POINT times) if fewer than MIN_SAMPLES_FOR_EVAL
    // are collected during the dwell window, or if the user did not
    // hold fixation for the required duration. This enforces the
    // spec clause "system shall dismiss and request retry if
    // insufficient samples collected".
    use eyetracker_inference::calibration::{
        classify_point, CalibrationPoint as InferenceCalibrationPoint,
        CalibrationSample as InferenceCalibrationSample, PointOutcome, MAX_RETRIES_PER_POINT,
    };
    // Approx 33ms per frame at the 30 FPS calibration polling rate.
    let frame_duration_ms: u64 = 33;

    for (i, point) in CALIBRATION_POINTS.iter().enumerate() {
        println!(
            "\n[{}/{}] Look at {} ({:.0}%, {:.0}%)",
            i + 1,
            CALIBRATION_POINTS.len(),
            point.label,
            point.x * 100.0,
            point.y * 100.0,
        );

        let mut attempt = 0;
        let sample = loop {
            attempt += 1;
            println!(
                "  Attempt {}/{} — press Enter when ready...",
                attempt, MAX_RETRIES_PER_POINT
            );

            input.clear();
            std::io::stdin().read_line(&mut input)?;

            let raw = collect_samples(&mut pipeline, point, Duration::from_secs(3))?;
            let inference_sample = InferenceCalibrationSample {
                point: InferenceCalibrationPoint {
                    x: raw.point.x,
                    y: raw.point.y,
                    label: raw.point.label.to_string(),
                },
                gaze_samples: raw.gaze_samples.clone(),
                timestamp: raw.timestamp,
            };
            let outcome = classify_point(&inference_sample, frame_duration_ms);
            let count = raw.gaze_samples.len();
            println!("  Collected {} samples → {:?}", count, outcome);

            match &outcome {
                PointOutcome::Stable { .. } => break raw,
                _ if attempt >= MAX_RETRIES_PER_POINT => {
                    println!(
                        "  Max retries ({}); accepting this point anyway.",
                        MAX_RETRIES_PER_POINT
                    );
                    break raw;
                }
                PointOutcome::InsufficientSamples { .. } => {
                    println!("  Insufficient samples — please look at the target and try again.");
                }
                PointOutcome::NoFixation { max_drift } => {
                    println!(
                        "  Gaze drifted ({:.1}% off) — keep your eyes on the target and try again.",
                        max_drift * 100.0
                    );
                }
            }
        };

        samples.push(sample);
    }

    pipeline.stop()?;

    // Compute calibration quality
    let quality = compute_calibration_quality(&samples);
    let success = quality > 0.3;

    println!("\n=== Calibration Complete ===");
    println!("Quality: {:.1}%", quality * 100.0);
    println!(
        "Success: {}",
        if success { "Yes" } else { "No - try again" }
    );

    let result = CalibrationResult {
        samples,
        quality,
        success,
    };

    if success {
        if let Err(e) = save_calibration(&result) {
            tracing::warn!("Failed to save calibration: {}", e);
        }
    }

    Ok(result)
}

/// Collect gaze samples for a specific target point
fn collect_samples(
    pipeline: &mut TrackingPipeline,
    point: &CalibrationPoint,
    duration: Duration,
) -> Result<CalibrationSample> {
    let start = Instant::now();
    let mut gaze_samples = Vec::new();

    while start.elapsed() < duration {
        match pipeline.process_frame() {
            Ok(result) => {
                if let Some(gaze) = result.gaze {
                    gaze_samples.push((gaze.combined.x, gaze.combined.y, gaze.combined.z));
                }
            }
            Err(e) => {
                tracing::warn!("Frame error during calibration: {}", e);
            }
        }
        // ~30fps polling
        std::thread::sleep(Duration::from_millis(33));
    }

    Ok(CalibrationSample {
        point: point.clone(),
        gaze_samples,
        timestamp: std::time::Instant::now(),
    })
}

/// Compute calibration quality score from collected samples
fn compute_calibration_quality(samples: &[CalibrationSample]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }

    let mut total_score = 0.0;

    for sample in samples {
        let count = sample.gaze_samples.len();
        if count < 5 {
            continue;
        }

        // Compute mean gaze vector for this sample
        let mean_x: f32 = sample.gaze_samples.iter().map(|s| s.0).sum::<f32>() / count as f32;
        let mean_y: f32 = sample.gaze_samples.iter().map(|s| s.1).sum::<f32>() / count as f32;
        let mean_z: f32 = sample.gaze_samples.iter().map(|s| s.2).sum::<f32>() / count as f32;

        // Compute variance (lower = more stable = better calibration)
        let variance: f32 = sample
            .gaze_samples
            .iter()
            .map(|s| (s.0 - mean_x).powi(2) + (s.1 - mean_y).powi(2) + (s.2 - mean_z).powi(2))
            .sum::<f32>()
            / count as f32;

        // Score: lower variance = higher quality
        // Typical variances are 0.01-0.1, so score = 1.0 / (1.0 + variance * 10)
        let stability_score = 1.0 / (1.0 + variance * 10.0);
        let coverage_score = (count as f32 / 30.0).min(1.0); // Expect ~30 samples per point

        total_score += stability_score * coverage_score;
    }

    total_score / samples.len() as f32
}

/// Get the path to the calibration file
fn calibration_path() -> Result<PathBuf> {
    let data_dir = dirs::data_local_dir()
        .ok_or_else(|| anyhow::anyhow!("Could not determine platform data directory"))?;
    let app_dir = data_dir.join("eyetracker");
    Ok(app_dir.join("cal.bin"))
}

/// Save calibration result to disk using bincode serialization
pub fn save_calibration(result: &CalibrationResult) -> Result<()> {
    let path = calibration_path()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let encoded = bincode::serialize(result)
        .map_err(|e| anyhow::anyhow!("Failed to serialize calibration: {}", e))?;
    std::fs::write(&path, encoded)?;
    println!("Calibration saved to: {}", path.display());
    Ok(())
}

/// Load calibration result from disk
pub fn load_calibration() -> Result<Option<CalibrationResult>> {
    let path = calibration_path()?;
    if !path.exists() {
        println!("No saved calibration found at: {}", path.display());
        return Ok(None);
    }
    let encoded = std::fs::read(&path)?;
    let result: CalibrationResult = bincode::deserialize(&encoded)
        .map_err(|e| anyhow::anyhow!("Failed to deserialize calibration: {}", e))?;
    println!("=== Calibration Loaded ===");
    println!("  From:    {}", path.display());
    println!("  Quality: {:.1}%", result.quality * 100.0);
    println!("  Points:  {}", result.samples.len());
    println!("  Success: {}", if result.success { "Yes" } else { "No" });
    Ok(Some(result))
}
