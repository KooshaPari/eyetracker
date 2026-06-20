//! Eye tracker calibration module
//!
//! Implements a 9-point calibration routine where the user looks at
//! target points on screen and gaze samples are collected to build
//! a calibration mapping.

use anyhow::Result;
use eyetracker_inference::{PipelineConfig, TrackingPipeline};
use std::time::{Duration, Instant};

/// Calibration point on screen
#[derive(Debug, Clone, Copy)]
pub struct CalibrationPoint {
    /// Normalized x (0.0 - 1.0)
    pub x: f32,
    /// Normalized y (0.0 - 1.0)
    pub y: f32,
    /// Label for display
    pub label: &'static str,
}

/// 9-point calibration grid positions (3x3)
const CALIBRATION_POINTS: &[CalibrationPoint] = &[
    CalibrationPoint { x: 0.1, y: 0.1, label: "Top-left" },
    CalibrationPoint { x: 0.5, y: 0.1, label: "Top-center" },
    CalibrationPoint { x: 0.9, y: 0.1, label: "Top-right" },
    CalibrationPoint { x: 0.1, y: 0.5, label: "Mid-left" },
    CalibrationPoint { x: 0.5, y: 0.5, label: "Center" },
    CalibrationPoint { x: 0.9, y: 0.5, label: "Mid-right" },
    CalibrationPoint { x: 0.1, y: 0.9, label: "Bottom-left" },
    CalibrationPoint { x: 0.5, y: 0.9, label: "Bottom-center" },
    CalibrationPoint { x: 0.9, y: 0.9, label: "Bottom-right" },
];

/// Calibration sample collected at a target point
#[derive(Debug, Clone)]
pub struct CalibrationSample {
    /// Which calibration point
    pub point: CalibrationPoint,
    /// Collected gaze vectors during the sample period
    pub gaze_samples: Vec<(f32, f32, f32)>, // (x, y, z) gaze vectors
    /// Timestamp of collection
    pub timestamp: std::time::Instant,
}

/// Calibration result mapping
#[derive(Debug, Clone)]
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

    for (i, point) in CALIBRATION_POINTS.iter().enumerate() {
        println!(
            "\n[{}/{}] Look at {} ({:.0}%, {:.0}%)",
            i + 1,
            CALIBRATION_POINTS.len(),
            point.label,
            point.x * 100.0,
            point.y * 100.0,
        );
        println!("Press Enter when ready...");

        input.clear();
        std::io::stdin().read_line(&mut input)?;

        // Collect samples for 3 seconds
        let sample = collect_samples(&mut pipeline, point, Duration::from_secs(3))?;
        let count = sample.gaze_samples.len();
        println!("  Collected {} samples (press Enter to continue)", count);

        input.clear();
        std::io::stdin().read_line(&mut input)?;

        samples.push(sample);
    }

    pipeline.stop()?;

    // Compute calibration quality
    let quality = compute_calibration_quality(&samples);
    let success = quality > 0.3;

    println!("\n=== Calibration Complete ===");
    println!("Quality: {:.1}%", quality * 100.0);
    println!("Success: {}", if success { "Yes" } else { "No - try again" });

    Ok(CalibrationResult {
        samples,
        quality,
        success,
    })
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
        point: *point,
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
            .map(|s| {
                (s.0 - mean_x).powi(2) + (s.1 - mean_y).powi(2) + (s.2 - mean_z).powi(2)
            })
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
