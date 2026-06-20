//! Main application logic for the eye tracker CLI

use anyhow::Result;
use eyetracker_inference::{PipelineConfig, TrackingPipeline, TrackingResult};
use ratatui::Terminal;
use std::io::Stdout;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use crate::ui;

/// Run the interactive TUI mode
pub fn run_tui(
    terminal: &mut Terminal<ratatui::backend::CrosstermBackend<Stdout>>,
    config: &PipelineConfig,
    duration_secs: u64,
) -> Result<()> {
    // Build pipeline
    let mut pipeline = TrackingPipeline::with_config(config.clone())?;
    pipeline.start()?;

    let (tx, rx) = mpsc::channel::<TrackingResult>();

    // Spawn processing thread
    let processing_config = config.clone();
    let processing_duration = duration_secs;
    let processing_handle = std::thread::spawn(move || {
        let mut local_pipeline = match TrackingPipeline::with_config(processing_config) {
            Ok(p) => p,
            Err(e) => {
                tracing::error!("Failed to create pipeline in worker: {}", e);
                return;
            }
        };
        if let Err(e) = local_pipeline.start() {
            tracing::error!("Failed to start pipeline in worker: {}", e);
            return;
        }
        let start = Instant::now();
        loop {
            if processing_duration > 0 && start.elapsed().as_secs() >= processing_duration {
                break;
            }
            match local_pipeline.process_frame() {
                Ok(result) => {
                    if tx.send(result).is_err() {
                        break;
                    }
                }
                Err(e) => {
                    tracing::warn!("Frame processing error: {}", e);
                }
            }
        }
        let _ = local_pipeline.stop();
    });

    // TUI event loop
    let result = ui::run_event_loop(
        terminal,
        &rx,
        |result: &TrackingResult| {
            let fps = if result.processing_time_ms > 0.0 {
                1000.0 / result.processing_time_ms
            } else {
                0.0
            };
            let gaze = result.gaze.as_ref().map(|g| {
                format!(
                    "({:.1}, {:.1}, {:.1})",
                    g.combined.x, g.combined.y, g.combined.z
                )
            });
            let confidence = result
                .gaze
                .as_ref()
                .map(|g| format!("{:.1}%", g.confidence * 100.0))
                .unwrap_or_else(|| "N/A".to_string());

            ui::DashboardData {
                fps,
                processing_ms: result.processing_time_ms,
                frame_number: result.frame.frame_number,
                gaze_vector: gaze,
                confidence,
                face_detected: result.face.is_some(),
                resolution: format!("{}x{}", result.frame.width, result.frame.height),
            }
        },
        duration_secs,
    );

    let _ = pipeline.stop();
    let _ = processing_handle.join();

    result
}

/// Run CSV dump mode (no TUI, just output CSV data)
pub fn run_csv_dump(config: &PipelineConfig, duration_secs: u64) -> Result<()> {
    let mut pipeline = TrackingPipeline::with_config(config.clone())?;
    pipeline.start()?;

    // CSV header
    println!("timestamp_ms,frame,processing_ms,gaze_x,gaze_y,gaze_z,confidence,face_detected");

    let start = Instant::now();
    loop {
        if duration_secs > 0 && start.elapsed().as_secs() >= duration_secs {
            break;
        }
        match pipeline.process_frame() {
            Ok(result) => {
                let timestamp = start.elapsed().as_secs_f64() * 1000.0;
                let (gx, gy, gz, conf) = result
                    .gaze
                    .as_ref()
                    .map(|g| (g.combined.x, g.combined.y, g.combined.z, g.confidence))
                    .unwrap_or((0.0, 0.0, 0.0, 0.0));
                println!(
                    "{:.1},{},{:.2},{:.4},{:.4},{:.4},{:.4},{}",
                    timestamp,
                    result.frame.frame_number,
                    result.processing_time_ms,
                    gx,
                    gy,
                    gz,
                    conf,
                    result.face.is_some(),
                );
            }
            Err(e) => {
                tracing::warn!("Frame error: {}", e);
            }
        }
    }

    pipeline.stop()?;
    Ok(())
}
