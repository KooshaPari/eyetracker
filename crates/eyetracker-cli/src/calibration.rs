//! Calibration module
//!
//! Handles 9-point calibration and accuracy validation.
use std::io;
use std::io::Write;

use anyhow::Result;
use crossterm::{
    event::{self, DisableBracketedPaste, DisableFocusChange, EnableBracketedPaste, EnableFocusChange, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Style},
    text::{Line, Text},
    widgets::{Block, Borders, Paragraph},
    Frame, Terminal,
};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

use eyetracker_camera::{Camera, CameraConfig};
use eyetracker_core::calibration::{Calibrator, CalibrationState};
use eyetracker_inference::{
    pipeline::InferencePipeline,
    processing::{preprocess_frame, PreprocessOptions},
};
use std::sync::{Arc, Mutex};
use eyetracker_domain::Point;
use eyetracker_inference::{
    preprocess_frame, GazeEstimationResult, InferencePipeline, PreprocessOptions,
};
use ratatui::{backend::CrosstermBackend, layout::*, style::*, text::*, widgets::*, Terminal};
use std::io::{self, Write};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Calibration point
#[derive(Debug, Clone)]
struct CalibrationPoint {
    /// Screen position (normalized 0-1)
    position: (f32, f32),
    /// Whether point has been calibrated
    completed: bool,
    /// Number of samples collected
    samples: usize,
}

impl CalibrationPoint {
    fn new(x: f32, y: f32) -> Self {
        Self {
            position: (x, y),
            completed: false,
            samples: 0,
        }
    }
}

/// Generate 9-point calibration grid
fn generate_calibration_points() -> Vec<CalibrationPoint> {
    vec![
        // Top row
        CalibrationPoint::new(0.1, 0.1),
        CalibrationPoint::new(0.5, 0.1),
        CalibrationPoint::new(0.9, 0.1),
        // Middle row
        CalibrationPoint::new(0.1, 0.5),
        CalibrationPoint::new(0.5, 0.5),
        CalibrationPoint::new(0.9, 0.5),
        // Bottom row
        CalibrationPoint::new(0.1, 0.9),
        CalibrationPoint::new(0.5, 0.9),
        CalibrationPoint::new(0.9, 0.9),
    ]
}

/// Run calibration
pub fn run_calibration(
    camera_index: usize,
    num_points: u32,
    save_path: Option<std::path::PathBuf>,
) -> Result<()> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let result = run_calibration_ui(&mut terminal, camera_index, num_points as usize);

    // Cleanup
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;

    // Save calibration if requested
    if let Some(path) = save_path {
        if let Ok(cal) = &result {
            save_calibration(cal, &path)?;
        }
    }

    result
}

fn run_calibration_ui<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    camera_index: usize,
    num_points: usize,
) -> Result<Calibrator> {
    // Create camera
    let mut config = CameraConfig::eye_tracking();
    config.camera_index = Some(camera_index);
    let mut camera = Camera::new(config)?;

    // Create inference pipeline
    let mut pipeline = InferencePipeline::real_time_pipeline();

    // Create calibrator
    let calibrator = Arc::new(Mutex::new(Calibrator::new()));
    let estimator = Arc::new(Mutex::new(GazeEstimator::new(None)));

    // Calibration state
    let mut points = generate_calibration_points();
    let mut current_point_index = 0;
    let mut calibrating = true;
    let mut sample_collection_start: Option<Instant> = None;
    let mut last_eye_position: Option<(f32, f32)> = None;

    println!("Starting calibration...");
    println!("Look at each point when it appears.");

    // Start camera
    camera.start()?;

    while calibrating {
        // Capture frame
        let frame = match camera.capture_frame() {
            Ok(f) => f,
            Err(e) => {
                tracing::warn!("Frame error: {}", e);
                continue;
            }
        };

        // Process frame
        let pixels = preprocess_frame(&frame.data, frame.width, frame.height, &PreprocessOptions::default());
        let result = pipeline.process_frame(&pixels, frame.width, frame.height);

        // Get gaze position
        let gaze_pos = result.screen_position().unwrap_or((0.5, 0.5));

        // Update UI
        terminal.draw(|f| {
            draw_calibration_ui(f, &points, current_point_index, gaze_pos);
        })?;

        // Handle calibration logic
        if current_point_index < points.len() {
            let point = &mut points[current_point_index];

            if !point.completed {
                // Check if gaze is on target
                let target = point.position;
                let dist = ((gaze_pos.0 - target.0).powi(2) + (gaze_pos.1 - target.1).powi(2)).sqrt();

                if dist < 0.05 {
                    // Gaze on target - start/continue sampling
                    if sample_collection_start.is_none() {
                        sample_collection_start = Some(Instant::now());
                        last_eye_position = Some(gaze_pos);
                    }

                    // Check if we've been fixating long enough (500ms per FR-EYE-CAL-001)
                    if let Some(start) = sample_collection_start {
                        if start.elapsed() >= Duration::from_millis(500) {
                            // Record sample
                            let mut cal = calibrator.lock().unwrap();
                            if cal.state() == CalibrationState::Idle {
                                cal.start_calibration();
                            }

                            // Convert gaze to screen coordinates
                            let eye_pt = Point::new(gaze_pos.0 as f64 * 1920.0, gaze_pos.1 as f64 * 1080.0);
                            let target_pt = Point::new(target.0 as f64 * 1920.0, target.1 as f64 * 1080.0);

                            if cal.record_sample(eye_pt, target_pt).is_ok() {
                                point.samples += 1;

                                if point.samples >= 3 {
                                    point.completed = true;
                                    sample_collection_start = None;
                                    println!(
                                        "Point {} completed ({} samples)",
                                        current_point_index + 1,
                                        point.samples
                                    );
                                }
                            }
                        }
                    }
                } else {
                    // Gaze off target - reset
                    sample_collection_start = None;
                }
            } else {
                // Point completed - move to next
                current_point_index += 1;
                if current_point_index >= points.len() {
                    // All points done - finalize calibration
                    let mut cal = calibrator.lock().unwrap();
                    match cal.finalize() {
                        Ok(_) => {
                            println!("Calibration successful!");
                            println!("Accuracy: {:.2} degrees", cal.accuracy());
                        }
                        Err(e) => {
                            eprintln!("Calibration failed: {}", e);
                        }
                    }
                    calibrating = false;
                }
            }
        }

        // Small delay
        std::thread::sleep(Duration::from_millis(16));
    }

    camera.stop()?;

    let cal = calibrator.lock().unwrap().clone();
    Ok(cal)
}

fn draw_calibration_ui<B: ratatui::backend::Backend>(
    f: &mut ratatui::Frame<B>,
    points: &[CalibrationPoint],
    current_index: usize,
    gaze_pos: (f32, f32),
) {
    let size = f.size();

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(10),
            Constraint::Length(5),
        ])
        .split(size);

    // Header
    let header = Paragraph::new("Calibration - Look at each point")
        .style(Style::new().on_blue().white())
        .alignment(Alignment::Center);
    f.render_widget(header, chunks[0]);

    // Calibration grid
    let grid_size = 3;
    let cell_width = size.width / 3;
    let cell_height = (size.height - 8) / 3;

    for (i, point) in points.iter().enumerate() {
        let row = i / grid_size;
        let col = i % grid_size;

        let x = col as u16 * cell_width + cell_width / 2;
        let y = 3 + row as u16 * cell_height + cell_height / 2;

        let symbol = if i == current_index {
            "◉"
        } else if point.completed {
            "●"
        } else {
            "○"
        };

        let style = if i == current_index {
            Style::new().red().bold()
        } else if point.completed {
            Style::new().green()
        } else {
            Style::new().white()
        };

        let paragraph = Paragraph::new(symbol)
            .style(style)
            .alignment(Alignment::Center);

        let area = Rect::new(x.saturating_sub(1), y.saturating_sub(1), 3, 3);
        f.render_widget(paragraph, area);
    }

    // Instructions
    let instructions = Paragraph::new(vec![
        Line::raw("Progress: "),
        Line::styled(
            format!("{}/{} points", current_index.min(points.len()), points.len()),
            Style::new().green(),
        ),
        Line::raw(" | "),
        Line::raw("Gaze: "),
        Line::styled(
            format!("({:.2}, {:.2})", gaze_pos.0, gaze_pos.1),
            Style::new().cyan(),
        ),
    ])
    .style(Style::new().on_black().white())
    .alignment(Alignment::Center);

    f.render_widget(instructions, chunks[2]);
}

fn save_calibration(calibrator: &Calibrator, path: &std::path::Path) -> Result<()> {
    let cal = calibrator
        .get_calibration()
        .ok_or_else(|| anyhow!("No calibration to save"))?;

    let data = serde_json::to_string_pretty(&cal)?;
    std::fs::write(path, data)?;

    println!("Calibration saved to: {:?}", path);
    Ok(())
}

/// Load calibration from file
pub fn load_calibration(path: &std::path::Path) -> Result<eyetracker_math::CalibrationMatrix> {
    let data = std::fs::read_to_string(path)?;
    let cal: eyetracker_math::CalibrationMatrix = serde_json::from_str(&data)?;
    Ok(cal)
}
