//! Main application logic for eye tracking

use anyhow::Result;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use eyetracker_camera::{Camera, CameraConfig};
use eyetracker_core::{Calibrator, GazeEstimator};
use eyetracker_inference::{
    preprocess_frame, InferencePipeline, PreprocessOptions,
};
use ratatui::{backend::CrosstermBackend, layout::*, style::*, text::*, widgets::*, Terminal};
use std::io;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Run the eye tracker
pub fn run_tracker(
    camera_index: usize,
    fps: u32,
    smooth: bool,
    debug: bool,
    _output: crate::OutputFormat,
) -> Result<()> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create camera
    let mut config = CameraConfig::eye_tracking();
    config.camera_index = Some(camera_index);
    config.target_fps = fps.min(60);

    let mut camera = Camera::new(config)?;

    // Create inference pipeline
    let mut pipeline = if smooth {
        InferencePipeline::real_time_pipeline()
    } else {
        InferencePipeline::high_accuracy_pipeline()
    };

    // Create calibrator and estimator
    let calibrator = Arc::new(Mutex::new(Calibrator::new()));
    let estimator = Arc::new(Mutex::new(GazeEstimator::new(None)));

    // Tracking state
    let tracking_state = Arc::new(Mutex::new(TrackingState::default()));

    // Setup terminal UI
    let result = run_ui(
        &mut terminal,
        &mut camera,
        &mut pipeline,
        calibrator,
        estimator,
        tracking_state.clone(),
        debug,
    );

    // Cleanup
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen, DisableMouseCapture)?;

    if let Err(e) = &result {
        eprintln!("Error: {}", e);
    }

    result
}

/// Tracking state
#[derive(Debug, Clone, Default)]
pub struct TrackingState {
    pub gaze_x: f64,
    pub gaze_y: f64,
    pub confidence: f64,
    pub fps: f64,
    pub frame_count: u64,
    pub face_detected: bool,
    pub last_update: Option<Instant>,
    pub latency_ms: f64,
}

impl TrackingState {
    pub fn update(&mut self, result: &eyetracker_inference::InferenceResult, latency: Duration) {
        if let Some(pos) = result.screen_position() {
            self.gaze_x = pos.0 as f64;
            self.gaze_y = pos.1 as f64;
            self.confidence = result
                .gaze
                .as_ref()
                .map(|g| g.combined.confidence as f64)
                .unwrap_or(0.0);
        }
        self.face_detected = result.face_detected;
        self.frame_count += 1;
        self.latency_ms = latency.as_secs_f64() * 1000.0;

        if let Some(last) = self.last_update {
            let elapsed = last.elapsed().as_secs_f64();
            if elapsed > 0.0 {
                self.fps = self.frame_count as f64 / elapsed;
            }
        }
        self.last_update = Some(Instant::now());
    }
}

fn run_ui<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    camera: &mut Camera,
    pipeline: &mut InferencePipeline,
    _calibrator: Arc<Mutex<Calibrator>>,
    _estimator: Arc<Mutex<GazeEstimator>>,
    state: Arc<Mutex<TrackingState>>,
    debug: bool,
) -> Result<()> {
    let mut running = true;

    // Start camera
    camera.start()?;

    while running {
        // Capture frame
        match camera.capture_frame() {
            Ok(frame) => {
                // Process frame using inference crate's preprocessing
                let pixels = preprocess_frame(&frame.data, frame.width, frame.height, &PreprocessOptions::default());
                let result = pipeline.process_frame(&pixels, frame.width, frame.height);

                // Update state
                {
                    let mut s = state.lock().unwrap();
                    s.update(&result, result.total_latency);
                }

                // Draw UI
                terminal.draw(|f| {
                    let state = state.lock().unwrap();
                    draw_ui(f, &state, debug);
                })?;
            }
            Err(e) => {
                tracing::warn!("Frame capture error: {}", e);
            }
        }

        // Handle input
        if event::poll(Duration::from_millis(16))? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') | KeyCode::Char('Q') => {
                        running = false;
                    }
                    KeyCode::Char('c') | KeyCode::Char('C') => {
                        tracing::info!("Calibration started");
                    }
                    KeyCode::Char('r') | KeyCode::Char('R') => {
                        tracing::info!("Tracking reset");
                    }
                    _ => {}
                }
            }
        }
    }

    camera.stop()?;
    Ok(())
}

fn draw_ui<B: ratatui::backend::Backend>(
    f: &mut ratatui::Frame<B>,
    state: &TrackingState,
    debug: bool,
) {
    let size = f.size();

fn draw_debug_view(state: &TrackingState) -> Vec<Line<'static>> {
    vec![
        Line::from(vec![
            Span::raw("FPS: "),
            Span::raw(format!("{:.1}", state.metrics.fps)),
            Span::raw(" | Latency: "),
            Span::raw(format!("{:.1}ms", state.metrics.latency_ms)),
            Span::raw(" | Faces: "),
            Span::raw(state.metrics.face_count.to_string()),
        ]),
        Line::from(vec![
            Span::raw("Gaze: ("),
            Span::raw(format!("{:.3}, {:.3}", state.gaze.x, state.gaze.y)),
            Span::raw(") | Confidence: "),
            Span::raw(format!("{:.2}", state.gaze.confidence)),
        ]),
        Line::from(vec![
            Span::raw("Calibration: "),
            Span::raw(if state.calibration.is_calibrated() {
                "READY"
            } else {
                "NOT CALIBRATED"
            }),
        ]),
    ]
}
    let header = Paragraph::new(Text::from(vec![
        Line::from(vec![
            Span::raw("EyeTracker CLI | "),
            Span::styled("Q", Style::new().bold()),
            Span::raw(" Quit | "),
            Span::styled("C", Style::new().bold()),
            Span::raw(" Calibrate | "),
            Span::styled("R", Style::new().bold()),
            Span::raw(" Reset"),
        ]),
    ]))
    .style(Style::new().on_blue().black())
    .alignment(Alignment::Center);

    // Main layout
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(10),
            Constraint::Length(3),
        ])
        .split(size);

    f.render_widget(header, chunks[0]);

    // Main content
    let main_content = if debug {
        draw_debug_view(state)
    } else {
        draw_gaze_view(state)
    };

    let main_widget = Paragraph::new(main_content)
        .block(Block::default().title("Eye Tracking").borders(Borders::ALL))
        .alignment(Alignment::Center);

    f.render_widget(main_widget, chunks[1]);

    // Footer
    let footer = Paragraph::new(Text::from(vec![
        Line::from(vec![
            Span::raw("Gaze: "),
            Span::styled(
                format!("({:.2}, {:.2})", state.gaze_x, state.gaze_y),
                Style::new().green(),
            ),
            Span::raw(" | Confidence: "),
            Span::styled(
                format!("{:.0}%", state.confidence * 100.0),
                if state.confidence > 0.7 {
                    Style::new().green()
                } else if state.confidence > 0.4 {
                    Style::new().yellow()
                } else {
                    Style::new().red()
                },
            ),
            Span::raw(" | FPS: "),
            Span::raw(format!("{:.1}", state.fps)),
            Span::raw(" | Latency: "),
            Span::raw(format!("{:.1}ms", state.latency_ms)),
        ]),
    ]))
    .style(Style::new().on_black().white())
    .alignment(Alignment::Center);

    f.render_widget(footer, chunks[2]);
}

fn draw_gaze_view(state: &TrackingState) -> Vec<Line<'static>> {
    let eye_symbol = if state.face_detected {
        if state.confidence > 0.7 {
            "[●]"
        } else {
            "[◐]"
        }
    } else {
        "[○]"
    };

    // Create a simple visual representation
    vec![
        Line::from(vec![
            Span::raw(eye_symbol),
        ]),
        Line::from(vec![
            Span::raw(""),
        ]),
        Line::from(vec![
            Span::raw("Looking at: "),
            Span::styled(
                format!("({:.2}, {:.2})", state.gaze_x, state.gaze_y),
                Style::new().green(),
            ),
        ]),
    ]
}
