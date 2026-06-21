//! Eye tracker CLI - Real-time eye tracking from webcam
//!
//! Provides a terminal UI showing gaze tracking data, calibration,
//! and performance metrics using the eyetracker pipeline.

mod app;
mod calibration;
mod mouse;
mod ui;

use anyhow::Result;
use clap::Parser;

/// Real-time eye tracking from webcam
#[derive(Parser, Debug)]
#[command(name = "eyetracker", version, about)]
struct Args {
    /// Camera device index
    #[arg(short = 'i', long, default_value = "0")]
    camera_index: usize,

    /// Camera resolution width
    #[arg(long, default_value = "640")]
    width: u32,

    /// Camera resolution height
    #[arg(long, default_value = "480")]
    height: u32,

    /// Target frame rate
    #[arg(short, long, default_value = "60")]
    fps: u32,

    /// Run calibration mode
    #[arg(short, long)]
    calibrate: bool,

    /// Load saved calibration on startup and print quality
    #[arg(long)]
    load_calibration: bool,

    /// List available cameras and exit
    #[arg(long)]
    list_cameras: bool,

    /// Gaze smoothing factor (0.0-0.95)
    #[arg(long, default_value = "0.6")]
    smoothing: f32,

    /// Screen distance in mm
    #[arg(long, default_value = "600")]
    screen_distance: f32,

    /// Show debug overlay with face mesh
    #[arg(long)]
    debug: bool,

    /// Dump gaze data as CSV to stdout
    #[arg(long)]
    csv: bool,

    /// Duration in seconds (0 = unlimited)
    #[arg(short, long, default_value = "0")]
    duration: u64,

    /// Dwell-click duration in ms (200-1000ms; FR-EYE-ACCESS-001).
    /// Values outside the spec range are clamped to the nearest boundary.
    #[arg(long, default_value = "500")]
    dwell_ms: u64,

    /// Disable mouse output (FR-EYE-ACCESS-001).
    /// When set, dwell-click and gaze-scroll actions are logged but no
    /// real mouse events are posted. Useful for headless tests or when
    /// the user wants to observe the pipeline without driving the cursor.
    #[arg(long)]
    no_mouse_output: bool,

    /// Screen width in pixels (for translating normalized gaze to display coords).
    /// Defaults to 1920 (Full HD); override for HiDPI or smaller displays.
    #[arg(long, default_value = "1920")]
    screen_width: u32,

    /// Screen height in pixels.
    #[arg(long, default_value = "1080")]
    screen_height: u32,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "eyetracker=info,warn".into()),
        )
        .with_target(false)
        .init();

    // Handle --list-cameras
    if args.list_cameras {
        let cameras = eyetracker_camera::list_cameras();
        if cameras.is_empty() {
            println!("No cameras found.");
            return Ok(());
        }
        println!("=== Available Cameras ===");
        for cam in &cameras {
            println!("  [{}] {} — {}", cam.index, cam.name, cam.description);
        }
        return Ok(());
    }

    // Handle --load-calibration
    if args.load_calibration {
        match calibration::load_calibration()? {
            Some(cal) => {
                println!("Calibration loaded (quality: {:.1}%)", cal.quality * 100.0);
            }
            None => {
                println!("No calibration file found. Run --calibrate first.");
            }
        }
        return Ok(());
    }

    // Build camera config
    let camera_config = eyetracker_camera::CameraConfig {
        target_fps: args.fps,
        width: args.width,
        height: args.height,
        camera_index: args.camera_index,
        low_light_mode: true,
    };

    // Build pipeline config
    let pipeline_config = eyetracker_inference::PipelineConfig {
        camera: camera_config,
        use_geometric_fallback: true,
        smoothing: args.smoothing,
        screen_distance_mm: args.screen_distance,
        debug_overlay: args.debug,
    };

    if args.calibrate {
        println!("Starting calibration mode...");
        calibration::run_calibration(&pipeline_config)?;
        return Ok(());
    }

    if args.csv {
        println!("Starting CSV dump mode...");
        let dwell_duration = std::time::Duration::from_millis(args.dwell_ms);
        app::run_csv_dump(
            &pipeline_config,
            args.duration,
            dwell_duration,
            args.no_mouse_output,
            args.screen_width,
            args.screen_height,
        )?;
        return Ok(());
    }

    // Run interactive TUI mode
    let mut terminal = ratatui::Terminal::new(ratatui::backend::CrosstermBackend::new(std::io::stdout()))?;
    let dwell_duration = std::time::Duration::from_millis(args.dwell_ms);
    let result = app::run_tui(
        &mut terminal,
        &pipeline_config,
        args.duration,
        dwell_duration,
        args.no_mouse_output,
        args.screen_width,
        args.screen_height,
    );
    let _ = crossterm::terminal::disable_raw_mode();
    result
}
