//! eyetracker-cli: Command-line eye tracker
//!
//! Real-time eye tracking using webcam input with TUI visualization.

mod app;
mod calibration;
mod ui;

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

/// Eye tracking CLI
#[derive(Parser, Debug)]
#[command(
    name = "eyetracker",
    about = "Real-time eye tracking CLI",
    version
)]
struct Cli {
    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,

    /// Camera device index
    #[arg(short, long, default_value = "0")]
    camera: usize,

    /// Configuration file
    #[arg(short, long)]
    config: Option<std::path::PathBuf>,

    /// Output format
    #[arg(short, long, value_enum, default_value = "json")]
    output: OutputFormat,

    /// Output file (default: stdout)
    #[arg(short, long)]
    output_file: Option<std::path::PathBuf>,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Debug, Clone, clap::ValueEnum)]
pub enum OutputFormat {
    /// JSON lines format
    Json,
    /// Text format
    Text,
    /// CSV format
    Csv,
    /// TUI visualization
    Tui,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Start real-time eye tracking
    Track {
        /// Frame rate limit
        #[arg(long, default_value = "30")]
        fps: u32,

        /// Enable smoothing
        #[arg(long, default_value = "true")]
        smooth: bool,

        /// Show debug visualization
        #[arg(long, default_value = "false")]
        debug: bool,
    },

    /// Run calibration
    Calibrate {
        /// Number of calibration points
        #[arg(long, default_value = "9")]
        points: u32,

        /// Save calibration to file
        #[arg(long)]
        save: Option<std::path::PathBuf>,
    },

    /// List available cameras
    ListCameras,

    /// Show configuration and status
    Status,

    /// Export gaze data to file
    Export {
        /// Input session file
        input: std::path::PathBuf,

        /// Output file
        output: std::path::PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    init_logging(cli.verbose);

    tracing::info!("Starting eyetracker CLI");
    tracing::info!("Version: {}", env!("CARGO_PKG_VERSION"));

    // Run command
    match cli.command.as_ref().unwrap_or(&Commands::Track { fps: 30, smooth: true, debug: false }) {
        Commands::Track { fps, smooth, debug } => {
            app::run_tracker(cli.camera, *fps, *smooth, *debug, cli.output)?;
        }
        Commands::Calibrate { points, save } => {
            calibration::run_calibration(cli.camera, *points, save.clone())?;
        }
        Commands::ListCameras => {
            list_cameras()?;
        }
        Commands::Status => {
            show_status()?;
        }
        Commands::Export { input, output } => {
            export_data(input, output)?;
        }
    }

    Ok(())
}

fn init_logging(verbose: bool) {
    let filter = if verbose {
        EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new("debug"))
    } else {
        EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new("info"))
    };

    tracing_subscriber::registry()
        .with(fmt::layer().with_target(true))
        .with(filter)
        .init();
}

fn list_cameras() -> Result<()> {
    println!("Available cameras:");
    println!("==================");

    match eyetracker_camera::list_cameras() {
        Ok(cameras) => {
            if cameras.is_empty() {
                println!("No cameras found");
            } else {
                for cam in &cameras {
                    println!("  [{}] {}", cam.index, cam.name);
                    println!("      Supported formats: {}", cam.supported_formats.join(", "));
                }
            }
        }
        Err(e) => {
            eprintln!("Error listing cameras: {}", e);
        }
    }

    Ok(())
}

fn show_status() -> Result<()> {
    println!("EyeTracker Status");
    println!("=================");
    println!("Version: {}", env!("CARGO_PKG_VERSION"));
    println!();

    // Check camera
    println!("Camera:");
    match eyetracker_camera::list_cameras() {
        Ok(cameras) => {
            println!("  Found {} camera(s)", cameras.len());
        }
        Err(e) => {
            println!("  Error: {}", e);
        }
    }

    println!();
    println!("To use eye tracking:");
    println!("  1. Run 'eyetracker list-cameras' to find your camera");
    println!("  2. Run 'eyetracker calibrate' to calibrate");
    println!("  3. Run 'eyetracker track' to start tracking");

    Ok(())
}

fn export_data(input: &std::path::Path, output: &std::path::Path) -> Result<()> {
    println!("Exporting from {:?} to {:?}", input, output);

    // Read session file
    let data = std::fs::read_to_string(input)?;
    let gaze_data: serde_json::Value = serde_json::from_str(&data)?;

    // Write output (placeholder - implement export formats)
    std::fs::write(output, serde_json::to_string_pretty(&gaze_data)?)?;

    println!("Export complete: {} bytes", std::fs::metadata(output)?.len());

    Ok(())
}
