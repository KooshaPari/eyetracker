//! Unified output emitter for the eyetracker CLI.
//!
//! Provides `Format` (selectable via `--format text|json`) and helpers
//! that write gaze frames, errors, and info messages in the chosen format.
//! All machine-readable output goes to stdout; all human-readable labels
//! and TUI chrome go to stderr so they don't corrupt piped output.

use serde::Serialize;

/// Output format selected by the `--format` flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Format {
    /// Human-readable text (default).
    Text,
    /// Newline-delimited JSON (NDJSON); one object per line.
    Json,
}

/// A single gaze frame emitted to stdout in the chosen format.
#[derive(Debug, Serialize)]
pub struct GazeFrame {
    /// Monotonic timestamp in milliseconds since the process started.
    pub ts_ms: u64,
    /// Normalized gaze X in [0, 1] (0 = left edge of screen).
    pub gaze_x: f32,
    /// Normalized gaze Y in [0, 1] (0 = top edge of screen).
    pub gaze_y: f32,
    /// Inferred gaze event type: "fixation", "saccade", or "unknown".
    pub event: &'static str,
    /// Inference latency for this frame in milliseconds.
    pub latency_ms: f32,
}

/// Typed error record emitted when the pipeline surfaces a recoverable error.
#[derive(Debug, Serialize)]
pub struct ErrorEvent {
    /// Monotonic timestamp in milliseconds.
    pub ts_ms: u64,
    /// Short error code (e.g. `CAMERA_UNAVAILABLE`, `CALIBRATION_LOAD_FAILED`).
    pub code: &'static str,
    /// Human-readable message (never a raw Rust stack trace).
    pub message: String,
    /// Optional recovery hint shown to the operator.
    pub hint: Option<&'static str>,
}

/// Emit a single gaze frame in the chosen format.
pub fn emit_gaze(frame: &GazeFrame, fmt: Format) {
    match fmt {
        Format::Text => {
            println!(
                "{:>8}ms  gaze=({:.4}, {:.4})  event={:<9}  latency={:.1}ms",
                frame.ts_ms, frame.gaze_x, frame.gaze_y, frame.event, frame.latency_ms
            );
        }
        Format::Json => {
            // serde_json::to_string never fails on this struct; unwrap is sound.
            println!("{}", serde_json::to_string(frame).unwrap());
        }
    }
}

/// Emit a recoverable error in the chosen format.
///
/// Errors always go to **stderr** so they do not corrupt a JSON pipe.
pub fn emit_error(ev: &ErrorEvent, fmt: Format) {
    match fmt {
        Format::Text => {
            eprintln!(
                "ERROR [{}] {} {}",
                ev.code,
                ev.message,
                ev.hint.map(|h| format!("(hint: {h})")).unwrap_or_default()
            );
        }
        Format::Json => {
            eprintln!("{}", serde_json::to_string(ev).unwrap());
        }
    }
}

/// Print a plain info/status message.
///
/// For JSON mode these go to stderr to avoid polluting the NDJSON stream.
pub fn info(msg: &str, fmt: Format) {
    match fmt {
        Format::Text => println!("{msg}"),
        Format::Json => eprintln!("{msg}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_gaze_frame_is_valid_json() {
        let frame = GazeFrame {
            ts_ms: 1234,
            gaze_x: 0.5,
            gaze_y: 0.3,
            event: "fixation",
            latency_ms: 12.5,
        };
        let s = serde_json::to_string(&frame).unwrap();
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["ts_ms"], 1234u64);
        assert_eq!(v["event"], "fixation");
    }

    #[test]
    fn json_error_event_is_valid_json() {
        let ev = ErrorEvent {
            ts_ms: 42,
            code: "CAMERA_UNAVAILABLE",
            message: "no camera at index 0".to_string(),
            hint: Some("check --camera-index or run --list-cameras"),
        };
        let s = serde_json::to_string(&ev).unwrap();
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["code"], "CAMERA_UNAVAILABLE");
        assert!(v["hint"].is_string());
    }

    #[test]
    fn format_value_enum_parses() {
        use clap::ValueEnum;
        assert!(Format::from_str("json", true).is_ok());
        assert!(Format::from_str("text", true).is_ok());
        assert!(Format::from_str("xml", true).is_err());
    }
}
