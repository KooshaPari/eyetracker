//! FocalPoint integration connector (FR-EYE-INTEROP-003)
//!
//! Publishes gaze events to a local Unix domain socket as a stream of
//! newline-delimited JSON tuples:
//!   {"window_id": "...", "gaze_x": 0.5, "gaze_y": 0.5, "ts": 1234567890}
//!
//! FocalPoint subscribers connect to the socket and consume the stream.
//! The connector is a thin shim that wraps the eye tracker pipeline and
//! forwards TrackingResult events to the bus.

use crate::pipeline::TrackingResult;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

/// Default socket path used by FocalPoint subscribers
pub const DEFAULT_SOCKET: &str = "/tmp/eyetracker-focalpoint.sock";

/// Gaze event payload sent over the FocalPoint bus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FocalPointGazeEvent {
    /// Window ID at the gaze point (0 if unknown)
    pub window_id: u64,
    /// Gaze x in normalized screen coords (0.0 - 1.0)
    pub gaze_x: f32,
    /// Gaze y in normalized screen coords (0.0 - 1.0)
    pub gaze_y: f32,
    /// Unix timestamp in milliseconds
    pub ts: u64,
    /// Smoothed vs raw (true = Kalman-filtered)
    pub smoothed: bool,
}

/// Connector that publishes gaze events to FocalPoint
pub struct FocalPointConnector {
    socket_path: PathBuf,
    stream: Mutex<Option<UnixStream>>,
}

impl FocalPointConnector {
    /// Create a new connector. Does not connect until `connect()` is called.
    pub fn new(socket_path: impl Into<PathBuf>) -> Self {
        Self {
            socket_path: socket_path.into(),
            stream: Mutex::new(None),
        }
    }

    /// Default connector using the standard FocalPoint socket path
    pub fn default_socket() -> Self {
        Self::new(DEFAULT_SOCKET)
    }

    /// Connect to the FocalPoint bus. Idempotent.
    pub fn connect(&self) -> Result<()> {
        let mut guard = self.stream.lock().unwrap();
        if guard.is_some() {
            return Ok(());
        }
        let stream = UnixStream::connect(&self.socket_path)
            .with_context(|| format!("connecting to FocalPoint at {:?}", self.socket_path))?;
        stream.set_nonblocking(false).ok();
        *guard = Some(stream);
        Ok(())
    }

    /// Disconnect from the bus
    pub fn disconnect(&self) {
        let mut guard = self.stream.lock().unwrap();
        *guard = None;
    }

    /// Returns true if currently connected
    pub fn is_connected(&self) -> bool {
        self.stream.lock().unwrap().is_some()
    }

    /// Publish a tracking result as a FocalPoint gaze event.
    /// Uses smoothed gaze when available, falling back to raw gaze.
    pub fn publish(&self, result: &TrackingResult) -> Result<()> {
        let (x, y, smoothed) = result
            .smoothed_gaze
            .map(|(x, y)| (x, y, true))
            .or_else(|| {
                result.gaze.as_ref().map(|g| (g.combined.x, g.combined.y, false))
            })
            .unwrap_or((0.0, 0.0, false));

        let event = FocalPointGazeEvent {
            window_id: 0, // window detection not yet wired in
            gaze_x: x,
            gaze_y: y,
            ts: unix_millis(),
            smoothed,
        };
        self.send_event(&event)
    }

    fn send_event(&self, event: &FocalPointGazeEvent) -> Result<()> {
        let mut guard = self.stream.lock().unwrap();
        let stream = guard
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Not connected to FocalPoint bus"))?;
        let json = serde_json::to_string(event)?;
        writeln!(stream, "{}", json)?;
        stream.flush()?;
        Ok(())
    }
}

impl Default for FocalPointConnector {
    fn default() -> Self {
        Self::default_socket()
    }
}

impl Drop for FocalPointConnector {
    fn drop(&mut self) {
        self.disconnect();
    }
}

fn unix_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::CalibrationResult;
    use eyetracker_camera::{Frame, PixelFormat};
    use std::os::unix::net::UnixListener;
    use std::time::Instant;

    fn dummy_result() -> TrackingResult {
        TrackingResult {
            frame: Frame {
                data: vec![0; 640 * 480 * 3],
                width: 640,
                height: 480,
                format: PixelFormat::Rgb8,
                timestamp: Instant::now(),
                frame_number: 1,
            },
            face: None,
            gaze: None,
            smoothed_gaze: Some((0.5, 0.5)),
            events: vec![],
            processing_time_ms: 10.0,
        }
    }

    #[test]
    fn test_default_socket_path() {
        let c = FocalPointConnector::default_socket();
        assert_eq!(c.socket_path.to_str().unwrap(), DEFAULT_SOCKET);
        assert!(!c.is_connected());
    }

    #[test]
    fn test_publish_requires_connection() {
        let c = FocalPointConnector::new("/tmp/nonexistent-eyetracker-test.sock");
        let result = c.publish(&dummy_result());
        assert!(result.is_err());
    }

    #[test]
    fn test_connect_and_publish_round_trip() {
        // Start a Unix listener on a temp socket
        let tmp = std::env::temp_dir().join(format!(
            "eyetracker-focalpoint-test-{}.sock",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&tmp);
        let listener = UnixListener::bind(&tmp).expect("bind listener");

        let c = FocalPointConnector::new(&tmp);
        c.connect().expect("connect");
        assert!(c.is_connected());

        // Accept a connection in another thread
        let accept = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            use std::io::Read;
            let mut buf = [0u8; 256];
            let n = stream.read(&mut buf).expect("read");
            String::from_utf8_lossy(&buf[..n]).to_string()
        });

        c.publish(&dummy_result()).expect("publish");
        let received = accept.join().unwrap();
        let _ = c.disconnect();

        // Verify the published event
        let v: serde_json::Value = serde_json::from_str(received.trim()).expect("parse json");
        assert_eq!(v["gaze_x"], 0.5);
        assert_eq!(v["gaze_y"], 0.5);
        assert_eq!(v["smoothed"], true);
        assert!(v["ts"].as_u64().unwrap() > 0);

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_publish_falls_back_to_raw_gaze() {
        let tmp = std::env::temp_dir().join(format!(
            "eyetracker-focalpoint-test-raw-{}.sock",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&tmp);
        let listener = UnixListener::bind(&tmp).expect("bind listener");
        let c = FocalPointConnector::new(&tmp);
        c.connect().expect("connect");

        let accept = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            use std::io::Read;
            let mut buf = [0u8; 256];
            let n = stream.read(&mut buf).expect("read");
            String::from_utf8_lossy(&buf[..n]).to_string()
        });

        // No smoothed gaze — should fall back to a raw gaze
        let mut r = dummy_result();
        r.smoothed_gaze = None;
        r.gaze = Some(crate::gaze_estimator::GazeResult {
            screen_point: crate::gaze_estimator::Point2D { x: 0.3, y: 0.4 },
            combined: crate::gaze_estimator::GazeVector { x: 0.3, y: 0.4, z: 0.0 },
            left: crate::gaze_estimator::GazeVector { x: 0.0, y: 0.0, z: 0.0 },
            right: crate::gaze_estimator::GazeVector { x: 0.0, y: 0.0, z: 0.0 },
            confidence: 0.9,
        });
        c.publish(&r).expect("publish");
        let received = accept.join().unwrap();
        let v: serde_json::Value = serde_json::from_str(received.trim()).expect("parse json");
        assert_eq!(v["gaze_x"], 0.3);
        assert_eq!(v["gaze_y"], 0.4);
        assert_eq!(v["smoothed"], false);

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_idempotent_connect() {
        let tmp = std::env::temp_dir().join(format!(
            "eyetracker-focalpoint-test-idem-{}.sock",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&tmp);
        let _listener = UnixListener::bind(&tmp).expect("bind listener");

        let c = FocalPointConnector::new(&tmp);
        c.connect().expect("first connect");
        c.connect().expect("second connect (idempotent)");
        assert!(c.is_connected());
        c.disconnect();
        assert!(!c.is_connected());

        let _ = std::fs::remove_file(&tmp);
    }

    // Smoke test for CalibrationResult (just to keep the import used)
    #[test]
    fn test_calibration_result_smoke() {
        let r = CalibrationResult {
            samples: vec![],
            quality: 0.5,
            success: false,
        };
        assert!(!r.success);
    }
}
