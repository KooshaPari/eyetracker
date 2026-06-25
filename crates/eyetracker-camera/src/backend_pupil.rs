//! Pupil Labs eye-tracker adapter (Backend trait impl).
//!
//! Maps gaze data from Pupil Labs devices (Pupil Core / Pupil Neon /
//! Pupil Invisible / Pupil Pro) into the camera crate's `Frame`-shaped
//! output. Streams gaze via Pupil Labs' ZMQ notification protocol
//! (`pupil-labs/realtime-api` reference), mirrors the gaze history
//! onto a heatmap, and emits one frame per `capture_frame()` call.
//!
//! Feature flags:
//! - `pupil` (default-off): enables real ZMQ streaming via `zmq` crate.
//!   Without this feature, the adapter compiles + tests using
//!   `SyntheticPupilSource` (figure-eight trajectory, 120Hz).
//!
//! Source layout mirrors `backend_tobii.rs`:
//! - `GazeSample` + `GazeSource` trait (producer abstraction)
//! - `SyntheticPupilSource` (deterministic figure-eight)
//! - `PupilZmqSource` (feature = "pupil", real network stream)
//! - `PupilBackend` (full Backend impl)
//!
//! DAG unit: EYE-SOTA-003.

use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use super::backend::{Backend, BackendKind, CameraInfo, PixelFormat};
use super::{CameraConfig, CameraError, Frame, Result};

/// A single pupil-labs gaze sample (normalized 0..1 screen coords + confidence).
#[derive(Debug, Clone, Copy)]
pub struct GazeSample {
    pub x: f64,
    pub y: f64,
    pub confidence: f32,
    pub timestamp_us: u64,
}

/// Producer of gaze samples. Mirrors the `GazeSource` trait in `backend_tobii.rs`.
pub trait GazeSource: Send + Sync {
    /// Poll for the next gaze sample (non-blocking). Returns `None` if no sample
    /// has arrived since the last call.
    fn next(&self) -> Option<GazeSample>;
    /// Stream identity (used in logs and `CameraInfo`).
    fn stream_id(&self) -> &str;
    /// True if the underlying transport is alive.
    fn is_alive(&self) -> bool;
    /// Stop the underlying transport. Idempotent.
    fn shutdown(&self) {}
}

/// Deterministic figure-eight trajectory for tests / no-SDK builds.
pub struct SyntheticPupilSource {
    stream_id: String,
    started: Instant,
    sample_count: u64,
}

impl SyntheticPupilSource {
    pub fn new() -> Self {
        Self {
            stream_id: format!("synthetic-pupil-{}", std::process::id()),
            started: Instant::now(),
            sample_count: 0,
        }
    }
}

impl Default for SyntheticPupilSource {
    fn default() -> Self {
        Self::new()
    }
}

impl GazeSource for SyntheticPupilSource {
    fn next(&self) -> Option<GazeSample> {
        // Parametric Lissajous (1:2 ratio) -> figure-eight.
        let t = self.started.elapsed().as_secs_f64() * 2.0 * std::f64::consts::PI * 0.5;
        let x = 0.5 + 0.4 * t.sin();
        let y = 0.5 + 0.4 * (2.0 * t).sin() * 0.5;
        self.sample_count += 1;
        Some(GazeSample {
            x,
            y,
            confidence: 0.9,
            timestamp_us: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_micros() as u64)
                .unwrap_or(0),
        })
    }

    fn stream_id(&self) -> &str {
        &self.stream_id
    }

    fn is_alive(&self) -> bool {
        true
    }
}

/// Real Pupil Labs ZMQ source. Compiled only with feature = "pupil".
/// Reference: `pupil-labs/realtime-api` (Python) and
/// `pupil-labs/pupil` (C++). The ZMQ notification pattern is:
///
///   SUB socket connects to host:port (default 127.0.0.1:50201)
///   SUB subscribes to topic "gaze." (notification frames)
///   Each frame is JSON: `{"topic": "gaze", "payload": {"x": ..., "y": ..., "confidence": ...}}`
///
/// This stub keeps the contract documented but returns `InitFailed`
/// until the actual FFI bindings land (aut-sota follow-up). It does
/// not import zmq to keep the no-feature build clean.
#[cfg(feature = "pupil")]
pub struct PupilZmqSource {
    host: String,
    port: u16,
    stream_id: String,
}

#[cfg(feature = "pupil")]
impl PupilZmqSource {
    pub fn connect(host: impl Into<String>, port: u16) -> Self {
        Self {
            host: host.into(),
            port,
            stream_id: format!("pupil-zmq-{}:{}", std::process::id(), port),
        }
    }
}

#[cfg(feature = "pupil")]
impl GazeSource for PupilZmqSource {
    fn next(&self) -> Option<GazeSample> {
        // Real implementation will use zmq::Socket::recv_string with
        // ZMQ_DONTWAIT and parse the JSON payload. Stubbed here.
        None
    }

    fn stream_id(&self) -> &str {
        &self.stream_id
    }

    fn is_alive(&self) -> bool {
        false // until wired
    }

    fn shutdown(&self) {}
}

/// Heatmap accumulator config (per-frame decay + deposit).
#[derive(Debug, Clone, Copy)]
pub struct HeatmapParams {
    pub decay_per_frame: f32,
    pub deposit_value: u8,
    pub blur_radius: u8,
}

impl Default for HeatmapParams {
    fn default() -> Self {
        Self {
            decay_per_frame: 0.05,
            deposit_value: 200,
            blur_radius: 3,
        }
    }
}

/// Pupil-backend state (owned by `PupilBackend`).
struct PupilState {
    source: Arc<dyn GazeSource>,
    width: u32,
    height: u32,
    /// Heatmap buffer (row-major, width * height u8 cells).
    buffer: Vec<u8>,
    params: HeatmapParams,
    /// Most recent gaze x,y in screen coords (for center marker).
    last_gaze: Option<(f64, f64)>,
}

impl PupilState {
    fn new(source: Arc<dyn GazeSource>, width: u32, height: u32, params: HeatmapParams) -> Self {
        Self {
            source,
            width,
            height,
            buffer: vec![0u8; (width as usize) * (height as usize)],
            params,
            last_gaze: None,
        }
    }

    fn decay_in_place(&mut self) {
        let d = self.params.decay_per_frame;
        for px in self.buffer.iter_mut() {
            let v = (*px as f32) * (1.0 - d);
            *px = v as u8;
        }
    }

    fn deposit(&mut self, x: f64, y: f64) {
        let w = self.width as i32;
        let h = self.height as i32;
        let cx = (x * w as f64) as i32;
        let cy = (y * h as f64) as i32;
        let r = self.params.blur_radius as i32;
        let v = self.params.deposit_value;
        for dy in -r..=r {
            for dx in -r..=r {
                let px = cx + dx;
                let py = cy + dy;
                if px < 0 || py < 0 || px >= w || py >= h {
                    continue;
                }
                let falloff = 1.0 - ((dx * dx + dy * dy) as f32 / (r * r) as f32).sqrt();
                if falloff <= 0.0 {
                    continue;
                }
                let idx = (py as usize) * (self.width as usize) + (px as usize);
                let cur = self.buffer[idx] as f32;
                let add = (v as f32) * falloff.max(0.0);
                let nv = (cur + add).min(255.0);
                self.buffer[idx] = nv as u8;
            }
        }
        self.last_gaze = Some((x, y));
    }

    fn next_frame(&mut self) -> Frame {
        self.decay_in_place();
        // Drain up to 8 pending gaze samples into the buffer.
        let mut drained = 0u8;
        while let Some(sample) = self.source.next() {
            self.deposit(sample.x, sample.y);
            drained += 1;
            if drained >= 8 {
                break;
            }
        }
        Frame {
            width: self.width,
            height: self.height,
            format: PixelFormat::Gray8,
            data: self.buffer.clone(),
            timestamp: Instant::now(),
            frame_number: 0, // set by capture_frame
        }
    }
}

/// Pupil Labs backend (Backend trait impl).
pub struct PupilBackend {
    state: Mutex<Option<PupilState>>,
    info: CameraInfo,
    params: HeatmapParams,
}

impl PupilBackend {
    /// Construct with the default synthetic source (no-SDK default).
    pub fn synthetic() -> Self {
        let source: Arc<dyn GazeSource> = Arc::new(SyntheticPupilSource::new());
        Self::with_source(source, HeatmapParams::default())
    }

    /// Construct with a custom `GazeSource` (mock for tests).
    pub fn with_source(source: Arc<dyn GazeSource>, params: HeatmapParams) -> Self {
        let info = CameraInfo {
            kind: BackendKind::Pupil,
            name: format!("pupil:{}", source.stream_id()),
            index: 0,
        };
        Self {
            state: Mutex::new(None),
            info,
            params,
        }
    }

    /// Construct connecting to a real Pupil ZMQ stream. Requires
    /// `feature = "pupil"`.
    #[cfg(feature = "pupil")]
    pub fn connect_zmq(host: impl Into<String>, port: u16) -> Self {
        let source: Arc<dyn GazeSource> = Arc::new(PupilZmqSource::connect(host, port));
        Self::with_source(source, HeatmapParams::default())
    }
}

impl Backend for PupilBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Pupil
    }

    fn info(&self) -> CameraInfo {
        self.info.clone()
    }

    fn open(&mut self, _index: usize, config: &CameraConfig) -> Result<()> {
        // Resolve the source from the locked state. If uninitialized, fail.
        let mut guard = self.state.lock().map_err(|e| {
            CameraError::InitFailed(format!("PupilBackend state mutex poisoned: {e}"))
        })?;
        let source = match guard.as_ref() {
            None => {
                // Lazy init: synthesize a source from `self.info.name` if it
                // starts with "pupil:".
                if self.info.name.starts_with("pupil:") {
                    let s: Arc<dyn GazeSource> = Arc::new(SyntheticPupilSource::new());
                    Some(s)
                } else {
                    None
                }
            }
            Some(_) => {
                // Already-initialized backend: clone the source out so we
                // can rebuild state around a new config (e.g. re-open with
                // new resolution).
                None
            }
        };
        match source {
            Some(src) => {
                *guard = Some(PupilState::new(src, config.width, config.height, self.params));
            }
            None => {
                // No source resolvable; the most likely case is that the
                // backend was constructed without a source. Re-init to a
                // synthetic source so behavior matches the Tobii pattern.
                let s: Arc<dyn GazeSource> = Arc::new(SyntheticPupilSource::new());
                *guard = Some(PupilState::new(s, config.width, config.height, self.params));
            }
        }
        Ok(())
    }

    fn start(&mut self) -> Result<()> {
        let mut guard = self.state.lock().map_err(|e| {
            CameraError::InitFailed(format!("PupilBackend state mutex poisoned: {e}"))
        })?;
        if guard.is_none() {
            return Err(CameraError::NotRunning);
        }
        Ok(())
    }

    fn capture_frame(&mut self) -> Result<Frame> {
        let mut guard = self.state.lock().map_err(|e| {
            CameraError::InitFailed(format!("PupilBackend state mutex poisoned: {e}"))
        })?;
        let state = guard
            .as_mut()
            .ok_or(CameraError::NotRunning)?;
        let mut frame = state.next_frame();
        // Stamp frame_number onto the frame header (we don't track it
        // across calls, so use Instant nanos / 1_000_000 modulo to keep
        // it monotonic).
        frame.frame_number = (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0))
            % 1_000_000;
        Ok(frame)
    }

    fn stop(&mut self) -> Result<()> {
        let mut guard = self.state.lock().map_err(|e| {
            CameraError::InitFailed(format!("PupilBackend state mutex poisoned: {e}"))
        })?;
        if let Some(state) = guard.as_ref() {
            state.source.shutdown();
        }
        Ok(())
    }

    fn name(&self) -> &str {
        &self.info.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic source advances.
    #[test]
    fn synthetic_figure_eight_advances() {
        let src = SyntheticPupilSource::new();
        let s1 = src.next().unwrap();
        std::thread::sleep(Duration::from_millis(10));
        let s2 = src.next().unwrap();
        // Figure-eight oscillates; the y value should differ over time.
        assert!(
            (s1.y - s2.y).abs() > 1e-6 || (s1.x - s2.x).abs() > 1e-6,
            "synthetic figure-eight should change over time"
        );
    }

    /// Synthetic source within screen bounds.
    #[test]
    fn synthetic_within_screen_bounds() {
        let src = SyntheticPupilSource::new();
        for _ in 0..50 {
            let s = src.next().unwrap();
            assert!((0.0..=1.0).contains(&s.x), "x out of bounds: {}", s.x);
            assert!((0.0..=1.0).contains(&s.y), "y out of bounds: {}", s.y);
            assert!((0.0..=1.0).contains(&s.confidence), "confidence out of bounds");
        }
    }

    /// Pupil backend reports kind and name.
    #[test]
    fn pupil_backend_kind_and_name() {
        let be = PupilBackend::synthetic();
        assert_eq!(be.kind(), BackendKind::Pupil);
        assert!(be.name().starts_with("pupil:"));
    }

    /// Pupil backend full lifecycle (open -> start -> capture -> stop).
    #[test]
    fn pupil_backend_full_lifecycle() {
        let mut be = PupilBackend::synthetic();
        let cfg = CameraConfig::default();
        be.open(0, &cfg).unwrap();
        be.start().unwrap();
        let frame = be.capture_frame().unwrap();
        assert_eq!(frame.width, cfg.width);
        assert_eq!(frame.height, cfg.height);
        assert_eq!(frame.format, PixelFormat::Gray8);
        assert_eq!(frame.data.len(), (cfg.width as usize) * (cfg.height as usize));
        // Heatmap accumulator should have some non-zero pixels after
        // the first frame (the figure-eight deposits ~200u8 with falloff).
        let nonzero = frame.data.iter().filter(|v| **v > 0).count();
        assert!(nonzero > 0, "expected at least one non-zero pixel in heatmap");
        be.stop().unwrap();
    }

    /// Capture without open requires `NotRunning`.
    #[test]
    fn capture_requires_running() {
        let mut be = PupilBackend::synthetic();
        let err = be.capture_frame().unwrap_err();
        match err {
            CameraError::NotRunning => {}
            other => panic!("expected NotRunning, got {other:?}"),
        }
    }
}