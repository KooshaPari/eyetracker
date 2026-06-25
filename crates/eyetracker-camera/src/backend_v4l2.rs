//! Linux Video4Linux2 (V4L2) camera backend.
//!
//! Linux cross-platform capture path for eyetracker-camera. Complements the
//! nokhwa-based [`Camera`](crate::Camera) which routes through the macOS/Windows
//! capture APIs. On Linux, `nokhwa::CameraFormat::new(...)` falls back to V4L2
//! anyway; this module exists to:
//!
//! 1. Expose Linux-specific knobs (`v4l2-ctl`-style controls) that the
//!    cross-platform Camera doesn't surface (gain, exposure, sharpness,
//!    backlight compensation).
//! 2. Provide a stable Linux API surface even when nokhwa's Linux backend
//!    changes (e.g. v4l2-rs vs libcamera switchover).
//! 3. Document the V4L2 contract for future Rust FFI work (see ADR-016).
//!
//! ## Build / feature
//!
//! Gated behind `#[cfg(feature = "v4l2")]`. Default build still uses nokhwa
//! for cross-platform tests. To enable on Linux:
//!
//! ```sh
//! cargo build --features v4l2
//! ```
//!
//! ## Contract
//!
//! Real FFI bindings (planned: `v4l2` crate or hand-rolled libc::ioctl calls)
//! will replace the placeholder constructor below. The public API is shaped
//! to survive that swap.

#![cfg(feature = "v4l2")]

use crate::{CameraConfig, CameraError, Frame, PixelFormat};
use anyhow::Result;
use std::time::Instant;

/// Linux Video4Linux2 (V4L2) capture backend.
///
/// Wraps a `/dev/video<N>` device and provides eye-tracker friendly defaults.
/// When real FFI bindings land, replace [`V4l2Camera::open`] with the FFI
/// init path; the rest of the public API stays stable.
pub struct V4l2Camera {
    /// Index of the underlying `/dev/video*` device.
    device_index: u32,
    /// Stored configuration (width, height, fps, ...).
    config: CameraConfig,
    /// Whether the camera is currently streaming.
    running: bool,
    /// Total frames captured since `start()`.
    frame_count: u64,
    /// Wall-clock instant of `start()`, used to compute `elapsed()`.
    start_time: Option<Instant>,
}

impl V4l2Camera {
    /// Open `/dev/video<device_index>` with the given configuration.
    ///
    /// Returns `CameraError::InitFailed` with a follow-up message until real
    /// FFI bindings are linked. The struct is otherwise constructed and
    /// usable for [`Self::kind`], [`Self::name`], [`Self::start`] / `stop`,
    /// and unit-testing the heatmap accumulator path.
    pub fn open(device_index: u32, config: CameraConfig) -> Result<Self> {
        if config.width == 0 || config.height == 0 {
            return Err(CameraError::InitFailed(
                "V4L2: width and height must be > 0".into(),
            ));
        }
        if config.target_fps == 0 {
            return Err(CameraError::InitFailed(
                "V4L2: target_fps must be > 0".into(),
            ));
        }
        // Real FFI path is not yet wired; record the intent so callers can
        // observe the deferred binding via the test suite.
        tracing::info!(
            "V4l2Camera::open: device=/dev/video{} res={}x{} fps={} (FFI pending)",
            device_index,
            config.width,
            config.height,
            config.target_fps,
        );

        Ok(Self {
            device_index,
            config,
            running: false,
            frame_count: 0,
            start_time: None,
        })
    }

    /// Short, stable identifier for logs / metrics ("v4l2").
    pub fn kind(&self) -> &'static str {
        "v4l2"
    }

    /// Human-readable backend label including the device index.
    pub fn name(&self) -> String {
        format!("V4L2(/dev/video{})", self.device_index)
    }

    /// Begin streaming. Idempotent.
    pub fn start(&mut self) -> Result<()> {
        if self.running {
            return Ok(());
        }
        // FFI: VIDIOC_STREAMON
        self.running = true;
        self.start_time = Some(Instant::now());
        tracing::info!("V4l2Camera::start: device=/dev/video{}", self.device_index);
        Ok(())
    }

    /// Capture one frame. Produces a deterministic gradient + center marker
    /// until real FFI is wired. Real implementation will mmap a v4l2 buffer,
    /// queue it, and copy into the [`Frame::data`] buffer.
    pub fn capture_frame(&mut self) -> Result<Frame> {
        if !self.running {
            return Err(CameraError::NotRunning);
        }
        let w = self.config.width as usize;
        let h = self.config.height as usize;
        let len = w.checked_mul(h).ok_or_else(|| {
            CameraError::InitFailed(format!(
                "V4L2: width * height overflow ({} * {})",
                w, h
            ))
        })?;
        let mut data = vec![0u8; len];

        // Deterministic synthetic pattern for tests:
        // - row 0 = 0, row h-1 = 255 (vertical gradient)
        // - center 16x16 = 200 (high-contrast marker)
        let center_y = h / 2;
        let center_x = w / 2;
        for y in 0..h {
            let row = if h > 1 {
                (y * 255 / (h - 1)) as u8
            } else {
                128
            };
            for x in 0..w {
                let v = if y.abs_diff(center_y) < 8 && x.abs_diff(center_x) < 8 {
                    200u8
                } else {
                    row
                };
                data[y * w + x] = v;
            }
        }

        self.frame_count = self.frame_count.saturating_add(1);
        let timestamp_us = self
            .start_time
            .map(|t| t.elapsed().as_micros() as u64)
            .unwrap_or(0);

        Ok(Frame {
            data,
            width: self.config.width,
            height: self.config.height,
            format: PixelFormat::Gray8,
            timestamp_us,
            frame_number: self.frame_count,
        })
    }

    /// Stop streaming. Idempotent.
    pub fn stop(&mut self) -> Result<()> {
        if !self.running {
            return Ok(());
        }
        // FFI: VIDIOC_STREAMOFF
        self.running = false;
        tracing::info!(
            "V4l2Camera::stop: device=/dev/video{} frames={}",
            self.device_index,
            self.frame_count
        );
        Ok(())
    }

    /// Total frames captured since `start()`.
    pub fn frames_captured(&self) -> u64 {
        self.frame_count
    }

    /// Wall-clock elapsed time since `start()`, or `None` if not started.
    pub fn elapsed(&self) -> Option<std::time::Duration> {
        self.start_time.map(|t| t.elapsed())
    }

    /// Stored configuration (immutable snapshot).
    pub fn config(&self) -> &CameraConfig {
        &self.config
    }

    /// `/dev/video<N>` index this camera was opened against.
    pub fn device_index(&self) -> u32 {
        self.device_index
    }
}

impl Drop for V4l2Camera {
    fn drop(&mut self) {
        if self.running {
            // Best-effort stop; ignore errors during drop.
            let _ = self.stop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> CameraConfig {
        CameraConfig::default()
    }

    #[test]
    fn v4l2_open_succeeds_with_default_config() {
        let cam = V4l2Camera::open(0, cfg()).expect("open default");
        assert_eq!(cam.kind(), "v4l2");
        assert_eq!(cam.device_index(), 0);
        assert_eq!(cam.frames_captured(), 0);
        assert!(!cam.running);
    }

    #[test]
    fn v4l2_name_includes_device_index() {
        let cam = V4l2Camera::open(2, cfg()).expect("open idx=2");
        assert!(cam.name().contains("/dev/video2"));
    }

    #[test]
    fn v4l2_open_rejects_zero_width() {
        let mut c = cfg();
        c.width = 0;
        let err = V4l2Camera::open(0, c).unwrap_err();
        assert!(matches!(err, CameraError::InitFailed(_)));
    }

    #[test]
    fn v4l2_open_rejects_zero_fps() {
        let mut c = cfg();
        c.target_fps = 0;
        let err = V4l2Camera::open(0, c).unwrap_err();
        assert!(matches!(err, CameraError::InitFailed(_)));
    }

    #[test]
    fn v4l2_capture_requires_running() {
        let mut cam = V4l2Camera::open(0, cfg()).expect("open");
        let err = cam.capture_frame().unwrap_err();
        assert!(matches!(err, CameraError::NotRunning));
    }

    #[test]
    fn v4l2_full_lifecycle() {
        let mut cam = V4l2Camera::open(0, cfg()).expect("open");
        cam.start().expect("start");
        assert!(cam.elapsed().is_some());
        let f1 = cam.capture_frame().expect("frame 1");
        let f2 = cam.capture_frame().expect("frame 2");
        assert_eq!(f1.frame_number, 1);
        assert_eq!(f2.frame_number, 2);
        assert!(f2.timestamp_us >= f1.timestamp_us);
        assert_eq!(f1.data.len(), 640 * 480);
        assert_eq!(f1.format, PixelFormat::Gray8);
        assert_eq!(cam.frames_captured(), 2);
        cam.stop().expect("stop");
        // Stop is idempotent
        cam.stop().expect("stop again");
    }

    #[test]
    fn v4l2_start_idempotent() {
        let mut cam = V4l2Camera::open(0, cfg()).expect("open");
        cam.start().expect("start 1");
        cam.start().expect("start 2 (no-op)");
        cam.stop().expect("stop");
    }

    #[test]
    fn v4l2_center_marker_present() {
        let mut cam = V4l2Camera::open(0, cfg()).expect("open");
        cam.start().expect("start");
        let frame = cam.capture_frame().expect("frame");
        // Center pixel (320, 240) on 640x480 should be the marker (200).
        let center = frame.data[240 * 640 + 320];
        assert_eq!(center, 200);
    }

    #[test]
    fn v4l2_high_res_config() {
        let cam = V4l2Camera::open(0, CameraConfig::high_res()).expect("open");
        assert_eq!(cam.config().width, 1280);
        assert_eq!(cam.config().height, 720);
    }

    #[test]
    fn v4l2_low_res_config() {
        let cam = V4l2Camera::open(0, CameraConfig::low_res()).expect("open");
        assert_eq!(cam.config().width, 320);
        assert_eq!(cam.config().height, 240);
        assert_eq!(cam.config().target_fps, 120);
    }

    #[test]
    fn v4l2_eye_tracking_config() {
        let cam = V4l2Camera::open(0, CameraConfig::eye_tracking()).expect("open");
        assert!(cam.config().low_light_mode);
        assert_eq!(cam.config().target_fps, 60);
    }

    #[test]
    fn v4l2_drop_stops_streaming() {
        {
            let mut cam = V4l2Camera::open(0, cfg()).expect("open");
            cam.start().expect("start");
        } // dropped here
        // No panic = drop ran cleanly.
    }
}
