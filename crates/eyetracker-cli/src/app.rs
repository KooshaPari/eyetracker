//! Main application logic for the eye tracker CLI
//!
//! Wires the full pipeline (camera → face detection → gaze → smoothing →
//! classification) with the drift monitor, multi-monitor calibration store,
//! and privacy manager. Surfaces all state to the TUI dashboard.

use anyhow::Result;
use eyetracker_inference::{
    accessibility::{AccessibilityAction, AccessibilityManager},
    classification::GazeEvent,
    drift_monitor::{DriftMonitor, DriftMonitorConfig, DriftSeverity, RecalibrationEvent},
    multi_monitor::{detect_active_display, MultiMonitorCalibration},
    privacy::{PrivacyManager, PrivacyMode},
    PipelineConfig, TrackingPipeline, TrackingResult,
};
use ratatui::Terminal;
use std::io::Stdout;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Instant;

use crate::mouse;
use crate::ui;

/// Shared state surfaced to the TUI
struct AppState {
    drift_monitor: DriftMonitor,
    monitor_store: MultiMonitorCalibration,
    privacy: PrivacyManager,
    active_display: eyetracker_inference::multi_monitor::DisplayId,
    /// Owns the dwell-click + scroll-by-gaze state machines. Updated on
    /// every frame via `tick_accessibility`; the resulting action is
    /// surfaced to the TUI via `last_accessibility_action`.
    accessibility: AccessibilityManager,
    /// Most recent recalibration event from the drift monitor. When Some
    /// and not dismissed, the TUI should surface a "Recalibrate?" prompt
    /// (FR-EYE-CAL-004).
    last_recalibration_event: Option<RecalibrationEvent>,
    /// Smoothed gaze observed at the moment the most recent drift event
    /// fired — used by the TUI to colour the drift panel.
    last_recalibration_position: Option<(f32, f32)>,
    /// Most recent accessibility action triggered (dwell-click / scroll).
    /// Single-frame signals like Click or DwellStarted linger here so the
    /// TUI can render a flash; the next frame clears or replaces it.
    last_accessibility_action: Option<eyetracker_inference::accessibility::AccessibilityAction>,
}

impl AppState {
    fn new(dwell_duration: std::time::Duration) -> Self {
        let active_display = detect_active_display()
            .unwrap_or_else(|_| eyetracker_inference::multi_monitor::DisplayId::synthetic("main"));
        let mut accessibility = AccessibilityManager::default();
        accessibility.dwell.set_dwell_duration(dwell_duration);
        let mut drift_monitor = DriftMonitor::new(DriftMonitorConfig::default());
        // Register the active display's baseline so the monitor has
        // something to compare against. The user will refine this in
        // --calibrate; for tracking-only mode we anchor at screen center
        // and let the monitor observe real drift.
        drift_monitor.register_baseline(active_display.clone(), 0.5, 0.5, 0.0);
        Self {
            drift_monitor,
            monitor_store: MultiMonitorCalibration::load().unwrap_or_default(),
            privacy: PrivacyManager::new(),
            active_display,
            accessibility,
            last_recalibration_event: None,
            last_recalibration_position: None,
            last_accessibility_action: None,
        }
    }

    /// Drift status string for the TUI panel. FR-EYE-CAL-004: this
    /// surfaces the most recent RecalibrationEvent (or "OK" if none).
    fn drift_status(&self) -> (String, String) {
        match &self.last_recalibration_event {
            Some(ev) if !self.drift_monitor.is_dismissed() => {
                let label = match ev.severity {
                    DriftSeverity::Critical => "RECALIBRATE",
                    DriftSeverity::Warning => "WARN",
                    DriftSeverity::None => "OK",
                };
                (label.to_string(), format!("{:.2}°", ev.drift_degrees))
            }
            _ => ("OK".to_string(), "—".to_string()),
        }
    }

    /// Whether the TUI should render the "Recalibrate?" prompt
    /// (FR-EYE-CAL-004).
    fn recalibration_pending(&self) -> bool {
        match &self.last_recalibration_event {
            Some(ev) => {
                !self.drift_monitor.is_dismissed() && matches!(ev.severity, DriftSeverity::Critical)
            }
            None => false,
        }
    }

    /// Format the active display label
    fn display_label(&self) -> String {
        let (w, h) = self.active_display.resolution;
        format!("{}  {}x{}", self.active_display.label, w, h)
    }

    /// Whether the active display has a calibration loaded
    fn display_calibrated(&self) -> bool {
        self.monitor_store
            .load_for(&self.active_display.uuid)
            .is_some()
    }

    /// Build the privacy banner text
    fn privacy_banner(&self) -> String {
        match self.privacy.mode {
            PrivacyMode::LocalOnly => "Local only\nNo data leaves device".to_string(),
            PrivacyMode::LocalWithExport => {
                let n = self.privacy.consent_count();
                format!(
                    "Local + export\n{} consent{}",
                    n,
                    if n == 1 { "" } else { "s" }
                )
            }
        }
    }

    /// Feed the latest gaze + fixation state into the accessibility
    /// detectors. Returns the action triggered this frame (or None).
    /// FR-EYE-ACCESS-001 (dwell-click) + FR-EYE-ACCESS-002 (scroll).
    fn tick_accessibility(
        &mut self,
        smoothed_gaze: Option<(f32, f32)>,
        is_fixating: bool,
        frame_w: f32,
        frame_h: f32,
    ) -> eyetracker_inference::accessibility::AccessibilityAction {
        // Convert pixel-space (relative to frame center) into normalized
        // [0,1] screen coordinates for the accessibility detectors.
        let Some((cx, cy)) = smoothed_gaze else {
            return eyetracker_inference::accessibility::AccessibilityAction::None;
        };
        let nx = (cx + frame_w / 2.0) / frame_w.max(1.0);
        let ny = (cy + frame_h / 2.0) / frame_h.max(1.0);
        let nx = nx.clamp(0.0, 1.0);
        let ny = ny.clamp(0.0, 1.0);

        // Dwell-click consumes the fixating flag.
        let dwell_action = self
            .accessibility
            .dwell
            .update(nx, ny, is_fixating, frame_w, frame_h);

        // Scroll-by-gaze is independent of fixation (top/bottom zones).
        let (scroll_action, _speed) = self.accessibility.scroll.update(ny);

        // Dwell state machine takes precedence over scroll while fixating.
        if !matches!(
            dwell_action,
            eyetracker_inference::accessibility::AccessibilityAction::None
        ) {
            return dwell_action;
        }
        scroll_action
    }

    /// Most recent accessibility action (for the TUI panel)
    #[allow(dead_code)]
    fn last_action(&self) -> Option<eyetracker_inference::accessibility::AccessibilityAction> {
        self.last_accessibility_action
    }
}

/// Run the interactive TUI mode
pub fn run_tui(
    terminal: &mut Terminal<ratatui::backend::CrosstermBackend<Stdout>>,
    config: &PipelineConfig,
    duration_secs: u64,
    dwell_duration: std::time::Duration,
    mouse_output_enabled: bool,
    screen_width: u32,
    screen_height: u32,
) -> Result<()> {
    let mut pipeline = TrackingPipeline::with_config(config.clone())?;
    pipeline.start()?;

    let (tx, rx) = mpsc::channel::<TrackingResult>();
    let state = std::sync::Arc::new(std::sync::Mutex::new(AppState::new(dwell_duration)));

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
    let state_clone = state.clone();
    let dismiss_flag = std::sync::Arc::new(AtomicBool::new(false));
    let dismiss_flag_closure = dismiss_flag.clone();
    let result = ui::run_event_loop(
        terminal,
        &rx,
        move |result: &TrackingResult| {
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
            let smoothed = result
                .smoothed_gaze
                .map(|(x, y)| format!("({:.1}, {:.1})", x, y));
            let events_summary = if result.events.is_empty() {
                String::new()
            } else {
                result
                    .events
                    .iter()
                    .map(|e| match e {
                        GazeEvent::FixationStart { .. } => "F+".to_string(),
                        GazeEvent::FixationEnd { .. } => "F-".to_string(),
                        GazeEvent::Saccade { .. } => "S".to_string(),
                    })
                    .collect::<Vec<_>>()
                    .join(",")
            };
            let confidence = result
                .gaze
                .as_ref()
                .map(|g| format!("{:.1}%", g.confidence * 100.0))
                .unwrap_or_else(|| "N/A".to_string());

            // Lock-free copy of state for the UI closure
            let (
                drift_status,
                drift_deg_str,
                display_label,
                display_calibrated,
                privacy_banner,
                recal_pending,
                last_action_dbg,
            ) = {
                if let Ok(mut s) = state_clone.lock() {
                    // FR-EYE-CAL-004: consume any pending dismiss request
                    if dismiss_flag_closure.swap(false, std::sync::atomic::Ordering::SeqCst) {
                        s.drift_monitor.dismiss();
                        s.last_recalibration_event = None;
                        s.last_recalibration_position = None;
                        tracing::info!("user dismissed recalibration event");
                    }
                    // Feed the actual smoothed gaze into the drift monitor
                    // so FR-EYE-CAL-004 auto-trigger fires from real data.
                    if let Some((x, y)) = result.smoothed_gaze {
                        if let Some(ev) = s.drift_monitor.record_sample(
                            x,
                            y,
                            result.gaze.as_ref().map(|g| g.confidence).unwrap_or(0.0),
                        ) {
                            // Capture the diagnostic fields before moving the
                            // event into AppState so the tracing macro can
                            // reference them by value.
                            let drift_deg = ev.drift_degrees;
                            let severity_dbg = format!("{:?}", ev.severity);
                            s.last_recalibration_event = Some(ev);
                            s.last_recalibration_position = Some((x, y));
                            tracing::warn!(
                                drift_degrees = drift_deg,
                                severity = %severity_dbg,
                                "drift monitor triggered recalibration event"
                            );
                        }
                    }
                    // FR-EYE-ACCESS-001/002: feed the live gaze + fixation
                    // state into the accessibility detectors.
                    let is_fixating = result
                        .events
                        .iter()
                        .any(|e| matches!(e, GazeEvent::FixationStart { .. }));
                    let action: AccessibilityAction = s.tick_accessibility(
                        result.smoothed_gaze,
                        is_fixating,
                        result.frame.width as f32,
                        result.frame.height as f32,
                    );
                    let action_dbg = format!("{:?}", action);
                    if !matches!(action, AccessibilityAction::None) {
                        tracing::debug!(action = %action_dbg, "accessibility action triggered");
                    }
                    // FR-EYE-ACCESS-001/002: dispatch the action to the OS so
                    // the user actually sees the click/scroll. On macOS this
                    // posts a CGEvent; on other platforms it's a no-op (still
                    // logged via tracing::debug).
                    if let Some((nx, ny)) = result.smoothed_gaze {
                        let frame_w = result.frame.width.max(1) as f32;
                        let frame_h = result.frame.height.max(1) as f32;
                        let nx_norm = (nx + frame_w / 2.0) / frame_w;
                        let ny_norm = (ny + frame_h / 2.0) / frame_h;
                        let (sx, sy) = mouse::normalized_to_screen(
                            nx_norm.clamp(0.0, 1.0),
                            ny_norm.clamp(0.0, 1.0),
                            screen_width,
                            screen_height,
                        );
                        mouse::dispatch(action, sx, sy, mouse_output_enabled);
                    }
                    s.last_accessibility_action = Some(action);
                    let (status, deg) = s.drift_status();
                    let label = s.display_label();
                    let cal = s.display_calibrated();
                    let banner = s.privacy_banner();
                    let pending = s.recalibration_pending();
                    (status, deg, label, cal, banner, pending, action_dbg)
                } else {
                    (
                        "-".to_string(),
                        "0.00".to_string(),
                        "-".to_string(),
                        false,
                        "Local only".to_string(),
                        false,
                        "None".to_string(),
                    )
                }
            };
            let _ = DriftSeverity::None; // keep import used
            let _ = Ordering::SeqCst; // keep import used

            ui::DashboardData {
                fps,
                processing_ms: result.processing_time_ms,
                frame_number: result.frame.frame_number,
                gaze_vector: gaze,
                smoothed_gaze: smoothed,
                confidence,
                face_detected: result.face.is_some(),
                resolution: format!("{}x{}", result.frame.width, result.frame.height),
                events: events_summary,
                drift_status,
                drift_degrees: drift_deg_str,
                display_label,
                display_calibrated,
                privacy_banner,
                recalibration_pending: recal_pending,
                last_accessibility_action: last_action_dbg,
            }
        },
        duration_secs,
        dismiss_flag,
    );

    let _ = pipeline.stop();
    let _ = processing_handle.join();

    result
}

/// Run CSV dump mode (no TUI, just output CSV data)
pub fn run_csv_dump(
    config: &PipelineConfig,
    duration_secs: u64,
    dwell_duration: std::time::Duration,
    mouse_output_enabled: bool,
    screen_width: u32,
    screen_height: u32,
) -> Result<()> {
    let mut pipeline = TrackingPipeline::with_config(config.clone())?;
    pipeline.start()?;

    // CSV header (extra columns for the accessibility action + dispatch target)
    println!(
        "timestamp_ms,frame,processing_ms,gaze_x,gaze_y,gaze_z,confidence,face_detected,accessibility_action,screen_x,screen_y"
    );

    let state = std::sync::Arc::new(std::sync::Mutex::new(AppState::new(dwell_duration)));
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

                // Tick accessibility on each frame; dispatch to OS (or no-op).
                let (action_dbg, sx, sy) = if let Ok(mut s) = state.lock() {
                    let is_fixating = result
                        .events
                        .iter()
                        .any(|e| matches!(e, GazeEvent::FixationStart { .. }));
                    let action: AccessibilityAction = s.tick_accessibility(
                        result.smoothed_gaze,
                        is_fixating,
                        result.frame.width as f32,
                        result.frame.height as f32,
                    );
                    let mut sx_f = 0.0;
                    let mut sy_f = 0.0;
                    if let Some((nx, ny)) = result.smoothed_gaze {
                        let frame_w = result.frame.width.max(1) as f32;
                        let frame_h = result.frame.height.max(1) as f32;
                        let nx_norm = (nx + frame_w / 2.0) / frame_w;
                        let ny_norm = (ny + frame_h / 2.0) / frame_h;
                        let (px, py) = mouse::normalized_to_screen(
                            nx_norm.clamp(0.0, 1.0),
                            ny_norm.clamp(0.0, 1.0),
                            screen_width,
                            screen_height,
                        );
                        sx_f = px;
                        sy_f = py;
                        mouse::dispatch(action, px, py, mouse_output_enabled);
                    }
                    (format!("{:?}", action), sx_f, sy_f)
                } else {
                    ("None".to_string(), 0.0, 0.0)
                };

                println!(
                    "{:.1},{},{:.2},{:.4},{:.4},{:.4},{:.4},{},{},{:.1},{:.1}",
                    timestamp,
                    result.frame.frame_number,
                    result.processing_time_ms,
                    gx,
                    gy,
                    gz,
                    conf,
                    result.face.is_some(),
                    action_dbg,
                    sx,
                    sy,
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
