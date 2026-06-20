//! Calibration drift monitoring and recalibration trigger (FR-EYE-CAL-004)
//!
//! Tracks runtime gaze accuracy drift relative to the baseline calibration.
//! When drift exceeds a configurable threshold (>2° from baseline), a
//! recalibration event is emitted. The user may dismiss once per session.

use crate::multi_monitor::DisplayId;
use crate::smoothing::GazeSmoother;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Drift severity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DriftSeverity {
    /// Drift within tolerance; no action required.
    None,
    /// Drift elevated; log only.
    Warning,
    /// Drift exceeded threshold; trigger recalibration.
    Critical,
}

/// Recalibration event emitted by the monitor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecalibrationEvent {
    /// Display that triggered the event
    pub display: DisplayId,
    /// Severity level
    pub severity: DriftSeverity,
    /// Estimated angular drift in degrees
    pub drift_degrees: f32,
    /// Baseline drift at calibration time
    pub baseline_drift_degrees: f32,
    /// Reason for the event
    pub reason: String,
}

/// Configuration for the drift monitor
#[derive(Debug, Clone)]
pub struct DriftMonitorConfig {
    /// Drift threshold (in degrees) for critical events (default 2.0)
    pub critical_threshold_deg: f32,
    /// Drift threshold (in degrees) for warning events (default 1.0)
    pub warning_threshold_deg: f32,
    /// Window for averaging samples
    pub window: Duration,
    /// Minimum samples before a drift event is emitted
    pub min_samples: usize,
}

impl Default for DriftMonitorConfig {
    fn default() -> Self {
        Self {
            critical_threshold_deg: 2.0,
            warning_threshold_deg: 1.0,
            window: Duration::from_secs(5),
            min_samples: 30,
        }
    }
}

/// Calibration drift monitor
///
/// Watches a stream of gaze samples and compares them to a baseline
/// calibration signature. Emits `RecalibrationEvent` when drift exceeds
/// the configured thresholds.
pub struct DriftMonitor {
    config: DriftMonitorConfig,
    /// Per-display baseline signatures (centroid in screen coords + drift)
    baselines: std::collections::HashMap<String, BaselineSignature>,
    /// Sliding window of recent samples (timestamp, x, y, confidence)
    samples: Vec<(Instant, f32, f32, f32)>,
    /// Whether the user has dismissed the recalibration dialog this session
    dismissed: bool,
    /// Last event emitted (for idempotency)
    last_event: Option<RecalibrationEvent>,
}

/// Calibration baseline signature
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BaselineSignature {
    /// Centroid x in normalized screen coords
    centroid_x: f32,
    /// Centroid y in normalized screen coords
    centroid_y: f32,
    /// Drift at calibration time (degrees)
    baseline_drift: f32,
    /// Display identifier
    display: DisplayId,
}

impl DriftMonitor {
    /// Create a new drift monitor
    pub fn new(config: DriftMonitorConfig) -> Self {
        Self {
            config,
            baselines: std::collections::HashMap::new(),
            samples: Vec::new(),
            dismissed: false,
            last_event: None,
        }
    }

    /// Register a calibration baseline for a display
    pub fn register_baseline(&mut self, display: DisplayId, centroid_x: f32, centroid_y: f32, baseline_drift: f32) {
        self.baselines.insert(
            display.uuid.clone(),
            BaselineSignature {
                centroid_x,
                centroid_y,
                baseline_drift,
                display,
            },
        );
    }

    /// Clear baseline for a display
    pub fn clear_baseline(&mut self, display_uuid: &str) {
        self.baselines.remove(display_uuid);
    }

    /// Reset the dismissed flag (called at session start)
    pub fn reset_dismissed(&mut self) {
        self.dismissed = false;
    }

    /// Dismiss the recalibration dialog for this session
    pub fn dismiss(&mut self) {
        self.dismissed = true;
    }

    /// Check if the user has dismissed the recalibration dialog
    pub fn is_dismissed(&self) -> bool {
        self.dismissed
    }

    /// Record a gaze sample and return any recalibration event triggered
    pub fn record_sample(&mut self, x: f32, y: f32, confidence: f32) -> Option<RecalibrationEvent> {
        let now = Instant::now();
        self.samples.push((now, x, y, confidence));

        // Trim samples outside the window
        let cutoff = now - self.config.window;
        self.samples.retain(|(t, _, _, _)| *t >= cutoff);

        if self.samples.len() < self.config.min_samples {
            return None;
        }

        // Compute current centroid
        let n = self.samples.len() as f32;
        let cx: f32 = self.samples.iter().map(|s| s.1).sum::<f32>() / n;
        let cy: f32 = self.samples.iter().map(|s| s.2).sum::<f32>() / n;

        // Compare against each baseline; emit the most severe event
        let mut worst_event: Option<RecalibrationEvent> = None;
        for baseline in self.baselines.values() {
            // Approximate angular drift using normalized screen coords.
            // For a typical screen at 600mm viewing distance, normalized 1.0
            // maps to roughly 30° of gaze angle; treat as linear approximation.
            let dx = cx - baseline.centroid_x;
            let dy = cy - baseline.centroid_y;
            let dist = (dx * dx + dy * dy).sqrt();
            let drift_deg = dist * 30.0; // approximation
            let total_drift = baseline.baseline_drift + drift_deg;

            let severity = if total_drift >= self.config.critical_threshold_deg {
                DriftSeverity::Critical
            } else if total_drift >= self.config.warning_threshold_deg {
                DriftSeverity::Warning
            } else {
                DriftSeverity::None
            };

            if severity == DriftSeverity::None {
                continue;
            }

            let event = RecalibrationEvent {
                display: baseline.display.clone(),
                severity,
                drift_degrees: total_drift,
                baseline_drift_degrees: baseline.baseline_drift,
                reason: format!(
                    "Drift {total_drift:.2}° exceeds {severity:?} threshold"
                ),
            };

            // Promote to most severe (Critical > Warning > None)
            let should_replace = match (&worst_event, severity) {
                (None, _) => true,
                (Some(e), DriftSeverity::Critical) if e.severity != DriftSeverity::Critical => true,
                _ => false,
            };
            if should_replace {
                worst_event = Some(event);
            }
        }

        // Suppress if user already dismissed this session
        if self.dismissed {
            worst_event = None;
        }

        // Idempotency: don't re-emit the same event within the window
        if let (Some(last), Some(new)) = (&self.last_event, &worst_event) {
            if last.severity == new.severity
                && (last.drift_degrees - new.drift_degrees).abs() < 0.1
            {
                return None;
            }
        }
        self.last_event = worst_event.clone();
        worst_event
    }

    /// Get the smoother used for noise reduction (helper reference)
    pub fn smoother_reference(&self) -> GazeSmoother {
        GazeSmoother::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multi_monitor::DisplayId;

    fn display(uuid: &str) -> DisplayId {
        DisplayId::synthetic(uuid)
    }

    #[test]
    fn test_no_baseline_no_event() {
        let mut monitor = DriftMonitor::new(DriftMonitorConfig::default());
        for _ in 0..50 {
            let event = monitor.record_sample(0.5, 0.5, 0.9);
            assert!(event.is_none());
        }
    }

    #[test]
    fn test_no_drift_no_event() {
        let mut monitor = DriftMonitor::new(DriftMonitorConfig::default());
        monitor.register_baseline(display("test"), 0.5, 0.5, 0.5);
        for _ in 0..50 {
            let event = monitor.record_sample(0.5, 0.5, 0.9);
            assert!(event.is_none(), "unexpected event: {event:?}");
        }
    }

    #[test]
    fn test_high_drift_triggers_critical() {
        let mut monitor = DriftMonitor::new(DriftMonitorConfig::default());
        monitor.register_baseline(display("test"), 0.1, 0.1, 0.0);

        // Feed 50 samples centered far from baseline (0.5, 0.5)
        // Distance ≈ sqrt(0.4^2 + 0.4^2) ≈ 0.566 → drift ≈ 17°  (well above 2°)
        let mut event_emitted = false;
        for i in 0..60 {
            let x = 0.5 + (i as f32 * 0.001).sin();
            let y = 0.5;
            if let Some(e) = monitor.record_sample(x, y, 0.9) {
                assert_eq!(e.severity, DriftSeverity::Critical);
                assert!(e.drift_degrees > 2.0);
                event_emitted = true;
                break;
            }
        }
        assert!(event_emitted, "expected a Critical drift event");
    }

    #[test]
    fn test_moderate_drift_triggers_warning() {
        let config = DriftMonitorConfig {
            warning_threshold_deg: 1.0,
            critical_threshold_deg: 5.0,
            min_samples: 30,
            ..Default::default()
        };
        let mut monitor = DriftMonitor::new(config);
        monitor.register_baseline(display("test"), 0.5, 0.5, 0.0);

        // Distance ≈ 0.05 → drift ≈ 1.5° (warning but not critical)
        let mut event_emitted = false;
        for _ in 0..60 {
            let x = 0.55 + (fastrand::f32() * 0.005);
            let y = 0.5;
            if let Some(e) = monitor.record_sample(x, y, 0.9) {
                assert_eq!(e.severity, DriftSeverity::Warning);
                event_emitted = true;
                break;
            }
        }
        assert!(event_emitted, "expected a Warning drift event");
    }

    #[test]
    fn test_dismiss_suppresses_events() {
        let mut monitor = DriftMonitor::new(DriftMonitorConfig::default());
        monitor.register_baseline(display("test"), 0.1, 0.1, 0.0);
        monitor.dismiss();

        for _ in 0..60 {
            let event = monitor.record_sample(0.5, 0.5, 0.9);
            assert!(event.is_none(), "event after dismiss: {event:?}");
        }
    }

    #[test]
    fn test_reset_dismissed_clears_flag() {
        let mut monitor = DriftMonitor::new(DriftMonitorConfig::default());
        monitor.dismiss();
        assert!(monitor.is_dismissed());
        monitor.reset_dismissed();
        assert!(!monitor.is_dismissed());
    }

    #[test]
    fn test_window_eviction() {
        let mut monitor = DriftMonitor::new(DriftMonitorConfig {
            window: Duration::from_millis(50),
            min_samples: 5,
            ..Default::default()
        });
        monitor.register_baseline(display("test"), 0.5, 0.5, 0.0);

        for _ in 0..10 {
            monitor.record_sample(0.5, 0.5, 0.9);
        }
        std::thread::sleep(Duration::from_millis(100));
        // After window expires, samples are evicted; need new ones
        let event = monitor.record_sample(0.5, 0.5, 0.9);
        assert!(event.is_none());
    }
}

// Minimal local RNG to avoid adding a new dep just for tests
mod fastrand {
    use std::cell::Cell;
    use std::time::{SystemTime, UNIX_EPOCH};

    thread_local! {
        static STATE: Cell<u64> = Cell::new({
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(1)
        });
    }

    pub fn f32() -> f32 {
        STATE.with(|s| {
            let mut x = s.get();
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            s.set(x);
            (x as f32 / u64::MAX as f32).abs()
        })
    }
}
