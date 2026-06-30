//! Accessibility features (FR-EYE-ACCESS-001, FR-EYE-ACCESS-002)
//!
//! - FR-EYE-ACCESS-001: Dwell-click selection. A fixation lasting at least
//!   the configured dwell time (200-1000ms, default 500ms) on a stable
//!   screen region triggers a click event. Cancellable via saccade to a
//!   safe zone (the screen edges).
//!
//! - FR-EYE-ACCESS-002: Scroll-by-gaze. Fixation in the upper 20% of the
//!   screen scrolls up; lower 20% scrolls down. Speed is proportional to
//!   the distance from the screen center (0% at 50% from center, max at 0/100%).

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Action triggered by accessibility logic
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccessibilityAction {
    /// No action
    None,
    /// Trigger mouse click
    Click,
    /// Scroll up (positive direction)
    ScrollUp,
    /// Scroll down (negative direction)
    ScrollDown,
    /// Dwell timer started
    DwellStarted,
    /// Dwell timer cancelled (user saccaded away)
    DwellCancelled,
}

/// Dwell-click configuration (FR-EYE-ACCESS-001)
#[derive(Debug, Clone)]
pub struct DwellClickConfig {
    /// Required fixation duration (200-1000ms; default 500ms)
    pub dwell_duration: Duration,
    /// Movement tolerance — saccade of this magnitude cancels the dwell
    pub movement_tolerance: f32,
    /// Safe zone thickness (pixels from edge; saccading into this zone cancels)
    pub safe_zone_thickness: f32,
}

impl Default for DwellClickConfig {
    fn default() -> Self {
        Self {
            dwell_duration: Duration::from_millis(500),
            movement_tolerance: 0.02,
            safe_zone_thickness: 0.05,
        }
    }
}

/// Dwell-click state machine
pub struct DwellClickDetector {
    config: DwellClickConfig,
    /// Last position at which dwell started
    dwell_start_pos: Option<(f32, f32)>,
    /// When the current dwell started
    dwell_start_time: Option<Instant>,
    /// True if a click has just fired (single-frame signal)
    click_pending: bool,
}

impl DwellClickDetector {
    pub fn new(config: DwellClickConfig) -> Self {
        Self {
            config,
            dwell_start_pos: None,
            dwell_start_time: None,
            click_pending: false,
        }
    }

    /// Feed a new gaze sample and the current fixation state.
    /// Returns the action triggered this frame.
    pub fn update(
        &mut self,
        x: f32,
        y: f32,
        is_fixating: bool,
        screen_w: f32,
        screen_h: f32,
    ) -> AccessibilityAction {
        // Clear single-frame click signal
        if self.click_pending {
            self.click_pending = false;
        }

        // In safe zone? Cancel any pending dwell.
        if x < self.config.safe_zone_thickness
            || x > 1.0 - self.config.safe_zone_thickness
            || y < self.config.safe_zone_thickness
            || y > 1.0 - self.config.safe_zone_thickness
        {
            if self.dwell_start_pos.is_some() {
                self.dwell_start_pos = None;
                self.dwell_start_time = None;
                return AccessibilityAction::DwellCancelled;
            }
            return AccessibilityAction::None;
        }

        if !is_fixating {
            // Not fixating — no dwell possible
            if self.dwell_start_pos.is_some() {
                self.dwell_start_pos = None;
                self.dwell_start_time = None;
                return AccessibilityAction::DwellCancelled;
            }
            return AccessibilityAction::None;
        }

        match (self.dwell_start_pos, self.dwell_start_time) {
            (Some((sx, sy)), Some(start)) => {
                let dx = (x - sx).abs();
                let dy = (y - sy).abs();
                if dx > self.config.movement_tolerance || dy > self.config.movement_tolerance {
                    // Saccade away — cancel
                    self.dwell_start_pos = Some((x, y));
                    self.dwell_start_time = Some(Instant::now());
                    return AccessibilityAction::DwellCancelled;
                }
                if start.elapsed() >= self.config.dwell_duration {
                    // Click! Reset.
                    self.dwell_start_pos = None;
                    self.dwell_start_time = None;
                    self.click_pending = true;
                    let _ = (screen_w, screen_h);
                    return AccessibilityAction::Click;
                }
                AccessibilityAction::None
            }
            _ => {
                // Start a new dwell
                self.dwell_start_pos = Some((x, y));
                self.dwell_start_time = Some(Instant::now());
                AccessibilityAction::DwellStarted
            }
        }
    }

    /// Reset state (e.g., when calibration changes)
    pub fn reset(&mut self) {
        self.dwell_start_pos = None;
        self.dwell_start_time = None;
        self.click_pending = false;
    }

    /// True if a click was triggered on the most recent update
    pub fn click_pending(&self) -> bool {
        self.click_pending
    }

    /// Configure the dwell duration. FR-EYE-ACCESS-001 requires
    /// 200-1000ms; values outside that range are clamped to the
    /// nearest boundary.
    pub fn set_dwell_duration(&mut self, duration: Duration) {
        const MIN: Duration = Duration::from_millis(200);
        const MAX: Duration = Duration::from_millis(1000);
        self.config.dwell_duration = if duration < MIN {
            MIN
        } else if duration > MAX {
            MAX
        } else {
            duration
        };
    }

    /// Current configured dwell duration (FR-EYE-ACCESS-001: 200-1000ms).
    pub fn dwell_duration(&self) -> Duration {
        self.config.dwell_duration
    }
}

/// Scroll-by-gaze configuration (FR-EYE-ACCESS-002)
#[derive(Debug, Clone)]
pub struct ScrollConfig {
    /// Top fraction of screen that triggers scroll-up (default 0.20)
    pub top_zone: f32,
    /// Bottom fraction of screen that triggers scroll-down (default 0.20)
    pub bottom_zone: f32,
    /// Maximum scroll speed in lines/second
    pub max_speed: f32,
}

impl Default for ScrollConfig {
    fn default() -> Self {
        Self {
            top_zone: 0.20,
            bottom_zone: 0.20,
            max_speed: 10.0,
        }
    }
}

/// Scroll state machine
pub struct ScrollDetector {
    config: ScrollConfig,
}

impl ScrollDetector {
    pub fn new(config: ScrollConfig) -> Self {
        Self { config }
    }

    /// Compute the scroll action for the current gaze position.
    /// `y` is normalized 0.0 (top) - 1.0 (bottom).
    /// Returns (action, speed_lines_per_second). Speed is 0 when no action.
    pub fn update(&self, y: f32) -> (AccessibilityAction, f32) {
        if y < self.config.top_zone {
            // Distance from the top edge; closer to top = faster
            let proximity = 1.0 - (y / self.config.top_zone);
            let speed = self.config.max_speed * proximity;
            return (AccessibilityAction::ScrollUp, speed);
        }
        if y > 1.0 - self.config.bottom_zone {
            let proximity = (y - (1.0 - self.config.bottom_zone)) / self.config.bottom_zone;
            let speed = self.config.max_speed * proximity;
            return (AccessibilityAction::ScrollDown, speed);
        }
        (AccessibilityAction::None, 0.0)
    }
}

/// Combined accessibility manager owning both detectors
pub struct AccessibilityManager {
    pub dwell: DwellClickDetector,
    pub scroll: ScrollDetector,
}

impl AccessibilityManager {
    pub fn new(dwell: DwellClickConfig, scroll: ScrollConfig) -> Self {
        Self {
            dwell: DwellClickDetector::new(dwell),
            scroll: ScrollDetector::new(scroll),
        }
    }
}

impl Default for AccessibilityManager {
    fn default() -> Self {
        Self::new(DwellClickConfig::default(), ScrollConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dwell_click_fires_after_dwell_duration() {
        let mut det = DwellClickDetector::new(DwellClickConfig {
            dwell_duration: Duration::from_millis(50),
            ..Default::default()
        });
        // Start dwell
        let a = det.update(0.5, 0.5, true, 1920.0, 1080.0);
        assert_eq!(a, AccessibilityAction::DwellStarted);
        // Wait for dwell to elapse
        std::thread::sleep(Duration::from_millis(60));
        let a = det.update(0.5, 0.5, true, 1920.0, 1080.0);
        assert_eq!(a, AccessibilityAction::Click);
    }

    #[test]
    fn test_dwell_click_cancels_on_saccade() {
        let mut det = DwellClickDetector::new(DwellClickConfig {
            dwell_duration: Duration::from_millis(100),
            movement_tolerance: 0.02,
            ..Default::default()
        });
        det.update(0.5, 0.5, true, 1920.0, 1080.0);
        // Saccade 0.1 away → cancels
        let a = det.update(0.6, 0.5, true, 1920.0, 1080.0);
        assert_eq!(a, AccessibilityAction::DwellCancelled);
    }

    #[test]
    fn test_dwell_click_cancels_in_safe_zone() {
        let mut det = DwellClickDetector::new(DwellClickConfig::default());
        det.update(0.5, 0.5, true, 1920.0, 1080.0);
        // Move to edge (safe zone)
        let a = det.update(0.01, 0.5, true, 1920.0, 1080.0);
        assert_eq!(a, AccessibilityAction::DwellCancelled);
    }

    #[test]
    fn test_dwell_click_requires_fixation() {
        let mut det = DwellClickDetector::new(DwellClickConfig {
            dwell_duration: Duration::from_millis(10),
            ..Default::default()
        });
        // Not fixating — should not start dwell
        let a = det.update(0.5, 0.5, false, 1920.0, 1080.0);
        assert_eq!(a, AccessibilityAction::None);
    }

    #[test]
    fn test_scroll_up_at_top() {
        let det = ScrollDetector::new(ScrollConfig::default());
        let (action, speed) = det.update(0.0);
        assert_eq!(action, AccessibilityAction::ScrollUp);
        assert!(speed > 0.0, "speed should be positive at top, got {speed}");
    }

    #[test]
    fn test_scroll_down_at_bottom() {
        let det = ScrollDetector::new(ScrollConfig::default());
        let (action, speed) = det.update(1.0);
        assert_eq!(action, AccessibilityAction::ScrollDown);
        assert!(
            speed > 0.0,
            "speed should be positive at bottom, got {speed}"
        );
    }

    #[test]
    fn test_scroll_no_action_in_middle() {
        let det = ScrollDetector::new(ScrollConfig::default());
        let (action, speed) = det.update(0.5);
        assert_eq!(action, AccessibilityAction::None);
        assert_eq!(speed, 0.0);
    }

    #[test]
    fn test_scroll_speed_proportional_to_proximity() {
        let det = ScrollDetector::new(ScrollConfig::default());
        let (_, speed_near_top) = det.update(0.01);
        let (_, speed_far_from_top) = det.update(0.15);
        assert!(
            speed_near_top > speed_far_from_top,
            "speed near top {speed_near_top} should be > speed far from top {speed_far_from_top}"
        );
    }

    #[test]
    fn test_dwell_reset() {
        let mut det = DwellClickDetector::new(DwellClickConfig::default());
        det.update(0.5, 0.5, true, 1920.0, 1080.0);
        det.reset();
        assert!(!det.click_pending());
    }

    #[test]
    fn test_accessibility_manager_default() {
        let mgr = AccessibilityManager::default();
        assert!(!mgr.dwell.click_pending());
    }
}
