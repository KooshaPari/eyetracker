//! Mouse output via macOS `CGEvent` (Core Graphics)
//!
//! Translates high-level accessibility actions (`Click`, `ScrollUp`,
//! `ScrollDown`) into the corresponding Quartz mouse events and posts
//! them to the active display.
//!
//! On non-macOS platforms this module compiles to a no-op so the rest
//! of the code can still depend on the same API surface.

use eyetracker_inference::accessibility::AccessibilityAction;
use tracing::{debug, warn};

#[cfg(target_os = "macos")]
mod platform {
    use super::*;

    /// Send a click + release at the absolute display coordinates.
    pub fn click_at(x: f64, y: f64, button: MouseButton) {
        let source =
            CGEventSource::new(CGEventSourceStateID::HIDSystemState).expect("CGEventSource::new");
        let down = CGEvent::new_mouse_event(
            source.clone(),
            button.down_event(),
            CGPoint::new(x, y),
            CGMouseButton::Left,
        )
        .expect("down event");
        let up = CGEvent::new_mouse_event(
            source.clone(),
            button.up_event(),
            CGPoint::new(x, y),
            CGMouseButton::Left,
        )
        .expect("up event");
        down.post(CGEventTapLocation::HID);
        up.post(CGEventTapLocation::HID);
        debug!("click posted at ({}, {})", x, y);
    }

    /// Send a vertical scroll delta. `lines` is in "lines" units (positive = up).
    pub fn scroll_at(x: f64, y: f64, lines: i32) {
        // wheel_event is gated behind the `highsierra` feature on core-graphics 0.23;
        // fall back to a no-op if not available so the build doesn't break on older
        // macOS or stripped environments.
        // The conversion: 1 line ≈ 10 wheel units (Apple convention).
        let delta = (lines * 10) as i64;
        let source =
            CGEventSource::new(CGEventSourceStateID::HIDSystemState).expect("CGEventSource::new");
        // Build a scroll event manually via the C-style API since the Rust binding
        // for `CGEventCreateScrollWheelEvent2` is not exposed in 0.23 by default.
        // SAFETY: `source` is a valid CGEventSource obtained from
        // `CGEventSource::new(HIDSystemState).expect(…)` above.
        // `ScrollEventUnit::LINE` is a valid enum discriminant.
        // `delta as i32` is sound because `lines` is bounded to [-i32::MAX/10,
        // i32::MAX/10] at every call site (accessibility.rs clamps scroll lines
        // to ±100). See docs/SAFETY.md § mouse.rs for the full invariant register.
        unsafe {
            use core_graphics::event::{CGEvent, CGEventTapLocation, ScrollEventUnit};
            let event =
                CGEvent::new_scroll_event(source, ScrollEventUnit::LINE, 1, delta as i32, 0, 0);
            if let Some(ev) = event {
                ev.post(CGEventTapLocation::HID);
                debug!("scroll posted at ({}, {}), lines={}", x, y, lines);
            } else {
                warn!("CGEvent::new_scroll_event returned None (highsierra feature missing?)");
            }
        }
    }
}

#[cfg(not(target_os = "macos"))]
mod platform {
    use super::*;

    pub fn click_at(_x: f64, _y: f64, _button: MouseButton) {
        warn!("mouse output is only supported on macOS; click ignored");
    }

    pub fn scroll_at(_x: f64, _y: f64, _lines: i32) {
        warn!("mouse output is only supported on macOS; scroll ignored");
    }
}

/// Which physical mouse button to use for a click.
#[derive(Debug, Clone, Copy)]
pub enum MouseButton {
    Left,
    Right,
}

impl MouseButton {
    #[cfg(target_os = "macos")]
    fn down_event(self) -> core_graphics::event::CGEventType {
        use core_graphics::event::CGEventType;
        match self {
            MouseButton::Left => CGEventType::LeftMouseDown,
            MouseButton::Right => CGEventType::RightMouseDown,
        }
    }

    #[cfg(target_os = "macos")]
    fn up_event(self) -> core_graphics::event::CGEventType {
        use core_graphics::event::CGEventType;
        match self {
            MouseButton::Left => CGEventType::LeftMouseUp,
            MouseButton::Right => CGEventType::RightMouseUp,
        }
    }
}

/// Translate a single accessibility action into one or more mouse events
/// posted at the given absolute display coordinates.
///
/// `enabled` is a runtime kill-switch (set to false for `--no-mouse-output`,
/// tests, or headless environments).
pub fn dispatch(action: AccessibilityAction, screen_x: f64, screen_y: f64, enabled: bool) {
    if !enabled {
        debug!(
            "mouse output disabled — action {:?} at ({}, {}) would have fired",
            action, screen_x, screen_y
        );
        return;
    }
    match action {
        AccessibilityAction::None => {}
        AccessibilityAction::Click => {
            platform::click_at(screen_x, screen_y, MouseButton::Left);
        }
        AccessibilityAction::ScrollUp(lines) => {
            platform::scroll_at(screen_x, screen_y, lines);
        }
        AccessibilityAction::ScrollDown(lines) => {
            platform::scroll_at(screen_x, screen_y, -lines);
        }
        AccessibilityAction::DwellStarted => {
            // Pure signal — no mouse event to post.
            debug!("dwell started at ({}, {})", screen_x, screen_y);
        }
    }
}

/// Convert a normalized [0, 1] gaze point to absolute display pixels
/// for the given screen width / height.
pub fn normalized_to_screen(nx: f32, ny: f32, width: u32, height: u32) -> (f64, f64) {
    let x = (nx as f64) * (width as f64);
    let y = (ny as f64) * (height as f64);
    debug_assert!(x.is_finite() && y.is_finite(), "non-finite screen coords");
    (x.max(0.0).min(width as f64), y.max(0.0).min(height as f64))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalized_to_screen_center() {
        let (x, y) = normalized_to_screen(0.5, 0.5, 1920, 1080);
        assert_eq!(x, 960.0);
        assert_eq!(y, 540.0);
    }

    #[test]
    fn test_normalized_to_screen_top_left() {
        let (x, y) = normalized_to_screen(0.0, 0.0, 1920, 1080);
        assert_eq!(x, 0.0);
        assert_eq!(y, 0.0);
    }

    #[test]
    fn test_normalized_to_screen_bottom_right() {
        let (x, y) = normalized_to_screen(1.0, 1.0, 800, 600);
        assert_eq!(x, 800.0);
        assert_eq!(y, 600.0);
    }

    #[test]
    fn test_normalized_to_screen_clamps_out_of_range() {
        // Should clamp to [0, width] / [0, height], not wrap or panic.
        let (x, y) = normalized_to_screen(-0.5, 1.5, 1000, 500);
        assert_eq!(x, 0.0);
        assert_eq!(y, 500.0);
    }

    #[test]
    fn test_dispatch_no_op_when_disabled() {
        // Should not panic on any variant when disabled.
        dispatch(AccessibilityAction::None, 100.0, 100.0, false);
        dispatch(AccessibilityAction::Click, 100.0, 100.0, false);
        dispatch(AccessibilityAction::ScrollUp(3), 100.0, 100.0, false);
        dispatch(AccessibilityAction::ScrollDown(3), 100.0, 100.0, false);
        dispatch(AccessibilityAction::DwellStarted, 100.0, 100.0, false);
    }

    #[test]
    fn test_dispatch_handles_non_finite_gracefully() {
        // NaN/Inf should be clamped or coerced to finite values by normalized_to_screen.
        let (x, y) = normalized_to_screen(f32::NAN, f32::NAN, 1920, 1080);
        assert!(x.is_finite());
        assert!(y.is_finite());
    }
}
