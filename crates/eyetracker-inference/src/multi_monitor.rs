//! Multi-monitor calibration support (FR-EYE-CAL-005)
//!
//! Stores separate calibration matrices per display, keyed by display UUID.
//! On focus change to a new monitor, the corresponding calibration is loaded.
//! If no calibration exists for the active display, the user is warned.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::calibration::CalibrationResult;

/// Per-display metadata for multi-monitor calibration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct DisplayId {
    /// OS-assigned display UUID (e.g., CGDisplayUUID on macOS)
    pub uuid: String,
    /// Human-readable label
    pub label: String,
    /// Display resolution (width x height)
    pub resolution: (u32, u32),
    /// Position in the global coordinate space
    pub position: (i32, i32),
}

impl DisplayId {
    /// Create a synthetic display ID (used in tests)
    pub fn synthetic(uuid: &str) -> Self {
        Self {
            uuid: uuid.to_string(),
            label: format!("Synthetic-{uuid}"),
            resolution: (1920, 1080),
            position: (0, 0),
        }
    }
}

/// Multi-monitor calibration store
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct MultiMonitorCalibration {
    /// Map from display UUID -> calibration result
    calibrations: HashMap<String, (DisplayId, CalibrationResult)>,
}

impl MultiMonitorCalibration {
    /// Create a new empty store
    pub fn new() -> Self {
        Self::default()
    }

    /// Load from the default location on disk
    pub fn load() -> Result<Self> {
        let path = store_path()?;
        if !path.exists() {
            return Ok(Self::new());
        }
        let encoded = std::fs::read(&path)?;
        let store: Self = bincode::deserialize(&encoded)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize multi-monitor calibration: {e}"))?;
        Ok(store)
    }

    /// Persist to the default location on disk
    pub fn save(&self) -> Result<()> {
        let path = store_path()?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let encoded = bincode::serialize(self)
            .map_err(|e| anyhow::anyhow!("Failed to serialize multi-monitor calibration: {e}"))?;
        std::fs::write(&path, encoded)?;
        Ok(())
    }

    /// Store a calibration for a specific display
    pub fn store(&mut self, display: DisplayId, calibration: CalibrationResult) {
        self.calibrations
            .insert(display.uuid.clone(), (display, calibration));
    }

    /// Load calibration for a specific display (returns None if not found)
    pub fn load_for(&self, display_uuid: &str) -> Option<&CalibrationResult> {
        self.calibrations.get(display_uuid).map(|(_, c)| c)
    }

    /// List all stored displays
    pub fn displays(&self) -> Vec<&DisplayId> {
        self.calibrations.values().map(|(d, _)| d).collect()
    }

    /// Number of stored calibrations
    pub fn len(&self) -> usize {
        self.calibrations.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.calibrations.is_empty()
    }

    /// Remove calibration for a display
    pub fn remove(&mut self, display_uuid: &str) -> Option<CalibrationResult> {
        self.calibrations.remove(display_uuid).map(|(_, c)| c)
    }
}

fn store_path() -> Result<PathBuf> {
    let data_dir = dirs::data_local_dir()
        .ok_or_else(|| anyhow::anyhow!("Could not determine platform data directory"))?;
    Ok(data_dir.join("eyetracker").join("multi_monitor_cal.bin"))
}

/// Detect currently active display (macOS implementation)
#[cfg(target_os = "macos")]
pub fn detect_active_display() -> Result<DisplayId> {
    use core_graphics::display::CGDisplay;

    let displays = CGDisplay::active_displays()
        .map_err(|e| anyhow::anyhow!("Failed to enumerate displays: {e:?}"))?;

    // Pick the main display (the one with origin at 0,0 is the convention
    // for the primary/main display on macOS).
    let mut main_id = displays.first().copied();
    for &id in &displays {
        let cg = CGDisplay::new(id);
        let bounds = cg.bounds();
        if bounds.origin.x == 0.0 && bounds.origin.y == 0.0 {
            main_id = Some(id);
            break;
        }
    }
    let display_id = main_id.unwrap_or(displays[0]);

    let cg = CGDisplay::new(display_id);
    let bounds = cg.bounds();
    Ok(DisplayId {
        uuid: format!("cg-{display_id}"),
        label: format!("Display {display_id}"),
        resolution: (bounds.size.width as u32, bounds.size.height as u32),
        position: (bounds.origin.x as i32, bounds.origin.y as i32),
    })
}

/// Stub for non-macOS platforms (Linux/Windows) — returns a synthetic main display
#[cfg(not(target_os = "macos"))]
pub fn detect_active_display() -> Result<DisplayId> {
    Ok(DisplayId::synthetic("main"))
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;
    use crate::calibration::{CalibrationPoint, CalibrationSample};

    fn dummy_calibration() -> CalibrationResult {
        CalibrationResult {
            samples: vec![CalibrationSample {
                point: CalibrationPoint {
                    x: 0.5,
                    y: 0.5,
                    label: "center".into(),
                },
                gaze_samples: vec![(0.5, 0.5, 0.0)],
                timestamp: Instant::now(),
            }],
            quality: 0.8,
            success: true,
        }
    }

    #[test]
    fn test_store_and_load_for_display() {
        let mut store = MultiMonitorCalibration::new();
        let display = DisplayId::synthetic("monitor-1");
        store.store(display.clone(), dummy_calibration());

        assert_eq!(store.len(), 1);
        assert!(store.load_for("monitor-1").is_some());
        assert!(store.load_for("monitor-2").is_none());
    }

    #[test]
    fn test_remove_display() {
        let mut store = MultiMonitorCalibration::new();
        let display = DisplayId::synthetic("monitor-1");
        store.store(display.clone(), dummy_calibration());

        let removed = store.remove("monitor-1");
        assert!(removed.is_some());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_list_displays() {
        let mut store = MultiMonitorCalibration::new();
        store.store(DisplayId::synthetic("a"), dummy_calibration());
        store.store(DisplayId::synthetic("b"), dummy_calibration());

        let displays = store.displays();
        assert_eq!(displays.len(), 2);
    }

    #[test]
    fn test_detect_active_display_returns_something() {
        let display = detect_active_display().expect("detect_active_display");
        assert!(!display.uuid.is_empty());
        assert!(display.resolution.0 > 0);
        assert!(display.resolution.1 > 0);
    }

    #[test]
    fn test_persistence_round_trip() {
        let tmp = std::env::temp_dir().join(format!(
            "eyetracker-multi-monitor-test-{}.bin",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&tmp);

        let mut store = MultiMonitorCalibration::new();
        store.store(DisplayId::synthetic("test-uuid"), dummy_calibration());
        let encoded = bincode::serialize(&store).unwrap();
        std::fs::write(&tmp, encoded).unwrap();

        let read = std::fs::read(&tmp).unwrap();
        let decoded: MultiMonitorCalibration = bincode::deserialize(&read).unwrap();
        assert_eq!(decoded.len(), 1);
        assert!(decoded.load_for("test-uuid").is_some());

        let _ = std::fs::remove_file(&tmp);
    }
}
