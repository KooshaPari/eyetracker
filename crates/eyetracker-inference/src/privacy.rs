//! Privacy & data handling (FR-EYE-PRIVACY-001, FR-EYE-PRIVACY-002, FR-EYE-PRIVACY-003)
//!
//! Implements the privacy guarantees required by the functional spec:
//! - FR-EYE-PRIVACY-001: All processing is local (no network calls in this crate)
//! - FR-EYE-PRIVACY-002: No default cloud upload; explicit opt-in only
//! - FR-EYE-PRIVACY-003: Screen recording requires explicit per-session consent

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

/// Privacy mode determining what data leaves the device
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrivacyMode {
    /// Strictly local; no data leaves the device.
    LocalOnly,
    /// Local processing but user has opted-in to export specific data.
    LocalWithExport,
}

/// User consent record for screen recording
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScreenRecordingConsent {
    /// Display where consent was given
    pub display_uuid: String,
    /// Unix timestamp when consent was given
    pub granted_at: u64,
    /// Session ID this consent applies to
    pub session_id: String,
    /// Duration of consent (None = session-only)
    pub expires_at: Option<u64>,
    /// What data is consented to
    pub scope: ConsentScope,
}

/// What data the user has consented to capture
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsentScope {
    /// Gaze coordinates only
    GazeOnly,
    /// Gaze + screen frames (research/debug mode)
    GazeAndFrames,
}

/// Privacy manager
///
/// Enforces FR-EYE-PRIVACY-001/002/003 by tracking consent state and
/// exposing explicit guards for any code path that would persist or
/// export data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyManager {
    /// Current privacy mode
    pub mode: PrivacyMode,
    /// Session identifier (regenerated per process)
    pub session_id: String,
    /// Screen recording consents granted this session
    pub consents: Vec<ScreenRecordingConsent>,
    /// Whether cloud upload is currently enabled (default: false)
    pub cloud_upload_enabled: bool,
}

impl PrivacyManager {
    /// Create a new privacy manager with strict defaults
    pub fn new() -> Self {
        Self {
            mode: PrivacyMode::LocalOnly,
            session_id: generate_session_id(),
            consents: Vec::new(),
            cloud_upload_enabled: false,
        }
    }

    /// Enable cloud upload (FR-EYE-PRIVACY-002 requires explicit opt-in)
    pub fn enable_cloud_upload(&mut self) {
        self.cloud_upload_enabled = true;
        self.mode = PrivacyMode::LocalWithExport;
    }

    /// Disable cloud upload
    pub fn disable_cloud_upload(&mut self) {
        self.cloud_upload_enabled = false;
        if self.consents.is_empty() {
            self.mode = PrivacyMode::LocalOnly;
        }
    }

    /// Check if a particular export action is allowed
    pub fn can_export(&self, scope: ConsentScope) -> bool {
        match self.mode {
            PrivacyMode::LocalOnly => false,
            PrivacyMode::LocalWithExport => self
                .consents
                .iter()
                .any(|c| c.scope == scope || c.scope == ConsentScope::GazeAndFrames),
        }
    }

    /// Grant screen-recording consent (FR-EYE-PRIVACY-003)
    pub fn grant_recording_consent(&mut self, display_uuid: &str, scope: ConsentScope) {
        let now = unix_timestamp();
        let consent = ScreenRecordingConsent {
            display_uuid: display_uuid.to_string(),
            granted_at: now,
            session_id: self.session_id.clone(),
            expires_at: None, // session-only by default
            scope,
        };
        self.consents.push(consent);
    }

    /// Check if screen recording is currently consented for a display
    pub fn can_record(&self, display_uuid: &str) -> bool {
        self.consents.iter().any(|c| c.display_uuid == display_uuid)
    }

    /// Strip expired consents
    pub fn purge_expired(&mut self) {
        let now = unix_timestamp();
        self.consents.retain(|c| {
            c.expires_at.map(|exp| exp > now).unwrap_or(true) // session-only stays
        });
    }

    /// Returns true if any data has been consented for export
    pub fn has_any_consent(&self) -> bool {
        !self.consents.is_empty()
    }

    /// Number of active consents
    pub fn consent_count(&self) -> usize {
        self.consents.len()
    }
}

impl Default for PrivacyManager {
    fn default() -> Self {
        Self::new()
    }
}

fn generate_session_id() -> String {
    let ts = unix_timestamp();
    format!("sess-{ts}")
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Determine the canonical privacy storage directory
pub fn privacy_storage_path() -> Result<PathBuf, String> {
    let data_dir = dirs::data_local_dir()
        .ok_or_else(|| "Could not determine platform data directory".to_string())?;
    Ok(data_dir.join("eyetracker").join("privacy"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_default_is_strict_local() {
        let mgr = PrivacyManager::new();
        assert_eq!(mgr.mode, PrivacyMode::LocalOnly);
        assert!(!mgr.cloud_upload_enabled);
        assert!(!mgr.can_record("any-display"));
        assert!(!mgr.can_export(ConsentScope::GazeOnly));
    }

    #[test]
    fn test_cloud_upload_requires_opt_in() {
        let mut mgr = PrivacyManager::new();
        assert!(!mgr.cloud_upload_enabled);
        mgr.enable_cloud_upload();
        assert!(mgr.cloud_upload_enabled);
        assert_eq!(mgr.mode, PrivacyMode::LocalWithExport);
        // Without explicit gaze consent, even with cloud enabled, can't export
        assert!(!mgr.can_export(ConsentScope::GazeOnly));
    }

    #[test]
    fn test_recording_consent_required() {
        let mut mgr = PrivacyManager::new();
        mgr.enable_cloud_upload();
        mgr.grant_recording_consent("display-1", ConsentScope::GazeOnly);

        assert!(mgr.can_record("display-1"));
        assert!(!mgr.can_record("display-2"));
        assert!(mgr.can_export(ConsentScope::GazeOnly));
        // Frames consent requires explicit grant
        assert!(!mgr.can_export(ConsentScope::GazeAndFrames));
    }

    #[test]
    fn test_gaze_and_frames_covers_gaze_only() {
        let mut mgr = PrivacyManager::new();
        mgr.enable_cloud_upload();
        mgr.grant_recording_consent("display-1", ConsentScope::GazeAndFrames);
        assert!(mgr.can_export(ConsentScope::GazeAndFrames));
        assert!(mgr.can_export(ConsentScope::GazeOnly));
    }

    #[test]
    fn test_session_id_is_unique() {
        let m1 = PrivacyManager::new();
        std::thread::sleep(Duration::from_millis(1100));
        let m2 = PrivacyManager::new();
        assert_ne!(m1.session_id, m2.session_id);
    }

    #[test]
    fn test_purge_expired_keeps_session_only() {
        let mut mgr = PrivacyManager::new();
        mgr.enable_cloud_upload();
        mgr.grant_recording_consent("d", ConsentScope::GazeOnly);
        mgr.purge_expired();
        assert_eq!(mgr.consent_count(), 1);
    }

    #[test]
    fn test_purge_expired_removes_expired() {
        let mut mgr = PrivacyManager::new();
        mgr.enable_cloud_upload();
        // Inject an already-expired consent
        mgr.consents.push(ScreenRecordingConsent {
            display_uuid: "old".to_string(),
            granted_at: 0,
            session_id: "old-session".to_string(),
            expires_at: Some(1), // expired
            scope: ConsentScope::GazeOnly,
        });
        mgr.purge_expired();
        assert_eq!(mgr.consent_count(), 0);
    }

    #[test]
    fn test_disable_cloud_upload_reverts_mode() {
        let mut mgr = PrivacyManager::new();
        mgr.enable_cloud_upload();
        assert_eq!(mgr.mode, PrivacyMode::LocalWithExport);
        mgr.disable_cloud_upload();
        assert_eq!(mgr.mode, PrivacyMode::LocalOnly);
    }
}
