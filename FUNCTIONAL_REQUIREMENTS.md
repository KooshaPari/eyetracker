# Eyetracker — Functional Requirements

## Calibration (FR-EYE-CAL-001 through FR-EYE-CAL-005)

**FR-EYE-CAL-001: 9-Point Calibration Protocol**
System shall present 9 calibration targets in a 3x3 grid and record gaze samples at each point. User must fixate for ≥500ms per target; system shall dismiss and request retry if insufficient samples collected.

**FR-EYE-CAL-002: Accuracy Validation & Drift Tolerance**
Post-calibration, system shall validate accuracy by computing residual error between predicted and actual gaze. Drift tolerance shall be ≤1.5° (angular); if exceeded, system shall flag as invalid and require recalibration.

**FR-EYE-CAL-003: Calibration Persistence**
Calibration matrix shall be serialized to disk (platform-appropriate: `~/Library/Application Support/eyetracker/cal.bin` on macOS) and reloaded on startup. User shall not be forced to recalibrate on every session.

**FR-EYE-CAL-004: Recalibration Trigger & Degradation Monitoring**
System shall monitor runtime accuracy drift (via periodic validation samples) and automatically trigger recalibration dialog when accuracy degrades >2° from baseline. User may dismiss dialog once per session.

**FR-EYE-CAL-005: Multi-Monitor Calibration**
System shall store separate calibration matrices per display (keyed by display UUID). When app focus moves to a new monitor, system shall load the corresponding calibration and warn if unavailable.

## Inference (FR-EYE-INFER-001 through FR-EYE-INFER-004)

**FR-EYE-INFER-001: Real-Time Inference Latency**
Gaze point computation (from eye image → calibrated screen coordinates) shall complete in ≤30ms wall-clock on target hardware (Apple Neural Engine / MediaPipe). Latency shall be measured and logged per inference.

**FR-EYE-INFER-002: Kalman 2D Smoothing**
Raw gaze estimates shall be fed through a 2D Kalman filter (velocity model) to reduce noise and jitter. Filter state shall be reset on saccade detection. Smoothed output shall be exposed as primary gaze stream.

**FR-EYE-INFER-003: Fixation Classification**
System shall classify gaze into fixation or saccade based on velocity threshold (fixation: velocity <30°/s for ≥100ms). Fixations shall include duration and centroid; saccades shall include duration and amplitude.

**FR-EYE-INFER-004: Saccade Detection**
System shall detect rapid eye movements (velocity >50°/s) and emit saccade events with millisecond precision. Saccade detection shall not be confused with blink-induced noise (blink duration ≤300ms, saccade ≥50ms).

## Privacy & Data Handling (FR-EYE-PRIVACY-001 through FR-EYE-PRIVACY-003)

**FR-EYE-PRIVACY-001: On-Device Processing**
All gaze inference, calibration, and fixation logic shall execute on-device (using Apple Neural Engine, MediaPipe, or equivalent). No gaze data shall be sent to cloud services without explicit user consent.

**FR-EYE-PRIVACY-002: No Default Cloud Upload**
By default, system shall not upload calibrations, raw gaze logs, or inference results to any external service. User may opt-in via settings to export data for debugging, with explicit destination confirmation.

**FR-EYE-PRIVACY-003: Optional Screen Recording Consent**
If screen recording is enabled for research/debugging, system shall require explicit user acknowledgment before any session data (gaze + screen frames) is persisted. Consent dialog shall appear once per session.

## Interoperability (FR-EYE-INTEROP-001 through FR-EYE-INTEROP-003)

**FR-EYE-INTEROP-001: UniFFI Swift Binding**
System shall expose core gaze inference and calibration APIs via UniFFI bindings for consumption by native Swift apps. Bindings shall include gaze event stream and calibration import/export.

**FR-EYE-INTEROP-002: JNI Kotlin Binding**
System shall expose core APIs via JNI for Android integration. Kotlin code shall call native gaze inference and calibration routines. Binding shall handle lifecycle (start/stop tracking) and permission gating.

**FR-EYE-INTEROP-003: FocalPoint Integration Connector**
System shall publish a focus-eye-tracker connector (Rust binary) that bridges eyetracker gaze stream to FocalPoint window-focus bus. Connector shall emit window-id + gaze-point tuples on shared bus.

## Accessibility (FR-EYE-ACCESS-001 through FR-EYE-ACCESS-002)

**FR-EYE-ACCESS-001: Dwell-Click Selection**
System shall support dwell-click (fixation ≥500ms on a screen region = mouse click). Dwell duration shall be configurable (200–1000ms) and cancellable via saccade to a safe zone.

**FR-EYE-ACCESS-002: Scroll-by-Gaze**
System shall support directional scrolling: fixation in upper 20% of screen scrolls up; lower 20% scrolls down. Scroll speed shall be proportional to distance from center (0% speed at 50% from center).

## Quality & Testing Requirements

All FRs shall be verified by automated test coverage in `tests/`. Each test shall reference the relevant FR ID. Calibration accuracy tests shall use synthetic gaze sequences. Inference latency shall be profiled on target hardware.
