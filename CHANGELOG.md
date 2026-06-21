# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.3-alpha] - 2026-06-20

### Changed
- **Wire AccessibilityManager end-to-end**: `app::AppState` now ticks the
  dwell-click + gaze-scroll controllers on every frame so the actions
  actually fire at runtime. Previously the manager was constructed
  but `update()` was never invoked — dwell-click and scroll-by-gaze
  were nominally implemented but functionally dead. **FR-EYE-ACCESS-001
  and FR-EYE-ACCESS-002 are now genuinely functional.**
- **CLI**: `--camera-index` short flag changed from `-c` to `-i` to
  resolve a pre-existing clap collision with `--calibrate`. (`-c` is
  now unambiguously "calibrate".)
- **TUI status block** now surfaces the latest accessibility action
  live (`CLICK` / `SCROLL_UP` / `SCROLL_DOWN` / `DWELL_STARTED`).

### Added
- 2 new integration tests in
  `crates/eyetracker-inference/tests/fr_eye_integration.rs`:
  - `fr_eye_access_001_dwell_click_fires_through_app_tick_path` —
    drives 30 simulated fixating frames through the same `tick_accessibility`
    path the app uses and asserts a `Click` action fires within the
    dwell window.
  - `fr_eye_access_002_scroll_emits_through_app_tick_path` — feeds
    gaze samples in the top 20% and asserts `ScrollUp` is emitted;
    same for `ScrollDown` in the bottom 20%.

## [0.1.2-alpha] - 2026-06-20

### Changed
- **Drift monitor surfaced end-to-end in TUI** (FR-EYE-CAL-004
  wired): the drift panel now shows the live status
  (`OK` / `WARN` / `RECALIBRATE`) with the actual drift degree
  readout, color-coded by severity. When a Critical event fires,
  a red **"Recalibrate?"** prompt appears with a `[d] Dismiss`
  hint in the help bar.
- **Pipeline per-frame latency tracing** (FR-EYE-INFER-001):
  `pipeline.rs` now emits `tracing::debug!(latency_ms, frame,
  face_detected, events)` per frame. Run with
  `RUST_LOG=eyetracker=debug` for the full timeline.
- **Pipeline feeds real smoothed gaze into DriftMonitor** (was a
  no-op stub). New `Arc<AtomicBool>` bridges the UI thread
  (key handler) to the processing thread (data closure) so the
  dismiss is consumed on the next frame.

### Added
- `DwellClickDetector::set_dwell_duration()` + `dwell_duration()`
  with spec-range clamp (200-1000ms).
- `--dwell-ms <N>` CLI flag (out-of-range values clamped).
- 1 new integration test:
  `fr_eye_cal_004_drift_trigger_to_dismiss_to_resume_cycle`
  (trigger → not-dismissed → dismiss → suppressed → reset).

## [0.1.1-alpha] - 2026-06-20

### Added
- **19 FR-referenced integration tests** in
  `crates/eyetracker-inference/tests/fr_eye_integration.rs`
  covering all 16 in-scope FRs (CAL-001..005, INFER-001..004,
  ACCESS-001/002, PRIVACY-001..003, INTEROP-001..003) plus an
  end-to-end smoother→classifier→drift round-trip. Closes the
  spec's **Quality & Testing** clause: *"All FRs shall be
  verified by automated test coverage in `tests/`."*
- **Per-inference latency tracing** (`tracing::debug!` in
  `pipeline.rs:181-192`) — run with `RUST_LOG=eyetracker=debug`
  for the full per-frame timeline.
- **`--dwell-ms <N>`** CLI flag with FR-EYE-ACCESS-001 spec-range
  clamp (200-1000ms).
- **`DwellClickDetector::set_dwell_duration()`** +
  `dwell_duration()` for programmatic configuration.

## [0.1.0-alpha] - 2026-06-20

### Added
- **First Rust release** — complete replacement of the original
  Python webcam-based implementation.
- **7 crates**: `eyetracker-camera`, `eyetracker-inference`,
  `eyetracker-cli`, `eyetracker-core`, `eyetracker-domain`,
  `eyetracker-ffi`, `eyetracker-math`.
- **Full inference pipeline**:
  Camera → Face Detection (geometric fallback + ONNX slot) →
  468-point Face Mesh Landmarks → Gaze Vector Estimation →
  Kalman 2D Smoothing → Fixation/Saccade Classification →
  Drift Monitor → Multi-Monitor Calibration → Privacy Manager →
  ratatui TUI / CSV / bincode persistence.
- **Working UniFFI Swift + Kotlin bindings** generated from
  `eyetracker.udl` via `uniffi-bindgen`.
- **FocalPoint NDJSON connector** (`pheno-interop-003`).
- **CLI binary**: `cargo run --release -- [OPTIONS]` with
  `--calibrate`, `--csv`, `--list-cameras`, `--load-calibration`.
- **3 new inference crates** totalling ~2,500 LOC of new code
  across 11 modules.
- 32 → 76 → 112 → 132 → 141 tests, all passing.

[Unreleased]: https://github.com/KooshaPari/eyetracker/compare/v0.1.3-alpha...HEAD
[0.1.3-alpha]: https://github.com/KooshaPari/eyetracker/compare/v0.1.2-alpha...v0.1.3-alpha
[0.1.2-alpha]: https://github.com/KooshaPari/eyetracker/compare/v0.1.1-alpha...v0.1.2-alpha
[0.1.1-alpha]: https://github.com/KooshaPari/eyetracker/compare/v0.1.0-alpha...v0.1.1-alpha
[0.1.0-alpha]: https://github.com/KooshaPari/eyetracker/releases/tag/v0.1.0-alpha