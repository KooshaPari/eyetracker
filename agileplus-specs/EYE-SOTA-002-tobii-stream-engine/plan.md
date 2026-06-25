# EYE-SOTA-002 — Implementation Plan

## Phase 1: Crate scaffold
- WP01 — Add `crates/eyetracker-tobii` to workspace
- WP02 — Cargo.toml: `tobii = "0.1"` (or pin exact version)
- WP03 — `lib.rs` re-exports + `error.rs` (`thiserror`)
- WP04 — feature flag `tobii-runtime` for optional native lib discovery

## Phase 2: HAL trait extension
- WP05 — `eyetracker-core::GazeSource` trait: `next_gaze(&mut self) -> Result<Gaze>`
- WP06 — `eyetracker-core::TrackerInfo` struct: id, model, firmware
- WP07 — Update `eyetracker-camera` to implement `GazeSource` (currently
  raw f64; wrap in trait)

## Phase 3: Stream Engine
- WP08 — `discovery.rs`: enumerate via `tobii_enumerate_local_addresses`
- WP09 — `stream.rs`: subscribe to gaze + position
- WP10 — `calibration.rs`: stub (out of scope for this WP, will land
  under EYE-SOTA-008)
- WP11 — `mapper.rs`: 3D-to-2D projection helper

## Phase 4: CLI
- WP12 — `examples/tobii_list.rs` — print discovered trackers
- WP13 — `examples/tobii_record.rs` — record 10s of gaze to CSV

## Phase 5: Quality
- WP14 — Unit tests for stream subscription lifecycle
- WP15 — `cargo +nightly fmt` + commit on `feat/EYE-SOTA-002` branch
- WP16 — Re-run `grade.sh --fast` — verify no regression
- WP17 — Open PR with [DAG: EYE-SOTA-002] prefix in title

## Dependencies on external state
- Tobii Stream Engine SDK 1.x (Windows: copy `tobii_research.h` to
  `crates/eyetracker-tobii/include/`)
- `tobii` crate: `cargo search tobii` to confirm latest version
