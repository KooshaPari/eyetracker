# eyetracker

> Phenotype eye-tracking core — Rust crates for 9-point calibration, gaze
> inference, fixation/saccade classification, and UniFFI bindings for
> Swift / Kotlin hosts.

> **Status:** alpha. APIs may change before `0.1.0`. The previous README
> described a `Tracker`/`Event` API that does not exist in the source; this
> README reflects what is actually in the workspace.

## Workspace

```
eyetracker/
└── crates/
    ├── eyetracker-domain/   # Point, Vector, GazeEstimate, FixationEvent, SaccadeEvent
    ├── eyetracker-math/     # KalmanFilter2D, CalibrationMatrix
    ├── eyetracker-core/     # Calibrator state machine + GazeEstimator + MotionClass
    └── eyetracker-ffi/      # UniFFI bindings (eyetracker.udl)
```

## Public API surface

### `eyetracker-domain`

```rust
pub struct Point          { pub x: f64, pub y: f64 }
pub struct Vector         { pub dx: f64, pub dy: f64 }
pub struct GazeEstimate   { pub position: Point, pub confidence: f64, pub timestamp: SystemTime }
pub struct FixationEvent  { pub centroid: Point, pub duration: Duration, pub start_time: SystemTime }
pub struct SaccadeEvent   { /* … */ }
```

There is intentionally **no** unifying `Event` enum and **no** `GazePoint`
type — emit `FixationEvent` / `SaccadeEvent` separately.

### `eyetracker-core`

```rust
pub enum  CalibrationState { Idle, Waiting, Sampling, Processing, Validating, Complete, Failed }
pub enum  EyetrackerError  { CalibrationFailed(String), InferenceError(String) }
pub struct Calibrator      { /* 9-point calibration workflow */ }
pub struct GazeEstimator   { /* applies calibration + smoothing */ }
pub enum   MotionClass     { /* fixation / saccade / smooth-pursuit classification */ }
```

### `eyetracker-math`

`KalmanFilter2D` for gaze smoothing, `CalibrationMatrix` for eye→screen
mapping.

## Quick start

```rust
use eyetracker_core::{Calibrator, EyetrackerError};
use eyetracker_domain::Point;

fn main() -> Result<(), EyetrackerError> {
    let mut cal = Calibrator::new();
    cal.start_calibration();

    // Feed at least 3 (eye_point, screen_target) pairs
    for (eye, target) in collect_calibration_samples() {
        cal.record_sample(eye, target)?;
    }

    cal.finalize()?;
    Ok(())
}

fn collect_calibration_samples() -> Vec<(Point, Point)> {
    // Replace with real capture logic
    vec![
        (Point::new(0.0, 0.0),   Point::new(0.0, 0.0)),
        (Point::new(640.0, 0.0), Point::new(960.0, 0.0)),
        (Point::new(0.0, 360.0), Point::new(0.0, 540.0)),
    ]
}
```

For runtime gaze inference, use `GazeEstimator` from `eyetracker-core`.

## UniFFI bindings

`eyetracker-ffi` exposes a deliberately small surface (`Point`,
`GazeEstimate`, `CalibrationState`, plus a couple of factory functions) via
`src/eyetracker.udl`:

```bash
cd crates/eyetracker-ffi
cargo build --release
# Generate Swift / Kotlin packages from the same .udl
uniffi-bindgen generate src/eyetracker.udl --language swift  --out-dir ./out/swift
uniffi-bindgen generate src/eyetracker.udl --language kotlin --out-dir ./out/kotlin
```

Drop the generated package and the compiled `libeyetracker_ffi` artifact into
the host Xcode/Gradle project.

## Development

```bash
cargo build --workspace
cargo test  --workspace
cargo clippy --workspace -- -D warnings
cargo fmt --check
```

## Cargo dependency

```toml
[dependencies]
eyetracker-core   = { git = "https://github.com/KooshaPari/eyetracker" }
eyetracker-domain = { git = "https://github.com/KooshaPari/eyetracker" }
```

## Contributing

Issues and PRs welcome. Please ensure new public types are reflected here —
earlier drafts of this README documented an API surface (`Tracker`,
`TrackerConfig`, `tracker.feed`, `Event::Fixation`, `Event::Saccade`,
`GazePoint`) that did not exist.

## License

[MIT License](LICENSE) © Koosha Pari.
