# EyeTracker Usage & Cookbook

## Architecture: Layered Crate Design

EyeTracker splits into four composable Rust crates (verified in `crates/`):

| Crate | Role | Purpose |
|-------|------|---------|
| **eyetracker-domain** | Core domain types | Point, Vector, GazeEstimate, FixationEvent, SaccadeEvent |
| **eyetracker-math** | Mathematical kernels | Kalman 2D filter, 3-point calibration solver (Cramer's rule) |
| **eyetracker-core** | Orchestration layer | Calibrator state machine, GazeEstimator, MotionClass |
| **eyetracker-ffi** | Language bindings | UniFFI (Swift/Kotlin), eyetracker.udl interface |

## Calibration State Machine (eyetracker-core::Calibrator)

```
┌─────────────────────────────────────────────┐
│ Idle → start_calibration()                  │
│   ↓                                         │
│ Waiting (user fixates on target)            │
│   ↓                                         │
│ Sampling (record_sample: collect ≥3 pairs) │
│   ↓                                         │
│ Processing (finalize: solve 3×3 matrix)    │
│   ↓                                         │
│ Validating (check accuracy ≤1.5°)          │
│   ├→ Complete (success)                    │
│   └→ Failed (accuracy exceeded)            │
└─────────────────────────────────────────────┘
```

**Traces to:** FR-EYE-CAL-001, FR-EYE-CAL-002

## Gaze Estimation (eyetracker-core::GazeEstimator)

Three-step pipeline:

1. **Calibration Transform**: Apply stored CalibrationMatrix (eye → screen)
2. **Kalman Filtering**: 2D Kalman filter (velocity model, α=0.5 blending)
3. **Motion Classification**: Detect fixation (v<30°/s) vs saccade (v>50°/s)

**Traces to:** FR-EYE-INFER-001, FR-EYE-INFER-003

## Usage Examples

### Rust: Full Calibration Workflow

```rust
use eyetracker_core::{Calibrator, CalibrationState, GazeEstimate};
use eyetracker_domain::Point;

// 1. Create and start calibration
let mut calibrator = Calibrator::new();
calibrator.start_calibration();
assert_eq!(calibrator.state(), CalibrationState::Waiting);

// 2. Record 3+ calibration samples (eye → screen target)
calibrator.record_sample(
    Point::new(50.0, 50.0),    // eye position
    Point::new(100.0, 100.0),  // screen target
)?;
calibrator.record_sample(Point::new(300.0, 50.0), Point::new(400.0, 100.0))?;
calibrator.record_sample(Point::new(150.0, 300.0), Point::new(200.0, 400.0))?;

// 3. Finalize (computes affine matrix via Cramer's rule)
calibrator.finalize()?;
assert_eq!(calibrator.state(), CalibrationState::Complete);
println!("Calibration accuracy: {:.2}°", calibrator.accuracy());

// 4. Extract calibration for gaze estimation
let calibration = calibrator.get_calibration().unwrap();
```

### Rust: Real-Time Gaze Estimation

```rust
use eyetracker_core::{GazeEstimator, MotionClass};
use std::time::SystemTime;

// Initialize with calibration from previous session
let mut estimator = GazeEstimator::with_calibration(calibration);

// Feed raw eye estimates (from vision model, hardware tracker, etc.)
let raw_gaze = GazeEstimate::new(
    Point::new(45.0, 48.0),  // raw eye coordinates
    0.92,                     // confidence [0.0–1.0]
    SystemTime::now(),
);

// Get smoothed screen position
let smoothed = estimator.estimate(&raw_gaze)?;
println!("Screen gaze: ({:.0}, {:.0})", smoothed.x, smoothed.y);

// Classify motion
match estimator.classify_motion() {
    MotionClass::Fixation => println!("Fixation detected"),
    MotionClass::Saccade => println!("Saccade detected"),
    MotionClass::Unknown => {},
}
```

### Swift (Phase-3B): Generate Bindings

```bash
cd crates/eyetracker-ffi
cargo build --release
uniffi-bindgen generate src/eyetracker.udl \
  --language swift \
  --out-dir ./bindings/swift
```

Then in Xcode:

```swift
import eyetracker

// Calibration
var cal = Calibrator()
cal.startCalibration()
let eye1 = createPoint(x: 50.0, y: 50.0)
let target1 = createPoint(x: 100.0, y: 100.0)
try cal.recordSample(eyePoint: eye1, screenTarget: target1)
// … (add 2 more samples)
try cal.finalize()
```

### Kotlin (Phase-3B): Generate Bindings

```bash
uniffi-bindgen generate src/eyetracker.udl \
  --language kotlin \
  --out-dir ./bindings/kotlin
```

Then in Android:

```kotlin
import com.example.eyetracker.*

val calibrator = Calibrator()
calibrator.startCalibration()
val sample = CalibrationSample(
    eyePoint = createPoint(50.0, 50.0),
    screenTarget = createPoint(100.0, 100.0)
)
// … (record 3 total, then finalize)
```

## Development

```bash
# Build all crates
cargo build --workspace

# Run tests (FR-traced via @test markers)
cargo test --workspace -- --nocapture

# Lint and format
cargo clippy --workspace -- -D warnings
cargo fmt --check
```

## Phase-3 Implementation Status

- **Phase-3A Complete** (2026-04-25): FFI scaffold in `eyetracker-ffi` with Calibrator + GazeEstimator wrappers
- **Phase-3B Ready**: UniFFI code generation (Swift/Kotlin bindings); disk budget resolved, can execute
- **Phase-3C Pending**: Native platform implementations
  - iOS: ARKit wrapper (Vision framework integration)
  - Android: CameraX wrapper (ML Kit Iris detection)

## Next Steps for Contributors

1. Generate Swift bindings: `uniffi-bindgen generate ... --language swift`
2. Generate Kotlin bindings: `uniffi-bindgen generate ... --language kotlin`
3. Integrate into iOS Xcode project (link libeyetracker_ffi)
4. Integrate into Android Gradle project (link libeyetracker_ffi)
5. Implement native eye tracking (ARKit, CameraX) wrapping the Rust core

See `FUNCTIONAL_REQUIREMENTS.md` for full FR traceability.
