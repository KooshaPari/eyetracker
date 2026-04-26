# eyetracker

> Phenotype eye-tracking framework — a Rust core for gaze estimation and
> fixation analysis, plus UniFFI-generated bindings for embedding in mobile
> and desktop hosts.

`eyetracker` provides the math, domain types, and runtime wiring needed to
turn a raw gaze stream (from a webcam, Tobii device, or AR headset) into
high-level events: fixations, saccades, smooth-pursuit segments, and
calibration-corrected screen coordinates. The core is `no_std`-friendly Rust;
the FFI layer exposes the same API to Swift, Kotlin, and any UniFFI target.

> **Status:** alpha. Public APIs may change before `0.1.0`.

## Workspace

```
eyetracker/
└── crates/
    ├── eyetracker-domain/   # core types: GazePoint, Fixation, Saccade, …
    ├── eyetracker-math/     # geometry, filters, calibration solvers
    ├── eyetracker-core/     # runtime: stream processing + event detection
    └── eyetracker-ffi/      # UniFFI bindings for Swift / Kotlin / etc.
```

## Install

### Rust

```toml
[dependencies]
eyetracker-core   = { git = "https://github.com/KooshaPari/eyetracker" }
eyetracker-domain = { git = "https://github.com/KooshaPari/eyetracker" }
```

### Swift / Kotlin

Build the FFI crate with UniFFI to produce a platform package:

```bash
cd crates/eyetracker-ffi
cargo build --release
uniffi-bindgen generate src/eyetracker.udl --language swift --out-dir ./out/swift
uniffi-bindgen generate src/eyetracker.udl --language kotlin --out-dir ./out/kotlin
```

Drop the generated package into your Xcode/Gradle project alongside the
compiled `libeyetracker_ffi` artifact.

## Quick start

```rust
use eyetracker_core::{Tracker, TrackerConfig};
use eyetracker_domain::{GazePoint, Event};

fn main() -> anyhow::Result<()> {
    let mut tracker = Tracker::new(TrackerConfig::default());

    for sample in raw_gaze_stream()? {
        let point = GazePoint::new(sample.x, sample.y, sample.timestamp);
        for event in tracker.feed(point) {
            match event {
                Event::Fixation(f) => println!("fixation @ {:?} for {:?}", f.center, f.duration),
                Event::Saccade(s)  => println!("saccade {:?} -> {:?}",     s.from,   s.to),
                _ => {}
            }
        }
    }
    Ok(())
}
```

## Development

```bash
cargo test --workspace
cargo clippy --workspace -- -D warnings
cargo fmt --check
```

## Contributing

Issues and PRs welcome. Please include tests for new detectors or filters,
and benchmarks for any change in the hot path (`eyetracker-core::Tracker::feed`).

## License

[MIT License](LICENSE) © Koosha Pari.
