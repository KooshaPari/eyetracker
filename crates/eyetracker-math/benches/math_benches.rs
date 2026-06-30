//! Criterion benchmarks for eyetracker-math hot paths.
//!
//! Run with: `cargo bench -p eyetracker-math`
//!
//! SLO targets (see FR-EYE-INFER-001, FR-EYE-INFER-002):
//!   - Kalman update:         < 1 µs per frame at 60 FPS
//!   - Calibration apply:     < 1 µs per frame
//!   - Calibration 3-point:   < 50 µs (one-shot, not on the hot path)

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use eyetracker_domain::Point;
use eyetracker_math::{CalibrationMatrix, KalmanFilter2D};

fn bench_kalman_update(c: &mut Criterion) {
    let mut kf = KalmanFilter2D::new(Point::new(0.5, 0.5));
    let measurement = Point::new(0.51, 0.49);
    c.bench_function("kalman_update", |b| {
        b.iter(|| {
            kf.update(black_box(measurement));
            black_box(kf.position())
        })
    });
}

fn bench_kalman_predict(c: &mut Criterion) {
    let mut kf = KalmanFilter2D::new(Point::new(0.5, 0.5));
    c.bench_function("kalman_predict_1ms", |b| {
        b.iter(|| {
            kf.predict(black_box(0.001));
            black_box(kf.position())
        })
    });
}

fn bench_calibration_apply(c: &mut Criterion) {
    let eye_points = [
        Point::new(0.0, 0.0),
        Point::new(1.0, 0.0),
        Point::new(0.0, 1.0),
    ];
    let screen_points = [
        Point::new(0.1, 0.1),
        Point::new(0.9, 0.1),
        Point::new(0.1, 0.9),
    ];
    let cal = CalibrationMatrix::from_3_point_calibration(&eye_points, &screen_points)
        .expect("calibration solve");
    let eye = Point::new(0.5, 0.5);
    c.bench_function("calibration_apply", |b| {
        b.iter(|| black_box(cal.apply(black_box(eye))))
    });
}

fn bench_calibration_solve(c: &mut Criterion) {
    let eye_points = [
        Point::new(0.0, 0.0),
        Point::new(1.0, 0.0),
        Point::new(0.0, 1.0),
    ];
    let screen_points = [
        Point::new(0.1, 0.1),
        Point::new(0.9, 0.1),
        Point::new(0.1, 0.9),
    ];
    c.bench_function("calibration_3point_solve", |b| {
        b.iter(|| {
            CalibrationMatrix::from_3_point_calibration(
                black_box(&eye_points),
                black_box(&screen_points),
            )
            .unwrap()
        })
    });
}

criterion_group!(
    benches,
    bench_kalman_update,
    bench_kalman_predict,
    bench_calibration_apply,
    bench_calibration_solve
);
criterion_main!(benches);
