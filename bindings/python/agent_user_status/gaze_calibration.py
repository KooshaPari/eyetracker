"""Calibration math and screen-target helpers for derived gaze tracking."""

from __future__ import annotations

import math
import time
from typing import Any, Protocol


class ScreenLike(Protocol):
    @property
    def width(self) -> int: ...

    @property
    def height(self) -> int: ...


def expanded_features(features: list[float]) -> list[float]:
    centered = [float(value) for value in features]
    squares = [value * value for value in centered]
    return [1.0, *centered, *squares, centered[0] * centered[2], centered[1] * centered[3]]


def fit_calibration(samples: list[tuple[list[float], float, float]], screen: ScreenLike) -> dict[str, Any]:
    if len(samples) < 12:
        raise ValueError("not enough calibration samples")
    import numpy as np

    x_mat = np.asarray([expanded_features(row[0]) for row in samples], dtype=float)
    y_x = np.asarray([row[1] for row in samples], dtype=float)
    y_y = np.asarray([row[2] for row in samples], dtype=float)
    ridge = 1e-4
    reg = np.eye(x_mat.shape[1]) * ridge
    reg[0, 0] = 0
    xtx = x_mat.T @ x_mat + reg
    weights_x = np.linalg.solve(xtx, x_mat.T @ y_x)
    weights_y = np.linalg.solve(xtx, x_mat.T @ y_y)
    pred_x = x_mat @ weights_x
    pred_y = x_mat @ weights_y
    error = np.sqrt((pred_x - y_x) ** 2 + (pred_y - y_y) ** 2)
    return {
        "version": 1,
        "kind": "mediapipe_iris_regression",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "screen_width": screen.width,
        "screen_height": screen.height,
        "feature_count": len(samples[0][0]),
        "weights_x": weights_x.tolist(),
        "weights_y": weights_y.tolist(),
        "mean_error_px": float(error.mean()),
        "p95_error_px": float(np.percentile(error, 95)),
        "sample_count": len(samples),
    }


def load_calibration_quality(calibration: dict[str, Any], screen: ScreenLike) -> dict[str, Any]:
    diagonal = math.hypot(float(screen.width), float(screen.height))
    mean_error = float(calibration.get("mean_error_px", 0.0) or 0.0)
    p95_error = float(calibration.get("p95_error_px", 0.0) or 0.0)
    sample_count = float(calibration.get("sample_count", 0) or 0)
    quality_score = max(0.0, min(1.0, 1.0 - min(1.0, p95_error / max(1.0, diagonal * 0.22))))
    if sample_count >= 18 and quality_score >= 0.82:
        quality_label = "excellent"
        recommended_action = "monitor"
    elif sample_count >= 14 and quality_score >= 0.66:
        quality_label = "usable"
        recommended_action = "monitor"
    elif sample_count >= 12 and quality_score >= 0.5:
        quality_label = "fragile"
        recommended_action = "recalibrate_when_convenient"
    else:
        quality_label = "poor"
        recommended_action = "recalibrate"
    return {
        "calibration_mean_error_px": round(mean_error, 2),
        "calibration_p95_error_px": round(p95_error, 2),
        "calibration_sample_count": int(sample_count),
        "calibration_quality_score": round(quality_score, 4),
        "calibration_quality_label": quality_label,
        "calibration_recommended_action": recommended_action,
    }


def projection_thresholds(calibration: dict[str, Any], screen: ScreenLike) -> tuple[float, float]:
    mean_error = float(calibration.get("mean_error_px", 0.0) or 0.0)
    p95_error = float(calibration.get("p95_error_px", 0.0) or 0.0)
    quality_score = float(calibration.get("calibration_quality_score", 0.0) or 0.0)
    if quality_score <= 0.0:
        diagonal = math.hypot(float(screen.width), float(screen.height))
        quality_score = max(0.0, min(1.0, 1.0 - min(1.0, p95_error / max(1.0, diagonal * 0.22))))
    shortest = float(min(screen.width, screen.height))
    diagonal = math.hypot(float(screen.width), float(screen.height))
    hold_threshold = max(90.0, shortest * 0.09, p95_error * 1.25, mean_error * 2.4)
    hold_threshold = min(hold_threshold, diagonal * 0.42)
    quality_factor = 0.74 + (quality_score * 0.42)
    hold_threshold *= quality_factor
    release_threshold = max(42.0, hold_threshold * 0.58, p95_error * 0.82, mean_error * 1.25)
    release_threshold = min(release_threshold, hold_threshold * 0.82)
    return (hold_threshold, release_threshold)


def predict(calibration: dict[str, Any], features: list[float]) -> tuple[float, float]:
    terms = expanded_features(features)
    weights_x = calibration["weights_x"]
    weights_y = calibration["weights_y"]
    if len(terms) != len(weights_x) or len(terms) != len(weights_y):
        raise ValueError("calibration feature shape does not match tracker")
    x = sum(term * weight for term, weight in zip(terms, weights_x, strict=True))
    y = sum(term * weight for term, weight in zip(terms, weights_y, strict=True))
    return (float(x), float(y))


def calibration_points(screen: ScreenLike) -> list[tuple[int, int]]:
    xs = [int(screen.width * ratio) for ratio in (0.12, 0.5, 0.88)]
    ys = [int(screen.height * ratio) for ratio in (0.14, 0.5, 0.86)]
    return [(x, y) for y in ys for x in xs]
