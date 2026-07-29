"""Helpers for normalizing derived eye-state payloads."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def as_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("numeric value must be finite")
    return parsed


def as_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    return int(value)


def bounded_float(value: Any, default: float, low: float, high: float, name: str) -> float:
    parsed = as_float(value, default)
    if parsed is None or parsed < low or parsed > high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return parsed


def bounded_int(value: Any, default: int, low: int, high: int, name: str) -> int:
    parsed = as_int(value, default)
    if parsed is None or parsed < low or parsed > high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return parsed


def build_eye_record(payload: dict[str, Any]) -> dict[str, Any]:
    score = bounded_float(payload.get("score"), 0.5, 0.0, 1.0, "score")
    max_age = bounded_int(payload.get("max_age_seconds"), 5, 1, 3600, "max_age_seconds")
    zone = payload.get("screen_zone")
    state = str(payload.get("state") or (f"looking_at_screen:{zone}" if zone else "looking_at_screen"))
    screen_x = payload.get("screen_x")
    screen_y = payload.get("screen_y")
    screen_width = payload.get("screen_width")
    screen_height = payload.get("screen_height")
    eye = {
        "score": score,
        "state": state,
        "screen_zone": zone,
        "screen_x": bounded_float(screen_x, 0.0, -100000.0, 100000.0, "screen_x") if screen_x is not None else None,
        "screen_y": bounded_float(screen_y, 0.0, -100000.0, 100000.0, "screen_y") if screen_y is not None else None,
        "screen_width": bounded_float(screen_width, 0.0, 1.0, 100000.0, "screen_width")
        if screen_width is not None
        else None,
        "screen_height": bounded_float(screen_height, 0.0, 1.0, 100000.0, "screen_height")
        if screen_height is not None
        else None,
        "observed_screen_x": bounded_float(
            payload.get("observed_screen_x"), 0.0, -100000.0, 100000.0, "observed_screen_x"
        )
        if payload.get("observed_screen_x") is not None
        else None,
        "observed_screen_y": bounded_float(
            payload.get("observed_screen_y"), 0.0, -100000.0, 100000.0, "observed_screen_y"
        )
        if payload.get("observed_screen_y") is not None
        else None,
        "projection_error_px": bounded_float(
            payload.get("projection_error_px"), 0.0, 0.0, 100000.0, "projection_error_px"
        )
        if payload.get("projection_error_px") is not None
        else None,
        "projection_offscreen_px": bounded_float(
            payload.get("projection_offscreen_px"), 0.0, 0.0, 100000.0, "projection_offscreen_px"
        )
        if payload.get("projection_offscreen_px") is not None
        else None,
        "projection_hold_active": bool(payload.get("projection_hold_active", False)),
        "projection_hold_reason": str(payload.get("projection_hold_reason") or "unknown")[:80],
        "projection_hold_hint": str(payload.get("projection_hold_hint") or "unknown")[:160]
        if payload.get("projection_hold_hint") is not None
        else None,
        "projection_hold_samples": bounded_int(
            payload.get("projection_hold_samples"), 0, 0, 100000, "projection_hold_samples"
        )
        if payload.get("projection_hold_samples") is not None
        else None,
        "projection_hold_threshold_px": bounded_float(
            payload.get("projection_hold_threshold_px"), 0.0, 0.0, 100000.0, "projection_hold_threshold_px"
        )
        if payload.get("projection_hold_threshold_px") is not None
        else None,
        "projection_release_threshold_px": bounded_float(
            payload.get("projection_release_threshold_px"), 0.0, 0.0, 100000.0, "projection_release_threshold_px"
        )
        if payload.get("projection_release_threshold_px") is not None
        else None,
        "projection_recovery_score": bounded_float(
            payload.get("projection_recovery_score"), 0.0, 0.0, 1.0, "projection_recovery_score"
        )
        if payload.get("projection_recovery_score") is not None
        else None,
        "projection_hold_budget_frames": bounded_int(
            payload.get("projection_hold_budget_frames"), 0, 0, 100000, "projection_hold_budget_frames"
        )
        if payload.get("projection_hold_budget_frames") is not None
        else None,
        "projection_anchor_x": bounded_float(
            payload.get("projection_anchor_x"), 0.0, -100000.0, 100000.0, "projection_anchor_x"
        )
        if payload.get("projection_anchor_x") is not None
        else None,
        "projection_anchor_y": bounded_float(
            payload.get("projection_anchor_y"), 0.0, -100000.0, 100000.0, "projection_anchor_y"
        )
        if payload.get("projection_anchor_y") is not None
        else None,
        "calibration_mean_error_px": bounded_float(
            payload.get("calibration_mean_error_px"), 0.0, 0.0, 100000.0, "calibration_mean_error_px"
        )
        if payload.get("calibration_mean_error_px") is not None
        else None,
        "calibration_p95_error_px": bounded_float(
            payload.get("calibration_p95_error_px"), 0.0, 0.0, 100000.0, "calibration_p95_error_px"
        )
        if payload.get("calibration_p95_error_px") is not None
        else None,
        "calibration_sample_count": bounded_int(
            payload.get("calibration_sample_count"), 0, 0, 100000, "calibration_sample_count"
        )
        if payload.get("calibration_sample_count") is not None
        else None,
        "calibration_quality_score": bounded_float(
            payload.get("calibration_quality_score"), 0.0, 0.0, 1.0, "calibration_quality_score"
        )
        if payload.get("calibration_quality_score") is not None
        else None,
        "calibration_quality_label": str(payload.get("calibration_quality_label") or "unknown")[:80]
        if payload.get("calibration_quality_label") is not None
        else None,
        "calibration_recommended_action": str(payload.get("calibration_recommended_action") or "unknown")[:80]
        if payload.get("calibration_recommended_action") is not None
        else None,
        "projection_recommended_action": str(payload.get("projection_recommended_action") or "unknown")[:80]
        if payload.get("projection_recommended_action") is not None
        else None,
        "correction_offset_x_px": bounded_float(
            payload.get("correction_offset_x_px"), 0.0, -100000.0, 100000.0, "correction_offset_x_px"
        )
        if payload.get("correction_offset_x_px") is not None
        else None,
        "correction_offset_y_px": bounded_float(
            payload.get("correction_offset_y_px"), 0.0, -100000.0, 100000.0, "correction_offset_y_px"
        )
        if payload.get("correction_offset_y_px") is not None
        else None,
        "correction_sample_count": bounded_int(
            payload.get("correction_sample_count"), 0, 0, 100000, "correction_sample_count"
        )
        if payload.get("correction_sample_count") is not None
        else None,
        "correction_reliability_score": bounded_float(
            payload.get("correction_reliability_score"), 0.0, 0.0, 1.0, "correction_reliability_score"
        )
        if payload.get("correction_reliability_score") is not None
        else None,
        "correction_updated_at": str(payload.get("correction_updated_at") or "")[:40]
        if payload.get("correction_updated_at") is not None
        else None,
        "head_yaw_deg": bounded_float(payload.get("head_yaw_deg"), 0.0, -90.0, 90.0, "head_yaw_deg")
        if payload.get("head_yaw_deg") is not None
        else None,
        "head_pitch_deg": bounded_float(payload.get("head_pitch_deg"), 0.0, -90.0, 90.0, "head_pitch_deg")
        if payload.get("head_pitch_deg") is not None
        else None,
        "head_roll_deg": bounded_float(payload.get("head_roll_deg"), 0.0, -90.0, 90.0, "head_roll_deg")
        if payload.get("head_roll_deg") is not None
        else None,
        "head_span_width_norm": bounded_float(
            payload.get("head_span_width_norm"), 0.0, 0.0, 1.0, "head_span_width_norm"
        )
        if payload.get("head_span_width_norm") is not None
        else None,
        "head_span_height_norm": bounded_float(
            payload.get("head_span_height_norm"), 0.0, 0.0, 1.0, "head_span_height_norm"
        )
        if payload.get("head_span_height_norm") is not None
        else None,
        "framing_quality": bounded_float(
            payload.get("framing_quality"), 0.0, 0.0, 1.0, "framing_quality"
        )
        if payload.get("framing_quality") is not None
        else None,
        "framing_state": str(payload.get("framing_state") or "unknown")[:80]
        if payload.get("framing_state") is not None
        else None,
        "confidence": bounded_float(payload.get("confidence"), score, 0.0, 1.0, "confidence"),
        "stability_score": bounded_float(payload.get("stability_score"), score, 0.0, 1.0, "stability_score"),
        "jump_px": bounded_float(payload.get("jump_px"), 0.0, 0.0, 100000.0, "jump_px"),
        "jitter_px": bounded_float(payload.get("jitter_px"), 0.0, 0.0, 100000.0, "jitter_px"),
        "velocity_px_s": bounded_float(payload.get("velocity_px_s"), 0.0, 0.0, 1000000.0, "velocity_px_s"),
        "targeting_reliable": bool(payload.get("targeting_reliable", score >= 0.5)),
        "filter_mode": str(payload.get("filter_mode") or "unknown")[:80],
        "max_age_seconds": max_age,
        "observed_at": now_iso(),
    }
    return {key: value for key, value in eye.items() if value is not None}
