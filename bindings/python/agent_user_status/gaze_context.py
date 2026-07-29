#!/usr/bin/env python3
"""Shared helpers for derived gaze reliability metadata.

The helpers in this module only read the already-published local dev state and
derive short-lived gating metadata. They never access raw camera frames or
other sensor payloads.
"""

from __future__ import annotations

import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

STATE_DIR = Path(os.environ.get("AGENT_IMESSAGE_STATE_DIR", "~/.local/share/agent-imessage/state")).expanduser()
DEV_STATE_PATH = STATE_DIR / "dev_monitor_state.json"
RELIABLE_STABILITY_SCORE = 0.42
RELIABLE_CONFIDENCE_SCORE = 0.4


def read_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text).astimezone(UTC)
    except ValueError:
        return None


def as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def current_eye_state() -> dict[str, Any] | None:
    state = read_json_file(DEV_STATE_PATH)
    eye = state.get("eye")
    return eye if isinstance(eye, dict) else None


def eye_age_seconds(eye: dict[str, Any]) -> float | None:
    observed = parse_dt(str(eye.get("observed_at") or ""))
    if observed is None:
        return None
    return max(0.0, (datetime.now(UTC) - observed).total_seconds())


def gaze_targeting_reliable(eye: dict[str, Any] | None) -> bool:
    if not eye:
        return False
    state = str(eye.get("state") or "")
    filter_mode = str(eye.get("filter_mode") or "")
    if state == "no_face" or filter_mode.startswith("projection_hold"):
        return False
    stability_score = float(eye.get("stability_score", eye.get("confidence", 0.0)) or 0.0)
    confidence = float(eye.get("confidence", 0.0) or 0.0)
    fresh = False
    max_age = eye.get("max_age_seconds")
    age_seconds = eye_age_seconds(eye)
    if age_seconds is not None:
        if isinstance(max_age, (int, float)) and math.isfinite(float(max_age)):
            fresh = age_seconds <= float(max_age)
        else:
            fresh = age_seconds <= 5.0
    return (
        fresh
        and bool(eye.get("targeting_reliable", False))
        and stability_score >= RELIABLE_STABILITY_SCORE
        and confidence >= RELIABLE_CONFIDENCE_SCORE
    )


def current_gaze_context() -> dict[str, Any] | None:
    eye = current_eye_state()
    if not eye:
        return None
    age_seconds = eye_age_seconds(eye)
    stability_score = float(eye.get("stability_score", eye.get("confidence", 0.0)) or 0.0)
    confidence = float(eye.get("confidence", 0.0) or 0.0)
    context = {
        "gaze_state": str(eye.get("state") or eye.get("screen_zone") or "unknown")[:80],
        "gaze_filter_mode": str(eye.get("filter_mode") or "unknown")[:80],
        "gaze_stability_score": round(stability_score, 4),
        "gaze_confidence": round(confidence, 4),
        "gaze_targeting_reliable": gaze_targeting_reliable(eye),
        "gaze_age_seconds": round(age_seconds, 2) if age_seconds is not None else None,
        "gaze_fresh": age_seconds is not None and age_seconds <= float(eye.get("max_age_seconds", 5) or 5),
    }
    if eye.get("screen_x") is not None and eye.get("screen_y") is not None:
        context["gaze_screen_x"] = round(float(eye.get("screen_x") or 0.0), 2)
        context["gaze_screen_y"] = round(float(eye.get("screen_y") or 0.0), 2)
    if eye.get("observed_screen_x") is not None and eye.get("observed_screen_y") is not None:
        context["gaze_raw_screen_x"] = round(float(eye.get("observed_screen_x") or 0.0), 2)
        context["gaze_raw_screen_y"] = round(float(eye.get("observed_screen_y") or 0.0), 2)
    if eye.get("screen_width") is not None and eye.get("screen_height") is not None:
        context["gaze_screen_width"] = int(float(eye.get("screen_width") or 0.0))
        context["gaze_screen_height"] = int(float(eye.get("screen_height") or 0.0))
    if eye.get("calibration_mean_error_px") is not None:
        context["gaze_calibration_mean_error_px"] = round(float(eye.get("calibration_mean_error_px") or 0.0), 2)
    if eye.get("calibration_p95_error_px") is not None:
        context["gaze_calibration_p95_error_px"] = round(float(eye.get("calibration_p95_error_px") or 0.0), 2)
    if eye.get("calibration_quality_score") is not None:
        context["gaze_calibration_quality_score"] = round(float(eye.get("calibration_quality_score") or 0.0), 4)
    if eye.get("calibration_quality_label") is not None:
        context["gaze_calibration_quality_label"] = str(eye.get("calibration_quality_label") or "unknown")[:80]
    if eye.get("calibration_recommended_action") is not None:
        context["gaze_calibration_recommended_action"] = str(
            eye.get("calibration_recommended_action") or "unknown"
        )[:80]
    if eye.get("projection_recommended_action") is not None:
        context["gaze_projection_recommended_action"] = str(eye.get("projection_recommended_action") or "unknown")[:80]
    if eye.get("projection_hold_reason") is not None:
        context["gaze_projection_hold_reason"] = str(eye.get("projection_hold_reason") or "unknown")[:80]
    if eye.get("projection_hold_hint") is not None:
        context["gaze_projection_hold_hint"] = str(eye.get("projection_hold_hint") or "unknown")[:160]
    if eye.get("projection_hold_budget_frames") is not None:
        context["gaze_projection_hold_budget_frames"] = int(eye.get("projection_hold_budget_frames") or 0)
    if eye.get("projection_offscreen_px") is not None:
        context["gaze_projection_offscreen_px"] = round(float(eye.get("projection_offscreen_px") or 0.0), 2)
    if eye.get("projection_anchor_x") is not None and eye.get("projection_anchor_y") is not None:
        context["gaze_projection_anchor_x"] = round(float(eye.get("projection_anchor_x") or 0.0), 2)
        context["gaze_projection_anchor_y"] = round(float(eye.get("projection_anchor_y") or 0.0), 2)
    if eye.get("correction_offset_x_px") is not None and eye.get("correction_offset_y_px") is not None:
        context["gaze_correction_offset_x_px"] = round(float(eye.get("correction_offset_x_px") or 0.0), 2)
        context["gaze_correction_offset_y_px"] = round(float(eye.get("correction_offset_y_px") or 0.0), 2)
    if eye.get("correction_reliability_score") is not None:
        context["gaze_correction_reliability_score"] = round(float(eye.get("correction_reliability_score") or 0.0), 4)
    if eye.get("correction_sample_count") is not None:
        context["gaze_correction_sample_count"] = int(eye.get("correction_sample_count") or 0)
    if eye.get("correction_updated_at") is not None:
        context["gaze_correction_updated_at"] = str(eye.get("correction_updated_at") or "")[:40]
    return {key: value for key, value in context.items() if value is not None}


def annotate_event_with_gaze(event: dict[str, Any]) -> dict[str, Any]:
    context = current_gaze_context()
    if context:
        event.update(context)
    return event


def is_gaze_reliable_event(event: dict[str, Any]) -> bool:
    if event.get("gaze_targeting_reliable") is False:
        return False
    if event.get("gaze_fresh") is False:
        return False
    if str(event.get("gaze_state") or "") == "no_face":
        return False
    if str(event.get("gaze_filter_mode") or "").startswith("projection_hold"):
        return False
    return True
