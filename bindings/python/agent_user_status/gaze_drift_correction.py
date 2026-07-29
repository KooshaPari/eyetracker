"""Learned drift correction for derived gaze coordinates."""

from __future__ import annotations

import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

STATE_DIR = Path(os.environ.get("AGENT_IMESSAGE_STATE_DIR", "~/.local/share/agent-imessage/state")).expanduser()
DRIFT_CORRECTION_PATH = STATE_DIR / "gaze_drift_correction.json"
INTERACTION_ROLES = {
    "terminal",
    "coding_terminal",
    "agent_terminal",
    "multi_agent_terminal",
    "editor",
    "gui_agent",
    "gui_chat",
}


class ScreenLike(Protocol):
    @property
    def width(self) -> int: ...

    @property
    def height(self) -> int: ...


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def read_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def write_json_file(path: Path, payload: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _screen_point(event: dict[str, Any]) -> tuple[float, float] | None:
    x = event.get("screen_x")
    y = event.get("screen_y")
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return (float(x), float(y))
    return None


def _gaze_point(event: dict[str, Any]) -> tuple[float, float] | None:
    raw_x = event.get("gaze_raw_screen_x")
    raw_y = event.get("gaze_raw_screen_y")
    if isinstance(raw_x, (int, float)) and isinstance(raw_y, (int, float)):
        return (float(raw_x), float(raw_y))
    x = event.get("gaze_screen_x")
    y = event.get("gaze_screen_y")
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return (float(x), float(y))
    return None


def _confidence(event: dict[str, Any]) -> float:
    for key in ("gaze_confidence", "confidence", "score"):
        value = event.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return clamp(float(value))
    return 0.5


def _stability(event: dict[str, Any]) -> float:
    value = event.get("gaze_stability_score")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return clamp(float(value))
    return _confidence(event)


def _window_role_weight(event: dict[str, Any], kind: str) -> float:
    role = str(event.get("window_role") or "").lower()
    if kind == "explicit_alignment":
        if role in INTERACTION_ROLES:
            return 1.2
        if role in {"browser", "communication", "media"}:
            return 0.95
        return 1.05
    if role in INTERACTION_ROLES:
        return 1.15
    if role in {"browser", "communication", "media"}:
        return 0.68
    if role == "unknown":
        return 0.9
    return 0.82


def learn_drift_correction(events: list[dict[str, Any]], screen: ScreenLike) -> dict[str, Any] | None:
    candidates: list[tuple[float, float, float, str]] = []
    for event in reversed(events[-180:]):
        if not isinstance(event, dict):
            continue
        if not bool(event.get("harmony_hint", False)) and str(event.get("kind") or "") != "explicit_alignment":
            continue
        kind = str(event.get("kind") or "")
        if kind not in {"cursor_click", "cursor_target", "explicit_alignment"}:
            continue
        if kind == "explicit_alignment" and event.get("gaze_fresh") is False:
            continue
        if event.get("learnable") is False and kind != "explicit_alignment":
            continue

        gaze = _gaze_point(event)
        target = _screen_point(event)
        if gaze is None or target is None:
            continue

        offset_x = target[0] - gaze[0]
        offset_y = target[1] - gaze[1]
        distance = math.hypot(offset_x, offset_y)
        max_distance = max(220.0, min(float(screen.width), float(screen.height)) * 0.25)
        if distance > max_distance and kind != "explicit_alignment":
            continue

        kind_weight = {
            "explicit_alignment": 1.35,
            "cursor_click": 1.0,
            "cursor_target": 0.85,
        }.get(kind, 0.7)
        weight = (
            kind_weight
            * _window_role_weight(event, kind)
            * (0.35 + 0.65 * _confidence(event))
            * (0.35 + 0.65 * _stability(event))
        )
        weight *= max(0.2, float(event.get("score", 0.5) or 0.5))
        candidates.append((offset_x, offset_y, weight, kind))

    if len(candidates) < 2:
        return None

    weight_sum = sum(item[2] for item in candidates)
    if weight_sum <= 0:
        return None

    offset_x = sum(item[0] * item[2] for item in candidates) / weight_sum
    offset_y = sum(item[1] * item[2] for item in candidates) / weight_sum
    residuals = [math.hypot(item[0] - offset_x, item[1] - offset_y) for item in candidates]
    spread = sum(residuals) / len(residuals)
    distinct_kinds = sorted({item[3] for item in candidates})
    reliability = clamp(
        0.22
        + min(0.44, len(candidates) / 10.0)
        + min(0.2, weight_sum / 10.0)
        + max(0.0, 0.14 * (1.0 - min(1.0, spread / 160.0)))
    )

    return {
        "version": 1,
        "kind": "gaze_drift_correction",
        "created_at": now_iso(),
        "screen_width": screen.width,
        "screen_height": screen.height,
        "screen_x_offset_px": round(offset_x, 2),
        "screen_y_offset_px": round(offset_y, 2),
        "sample_count": len(candidates),
        "weighted_sample_count": round(weight_sum, 4),
        "reliable_samples": len(candidates),
        "spread_px": round(spread, 2),
        "reliability_score": round(reliability, 4),
        "source_kinds": distinct_kinds,
    }


def persist_drift_correction(
    events: list[dict[str, Any]],
    screen: ScreenLike,
    path: Path = DRIFT_CORRECTION_PATH,
) -> dict[str, Any] | None:
    correction = learn_drift_correction(events, screen)
    if correction is None:
        return None
    write_json_file(path, correction)
    return correction


def load_drift_correction(path: Path = DRIFT_CORRECTION_PATH) -> dict[str, Any] | None:
    correction = read_json_file(path)
    if not correction:
        return None
    if correction.get("kind") != "gaze_drift_correction":
        return None
    return correction


def apply_drift_correction(
    point_xy: tuple[float, float],
    screen: ScreenLike,
    correction: dict[str, Any] | None,
) -> tuple[float, float]:
    if not correction:
        return point_xy
    reliability = correction.get("reliability_score", 1.0)
    if not isinstance(reliability, (int, float)) or float(reliability) < 0.35:
        return point_xy
    offset_x = correction.get("screen_x_offset_px", 0.0)
    offset_y = correction.get("screen_y_offset_px", 0.0)
    if not isinstance(offset_x, (int, float)) or not isinstance(offset_y, (int, float)):
        return point_xy
    x = max(0.0, min(float(screen.width - 1), point_xy[0] + float(offset_x)))
    y = max(0.0, min(float(screen.height - 1), point_xy[1] + float(offset_y)))
    return (x, y)
