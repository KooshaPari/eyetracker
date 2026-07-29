"""Publish derived eye state to the local status backend."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol


class PublishError(RuntimeError):
    pass


class ScreenLike(Protocol):
    width: int
    height: int


@dataclass(frozen=True)
class EyePublishConfig:
    statusd_url: str
    timeout_seconds: float = 1.2


def post_eye(
    point_xy: tuple[float, float],
    screen: ScreenLike,
    confidence: float,
    max_age: int,
    config: EyePublishConfig,
    state: str = "looking_at_screen",
    extra: dict[str, Any] | None = None,
) -> None:
    x = max(0.0, min(float(screen.width - 1), point_xy[0]))
    y = max(0.0, min(float(screen.height - 1), point_xy[1]))
    payload = {
        "screen_x": x,
        "screen_y": y,
        "screen_width": screen.width,
        "screen_height": screen.height,
        "score": max(0.0, min(1.0, confidence)),
        "confidence": max(0.0, min(1.0, confidence)),
        "state": state,
        "max_age_seconds": max_age,
    }
    if extra:
        payload.update(extra)
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{config.statusd_url.rstrip('/')}/dev/eye",
        data=data,
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=config.timeout_seconds) as response:
            response.read()
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        raise PublishError(f"statusd publish failed: {exc}") from exc
