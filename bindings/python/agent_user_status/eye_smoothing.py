"""Adaptive smoothing for derived gaze coordinates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


class ScreenLike(Protocol):
    @property
    def width(self) -> int: ...

    @property
    def height(self) -> int: ...


def projection_error(point_xy: tuple[float, float], screen: ScreenLike) -> float:
    """Euclidean distance from *point_xy* to the nearest on-screen coordinate.

    Points inside the screen bounds return 0.0. Off-screen points return the
    distance to the nearest edge or corner of the screen rectangle.

    Parameters
    ----------
    point_xy:
        The (x, y) raw projection point in screen-pixel space.
    screen:
        An object with ``.width`` and ``.height`` int attributes.

    Returns
    -------
    float
        Non-negative distance in pixels.

    Examples
    --------
    >>> from dataclasses import dataclass
    >>> @dataclass(frozen=True)
    ... class Screen:
    ...     width: int
    ...     height: int

    A point well inside the screen yields zero error:

    >>> projection_error((960.0, 540.0), Screen(1920, 1080))
    0.0

    A point to the left of the screen reports the horizontal offset:

    >>> projection_error((-50.0, 540.0), Screen(1920, 1080))
    50.0

    A point beyond the bottom-right corner reports the diagonal distance:

    >>> error = projection_error((2000.0, 1200.0), Screen(1920, 1080))
    >>> round(error, 2)
    145.61
    """
    clamped_x = max(0.0, min(float(screen.width - 1), point_xy[0]))
    clamped_y = max(0.0, min(float(screen.height - 1), point_xy[1]))
    return math.hypot(point_xy[0] - clamped_x, point_xy[1] - clamped_y)


@dataclass
class OneEuroAxis:
    min_cutoff: float
    beta: float
    derivative_cutoff: float
    value: float | None = None
    derivative: float = 0.0
    last_time: float | None = None

    def update(self, value: float, timestamp: float) -> float:
        if self.value is None or self.last_time is None:
            self.value = value
            self.last_time = timestamp
            return value

        dt = max(1e-3, timestamp - self.last_time)
        raw_derivative = (value - self.value) / dt
        self.derivative = self._smooth(raw_derivative, self.derivative, self.derivative_cutoff, dt)
        cutoff = self.min_cutoff + self.beta * abs(self.derivative)
        self.value = self._smooth(value, self.value, cutoff, dt)
        self.last_time = timestamp
        return self.value

    def _smooth(self, value: float, previous: float, cutoff: float, dt: float) -> float:
        alpha = self._alpha(cutoff, dt)
        return previous + alpha * (value - previous)

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * max(1e-6, cutoff))
        return 1.0 / (1.0 + tau / dt)


class AdaptiveGazeSmoother:
    """Two-axis One Euro filter with bounded jumps and stability diagnostics."""

    def __init__(
        self,
        min_cutoff: float = 1.15,
        beta: float = 0.012,
        derivative_cutoff: float = 1.0,
        max_jump_px: float = 620.0,
    ) -> None:
        self.x = OneEuroAxis(min_cutoff, beta, derivative_cutoff)
        self.y = OneEuroAxis(min_cutoff, beta, derivative_cutoff)
        self.max_jump_px = max_jump_px
        self.jump_ema = 0.0
        self.residual_ema = 0.0
        self.velocity_ema = 0.0
        self.last_snapshot = {
            "stability_score": 1.0,
            "jump_px": 0.0,
            "jitter_px": 0.0,
            "velocity_px_s": 0.0,
            "targeting_reliable": True,
            "filter_mode": "bootstrap",
        }

    def update(self, point: tuple[float, float], timestamp: float, confidence: float = 1.0) -> tuple[float, float]:
        if self.x.value is None or self.y.value is None or self.x.last_time is None:
            initial = (self.x.update(point[0], timestamp), self.y.update(point[1], timestamp))
            self.last_snapshot = {
                "stability_score": max(0.0, min(1.0, confidence)),
                "jump_px": 0.0,
                "jitter_px": 0.0,
                "velocity_px_s": 0.0,
                "targeting_reliable": confidence >= 0.45,
                "filter_mode": "bootstrap",
            }
            return initial

        current = (self.x.value, self.y.value)
        dt = max(1e-3, timestamp - self.x.last_time)
        dx = point[0] - current[0]
        dy = point[1] - current[1]
        jump_px = math.hypot(dx, dy)
        velocity_px_s = jump_px / dt
        deadband_px = max(8.0, 18.0 - confidence * 7.0)
        clamp_px = max(85.0, self.max_jump_px * (0.2 + confidence * 0.8))
        mode = "tracking"

        if jump_px <= deadband_px:
            point = current
            mode = "deadband_hold"
        elif confidence < 0.34 and jump_px > min(clamp_px * 0.45, 220.0):
            point = current
            mode = "confidence_hold"
        elif jump_px > clamp_px:
            scale = clamp_px / jump_px
            point = (current[0] + dx * scale, current[1] + dy * scale)
            mode = "jump_clamp"

        filtered = (self.x.update(point[0], timestamp), self.y.update(point[1], timestamp))
        residual_px = math.hypot(point[0] - filtered[0], point[1] - filtered[1])
        self.jump_ema = self.jump_ema * 0.82 + jump_px * 0.18
        self.residual_ema = self.residual_ema * 0.82 + residual_px * 0.18
        self.velocity_ema = self.velocity_ema * 0.84 + velocity_px_s * 0.16
        stability_score = max(
            0.0,
            min(
                1.0,
                0.52 * confidence
                + 0.24 * (1.0 - min(1.0, self.jump_ema / 180.0))
                + 0.24 * (1.0 - min(1.0, self.residual_ema / 90.0)),
            ),
        )
        targeting_reliable = confidence >= 0.4 and stability_score >= 0.42 and self.jump_ema <= 220.0
        self.last_snapshot = {
            "stability_score": round(stability_score, 4),
            "jump_px": round(jump_px, 2),
            "jitter_px": round(self.residual_ema, 2),
            "velocity_px_s": round(self.velocity_ema, 2),
            "targeting_reliable": targeting_reliable,
            "filter_mode": mode,
        }
        return filtered

    def reset(self, point: tuple[float, float], timestamp: float, confidence: float = 1.0) -> None:
        self.x.value = point[0]
        self.y.value = point[1]
        self.x.derivative = 0.0
        self.y.derivative = 0.0
        self.x.last_time = timestamp
        self.y.last_time = timestamp
        self.jump_ema = 0.0
        self.residual_ema = 0.0
        self.velocity_ema = 0.0
        self.last_snapshot = {
            "stability_score": max(0.0, min(1.0, confidence)),
            "jump_px": 0.0,
            "jitter_px": 0.0,
            "velocity_px_s": 0.0,
            "targeting_reliable": confidence >= 0.45,
            "filter_mode": "reseed",
        }

    def snapshot(self) -> dict[str, float | bool | str]:
        return dict(self.last_snapshot)

    def current(self) -> tuple[float, float] | None:
        if self.x.value is None or self.y.value is None:
            return None
        return (self.x.value, self.y.value)
