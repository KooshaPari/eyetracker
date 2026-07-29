"""Privacy-safe gaze calibration evaluation counters."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

REPEATED_GAZE_SAMPLE = "repeated_gaze_sample"
STUCK_GAZE_SAMPLE = "stuck_gaze_sample"


@dataclass
class GazeSampleStuckDetector:
    """Detect repeated derived gaze coordinates without retaining a raw stream."""

    repeated_threshold: int = 2
    stuck_threshold: int = 4
    coordinate_precision_px: float = 1.0
    _last_key: tuple[int, int] | None = None
    _repeat_count: int = 0
    repeated_total: int = 0
    stuck_total: int = 0

    def inspect(self, observed: tuple[float, float]) -> str | None:
        key = self._key(observed)
        if key != self._last_key:
            self._last_key = key
            self._repeat_count = 1
            return None

        self._repeat_count += 1
        if self._repeat_count >= self.stuck_threshold:
            self.stuck_total += 1
            return STUCK_GAZE_SAMPLE
        if self._repeat_count >= self.repeated_threshold:
            self.repeated_total += 1
            return REPEATED_GAZE_SAMPLE
        return None

    def reset(self) -> None:
        self._last_key = None
        self._repeat_count = 0

    def summary(self) -> dict[str, int]:
        return {
            "repeated_gaze_sample_count": self.repeated_total,
            "stuck_gaze_sample_count": self.stuck_total,
        }

    def _key(self, observed: tuple[float, float]) -> tuple[int, int]:
        precision = max(self.coordinate_precision_px, 0.1)
        return (round(observed[0] / precision), round(observed[1] / precision))


@dataclass
class TargetEvaluation:
    target_index: int
    target_x: int
    target_y: int
    accepted: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    errors: list[float] = field(default_factory=list)

    def accept(self, observed: tuple[float, float]) -> None:
        self.accepted += 1
        self.errors.append(math.hypot(observed[0] - self.target_x, observed[1] - self.target_y))

    def reject(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1

    def summary(self) -> dict[str, Any]:
        return {
            "target_index": self.target_index,
            "target_x": self.target_x,
            "target_y": self.target_y,
            "accepted": self.accepted,
            "rejected": dict(sorted(self.rejected.items())),
            "mean_error_px": round(sum(self.errors) / len(self.errors), 2) if self.errors else None,
            "max_error_px": round(max(self.errors), 2) if self.errors else None,
        }


@dataclass
class EvaluationCounters:
    targets: list[TargetEvaluation] = field(default_factory=list)
    stuck_detector: GazeSampleStuckDetector = field(default_factory=GazeSampleStuckDetector)

    def begin_target(self, target_index: int, target_x: int, target_y: int) -> TargetEvaluation:
        target = TargetEvaluation(target_index=target_index, target_x=target_x, target_y=target_y)
        self.targets.append(target)
        return target

    def inspect_observed_sample(self, observed: tuple[float, float]) -> str | None:
        return self.stuck_detector.inspect(observed)

    def reset_sample_detector(self) -> None:
        self.stuck_detector.reset()

    @property
    def errors(self) -> list[float]:
        return [error for target in self.targets for error in target.errors]

    def rejected_totals(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for target in self.targets:
            for reason, count in target.rejected.items():
                totals[reason] = totals.get(reason, 0) + count
        return dict(sorted(totals.items()))

    def summary(self, hold_threshold_px: float) -> dict[str, Any]:
        errors = self.errors
        hold_count = sum(1 for error in errors if error > hold_threshold_px)
        return {
            "sample_count": len(errors),
            "accepted_total": len(errors),
            "rejected_total": sum(self.rejected_totals().values()),
            "rejected_by_reason": self.rejected_totals(),
            **self.stuck_detector.summary(),
            "projection_hold_candidate_count": hold_count,
            "projection_hold_rate": round(hold_count / len(errors), 4) if errors else 0.0,
            "targets": [target.summary() for target in self.targets],
        }
