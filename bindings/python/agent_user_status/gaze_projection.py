"""Projection-hold recovery and stable sampling helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from agent_user_status.eye_smoothing import projection_error


class ScreenLike(Protocol):
    @property
    def width(self) -> int: ...

    @property
    def height(self) -> int: ...


@dataclass(frozen=True)
class ProjectionDecision:
    publish_point: tuple[float, float]
    smooth_point: tuple[float, float] | None
    anchor_point: tuple[float, float] | None
    mode: str
    hold_active: bool
    hold_reason: str
    hold_hint: str
    should_reset: bool
    projection_error_px: float
    projection_offscreen_px: float
    hold_threshold_px: float
    release_threshold_px: float
    recovery_score: float
    stable_frames: int
    hold_budget_frames: int
    targeting_reliable: bool


class StableSampleGate:
    def __init__(self, min_confidence: float = 0.35, min_frames: int = 2) -> None:
        self.min_confidence = min_confidence
        self.min_frames = max(1, min_frames)
        self.frames = 0
        self.last_confidence: float | None = None
        self.max_confidence_delta = 0.16

    def update(self, confidence: float) -> bool:
        if confidence >= self.min_confidence:
            if (
                self.last_confidence is not None
                and self.frames > 0
                and abs(confidence - self.last_confidence) > self.max_confidence_delta
            ):
                self.frames = 0
            else:
                self.frames += 1
            self.last_confidence = confidence
        else:
            self.frames = 0
            self.last_confidence = confidence
        return self.ready()

    def ready(self) -> bool:
        return self.frames >= self.min_frames

    def reset(self) -> None:
        self.frames = 0
        self.last_confidence = None


def clamp_point(point: tuple[float, float], screen: ScreenLike) -> tuple[float, float]:
    return (
        max(0.0, min(float(screen.width - 1), point[0])),
        max(0.0, min(float(screen.height - 1), point[1])),
    )


def in_bounds(point: tuple[float, float], screen: ScreenLike) -> bool:
    return 0.0 <= point[0] <= float(screen.width - 1) and 0.0 <= point[1] <= float(screen.height - 1)


def screen_center(screen: ScreenLike) -> tuple[float, float]:
    return (float(screen.width - 1) / 2.0, float(screen.height - 1) / 2.0)


class ProjectionHoldGate:
    def __init__(
        self,
        hold_threshold_px: float,
        release_threshold_px: float,
        calibration_quality_score: float = 0.0,
        calibration_recommended_action: str = "monitor",
        min_confidence: float = 0.36,
        enter_frames: int = 2,
        release_frames: int = 3,
    ) -> None:
        self.hold_threshold_px = hold_threshold_px
        self.release_threshold_px = release_threshold_px
        self.calibration_quality_score = max(0.0, min(1.0, calibration_quality_score))
        self.calibration_recommended_action = calibration_recommended_action
        self.min_confidence = min_confidence
        self.enter_frames = max(1, enter_frames)
        self.release_frames = max(1, release_frames)
        self.hold_frames = 0
        self.recovery_frames = 0
        self.hold_active = False
        self.last_trusted_point: tuple[float, float] | None = None
        self.last_error_px = 0.0

    def _quality_label(self) -> str:
        if self.calibration_quality_score <= 0.35:
            return "poor"
        if self.calibration_quality_score <= 0.55:
            return "fragile"
        if self.calibration_quality_score <= 0.78:
            return "usable"
        return "excellent"

    def _hold_budget_frames(self) -> int:
        if self.calibration_quality_score <= 0.35:
            return 2
        if self.calibration_quality_score <= 0.55:
            return 3
        return 0

    def _effective_release_frames(self) -> int:
        if self.calibration_quality_score <= 0.35:
            return max(self.release_frames, 4)
        if self.calibration_quality_score <= 0.55:
            return max(self.release_frames, 3)
        return self.release_frames

    def _anchor_point(
        self,
        fallback_point: tuple[float, float] | None,
        screen: ScreenLike,
    ) -> tuple[float, float]:
        if self.last_trusted_point is not None:
            return clamp_point(self.last_trusted_point, screen)
        if fallback_point is not None and in_bounds(fallback_point, screen):
            return clamp_point(fallback_point, screen)
        return screen_center(screen)

    def _hold_reason(
        self,
        raw_point: tuple[float, float],
        screen: ScreenLike,
        error_px: float,
        confidence: float,
        stability_score: float,
        recovery_signal: bool,
    ) -> str:
        if recovery_signal:
            return "recovering"
        if self.calibration_quality_score <= 0.35:
            return "poor_calibration_fit"
        if not in_bounds(raw_point, screen):
            return "offscreen_jump"
        if confidence < self.min_confidence:
            return "low_confidence"
        if stability_score < 0.32:
            return "unstable_projection"
        if error_px >= self.hold_threshold_px:
            return "projection_outlier"
        return "projection_pending"

    def _recovery_score(self, error_px: float, confidence: float, stability_score: float) -> float:
        if self.release_threshold_px >= self.hold_threshold_px:
            proximity = 1.0 if error_px <= self.release_threshold_px else 0.0
        else:
            proximity = 1.0 - min(
                1.0,
                max(0.0, (error_px - self.release_threshold_px) / (self.hold_threshold_px - self.release_threshold_px)),
            )
        quality = max(
            0.0,
            min(
                1.0,
                0.45 * (confidence / max(self.min_confidence, 1e-6))
                + 0.35 * stability_score
                + 0.20 * self.calibration_quality_score,
            ),
        )
        return max(0.0, min(1.0, 0.65 * proximity + 0.35 * quality))

    def _hold_hint(
        self,
        reason: str,
        recovery_signal: bool,
        effective_release_frames: int,
        budget_frames: int,
    ) -> str:
        if recovery_signal:
            return (
                f"projection is back inside screen bounds; release after one more stable frame "
                f"({self.release_threshold_px:.0f}px release / {self.hold_threshold_px:.0f}px hold)"
            )
        if reason == "poor_calibration_fit":
            return (
                f"poor calibration fit ({self._quality_label()}); recalibrate soon; "
                f"degraded hold stays active until recovery is sustained for {effective_release_frames} frames "
                f"({self.release_threshold_px:.0f}px release / {self.hold_threshold_px:.0f}px hold)"
            )
        if reason == "offscreen_jump":
            return (
                f"projection is offscreen; move gaze back inside the screen and keep it steady for "
                f"{effective_release_frames} frames "
                f"({self.release_threshold_px:.0f}px release / {self.hold_threshold_px:.0f}px hold)"
            )
        if reason == "low_confidence":
            return (
                f"camera confidence is too low; hold keeps the last trusted point and will release after "
                f"{effective_release_frames} stable frames "
                f"({self.release_threshold_px:.0f}px release / {self.hold_threshold_px:.0f}px hold)"
            )
        if reason == "unstable_projection":
            return (
                "projection is unstable; slow down head motion and keep gaze steady for "
                f"{effective_release_frames} frames "
                f"({self.release_threshold_px:.0f}px release / {self.hold_threshold_px:.0f}px hold)"
            )
        if reason == "projection_outlier":
            return (
                f"projection jumped past the hold threshold ({self.hold_threshold_px:.0f}px); "
                f"recalibrate if this repeats"
            )
        if budget_frames > 0:
            return (
                f"hold budget is {budget_frames} frames before degraded release; "
                "recalibrate if the fit stays poor "
                f"({self.release_threshold_px:.0f}px release / {self.hold_threshold_px:.0f}px hold)"
            )
        return "freeze at the last trusted point until the projection returns in bounds"

    def update(
        self,
        raw_point: tuple[float, float],
        screen: ScreenLike,
        confidence: float,
        stability_score: float,
        fallback_point: tuple[float, float] | None = None,
    ) -> ProjectionDecision:
        error_px = projection_error(raw_point, screen)
        raw_in_bounds = in_bounds(raw_point, screen)
        improving = error_px <= self.last_error_px * 0.86 if self.last_error_px > 0 else False
        effective_release_frames = self._effective_release_frames()
        hold_budget_frames = self._hold_budget_frames()
        recovery_signal = raw_in_bounds and (
            error_px <= self.release_threshold_px
            or (improving and error_px <= self.hold_threshold_px * 1.15)
            or (
                confidence >= self.min_confidence + 0.06
                and stability_score >= 0.33
                and error_px <= self.hold_threshold_px
            )
        )
        self.last_error_px = error_px
        anchor_point = self._anchor_point(fallback_point, screen)
        hold_reason = self._hold_reason(raw_point, screen, error_px, confidence, stability_score, recovery_signal)

        if self.hold_active:
            self.hold_frames += 1
            if recovery_signal:
                self.recovery_frames += 1
            else:
                self.recovery_frames = 0
            if self.recovery_frames >= effective_release_frames:
                self.hold_active = False
                self.hold_frames = 0
                self.recovery_frames = 0
                self.last_trusted_point = raw_point if raw_in_bounds else anchor_point
                return ProjectionDecision(
                    publish_point=raw_point,
                    smooth_point=raw_point,
                    anchor_point=anchor_point,
                    mode="projection_hold_recovering",
                    hold_active=False,
                    hold_reason="recovered",
                    hold_hint="resume tracking; projection is back inside screen bounds",
                    should_reset=True,
                    projection_error_px=round(error_px, 2),
                    projection_offscreen_px=round(error_px if not raw_in_bounds else 0.0, 2),
                    hold_threshold_px=round(self.hold_threshold_px, 2),
                    release_threshold_px=round(self.release_threshold_px, 2),
                    recovery_score=round(self._recovery_score(error_px, confidence, stability_score), 4),
                    stable_frames=self.recovery_frames,
                    hold_budget_frames=hold_budget_frames,
                    targeting_reliable=confidence >= self.min_confidence and stability_score >= 0.38 and raw_in_bounds,
                )
            if hold_budget_frames > 0 and self.hold_frames >= hold_budget_frames:
                self.hold_frames = hold_budget_frames
                return ProjectionDecision(
                    publish_point=anchor_point,
                    smooth_point=None,
                    anchor_point=anchor_point,
                    mode="projection_hold_degraded",
                    hold_active=True,
                    hold_reason=hold_reason,
                    hold_hint=(
                        f"poor calibration fit ({self._quality_label()}); degraded hold remains active until "
                        f"recovery is sustained for {effective_release_frames} frames; "
                        f"{self.calibration_recommended_action.replace('_', ' ')}"
                    ),
                    should_reset=False,
                    projection_error_px=round(error_px, 2),
                    projection_offscreen_px=round(error_px if not raw_in_bounds else 0.0, 2),
                    hold_threshold_px=round(self.hold_threshold_px, 2),
                    release_threshold_px=round(self.release_threshold_px, 2),
                    recovery_score=round(self._recovery_score(error_px, confidence, stability_score), 4),
                    stable_frames=self.hold_frames,
                    hold_budget_frames=hold_budget_frames,
                    targeting_reliable=False,
                )

            anchor = anchor_point
            return ProjectionDecision(
                publish_point=anchor,
                smooth_point=None,
                anchor_point=anchor,
                mode="projection_hold",
                hold_active=True,
                hold_reason=hold_reason,
                hold_hint=self._hold_hint(hold_reason, recovery_signal, effective_release_frames, hold_budget_frames),
                should_reset=False,
                projection_error_px=round(error_px, 2),
                projection_offscreen_px=round(error_px if not raw_in_bounds else 0.0, 2),
                hold_threshold_px=round(self.hold_threshold_px, 2),
                release_threshold_px=round(self.release_threshold_px, 2),
                recovery_score=round(
                    max(
                        self.recovery_frames / effective_release_frames,
                        self._recovery_score(error_px, confidence, stability_score),
                    ),
                    4,
                ),
                stable_frames=self.recovery_frames,
                hold_budget_frames=hold_budget_frames,
                targeting_reliable=False,
            )

        if error_px <= self.hold_threshold_px and confidence >= self.min_confidence and stability_score >= 0.28:
            self.hold_frames = 0
            self.last_trusted_point = raw_point
            return ProjectionDecision(
                publish_point=raw_point,
                smooth_point=raw_point,
                anchor_point=raw_point,
                mode="tracking",
                hold_active=False,
                hold_reason="tracking",
                hold_hint="tracking is stable",
                should_reset=False,
                projection_error_px=round(error_px, 2),
                projection_offscreen_px=0.0,
                hold_threshold_px=round(self.hold_threshold_px, 2),
                release_threshold_px=round(self.release_threshold_px, 2),
                recovery_score=0.0,
                stable_frames=0,
                hold_budget_frames=hold_budget_frames,
                targeting_reliable=True,
            )

        self.hold_frames += 1
        self.recovery_frames = 0
        if raw_in_bounds:
            self.last_trusted_point = raw_point
        if self.hold_frames < self.enter_frames:
            clamped = anchor_point if not raw_in_bounds else clamp_point(raw_point, screen)
            return ProjectionDecision(
                publish_point=clamped,
                smooth_point=clamped,
                anchor_point=anchor_point,
                mode="projection_pending",
                hold_active=False,
                hold_reason=hold_reason,
                hold_hint="await one more stable frame before freezing the point",
                should_reset=False,
                projection_error_px=round(error_px, 2),
                projection_offscreen_px=round(error_px if not raw_in_bounds else 0.0, 2),
                hold_threshold_px=round(self.hold_threshold_px, 2),
                release_threshold_px=round(self.release_threshold_px, 2),
                recovery_score=round(self._recovery_score(error_px, confidence, stability_score), 4),
                stable_frames=self.hold_frames,
                hold_budget_frames=hold_budget_frames,
                targeting_reliable=raw_in_bounds and confidence >= self.min_confidence and stability_score >= 0.28,
            )

        self.hold_active = True
        anchor = anchor_point
        return ProjectionDecision(
            publish_point=anchor,
            smooth_point=None,
            anchor_point=anchor,
            mode="projection_hold",
            hold_active=True,
            hold_reason=hold_reason,
            hold_hint="freeze at the last trusted point until the projection returns in bounds",
            should_reset=False,
            projection_error_px=round(error_px, 2),
            projection_offscreen_px=round(error_px if not raw_in_bounds else 0.0, 2),
            hold_threshold_px=round(self.hold_threshold_px, 2),
            release_threshold_px=round(self.release_threshold_px, 2),
            recovery_score=round(self._recovery_score(error_px, confidence, stability_score), 4),
            stable_frames=self.hold_frames,
            hold_budget_frames=hold_budget_frames,
            targeting_reliable=False,
        )
