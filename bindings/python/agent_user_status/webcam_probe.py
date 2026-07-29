"""Derived-only webcam acquisition diagnostics."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from agent_user_status.webcam_support import FaceTracker, GazeSample, frame_sample


@dataclass
class PresenceProbeSummary:
    ok: bool
    camera: int
    requested_width: int
    requested_height: int
    frame_width: int | None
    frame_height: int | None
    frames_requested: int
    frames_read: int
    frames_unavailable: int
    presence_samples: int
    missing_presence_samples: int
    low_confidence_samples: int
    max_sample_confidence: float
    mean_sample_confidence: float
    diagnosis: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "camera": self.camera,
            "requested_width": self.requested_width,
            "requested_height": self.requested_height,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "frames_requested": self.frames_requested,
            "frames_read": self.frames_read,
            "frames_unavailable": self.frames_unavailable,
            "presence_samples": self.presence_samples,
            "missing_presence_samples": self.missing_presence_samples,
            "low_confidence_samples": self.low_confidence_samples,
            "max_sample_confidence": round(self.max_sample_confidence, 4),
            "mean_sample_confidence": round(self.mean_sample_confidence, 4),
            "diagnosis": self.diagnosis,
        }


def summarize_presence_probe(
    *,
    camera: int,
    requested_width: int,
    requested_height: int,
    frame_width: int | None,
    frame_height: int | None,
    frames_requested: int,
    frames_read: int,
    confidences: list[float],
    min_sample_confidence: float,
) -> PresenceProbeSummary:
    presence_samples = len(confidences)
    frames_unavailable = max(0, frames_requested - frames_read)
    low_confidence_samples = sum(1 for value in confidences if value < min_sample_confidence)
    accepted_confidences = [value for value in confidences if value >= min_sample_confidence]
    max_sample_confidence = max(confidences) if confidences else 0.0
    mean_sample_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    missing_presence_samples = max(0, frames_read - presence_samples)

    if frames_read == 0:
        diagnosis = "camera_no_frames"
    elif presence_samples == 0:
        diagnosis = "presence_not_detected"
    elif not accepted_confidences:
        diagnosis = "presence_low_confidence"
    else:
        diagnosis = "presence_detected"

    return PresenceProbeSummary(
        ok=diagnosis == "presence_detected",
        camera=camera,
        requested_width=requested_width,
        requested_height=requested_height,
        frame_width=frame_width,
        frame_height=frame_height,
        frames_requested=frames_requested,
        frames_read=frames_read,
        frames_unavailable=frames_unavailable,
        presence_samples=presence_samples,
        missing_presence_samples=missing_presence_samples,
        low_confidence_samples=low_confidence_samples,
        max_sample_confidence=max_sample_confidence,
        mean_sample_confidence=mean_sample_confidence,
        diagnosis=diagnosis,
    )


def probe_presence(
    *,
    cap: Any,
    tracker: FaceTracker,
    camera: int,
    requested_width: int,
    requested_height: int,
    frames: int,
    warmup_frames: int,
    min_sample_confidence: float,
    frame_delay_seconds: float,
) -> PresenceProbeSummary:
    frame_width: int | None = None
    frame_height: int | None = None
    frames_read = 0
    confidences: list[float] = []

    for index in range(max(0, warmup_frames) + max(1, frames)):
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        if frame_width is None:
            frame_width = int(frame.shape[1])
            frame_height = int(frame.shape[0])
        if index < warmup_frames:
            continue
        frames_read += 1
        sample: GazeSample | None = frame_sample(tracker, frame)
        if sample is not None:
            confidences.append(float(sample.confidence))
        if frame_delay_seconds > 0:
            time.sleep(frame_delay_seconds)

    return summarize_presence_probe(
        camera=camera,
        requested_width=requested_width,
        requested_height=requested_height,
        frame_width=frame_width,
        frame_height=frame_height,
        frames_requested=max(1, frames),
        frames_read=frames_read,
        confidences=confidences,
        min_sample_confidence=min_sample_confidence,
    )
