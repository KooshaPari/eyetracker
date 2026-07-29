"""Live derived-gaze publishing loop."""

from __future__ import annotations

import signal
import time
from pathlib import Path
from typing import Any

from agent_user_status.eye_publish import EyePublishConfig, PublishError, post_eye
from agent_user_status.eye_smoothing import AdaptiveGazeSmoother
from agent_user_status.gaze_calibration import load_calibration_quality, predict, projection_thresholds
from agent_user_status.gaze_drift_correction import apply_drift_correction, load_drift_correction
from agent_user_status.gaze_projection import ProjectionHoldGate
from agent_user_status.webcam_support import (
    FaceTrackerThresholds,
    frame_sample,
    open_camera,
    screen_size,
)


def load_calibration(path: Path) -> dict[str, Any]:
    import json

    from agent_user_status.webcam_support import TrackerError

    if not path.exists():
        raise TrackerError(f"calibration missing: run `agent-user-status-webcam-eye-tracker calibrate` first ({path})")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != "mediapipe_iris_regression":
        raise TrackerError(f"unsupported calibration kind in {path}")
    return payload


def run_tracker(
    args: Any,
    *,
    calibration_path: Path,
    statusd_url: str,
    thresholds: FaceTrackerThresholds,
) -> int:
    from agent_user_status.webcam_support import create_face_mesh

    screen = screen_size()
    calibration = load_calibration(calibration_path)
    calibration_quality = load_calibration_quality(calibration, screen)
    hold_threshold_px, release_threshold_px = projection_thresholds(calibration, screen)
    cap = open_camera(args.camera, args.width, args.height)
    face_mesh = create_face_mesh(thresholds)
    publisher = EyePublishConfig(statusd_url=statusd_url)
    smoother = AdaptiveGazeSmoother(
        min_cutoff=args.min_cutoff,
        beta=args.beta,
        derivative_cutoff=args.derivative_cutoff,
        max_jump_px=args.max_jump_px,
    )
    hold_gate = ProjectionHoldGate(
        hold_threshold_px=hold_threshold_px,
        release_threshold_px=release_threshold_px,
        calibration_quality_score=float(calibration_quality.get("calibration_quality_score", 0.0) or 0.0),
        calibration_recommended_action=str(
            calibration_quality.get("calibration_recommended_action") or "monitor"
        ),
        min_confidence=args.min_sample_confidence,
    )
    frame_period = 1.0 / args.hz
    last_missing_presence_post = 0.0
    stopped = False

    def stop(_signum: int, _frame: Any) -> None:
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        while not stopped:
            started = time.monotonic()
            ok, frame = cap.read()
            if ok and frame is not None:
                sample = frame_sample(face_mesh, frame)
                if sample is not None:
                    _publish_sample(
                        args=args,
                        sample=sample,
                        screen=screen,
                        calibration=calibration,
                        calibration_quality=calibration_quality,
                        smoother=smoother,
                        hold_gate=hold_gate,
                        publisher=publisher,
                    )
                elif started - last_missing_presence_post >= 1.0:
                    _publish_missing_presence(screen, publisher)
                    last_missing_presence_post = started
            sleep_for = frame_period - (time.monotonic() - started)
            if sleep_for > 0:
                time.sleep(sleep_for)
        return 0
    finally:
        cap.release()
        face_mesh.close()


def _publish_missing_presence(screen: Any, publisher: EyePublishConfig) -> None:
    try:
        post_eye(
            (screen.width / 2, screen.height / 2),
            screen,
            0.0,
            max_age=2,
            config=publisher,
            state="presence_missing",
        )
    except PublishError:
        pass


def _publish_sample(
    *,
    args: Any,
    sample: Any,
    screen: Any,
    calibration: dict[str, Any],
    calibration_quality: dict[str, Any],
    smoother: AdaptiveGazeSmoother,
    hold_gate: ProjectionHoldGate,
    publisher: EyePublishConfig,
) -> None:
    raw_point = predict(calibration, sample.features)
    correction = load_drift_correction()
    corrected_point = apply_drift_correction(raw_point, screen, correction)
    confidence = min(1.0, sample.confidence * args.confidence_scale)
    stable_state = smoother.snapshot()
    decision = hold_gate.update(
        corrected_point,
        screen,
        confidence,
        float(stable_state.get("stability_score", confidence) or confidence),
        smoother.current(),
    )
    if decision.should_reset:
        smoother.reset(decision.smooth_point or decision.publish_point, time.monotonic(), confidence=confidence)
    if decision.smooth_point is None:
        smoothed = decision.publish_point
        stability = {
            **smoother.snapshot(),
            "stability_score": 0.0,
            "targeting_reliable": False,
            "filter_mode": decision.mode,
        }
    else:
        smoothed = smoother.update(
            decision.smooth_point,
            time.monotonic(),
            confidence=confidence if decision.targeting_reliable else min(confidence, 0.4),
        )
        stability = smoother.snapshot()
    try:
        post_eye(
            smoothed,
            screen,
            confidence,
            max_age=max(2, int(args.max_age_seconds)),
            config=publisher,
            extra=_publish_metadata(
                stability=stability,
                sample_telemetry=sample.telemetry,
                calibration_quality=calibration_quality,
                raw_point=raw_point,
                correction=correction,
                decision=decision,
            ),
        )
    except PublishError:
        pass


def _publish_metadata(
    *,
    stability: dict[str, Any],
    sample_telemetry: dict[str, Any],
    calibration_quality: dict[str, Any],
    raw_point: tuple[float, float],
    correction: dict[str, Any] | None,
    decision: Any,
) -> dict[str, Any]:
    return {
        **stability,
        **sample_telemetry,
        **calibration_quality,
        "observed_screen_x": round(raw_point[0], 2),
        "observed_screen_y": round(raw_point[1], 2),
        "correction_offset_x_px": correction.get("screen_x_offset_px") if correction else None,
        "correction_offset_y_px": correction.get("screen_y_offset_px") if correction else None,
        "correction_sample_count": correction.get("sample_count") if correction else None,
        "correction_reliability_score": correction.get("reliability_score") if correction else None,
        "correction_updated_at": correction.get("created_at") if correction else None,
        "projection_error_px": decision.projection_error_px,
        "projection_offscreen_px": decision.projection_offscreen_px,
        "projection_hold_active": decision.hold_active,
        "projection_hold_reason": decision.hold_reason,
        "projection_hold_hint": decision.hold_hint,
        "projection_hold_samples": decision.stable_frames,
        "projection_hold_threshold_px": decision.hold_threshold_px,
        "projection_release_threshold_px": decision.release_threshold_px,
        "projection_recovery_score": decision.recovery_score,
        "projection_hold_budget_frames": decision.hold_budget_frames,
        "projection_anchor_x": round(decision.anchor_point[0], 2) if decision.anchor_point else None,
        "projection_anchor_y": round(decision.anchor_point[1], 2) if decision.anchor_point else None,
        "projection_recommended_action": calibration_quality.get("calibration_recommended_action"),
        "filter_mode": decision.mode,
        "targeting_reliable": bool(
            decision.targeting_reliable and bool(stability.get("targeting_reliable", True))
        ),
    }
