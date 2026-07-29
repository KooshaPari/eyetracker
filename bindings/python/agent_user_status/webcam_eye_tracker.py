#!/usr/bin/env python3
"""Opt-in webcam gaze collector that publishes only derived screen coordinates."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from agent_user_status.gaze_calibration import (
    calibration_points,
    fit_calibration,
    load_calibration_quality,
    predict,
    projection_thresholds,
)
from agent_user_status.gaze_evaluation import EvaluationCounters
from agent_user_status.gaze_projection import StableSampleGate
from agent_user_status.webcam_probe import probe_presence
from agent_user_status.webcam_runtime import run_tracker
from agent_user_status.webcam_support import (
    FaceTrackerThresholds,
    TrackerError,
    create_face_mesh,
    frame_sample,
    import_cv2,
    import_mediapipe,
    import_numpy,
    open_camera,
    screen_size,
)

STATE_DIR = Path(os.environ.get("AGENT_IMESSAGE_STATE_DIR", "~/.local/share/agent-imessage/state")).expanduser()
CALIBRATION_PATH = Path(
    os.environ.get("AGENT_USER_STATUS_EYE_CALIBRATION", str(STATE_DIR / "eye_calibration.json"))
).expanduser()
STATUSD_URL = os.environ.get("AGENT_USER_STATUSD_URL", "http://127.0.0.1:8765")


def default_camera() -> int:
    try:
        return int(os.environ.get("AGENT_USER_STATUS_EYE_CAMERA", "0"))
    except ValueError:
        return 0


def load_calibration(path: Path = CALIBRATION_PATH) -> dict[str, Any]:
    if not path.exists():
        raise TrackerError(f"calibration missing: run `agent-user-status-webcam-eye-tracker calibrate` first ({path})")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != "mediapipe_iris_regression":
        raise TrackerError(f"unsupported calibration kind in {path}")
    return payload


def summarize_errors(errors: list[float]) -> dict[str, float]:
    if not errors:
        return {"mean_error_px": 0.0, "p95_error_px": 0.0, "max_error_px": 0.0}
    np = import_numpy()
    ordered = np.asarray(errors, dtype=float)
    return {
        "mean_error_px": float(ordered.mean()),
        "p95_error_px": float(np.percentile(ordered, 95)),
        "max_error_px": float(ordered.max()),
    }


def tracker_thresholds(args: argparse.Namespace) -> FaceTrackerThresholds:
    return FaceTrackerThresholds(
        detection=float(args.min_face_detection_confidence),
        presence=float(args.min_face_presence_confidence),
        tracking=float(args.min_tracking_confidence),
    )


def command_calibrate(args: argparse.Namespace) -> int:
    cv2 = import_cv2()
    screen = screen_size()
    cap = open_camera(args.camera, args.width, args.height)
    face_mesh = create_face_mesh(tracker_thresholds(args))
    samples: list[tuple[list[float], float, float]] = []
    sample_gate = StableSampleGate(args.min_sample_confidence, args.min_consecutive_frames)
    window_name = "Agent User Status Eye Calibration"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    points = calibration_points(screen)
    try:
        for idx, (target_x, target_y) in enumerate(points, start=1):
            started = time.monotonic()
            kept = 0
            while time.monotonic() - started < args.seconds_per_point:
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                sample = frame_sample(face_mesh, frame)
                elapsed = time.monotonic() - started
                if sample is None or sample.confidence < args.min_sample_confidence:
                    sample_gate.reset()
                elif elapsed > args.settle_seconds and sample_gate.update(sample.confidence):
                    samples.append((sample.features, float(target_x), float(target_y)))
                    kept += 1

                canvas = cv2.UMat(screen.height, screen.width, cv2.CV_8UC3).get()
                canvas[:] = (18, 18, 18)
                cv2.circle(canvas, (target_x, target_y), 18, (87, 207, 136), -1)
                cv2.circle(canvas, (target_x, target_y), 34, (255, 255, 255), 2)
                cv2.putText(
                    canvas,
                    f"{idx}/{len(points)}  samples {kept}  look at the dot",
                    (40, 54),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (240, 240, 240),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(window_name, canvas)
                if cv2.waitKey(1) == 27:
                    raise TrackerError("calibration cancelled")
        calibration = fit_calibration(samples, screen)
        calibration.update(load_calibration_quality(calibration, screen))
        CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
        CALIBRATION_PATH.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
        print(json.dumps({"ok": True, "calibration": str(CALIBRATION_PATH), **calibration}, indent=2))
        return 0
    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_mesh.close()


def command_run(args: argparse.Namespace) -> int:
    return run_tracker(
        args,
        calibration_path=CALIBRATION_PATH,
        statusd_url=STATUSD_URL,
        thresholds=tracker_thresholds(args),
    )


def command_evaluate(args: argparse.Namespace) -> int:
    cv2 = import_cv2()
    screen = screen_size()
    calibration = load_calibration()
    hold_threshold_px, release_threshold_px = projection_thresholds(calibration, screen)
    cap = open_camera(args.camera, args.width, args.height)
    face_mesh = create_face_mesh(tracker_thresholds(args))
    counters = EvaluationCounters()
    sample_gate = StableSampleGate(args.min_sample_confidence, args.min_consecutive_frames)
    window_name = "Agent User Status Eye Evaluation"
    points = calibration_points(screen)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    try:
        for idx, (target_x, target_y) in enumerate(points, start=1):
            started = time.monotonic()
            target = counters.begin_target(idx, target_x, target_y)
            while time.monotonic() - started < args.seconds_per_point:
                ok, frame = cap.read()
                if not ok or frame is None:
                    target.reject("camera_frame_unavailable")
                    continue
                sample = frame_sample(face_mesh, frame)
                if sample is None:
                    target.reject("no_face_sample")
                    sample_gate.reset()
                    continue
                if sample.confidence < args.min_sample_confidence:
                    target.reject("low_confidence")
                    sample_gate.reset()
                    continue
                if time.monotonic() - started <= args.settle_seconds:
                    target.reject("settling")
                    continue
                if not sample_gate.update(sample.confidence):
                    target.reject("unstable_confidence")
                    continue

                observed = predict(calibration, sample.features)
                sample_health = counters.inspect_observed_sample(observed)
                if sample_health:
                    target.reject(sample_health)
                    continue
                target.accept(observed)

                canvas = cv2.UMat(screen.height, screen.width, cv2.CV_8UC3).get()
                canvas[:] = (18, 18, 18)
                cv2.circle(canvas, (target_x, target_y), 18, (87, 207, 136), -1)
                cv2.circle(canvas, (target_x, target_y), 34, (255, 255, 255), 2)
                cv2.putText(
                    canvas,
                    f"{idx}/{len(points)}  look at the dot",
                    (40, 54),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (240, 240, 240),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(window_name, canvas)
                if cv2.waitKey(1) == 27:
                    raise TrackerError("evaluation cancelled")

        summary = summarize_errors(counters.errors)
        summary.update(
            {
                "hold_threshold_px": round(hold_threshold_px, 2),
                "release_threshold_px": round(release_threshold_px, 2),
                **counters.summary(hold_threshold_px),
                **load_calibration_quality(calibration, screen),
            }
        )
        print(json.dumps({"ok": True, "evaluation": summary}, indent=2))
        return 0
    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_mesh.close()


def command_check(args: argparse.Namespace) -> int:
    payload = {
        "python": sys.executable,
        "calibration": str(CALIBRATION_PATH),
        "calibrated": CALIBRATION_PATH.exists(),
        "statusd_url": STATUSD_URL,
    }
    for name, loader in (("cv2", import_cv2), ("numpy", import_numpy), ("mediapipe", import_mediapipe)):
        try:
            module = loader()
            payload[name] = getattr(module, "__version__", "ok")
        except Exception as exc:
            payload[name] = f"missing: {exc}"
    print(json.dumps(payload, indent=2))
    return 0


def command_probe(args: argparse.Namespace) -> int:
    cap = open_camera(args.camera, args.width, args.height)
    face_mesh = create_face_mesh(tracker_thresholds(args))
    try:
        result = probe_presence(
            cap=cap,
            tracker=face_mesh,
            camera=args.camera,
            requested_width=args.width,
            requested_height=args.height,
            frames=args.frames,
            warmup_frames=args.warmup_frames,
            min_sample_confidence=args.min_sample_confidence,
            frame_delay_seconds=args.frame_delay_seconds,
        )
        print(json.dumps(result.to_dict(), indent=2))
        return 0
    finally:
        cap.release()
        face_mesh.close()


def add_camera_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--camera", type=int, default=default_camera())
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)


def add_tracker_threshold_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--min-face-detection-confidence", type=float, default=0.2)
    parser.add_argument("--min-face-presence-confidence", type=float, default=0.2)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.2)


def add_acquisition_args(parser: argparse.ArgumentParser) -> None:
    add_camera_args(parser)
    add_tracker_threshold_args(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Opt-in webcam gaze tracker for agent-user-status")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="Check optional eye-tracking dependencies")
    check.set_defaults(func=command_check)

    probe = sub.add_parser("probe-camera", help="Open the camera and read one frame without storing it")
    add_acquisition_args(probe)
    probe.add_argument("--frames", type=int, default=45)
    probe.add_argument("--warmup-frames", type=int, default=8)
    probe.add_argument("--min-sample-confidence", type=float, default=0.2)
    probe.add_argument("--frame-delay-seconds", type=float, default=0.02)
    probe.set_defaults(func=command_probe)

    calibrate = sub.add_parser("calibrate", help="Collect 9-point calibration from the MacBook webcam")
    add_acquisition_args(calibrate)
    calibrate.add_argument("--seconds-per-point", type=float, default=2.0)
    calibrate.add_argument("--settle-seconds", type=float, default=0.55)
    calibrate.add_argument("--min-sample-confidence", type=float, default=0.1)
    calibrate.add_argument("--min-consecutive-frames", type=int, default=2)
    calibrate.set_defaults(func=command_calibrate)

    evaluate = sub.add_parser("evaluate", help="Evaluate a saved calibration against a 9-point screen target")
    add_acquisition_args(evaluate)
    evaluate.add_argument("--seconds-per-point", type=float, default=1.6)
    evaluate.add_argument("--settle-seconds", type=float, default=0.4)
    evaluate.add_argument("--min-sample-confidence", type=float, default=0.1)
    evaluate.add_argument("--min-consecutive-frames", type=int, default=2)
    evaluate.set_defaults(func=command_evaluate)

    run = sub.add_parser("run", help="Publish calibrated derived gaze coordinates to statusd")
    add_acquisition_args(run)
    run.add_argument("--hz", type=float, default=12.0)
    run.add_argument("--min-cutoff", type=float, default=1.15)
    run.add_argument("--beta", type=float, default=0.012)
    run.add_argument("--derivative-cutoff", type=float, default=1.0)
    run.add_argument("--max-jump-px", type=float, default=620.0)
    run.add_argument("--confidence-scale", type=float, default=1.0)
    run.add_argument("--min-sample-confidence", type=float, default=0.1)
    run.add_argument("--max-age-seconds", type=int, default=3)
    run.set_defaults(func=command_run)
    return parser


def main() -> int:
    try:
        args = build_parser().parse_args()
        return int(args.func(args))
    except TrackerError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
