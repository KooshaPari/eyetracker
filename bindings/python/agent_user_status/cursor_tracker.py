#!/usr/bin/env python3
"""Cursor activity tracker for the user-status runtime.

This is not eye tracking and must not drive the monitor dot. It publishes
short-lived derived input activity when the cursor moves.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request

from AppKit import NSEvent, NSScreen  # type: ignore[reportAttributeAccessIssue]

ACTION_URL = "http://127.0.0.1:8765/action"
CORRECTION_URL = "http://127.0.0.1:8765/correction/event"


def post(url: str, payload: dict[str, object]) -> None:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=1.0) as response:
        response.read()


def screen_size() -> tuple[float, float]:
    screen = NSScreen.mainScreen()
    frame = screen.frame() if screen else None
    if frame is None:
        return 1440.0, 900.0
    return float(frame.size.width), float(frame.size.height)


def mouse_button_down() -> bool:
    try:
        return int(NSEvent.pressedMouseButtons()) != 0
    except Exception:
        return False


def post_correction_click(x: int, y: int, args: argparse.Namespace) -> None:
    width, height = screen_size()
    post(
        CORRECTION_URL,
        {
            "kind": "cursor_click",
            "screen_x": x,
            "screen_y": y,
            "screen_width": width,
            "screen_height": height,
            "score": args.correction_score,
            "state": "cursor_click_target",
            "harmony_hint": True,
            "input_modality": "mouse",
            "max_age_seconds": max(2, int(args.correction_max_age_seconds)),
        },
    )


def command_run(args: argparse.Namespace) -> int:
    poll_interval = 1.0 / args.poll_hz
    activity_interval = 1.0 / args.hz
    last: tuple[int, int] | None = None
    last_activity_post = 0.0
    was_down = False
    while True:
        point = NSEvent.mouseLocation()
        x, y = int(point.x), int(point.y)
        now = time.monotonic()
        down = mouse_button_down()
        if args.correction_clicks and down and not was_down:
            try:
                post_correction_click(x, y, args)
            except Exception:
                pass
        was_down = down

        should_post_activity = now - last_activity_post >= activity_interval
        if args.only_changed and last == (x, y):
            time.sleep(poll_interval)
            continue
        last = (x, y)
        if not should_post_activity:
            time.sleep(poll_interval)
            continue
        payload = {
            "direction": "input",
            "kind": "mouse_move",
            "score": args.score,
            "state": "mouse_move",
            "max_age_seconds": max(2, int(args.max_age_seconds)),
        }
        try:
            post(ACTION_URL, payload)
            last_activity_post = now
        except Exception:
            # Keep loop resilient to transient local endpoint availability issues.
            pass
        time.sleep(poll_interval)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish cursor movement as input activity, not visual gaze")
    parser.add_argument("--hz", type=float, default=0.5)
    parser.add_argument("--poll-hz", type=float, default=12.0)
    parser.add_argument("--score", type=float, default=0.7)
    parser.add_argument("--max-age-seconds", type=int, default=30)
    parser.add_argument("--only-changed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--correction-clicks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--correction-score", type=float, default=0.78)
    parser.add_argument("--correction-max-age-seconds", type=int, default=45)
    parser.set_defaults(func=command_run)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
