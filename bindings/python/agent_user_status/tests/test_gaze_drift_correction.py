from __future__ import annotations

from dataclasses import dataclass

from agent_user_status.gaze_drift_correction import apply_drift_correction, learn_drift_correction


@dataclass(frozen=True)
class Screen:
    width: int
    height: int


def test_learn_drift_correction_uses_reliable_alignment_events() -> None:
    events = [
        {
            "kind": "cursor_click",
            "screen_x": 500.0,
            "screen_y": 300.0,
            "gaze_screen_x": 480.0,
            "gaze_screen_y": 290.0,
            "gaze_targeting_reliable": True,
            "gaze_fresh": True,
            "harmony_hint": True,
            "score": 0.9,
            "confidence": 0.92,
        },
        {
            "kind": "explicit_alignment",
            "screen_x": 900.0,
            "screen_y": 650.0,
            "gaze_screen_x": 880.0,
            "gaze_screen_y": 635.0,
            "gaze_targeting_reliable": True,
            "gaze_fresh": True,
            "harmony_hint": True,
            "score": 0.95,
            "confidence": 0.96,
        },
        {
            "kind": "cursor_click",
            "screen_x": 700.0,
            "screen_y": 380.0,
            "gaze_screen_x": 100.0,
            "gaze_screen_y": 80.0,
            "gaze_targeting_reliable": False,
            "gaze_fresh": True,
            "harmony_hint": True,
            "score": 0.9,
            "confidence": 0.4,
        },
    ]

    correction = learn_drift_correction(events, Screen(width=1440, height=900))

    assert correction is not None
    assert correction["sample_count"] == 2
    assert correction["reliable_samples"] == 2
    assert correction["screen_x_offset_px"] == 20.0
    assert correction["screen_y_offset_px"] == 13.29
    assert correction["reliability_score"] > 0.5


def test_apply_drift_correction_clamps_to_screen_bounds() -> None:
    screen = Screen(width=1440, height=900)

    corrected = apply_drift_correction((10.0, 12.0), screen, {"screen_x_offset_px": -50.0, "screen_y_offset_px": -22.0})

    assert corrected == (0.0, 0.0)


def test_learn_drift_correction_scores_terminal_windows_higher_than_browser_windows() -> None:
    screen = Screen(width=1440, height=900)
    terminal_events = [
        {
            "kind": "cursor_click",
            "screen_x": 500.0,
            "screen_y": 300.0,
            "gaze_screen_x": 480.0,
            "gaze_screen_y": 290.0,
            "gaze_targeting_reliable": True,
            "gaze_fresh": True,
            "harmony_hint": True,
            "window_role": "terminal",
            "score": 0.9,
            "confidence": 0.92,
        },
        {
            "kind": "cursor_click",
            "screen_x": 900.0,
            "screen_y": 650.0,
            "gaze_screen_x": 880.0,
            "gaze_screen_y": 635.0,
            "gaze_targeting_reliable": True,
            "gaze_fresh": True,
            "harmony_hint": True,
            "window_role": "terminal",
            "score": 0.95,
            "confidence": 0.96,
        },
    ]
    browser_events = [
        {**event, "window_role": "browser"} for event in terminal_events
    ]

    terminal = learn_drift_correction(terminal_events, screen)
    browser = learn_drift_correction(browser_events, screen)

    assert terminal is not None
    assert browser is not None
    assert terminal["reliability_score"] > browser["reliability_score"]


def test_explicit_alignment_can_seed_passive_correction_during_recalibration_need() -> None:
    screen = Screen(width=1440, height=900)
    events = [
        {
            "kind": "explicit_alignment",
            "screen_x": 1020.0,
            "screen_y": 740.0,
            "gaze_screen_x": 100.0,
            "gaze_screen_y": 100.0,
            "gaze_targeting_reliable": False,
            "gaze_fresh": False,
            "learnable": False,
            "score": 1.0,
            "confidence": 1.0,
        },
        {
            "kind": "explicit_alignment",
            "screen_x": 420.0,
            "screen_y": 240.0,
            "gaze_screen_x": 360.0,
            "gaze_screen_y": 300.0,
            "gaze_targeting_reliable": False,
            "gaze_fresh": True,
            "learnable": False,
            "score": 0.92,
            "confidence": 0.88,
        },
        {
            "kind": "explicit_alignment",
            "screen_x": 820.0,
            "screen_y": 640.0,
            "gaze_screen_x": 760.0,
            "gaze_screen_y": 700.0,
            "gaze_targeting_reliable": False,
            "gaze_fresh": True,
            "learnable": False,
            "score": 0.95,
            "confidence": 0.9,
        },
    ]

    correction = learn_drift_correction(events, screen)

    assert correction is not None
    assert correction["source_kinds"] == ["explicit_alignment"]
    assert correction["screen_x_offset_px"] == 60.0
    assert correction["screen_y_offset_px"] == -60.0
