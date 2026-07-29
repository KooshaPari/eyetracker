from __future__ import annotations

from typing import cast

from agent_user_status.eye_smoothing import AdaptiveGazeSmoother


def test_smoother_holds_small_deadband_motion() -> None:
    smoother = AdaptiveGazeSmoother()
    assert smoother.update((100.0, 100.0), timestamp=0.0, confidence=1.0) == (100.0, 100.0)

    filtered = smoother.update((103.0, 104.0), timestamp=0.1, confidence=1.0)

    assert filtered == (100.0, 100.0)
    assert smoother.snapshot()["filter_mode"] == "deadband_hold"


def test_smoother_holds_low_confidence_jump() -> None:
    smoother = AdaptiveGazeSmoother()
    smoother.update((100.0, 100.0), timestamp=0.0, confidence=1.0)

    filtered = smoother.update((500.0, 100.0), timestamp=0.1, confidence=0.1)

    assert filtered == (100.0, 100.0)
    assert smoother.snapshot()["filter_mode"] == "confidence_hold"
    assert smoother.snapshot()["targeting_reliable"] is False


def test_smoother_clamps_large_reliable_jump_before_filtering() -> None:
    smoother = AdaptiveGazeSmoother(max_jump_px=100.0)
    smoother.update((0.0, 0.0), timestamp=0.0, confidence=1.0)

    filtered = smoother.update((300.0, 0.0), timestamp=0.1, confidence=1.0)

    assert 0.0 < filtered[0] <= 100.0
    assert filtered[1] == 0.0
    assert smoother.current() == filtered
    assert smoother.snapshot()["filter_mode"] == "jump_clamp"


def test_smoother_ema_tracks_jump_velocity_and_residual() -> None:
    smoother = AdaptiveGazeSmoother()
    smoother.update((100.0, 100.0), timestamp=0.0, confidence=1.0)

    smoother.update((150.0, 100.0), timestamp=0.2, confidence=0.9)
    first = smoother.snapshot()
    smoother.update((200.0, 100.0), timestamp=0.4, confidence=0.9)
    second = smoother.snapshot()

    assert first["filter_mode"] == "tracking"
    assert second["filter_mode"] == "tracking"
    assert cast(float, second["velocity_px_s"]) > 0.0
    assert cast(float, second["jitter_px"]) >= cast(float, first["jitter_px"])


def test_smoother_reset_reseeds_state_and_clears_ema() -> None:
    smoother = AdaptiveGazeSmoother()
    smoother.update((100.0, 100.0), timestamp=0.0, confidence=1.0)
    smoother.update((300.0, 260.0), timestamp=0.1, confidence=0.9)

    smoother.reset((640.0, 360.0), timestamp=2.0, confidence=0.8)

    assert smoother.current() == (640.0, 360.0)
    assert smoother.jump_ema == 0.0
    assert smoother.residual_ema == 0.0
    assert smoother.velocity_ema == 0.0
    assert smoother.snapshot() == {
        "stability_score": 0.8,
        "jump_px": 0.0,
        "jitter_px": 0.0,
        "velocity_px_s": 0.0,
        "targeting_reliable": True,
        "filter_mode": "reseed",
    }
