from __future__ import annotations

from agent_user_status.webcam_probe import summarize_presence_probe


def test_presence_probe_reports_detected_samples() -> None:
    summary = summarize_presence_probe(
        camera=0,
        requested_width=1280,
        requested_height=720,
        frame_width=1280,
        frame_height=720,
        frames_requested=4,
        frames_read=4,
        confidences=[0.21, 0.44],
        min_sample_confidence=0.2,
    )

    assert summary.ok is True
    assert summary.diagnosis == "presence_detected"
    assert summary.presence_samples == 2
    assert summary.missing_presence_samples == 2
    assert summary.max_sample_confidence == 0.44


def test_presence_probe_reports_missing_presence_without_sensitive_terms() -> None:
    summary = summarize_presence_probe(
        camera=0,
        requested_width=1280,
        requested_height=720,
        frame_width=1280,
        frame_height=720,
        frames_requested=3,
        frames_read=3,
        confidences=[],
        min_sample_confidence=0.2,
    )
    payload = summary.to_dict()

    assert payload["ok"] is False
    assert payload["diagnosis"] == "presence_not_detected"
    assert "face" not in str(payload).lower()
    assert "landmark" not in str(payload).lower()
