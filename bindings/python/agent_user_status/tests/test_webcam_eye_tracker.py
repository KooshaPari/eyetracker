from __future__ import annotations

from pathlib import Path


def test_tracker_uses_privacy_safe_missing_presence_state() -> None:
    source = Path("src/agent_user_status/webcam_runtime.py").read_text(encoding="utf-8")

    assert 'state="presence_missing"' in source
    assert 'state="no_face"' not in source
