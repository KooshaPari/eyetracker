from __future__ import annotations

import pytest

from agent_user_status.eye_publish import EyePublishConfig, PublishError, post_eye


class Screen:
    width = 1440
    height = 900


def test_post_eye_wraps_connection_reset_as_publish_error(monkeypatch) -> None:
    def raise_connection_reset(*_args, **_kwargs):
        raise ConnectionResetError("reset")

    monkeypatch.setattr("urllib.request.urlopen", raise_connection_reset)

    with pytest.raises(PublishError, match="statusd publish failed"):
        post_eye(
            (100.0, 200.0),
            Screen(),
            confidence=0.7,
            max_age=5,
            config=EyePublishConfig(statusd_url="http://127.0.0.1:8765"),
        )
