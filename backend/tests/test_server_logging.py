from __future__ import annotations

from app.utils.server_logging import server_log


def test_server_log_writes_to_stdout(capsys):
    server_log("[TestLog] model loaded and ready")

    captured = capsys.readouterr()
    assert "[TestLog] model loaded and ready" in captured.out
