from __future__ import annotations

import types
from types import SimpleNamespace

import pytest

from app.services.asr_backends.qwen3_backend import Qwen3ASRBackend


class _FakeQwen3ASRModel:
    calls: list[dict] = []

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs):
        cls.calls.append({"model_id": model_id, "kwargs": kwargs})
        return cls()

    def transcribe(self, **kwargs):
        self.calls.append({"transcribe": kwargs})
        return [SimpleNamespace(text="テストです", language="Japanese", time_stamps=(0.0, 1.0))]


def _install_fake_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        __import__("sys").modules,
        "torch",
        SimpleNamespace(
            float16="float16",
            float32="float32",
            cuda=SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None),
        ),
    )


def test_qwen3_uses_official_qwen_asr_backend(monkeypatch: pytest.MonkeyPatch):
    _FakeQwen3ASRModel.calls = []
    _install_fake_torch(monkeypatch)

    fake_qwen3_module = types.ModuleType("qwen_asr.inference.qwen3_asr")
    fake_qwen3_module.Qwen3ASRModel = _FakeQwen3ASRModel

    monkeypatch.setitem(__import__("sys").modules, "qwen_asr", types.ModuleType("qwen_asr"))
    monkeypatch.setitem(__import__("sys").modules, "qwen_asr.inference", types.ModuleType("qwen_asr.inference"))
    monkeypatch.setitem(__import__("sys").modules, "qwen_asr.inference.qwen3_asr", fake_qwen3_module)

    backend = Qwen3ASRBackend("qwen3-asr-0.6b", device="cpu")
    result = backend.transcribe(b"fake-wav", language="ja", return_timestamps=True)
    second_result = backend.transcribe(b"fake-wav", language="ja", return_timestamps=True)

    assert result["text"] == "テストです"
    assert second_result["text"] == "テストです"
    assert result["model_id"] == "qwen3-asr-0.6b"
    assert result["language"] == "Japanese"
    assert result["chunks"] == [{"text": "テストです", "timestamp": (0.0, 1.0)}]
    load_calls = [call for call in _FakeQwen3ASRModel.calls if "model_id" in call]
    assert len(load_calls) == 1
    assert load_calls[0]["model_id"] == "Qwen/Qwen3-ASR-0.6B"
    assert load_calls[0]["kwargs"]["device_map"] == "cpu"


def test_qwen3_missing_qwen_asr_does_not_fallback_to_generic_pipeline(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fake_torch(monkeypatch)

    def fake_import_module(name: str):
        if name == "qwen_asr.inference.qwen3_asr":
            raise ImportError("No module named qwen_asr")
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr("app.services.asr_backends.qwen3_backend.importlib.import_module", fake_import_module)

    backend = Qwen3ASRBackend("qwen3-asr-0.6b", device="cpu")

    with pytest.raises(RuntimeError) as excinfo:
        backend.transcribe(b"fake-wav")

    message = str(excinfo.value)
    assert "qwen-asr" in message
    assert "generic transformers ASR pipeline cannot load Qwen3-ASR" in message
