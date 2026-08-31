import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np


FINAL = 1


class _Tensor:
    def __init__(self, name, data):
        self.name = name
        self._data = data

    def as_numpy(self):
        return self._data


class _TritonError:
    def __init__(self, message):
        self.message = message


class _InferenceResponse:
    def __init__(self, output_tensors=None, error=None):
        self.output_tensors = output_tensors or []
        self.error = error


class _Sender:
    def __init__(self, cancel_after=None):
        self.sent = []
        self.cancel_after = cancel_after

    def is_cancelled(self):
        return self.cancel_after is not None and len(self.sent) >= self.cancel_after

    def send(self, response=None, flags=0):
        self.sent.append((response, flags))


class _Request:
    def __init__(self, inputs, sender=None):
        self.inputs = inputs
        self.sender = sender or _Sender()

    def get_response_sender(self):
        return self.sender


class _Logger:
    messages = []

    @classmethod
    def log_error(cls, message):
        cls.messages.append(message)


def _load_model(project_root, monkeypatch, max_output_tokens=4):
    pb_utils = types.ModuleType("triton_python_backend_utils")
    pb_utils.Tensor = _Tensor
    pb_utils.TritonError = _TritonError
    pb_utils.InferenceResponse = _InferenceResponse
    pb_utils.Logger = _Logger
    pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL = FINAL
    pb_utils.get_input_tensor_by_name = lambda request, name: request.inputs.get(name)
    monkeypatch.setitem(sys.modules, "triton_python_backend_utils", pb_utils)

    path = (
        Path(project_root)
        / "models"
        / "_templates"
        / "decoupled_streaming"
        / "1"
        / "model.py"
    )
    spec = importlib.util.spec_from_file_location("decoupled_template", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.TritonPythonModel()
    model.initialize(
        {
            "model_config": json.dumps(
                {
                    "parameters": {
                        "max_output_tokens": {
                            "string_value": str(max_output_tokens)
                        }
                    }
                }
            )
        }
    )
    return model


def _request(max_tokens, sender=None):
    return _Request(
        {
            "INPUT_TEXT": _Tensor("INPUT_TEXT", np.array([[b"hello"]], dtype=object)),
            "MAX_TOKENS": _Tensor(
                "MAX_TOKENS", np.array([[max_tokens]], dtype=np.int32)
            ),
        },
        sender,
    )


def test_decoupled_config_declares_output_limit(project_root):
    config = (
        Path(project_root)
        / "models"
        / "_templates"
        / "decoupled_streaming"
        / "config.pbtxt"
    ).read_text(encoding="utf-8")

    assert "decoupled: true" in config
    assert 'key: "max_output_tokens"' in config


def test_decoupled_rejects_token_count_above_limit(project_root, monkeypatch):
    model = _load_model(project_root, monkeypatch, max_output_tokens=4)
    request = _request(5)

    model.execute([request])

    response, flags = request.sender.sent[0]
    assert flags == FINAL
    assert response.error.message == "MAX_TOKENS must be between 1 and 4"


def test_decoupled_stops_generation_and_sends_final_on_cancellation(
    project_root, monkeypatch
):
    model = _load_model(project_root, monkeypatch)
    sender = _Sender(cancel_after=1)
    request = _request(4, sender)

    model.execute([request])

    assert len(sender.sent) == 2
    assert sender.sent[0][1] == 0
    assert sender.sent[0][0].output_tensors[0]._data.tolist() == ["token_0"]
    assert sender.sent[1] == (None, FINAL)


def test_decoupled_isolates_unexpected_generation_failure(
    project_root, monkeypatch
):
    model = _load_model(project_root, monkeypatch)
    _Logger.messages = []

    def generate(prompt, max_tokens):
        if prompt == "break":
            raise RuntimeError("private model path")
        return iter(["ok"])

    model._generate_tokens = generate
    failed = _Request(
        {
            "INPUT_TEXT": _Tensor(
                "INPUT_TEXT", np.array([[b"break"]], dtype=object)
            ),
            "MAX_TOKENS": _Tensor(
                "MAX_TOKENS", np.array([[1]], dtype=np.int32)
            ),
        }
    )
    succeeded = _request(1)

    model.execute([failed, succeeded])

    assert failed.sender.sent[0][0].error.message == (
        "Streaming request failed; see server logs"
    )
    assert failed.sender.sent[0][1] == FINAL
    assert succeeded.sender.sent[0][0].output_tensors[0]._data.tolist() == ["ok"]
    assert succeeded.sender.sent[0][1] == FINAL
    assert "private model path" in _Logger.messages[0]
