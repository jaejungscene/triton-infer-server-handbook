import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


class _Tensor:
    def __init__(self, name, data):
        self.name = name
        self._data = data

    def as_numpy(self):
        return self._data


class _TritonError:
    def __init__(self, message):
        self._message = message

    def message(self):
        return self._message


class _InferenceResponse:
    def __init__(self, output_tensors=None, error=None):
        self.output_tensors = output_tensors or []
        self._error = error

    def has_error(self):
        return self._error is not None

    def error(self):
        return self._error


class _InferenceRequest:
    queued_responses = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def exec(self):
        return type(self).queued_responses.pop(0)


class _Request:
    def __init__(self, input_tensor):
        self.input_tensor = input_tensor

    def trace(self):
        return None


def _load_template(project_root, monkeypatch):
    pb_utils = types.ModuleType("triton_python_backend_utils")
    pb_utils.Tensor = _Tensor
    pb_utils.TritonError = _TritonError
    pb_utils.InferenceResponse = _InferenceResponse
    pb_utils.InferenceRequest = _InferenceRequest
    pb_utils.get_input_tensor_by_name = lambda request, name: request.input_tensor
    pb_utils.get_output_tensor_by_name = lambda response, name: next(
        (tensor for tensor in response.output_tensors if tensor.name == name), None
    )
    monkeypatch.setitem(sys.modules, "triton_python_backend_utils", pb_utils)

    path = Path(project_root) / "models" / "_templates" / "bls_model" / "1" / "model.py"
    spec = importlib.util.spec_from_file_location("bls_template", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TritonPythonModel()


def test_bls_config_matches_the_preprocessor_input_contract(project_root):
    config_path = (
        Path(project_root) / "models" / "_templates" / "bls_model" / "config.pbtxt"
    )
    config = config_path.read_text(encoding="utf-8")

    assert 'name: "INPUT"' in config
    assert "data_type: TYPE_UINT8" in config
    assert "dims: [-1, -1, 3]" in config
    assert "dims: [224, 224, 3]" in config


def test_bls_returns_request_error_when_downstream_output_is_missing(
    project_root, monkeypatch
):
    model = _load_template(project_root, monkeypatch)
    _InferenceRequest.queued_responses = [
        _InferenceResponse(output_tensors=[_Tensor("PREPROCESSED", np.ones((1, 3)))]),
        _InferenceResponse(output_tensors=[]),
    ]
    request = _Request(_Tensor("INPUT", np.ones((1, 8, 8, 3), dtype=np.uint8)))

    response = model.execute([request])[0]

    assert response.has_error()
    assert response.error().message() == "Inferencer response is missing RAW_OUTPUT"


def test_bls_returns_request_error_when_input_is_missing(project_root, monkeypatch):
    model = _load_template(project_root, monkeypatch)

    response = model.execute([_Request(None)])[0]

    assert response.has_error()
    assert response.error().message() == "Missing required input: INPUT"
