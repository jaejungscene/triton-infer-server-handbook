import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.modules.pop("client", None)

from client.base import TritonConfig  # noqa: E402
from client.streaming_client import TritonStreamingClient  # noqa: E402


class _FakeInferInput:
    def __init__(self, name, shape, datatype):
        self.name = name
        self.shape = shape
        self.datatype = datatype
        self.data = None

    def set_data_from_numpy(self, data):
        self.data = data


class _FakeRequestedOutput:
    def __init__(self, name):
        self.name = name


class _FakeResult:
    def __init__(self, output, final=False):
        self.output = output
        self.final = final

    def as_numpy(self, output_name):
        return self.output

    def get_response(self):
        return SimpleNamespace(
            parameters={
                "triton_final_response": SimpleNamespace(bool_param=self.final)
            }
        )


class _FakeInferenceServerClient:
    emit_responses = True
    last_request = None

    def __init__(self, **kwargs):
        self.callback = None

    def start_stream(self, callback):
        self.callback = callback

    def async_stream_infer(self, **kwargs):
        type(self).last_request = kwargs
        if not type(self).emit_responses:
            return
        self.callback(_FakeResult(np.array([[b"token"]], dtype=object)), None)
        self.callback(_FakeResult(None, final=True), None)

    def stop_stream(self, cancel_requests=False):
        pass

    def close(self):
        pass


@pytest.fixture(autouse=True)
def fake_grpc_module(monkeypatch):
    grpc_module = types.ModuleType("tritonclient.grpc")
    grpc_module.InferInput = _FakeInferInput
    grpc_module.InferRequestedOutput = _FakeRequestedOutput
    grpc_module.InferenceServerClient = _FakeInferenceServerClient
    grpc_module.SslOptions = SimpleNamespace

    tritonclient_module = types.ModuleType("tritonclient")
    tritonclient_module.grpc = grpc_module
    monkeypatch.setitem(sys.modules, "tritonclient", tritonclient_module)
    monkeypatch.setitem(sys.modules, "tritonclient.grpc", grpc_module)
    _FakeInferenceServerClient.emit_responses = True
    _FakeInferenceServerClient.last_request = None


def test_template_stream_decodes_nested_byte_output():
    with TritonStreamingClient(TritonConfig(timeout=1)) as client:
        tokens = list(client.stream_infer("decoupled_streaming", "hello"))

    assert tokens == ["token"]


def test_vllm_stream_uses_serving_model_tensor_contract():
    with TritonStreamingClient(TritonConfig(timeout=1)) as client:
        tokens = list(
            client.stream_generate_vllm(
                "llm_vllm",
                "hello",
                max_tokens=16,
                sampling_parameters={"temperature": 0.2},
            )
        )

    request = _FakeInferenceServerClient.last_request
    assert tokens == ["token"]
    assert [input_tensor.name for input_tensor in request["inputs"]] == [
        "text_input",
        "stream",
        "sampling_parameters",
    ]
    parameters = json.loads(request["inputs"][2].data[0])
    assert parameters == {"temperature": 0.2, "max_tokens": 16}
    assert request["outputs"][0].name == "text_output"


def test_stream_fails_after_idle_timeout():
    _FakeInferenceServerClient.emit_responses = False
    client = TritonStreamingClient(TritonConfig(timeout=0.01))

    with pytest.raises(TimeoutError, match="No streaming response"):
        list(client.stream_infer("decoupled_streaming", "hello"))

    client.close()
