import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from client.base import (  # noqa: E402
    TritonConfig,
    collect_numpy_outputs,
    validate_numpy_request,
)
import client.grpc_client as grpc_client_module  # noqa: E402
import client.http_client as http_client_module  # noqa: E402


class _FakeProtocolClient:
    last_init = None
    last_call = None

    def __init__(self, **kwargs):
        type(self).last_init = kwargs

    def is_server_ready(self, **kwargs):
        type(self).last_call = kwargs
        return True

    def is_model_ready(self, model_name, model_version, **kwargs):
        type(self).last_call = kwargs
        return True

    def infer(self, **kwargs):
        type(self).last_call = kwargs
        return _Result()

    def close(self):
        pass


class _InferInput:
    def __init__(self, name, shape, datatype):
        self.name = name
        self.shape = shape
        self.datatype = datatype

    def set_data_from_numpy(self, data):
        self.data = data


class _InferRequestedOutput:
    def __init__(self, name):
        self.name = name


class _Result:
    def as_numpy(self, name):
        if name == "MISSING":
            return None
        return np.array([name], dtype=object)


@pytest.fixture
def fake_protocol_modules(monkeypatch):
    http_module = types.ModuleType("tritonclient.http")
    http_module.InferenceServerClient = _FakeProtocolClient
    http_module.InferInput = _InferInput
    http_module.InferRequestedOutput = _InferRequestedOutput
    grpc_module = types.ModuleType("tritonclient.grpc")
    grpc_module.InferenceServerClient = _FakeProtocolClient
    grpc_module.InferInput = _InferInput
    grpc_module.InferRequestedOutput = _InferRequestedOutput

    tritonclient_module = types.ModuleType("tritonclient")
    tritonclient_module.http = http_module
    tritonclient_module.grpc = grpc_module
    monkeypatch.setitem(sys.modules, "tritonclient", tritonclient_module)
    monkeypatch.setitem(sys.modules, "tritonclient.http", http_module)
    monkeypatch.setitem(sys.modules, "tritonclient.grpc", grpc_module)
    _FakeProtocolClient.last_init = None
    _FakeProtocolClient.last_call = None


def test_config_rejects_invalid_timeout_and_partial_mtls_identity():
    with pytest.raises(ValueError, match="timeout"):
        TritonConfig(timeout=0)
    with pytest.raises(ValueError, match="configured together"):
        TritonConfig(ssl=True, ssl_cert="client.crt")
    with pytest.raises(ValueError, match="finite"):
        TritonConfig(timeout=float("nan"))
    with pytest.raises(ValueError, match="without a URL scheme"):
        TritonConfig(url="https://localhost:8000")
    with pytest.raises(ValueError, match="host:port"):
        TritonConfig(grpc_url="localhost:0")
    with pytest.raises(ValueError, match="ssl must be enabled"):
        TritonConfig(ssl_root_cert="ca.crt")
    with pytest.raises(ValueError, match="headers"):
        TritonConfig(headers={"authorization": "token\nforged"})


def test_http_applies_timeout_headers_and_tls_context(
    monkeypatch, fake_protocol_modules
):
    tls_context = object()
    monkeypatch.setattr(
        http_client_module, "http_ssl_context", lambda config: tls_context
    )
    config = TritonConfig(
        timeout=7,
        ssl=True,
        ssl_cert="client.crt",
        ssl_key="client.key",
        headers={"authorization": "Bearer test"},
    )

    client = http_client_module.TritonHTTPClient(config)
    assert _FakeProtocolClient.last_init["connection_timeout"] == 7
    assert _FakeProtocolClient.last_init["network_timeout"] == 7
    assert _FakeProtocolClient.last_init["ssl_context_factory"]() is tls_context

    assert client.is_server_ready() is True
    assert _FakeProtocolClient.last_call["headers"] == config.headers

    client.infer("model", [], [])
    assert _FakeProtocolClient.last_call["headers"] == config.headers


def test_grpc_applies_deadline_headers_and_mtls_material(
    tmp_path, fake_protocol_modules
):
    root_cert = tmp_path / "ca.crt"
    client_cert = tmp_path / "client.crt"
    client_key = tmp_path / "client.key"
    root_cert.write_bytes(b"root")
    client_cert.write_bytes(b"cert")
    client_key.write_bytes(b"key")
    config = TritonConfig(
        timeout=9,
        ssl=True,
        ssl_root_cert=str(root_cert),
        ssl_cert=str(client_cert),
        ssl_key=str(client_key),
        headers={"authorization": "Bearer test"},
    )

    client = grpc_client_module.TritonGRPCClient(config)
    assert _FakeProtocolClient.last_init["root_certificates"] == b"root"
    assert _FakeProtocolClient.last_init["certificate_chain"] == b"cert"
    assert _FakeProtocolClient.last_init["private_key"] == b"key"

    assert client.is_model_ready("model") is True
    assert _FakeProtocolClient.last_call["client_timeout"] == 9
    assert _FakeProtocolClient.last_call["headers"] == config.headers

    client.infer("model", [], [])
    assert _FakeProtocolClient.last_call["client_timeout"] == 9
    assert _FakeProtocolClient.last_call["headers"] == config.headers


@pytest.mark.parametrize(
    "client_type",
    [
        http_client_module.TritonHTTPClient,
        grpc_client_module.TritonGRPCClient,
    ],
)
def test_protocol_clients_validate_requests_and_complete_outputs(
    client_type, fake_protocol_modules
):
    client = client_type(TritonConfig())
    outputs = client.infer_numpy(
        "model",
        {"INPUT": np.array([[1]], dtype=np.int32)},
        ["OUTPUT"],
    )
    assert outputs["OUTPUT"].tolist() == ["OUTPUT"]

    with pytest.raises(ValueError, match="duplicates"):
        client.infer_numpy("model", {"INPUT": np.ones(1)}, ["OUTPUT", "OUTPUT"])
    with pytest.raises(RuntimeError, match="missing requested output"):
        client.infer_numpy("model", {"INPUT": np.ones(1)}, ["MISSING"])


def test_numpy_request_validation_rejects_malformed_payloads():
    with pytest.raises(ValueError, match="model_name"):
        validate_numpy_request("", {"INPUT": np.ones(1)}, ["OUTPUT"])
    with pytest.raises(ValueError, match="input_data"):
        validate_numpy_request("model", {}, ["OUTPUT"])
    with pytest.raises(TypeError, match="NumPy"):
        validate_numpy_request("model", {"INPUT": [1]}, ["OUTPUT"])
    with pytest.raises(ValueError, match="must not be empty"):
        validate_numpy_request("model", {"INPUT": np.array([])}, ["OUTPUT"])
    with pytest.raises(ValueError, match="output_names"):
        validate_numpy_request("model", {"INPUT": np.ones(1)}, [])


def test_output_collection_rejects_non_array_values():
    class InvalidResult:
        def as_numpy(self, name):
            return "not-an-array"

    with pytest.raises(RuntimeError, match="not a NumPy array"):
        collect_numpy_outputs(InvalidResult(), ["OUTPUT"])
