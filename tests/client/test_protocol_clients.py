import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from client.base import TritonConfig  # noqa: E402
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
        return object()

    def close(self):
        pass


@pytest.fixture
def fake_protocol_modules(monkeypatch):
    http_module = types.ModuleType("tritonclient.http")
    http_module.InferenceServerClient = _FakeProtocolClient
    grpc_module = types.ModuleType("tritonclient.grpc")
    grpc_module.InferenceServerClient = _FakeProtocolClient

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
