import asyncio
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from client.async_client import TritonAsyncClient  # noqa: E402
from client.base import TritonConfig, numpy_to_triton_dtype  # noqa: E402


class _InferInput:
    def __init__(self, name, shape, datatype):
        self.name = name
        self.shape = shape
        self.datatype = datatype
        self.data = None

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


class _AsyncProtocolClient:
    last_call = None

    def __init__(self, **kwargs):
        self.closed = False

    async def is_server_ready(self, **kwargs):
        type(self).last_call = kwargs
        return True

    async def is_model_ready(self, model_name, model_version, **kwargs):
        type(self).last_call = kwargs
        return True

    async def infer(self, **kwargs):
        type(self).last_call = kwargs
        return _Result()

    async def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def fake_grpc_aio(monkeypatch):
    grpc_module = types.ModuleType("tritonclient.grpc")
    grpc_module.InferInput = _InferInput
    grpc_module.InferRequestedOutput = _InferRequestedOutput
    aio_module = types.ModuleType("tritonclient.grpc.aio")
    aio_module.InferenceServerClient = _AsyncProtocolClient
    grpc_module.aio = aio_module
    tritonclient_module = types.ModuleType("tritonclient")
    tritonclient_module.grpc = grpc_module

    monkeypatch.setitem(sys.modules, "tritonclient", tritonclient_module)
    monkeypatch.setitem(sys.modules, "tritonclient.grpc", grpc_module)
    monkeypatch.setitem(sys.modules, "tritonclient.grpc.aio", aio_module)
    _AsyncProtocolClient.last_call = None


def test_async_client_uses_native_transport_and_closes():
    async def run_inference():
        config = TritonConfig(timeout=4, headers={"authorization": "Bearer test"})
        client = TritonAsyncClient(config)

        async with client:
            outputs = await client.infer_numpy(
                "text_classifier",
                {"INPUT_TEXT": np.array([[b"hello"]], dtype=object)},
                ["OUTPUT_CLASS"],
            )
        return config, client, outputs

    config, client, outputs = asyncio.run(run_inference())

    request = _AsyncProtocolClient.last_call
    assert request["client_timeout"] == 4
    assert request["headers"] == config.headers
    assert request["inputs"][0].datatype == "BYTES"
    assert outputs["OUTPUT_CLASS"].tolist() == ["OUTPUT_CLASS"]
    assert client._client.closed is True


def test_numpy_dtype_contract_includes_unsigned_and_rejects_unicode():
    assert numpy_to_triton_dtype(np.dtype("uint64")) == "UINT64"
    assert numpy_to_triton_dtype(np.dtype(object)) == "BYTES"
    with pytest.raises(ValueError, match="Unsupported"):
        numpy_to_triton_dtype(np.dtype("U10"))


def test_async_client_rejects_invalid_requests_and_incomplete_responses():
    async def run_invalid_requests():
        client = TritonAsyncClient()
        try:
            with pytest.raises(ValueError, match="input_data"):
                await client.infer_numpy("model", {}, ["OUTPUT"])
            with pytest.raises(RuntimeError, match="missing requested output"):
                await client.infer_numpy(
                    "model", {"INPUT": np.ones(1)}, ["MISSING"]
                )
        finally:
            await client.close()

    asyncio.run(run_invalid_requests())
