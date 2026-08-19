import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.modules.pop("client", None)

from client.base import TritonConfig  # noqa: E402
from client.shared_memory_client import TritonSHMClient  # noqa: E402


class _FakeInferInput:
    def __init__(self, name, shape, datatype):
        self.name = name

    def set_shared_memory(self, region_name, byte_size):
        self.region_name = region_name


class _FakeRequestedOutput:
    def __init__(self, name):
        self.name = name

    def set_shared_memory(self, region_name, byte_size):
        self.region_name = region_name


class _FakeResult:
    def __init__(self, outputs):
        self.outputs = outputs

    def as_numpy(self, name):
        return self.outputs.get(name)


class _FakeHTTPClient:
    fail_infer = False
    last_instance = None

    def __init__(self, **kwargs):
        self.registered = []
        self.unregistered = []
        self.source_output = np.array([[1.0]], dtype=np.float32)
        type(self).last_instance = self

    def register_system_shared_memory(self, name, key, byte_size, headers=None):
        self.registered.append(name)

    def unregister_system_shared_memory(self, name, headers=None):
        self.unregistered.append(name)

    def infer(self, model_name, inputs, outputs, model_version, headers=None):
        if type(self).fail_infer:
            raise RuntimeError("inference failed")
        return _FakeResult({output.name: self.source_output for output in outputs})

    def close(self):
        pass


class _FakeSharedMemory(types.ModuleType):
    fail_set = False

    def __init__(self):
        super().__init__("tritonclient.utils.shared_memory")
        self.created = []
        self.destroyed = []

    def create_shared_memory_region(self, name, key, byte_size):
        self.created.append(name)
        return name

    def set_shared_memory_region(self, handle, values):
        if type(self).fail_set:
            raise RuntimeError("shared memory write failed")

    def destroy_shared_memory_region(self, handle):
        self.destroyed.append(handle)


@pytest.fixture(autouse=True)
def fake_tritonclient(monkeypatch):
    http_module = types.ModuleType("tritonclient.http")
    http_module.InferenceServerClient = _FakeHTTPClient
    http_module.InferInput = _FakeInferInput
    http_module.InferRequestedOutput = _FakeRequestedOutput

    shared_memory_module = _FakeSharedMemory()
    utils_module = types.ModuleType("tritonclient.utils")
    utils_module.shared_memory = shared_memory_module
    tritonclient_module = types.ModuleType("tritonclient")
    tritonclient_module.http = http_module
    tritonclient_module.utils = utils_module

    monkeypatch.setitem(sys.modules, "tritonclient", tritonclient_module)
    monkeypatch.setitem(sys.modules, "tritonclient.http", http_module)
    monkeypatch.setitem(sys.modules, "tritonclient.utils", utils_module)
    monkeypatch.setitem(
        sys.modules, "tritonclient.utils.shared_memory", shared_memory_module
    )
    _FakeHTTPClient.fail_infer = False
    _FakeSharedMemory.fail_set = False
    yield shared_memory_module


def _infer(client):
    return client.infer_with_shm(
        model_name="sample",
        input_data={"input": np.array([[1.0]], dtype=np.float32)},
        output_names=["output"],
        output_shapes={"output": (1, 1)},
        output_dtypes={"output": np.float32},
    )


def test_regions_are_unique_and_cleaned_per_request(fake_tritonclient):
    client = TritonSHMClient(TritonConfig())

    first = _infer(client)
    second = _infer(client)

    assert len(set(fake_tritonclient.created)) == 4
    assert sorted(fake_tritonclient.destroyed) == sorted(fake_tritonclient.created)
    assert client._registered_regions == []
    assert client._region_handles == {}
    assert first["output"][0, 0] == 1.0
    assert second["output"][0, 0] == 1.0
    client.close()


def test_inference_failure_still_cleans_regions(fake_tritonclient):
    client = TritonSHMClient(TritonConfig())
    _FakeHTTPClient.fail_infer = True

    with pytest.raises(RuntimeError, match="inference failed"):
        _infer(client)

    assert sorted(fake_tritonclient.destroyed) == sorted(fake_tritonclient.created)
    assert client._registered_regions == []
    client.close()


def test_invalid_request_metadata_fails_before_allocation(fake_tritonclient):
    client = TritonSHMClient(TritonConfig())

    with pytest.raises(ValueError, match="exactly match"):
        client.infer_with_shm(
            model_name="sample",
            input_data={"input": np.array([[1.0]], dtype=np.float32)},
            output_names=["output"],
            output_shapes={},
            output_dtypes={},
        )

    assert fake_tritonclient.created == []

    with pytest.raises(ValueError, match="positive integer dimensions"):
        client.infer_with_shm(
            model_name="sample",
            input_data={"input": np.array([[1.0]], dtype=np.float32)},
            output_names=["output"],
            output_shapes={"output": (1, 0)},
            output_dtypes={"output": np.float32},
        )
    with pytest.raises(TypeError, match="NumPy"):
        client.infer_with_shm(
            model_name="sample",
            input_data={"input": [1.0]},
            output_names=["output"],
            output_shapes={"output": (1, 1)},
            output_dtypes={"output": np.float32},
        )
    assert fake_tritonclient.created == []
    client.close()


def test_region_is_cleaned_when_input_write_fails(fake_tritonclient):
    client = TritonSHMClient(TritonConfig())
    _FakeSharedMemory.fail_set = True

    with pytest.raises(RuntimeError, match="shared memory write failed"):
        _infer(client)

    assert fake_tritonclient.destroyed == fake_tritonclient.created
    assert client._registered_regions == []
    assert client._region_handles == {}
    client.close()


def test_closed_client_cannot_allocate_regions(fake_tritonclient):
    client = TritonSHMClient(TritonConfig())
    client.close()

    with pytest.raises(RuntimeError, match="client is closed"):
        _infer(client)

    assert fake_tritonclient.created == []
