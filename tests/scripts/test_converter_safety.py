import hashlib
import importlib.util
from pathlib import Path

import pytest


def _load_common(project_root):
    path = Path(project_root) / "scripts" / "convert" / "common.py"
    spec = importlib.util.spec_from_file_location("converter_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validated_input_checks_checksum_and_rejects_symlink(project_root, tmp_path):
    common = _load_common(project_root)
    source = tmp_path / "model.pt"
    source.write_bytes(b"trusted-model")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    assert common.validated_input(str(source), digest) == source.resolve()
    with pytest.raises(common.ConversionError, match="SHA-256 mismatch"):
        common.validated_input(str(source), "0" * 64)

    link = tmp_path / "linked.pt"
    link.symlink_to(source)
    with pytest.raises(common.ConversionError, match="regular file"):
        common.validated_input(str(link))


def test_atomic_output_preserves_previous_artifact_on_failure(project_root, tmp_path):
    common = _load_common(project_root)
    output = tmp_path / "model.onnx"
    output.write_bytes(b"previous")

    with pytest.raises(RuntimeError, match="conversion failed"):
        with common.atomic_output(output) as temporary_output:
            temporary_output.write_bytes(b"incomplete")
            raise RuntimeError("conversion failed")

    assert output.read_bytes() == b"previous"
    assert list(tmp_path.glob(".model-*.onnx")) == []


def test_atomic_output_publishes_nonempty_artifact(project_root, tmp_path):
    common = _load_common(project_root)
    output = tmp_path / "model.onnx"

    with common.atomic_output(output) as temporary_output:
        temporary_output.write_bytes(b"validated")

    assert output.read_bytes() == b"validated"


def test_validated_output_rejects_symlink(project_root, tmp_path):
    common = _load_common(project_root)
    source = tmp_path / "source.pt"
    source.write_bytes(b"source")
    target = tmp_path / "target.onnx"
    target.write_bytes(b"target")
    output_link = tmp_path / "model.onnx"
    output_link.symlink_to(target)

    with pytest.raises(common.ConversionError, match="output must not be a symlink"):
        common.validated_output(str(output_link), source)


@pytest.mark.parametrize("shape", ["", "1,0,224", "1,-3,224", "1,width,224"])
def test_parse_shape_rejects_invalid_dimensions(project_root, shape):
    common = _load_common(project_root)

    with pytest.raises(common.ConversionError, match="shape"):
        common.parse_shape(shape)


def test_pickle_checkpoint_requires_explicit_opt_in(project_root, tmp_path):
    common = _load_common(project_root)

    class FakeJit:
        @staticmethod
        def load(path, map_location):
            raise RuntimeError("not TorchScript")

    class FakeTorch:
        jit = FakeJit()

        @staticmethod
        def load(*args, **kwargs):
            raise AssertionError("unsafe loader must not run")

    with pytest.raises(common.ConversionError, match="allow-unsafe-pickle"):
        common.load_pytorch_module(FakeTorch(), tmp_path / "model.pt", False)
