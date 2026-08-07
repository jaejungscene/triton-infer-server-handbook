"""Unit tests for the disabled-by-default object detection pipeline helpers."""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


def _load_model(project_root, relative_path, module_name):
    sys.modules.setdefault("triton_python_backend_utils", types.ModuleType("pb_utils"))
    path = Path(project_root, relative_path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_letterbox_batch_has_configured_shape_and_metadata(project_root):
    model = _load_model(
        project_root,
        "models/serving/vision/object_detection/preprocessor/1/model.py",
        "od_preprocessor_model",
    )
    image = np.full((2, 320, 640, 3), 255, dtype=np.uint8)

    output, metadata = model.letterbox_batch(image)

    assert output.shape == (2, 3, 640, 640)
    assert output.dtype == np.float32
    np.testing.assert_allclose(metadata[0], [1.0, 0.0, 160.0, 640.0, 320.0])
    np.testing.assert_allclose(output[:, :, :160], 114.0 / 255.0)
    np.testing.assert_allclose(output[:, :, 160:480], 1.0)


def test_postprocess_is_batched_class_aware_and_recovers_scale(project_root):
    model = _load_model(
        project_root,
        "models/serving/vision/object_detection/postprocessor/1/model.py",
        "od_postprocessor_model",
    )
    detections = np.array(
        [[
            [10, 170, 110, 270, 1.0, 0.9, 1],
            [12, 172, 108, 268, 1.0, 0.8, 1],
            [10, 170, 110, 270, 1.0, 0.7, 2],
            [0, 0, 1, 1, 1.0, 0.1, 1],
        ]],
        dtype=np.float32,
    )
    metadata = np.array([[1.0, 0.0, 160.0, 640.0, 320.0]], dtype=np.float32)

    boxes, scores, class_ids, counts = model.postprocess_batch(
        detections, metadata, 0.5, 0.45, 4
    )

    assert counts.tolist() == [[2]]
    np.testing.assert_allclose(boxes[0, :2], [[10, 10, 110, 110], [10, 10, 110, 110]])
    np.testing.assert_allclose(scores[0, :2], [0.9, 0.7])
    assert class_ids[0].tolist() == [1, 2, -1, -1]
