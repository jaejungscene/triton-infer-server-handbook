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


def test_ensemble_template_preprocessor_resizes_to_declared_shape(project_root):
    model = _load_model(
        project_root,
        "models/_templates/ensemble_pipeline/preprocessor/1/model.py",
        "ensemble_preprocessor_template",
    )
    images = np.full((2, 32, 48, 3), 255, dtype=np.uint8)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    output = model.preprocess_batch(images, 224, 224, mean, std)

    assert output.shape == (2, 3, 224, 224)
    assert output.dtype == np.float32
    assert output.flags.c_contiguous
    np.testing.assert_allclose(output[0, :, 0, 0], (1.0 - mean) / std)


def test_ensemble_template_postprocessor_applies_stable_softmax(project_root):
    model = _load_model(
        project_root,
        "models/_templates/ensemble_pipeline/postprocessor/1/model.py",
        "ensemble_postprocessor_template",
    )
    logits = np.array([[1000.0, 1001.0, 1002.0]], dtype=np.float32)

    output = model.postprocess_batch(logits, "softmax")

    np.testing.assert_allclose(output.sum(axis=-1), [1.0])
    assert output.argmax(axis=-1).tolist() == [2]
