"""Boundary tests for the required deterministic smoke model."""

import importlib.util
import sys
import types
from pathlib import Path


def _load_model(project_root):
    sys.modules.setdefault("triton_python_backend_utils", types.ModuleType("pb_utils"))
    path = Path(
        project_root,
        "models",
        "serving",
        "nlp",
        "text_classifier",
        "1",
        "model.py",
    )
    spec = importlib.util.spec_from_file_location("text_classifier_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_classifier_matches_complete_casefolded_words(project_root):
    model = _load_model(project_root)

    assert model.classify_text("A GOOD and stable release") == ("positive", 0.91)
    assert model.classify_text("This is unstable") == ("negative", 0.89)
    assert model.classify_text("빠르다 그리고 좋다") == ("positive", 0.91)


def test_classifier_does_not_match_substrings(project_root):
    model = _load_model(project_root)

    assert model.classify_text("goodbye errorless stability") == ("neutral", 0.62)
    assert model.classify_text("fast and slow") == ("neutral", 0.62)
