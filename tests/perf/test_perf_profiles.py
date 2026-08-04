import json
import sys
from pathlib import Path

import pytest

PERF_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PERF_DIR))

from profile_args import profile_arguments  # noqa: E402


def _model_names(path):
    return set(json.loads(path.read_text(encoding="utf-8"))["models"])


def test_every_baseline_has_a_reproducible_profile():
    baseline_models = _model_names(PERF_DIR / "baseline.json")
    profile_models = _model_names(PERF_DIR / "profiles.json")
    assert profile_models == baseline_models

    for model in sorted(profile_models):
        arguments = profile_arguments(
            PERF_DIR / "profiles.json", model, PERF_DIR
        )
        assert "--input-data" in arguments
        assert "--batch-size" in arguments


def test_text_profile_uses_versioned_real_inputs():
    arguments = profile_arguments(
        PERF_DIR / "profiles.json", "text_classifier", PERF_DIR
    )
    input_path = Path(arguments[arguments.index("--input-data") + 1])
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    assert len(payload["data"]) >= 3
    assert all("INPUT_TEXT" in request for request in payload["data"])


def test_unknown_model_fails_instead_of_using_random_input():
    with pytest.raises(ValueError, match="No performance profile"):
        profile_arguments(PERF_DIR / "profiles.json", "unknown", PERF_DIR)
