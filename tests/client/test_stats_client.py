import io
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from client import stats_client  # noqa: E402


class _FakeResponse:
    def __init__(self, payload):
        self._body = io.BytesIO(json.dumps(payload).encode("utf-8"))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self, size=-1):
        return self._body.read(size)


def test_model_stats_encodes_path_and_applies_timeout_and_headers(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["authorization"] = request.get_header("Authorization")
        captured["timeout"] = timeout
        return _FakeResponse({"model_stats": []})

    monkeypatch.setattr(stats_client.urllib.request, "urlopen", fake_urlopen)

    result = stats_client.get_model_stats(
        "https://triton.example.com/",
        "team/model",
        "v 1",
        timeout=3.5,
        headers={"Authorization": "Bearer secret"},
    )

    assert result == {"model_stats": []}
    assert captured == {
        "url": "https://triton.example.com/v2/models/team%2Fmodel/versions/v%201/stats",
        "authorization": "Bearer secret",
        "timeout": 3.5,
    }


@pytest.mark.parametrize(
    "url",
    [
        "localhost:8000",
        "https://user:secret@triton.example.com",
        "https://triton.example.com/prefix",
        "https://triton.example.com?tenant=a",
        " https://triton.example.com",
    ],
)
def test_stats_rejects_non_origin_urls(url):
    with pytest.raises(ValueError, match=r"HTTP\(S\) origin"):
        stats_client.get_all_model_stats(url)


def test_stats_rejects_invalid_timeout_header_and_empty_model():
    with pytest.raises(ValueError, match="finite"):
        stats_client.get_all_model_stats("http://localhost:8000", timeout=float("nan"))
    with pytest.raises(ValueError, match="timeout"):
        stats_client.get_all_model_stats("http://localhost:8000", timeout=0)
    with pytest.raises(ValueError, match="headers"):
        stats_client.get_all_model_stats(
            "http://localhost:8000", headers={"Authorization": "token\nforged"}
        )
    with pytest.raises(ValueError, match="model_name"):
        stats_client.get_model_stats("http://localhost:8000", "")


def test_stats_rejects_oversized_response(monkeypatch):
    monkeypatch.setattr(stats_client, "MAX_RESPONSE_BYTES", 16)
    monkeypatch.setattr(
        stats_client.urllib.request,
        "urlopen",
        lambda request, timeout: _FakeResponse({"model_stats": ["too large"]}),
    )

    with pytest.raises(RuntimeError, match="exceeds 16 bytes"):
        stats_client.get_all_model_stats("http://localhost:8000")


def test_summary_accepts_protobuf_json_integer_strings(capsys):
    stats_client.print_summary(
        {
            "model_stats": [
                {
                    "name": "classifier",
                    "version": "1",
                    "inference_count": "8",
                    "execution_count": "2",
                    "inference_stats": {
                        "queue": {"count": "8", "ns": "8000000"},
                    },
                }
            ]
        }
    )

    output = capsys.readouterr().out
    assert "총 inference 수 : 8" in output
    assert "평균 배치 크기  : 4.00" in output
    assert "평균    1.000 ms" in output
