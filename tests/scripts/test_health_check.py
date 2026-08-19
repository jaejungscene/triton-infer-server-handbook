import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


class _TritonHandler(BaseHTTPRequestHandler):
    ready_models = [{"name": "text_classifier", "version": "1", "state": "READY"}]
    expected_authorization = None

    def _authorized(self):
        expected = type(self).expected_authorization
        if expected is None or self.headers.get("Authorization") == expected:
            return True
        self.send_response(401)
        self.end_headers()
        return False

    def do_GET(self):
        if not self._authorized():
            return
        if self.path in {"/v2/health/live", "/v2/health/ready"}:
            self.send_response(200)
            self.end_headers()
            return
        self.send_error(404)

    def do_POST(self):
        if not self._authorized():
            return
        if self.path != "/v2/repository/index":
            self.send_error(404)
            return
        content_length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(content_length)
        payload = json.dumps(type(self).ready_models).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass


@pytest.fixture
def fake_triton():
    _TritonHandler.ready_models = [
        {"name": "text_classifier", "version": "1", "state": "READY"}
    ]
    _TritonHandler.expected_authorization = None
    server = ThreadingHTTPServer(("127.0.0.1", 0), _TritonHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()


def _run_health_check(project_root, triton_url, env=None):
    return subprocess.run(
        [f"{project_root}/scripts/health_check.sh", triton_url],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_health_check_passes_with_ready_model(project_root, fake_triton):
    _TritonHandler.ready_models = [
        {"name": "text_classifier", "version": "1", "state": "READY"}
    ]

    result = _run_health_check(project_root, fake_triton)

    assert result.returncode == 0
    assert "text_classifier" in result.stdout


def test_health_check_fails_without_ready_models(project_root, fake_triton):
    _TritonHandler.ready_models = []

    result = _run_health_check(project_root, fake_triton)

    assert result.returncode == 1
    assert "no ready models" in result.stderr


def test_health_check_authenticates_every_request(project_root, fake_triton):
    _TritonHandler.expected_authorization = "Bearer test-token"
    env = os.environ.copy()
    env["TRITON_AUTH_TOKEN"] = "test-token"

    result = _run_health_check(project_root, fake_triton, env=env)

    assert result.returncode == 0


def test_health_check_rejects_unsafe_urls_and_header_injection(project_root):
    unsafe_url = _run_health_check(project_root, "file:///tmp/triton")
    assert unsafe_url.returncode == 2
    assert "HTTP(S) origin" in unsafe_url.stderr

    env = os.environ.copy()
    env["TRITON_AUTH_TOKEN"] = "token\nX-Forged: true"
    unsafe_token = _run_health_check(
        project_root, "http://localhost:8000", env=env
    )
    assert unsafe_token.returncode == 2
    assert "line breaks" in unsafe_token.stderr
