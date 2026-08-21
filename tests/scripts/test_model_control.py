import os
import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


class _ModelControlHandler(BaseHTTPRequestHandler):
    ready = False
    actions = []
    authorization_headers = []
    index_status = 200

    def do_POST(self):
        type(self).authorization_headers.append(self.headers.get("Authorization"))
        if self.path == "/v2/repository/index":
            content_length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(content_length)
            if type(self).index_status != 200:
                self.send_error(type(self).index_status)
                return
            payload = json.dumps(
                [{"name": "sample_model", "state": "READY"}]
                if type(self).ready
                else []
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path.endswith("/load"):
            type(self).ready = True
            type(self).actions.append("load")
            self.send_response(200)
            self.end_headers()
            return
        if self.path.endswith("/unload"):
            type(self).ready = False
            type(self).actions.append("unload")
            self.send_response(200)
            self.end_headers()
            return
        self.send_error(404)

    def do_GET(self):
        type(self).authorization_headers.append(self.headers.get("Authorization"))
        if self.path.endswith("/ready"):
            self.send_response(200 if type(self).ready else 503)
            self.end_headers()
            return
        self.send_error(404)

    def log_message(self, format, *args):
        pass


@pytest.fixture
def model_control_server():
    _ModelControlHandler.ready = False
    _ModelControlHandler.actions = []
    _ModelControlHandler.authorization_headers = []
    _ModelControlHandler.index_status = 200
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ModelControlHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()


def _run_script(project_root, script_name, *args, env=None):
    script = Path(project_root) / "scripts" / "model_control" / script_name
    return subprocess.run(
        [str(script), *args],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_load_and_unload_verify_readiness(project_root, model_control_server):
    load_result = _run_script(
        project_root, "load.sh", "sample_model", model_control_server, "5"
    )
    unload_result = _run_script(
        project_root, "unload.sh", "sample_model", model_control_server, "5"
    )

    assert load_result.returncode == 0
    assert unload_result.returncode == 0
    assert _ModelControlHandler.actions == ["load", "unload"]


def test_reload_does_not_ignore_unload(project_root, model_control_server):
    _ModelControlHandler.ready = True

    result = _run_script(
        project_root, "reload.sh", "sample_model", model_control_server, "5"
    )

    assert result.returncode == 0
    assert _ModelControlHandler.actions == ["unload", "load"]
    assert "availability gap" in result.stdout


def test_model_name_rejects_url_path_injection(project_root):
    result = _run_script(project_root, "load.sh", "../other-model")

    assert result.returncode == 2
    assert "Invalid model name" in result.stderr


def test_model_control_rejects_non_http_origin(project_root):
    result = _run_script(project_root, "load.sh", "sample_model", "file:///tmp")

    assert result.returncode == 2
    assert "HTTP(S) origin" in result.stderr

    invalid_port = _run_script(
        project_root, "load.sh", "sample_model", "http://localhost:70000"
    )
    assert invalid_port.returncode == 2
    assert "valid HTTP(S) origin" in invalid_port.stderr


def test_model_control_sends_auth_token_to_control_and_readiness_requests(
    project_root, model_control_server
):
    env = os.environ.copy()
    env["TRITON_AUTH_TOKEN"] = "test-token"

    result = _run_script(
        project_root,
        "load.sh",
        "sample_model",
        model_control_server,
        "5",
        env=env,
    )

    assert result.returncode == 0
    assert _ModelControlHandler.authorization_headers == [
        "Bearer test-token",
        "Bearer test-token",
    ]


def test_model_control_rejects_header_injection(project_root):
    env = os.environ.copy()
    env["TRITON_AUTH_TOKEN"] = "token\nX-Forged: true"

    result = _run_script(project_root, "load.sh", "sample_model", env=env)

    assert result.returncode == 2
    assert "line breaks" in result.stderr


def test_unload_does_not_treat_repository_http_errors_as_success(
    project_root, model_control_server
):
    _ModelControlHandler.ready = True
    _ModelControlHandler.index_status = 401

    result = _run_script(
        project_root, "unload.sh", "sample_model", model_control_server, "5"
    )

    assert result.returncode == 1
    assert "Repository Index failed (HTTP 401)" in result.stderr
