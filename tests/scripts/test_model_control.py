import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


class _ModelControlHandler(BaseHTTPRequestHandler):
    ready = False
    actions = []

    def do_POST(self):
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
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ModelControlHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()


def _run_script(project_root, script_name, *args):
    script = Path(project_root) / "scripts" / "model_control" / script_name
    return subprocess.run(
        [str(script), *args],
        check=False,
        capture_output=True,
        text=True,
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
