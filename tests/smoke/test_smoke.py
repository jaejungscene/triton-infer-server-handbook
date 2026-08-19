"""
test_smoke.py — 배포 직후 빠른 정상 확인

서버가 살아있는지, 모델이 로드되었는지, metrics가 노출되는지 확인합니다.
"""

import requests


def _ready_models(triton_url, headers):
    response = requests.post(
        f"{triton_url.rstrip('/')}/v2/repository/index",
        json={"ready": True},
        headers=headers,
        timeout=10,
    )
    response.raise_for_status()
    models = response.json()
    assert isinstance(models, list), "Repository Index response must be a JSON array"
    return [model for model in models if model.get("name")]


class TestServerHealth:
    """Triton 서버 상태 확인"""

    def test_server_live(self, triton_url, triton_headers):
        """서버가 살아있는지 확인 (/v2/health/live)"""
        response = requests.get(
            f"{triton_url}/v2/health/live", headers=triton_headers, timeout=10
        )
        assert response.status_code == 200

    def test_server_ready(self, triton_url, triton_headers):
        """서버가 요청 처리 준비 완료 (/v2/health/ready)"""
        response = requests.get(
            f"{triton_url}/v2/health/ready", headers=triton_headers, timeout=10
        )
        assert response.status_code == 200

    def test_server_metadata(self, triton_url, triton_headers):
        """서버 메타데이터 조회"""
        response = requests.get(
            f"{triton_url}/v2", headers=triton_headers, timeout=10
        )
        assert response.status_code == 200
        data = response.json()
        assert "name" in data


class TestModelStatus:
    """로드된 모델 상태 확인"""

    def test_models_loaded(self, triton_url, triton_headers):
        """최소 1개 이상의 모델이 로드됨"""
        models = _ready_models(triton_url, triton_headers)
        assert models, "No ready models returned by Repository Index API"

    def test_model_ready(self, triton_url, triton_headers):
        """로드된 모델이 READY 상태"""
        models = _ready_models(triton_url, triton_headers)
        for model in models:
            name = model["name"]
            ready_response = requests.get(
                f"{triton_url}/v2/models/{name}/ready",
                headers=triton_headers,
                timeout=10,
            )
            assert ready_response.status_code == 200, f"Model {name} is not ready"


class TestMetrics:
    """Prometheus 메트릭 노출 확인"""

    def test_metrics_endpoint(self, triton_metrics_url, triton_metrics_headers):
        """메트릭 엔드포인트 접근 가능"""
        response = requests.get(
            f"{triton_metrics_url}/metrics",
            headers=triton_metrics_headers,
            timeout=10,
        )
        assert response.status_code == 200
        assert "nv_inference" in response.text or "# HELP" in response.text
