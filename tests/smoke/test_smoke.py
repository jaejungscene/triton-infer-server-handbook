"""
test_smoke.py — 배포 직후 빠른 정상 확인

서버가 살아있는지, 모델이 로드되었는지, metrics가 노출되는지 확인합니다.
"""

import requests


class TestServerHealth:
    """Triton 서버 상태 확인"""

    def test_server_live(self, triton_url):
        """서버가 살아있는지 확인 (/v2/health/live)"""
        response = requests.get(f"{triton_url}/v2/health/live", timeout=10)
        assert response.status_code == 200

    def test_server_ready(self, triton_url):
        """서버가 요청 처리 준비 완료 (/v2/health/ready)"""
        response = requests.get(f"{triton_url}/v2/health/ready", timeout=10)
        assert response.status_code == 200

    def test_server_metadata(self, triton_url):
        """서버 메타데이터 조회"""
        response = requests.get(f"{triton_url}/v2", timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert "name" in data


class TestModelStatus:
    """로드된 모델 상태 확인"""

    def test_models_loaded(self, triton_url):
        """최소 1개 이상의 모델이 로드됨"""
        response = requests.get(f"{triton_url}/v2/models", timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert len(data.get("models", [])) > 0, "No models loaded"

    def test_model_ready(self, triton_url):
        """로드된 모델이 READY 상태"""
        response = requests.get(f"{triton_url}/v2/models", timeout=10)
        models = response.json().get("models", [])
        for model in models:
            name = model["name"]
            ready_response = requests.get(
                f"{triton_url}/v2/models/{name}/ready", timeout=10
            )
            assert ready_response.status_code == 200, f"Model {name} is not ready"


class TestMetrics:
    """Prometheus 메트릭 노출 확인"""

    def test_metrics_endpoint(self, triton_metrics_url):
        """메트릭 엔드포인트 접근 가능"""
        response = requests.get(f"{triton_metrics_url}/metrics", timeout=10)
        assert response.status_code == 200
        assert "nv_inference" in response.text or "# HELP" in response.text
