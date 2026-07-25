# Tests 디렉토리 — 테스트 전략 및 실행 방법

## 테스트 계층

| 계층 | 디렉토리 | 목적 | 실행 시점 |
|------|----------|------|-----------|
| **Config** | `tests/config/` | config.pbtxt 유효성, manifest 정합성 | PR (CI) |
| **Smoke** | `tests/smoke/` | 서버 기동 + 모델 로드 + metrics 노출 확인 | 배포 직후 |
| **Integration** | `tests/integration/` | 파이프라인 E2E, 기능별 동작 검증 | Staging 배포 후 |
| **Performance** | `tests/perf/` | throughput/latency 기준선 비교 | 주간/수동 |

## 실행 방법

```bash
# 최초 1회
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-dev.txt

# 전체 실행
pytest tests/

# Config 검증만 (CI에서 주로 사용)
pytest tests/config/

# Smoke test (Triton 서버 실행 중이어야 함)
pytest tests/smoke/ \
  --triton-url http://localhost:8000 \
  --triton-metrics-url http://localhost:8002

# Integration test
pytest tests/integration/ \
  --triton-url http://localhost:8000 \
  --triton-metrics-url http://localhost:8002

# Performance test
docker run --rm \
  --add-host host.docker.internal:host-gateway \
  -e TRITON_URL=host.docker.internal:8000 \
  -v "$PWD:/workspace" -w /workspace \
  nvcr.io/nvidia/tritonserver:24.08-py3-sdk \
  ./tests/perf/run_perf_analyzer.sh
```

`tests/config/`는 Triton 서버가 없어도 실행됩니다. `tests/smoke/`와
`tests/integration/`은 이미 기동 중인 Triton 서버가 필요합니다.

성능 검증은 Repository Index API에서 ready 모델을 조회한 뒤 모델별 CSV를 생성하고,
`tests/perf/baseline.json`의 동일 concurrency 기준과 비교합니다. `perf_analyzer`의 latency
CSV 값은 microsecond이므로 비교기가 millisecond로 변환합니다. 기준값은 GPU 종류, Triton
버전, 모델 artifact와 입력 데이터에 종속되므로 production 적용 전 전용 runner에서 다시
측정해 갱신해야 합니다. 기준선에 없는 모델이나 결과가 하나도 없는 실행은 성공으로
간주하지 않습니다.

## 환경변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `TRITON_HTTP_URL` | `http://localhost:8000` | Triton HTTP endpoint |
| `TRITON_GRPC_URL` | `localhost:8001` | Triton gRPC endpoint |
| `TRITON_METRICS_URL` | HTTP URL에서 8002로 추론 | Triton Prometheus metrics endpoint |
