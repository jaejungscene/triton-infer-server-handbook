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

# 기본 전체 실행 (live smoke/integration은 skip)
pytest tests/

# Config 검증만 (CI에서 주로 사용)
pytest tests/config/

# Smoke test (Triton 서버 실행 중이어야 함)
pytest tests/smoke/ \
  --run-live \
  --triton-url http://localhost:8000 \
  --triton-metrics-url http://localhost:8002

# Integration test
pytest tests/integration/ \
  --run-live \
  --triton-url http://localhost:8000 \
  --triton-metrics-url http://localhost:8002

# Performance test
docker run --rm \
  --add-host host.docker.internal:host-gateway \
  -e TRITON_URL=host.docker.internal:8000 \
  -v "$PWD:/workspace" -w /workspace \
  nvcr.io/nvidia/tritonserver:24.08-py3-sdk@sha256:af34153227000b64d1ed4faf9612570a44d414ab8aa0e1dc143f18c19d71a5a7 \
  ./tests/perf/run_perf_analyzer.sh
```

`tests/config/`는 Triton 서버가 없어도 실행됩니다. `tests/smoke/`와
`tests/integration/`은 이미 기동 중인 Triton 서버가 필요합니다.
두 live suite는 endpoint를 실수로 호출하지 않도록 기본 실행에서 skip되며 반드시
`--run-live`를 지정해야 합니다.
Smoke test는 Repository Index API에서 ready 모델이 최소 하나 확인되어야 통과합니다.
서버 health endpoint만 200이고 모델이 하나도 로드되지 않은 상태는 배포 성공으로 보지 않습니다.

Integration suite의 `test_text_classifier.py`는 기본 manifest의 필수 E2E gate입니다. 모델
readiness, BYTES input, label/confidence output 계약을 실제 inference로 확인하며 실패를 skip하지
않습니다. `test_cache.py`도 같은 필수 모델에 고유 입력을 두 번 보내 miss와 hit counter가
각각 증가하는지 강제하므로, model config 또는 server `--cache-config`가 빠지면 staging을
실패시킵니다. LLM/vision처럼 manifest에서 기본 비활성인 선택 모델은 로드되지 않았을 때만
skip하고, 일단 ready인 모델의 연결·설정·inference 오류는 배포 실패로 처리합니다.

Main release CI도 candidate image를 production과 같은 `explicit + --load-model=*`, local cache,
rate limiter 인자로 기동한 뒤 smoke, 필수 text classifier, cache miss/hit 계약을 모두 통과해야
SHA release tag를 생성합니다. 따라서 staging 이전에도 image 안 모델과 production server
인자의 결합 오류를 차단합니다.

성능 검증은 Repository Index API에서 ready 모델을 조회한 뒤 모델별 CSV를 생성하고,
`tests/perf/profiles.json`의 input data·batch size·shape로 부하를 만든 다음
`tests/perf/baseline.json`의 동일 concurrency 기준과 비교합니다. `perf_analyzer`의 latency
CSV 값은 microsecond이므로 비교기가 millisecond로 변환합니다. 기준값은 GPU 종류, Triton
버전, 모델 artifact와 입력 데이터에 종속되므로 production 적용 전 전용 runner에서 다시
측정해 갱신해야 합니다. 기준선에 없는 모델이나 결과가 하나도 없는 실행은 성공으로
간주하지 않습니다. 새 모델을 benchmark 대상에 추가할 때 profile과 baseline을 함께 추가해야
하며, profile이 없는 ready 모델은 임의 random input으로 측정하지 않고 즉시 실패합니다.

## 환경변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `TRITON_HTTP_URL` | `http://localhost:8000` | Triton HTTP endpoint |
| `TRITON_GRPC_URL` | `localhost:8001` | Triton gRPC endpoint |
| `TRITON_METRICS_URL` | HTTP URL에서 8002로 추론 | Triton Prometheus metrics endpoint |
| `TRITON_AUTH_TOKEN` | 없음 | HTTP/gRPC health·metadata·inference Bearer token |
| `TRITON_METRICS_AUTH_TOKEN` | inference token 사용 | metrics endpoint 전용 Bearer token |

인증 token은 pytest 인수로 넘기지 않고 환경변수로 주입합니다. metrics gateway가 inference와
다른 credential을 요구하면 `TRITON_METRICS_AUTH_TOKEN`을 별도로 설정합니다. 두 token 모두
줄바꿈을 포함하면 header injection 위험으로 실행 전에 실패합니다.
