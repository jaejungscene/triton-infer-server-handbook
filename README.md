# Triton Inference Server — Production Handbook

실무에서 바로 활용하고, 확장·유지보수할 수 있는 Triton 추론 서버 가이드라인 레포입니다.

> 운영 템플릿과 학습용 핸드북을 함께 제공합니다. 실제 production 적용 전에는
> 모델 바이너리, artifact 저장소, GPU runner, 도메인별 SLO를 각 팀 환경에 맞게 채워야 합니다.

---

## Quick Start

사전 조건:

- Docker Compose
- NVIDIA GPU + NVIDIA Container Toolkit (Triton 컨테이너 실행 시)
- Python 3.11+ (검증 스크립트와 테스트 실행 시)

기본 manifest는 바로 로드 가능한 `text_classifier`만 활성화합니다. ONNX/TensorRT/FIL/LLM
예제는 실제 모델 artifact와 backend 이미지를 준비한 뒤 `models/serving/manifest.yaml`에서
`enabled: true`로 전환합니다.

```bash
# 1. 환경설정
cp .env.example .env.dev

# 2. 모델 빌드 (models/serving/ → model_repository/)
./scripts/build.sh --env dev

# 3. 서버 기동
docker compose -f deploy/docker/docker-compose.yml up -d

# 4. Health Check
./scripts/health_check.sh
```

로컬 검증만 수행할 때는 Docker 없이 아래 명령으로 시작할 수 있습니다.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-dev.txt
./scripts/validate.sh
pytest tests/config/
```

---

## 프로젝트 구조

```
triton-inference-server/
│
├── models/                              # 모델 소스 코드 (Git 관리 대상)
│   ├── _templates/                      # 새 모델 추가 시 복사하여 시작
│   │   ├── single_model/                # 단일 모델 (ONNX / TensorRT / PyTorch)
│   │   │   ├── 1/model.onnx            # 예시 파일명, 실제 바이너리는 artifact로 배치
│   │   │   └── config.pbtxt
│   │   ├── ensemble_pipeline/           # 전처리 → 추론 → 후처리 DAG
│   │   │   ├── preprocessor/
│   │   │   ├── inferencer/
│   │   │   ├── postprocessor/
│   │   │   └── pipeline/config.pbtxt   # ensemble 스텝 연결 정의
│   │   ├── bls_model/                   # Python 내 다중 모델 호출·분기
│   │   │   └── 1/model.py              # pbutils.InferenceServerClient 직접 호출
│   │   └── decoupled_streaming/         # LLM 토큰 스트리밍 (비동기 응답)
│   │       └── 1/model.py
│   │
│   └── serving/                         # 실제 운영 모델 (도메인별 분류)
│       ├── manifest.yaml                # source→target 매핑 (빌드의 Single Source of Truth)
│       ├── vision/
│       │   ├── classification/
│       │   │   └── resnet50/            # ONNX 단일 모델
│       │   │       ├── 1/model.onnx    # 실제 운영에서는 artifact 저장소/CI가 배치
│       │   │       └── config.pbtxt
│       │   └── object_detection/        # YOLOX 앙상블 파이프라인
│       │       ├── preprocessor/        # Python 백엔드 (이미지 리사이즈)
│       │       ├── yolox/               # TensorRT 백엔드
│       │       ├── postprocessor/       # Python 백엔드 (NMS)
│       │       └── pipeline/            # Ensemble config (DAG 연결)
│       ├── nlp/
│       │   ├── text_classifier/         # Python 백엔드 (CPU smoke 예제)
│       │   └── llm/                     # vLLM + Decoupled Streaming
│       └── tabular/
│           └── anomaly_detector/        # FIL (Forest Inference Library)
│
├── model_repository/                    # 런타임 마운트 대상 (Git 제외, CI가 생성)
│   └── .gitkeep
│
├── configs/                             # 서버 레벨 설정 (환경별 + 기능별)
│   ├── base.txt                         # 공통 인수 (포트, 메트릭, strict config)
│   ├── dev.txt                          # poll 모드, verbose 로그
│   ├── staging.txt                      # explicit 모드
│   ├── prod.txt                         # explicit + 응답 캐시 + rate limit + 스레드 8
│   ├── tls/
│   │   ├── grpc_tls.txt                 # gRPC TLS 인증서 경로 설정
│   │   └── http_tls.txt                 # HTTP TLS 인증서 경로 설정
│   ├── tracing/
│   │   ├── otel.txt                     # OpenTelemetry 엔드포인트 설정
│   │   └── triton_native.txt            # Triton 네이티브 트레이싱
│   ├── cache/
│   │   ├── local_cache.txt              # 인메모리 응답 캐시 (단일 노드)
│   │   └── redis_cache.txt              # Redis 캐시 플러그인 사용 시 참고 설정
│   ├── gpu/
│   │   ├── mps.md                       # CUDA Multi-Process Service 설정 가이드
│   │   └── numa.txt                     # NUMA 최적화 인수
│   └── repository/
│       ├── s3.txt                       # AWS S3 모델 저장소 연동
│       ├── gcs.txt                      # Google Cloud Storage 연동
│       └── azure.txt                    # Azure Blob Storage 연동
│
├── scripts/                             # 운영 자동화 스크립트
│   ├── build.sh                         # manifest.yaml → model_repository/ 빌드
│   ├── validate.sh                      # config.pbtxt 문법·필드 검증
│   ├── health_check.sh                  # /v2/health/ready + 모델 상태 확인
│   ├── convert/                         # 모델 포맷 변환
│   │   ├── to_onnx.py                   # PyTorch → ONNX
│   │   ├── to_tensorrt.sh               # ONNX → TensorRT (trtexec 래퍼)
│   │   ├── to_torchscript.py            # PyTorch → TorchScript
│   │   └── to_fil.py                    # sklearn / XGBoost → FIL
│   └── model_control/                   # 런타임 모델 관리 (서버 재시작 불필요)
│       ├── load.sh                      # POST /v2/repository/models/{name}/load
│       ├── unload.sh                    # POST /v2/repository/models/{name}/unload
│       └── reload.sh                    # unload → load 순차 실행 (무중단 교체)
│
├── client/                              # Triton 클라이언트 라이브러리
│   ├── base.py                          # 추상 기본 클라이언트
│   ├── http_client.py                   # HTTP/REST (KServe v2 프로토콜)
│   ├── grpc_client.py                   # gRPC (고성능 바이너리 통신)
│   ├── async_client.py                  # 비동기 클라이언트 (asyncio)
│   ├── streaming_client.py              # gRPC 스트리밍 (LLM 토큰 단위 응답)
│   ├── shared_memory_client.py          # 공유 메모리 (CPU/GPU 데이터 복사 오버헤드 제거)
│   ├── stats_client.py                  # Statistics API (/v2/models/{name}/stats)
│   └── README.md                        # 프로토콜별 선택 가이드
│
├── deploy/                              # 배포 구성
│   ├── docker/
│   │   ├── Dockerfile                   # 프로덕션 런타임 이미지
│   │   ├── Dockerfile.converter          # 모델 변환용 이미지 (CI 전용)
│   │   ├── docker-compose.yml           # 개발 환경 (GPU 1개, poll 모드)
│   │   └── docker-compose.prod.yml      # 프로덕션 (Triton + Prometheus + Grafana, Redis profile 선택)
│   │
│   ├── helm/                            # Helm Chart (업계 표준 K8s 패키징)
│   │   └── triton/
│   │       ├── Chart.yaml               # 차트 메타데이터 (이름, 버전)
│   │       ├── values.yaml              # 기본값 (모든 환경 공통)
│   │       ├── values.dev.yaml          # dev 오버라이드 (poll, 리소스 최소)
│   │       ├── values.staging.yaml      # staging 오버라이드 (explicit, 2 replicas)
│   │       ├── values.prod.yaml         # prod 오버라이드 (HPA, PDB, 캐시, rate-limit)
│   │       └── templates/
│   │           ├── _helpers.tpl         # 이름·레이블 생성 헬퍼
│   │           ├── deployment.yaml      # Triton Pod 스펙 (GPU tolerations)
│   │           ├── service.yaml         # ClusterIP (HTTP 8000, gRPC 8001, metrics 8002)
│   │           ├── configmap.yaml       # 서버 인수 주입
│   │           ├── hpa.yaml             # HorizontalPodAutoscaler (CPU / GPU 메트릭)
│   │           ├── pvc.yaml             # PersistentVolumeClaim (50Gi, ReadWriteMany)
│   │           └── pdb.yaml             # PodDisruptionBudget (minAvailable)
│   │
│   └── k8s/                             # Kustomize (환경별 오버레이)
│       ├── base/                        # 기본 매니페스트
│       │   ├── deployment.yaml
│       │   ├── service.yaml
│       │   ├── configmap.yaml
│       │   ├── pvc.yaml
│       │   └── kustomization.yaml
│       └── overlays/
│           ├── dev/                     # 1 replica, 리소스 최소
│           ├── staging/                 # 2 replicas, 중간 리소스
│           ├── prod/                    # 3+ replicas, HPA, PDB
│           ├── multi-gpu/               # 단일 노드 다중 GPU
│           └── multi-node/              # 멀티 노드 (HPA + PDB)
│
├── monitoring/                          # 메트릭·트레이싱·알림
│   ├── prometheus/
│   │   ├── scrape_config.yml            # triton-server:8002/metrics 스크랩 설정
│   │   └── triton_rules.yml             # 알림 규칙 (latency, error rate, GPU 사용률 등)
│   ├── grafana/
│   │   └── triton_dashboard.json        # 사전 구성된 Grafana 대시보드
│   └── otel/
│       └── otel-collector-config.yaml   # OTel Collector (수신: OTLP → 전달: Jaeger / Zipkin)
│
├── tests/                               # 테스트 피라미드
│   ├── config/                          # config.pbtxt 검증 (PR마다 실행)
│   ├── smoke/                           # 서버 기동 + 모델 로드 + metrics 확인 (배포 직후)
│   ├── integration/                     # E2E 파이프라인 테스트 (staging)
│   ├── perf/                            # 성능 기준선 비교 (주간)
│   │   └── run_perf_analyzer.sh         # perf_analyzer 래퍼 스크립트
│   ├── conftest.py                      # 공통 pytest 픽스처
│   └── README.md
│
├── extensions/                          # 확장 포인트
│   ├── backends/                        # 커스텀 C++ 백엔드
│   ├── caches/                          # 커스텀 캐시 구현체
│   └── custom_ops/                      # 커스텀 연산자
│
├── .github/
│   ├── actions/
│   │   └── triton-health-check/
│   │       └── action.yml               # 재사용 가능한 헬스체크 액션
│   └── workflows/
│       ├── ci-validate.yml              # PR: config 검증, lint (GPU 불필요)
│       ├── ci-build-test.yml            # main push: 빌드 → smoke test → GHCR push
│       ├── cd-staging.yml               # staging 자동 배포 + integration test
│       ├── cd-production.yml            # prod 수동 배포 + 자동 롤백
│       └── perf-benchmark.yml           # 주간 성능 기준선 비교
│
├── .env.example                         # 환경변수 템플릿 (이미지, 포트, 캐시, TLS 등)
├── .env.dev                             # 개발 환경 (Git 추적 허용)
├── .gitignore
├── README.md
└── CONTRIBUTING.md
```

---

## 운영 핸드북

처음 보는 사람은 아래 순서로 읽으면 됩니다.

| 문서 | 내용 |
|------|------|
| [Architecture](docs/architecture.md) | 요청 처리 경로, 모델 레포지토리 전략, serving 패턴, 관측성 |
| [Production Adoption](docs/production-adoption.md) | production 도입 단계, release checklist, rollback 기준 |
| [Practical Scenarios](docs/scenarios.md) | 단일 모델, ensemble, GPU OOM, cache, LLM streaming, release 시나리오 |

핵심 운영 원칙은 단순합니다.

1. dev는 빠른 실험을 위해 `poll`을 허용합니다.
2. staging/prod는 `explicit` 모드로 load/unload를 통제합니다.
3. production은 image tag와 model repository revision을 고정합니다.
4. 모든 모델 변경은 config 검증, smoke test, metrics 확인을 거칩니다.
5. 성능 튜닝은 `config.pbtxt` 변경 전후의 latency, throughput, GPU memory를 숫자로 비교합니다.

---

## Triton 기능 ↔ 파일 매핑

### ⚡ 성능 최적화

| 기능 | 설정 위치 | 설명 |
|------|-----------|------|
| **Dynamic Batching** | `config.pbtxt` → `dynamic_batching { preferred_batch_size: [4, 8] max_queue_delay_microseconds: 100 }` | 여러 요청을 자동으로 묶어 배치 처리, throughput 극대화 |
| **Concurrent Execution** | `config.pbtxt` → `instance_group { count: 2 kind: KIND_GPU }` | 동일 모델을 GPU 인스턴스 여러 개로 병렬 실행 |
| **Model Warmup** | `config.pbtxt` → `model_warmup { name: "warmup" batch_size: 1 ... }` | 서버 시작 시 미리 워밍업하여 첫 요청 지연 제거 |
| **Response Cache** | `configs/cache/` + `config.pbtxt` → `response_cache { enable: true }` | 동일 입력에 대한 결과 캐싱 (local cache 기본, Redis는 플러그인 검증 후) |
| **Rate Limiter** | `configs/prod.txt` + `config.pbtxt` → `instance_group { rate_limiter { ... } }` | 리소스 기반 요청 제한 및 우선순위 조절 |

### 🗂️ 모델 관리

| 기능 | 설정 위치 | 설명 |
|------|-----------|------|
| **Model Repository** | `configs/repository/` (s3/gcs/azure) | 로컬·S3·GCS·Azure Blob 등 저장소 연동 |
| **Model Versioning** | `models/serving/{model}/1/`, `2/` ... | 여러 버전 동시 운용, 특정 버전 라우팅 가능 |
| **Dynamic Load/Unload** | `scripts/model_control/load.sh`, `unload.sh`, `reload.sh` | 런타임 모델 교체 (서버 재시작 불필요) |
| **Model Configuration** | `models/serving/**/config.pbtxt` | 입출력 shape, 배치 설정, 인스턴스 수 명세 |

### 🔗 Ensemble & BLS

| 기능 | 설정 위치 | 설명 |
|------|-----------|------|
| **Ensemble Pipeline** | `models/_templates/ensemble_pipeline/` / `models/serving/vision/object_detection/pipeline/` | 여러 모델을 DAG 형태로 연결 (전처리→추론→후처리) |
| **BLS** | `models/_templates/bls_model/1/model.py` | Python 백엔드 내에서 다른 모델 직접 호출, 분기 로직 구현 |

### 🌐 프로토콜 및 인터페이스

| 기능 | 설정 위치 | 설명 |
|------|-----------|------|
| **HTTP/REST** | `client/http_client.py` | KServe v2 Inference Protocol 표준 |
| **gRPC** | `client/grpc_client.py` | 고성능 바이너리 통신 |
| **Shared Memory** | `client/shared_memory_client.py` | CPU/GPU 공유 메모리로 데이터 복사 오버헤드 제거 |
| **Streaming** | `client/streaming_client.py` / `models/_templates/decoupled_streaming/` | 토큰 단위 스트리밍 응답 (LLM 서빙) |

### 📊 모니터링 & 관측성

| 기능 | 설정 위치 | 설명 |
|------|-----------|------|
| **Prometheus Metrics** | `monitoring/prometheus/scrape_config.yml` | throughput, latency, 큐 대기, GPU 사용률 수집 |
| **Alert Rules** | `monitoring/prometheus/triton_rules.yml` | 평균 latency > 200ms, error rate > 5%, GPU > 90% 등 알림 |
| **Grafana Dashboard** | `monitoring/grafana/triton_dashboard.json` | 사전 구성된 시각화 대시보드 |
| **OpenTelemetry Tracing** | `monitoring/otel/otel-collector-config.yaml` + `configs/tracing/otel.txt` | 요청 추적 (Jaeger / Zipkin 전달) |
| **Health Check** | `scripts/health_check.sh` / `/v2/health/live`, `/v2/health/ready` | 서버·모델 상태 확인 |
| **Statistics API** | `client/stats_client.py` / `GET /v2/models/{name}/stats` | 모델별 추론 횟수·큐 대기·연산 시간 상세 조회 |

### 🔒 기타

| 기능 | 설정 위치 | 설명 |
|------|-----------|------|
| **TLS/SSL** | `configs/tls/grpc_tls.txt`, `configs/tls/http_tls.txt` | gRPC/HTTP 인증서 기반 보안 통신 |
| **CUDA MPS** | `configs/gpu/mps.md` | GPU 자원 공유 효율화 (Multi-Process Service) |
| **Kubernetes + Helm** | `deploy/helm/triton/` | `helm install/upgrade` 한 줄 배포, HPA·PDB 내장 |
| **Kustomize** | `deploy/k8s/overlays/` | 환경별 overlay 방식 K8s 배포 |

---

## 배포 방식 비교 (Helm vs Kustomize)

| | Helm (`deploy/helm/`) | Kustomize (`deploy/k8s/`) |
|---|---|---|
| **배포 명령** | `helm upgrade --install triton ...` | `kubectl apply -k overlays/prod` |
| **설정 방식** | `values.prod.yaml` 하나로 오버라이드 | 디렉토리별 overlay 파일 |
| **롤백** | `helm rollback triton` 내장 | 수동 처리 필요 |
| **패키지 공유** | Helm Registry로 배포 가능 | 불가 |
| **권장 상황** | 신규 프로젝트, 외부 공유 | 기존 Kustomize 인프라 사용 시 |

```bash
# Helm — dev
helm install triton deploy/helm/triton -f deploy/helm/triton/values.dev.yaml

# Helm — prod (이미지 태그 지정)
helm upgrade --install triton deploy/helm/triton \
  -f deploy/helm/triton/values.prod.yaml \
  --set image.tag=sha-abc1234

# Helm 롤백
helm rollback triton 1

# Kustomize — staging
kubectl apply -k deploy/k8s/overlays/staging
```

---

## 새 모델 추가 절차

1. **템플릿 복사** — 서빙 패턴에 맞는 템플릿 선택
   ```bash
   cp -r models/_templates/ensemble_pipeline/ models/serving/vision/my_model/
   ```

2. **config.pbtxt 수정** — input/output shape, backend, instance_group 등

3. **manifest.yaml 등록** — source→target 매핑 추가
   ```yaml
   - source: vision/my_model/inferencer
     target: my_model_inferencer
     tags: [gpu, tensorrt]
   ```

4. **빌드 & 테스트**
   ```bash
   ./scripts/build.sh --env dev
   docker compose -f deploy/docker/docker-compose.yml up -d
   pytest tests/smoke/
   ```

---

## 서빙 패턴 (Templates)

| 패턴 | 템플릿 경로 | 용도 |
|------|-------------|------|
| **단일 모델** | `models/_templates/single_model/` | ONNX, TensorRT, PyTorch 단독 서빙 |
| **Ensemble** | `models/_templates/ensemble_pipeline/` | 전처리→추론→후처리 DAG 파이프라인 |
| **BLS** | `models/_templates/bls_model/` | Python 내 다중 모델 호출, 분기 로직 |
| **Decoupled Streaming** | `models/_templates/decoupled_streaming/` | LLM 토큰 스트리밍 등 비동기 응답 |

---

## 환경별 배포

| 환경 | 설정 | 배포 방식 |
|------|------|-----------|
| **dev** | `--model-control-mode=poll` | `docker compose up` |
| **staging** | `--model-control-mode=explicit` | `helm install ... -f values.staging.yaml` 또는 `kubectl apply -k overlays/staging` |
| **prod** | explicit + cache + rate-limit + HPA | `helm upgrade --install ... -f values.prod.yaml` 또는 `kubectl apply -k overlays/prod` (수동 승인) |

---

## CI/CD 파이프라인

```
PR 생성 → ci-validate (lint, config 검증)
    ↓
main merge → ci-build-test (빌드 + smoke test + GHCR push)
    ↓
자동 → cd-staging (staging 배포 + integration test)
    ↓
수동 승인 → cd-production (prod 배포 + perf baseline 비교)
```

---

## Troubleshooting

실무에서 자주 만나는 문제와 해결 방법입니다.

### 모델 로드 실패

```
Error: failed to load model 'my_model'
```

**원인 및 해결**:
1. `config.pbtxt` 문법 오류 → `./scripts/validate.sh` 로 검증
2. 모델 파일 없음 → `./scripts/build.sh --env dev` 재실행 후 `model_repository/` 확인
3. 버전 디렉토리 누락 → `models/serving/my_model/1/` 디렉토리 존재 확인
4. backend 이름 오타 → `backend: "onnxruntime"` / `"tensorrt"` / `"python"` 철자 확인

---

### GPU Out of Memory (OOM)

```
CUDA out of memory. Tried to allocate ...
```

**원인 및 해결**:
1. `instance_group` count 줄이기
   ```protobuf
   instance_group [ { kind: KIND_GPU count: 1 } ]  # count 2→1
   ```
2. `rate_limiter` 설정으로 동시 실행 제한
   ```protobuf
   instance_group [{ rate_limiter { resources [{ name: "GPU" count: 1 }] } }]
   ```
3. 동적 배치 최대 배치 크기 축소 (`max_batch_size` 조정)

---

### 첫 요청이 느림 (Cold Start)

**원인**: GPU 커널 JIT 컴파일 지연

**해결**: `config.pbtxt`에 `model_warmup` 추가

```protobuf
model_warmup [
  {
    name: "warmup"
    batch_size: 1
    inputs { key: "INPUT" value: { data_type: TYPE_FP32 dims: [3, 224, 224] } }
  }
]
```

---

### Dynamic Batching 효과가 없음

**증상**: `execution_count ≈ inference_count` (배치 크기가 항상 1)

**해결**:
```protobuf
dynamic_batching {
  preferred_batch_size: [ 4, 8 ]
  max_queue_delay_microseconds: 5000   # 기본값 0 → 5ms로 늘려서 요청 대기
}
```

> **확인**: `python client/stats_client.py --model my_model` 으로 평균 배치 크기 모니터링

---

### gRPC 연결 오류

```
StatusCode.UNAVAILABLE: failed to connect to all addresses
```

**체크리스트**:
1. 포트 확인 — gRPC는 **8001** (HTTP 8000 아님)
2. TLS 미스매치 — 서버가 TLS인데 클라이언트가 plaintext인 경우
   ```python
   # TLS 사용 시
   channel = grpc.secure_channel("host:8001", grpc.ssl_channel_credentials())
   # TLS 미사용 시
   channel = grpc.insecure_channel("host:8001")
   ```
3. K8s 환경 — Service 이름으로 접근 (`triton:8001`, 외부는 Ingress 통해서)

---

### OpenTelemetry 트레이스가 Jaeger에 안 보임

**체크리스트**:
1. `configs/tracing/otel.txt` 내 `--trace-config` 값 확인
2. OTel Collector 실행 여부 — `docker compose ps` 또는 `kubectl get pods`
3. Triton 서버 기동 시 `--trace-config=triton-server-name=my-service` 인수 포함 확인
4. Jaeger UI 포트 — 기본 16686 (docker-compose.prod.yml에서 확인)

---

### 응답 캐시가 동작하지 않음

**체크리스트**:
1. 서버 인수에 `--cache-config=local,size=...` 지정 확인 (`configs/prod.txt`)
2. `config.pbtxt`에 `response_cache { enable: true }` 추가 확인
3. 입력이 완전히 동일한지 확인 (바이트 단위 일치해야 캐시 히트)
4. `python client/stats_client.py --model my_model` 으로 캐시 히트율 확인

---

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md) 참조
