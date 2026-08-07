# 실무 시나리오

이 문서는 이 저장소를 공부하거나 실무에 적용할 때 따라갈 수 있는 짧은 시나리오입니다.
각 시나리오는 "언제 필요한가", "무엇을 바꾸나", "어떻게 확인하나"를 기준으로 봅니다.

## 시나리오 1. 단일 ONNX 모델을 서빙한다

상황:
이미 학습된 classifier가 있고 online API에서 낮은 latency로 호출해야 합니다.

작업:

1. `models/_templates/single_model/`을 복사합니다.
2. `config.pbtxt`에서 input/output name, shape, dtype을 실제 모델과 맞춥니다.
3. `1/model.onnx`를 version directory에 둡니다.
4. `models/serving/manifest.yaml`에 source/target/tags를 추가합니다.
5. `./scripts/build.sh --env dev --clean`으로 repository를 만듭니다.

확인:

- `./scripts/validate.sh`
- `docker compose --env-file .env.dev -f deploy/docker/docker-compose.yml up -d`
- `./scripts/health_check.sh`
- `python client/stats_client.py --model <model_name>`

주의:
`max_batch_size > 0`이면 input shape에 batch 차원을 제외한 shape만 적습니다. 클라이언트
요청에는 batch 차원이 포함됩니다.

## 시나리오 2. 전처리와 후처리를 서버 안으로 넣는다

상황:
클라이언트마다 이미지 resize, normalize, NMS 구현이 달라서 결과가 흔들립니다.

선택:

- 처리 순서가 고정이면 Ensemble
- 요청 내용에 따라 다른 모델을 호출해야 하면 BLS

작업:

1. `models/_templates/ensemble_pipeline/`을 복사합니다.
2. preprocessor/postprocessor는 Python backend로 구현합니다.
3. inferencer는 ONNX 또는 TensorRT backend를 사용합니다.
4. `pipeline/config.pbtxt`의 `input_map`, `output_map`을 실제 tensor name과 맞춥니다.
5. manifest에는 구성 모델과 pipeline 모델을 모두 등록합니다.

확인:

- 구성 모델 각각 `/v2/models/{name}/ready` 확인
- pipeline 모델 inference 확인
- `/v2/models/{pipeline}/stats`와 구성 모델 stats를 함께 확인

주의:
Ensemble model 자체는 실제 compute를 하지 않습니다. 병목은 구성 모델의 scheduler, backend,
CPU/GPU 자원에서 발생합니다.

## 시나리오 3. GPU OOM을 줄인다

상황:
여러 모델을 한 GPU에 올렸더니 peak traffic에서 OOM이 발생합니다.

작업 순서:

1. `instance_group.count`를 줄여 모델 동시 실행 수를 낮춥니다.
2. `--rate-limit=execution_count`를 켜고 `instance_group.rate_limiter.resources`를 설정합니다.
3. `dynamic_batching.max_queue_delay_microseconds`를 늘려 throughput을 올릴 수 있는지 확인합니다.
4. 그래도 부족하면 모델별 replica 또는 GPU pool을 분리합니다.

확인:

- `nv_gpu_memory_used_bytes / nv_gpu_memory_total_bytes`
- `nv_inference_queue_duration_us`
- `/v2/models/{name}/stats`의 `execution_count`

주의:
Rate limiter는 OOM을 줄일 수 있지만 queue time과 tail latency를 늘릴 수 있습니다. latency
SLO가 엄격한 모델은 별도 GPU 또는 별도 deployment로 분리하는 편이 낫습니다.

## 시나리오 4. 반복 요청을 캐싱한다

상황:
동일한 feature vector나 이미지 hash에 대해 같은 모델을 자주 호출합니다.

작업:

1. 서버 인자에 `--cache-config=local,size=<bytes>`를 추가합니다.
2. 모델 `config.pbtxt`에 `response_cache { enable: true }`를 추가합니다.
3. deterministic output인지 검증합니다.
4. staging에서 동일 입력을 반복 호출하고 cache metric을 확인합니다.

확인:

- `nv_cache_num_hits_per_model`
- `nv_cache_num_misses_per_model`
- cache hit일 때 compute latency 감소 여부

주의:
모델명, 버전, input tensor name/shape/dtype/data가 cache key에 포함됩니다. 같은 의미의 입력이라도
shape이나 dtype이 다르면 hit가 나지 않습니다.

## 시나리오 5. LLM token streaming을 제공한다

상황:
사용자가 긴 답변 생성을 기다리지 않고 token 단위로 받게 하고 싶습니다.

작업:

1. 모델 config에 `model_transaction_policy { decoupled: true }`를 설정합니다.
2. Python backend, vLLM backend, TensorRT-LLM backend 중 하나를 선택합니다.
3. 클라이언트는 `client/streaming_client.py`처럼 gRPC streaming을 사용합니다.
4. timeout, cancellation, max token, backpressure 정책을 API contract에 넣습니다.

확인:

- 일반 HTTP infer가 아니라 streaming client로 테스트
- 마지막 response에 final flag가 전달되는지 확인
- client disconnect/cancel 시 backend가 중단 가능한지 확인

주의:
Decoupled 모델은 요청 1개에 0개, 1개, 여러 개의 응답을 보낼 수 있습니다. downstream은 일반
request/response 모델과 다르게 설계해야 합니다.

## 시나리오 6. Production release를 수행한다

상황:
staging 검증이 끝난 새 모델 revision을 production에 반영합니다.

작업:

1. model artifact revision과 container image digest를 고정합니다. `latest`나 mutable tag는 사용하지 않습니다.
2. GitHub workflow에는 main commit SHA를 입력하고 실제 Deployment가 해석된 digest를 쓰는지 확인합니다.
3. production은 `explicit` 모드로 기동합니다.
4. 배포 직후 smoke test와 metrics 확인을 수행합니다.
5. 문제가 있으면 Helm rollback 또는 이전 model repository revision으로 되돌립니다.

확인:

- `/v2/health/live`
- `/v2/health/ready`
- `POST /v2/repository/index`의 ready model 목록
- `/v2/models/{name}/stats`
- Prometheus alert 상태

주의:
production에서 repository를 직접 덮어쓰는 방식은 피합니다. artifact를 immutable하게 만들고,
배포 단위가 정확히 어떤 모델/이미지 조합인지 추적 가능해야 합니다. 외부 Ingress에서는
repository API가 차단되므로 ready model 확인과 load/unload는 내부 운영 endpoint를 사용합니다.
장애가 나면 [Production 장애 대응 Runbook](runbook.md)의 full release rollback과 긴급
Deployment rollback 차이를 먼저 확인합니다.
