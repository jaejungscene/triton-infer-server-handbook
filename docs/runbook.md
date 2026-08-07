# Production 장애 대응 Runbook

이 문서는 production Triton 경보를 받은 당직자가 코드 구조를 몰라도 초기 진단, 완화,
rollback, 복구 확인을 수행할 수 있도록 정리한 절차입니다. 명령 예시는 Kustomize production
배포와 `app=triton-server` label을 기준으로 합니다.

## 대응 원칙

1. 사용자 영향, 시작 시각, 변경 시각을 먼저 기록합니다.
2. 원인을 찾기 전에도 영향이 커지면 traffic 우회나 검증된 release rollback을 우선합니다.
3. 여러 Pod를 동시에 재시작하지 않습니다. 증거를 남기고 한 장애 도메인씩 조치합니다.
4. 외부 Ingress는 inference API만 허용합니다. repository/trace/shared-memory 제어는
   `kubectl port-forward` 등 RBAC가 적용된 내부 운영 경로에서만 수행합니다.
5. secret, request payload, 고객 입력은 log나 incident 문서에 붙이지 않습니다.

## 최초 5분

```bash
export NS=production
export APP=triton-server

kubectl config current-context
kubectl get deployment,pod -n "${NS}" -l "app=${APP}" -o wide
kubectl rollout status deployment/triton-server -n "${NS}" --timeout=30s
kubectl get events -n "${NS}" --sort-by=.lastTimestamp | tail -n 30
```

다음 정보를 incident timeline에 남깁니다.

- alert 이름, firing 시작 시각, `environment`, `model`, `version`, `instance`
- 직전 배포 SHA와 실제 Pod `imageID` digest
- 영향받는 요청 비율, region/tenant/model 범위
- ready Pod 수, restart 수, Pending/Evicted/OOMKilled 여부
- error rate, 평균 latency, queue time, GPU memory의 변경 전후 값

실제 배포 바이트는 tag가 아니라 `imageID`의 digest로 확인합니다. Deployment의 기대 image도
`registry/repository@sha256:...` 형식이어야 하며, 아래 `imageID`와 digest가 일치해야 합니다.

```bash
kubectl get pods -n "${NS}" -l "app=${APP}" \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.containerStatuses[0].imageID}{"\n"}{end}'
kubectl rollout history deployment/triton-server -n "${NS}"
```

## 빠른 분류

| 관찰 | 우선 의심 | 첫 조치 |
|------|-----------|---------|
| `TritonMetricsMissing`, inference 정상 | scrape discovery/NetworkPolicy/Prometheus | target label과 monitoring 경로 확인 |
| `TritonServerDown`, ready 실패 | process crash, model load, node/GPU | Pod 상태·이전 log·event 확인 |
| error rate만 증가 | 입력 contract, model/backend, 새 release | 모델별 error와 직전 변경 비교 |
| queue와 latency 동반 증가 | GPU 포화, batch/rate limit, traffic 급증 | queue/GPU/replica/capacity 확인 |
| GPU memory 90% 이상 또는 OOM | instance 수, batch, 모델 중복 | 신규 traffic 제한 후 rollback 검토 |
| 특정 모델만 not ready | artifact/config/backend | Repository Index와 해당 Pod log 확인 |
| cache hit 0, miss만 증가 | 입력 변동 또는 cache 설정 누락 | model config와 server 인자 확인 |

## 내부 상태 확인

외부 Ingress의 `/v2/repository/**`는 차단되어 있으므로 아래 port-forward를 사용합니다.

```bash
kubectl port-forward -n "${NS}" service/triton-server 18000:8000 18002:8002
```

다른 터미널에서 상태를 확인합니다.

```bash
curl -fsS http://127.0.0.1:18000/v2/health/live
curl -fsS http://127.0.0.1:18000/v2/health/ready
curl -fsS -X POST -H 'Content-Type: application/json' \
  -d '{"ready":true}' http://127.0.0.1:18000/v2/repository/index
curl -fsS http://127.0.0.1:18002/metrics | grep '^nv_inference_' | head
```

Pod log는 현재와 직전 container를 모두 보되 입력 데이터가 포함되지 않았는지 확인한 뒤
공유합니다.

```bash
POD="$(kubectl get pod -n "${NS}" -l "app=${APP}" -o jsonpath='{.items[0].metadata.name}')"
kubectl logs -n "${NS}" "${POD}" --since=15m
kubectl logs -n "${NS}" "${POD}" --previous --tail=200
kubectl describe pod -n "${NS}" "${POD}"
```

## 경보별 절차

### ServerDown 또는 MetricsMissing

1. `live`와 `ready`를 분리합니다. live 실패는 process/node 문제, ready만 실패하면 모델 load나
   warmup 문제일 가능성이 큽니다.
2. `MetricsMissing`인데 inference가 정상이면 scrape target에 `service="triton"`,
   `environment="production"` label이 있는지 확인합니다.
3. monitoring namespace에서 8002 접근이 NetworkPolicy와 CNI에 의해 허용되는지 확인합니다.
4. 모든 Pod가 같은 digest에서 실패하고 직전 배포와 시각이 겹치면 full release rollback을
   우선합니다.

### HighErrorRate

1. `environment`, `model`, `version`별로 범위를 좁힙니다.
2. `/v2/models/{model}/stats`와 backend log에서 inference failure 증가를 확인합니다.
3. 새 input schema, dtype, shape, model version 변경과 시간상 상관관계를 확인합니다.
4. 일부 요청만 잘못되었으면 gateway에서 해당 traffic을 제한하고, 새 release 전체가 문제면
   이전 검증 release로 되돌립니다.

### HighLatency 또는 QueueBacklog

1. 요청률 증가, queue duration, compute duration, GPU utilization을 같은 5분 구간으로 봅니다.
2. queue만 증가하면 replica/instance/rate limiter 병목을, compute도 증가하면 모델/GPU 회귀를
   우선 의심합니다.
3. HPA가 `maxReplicas`에 도달했는지, 새 Pod가 GPU capacity 부족으로 Pending인지 확인합니다.
4. SLO를 회복하지 못하면 비핵심 traffic을 제한하거나 이전 release로 rollback합니다.

### GPU memory 또는 OOM

1. `kubectl describe pod`에서 `OOMKilled`와 node GPU allocation을 확인합니다.
2. 동일 GPU에 의도보다 많은 model instance/Pod가 배치되었는지 확인합니다.
3. 운영 중 `instance_group`이나 batch size를 즉석 변경하지 않습니다. 검증된 이전 release로
   우회하고 staging에서 memory peak를 재측정합니다.
4. PDB를 무시한 동시 재시작은 남은 정상 replica까지 제거할 수 있으므로 금지합니다.

## Rollback 선택

### 전체 release rollback: 기본 선택

GitHub `CD - Production`을 이전 정상 40자리 main commit SHA로 실행하고 `rollback=false`를
선택합니다. workflow는 그 revision의 manifest와 test를 checkout하고 SHA tag를 registry
digest로 해석한 뒤 적용합니다. Deployment image뿐 아니라 Git으로 관리되는 Ingress,
NetworkPolicy, HPA, PDB 변경도 이전 revision으로 맞추는 방법입니다.

### 긴급 Deployment rollback

`rollback=true` workflow 또는 다음 명령은 Kubernetes Deployment의 직전 ReplicaSet만
복원합니다.

```bash
kubectl rollout undo deployment/triton-server -n "${NS}"
kubectl rollout status deployment/triton-server -n "${NS}" --timeout=300s
```

이 방식은 Ingress, NetworkPolicy, HPA, PDB, Secret, 외부 PVC/object-storage model revision을
되돌리지 않습니다. 자동 smoke 실패 rollback도 같은 범위입니다. 응급 완화 후에는 반드시
이전 정상 SHA로 전체 release를 다시 적용해 cluster 상태를 일치시킵니다.

PVC/object storage를 선택한 환경은 별도 immutable model revision과 checksum을 먼저 복원한
뒤 Pod를 교체합니다. 가변 경로를 덮어써서 rollback하지 않습니다.

## 복구 확인

복구 완료는 다음 조건을 모두 충족할 때 선언합니다.

- 모든 Pod가 의도한 동일 image digest이며 Deployment available replica가 목표 수와 일치
- `/live`, `/ready`, 필수 `text_classifier` inference와 response-cache miss/hit gate 통과
- 10분 이상 error rate, queue time, GPU memory가 정상 범위
- `TritonServerDown`, `TritonMetricsMissing`이 resolved이고 새 critical alert 없음
- 외부 Ingress에서 inference는 성공하고 `/v2/repository/**`는 계속 차단
- 임시 traffic 제한, port-forward, debug 설정을 제거하고 incident timeline 갱신

```bash
pytest tests/smoke/ tests/integration/test_text_classifier.py \
  tests/integration/test_cache.py \
  --run-live \
  --triton-url http://127.0.0.1:18000 \
  --triton-metrics-url http://127.0.0.1:18002
```

## 사후 조치

- 원인, 탐지 시각, 완화 시각, 사용자 영향, 실제 image digest를 기록합니다.
- alert가 늦었으면 rule과 `promtool` test를 함께 수정합니다.
- 수동 조치가 필요했다면 CI/CD 또는 이 runbook에 재현 가능한 단계로 반영합니다.
- SLO 회귀라면 동일 GPU·입력·concurrency의 perf baseline을 다시 측정합니다.
- secret이나 고객 payload가 incident artifact에 포함되지 않았는지 확인합니다.
