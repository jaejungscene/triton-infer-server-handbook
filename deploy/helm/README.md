# Triton Helm Chart

NVIDIA Triton Inference Server를 Kubernetes에 배포하는 Helm 차트입니다.

## 사전 준비

```bash
# Helm 설치 확인
helm version

# GPU 노드 레이블 확인
kubectl get nodes --show-labels | grep nvidia
```

## 빠른 시작

```bash
# dev 환경 (최초 설치)
helm install triton deploy/helm/triton \
  -f deploy/helm/triton/values.dev.yaml \
  --namespace triton --create-namespace

# staging 환경
helm install triton deploy/helm/triton \
  -f deploy/helm/triton/values.staging.yaml \
  --namespace triton

# prod 환경 (특정 이미지 태그 고정)
helm upgrade --install triton deploy/helm/triton \
  -f deploy/helm/triton/values.prod.yaml \
  --set image.repository=ghcr.io/your-org/triton-infer-server-handbook \
  --set image.tag=sha-abc1234 \
  --namespace triton
```

## 환경별 values 파일

| 파일 | replicas | 모델 제어 | HPA | PDB | 설명 |
|------|----------|-----------|-----|-----|------|
| `values.dev.yaml` | 1 | poll (파일 감지) | 없음 | 없음 | 개발용, 16 MiB cache, verbose 로그 |
| `values.staging.yaml` | 2 | explicit | 없음 | 사용 (min 1) | 스테이징 |
| `values.prod.yaml` | 3 | explicit + 캐시 | 사용 (3~10) | 사용 (min 2) | 프로덕션 |

`tritonArgs`는 Kubernetes `args`에 그대로 렌더링되며 이미지의
`ENTRYPOINT ["tritonserver"]`에 전달됩니다. 따라서 실행 파일 이름을 반복하지 않고
`--model-repository=/models` 같은 서버 플래그만 나열합니다. 다른 이미지를 쓸 때는 그
이미지가 같은 entrypoint 계약을 제공하는지 배포 전에 확인합니다.

staging/prod는 `explicit`와 `--load-model=*`를 함께 사용합니다. `explicit`만 지정하면
시작 시 아무 모델도 로드되지 않으므로, 저장소 전체가 배포 승인 단위인 환경에서만 이
조합을 사용합니다. 모델을 개별 승인하는 조직은 `--load-model=*`를 제거하고 배포 파이프라인의
명시적 load 단계가 성공한 뒤에만 트래픽을 연결해야 합니다.

기본 활성 모델인 `text_classifier`가 `response_cache`를 사용하므로 모든 환경에 local cache를
설정합니다. 용량은 dev 16 MiB, staging 32 MiB, prod 64 MiB의 예시값이며 실제 hit ratio와
메모리 예산을 측정해 조정합니다.

HPA가 활성화되면 Deployment의 `spec.replicas`는 렌더링하지 않습니다. Helm upgrade가
HPA가 계산한 replica 수를 다시 기본값으로 덮지 않게 하기 위한 설정입니다. startup probe가
통과하기 전에는 liveness probe가 동작하지 않으며, 종료 시에는 preStop과 grace period로
endpoint 전파 및 진행 중 요청 정리 시간을 확보합니다.

기본 chart는 `scripts/build.sh` 결과를 `/models`에 포함한 release image를 전제로 하며 PVC를
마운트하지 않습니다. private registry나 외부 모델 PVC는 values로 명시합니다.

```yaml
imagePullSecrets:
  - name: registry-credentials
persistence:
  enabled: true
  existingClaim: shared-model-repository
envFrom:
  - secretRef:
      name: triton-cloud-credentials
```

PVC를 켜면 image의 `/models`가 가려집니다. PVC revision을 image SHA와 별도로 추적·검증하고
원자적으로 게시하는 pipeline이 없는 경우에는 기본 image-bundled 방식을 유지합니다.

staging/prod는 hostname 기준 topology spread를 기본 활성화합니다. GPU 여유 노드가 적은
클러스터에서는 `whenUnsatisfiable: ScheduleAnyway`이므로 배포를 막지는 않으며, 강제 분산이
필요하면 `DoNotSchedule`로 바꾸고 GPU 용량을 먼저 확보합니다.

staging/prod는 NetworkPolicy도 활성화합니다. 기본값은 같은 namespace의 Pod, 이름이
`ingress-nginx`인 namespace의 HTTP/gRPC, 이름이 `monitoring`인 namespace의 metrics 접근만
허용합니다. 클러스터 구성이 다르면 `networkPolicy.*NamespaceSelector`를 배포 전에
조정해야 하며, 정책을 적용한 뒤 inference와 scrape를 각각 확인합니다.

## 주요 배포 명령어

```bash
# 현재 배포 상태 확인
helm status triton -n triton

# 설정값 확인
helm get values triton -n triton

# 이미지 태그만 교체 (무중단 롤링 업데이트)
helm upgrade triton deploy/helm/triton \
  -f deploy/helm/triton/values.prod.yaml \
  --set image.repository=ghcr.io/your-org/triton-infer-server-handbook \
  --set image.tag=sha-newversion \
  --namespace triton

# 이전 버전으로 롤백
helm rollback triton 1 -n triton

# 배포 히스토리 확인
helm history triton -n triton

# 삭제
helm uninstall triton -n triton
```

## 템플릿 확인 (dry-run)

```bash
# 렌더링 결과 미리 보기
helm template triton deploy/helm/triton/ \
  -f deploy/helm/triton/values.prod.yaml

# K8s API 서버 검증 (클러스터 연결 필요)
helm install triton deploy/helm/triton/ \
  -f deploy/helm/triton/values.prod.yaml \
  --dry-run --debug
```

## Kustomize와의 비교

| | 이 Helm Chart | deploy/k8s/ (Kustomize) |
|---|---|---|
| 배포 | `helm upgrade --install` | `kubectl apply -k` |
| 롤백 | `helm rollback` (내장) | 수동 |
| 환경 설정 | values.yaml 오버라이드 | overlay 디렉토리 |
| 권장 상황 | 신규 프로젝트, 외부 공유 | 기존 Kustomize 인프라 |

## Secret 주입

클라우드 저장소(S3/GCS) 또는 TLS 사용 시 먼저 Secret을 생성하세요:

```bash
# deploy/k8s/base/secret-template.yaml 참고
kubectl create secret generic triton-cloud-credentials \
  --from-literal=AWS_ACCESS_KEY_ID=AKIAXXXXXXXX \
  --from-literal=AWS_SECRET_ACCESS_KEY=XXXXXXXX \
  --namespace triton
```
