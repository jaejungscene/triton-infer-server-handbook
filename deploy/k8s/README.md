# Kubernetes 배포 가이드

## 구조

```
k8s/
├── base/                # 공통 리소스 (Kustomize base)
│   ├── deployment.yaml  # Triton Deployment (GPU tolerations, resource limits)
│   ├── service.yaml     # ClusterIP (HTTP 8000, gRPC 8001, metrics 8002)
│   └── kustomization.yaml
├── components/
│   └── model-pvc/       # 외부 공유 repository를 선택할 때만 추가
├── ingress/             # prod overlay만 선택하는 HTTP/gRPC Ingress
└── overlays/
    ├── dev/             # replicas=1, resource 최소
    ├── dev-pvc/         # 외부 PVC component 조합 예시
    ├── staging/         # replicas=2
    ├── prod/            # replicas=3+, resource 최대, HPA
    ├── multi-gpu/       # 단일 노드 multi-GPU
    └── multi-node/      # 멀티 노드 + HPA + PDB
```

## 배포 명령

기본 Deployment의 `triton-server:local`은 placeholder입니다. CI가 만든 immutable image로
반드시 교체합니다. 해당 image에는 같은 commit에서 검증한 `/models`가 포함됩니다.

```bash
# Dev
kubectl apply -k deploy/k8s/overlays/dev

# Production
kubectl apply -f deploy/k8s/overlays/prod/namespace.yaml
htpasswd -c auth triton-client
kubectl create secret generic triton-ingress-basic-auth \
  --from-file=auth --namespace production
kubectl create secret tls triton-tls-secret \
  --cert=path/to/tls.crt --key=path/to/tls.key --namespace production
kubectl apply -k deploy/k8s/overlays/prod

# 상태 확인
kubectl get pods -n production -l app=triton-server
kubectl logs -n production -f deployment/triton-server

# 로컬 검증용 포트 포워딩
kubectl port-forward -n production svc/triton-server 8000:8000 8001:8001 8002:8002
```

## 트레이드오프

| 결정 | 선택 | 이유 |
|------|------|------|
| Service 타입 | ClusterIP | 클러스터 내부 안정 주소를 제공하고, 외부 노출은 Ingress/LB에서 분리 |
| Ingress | prod에서만 opt-in, HTTP/gRPC 분리 | dev의 우발적 외부 노출을 막고 protocol별 정책을 독립 적용 |
| 외부 인증 | TLS + basic auth 기본선 | Triton 자체에 tenant 인증이 없고 repository control API도 같은 port에 있기 때문 |
| rollout | `maxUnavailable: 0`, startup probe | 모델 로딩 중 재시작을 막고 기존 replica를 유지한 채 교체 |
| 종료 | 10초 preStop + 60초 grace period | endpoint 전파와 진행 중 요청 종료 시간을 확보 |
| Pod 권한 | SA token 미마운트, capability 제거 | Kubernetes API와 Linux capability가 필요 없는 추론 Pod의 공격 표면 축소 |
| NetworkPolicy | staging/prod ingress 제한 | prod는 ingress controller와 monitoring만 허용하고 staging만 같은 namespace 접근 허용 |
| 모델 전달 | image에 `/models` 포함 | server 코드·의존성·모델을 한 SHA로 승격하고 rollback 단순화 |
| 외부 PVC | component로 opt-in | image 크기가 과도하거나 별도 artifact 승격 체계가 있을 때만 사용 |
| GPU 스케줄링 | tolerations + nodeSelector | GPU 노드에만 배치 |
| 스케일링 | CPU HPA 기본선 | 별도 metrics adapter 없이 동작하며 GPU/queue metric은 adapter 검증 후 교체 |

## 환경별 Triton 인자

Kustomize는 base Deployment의 기본 인자를 환경별 overlay patch로 교체합니다.
Helm도 동일하게 `values*.yaml`의 `tritonArgs`로 실행 인자를 주입합니다. 두 방식 모두
프로젝트 이미지의 `ENTRYPOINT ["tritonserver"]`를 전제로 하므로 `args`에는 실행 파일
이름이 아니라 `--model-repository`부터 시작하는 플래그만 둡니다.

| 환경 | 핵심 인자 | 목적 |
|------|-----------|------|
| dev | `poll`, `--repository-poll-secs=5`, local cache 16 MiB | 모델 수정 후 빠른 재로드와 cache 계약 유지 |
| staging | `explicit`, `--load-model=*`, local cache 32 MiB | 검증된 저장소 전체를 시작 시 로드하고 이후 변경은 API로 제한 |
| prod | `explicit`, `--load-model=*`, cache, rate-limit | 배포 직후 서비스 가능한 모델 세트와 예측 가능한 변경 정책 확보 |

`explicit`만 지정하면 Triton은 시작 시 모델을 하나도 로드하지 않습니다. 이 예시는 배포
artifact 자체가 승인된 모델 세트라는 전제에서 `--load-model=*`를 함께 사용합니다. 모델별
승인을 따로 운영한다면 이 옵션을 제거하고 rollout 전에 `scripts/model_control/load.sh`로
필수 모델을 로드하는 별도 배포 단계를 두어야 합니다.

수십 GB 모델이라 image 배포가 비효율적이면 대상 overlay의 `kustomization.yaml`에 아래
component를 추가할 수 있습니다.

```yaml
components:
  - ../../components/model-pvc
```

이 component는 `/models`를 PVC로 덮어씁니다. 따라서 image build와 별개로 PVC에 immutable
revision을 원자적으로 배치하고, rollout 전에 파일 checksum과 필수 artifact를 검증하는
pipeline이 있어야 합니다. 빈 PVC를 연결하면 image 안 모델은 보이지 않습니다.

staging/prod의 NetworkPolicy는 ingress controller가 `ingress-nginx`, Prometheus가
`monitoring` namespace에 있다고 가정합니다. staging은 같은 namespace Pod의 직접 접근을
허용하지만 prod는 허용하지 않으므로 내부 호출도 gateway를 통하거나 별도 허용 rule을
명시해야 합니다. 실제 namespace가 다르면
`network_policy.yaml`의 selector를 먼저 바꾸십시오. 사용 중인 CNI가 NetworkPolicy를
지원하는지와 ingress controller가 host network를 쓰는지도 staging에서 확인해야 합니다.

prod와 multi-node HPA는 Kubernetes Metrics Server만으로 동작하는 CPU utilization을 안전한
기본선으로 사용하고, 5분 scale-down 안정화로 GPU 모델 cold start 중 축소 진동을 줄입니다.
GPU 사용률이나 Triton queue time으로 확장하려면 Prometheus Adapter/KEDA가 노출하는 metric
이름과 단위를 staging에서 검증한 뒤 HPA `metrics`를 교체합니다. DCGM exporter metric 이름을
adapter 설정 없이 HPA에 직접 적는 것만으로는 autoscaling이 동작하지 않습니다.

production Ingress는 `triton-ingress-basic-auth` Secret이 없으면 정상 인증을 구성할 수 없도록
의도했습니다. basic auth는 예제를 안전하게 시작하기 위한 최소선입니다. 실제 서비스는 API
gateway나 service mesh에서 OIDC/mTLS, tenant 권한, 요청 크기 제한, rate limit을 적용하고
`/v2/repository/**` 모델 제어 경로는 배포 주체만 호출할 수 있도록 inference 경로와 분리합니다.
