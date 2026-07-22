# Kubernetes 배포 가이드

## 구조

```
k8s/
├── base/                # 공통 리소스 (Kustomize base)
│   ├── deployment.yaml  # Triton Deployment (GPU tolerations, resource limits)
│   ├── service.yaml     # ClusterIP (HTTP 8000, gRPC 8001, metrics 8002)
│   ├── pvc.yaml         # model_repository PVC
│   └── kustomization.yaml
└── overlays/
    ├── dev/             # replicas=1, resource 최소
    ├── staging/         # replicas=2
    ├── prod/            # replicas=3+, resource 최대, HPA
    ├── multi-gpu/       # 단일 노드 multi-GPU
    └── multi-node/      # 멀티 노드 + HPA + PDB
```

## 배포 명령

```bash
# Dev
kubectl apply -k deploy/k8s/overlays/dev

# Production
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
| PVC | ReadWriteMany | 여러 pod이 같은 모델 공유 |
| GPU 스케줄링 | tolerations + nodeSelector | GPU 노드에만 배치 |
| 스케일링 | HPA (GPU utilization) | GPU 사용률 기반 자동 스케일링 |

## 환경별 Triton 인자

Kustomize는 base Deployment의 기본 인자를 환경별 overlay patch로 교체합니다.
Helm도 동일하게 `values*.yaml`의 `tritonArgs`로 실행 인자를 주입합니다.

| 환경 | 핵심 인자 | 목적 |
|------|-----------|------|
| dev | `--model-control-mode=poll`, `--repository-poll-secs=5` | 모델 수정 후 빠른 재로드 |
| staging | `--model-control-mode=explicit`, `--log-verbose=1` | 운영과 같은 로드 정책으로 검증 |
| prod | explicit, cache, rate-limit, thread count | 예측 가능한 배포와 처리량 튜닝 |
