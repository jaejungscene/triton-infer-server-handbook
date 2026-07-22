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
  --set image.tag=sha-abc1234 \
  --namespace triton
```

## 환경별 values 파일

| 파일 | replicas | 모델 제어 | HPA | PDB | 설명 |
|------|----------|-----------|-----|-----|------|
| `values.dev.yaml` | 1 | poll (파일 감지) | 없음 | 없음 | 개발용, verbose 로그 |
| `values.staging.yaml` | 2 | explicit | 없음 | 사용 (min 1) | 스테이징 |
| `values.prod.yaml` | 3 | explicit + 캐시 | 사용 (3~10) | 사용 (min 2) | 프로덕션 |

`tritonArgs`는 Kubernetes `args`에 그대로 렌더링됩니다. 따라서 첫 항목은
`tritonserver`여야 하며, 그 뒤에 `--model-repository=/models` 같은 서버 플래그를
나열합니다.

## 주요 배포 명령어

```bash
# 현재 배포 상태 확인
helm status triton -n triton

# 설정값 확인
helm get values triton -n triton

# 이미지 태그만 교체 (무중단 롤링 업데이트)
helm upgrade triton deploy/helm/triton \
  -f deploy/helm/triton/values.prod.yaml \
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
