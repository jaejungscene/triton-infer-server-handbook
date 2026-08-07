# Configs 디렉토리 — 서버 레벨 설정

Triton 서버 시작 시 사용되는 CLI 인자를 환경별 + 모듈별로 분리합니다.

## 구조

```
configs/
├── base.txt          # 모든 환경 공통 (로깅, strict config 등)
├── dev.txt           # 개발: poll 모드, 짧은 폴링 주기
├── staging.txt       # 스테이징: explicit 모드
├── prod.txt          # 운영: explicit + cache + rate-limit
├── tls/              # TLS/SSL 인증서 기반 보안 통신
├── tracing/          # OpenTelemetry / Triton trace 설정
├── cache/            # Response Cache (Local / Redis)
├── gpu/              # CUDA MPS, NUMA 최적화
└── repository/       # 클라우드 모델 저장소 (S3/GCS/Azure)
```

## 사용법과 역할

`configs/*.txt`는 Triton 서버 인자를 환경별/기능별로 설명하는 운영 기준 파일입니다.
현재 `scripts/build.sh`는 모델 repository를 생성하며, 이 설정 파일들을 자동 합성하지는
않습니다. 실제 서버 기동 인자는 아래 위치에 반영합니다.

| 실행 방식 | 설정 위치 |
|-----------|-----------|
| Docker dev/prod | `deploy/docker/docker-compose*.yml` |
| Helm | `deploy/helm/triton/values*.yaml` |
| Kustomize | `deploy/k8s/base`, `deploy/k8s/overlays/*` |

값을 바꿀 때는 `configs/*.txt`와 실제 배포 설정을 함께 갱신합니다.

## 운영 판단 기준

| 설정 | dev | staging | prod |
|------|-----|---------|------|
| `model-control-mode` | poll (자동 감지) | explicit | explicit |
| `response-cache` | on (local 16 MiB) | on (local 32 MiB) | on (local 64 MiB, redis는 별도 플러그인 검증 후) |
| `rate-limit` | off | off | on |
| `tls` | off | gateway/mesh에서 선택 | Kustomize Ingress termination, direct TLS는 opt-in |
| `tracing` | opt-in | collector 준비 후 opt-in | collector 준비 후 opt-in |

TLS와 tracing 행은 권장 운영 방향이며 현재 Compose/Helm/Kustomize가 `configs/` 파일을 자동으로
읽는다는 뜻이 아닙니다. 해당 인수, 인증서 mount, collector 또는 gateway 리소스를 실제 배포
manifest에 반영한 뒤 staging에서 연결을 검증해야 합니다.
