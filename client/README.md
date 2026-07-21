# Client Libraries — Triton 추론 클라이언트

## 프로토콜 선택 기준

| 프로토콜 | 사용 시나리오 | 장점 | 단점 |
|----------|-------------|------|------|
| **HTTP/REST** | 첫 연동, 디버깅, 저빈도 | 간편, curl 호환, 방화벽 친화 | gRPC 대비 오버헤드 |
| **gRPC** | 고성능 서비스 간 통신 | 낮은 latency, 바이너리 직렬화 | 인프라 설정 필요 |
| **gRPC Streaming** | LLM 토큰 스트리밍 | 실시간 응답, decoupled 모델 | HTTP 불가 |
| **Shared Memory** | 같은 노드, 극한 성능 | 복사 오버헤드 제거 | 같은 머신 필수 |
| **Statistics API** | 모델 성능 분석, 디버깅 | 큐/연산 단계별 시간 조회 | 추론 성능에 영향 없음 |

## 설치

```bash
python -m pip install -r requirements-integration.txt

# 개별 설치가 필요하면 Triton 서버 버전과 맞춰 고정합니다.
# 예: python -m pip install "tritonclient[grpc]==2.49.0"
```

## 클라이언트 목록

| 파일 | 설명 |
|------|------|
| `base.py` | 공통 추상 클라이언트 (설정, 유틸리티) |
| `http_client.py` | REST / KServe v2 프로토콜 |
| `grpc_client.py` | gRPC 고성능 클라이언트 |
| `async_client.py` | 비동기 (asyncio) 클라이언트 |
| `streaming_client.py` | gRPC decoupled streaming (LLM용) |
| `shared_memory_client.py` | CPU/CUDA Shared Memory 클라이언트 |
| `stats_client.py` | Statistics API — 모델별 추론 통계 조회 |
