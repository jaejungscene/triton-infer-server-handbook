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

## Streaming contract

Python backend template과 vLLM backend는 tensor 계약이 다릅니다.

```python
with TritonStreamingClient(TritonConfig(timeout=30)) as client:
    # models/_templates/decoupled_streaming 계약
    for token in client.stream_infer("decoupled_streaming", "Hello"):
        print(token, end="")

    # models/serving/nlp/llm vLLM 계약
    for token in client.stream_generate_vllm(
        "llm_vllm", "Hello", sampling_parameters={"temperature": 0.2}
    ):
        print(token, end="")
```

`TritonConfig.timeout`은 HTTP connect/network timeout, gRPC unary request deadline, streaming
idle timeout에 공통 적용됩니다. timeout이나 오류가 발생하면 해당 요청을 실패로 처리하며,
stream은 취소합니다. 한 streaming client instance에서는 stream을 직렬화하므로 높은
동시성이 필요하면 worker별 client를 생성합니다.

```python
config = TritonConfig(
    ssl=True,
    ssl_root_cert="/certs/ca.crt",
    ssl_cert="/certs/client.crt",  # mTLS certificate chain
    ssl_key="/certs/client.key",
    headers={"authorization": "Bearer ..."},
    timeout=10,
)
```

`ssl_cert`와 `ssl_key`는 반드시 함께 설정합니다. HTTP와 gRPC 모두 CA 검증을 기본으로 하며,
인증 header는 health/model/inference 요청과 stream handshake에 전달됩니다. production에서는
token을 코드나 저장소에 넣지 말고 secret manager에서 주입합니다.

Shared-memory client는 요청마다 충돌하지 않는 region 이름을 만들고 inference 종료 시 즉시
해제합니다. 반환 NumPy 배열은 region 해제 전에 복사되므로 client 수명과 독립적입니다.
출력 shape/dtype은 실제 모델 계약과 정확히 일치해야 하며 빈 tensor는 허용하지 않습니다.

Statistics API CLI는 10초 timeout을 기본 적용하고 HTTPS 인증서 검증을 유지합니다. 인증이
필요하면 token을 명령행 인수로 노출하지 말고 환경변수로 전달합니다. `inference_count`는 요청
수가 아니라 batch 원소를 포함한 inference 수이므로 `execution_count`로 나눈 값이 평균 동적
batch 크기입니다.

```bash
export TRITON_AUTH_TOKEN='<load from secret manager>'
python client/stats_client.py \
  --url https://triton.example.com --model text_classifier --timeout 5
unset TRITON_AUTH_TOKEN
```
