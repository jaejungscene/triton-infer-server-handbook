# Custom Cache 확장 지점

Triton은 [TRITONCACHE API](https://github.com/triton-inference-server/core/blob/main/include/triton/core/tritoncache.h)를 통해 custom cache 구현을 지원합니다.

## 캐시 구현

- `local` — Triton 기본 인메모리 캐시 (프로세스 내)
- `redis` — 별도 plugin build와 release image 포함이 필요한 참고 구현

## Custom Cache 개발 절차

1. TRITONCACHE API 헤더의 인터페이스 구현
2. `libtritoncache_<name>.so` 빌드
3. `/opt/tritonserver/caches/<name>/` 에 배치
4. `--cache-config=<name>,<key>=<value>` 로 사용

## 사용 예

```bash
tritonserver \
  --cache-config=my_cache,endpoint=memcached:11211 \
  --model-repository=/models
```

## 참고

`docker-compose.prod.yml`의 `redis-cache` profile은 Redis 서버만 시작합니다. Redis cache를
사용하려면 plugin을 Triton image의 cache 경로에 포함하고 `--cache-config=redis,...` 인수로
교체한 image를 staging에서 먼저 검증합니다.

- [local_cache](https://github.com/triton-inference-server/local_cache) — 인메모리 캐시 구현
- [redis_cache](https://github.com/triton-inference-server/redis_cache) — Redis 캐시 구현
