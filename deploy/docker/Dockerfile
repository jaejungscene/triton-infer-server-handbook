# =============================================================================
# Dockerfile — Triton Inference Server 운영 이미지
# =============================================================================
# 빌드:
#   docker build -f deploy/docker/Dockerfile -t triton-server .
#
# scripts/build.sh가 생성한 model_repository를 이미지에 포함
# image SHA가 server runtime과 model artifact를 함께 식별
# =============================================================================

ARG TRITON_VERSION=24.08
FROM nvcr.io/nvidia/tritonserver:${TRITON_VERSION}-py3

ARG VCS_REF=unknown
LABEL org.opencontainers.image.revision="${VCS_REF}"

# Python backend 의존성 설치
COPY requirements-model.txt /tmp/requirements-model.txt
RUN pip install --no-cache-dir -r /tmp/requirements-model.txt

# CI에서 검증·staging한 immutable model repository
COPY model_repository/ /models/

# 스크립트 복사 (health check 등)
COPY scripts/health_check.sh /opt/scripts/health_check.sh
RUN chmod +x /opt/scripts/health_check.sh

# Prometheus metrics
EXPOSE 8000 8001 8002

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/v2/health/ready || exit 1

ENTRYPOINT ["tritonserver"]
CMD ["--model-repository=/models"]
