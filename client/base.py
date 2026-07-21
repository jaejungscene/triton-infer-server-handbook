"""
Base Client — 공통 설정 및 유틸리티

모든 Triton 클라이언트의 기반 클래스.
서버 URL, 타임아웃, health check 등 공통 로직을 포함합니다.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class TritonConfig:
    """Triton 서버 연결 설정"""

    url: str = "localhost:8000"
    grpc_url: str = "localhost:8001"
    timeout: float = 30.0
    verbose: bool = False
    ssl: bool = False
    ssl_cert: str = ""
    ssl_key: str = ""
    ssl_root_cert: str = ""
    headers: dict = field(default_factory=dict)


class BaseTritonClient(ABC):
    """Triton 클라이언트 추상 기반 클래스"""

    def __init__(self, config: TritonConfig):
        self.config = config
        self._client = None

    @abstractmethod
    def is_server_ready(self) -> bool:
        """서버가 요청을 받을 준비가 되었는지 확인"""

    @abstractmethod
    def is_model_ready(self, model_name: str, model_version: str = "") -> bool:
        """특정 모델이 서빙 가능한 상태인지 확인"""

    @abstractmethod
    def infer(self, model_name: str, inputs: list, outputs: list, **kwargs):
        """추론 요청 전송"""

    def close(self):
        """클라이언트 리소스 정리"""
        if self._client and hasattr(self._client, "close"):
            self._client.close()
        self._client = None
