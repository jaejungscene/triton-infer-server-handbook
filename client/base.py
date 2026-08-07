"""
Base Client — 공통 설정 및 유틸리티

모든 Triton 클라이언트의 기반 클래스.
서버 URL, 타임아웃, health check 등 공통 로직을 포함합니다.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
import ssl
from urllib.parse import urlsplit

import numpy as np


_NUMPY_TO_TRITON_DTYPE = {
    np.dtype(np.bool_): "BOOL",
    np.dtype(np.uint8): "UINT8",
    np.dtype(np.uint16): "UINT16",
    np.dtype(np.uint32): "UINT32",
    np.dtype(np.uint64): "UINT64",
    np.dtype(np.int8): "INT8",
    np.dtype(np.int16): "INT16",
    np.dtype(np.int32): "INT32",
    np.dtype(np.int64): "INT64",
    np.dtype(np.float16): "FP16",
    np.dtype(np.float32): "FP32",
    np.dtype(np.float64): "FP64",
    np.dtype(object): "BYTES",
}


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
    headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be a finite value greater than zero")
        self._validate_endpoint("url", self.url)
        self._validate_endpoint("grpc_url", self.grpc_url)
        if bool(self.ssl_cert) != bool(self.ssl_key):
            raise ValueError("ssl_cert and ssl_key must be configured together")
        if not self.ssl and any((self.ssl_cert, self.ssl_key, self.ssl_root_cert)):
            raise ValueError("ssl must be enabled when TLS certificate paths are set")
        if not isinstance(self.headers, dict) or not all(
            isinstance(key, str)
            and isinstance(value, str)
            and key
            and "\r" not in key
            and "\n" not in key
            and "\r" not in value
            and "\n" not in value
            for key, value in self.headers.items()
        ):
            raise ValueError("headers must contain non-empty string keys and safe string values")
        self.headers = dict(self.headers)

    @staticmethod
    def _validate_endpoint(field_name: str, endpoint: str) -> None:
        if not isinstance(endpoint, str) or endpoint.strip() != endpoint or "://" in endpoint:
            raise ValueError(f"{field_name} must use host:port without a URL scheme")
        try:
            parsed = urlsplit(f"//{endpoint}")
            port = parsed.port
        except ValueError as error:
            raise ValueError(f"{field_name} must contain a valid host and port") from error
        if (
            not parsed.hostname
            or port is None
            or port <= 0
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(f"{field_name} must use host:port without path or credentials")


def numpy_to_triton_dtype(dtype: np.dtype) -> str:
    """Return the KServe datatype for a supported NumPy wire representation."""
    normalized = np.dtype(dtype)
    try:
        return _NUMPY_TO_TRITON_DTYPE[normalized]
    except KeyError as error:
        raise ValueError(
            f"Unsupported NumPy dtype for Triton inference: {normalized}"
        ) from error


def _read_optional_bytes(path: str) -> bytes | None:
    if not path:
        return None
    with open(path, "rb") as material_file:
        return material_file.read()


def grpc_tls_kwargs(config: TritonConfig) -> dict:
    """Build Python tritonclient gRPC TLS constructor arguments."""
    if not config.ssl:
        return {}
    return {
        "root_certificates": _read_optional_bytes(config.ssl_root_cert),
        "private_key": _read_optional_bytes(config.ssl_key),
        "certificate_chain": _read_optional_bytes(config.ssl_cert),
    }


def http_ssl_context(config: TritonConfig) -> ssl.SSLContext | None:
    """Build a verified HTTP TLS context, optionally with an mTLS identity."""
    if not config.ssl:
        return None
    context = ssl.create_default_context(cafile=config.ssl_root_cert or None)
    if config.ssl_cert:
        context.load_cert_chain(config.ssl_cert, config.ssl_key)
    return context


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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False
