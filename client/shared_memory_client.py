"""
Shared Memory Client — CPU/CUDA 공유 메모리 클라이언트

같은 노드에서 데이터 복사 오버헤드를 제거하여 latency를 최소화합니다.

라이프사이클:
    1. SHM 영역 생성/등록 (register)
    2. 입력 데이터를 SHM에 기록
    3. Triton 추론 요청 (SHM 참조로 전달)
    4. 결과를 SHM에서 읽기
    5. SHM 영역 해제 (unregister + cleanup)

⚠️ 주의:
    - 같은 머신에서만 사용 가능 (네트워크 통신 불가)
    - CUDA SHM은 GPU 메모리 사용 → 메모리 관리 필수
    - 반드시 unregister로 정리해야 메모리 누수 방지

사용 예:
    from client.shared_memory_client import TritonSHMClient, TritonConfig

    config = TritonConfig(url="localhost:8000")
    with TritonSHMClient(config, use_cuda=False) as client:
        result = client.infer_with_shm(
            model_name="resnet50",
            input_data={"input": numpy_array},
            output_names=["output"],
            output_shapes={"output": (1, 1000)},
            output_dtypes={"output": np.float32},
        )
"""

import logging
import math
import uuid

import numpy as np

from .base import (
    TritonConfig,
    collect_numpy_outputs,
    http_ssl_context,
    numpy_to_triton_dtype,
    validate_numpy_request,
)

_LOGGER = logging.getLogger(__name__)


class TritonSHMClient:
    """CPU/CUDA Shared Memory 기반 Triton 클라이언트"""

    def __init__(self, config: TritonConfig | None = None, use_cuda: bool = False):
        self.config = config or TritonConfig()
        self.use_cuda = use_cuda

        import tritonclient.http as httpclient

        ssl_context = http_ssl_context(self.config)
        self._client = httpclient.InferenceServerClient(
            url=self.config.url,
            verbose=self.config.verbose,
            connection_timeout=self.config.timeout,
            network_timeout=self.config.timeout,
            ssl=self.config.ssl,
            ssl_context_factory=(lambda: ssl_context) if ssl_context else None,
        )
        self._httpclient = httpclient

        if use_cuda:
            import tritonclient.utils.cuda_shared_memory as cuda_shm

            self._shm = cuda_shm
        else:
            import tritonclient.utils.shared_memory as cpu_shm

            self._shm = cpu_shm

        self._registered_regions: list[str] = []
        self._region_handles: dict[str, object] = {}
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("shared-memory client is closed")

    def _track_region(self, region_name: str, shm_handle: object) -> None:
        self._registered_regions.append(region_name)
        self._region_handles[region_name] = shm_handle

    def infer_with_shm(
        self,
        model_name: str,
        input_data: dict[str, np.ndarray],
        output_names: list[str],
        output_shapes: dict[str, tuple],
        output_dtypes: dict[str, np.dtype],
        model_version: str = "",
    ) -> dict[str, np.ndarray]:
        """Shared Memory를 사용한 추론"""
        self._ensure_open()
        validate_numpy_request(model_name, input_data, output_names)
        if set(output_shapes) != set(output_names) or set(output_dtypes) != set(
            output_names
        ):
            raise ValueError(
                "Output metadata keys must exactly match output_names: "
                f"shapes={sorted(output_shapes)}, dtypes={sorted(output_dtypes)}"
            )

        prepared_inputs = {}
        for name, data in input_data.items():
            self._numpy_to_triton_dtype(data.dtype)
            prepared_inputs[name] = np.ascontiguousarray(data)

        output_byte_sizes = {}
        for name in output_names:
            shape = output_shapes[name]
            if not isinstance(shape, (tuple, list)) or not all(
                isinstance(dimension, int)
                and not isinstance(dimension, bool)
                and dimension > 0
                for dimension in shape
            ):
                raise ValueError(
                    f"Output {name} shape must contain only positive integer dimensions"
                )
            dtype = np.dtype(output_dtypes[name])
            self._numpy_to_triton_dtype(dtype)
            output_byte_sizes[name] = math.prod(shape) * dtype.itemsize

        inputs = []
        outputs = []
        request_regions = []
        request_prefix = f"triton_{uuid.uuid4().hex}"

        try:
            # ── Input SHM 등록 ──
            for index, (name, data) in enumerate(prepared_inputs.items()):
                region_name = f"{request_prefix}_input_{index}"
                byte_size = data.nbytes
                if byte_size <= 0:
                    raise ValueError(f"Input {name} must not be empty")

                if self.use_cuda:
                    shm_handle = self._shm.create_shared_memory_region(
                        region_name, byte_size, 0
                    )
                    self._track_region(region_name, shm_handle)
                    request_regions.append(region_name)
                    self._shm.set_shared_memory_region(shm_handle, [data])
                else:
                    shm_handle = self._shm.create_shared_memory_region(
                        region_name, f"/{region_name}", byte_size
                    )
                    self._track_region(region_name, shm_handle)
                    request_regions.append(region_name)
                    self._shm.set_shared_memory_region(shm_handle, [data])

                if self.use_cuda:
                    self._client.register_cuda_shared_memory(
                        region_name,
                        self._shm.get_raw_handle(shm_handle),
                        0,
                        byte_size,
                        headers=self.config.headers,
                    )
                else:
                    self._client.register_system_shared_memory(
                        region_name,
                        f"/{region_name}",
                        byte_size,
                        headers=self.config.headers,
                    )

                inp = self._httpclient.InferInput(
                    name, list(data.shape), self._numpy_to_triton_dtype(data.dtype)
                )
                inp.set_shared_memory(region_name, byte_size)
                inputs.append(inp)

            # ── Output SHM 등록 ──
            for index, name in enumerate(output_names):
                region_name = f"{request_prefix}_output_{index}"
                shape = output_shapes[name]
                dtype = output_dtypes[name]
                byte_size = output_byte_sizes[name]

                if self.use_cuda:
                    shm_handle = self._shm.create_shared_memory_region(
                        region_name, byte_size, 0
                    )
                else:
                    shm_handle = self._shm.create_shared_memory_region(
                        region_name, f"/{region_name}", byte_size
                    )

                self._track_region(region_name, shm_handle)
                request_regions.append(region_name)

                if self.use_cuda:
                    self._client.register_cuda_shared_memory(
                        region_name,
                        self._shm.get_raw_handle(shm_handle),
                        0,
                        byte_size,
                        headers=self.config.headers,
                    )
                else:
                    self._client.register_system_shared_memory(
                        region_name,
                        f"/{region_name}",
                        byte_size,
                        headers=self.config.headers,
                    )

                out = self._httpclient.InferRequestedOutput(name)
                out.set_shared_memory(region_name, byte_size)
                outputs.append(out)

            result = self._client.infer(
                model_name=model_name,
                inputs=inputs,
                outputs=outputs,
                model_version=model_version,
                headers=self.config.headers,
            )

            return {
                name: output.copy()
                for name, output in collect_numpy_outputs(
                    result, output_names
                ).items()
            }
        finally:
            self._cleanup_regions(request_regions)

    def _cleanup_regions(self, region_names):
        for region_name in reversed(region_names):
            if region_name not in self._region_handles:
                continue
            try:
                if self.use_cuda:
                    self._client.unregister_cuda_shared_memory(
                        region_name, headers=self.config.headers
                    )
                else:
                    self._client.unregister_system_shared_memory(
                        region_name, headers=self.config.headers
                    )
            except Exception as exc:
                _LOGGER.debug(
                    "Failed to unregister shared memory region %s: %s",
                    region_name,
                    exc,
                )

            shm_handle = self._region_handles.pop(region_name)
            try:
                self._shm.destroy_shared_memory_region(shm_handle)
            except Exception as exc:
                _LOGGER.debug(
                    "Failed to destroy shared memory region %s: %s", region_name, exc
                )
            try:
                self._registered_regions.remove(region_name)
            except ValueError:
                pass

    def cleanup(self):
        """등록된 모든 SHM 영역 해제 — 반드시 호출"""
        self._cleanup_regions(list(self._registered_regions))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False

    def close(self):
        """SHM 영역과 Triton HTTP client를 정리"""
        if self._closed:
            return
        self._closed = True
        self.cleanup()
        if self._client and hasattr(self._client, "close"):
            try:
                self._client.close()
            except Exception as close_exc:
                _LOGGER.debug("Failed to close Triton shared memory client: %s", close_exc)
        self._client = None

    def __del__(self):
        try:
            self.close()
        except Exception as exc:
            _LOGGER.debug("Failed to clean up Triton shared memory client: %s", exc)

    def cleanup_all(self):
        """현재 프로세스가 가진 모든 SHM 영역을 정리하는 수동 복구 도구"""
        self.cleanup()
        try:
            if hasattr(self._shm, "destroy_shared_memory_region_all"):
                self._shm.destroy_shared_memory_region_all()
        except Exception as exc:
            _LOGGER.debug("Failed to destroy all shared memory regions: %s", exc)

    @staticmethod
    def _numpy_to_triton_dtype(dtype: np.dtype) -> str:
        triton_dtype = numpy_to_triton_dtype(dtype)
        if triton_dtype == "BYTES":
            raise ValueError("BYTES tensors are not supported by this shared-memory wrapper")
        return triton_dtype
