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

import numpy as np

from .base import TritonConfig

_LOGGER = logging.getLogger(__name__)


class TritonSHMClient:
    """CPU/CUDA Shared Memory 기반 Triton 클라이언트"""

    def __init__(self, config: TritonConfig | None = None, use_cuda: bool = False):
        self.config = config or TritonConfig()
        self.use_cuda = use_cuda

        import tritonclient.http as httpclient

        self._client = httpclient.InferenceServerClient(
            url=self.config.url,
            verbose=self.config.verbose,
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

        inputs = []
        outputs = []

        # ── Input SHM 등록 ──
        for name, data in input_data.items():
            region_name = f"input_{name}"
            byte_size = data.nbytes

            # SHM 영역 생성
            if self.use_cuda:
                shm_handle = self._shm.create_shared_memory_region(region_name, byte_size, 0)
                self._shm.set_shared_memory_region(shm_handle, [data])
            else:
                shm_handle = self._shm.create_shared_memory_region(region_name, f"/{region_name}", byte_size)
                self._shm.set_shared_memory_region(shm_handle, [data])

            # Triton에 등록
            if self.use_cuda:
                self._client.register_cuda_shared_memory(region_name, self._shm.get_raw_handle(shm_handle), 0, byte_size)
            else:
                self._client.register_system_shared_memory(region_name, f"/{region_name}", byte_size)

            self._registered_regions.append(region_name)
            self._region_handles[region_name] = shm_handle

            # Input 구성
            inp = self._httpclient.InferInput(name, list(data.shape), self._numpy_to_triton_dtype(data.dtype))
            inp.set_shared_memory(region_name, byte_size)
            inputs.append(inp)

        # ── Output SHM 등록 ──
        for name in output_names:
            region_name = f"output_{name}"
            shape = output_shapes[name]
            dtype = output_dtypes[name]
            byte_size = int(np.prod(shape) * np.dtype(dtype).itemsize)

            if self.use_cuda:
                shm_handle = self._shm.create_shared_memory_region(region_name, byte_size, 0)
                self._client.register_cuda_shared_memory(region_name, self._shm.get_raw_handle(shm_handle), 0, byte_size)
            else:
                shm_handle = self._shm.create_shared_memory_region(region_name, f"/{region_name}", byte_size)
                self._client.register_system_shared_memory(region_name, f"/{region_name}", byte_size)

            self._registered_regions.append(region_name)
            self._region_handles[region_name] = shm_handle

            out = self._httpclient.InferRequestedOutput(name)
            out.set_shared_memory(region_name, byte_size)
            outputs.append(out)

        # ── 추론 ──
        result = self._client.infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs,
            model_version=model_version,
        )

        return {name: result.as_numpy(name) for name in output_names}

    def cleanup(self):
        """등록된 모든 SHM 영역 해제 — 반드시 호출"""
        for region_name in self._registered_regions:
            try:
                if self.use_cuda:
                    self._client.unregister_cuda_shared_memory(region_name)
                else:
                    self._client.unregister_system_shared_memory(region_name)
            except Exception as exc:
                _LOGGER.debug("Failed to unregister shared memory region %s: %s", region_name, exc)

            shm_handle = self._region_handles.get(region_name)
            if shm_handle is None:
                continue

            try:
                self._shm.destroy_shared_memory_region(shm_handle)
            except Exception as exc:
                _LOGGER.debug("Failed to destroy shared memory region %s: %s", region_name, exc)

        self._registered_regions.clear()
        self._region_handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False

    def close(self):
        """SHM 영역과 Triton HTTP client를 정리"""
        self.cleanup()
        if self._client and hasattr(self._client, "close"):
            try:
                self._client.close()
            except Exception as close_exc:
                _LOGGER.debug("Failed to close Triton shared memory client: %s", close_exc)

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
        mapping = {
            np.float32: "FP32",
            np.float16: "FP16",
            np.int32: "INT32",
            np.int64: "INT64",
            np.uint8: "UINT8",
        }
        if dtype.type not in mapping:
            raise ValueError(f"Unsupported NumPy dtype for Triton shared memory inference: {dtype}")
        return mapping[dtype.type]
