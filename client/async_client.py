"""
Async Client — 비동기 (asyncio) Triton 클라이언트

FastAPI, aiohttp 등 비동기 웹 프레임워크와 연동 시 사용.

사용 예:
    import asyncio
    from client.async_client import TritonAsyncClient, TritonConfig

    async def main():
        config = TritonConfig(grpc_url="localhost:8001")
        client = TritonAsyncClient(config)

        result = await client.infer_numpy(
            model_name="resnet50",
            input_data={"input": numpy_array},
            output_names=["output"],
        )

    asyncio.run(main())
"""

import asyncio
from functools import partial

import numpy as np

from .base import TritonConfig
from .grpc_client import TritonGRPCClient


class TritonAsyncClient:
    """비동기 래퍼 — gRPC 클라이언트의 callback 기반 async_infer 활용"""

    def __init__(self, config: TritonConfig | None = None):
        self.config = config or TritonConfig()
        self._sync_client = TritonGRPCClient(self.config)

    async def is_server_ready(self) -> bool:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._sync_client.is_server_ready)

    async def is_model_ready(self, model_name: str, model_version: str = "") -> bool:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(self._sync_client.is_model_ready, model_name, model_version))

    async def infer_numpy(
        self,
        model_name: str,
        input_data: dict[str, np.ndarray],
        output_names: list[str],
        model_version: str = "",
    ) -> dict[str, np.ndarray]:
        """비동기 NumPy 추론"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            partial(
                self._sync_client.infer_numpy,
                model_name=model_name,
                input_data=input_data,
                output_names=output_names,
                model_version=model_version,
            ),
        )

    def close(self):
        self._sync_client.close()
