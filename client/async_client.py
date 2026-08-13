"""Native asyncio gRPC client for unary Triton inference."""

import numpy as np

from .base import (
    TritonConfig,
    collect_numpy_outputs,
    grpc_tls_kwargs,
    numpy_to_triton_dtype,
    validate_numpy_request,
)


class TritonAsyncClient:
    """Use Triton's grpc.aio transport so cancellation reaches in-flight RPCs."""

    def __init__(self, config: TritonConfig | None = None):
        self.config = config or TritonConfig()
        import tritonclient.grpc as grpcclient
        import tritonclient.grpc.aio as grpc_aio

        self._grpcclient = grpcclient
        self._client = grpc_aio.InferenceServerClient(
            url=self.config.grpc_url,
            verbose=self.config.verbose,
            ssl=self.config.ssl,
            **grpc_tls_kwargs(self.config),
        )
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("async client is closed")

    async def is_server_ready(self) -> bool:
        self._ensure_open()
        return await self._client.is_server_ready(
            headers=self.config.headers,
            client_timeout=self.config.timeout,
        )

    async def is_model_ready(self, model_name: str, model_version: str = "") -> bool:
        self._ensure_open()
        return await self._client.is_model_ready(
            model_name,
            model_version,
            headers=self.config.headers,
            client_timeout=self.config.timeout,
        )

    async def infer_numpy(
        self,
        model_name: str,
        input_data: dict[str, np.ndarray],
        output_names: list[str],
        model_version: str = "",
    ) -> dict[str, np.ndarray]:
        """Run one native asynchronous inference request."""
        self._ensure_open()
        validate_numpy_request(model_name, input_data, output_names)
        inputs = []
        for name, data in input_data.items():
            infer_input = self._grpcclient.InferInput(
                name, list(data.shape), numpy_to_triton_dtype(data.dtype)
            )
            infer_input.set_data_from_numpy(data)
            inputs.append(infer_input)
        outputs = [
            self._grpcclient.InferRequestedOutput(name) for name in output_names
        ]
        result = await self._client.infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs,
            headers=self.config.headers,
            client_timeout=self.config.timeout,
            model_version=model_version,
        )
        return collect_numpy_outputs(result, output_names)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._client.close()

    async def __aenter__(self):
        self._ensure_open()
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        await self.close()
        return False
