"""
HTTP Client — REST / KServe v2 Protocol

사용 예:
    from client.http_client import TritonHTTPClient, TritonConfig

    config = TritonConfig(url="localhost:8000")
    client = TritonHTTPClient(config)

    result = client.infer_numpy(
        model_name="resnet50",
        input_data={"input": numpy_array},
        output_names=["output"],
    )
"""

import numpy as np

from .base import (
    BaseTritonClient,
    TritonConfig,
    http_ssl_context,
    numpy_to_triton_dtype,
)


class TritonHTTPClient(BaseTritonClient):
    """HTTP/REST 기반 Triton 클라이언트"""

    def __init__(self, config: TritonConfig | None = None):
        super().__init__(config or TritonConfig())
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

    def is_server_ready(self) -> bool:
        return self._client.is_server_ready(headers=self.config.headers)

    def is_model_ready(self, model_name: str, model_version: str = "") -> bool:
        return self._client.is_model_ready(
            model_name, model_version, headers=self.config.headers
        )

    def get_model_config(self, model_name: str, model_version: str = ""):
        return self._client.get_model_config(
            model_name, model_version, headers=self.config.headers
        )

    def infer(self, model_name: str, inputs: list, outputs: list, **kwargs):
        return self._client.infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs,
            headers=self.config.headers,
            request_id=kwargs.get("request_id", ""),
            model_version=kwargs.get("model_version", ""),
        )

    def infer_numpy(
        self,
        model_name: str,
        input_data: dict[str, np.ndarray],
        output_names: list[str],
        model_version: str = "",
    ) -> dict[str, np.ndarray]:
        """NumPy 배열로 간편 추론 — 가장 흔한 사용 패턴"""
        inputs = []
        for name, data in input_data.items():
            inp = self._httpclient.InferInput(
                name, list(data.shape), numpy_to_triton_dtype(data.dtype)
            )
            inp.set_data_from_numpy(data)
            inputs.append(inp)

        outputs = [self._httpclient.InferRequestedOutput(name) for name in output_names]

        result = self.infer(model_name, inputs, outputs, model_version=model_version)

        return {name: result.as_numpy(name) for name in output_names}
