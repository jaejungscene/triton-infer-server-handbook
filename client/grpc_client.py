"""
gRPC Client — 고성능 바이너리 프로토콜

HTTP 대비 낮은 latency, 높은 throughput. 서비스 간 통신에 권장.

사용 예:
    from client.grpc_client import TritonGRPCClient, TritonConfig

    config = TritonConfig(grpc_url="localhost:8001")
    client = TritonGRPCClient(config)

    result = client.infer_numpy(
        model_name="resnet50",
        input_data={"input": numpy_array},
        output_names=["output"],
    )
"""

import numpy as np

from .base import BaseTritonClient, TritonConfig, grpc_tls_kwargs, numpy_to_triton_dtype


class TritonGRPCClient(BaseTritonClient):
    """gRPC 기반 Triton 클라이언트"""

    def __init__(self, config: TritonConfig | None = None):
        super().__init__(config or TritonConfig())
        import tritonclient.grpc as grpcclient

        self._client = grpcclient.InferenceServerClient(
            url=self.config.grpc_url,
            verbose=self.config.verbose,
            ssl=self.config.ssl,
            **grpc_tls_kwargs(self.config),
        )
        self._grpcclient = grpcclient

    def is_server_ready(self) -> bool:
        return self._client.is_server_ready(
            headers=self.config.headers,
            client_timeout=self.config.timeout,
        )

    def is_model_ready(self, model_name: str, model_version: str = "") -> bool:
        return self._client.is_model_ready(
            model_name,
            model_version,
            headers=self.config.headers,
            client_timeout=self.config.timeout,
        )

    def get_model_metadata(self, model_name: str, model_version: str = ""):
        return self._client.get_model_metadata(
            model_name,
            model_version,
            headers=self.config.headers,
            client_timeout=self.config.timeout,
        )

    def infer(self, model_name: str, inputs: list, outputs: list, **kwargs):
        return self._client.infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs,
            headers=self.config.headers,
            client_timeout=self.config.timeout,
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
        """NumPy 배열로 간편 추론"""
        inputs = []
        for name, data in input_data.items():
            inp = self._grpcclient.InferInput(
                name, list(data.shape), numpy_to_triton_dtype(data.dtype)
            )
            inp.set_data_from_numpy(data)
            inputs.append(inp)

        outputs = [self._grpcclient.InferRequestedOutput(name) for name in output_names]

        result = self.infer(model_name, inputs, outputs, model_version=model_version)

        return {name: result.as_numpy(name) for name in output_names}
