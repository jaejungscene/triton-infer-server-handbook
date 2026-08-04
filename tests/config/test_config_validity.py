"""
test_config_validity.py — config.pbtxt 및 manifest.yaml 자동 검증

CI에서 자동 실행되어 잘못된 설정이 배포되는 것을 방지합니다.
"""

import os

import pytest

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False


@pytest.fixture
def models_dir(project_root):
    return os.path.join(project_root, "models")


@pytest.fixture
def serving_dir(project_root):
    return os.path.join(project_root, "models", "serving")


class TestConfigFiles:
    """config.pbtxt 파일 유효성 검사"""

    def _find_configs(self, base_dir):
        configs = []
        for root, _, files in os.walk(base_dir):
            for f in files:
                if f == "config.pbtxt":
                    configs.append(os.path.join(root, f))
        return configs

    def test_all_configs_exist(self, models_dir):
        """models/ 하위에 config.pbtxt 파일이 하나 이상 존재"""
        configs = self._find_configs(models_dir)
        assert len(configs) > 0, "No config.pbtxt files found"

    def test_configs_have_backend_or_platform(self, models_dir):
        """모든 config.pbtxt에 backend 또는 platform이 정의됨"""
        configs = self._find_configs(models_dir)
        for config_path in configs:
            with open(config_path) as f:
                content = f.read()
            # ensemble은 platform: "ensemble" 사용
            has_backend = "backend:" in content or 'platform:' in content
            assert has_backend, f"Missing backend/platform in {config_path}"

    def test_configs_balanced_brackets(self, models_dir):
        """모든 config.pbtxt의 괄호가 균형"""
        configs = self._find_configs(models_dir)
        for config_path in configs:
            with open(config_path) as f:
                content = f.read()

            # 주석 제거 (# 로 시작하는 줄)
            lines = [line for line in content.splitlines() if not line.strip().startswith("#")]
            clean = "\n".join(lines)

            assert clean.count("[") == clean.count("]"), f"Unbalanced [] in {config_path}"
            assert clean.count("{") == clean.count("}"), f"Unbalanced {{}} in {config_path}"

    def test_version_directories_exist(self, models_dir):
        """config.pbtxt가 있는 디렉토리에 버전 디렉토리(1/)가 존재"""
        configs = self._find_configs(models_dir)
        for config_path in configs:
            model_dir = os.path.dirname(config_path)
            # ensemble/pipeline은 1/.gitkeep만 필요
            with open(config_path) as f:
                content = f.read()
            if 'platform: "ensemble"' in content:
                continue  # ensemble은 버전 디렉토리 체크 스킵 가능

            version_dirs = [
                d for d in os.listdir(model_dir)
                if os.path.isdir(os.path.join(model_dir, d)) and d.isdigit()
            ]
            # configs/ 디렉토리는 버전이 아님
            assert len(version_dirs) > 0 or "configs" in os.listdir(model_dir), \
                f"No version directory in {model_dir}"

    def test_health_checks_use_repository_index_for_model_listing(self, project_root):
        checked_files = [
            os.path.join(project_root, "scripts", "health_check.sh"),
            os.path.join(project_root, "tests", "smoke", "test_smoke.py"),
        ]
        for checked_file in checked_files:
            with open(checked_file) as source_file:
                content = source_file.read()
            assert "/v2/repository/index" in content
            assert '"/v2/models"' not in content


@pytest.mark.skipif(not HAS_YAML, reason="PyYAML not installed")
class TestManifest:
    """manifest.yaml 정합성 검사"""

    def test_manifest_exists(self, serving_dir):
        manifest_path = os.path.join(serving_dir, "manifest.yaml")
        assert os.path.exists(manifest_path), "manifest.yaml not found"

    def test_manifest_sources_exist(self, serving_dir):
        """manifest의 모든 source 경로가 실제로 존재"""
        manifest_path = os.path.join(serving_dir, "manifest.yaml")
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)

        for model in manifest.get("models", []):
            source_path = os.path.join(serving_dir, model["source"])
            assert os.path.exists(source_path), \
                f"Source path not found: {model['source']} (expected: {source_path})"

    def test_manifest_targets_unique(self, serving_dir):
        """manifest의 target 이름이 모두 고유"""
        manifest_path = os.path.join(serving_dir, "manifest.yaml")
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)

        targets = [m["target"] for m in manifest.get("models", [])]
        assert len(targets) == len(set(targets)), \
            f"Duplicate targets found: {[t for t in targets if targets.count(t) > 1]}"

    def test_manifest_has_required_fields(self, serving_dir):
        """manifest의 모든 모델 항목에 source, target 필드가 존재"""
        manifest_path = os.path.join(serving_dir, "manifest.yaml")
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)

        for i, model in enumerate(manifest.get("models", [])):
            assert "source" in model, f"Model {i} missing 'source' field"
            assert "target" in model, f"Model {i} missing 'target' field"

    def test_enabled_manifest_models_have_runtime_payload(self, serving_dir):
        """enabled 모델은 Triton이 로드할 수 있는 런타임 payload를 포함해야 함"""
        manifest_path = os.path.join(serving_dir, "manifest.yaml")
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)

        for model in manifest.get("models", []):
            if not model.get("enabled", True):
                continue

            source_path = os.path.join(serving_dir, model["source"])
            required_files = model.get("required_files")
            if required_files:
                missing = [
                    rel_path
                    for rel_path in required_files
                    if not os.path.exists(os.path.join(source_path, rel_path))
                ]
                assert not missing, f"Enabled model {model['target']} missing files: {missing}"
                continue

            payload_candidates = [
                "1/model.py",
                "1/model.json",
                "1/model.onnx",
                "1/model.plan",
                "1/model.xgboost",
            ]
            assert any(
                os.path.exists(os.path.join(source_path, candidate))
                for candidate in payload_candidates
            ), f"Enabled model {model['target']} has no runtime payload"


@pytest.mark.skipif(not HAS_YAML, reason="PyYAML not installed")
class TestDeploymentRuntimeArgs:
    """Helm/Kustomize가 Triton 서버를 실행 가능한 인자로 렌더링하는지 검사"""

    def _load_yaml(self, path):
        with open(path) as f:
            return yaml.safe_load(f)

    def test_helm_values_start_with_tritonserver(self, project_root):
        values_dir = os.path.join(project_root, "deploy", "helm", "triton")
        for filename in (
            "values.yaml",
            "values.dev.yaml",
            "values.staging.yaml",
            "values.prod.yaml",
        ):
            values = self._load_yaml(os.path.join(values_dir, filename))
            args = values.get("tritonArgs", [])

            assert args, f"{filename} has no tritonArgs"
            assert args[0] == "tritonserver", \
                f"{filename} must keep tritonserver as the first Kubernetes arg"
            assert "--model-repository=/models" in args, \
                f"{filename} must set the model repository"
            assert "--allow-metrics=true" in args, \
                f"{filename} must expose Prometheus metrics"
            if "--model-control-mode=explicit" in args:
                assert "--load-model=*" in args, \
                    f"{filename} explicit mode must bootstrap the validated model set"

    def test_kustomize_env_overlays_patch_runtime_args(self, project_root):
        overlay_expectations = {
            "dev": {"--model-control-mode=poll"},
            "staging": {"--model-control-mode=explicit", "--load-model=*"},
            "prod": {
                "--model-control-mode=explicit",
                "--load-model=*",
                "--cache-config=local,size=67108864",
                "--rate-limit=execution_count",
            },
            "multi-gpu": {"--model-control-mode=explicit", "--load-model=*"},
            "multi-node": {"--model-control-mode=explicit", "--load-model=*"},
        }
        overlays_dir = os.path.join(project_root, "deploy", "k8s", "overlays")

        for overlay, expected_args in overlay_expectations.items():
            kustomization_path = os.path.join(overlays_dir, overlay, "kustomization.yaml")
            kustomization = self._load_yaml(kustomization_path)
            patch_paths = {
                patch["path"]
                for patch in kustomization.get("patches", [])
                if isinstance(patch, dict) and "path" in patch
            }
            assert "triton_args_patch.yaml" in patch_paths, \
                f"{overlay} overlay must patch Triton runtime args"

            patch_path = os.path.join(overlays_dir, overlay, "triton_args_patch.yaml")
            patch = self._load_yaml(patch_path)
            args = patch["spec"]["template"]["spec"]["containers"][0]["args"]

            assert args[0] == "tritonserver", \
                f"{overlay} overlay must keep tritonserver as the first arg"
            assert "--model-repository=/models" in args, \
                f"{overlay} overlay must set the model repository"
            assert expected_args.issubset(set(args)), \
                f"{overlay} overlay missing expected args: {expected_args - set(args)}"

    def test_restricted_overlays_include_network_policy(self, project_root):
        overlays_dir = os.path.join(project_root, "deploy", "k8s", "overlays")

        for overlay in ("staging", "prod"):
            overlay_dir = os.path.join(overlays_dir, overlay)
            kustomization = self._load_yaml(
                os.path.join(overlay_dir, "kustomization.yaml")
            )
            assert "network_policy.yaml" in kustomization.get("resources", [])

            policy = self._load_yaml(os.path.join(overlay_dir, "network_policy.yaml"))
            assert policy["kind"] == "NetworkPolicy"
            assert policy["spec"]["policyTypes"] == ["Ingress"]
            assert policy["spec"]["podSelector"]["matchLabels"]["app"] == \
                "triton-server"

    def test_http_and_grpc_use_separate_ingresses(self, project_root):
        base_dir = os.path.join(project_root, "deploy", "k8s", "base")
        http_ingress = self._load_yaml(os.path.join(base_dir, "ingress-http.yaml"))
        grpc_ingress = self._load_yaml(os.path.join(base_dir, "ingress-grpc.yaml"))

        assert "nginx.ingress.kubernetes.io/backend-protocol" not in \
            http_ingress["metadata"].get("annotations", {})
        assert grpc_ingress["metadata"]["annotations"][
            "nginx.ingress.kubernetes.io/backend-protocol"
        ] == "GRPC"

        http_port = http_ingress["spec"]["rules"][0]["http"]["paths"][0][
            "backend"
        ]["service"]["port"]["name"]
        grpc_port = grpc_ingress["spec"]["rules"][0]["http"]["paths"][0][
            "backend"
        ]["service"]["port"]["name"]
        assert http_port == "http"
        assert grpc_port == "grpc"

    def test_deployments_define_safe_rollout_and_startup(self, project_root):
        base_deployment = self._load_yaml(
            os.path.join(project_root, "deploy", "k8s", "base", "deployment.yaml")
        )
        spec = base_deployment["spec"]
        pod_spec = spec["template"]["spec"]
        container = pod_spec["containers"][0]

        assert spec["strategy"]["rollingUpdate"]["maxUnavailable"] == 0
        assert pod_spec["automountServiceAccountToken"] is False
        assert pod_spec["terminationGracePeriodSeconds"] >= 60
        assert container["startupProbe"]["failureThreshold"] >= 30
        assert container["securityContext"]["allowPrivilegeEscalation"] is False

        helm_values = self._load_yaml(
            os.path.join(project_root, "deploy", "helm", "triton", "values.yaml")
        )
        assert helm_values["updateStrategy"]["rollingUpdate"]["maxUnavailable"] == 0
        assert helm_values["automountServiceAccountToken"] is False
        assert helm_values["startupProbe"]["failureThreshold"] >= 30
