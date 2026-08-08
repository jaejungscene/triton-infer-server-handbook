"""
test_config_validity.py — config.pbtxt 및 manifest.yaml 자동 검증

CI에서 자동 실행되어 잘못된 설정이 배포되는 것을 방지합니다.
"""

import os
import re

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

    def test_manifest_targets_match_triton_model_names(self, serving_dir):
        """manifest target과 config.pbtxt name은 동일해야 repository load가 성공함"""
        manifest_path = os.path.join(serving_dir, "manifest.yaml")
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)

        for model in manifest.get("models", []):
            config_path = os.path.join(serving_dir, model["source"], "config.pbtxt")
            with open(config_path) as config_file:
                config = config_file.read()
            match = re.search(r'^\s*name\s*:\s*"([A-Za-z0-9._-]+)"', config, re.MULTILINE)
            assert match, f"Configured model name not found: {config_path}"
            assert match.group(1) == model["target"], (
                f"Manifest target {model['target']} does not match "
                f"config name {match.group(1)}"
            )

    def test_manifest_environments_are_supported(self, serving_dir):
        manifest_path = os.path.join(serving_dir, "manifest.yaml")
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)

        supported = {"dev", "staging", "prod"}
        for model in manifest.get("models", []):
            environments = model.get("environments", supported)
            assert environments, f"Empty environments for {model['target']}"
            assert set(environments).issubset(supported), (
                f"Unsupported environments for {model['target']}: {environments}"
            )

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

    def test_string_warmup_uses_generated_data(self, serving_dir):
        config_path = os.path.join(
            serving_dir, "nlp", "text_classifier", "config.pbtxt"
        )
        with open(config_path) as config_file:
            config = config_file.read()

        assert "model_warmup" in config
        assert "zero_data: true" in config
        assert "input_data_file" not in config


@pytest.mark.skipif(not HAS_YAML, reason="PyYAML not installed")
class TestDeploymentRuntimeArgs:
    """Helm/Kustomize가 Triton 서버를 실행 가능한 인자로 렌더링하는지 검사"""

    def _load_yaml(self, path):
        with open(path) as f:
            return yaml.safe_load(f)

    def test_helm_values_only_contain_entrypoint_arguments(self, project_root):
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
            assert args[0].startswith("--"), \
                f"{filename} must contain flags for the image entrypoint"
            assert "tritonserver" not in args, \
                f"{filename} must not repeat the image entrypoint"
            assert "--model-repository=/models" in args, \
                f"{filename} must set the model repository"
            assert "--allow-metrics=true" in args, \
                f"{filename} must expose Prometheus metrics"
            assert any(arg.startswith("--cache-config=") for arg in args), \
                f"{filename} must configure the enabled model response cache"
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

            assert args[0].startswith("--"), \
                f"{overlay} overlay must contain flags for the image entrypoint"
            assert "tritonserver" not in args, \
                f"{overlay} overlay must not repeat the image entrypoint"
            assert "--model-repository=/models" in args, \
                f"{overlay} overlay must set the model repository"
            assert any(arg.startswith("--cache-config=") for arg in args), \
                f"{overlay} must configure the enabled model response cache"
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
            expected_policy_types = (
                ["Ingress", "Egress"] if overlay == "prod" else ["Ingress"]
            )
            assert policy["spec"]["policyTypes"] == expected_policy_types
            assert policy["spec"]["podSelector"]["matchLabels"]["app"] == \
                "triton-server"

        prod_policy = self._load_yaml(
            os.path.join(overlays_dir, "prod", "network_policy.yaml")
        )
        assert not any(
            source == {"podSelector": {}}
            for rule in prod_policy["spec"]["ingress"]
            for source in rule.get("from", [])
        ), "production must not trust every pod in its namespace"
        assert prod_policy["spec"]["egress"] == [], \
            "production must default-deny outbound traffic"

    def test_production_has_hpa_and_pdb(self, project_root):
        prod_dir = os.path.join(project_root, "deploy", "k8s", "overlays", "prod")
        kustomization = self._load_yaml(os.path.join(prod_dir, "kustomization.yaml"))
        resources = set(kustomization["resources"])

        assert {"hpa.yaml", "pdb.yaml"}.issubset(resources)
        hpa = self._load_yaml(os.path.join(prod_dir, "hpa.yaml"))
        assert hpa["spec"]["minReplicas"] >= 3
        assert hpa["spec"]["behavior"]["scaleDown"][
            "stabilizationWindowSeconds"
        ] >= 300
        assert hpa["spec"]["metrics"][0]["resource"]["name"] == "cpu"

        pdb = self._load_yaml(os.path.join(prod_dir, "pdb.yaml"))
        assert pdb["spec"]["minAvailable"] >= 2

        helm_prod = self._load_yaml(
            os.path.join(
                project_root, "deploy", "helm", "triton", "values.prod.yaml"
            )
        )
        assert helm_prod["networkPolicy"]["allowSameNamespace"] is False
        assert helm_prod["networkPolicy"]["egress"] == {
            "enabled": True,
            "rules": [],
        }
        assert helm_prod["image"]["requireDigest"] is True
        assert helm_prod["topologySpread"]["enabled"] is True
        assert helm_prod["topologySpread"]["whenUnsatisfiable"] == \
            "DoNotSchedule"

        replica_patch = self._load_yaml(
            os.path.join(prod_dir, "replica_patch.yaml")
        )
        constraints = replica_patch["spec"]["template"]["spec"][
            "topologySpreadConstraints"
        ]
        assert constraints[0]["topologyKey"] == "kubernetes.io/hostname"
        assert constraints[0]["whenUnsatisfiable"] == "DoNotSchedule"

    def test_restricted_namespaces_enforce_pod_security_baseline(self, project_root):
        overlays_dir = os.path.join(project_root, "deploy", "k8s", "overlays")
        for environment in ("staging", "prod"):
            namespace = self._load_yaml(
                os.path.join(overlays_dir, environment, "namespace.yaml")
            )
            labels = namespace["metadata"]["labels"]
            assert labels["pod-security.kubernetes.io/enforce"] == "baseline"
            assert labels["pod-security.kubernetes.io/audit"] == "restricted"
            assert labels["pod-security.kubernetes.io/warn"] == "restricted"

    def test_ingress_is_opt_in_and_production_is_authenticated(self, project_root):
        base_dir = os.path.join(project_root, "deploy", "k8s", "base")
        base = self._load_yaml(os.path.join(base_dir, "kustomization.yaml"))
        assert not any("ingress" in resource for resource in base.get("resources", []))

        prod_dir = os.path.join(project_root, "deploy", "k8s", "overlays", "prod")
        prod = self._load_yaml(os.path.join(prod_dir, "kustomization.yaml"))
        assert "../../ingress" in prod.get("resources", [])

        for filename in ("ingress_http_patch.yaml", "ingress_grpc_patch.yaml"):
            patch = self._load_yaml(os.path.join(prod_dir, filename))
            annotations = patch["metadata"]["annotations"]
            assert annotations["nginx.ingress.kubernetes.io/ssl-redirect"] == "true"
            assert annotations["nginx.ingress.kubernetes.io/auth-type"] == "basic"
            assert annotations["nginx.ingress.kubernetes.io/auth-secret"] == \
                "triton-ingress-basic-auth"

    def test_http_and_grpc_use_separate_ingresses(self, project_root):
        ingress_dir = os.path.join(project_root, "deploy", "k8s", "ingress")
        http_ingress = self._load_yaml(
            os.path.join(ingress_dir, "ingress-http.yaml")
        )
        grpc_ingress = self._load_yaml(
            os.path.join(ingress_dir, "ingress-grpc.yaml")
        )

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

    def test_ingresses_expose_inference_api_only(self, project_root):
        ingress_dir = os.path.join(project_root, "deploy", "k8s", "ingress")
        http_ingress = self._load_yaml(
            os.path.join(ingress_dir, "ingress-http.yaml")
        )
        grpc_ingress = self._load_yaml(
            os.path.join(ingress_dir, "ingress-grpc.yaml")
        )

        http_paths = {
            (path["path"], path["pathType"])
            for path in http_ingress["spec"]["rules"][0]["http"]["paths"]
        }
        assert http_paths == {
            ("/v2", "Exact"),
            ("/v2/health", "Prefix"),
            ("/v2/models", "Prefix"),
        }

        grpc_paths = grpc_ingress["spec"]["rules"][0]["http"]["paths"]
        assert all(path["pathType"] == "Exact" for path in grpc_paths)
        assert all("Repository" not in path["path"] for path in grpc_paths)
        assert all("SharedMemory" not in path["path"] for path in grpc_paths)
        assert {path["path"].rsplit("/", 1)[-1] for path in grpc_paths} >= {
            "ModelInfer",
            "ModelStreamInfer",
            "ServerReady",
        }

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

    def test_production_compose_uses_release_image_contract(self, project_root):
        compose = self._load_yaml(
            os.path.join(
                project_root, "deploy", "docker", "docker-compose.prod.yml"
            )
        )
        triton = compose["services"]["triton"]

        assert triton["image"].startswith("${TRITON_IMAGE:?")
        assert "volumes" not in triton, \
            "production must use the model repository embedded in the release image"
        assert triton["command"][0].startswith("--")
        assert "tritonserver" not in triton["command"]
        assert triton["pull_policy"] == "always"

    def test_helm_production_requires_a_valid_image_digest(self, project_root):
        chart_dir = os.path.join(project_root, "deploy", "helm", "triton")
        deployment_path = os.path.join(chart_dir, "templates", "deployment.yaml")
        with open(deployment_path) as deployment_file:
            deployment = deployment_file.read()

        assert "image.requireDigest=true" in deployment
        assert 'regexMatch "^sha256:[0-9a-f]{64}$"' in deployment
        assert (
            'image: "{{ .Values.image.repository }}@{{ .Values.image.digest }}"'
            in deployment
        )

    def test_development_compose_environment_controls_are_effective(self, project_root):
        compose = self._load_yaml(
            os.path.join(project_root, "deploy", "docker", "docker-compose.yml")
        )
        triton = compose["services"]["triton"]

        assert triton["image"].startswith("${TRITON_DEV_IMAGE:-")
        assert all("${HOST_BIND_ADDRESS:-" in port for port in triton["ports"])
        assert "${MODEL_REPOSITORY_PATH:-" in triton["volumes"][0]
        assert any("${CACHE_SIZE:-" in arg for arg in triton["command"].split())

    def test_readme_does_not_claim_perf_runs_during_production_deploy(
        self, project_root
    ):
        with open(os.path.join(project_root, "README.md")) as readme_file:
            readme = readme_file.read()

        assert "prod 배포 + perf baseline 비교" not in readme
        assert "perf-benchmark.yml" in readme


class TestReleaseWorkflow:
    """배포 image와 manifest/test revision이 어긋나지 않는지 검사"""

    def test_production_checks_out_requested_main_revision(self, project_root):
        workflow_path = os.path.join(
            project_root, ".github", "workflows", "cd-production.yml"
        )
        with open(workflow_path) as workflow_file:
            workflow = workflow_file.read()

        assert "ref: ${{ github.event.inputs.image_tag }}" in workflow
        assert "fetch-depth: 0" in workflow
        assert 'git merge-base --is-ancestor "${IMAGE_TAG}" origin/main' in workflow
        assert 'test "$(git rev-parse HEAD)" = "${IMAGE_TAG}"' in workflow

    def test_pr_ci_validates_all_deployment_formats(self, project_root):
        workflow_path = os.path.join(
            project_root, ".github", "workflows", "ci-validate.yml"
        )
        with open(workflow_path) as workflow_file:
            workflow = workflow_file.read()

        for command in (
            "kustomize build",
            "helm lint",
            "helm template",
            "docker compose",
            "promtool",
            "bash -n",
        ):
            assert command in workflow, f"PR CI does not run {command}"
        assert "- 'monitoring/**'" in workflow

    def test_pr_ci_runs_offline_unit_suites(self, project_root):
        workflow_path = os.path.join(
            project_root, ".github", "workflows", "ci-validate.yml"
        )
        with open(workflow_path) as workflow_file:
            workflow = workflow_file.read()

        assert "name: Unit Tests" in workflow
        assert "pytest tests/client/ tests/scripts/ tests/perf/" in workflow

    def test_deployment_workflows_explicitly_enable_live_tests(self, project_root):
        workflow_expectations = {
            "ci-build-test.yml": "pytest tests/smoke/ --run-live",
            "cd-staging.yml": "pytest tests/integration/ --run-live",
            "cd-production.yml": "pytest tests/smoke/ --run-live",
        }
        for filename, command in workflow_expectations.items():
            path = os.path.join(project_root, ".github", "workflows", filename)
            with open(path) as workflow_file:
                assert command in workflow_file.read()

    def test_main_build_tests_the_published_digest(self, project_root):
        workflow_path = os.path.join(
            project_root, ".github", "workflows", "ci-build-test.yml"
        )
        with open(workflow_path) as workflow_file:
            workflow = workflow_file.read()

        assert "./scripts/build.sh --env prod --clean" in workflow
        assert "./scripts/build.sh --env dev --clean" not in workflow
        assert "- 'client/**'" in workflow
        assert "- 'tests/**'" in workflow
        assert "--cache-config=local,size=16777216" in workflow
        assert "id: build" in workflow
        assert "push: true" in workflow
        assert "${{ env.IMAGE_NAME }}:candidate-${{ github.sha }}" in workflow
        assert "@${{ steps.build.outputs.digest }}" in workflow
        assert workflow.count("uses: docker/build-push-action@") == 1
        assert "docker buildx imagetools create" in workflow
        assert 'test "${PROMOTED_DIGEST}" = "${CANDIDATE_DIGEST}"' in workflow
        assert workflow.index("- name: Run smoke tests") < workflow.index(
            "- name: Promote tested digest to release tag"
        )

    def test_deploy_workflows_pin_the_resolved_digest(self, project_root):
        for filename in ("cd-staging.yml", "cd-production.yml"):
            path = os.path.join(project_root, ".github", "workflows", filename)
            with open(path) as workflow_file:
                workflow = workflow_file.read()

            assert "docker buildx imagetools inspect" in workflow
            assert "^sha256:[0-9a-f]{64}$" in workflow
            assert 'IMAGE_REF=${REGISTRY}/${IMAGE_NAME}@${DIGEST}' in workflow
            assert 'kustomize edit set image "triton-server=${IMAGE_REF}"' in workflow

    def test_perf_benchmark_uses_release_runtime_contract(self, project_root):
        workflow_path = os.path.join(
            project_root, ".github", "workflows", "perf-benchmark.yml"
        )
        with open(workflow_path) as workflow_file:
            workflow = workflow_file.read()

        assert "./scripts/build.sh --env prod --clean" in workflow
        assert "--file deploy/docker/Dockerfile" in workflow
        assert "--tag triton-server:perf" in workflow
        assert "--cache-config=local,size=67108864" in workflow
        assert "-v $(pwd)/model_repository:/models:ro" not in workflow

    def test_deploy_workflows_verify_kube_context(self, project_root):
        workflow_expectations = {
            "cd-staging.yml": "KUBE_CONTEXT_STAGING",
            "cd-production.yml": "KUBE_CONTEXT_PROD",
        }
        for filename, variable in workflow_expectations.items():
            path = os.path.join(project_root, ".github", "workflows", filename)
            with open(path) as workflow_file:
                workflow = workflow_file.read()

            assert f"kubectl config use-context \"${{{variable}}}\"" in workflow
            assert f"kubectl config current-context)\" = \"${{{variable}}}\"" in workflow

    def test_composite_action_inputs_are_not_interpolated_in_shell(self, project_root):
        action_path = os.path.join(
            project_root, ".github", "actions", "triton-health-check", "action.yml"
        )
        with open(action_path) as action_file:
            action = yaml.safe_load(action_file)

        for step in action["runs"]["steps"]:
            assert "${{ inputs." not in step.get("run", ""), \
                "action inputs must flow through env before reaching shell"

    def test_requirement_versions_are_exactly_pinned(self, project_root):
        for filename in (
            "requirements-dev.txt",
            "requirements-integration.txt",
            "requirements-model.txt",
            "requirements-converter.txt",
        ):
            path = os.path.join(project_root, filename)
            with open(path) as requirements_file:
                requirements = [
                    line.strip()
                    for line in requirements_file
                    if line.strip() and not line.lstrip().startswith("#")
                ]

            for requirement in requirements:
                if requirement.startswith("-r "):
                    continue
                assert re.match(r"^[A-Za-z0-9_.-]+(?:\[[^]]+\])?==[^=]+$", requirement), \
                    f"{filename}: dependency must be exactly pinned: {requirement}"

    @pytest.mark.skipif(not HAS_YAML, reason="PyYAML not installed")
    def test_external_actions_are_pinned_to_commit_sha(self, project_root):
        workflows_dir = os.path.join(project_root, ".github", "workflows")

        def collect_uses(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    if key == "uses" and isinstance(value, str):
                        yield value
                    else:
                        yield from collect_uses(value)
            elif isinstance(node, list):
                for value in node:
                    yield from collect_uses(value)

        for filename in os.listdir(workflows_dir):
            if not filename.endswith((".yml", ".yaml")):
                continue
            path = os.path.join(workflows_dir, filename)
            with open(path) as workflow_file:
                workflow = yaml.safe_load(workflow_file)
            for action in collect_uses(workflow):
                if action.startswith("./"):
                    continue
                _, separator, revision = action.rpartition("@")
                assert separator and re.fullmatch(r"[0-9a-f]{40}", revision), \
                    f"{filename}: external action must use a full commit SHA: {action}"


class TestImmutableModelRelease:
    """검증된 model repository가 release image와 함께 승격되는지 검사"""

    def test_serving_image_packages_built_repository(self, project_root):
        dockerfile = os.path.join(project_root, "deploy", "docker", "Dockerfile")
        with open(dockerfile) as dockerfile_content:
            contents = dockerfile_content.read()
        assert "COPY model_repository/ /models/" in contents
        assert "org.opencontainers.image.revision" in contents

        dockerignore = os.path.join(project_root, ".dockerignore")
        with open(dockerignore) as dockerignore_content:
            ignore_rules = dockerignore_content.read()
        assert "!model_repository/**" in ignore_rules

    def test_ci_smoke_tests_the_bundled_repository(self, project_root):
        workflow_path = os.path.join(
            project_root, ".github", "workflows", "ci-build-test.yml"
        )
        with open(workflow_path) as workflow_file:
            workflow = workflow_file.read()
        start_step = workflow.split("- name: Start Triton server", 1)[1].split(
            "- name: Wait for server ready", 1
        )[0]
        assert "VCS_REF=${{ github.sha }}" in workflow
        assert "model_repository:/models" not in start_step

    def test_kustomize_uses_image_models_unless_pvc_is_opted_in(
        self, project_root
    ):
        base_dir = os.path.join(project_root, "deploy", "k8s", "base")
        with open(os.path.join(base_dir, "deployment.yaml")) as deployment_file:
            deployment = yaml.safe_load(deployment_file)
        pod_spec = deployment["spec"]["template"]["spec"]
        container = pod_spec["containers"][0]
        assert container["image"] == "triton-server:local"
        assert "volumeMounts" not in container
        assert "volumes" not in pod_spec

        component_dir = os.path.join(
            project_root, "deploy", "k8s", "components", "model-pvc"
        )
        assert os.path.isfile(os.path.join(component_dir, "pvc.yaml"))
        assert os.path.isfile(os.path.join(component_dir, "volume_patch.yaml"))
        pvc_overlay = os.path.join(
            project_root, "deploy", "k8s", "overlays", "dev-pvc", "kustomization.yaml"
        )
        with open(pvc_overlay) as overlay_file:
            overlay = yaml.safe_load(overlay_file)
        assert "../../components/model-pvc" in overlay["components"]

    def test_helm_does_not_mask_bundled_models_by_default(self, project_root):
        values_path = os.path.join(
            project_root, "deploy", "helm", "triton", "values.yaml"
        )
        with open(values_path) as values_file:
            values = yaml.safe_load(values_file)
        assert values["image"] == {
            "repository": "triton-server",
            "tag": "local",
            "digest": "",
            "requireDigest": False,
            "pullPolicy": "IfNotPresent",
        }
        assert values["persistence"]["enabled"] is False
