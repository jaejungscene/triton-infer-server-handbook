# Contributing Guide

## 모델 추가

1. `models/_templates/` 에서 적절한 템플릿을 복사합니다.
2. `models/serving/{domain}/` 아래에 배치합니다.
3. `models/serving/manifest.yaml`에 source→target 매핑을 추가합니다.
4. `config.pbtxt`를 수정합니다 (input/output shape, backend).
5. PR을 생성합니다. CI가 자동으로 config 검증을 수행합니다.

## 브랜치 전략

- `main` — 항상 배포 가능 상태
- `feature/*` — 기능 개발
- `hotfix/*` — 긴급 수정

## PR 규칙

- `models/` 하위 변경 시 ML팀 리뷰 필수 (CODEOWNERS)
- config.pbtxt 변경 시 `scripts/validate.sh` 로컬 실행 권장
- 성능에 영향을 줄 수 있는 변경은 perf test 결과 첨부

## 로컬 검증

PR CI는 Python/config뿐 아니라 모든 Kustomize overlay, Helm values, Docker Compose,
Prometheus rule, shell 문법을 검사합니다. 제출 전 최소 검증은 다음과 같습니다.

```bash
./scripts/validate.sh
pytest tests/
ruff check models/ client/ tests/ scripts/ --select E,W,F --ignore E501
helm lint deploy/helm/triton \
  -f deploy/helm/triton/values.prod.yaml \
  --set image.digest=sha256:0000000000000000000000000000000000000000000000000000000000000000
kustomize build deploy/k8s/overlays/prod >/dev/null
promtool check config monitoring/prometheus/scrape_config.yml
promtool check rules monitoring/prometheus/triton_rules.yml
TRITON_IMAGE=triton-server:validation GRAFANA_ADMIN_PASSWORD=local docker compose \
  -f deploy/docker/docker-compose.prod.yml config --quiet
```

PR의 offline unit job은 `tests/models`, `tests/client`, `tests/scripts`, `tests/perf`를 모두
실행합니다. `pytest tests/`는 서버가 필요 없는 suite를 실행하고 smoke/integration은 skip합니다. 로컬
Triton에 실제 요청을 보내려면 대상 디렉터리와 `--run-live`를 함께 지정합니다.
Markdown만 바꾼 PR도 CI 대상이며 내부 파일 링크와 닫히지 않은 code fence를 config suite에서
검사합니다.

GitHub Actions의 외부 `uses:`는 release tag가 아니라 40자리 commit SHA로 고정합니다. 사람이
버전을 알아볼 수 있도록 같은 줄에 `# v4`처럼 tag를 주석으로 남기고, 버전 갱신 PR에서는
upstream release note와 새 SHA를 함께 검토합니다. repository 내부 composite action은
`./.github/actions/...` 상대 경로를 사용합니다.

`requirements*.txt`의 실행 의존성은 범위가 아닌 `==` 버전으로 고정합니다. 버전 갱신은
관련 패키지를 한 번에 무작정 올리지 않고 unit/config/smoke 결과와 converter image build를
확인한 별도 PR로 수행합니다.

Triton container는 사람이 읽는 release tag와 immutable digest를 함께 기록합니다. 버전을
올릴 때 Dockerfile 두 개, dev Compose, perf SDK, `tests/perf/baseline.json`의 tag/digest를 같은
PR에서 갱신하고 새 digest의 출처와 smoke/perf 결과를 남깁니다.

production Compose의 Redis, Prometheus, Grafana와 CI의 Prometheus 도구 image도
`tag@sha256:<manifest-list-digest>`로 고정합니다. 버전을 올릴 때 `docker buildx imagetools inspect`
로 multi-architecture index digest를 확인하고 Compose와 CI의 Prometheus 참조를 함께 갱신합니다.

main CI는 `candidate-<commit SHA>` image만 게시합니다. candidate는 GPU 검증 전 artifact이므로
staging이나 production에 사용하지 않습니다. NVIDIA GPU runner가 준비된 시점에 main에서
`CI - GPU Release`를 수동 실행하고, runtime contract를 통과해 `<commit SHA>` release tag가
생성된 뒤에만 staging과 production 절차를 진행합니다.

## 커밋 메시지

```
type(scope): description

feat(models): add yolox ensemble pipeline
fix(configs): correct rate limiter resource count
ci(workflows): add staging deployment step
```
