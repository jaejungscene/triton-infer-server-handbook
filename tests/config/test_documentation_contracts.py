"""Operational documentation must match the enforced deployment boundaries."""

from pathlib import Path


def _read(project_root, relative_path):
    return Path(project_root, relative_path).read_text(encoding="utf-8")


def test_handbook_links_the_incident_runbook(project_root):
    readme = _read(project_root, "README.md")
    runbook = _read(project_root, "docs/runbook.md")

    assert "[Production 장애 대응 Runbook](docs/runbook.md)" in readme
    assert "imageID" in runbook
    assert "@sha256:" in runbook
    assert "이 방식은 Ingress, NetworkPolicy, HPA, PDB" in runbook
    assert "외부 Ingress의 `/v2/repository/**`는 차단" in runbook


def test_release_docs_distinguish_sha_selector_from_image_digest(project_root):
    adoption = _read(project_root, "docs/production-adoption.md")
    architecture = _read(project_root, "docs/architecture.md")
    scenarios = _read(project_root, "docs/scenarios.md")

    assert "이 SHA는\nrelease 선택자" in adoption
    assert "registry digest로 smoke test" in architecture
    assert "container image digest를 고정" in scenarios
    assert "- `/v2/models`" not in scenarios
