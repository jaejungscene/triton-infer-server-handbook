import json
import os

import yaml


RUNBOOK_URL = (
    "https://github.com/jaejungscene/triton-infer-server-handbook/"
    "blob/main/docs/runbook.md"
)


def test_prometheus_rules_do_not_clamp_low_traffic_denominators(project_root):
    rules_path = os.path.join(
        project_root, "monitoring", "prometheus", "triton_rules.yml"
    )
    with open(rules_path) as rules_file:
        rules = yaml.safe_load(rules_file)

    expressions = [
        rule["expr"]
        for group in rules["groups"]
        for rule in group["rules"]
        if "expr" in rule
    ]
    assert not any("clamp_min" in expression for expression in expressions)
    ratio_expressions = [
        expression
        for expression in expressions
        if "nv_inference_request_success" in expression
    ]
    assert all("environment" in expression for expression in ratio_expressions)
    assert all(">= 0.1" in expression for expression in ratio_expressions)


def test_missing_metrics_rule_tracks_the_required_production_environment(project_root):
    rules_path = os.path.join(
        project_root, "monitoring", "prometheus", "triton_rules.yml"
    )
    with open(rules_path) as rules_file:
        rules = yaml.safe_load(rules_file)

    missing_rule = next(
        rule
        for group in rules["groups"]
        for rule in group["rules"]
        if rule.get("alert") == "TritonMetricsMissing"
    )
    assert (
        'up{service="triton",environment="production"}'
        in missing_rule["expr"]
    )


def test_alerts_include_routing_context_and_runbook(project_root):
    rules_path = os.path.join(
        project_root, "monitoring", "prometheus", "triton_rules.yml"
    )
    with open(rules_path) as rules_file:
        groups = yaml.safe_load(rules_file)["groups"]

    alerts = {
        rule["alert"]: rule
        for group in groups
        for rule in group["rules"]
        if "alert" in rule
    }
    assert len(alerts) == 7
    for rule in alerts.values():
        annotations = rule["annotations"]
        assert annotations["runbook_url"] == RUNBOOK_URL
        assert "$labels.environment" in annotations["summary"]

    for alert_name in (
        "TritonGPUHighUtilization",
        "TritonGPUMemoryHigh",
        "TritonServerDown",
    ):
        assert "$labels.instance" in alerts[alert_name]["annotations"]["summary"]

    for alert_name in ("TritonGPUHighUtilization", "TritonGPUMemoryHigh"):
        assert (
            "$labels.gpu_uuid"
            in alerts[alert_name]["annotations"]["description"]
        )


def test_dashboard_only_uses_metrics_exported_by_default(project_root):
    dashboard_path = os.path.join(
        project_root, "monitoring", "grafana", "triton_dashboard.json"
    )
    with open(dashboard_path) as dashboard_file:
        dashboard = json.load(dashboard_file)

    expressions = [
        target["expr"]
        for panel in dashboard["panels"]
        for target in panel.get("targets", [])
    ]
    assert not any("_bucket" in expression for expression in expressions)
    assert not any("nv_inference_pending_request_count" in expression for expression in expressions)
    assert dashboard["datasource"]["uid"] == "prometheus"
    assert dashboard["templating"]["list"][0]["name"] == "environment"
    assert all(
        'environment=~"$environment"' in expression
        for expression in expressions
    )
    legends = [
        target["legendFormat"]
        for panel in dashboard["panels"]
        for target in panel.get("targets", [])
    ]
    assert all("{{ environment }}" in legend for legend in legends)

    panels = {panel["title"]: panel for panel in dashboard["panels"]}
    latency_expressions = [
        target["expr"]
        for target in panels["Average Request / Queue Latency (ms)"]["targets"]
    ]
    assert all(">= 0.1" in expression for expression in latency_expressions)
    assert ">= 0.1" in panels["Error Rate"]["targets"][0]["expr"]
    assert "> 0" in panels["Cache Hit Rate"]["targets"][0]["expr"]
