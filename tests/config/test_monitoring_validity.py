import json
import os

import yaml


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
