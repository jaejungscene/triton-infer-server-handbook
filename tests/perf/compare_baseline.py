#!/usr/bin/env python3
"""Compare perf_analyzer CSV exports with per-model performance floors."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _find_column(fieldnames: list[str], aliases: set[str]) -> str:
    for fieldname in fieldnames:
        if _normalized(fieldname) in aliases:
            return fieldname
    raise ValueError(f"missing CSV column; expected one of {sorted(aliases)}")


def _read_measurement(csv_path: Path, concurrency: int) -> tuple[float, float]:
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        concurrency_column = _find_column(fieldnames, {"concurrency"})
        throughput_column = _find_column(
            fieldnames,
            {"inferencesecond", "inferencessecond", "throughput"},
        )
        p95_column = _find_column(
            fieldnames,
            {"p95latency", "p95latencyus", "p95latencyusec"},
        )

        for row in reader:
            if int(float(row[concurrency_column])) != concurrency:
                continue
            throughput = float(row[throughput_column])
            p95_latency_ms = float(row[p95_column]) / 1000.0
            return throughput, p95_latency_ms

    raise ValueError(f"no measurement for concurrency={concurrency}")


def compare(baseline_path: Path, results_dir: Path, selected_model: str | None) -> int:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    model_targets = baseline.get("models", baseline)
    if not isinstance(model_targets, dict):
        raise ValueError("baseline must contain a 'models' object")

    if selected_model and selected_model not in model_targets:
        print(f"ERROR: no baseline is defined for {selected_model}", file=sys.stderr)
        return 2

    failures: list[str] = []
    result_models = {
        path.name.removesuffix("_perf.csv")
        for path in results_dir.glob("*_perf.csv")
    }
    unknown_models = sorted(result_models - set(model_targets))
    failures.extend(
        f"{model_name}: benchmark result has no configured baseline"
        for model_name in unknown_models
    )

    evaluated = 0
    for model_name, targets in model_targets.items():
        if model_name.startswith("_") or (selected_model and model_name != selected_model):
            continue

        csv_path = results_dir / f"{model_name}_perf.csv"
        if not csv_path.exists():
            if selected_model:
                failures.append(f"{model_name}: result CSV is missing")
            else:
                print(f"SKIP {model_name}: model was not benchmarked")
            continue

        evaluated += 1
        try:
            concurrency = int(targets["concurrency"])
            min_throughput = float(targets["min_throughput"])
            max_p95_latency_ms = float(targets["max_p95_latency_ms"])
            throughput, p95_latency_ms = _read_measurement(csv_path, concurrency)
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(f"{model_name}: invalid result or baseline ({exc})")
            continue

        print(
            f"CHECK {model_name}: throughput={throughput:.2f} infer/sec "
            f"(min={min_throughput:.2f}), p95={p95_latency_ms:.2f} ms "
            f"(max={max_p95_latency_ms:.2f})"
        )
        if throughput < min_throughput:
            failures.append(
                f"{model_name}: throughput {throughput:.2f} < {min_throughput:.2f} infer/sec"
            )
        if p95_latency_ms > max_p95_latency_ms:
            failures.append(
                f"{model_name}: p95 {p95_latency_ms:.2f} > {max_p95_latency_ms:.2f} ms"
            )

    if evaluated == 0:
        failures.append("no benchmark result matched a configured baseline")

    if failures:
        print("\nPERFORMANCE REGRESSION", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("\nAll measured models are within baseline")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--model")
    args = parser.parse_args()

    try:
        return compare(args.baseline, args.results_dir, args.model)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
