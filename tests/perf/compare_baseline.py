#!/usr/bin/env python3
"""Compare perf_analyzer CSV exports with per-model performance floors."""

from __future__ import annotations

import argparse
import csv
import json
import math
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


def _finite_float(value, field: str, *, allow_zero: bool) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric") from error
    if not math.isfinite(parsed) or parsed < 0 or (not allow_zero and parsed == 0):
        qualifier = "non-negative" if allow_zero else "greater than zero"
        raise ValueError(f"{field} must be finite and {qualifier}")
    return parsed


def _validated_targets(document: object) -> dict[str, dict]:
    if not isinstance(document, dict) or not isinstance(document.get("models"), dict):
        raise ValueError("baseline must contain a 'models' object")
    model_targets = document["models"]
    if not model_targets:
        raise ValueError("baseline models must not be empty")

    validated = {}
    for model_name, target in model_targets.items():
        if not isinstance(model_name, str) or not re.fullmatch(
            r"[A-Za-z0-9._-]+", model_name
        ):
            raise ValueError(f"invalid baseline model name: {model_name!r}")
        if not isinstance(target, dict):
            raise ValueError(f"baseline target for {model_name} must be an object")
        concurrency = target.get("concurrency")
        if (
            not isinstance(concurrency, int)
            or isinstance(concurrency, bool)
            or concurrency <= 0
        ):
            raise ValueError(f"{model_name}.concurrency must be a positive integer")
        validated[model_name] = {
            "concurrency": concurrency,
            "min_throughput": _finite_float(
                target.get("min_throughput"),
                f"{model_name}.min_throughput",
                allow_zero=False,
            ),
            "max_p95_latency_ms": _finite_float(
                target.get("max_p95_latency_ms"),
                f"{model_name}.max_p95_latency_ms",
                allow_zero=False,
            ),
        }
    return validated


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

        matches = []
        for row_number, row in enumerate(reader, start=2):
            measured_concurrency = _finite_float(
                row.get(concurrency_column),
                f"row {row_number} concurrency",
                allow_zero=False,
            )
            if not measured_concurrency.is_integer():
                raise ValueError(f"row {row_number} concurrency must be an integer")
            if int(measured_concurrency) != concurrency:
                continue
            throughput = _finite_float(
                row.get(throughput_column),
                f"row {row_number} throughput",
                allow_zero=True,
            )
            p95_latency_us = _finite_float(
                row.get(p95_column),
                f"row {row_number} p95 latency",
                allow_zero=True,
            )
            matches.append((throughput, p95_latency_us / 1000.0))

    if not matches:
        raise ValueError(f"no measurement for concurrency={concurrency}")
    if len(matches) > 1:
        raise ValueError(f"duplicate measurements for concurrency={concurrency}")
    return matches[0]


def _reject_json_constant(value: str):
    raise ValueError(f"baseline contains non-finite JSON value: {value}")


def compare(baseline_path: Path, results_dir: Path, selected_model: str | None) -> int:
    baseline = json.loads(
        baseline_path.read_text(encoding="utf-8"),
        parse_constant=_reject_json_constant,
    )
    model_targets = _validated_targets(baseline)

    if selected_model and selected_model not in model_targets:
        raise ValueError(f"no baseline is defined for {selected_model}")

    failures: list[str] = []
    result_models = {
        path.name.removesuffix("_perf.csv")
        for path in results_dir.glob("*_perf.csv")
    }
    unknown_models = sorted(result_models - set(model_targets))
    if unknown_models:
        raise ValueError(
            f"benchmark results have no configured baseline: {', '.join(unknown_models)}"
        )

    evaluated = 0
    for model_name, targets in model_targets.items():
        if selected_model and model_name != selected_model:
            continue

        csv_path = results_dir / f"{model_name}_perf.csv"
        if not csv_path.exists():
            if selected_model:
                raise ValueError(f"{model_name}: result CSV is missing")
            else:
                print(f"SKIP {model_name}: model was not benchmarked")
            continue

        evaluated += 1
        concurrency = targets["concurrency"]
        min_throughput = targets["min_throughput"]
        max_p95_latency_ms = targets["max_p95_latency_ms"]
        throughput, p95_latency_ms = _read_measurement(csv_path, concurrency)

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
        raise ValueError("no benchmark result matched a configured baseline")

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
