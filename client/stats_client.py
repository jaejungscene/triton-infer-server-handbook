"""
Triton Statistics API 클라이언트
GET /v2/models/{name}/stats
GET /v2/models/{name}/versions/{version}/stats

사용 예:
    python client/stats_client.py --url http://localhost:8000 --model resnet50
    python client/stats_client.py --url http://localhost:8000 --model yolox --version 1
"""

import argparse
import json
import os
import urllib.error
import urllib.request
from collections.abc import Mapping
from urllib.parse import quote, urlsplit


DEFAULT_TIMEOUT_SECONDS = 10.0


def _validated_base_url(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("url must be an absolute HTTP(S) Triton endpoint")
    return url.rstrip("/")


def _request_json(
    endpoint: str,
    timeout: float,
    headers: Mapping[str, str] | None,
) -> dict:
    if timeout <= 0:
        raise ValueError("timeout must be greater than zero")

    request = urllib.request.Request(endpoint, headers=dict(headers or {}))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"서버 연결 실패: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Triton statistics response is not valid JSON") from exc

    if not isinstance(payload, dict):
        raise RuntimeError("Triton statistics response must be a JSON object")
    return payload


def get_model_stats(
    url: str,
    model_name: str,
    version: str | None = None,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    headers: Mapping[str, str] | None = None,
) -> dict:
    """
    Triton Statistics API를 호출하여 모델별 추론 통계를 반환합니다.

    Args:
        url: Triton 서버 HTTP URL (예: http://localhost:8000)
        model_name: 모델 이름
        version: 모델 버전 (None이면 최신 버전)

    Returns:
        model_stats 딕셔너리
    """
    if not model_name:
        raise ValueError("model_name must not be empty")

    base_url = _validated_base_url(url)
    encoded_model = quote(model_name, safe="")
    if version:
        encoded_version = quote(version, safe="")
        endpoint = (
            f"{base_url}/v2/models/{encoded_model}/versions/{encoded_version}/stats"
        )
    else:
        endpoint = f"{base_url}/v2/models/{encoded_model}/stats"

    return _request_json(endpoint, timeout, headers)


def get_all_model_stats(
    url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    headers: Mapping[str, str] | None = None,
) -> dict:
    """서버 전체 모델 통계를 반환합니다. GET /v2/models/stats"""
    endpoint = f"{_validated_base_url(url)}/v2/models/stats"
    return _request_json(endpoint, timeout, headers)


def _as_non_negative_int(value: int | str, field: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if parsed < 0:
        raise ValueError(f"{field} must not be negative")
    return parsed


def _ns_to_ms(ns: int | str) -> float:
    return round(_as_non_negative_int(ns, "duration ns") / 1_000_000, 3)


def print_summary(stats: dict) -> None:
    """
    model_stats 응답을 사람이 읽기 좋은 형태로 출력합니다.

    주요 지표:
      - inference_count  : 총 추론 요청 수
      - execution_count  : 실제 배치 실행 횟수 (dynamic batching 시 < inference_count)
      - queue 대기시간    : 요청이 큐에서 기다린 누적 시간
      - compute_infer    : 실제 GPU/CPU 연산 시간
      - compute_input    : 입력 전처리 시간
      - compute_output   : 출력 후처리 시간
    """
    model_stats_list = stats.get("model_stats", [])
    if not model_stats_list:
        print("통계 데이터가 없습니다.")
        return

    for entry in model_stats_list:
        name = entry.get("name", "unknown")
        version = entry.get("version", "-")
        inf_count = _as_non_negative_int(
            entry.get("inference_count", 0), "inference_count"
        )
        exec_count = _as_non_negative_int(
            entry.get("execution_count", 0), "execution_count"
        )

        print(f"\n{'=' * 55}")
        print(f"  모델: {name}  (버전: {version})")
        print(f"{'=' * 55}")
        print(f"  총 inference 수 : {inf_count:,}")
        print(f"  실제 실행 횟수  : {exec_count:,}")

        if exec_count > 0 and inf_count > 0:
            batch_eff = inf_count / exec_count
            print(f"  평균 배치 크기  : {batch_eff:.2f}  (Dynamic Batching 효과)")

        inf_stats = entry.get("inference_stats", {})
        if inf_stats:
            print()
            print("  [단계별 누적 시간 (ms)]")

            for key, label in [
                ("queue",          "큐 대기"),
                ("compute_input",  "입력 처리"),
                ("compute_infer",  "추론 연산"),
                ("compute_output", "출력 처리"),
                ("success",        "전체 성공"),
                ("fail",           "실패"),
            ]:
                stat = inf_stats.get(key)
                if stat:
                    count = _as_non_negative_int(
                        stat.get("count", 0), f"inference_stats.{key}.count"
                    )
                    total_ms = _ns_to_ms(stat.get("ns", 0))
                    avg_ms = round(total_ms / count, 3) if count > 0 else 0
                    print(f"    {label:<12}: 총 {total_ms:>10,.3f} ms  |  평균 {avg_ms:>8,.3f} ms  |  횟수 {count:,}")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Triton Statistics API 클라이언트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 특정 모델 통계
  python client/stats_client.py --model resnet50

  # 특정 버전 통계
  python client/stats_client.py --model yolox --version 1

  # 전체 모델 통계
  python client/stats_client.py --all

  # JSON 원본 출력
  python client/stats_client.py --model resnet50 --json
        """,
    )
    parser.add_argument("--url", default="http://localhost:8000", help="Triton HTTP URL")
    parser.add_argument("--model", help="모델 이름")
    parser.add_argument("--version", help="모델 버전 (생략 시 최신)")
    parser.add_argument("--all", action="store_true", help="전체 모델 통계 조회")
    parser.add_argument("--json", action="store_true", help="JSON 원본 출력")
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="HTTP timeout seconds (default: 10)",
    )
    args = parser.parse_args()

    token = os.getenv("TRITON_AUTH_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else None

    if args.all:
        stats = get_all_model_stats(
            args.url, timeout=args.timeout, headers=headers
        )
    elif args.model:
        stats = get_model_stats(
            args.url,
            args.model,
            args.version,
            timeout=args.timeout,
            headers=headers,
        )
    else:
        parser.error("--model 또는 --all 중 하나를 지정하세요.")

    if args.json:
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    else:
        print_summary(stats)


if __name__ == "__main__":
    main()
