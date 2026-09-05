#!/usr/bin/env python3
"""Fetch manifest-declared model artifacts into a verified staging directory."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import ssl
import sys
import tempfile
from typing import BinaryIO
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlsplit, urlunsplit
from urllib.request import (
    HTTPRedirectHandler,
    HTTPSHandler,
    Request,
    build_opener,
)

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "models" / "serving" / "manifest.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / ".artifacts"
CHUNK_SIZE = 64 * 1024
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
ENV_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
HOST_PATTERN = re.compile(r"^[A-Za-z0-9.-]+$")
SAFE_TARGET_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
ALLOWED_ARTIFACT_FIELDS = {
    "path",
    "uri",
    "sha256",
    "size_bytes",
    "auth_token_env",
}


class ArtifactError(ValueError):
    """Expected manifest, transport, or integrity failure."""


@dataclass(frozen=True)
class ArtifactSpec:
    target: str
    path: str
    uri: str
    sha256: str
    size_bytes: int
    auth_token_env: str | None = None


def _relative_artifact_path(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ArtifactError(f"{field} must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ArtifactError(f"{field} must be a normalized relative path")
    if any(part in {"", "."} for part in path.parts):
        raise ArtifactError(f"{field} contains an invalid path component")
    return value


def _parse_tags(value: str) -> set[str]:
    tags = set(filter(None, value.split(",")))
    if any(SAFE_TARGET_PATTERN.fullmatch(tag) is None for tag in tags):
        raise ArtifactError("--tags must contain comma-separated safe names")
    return tags


def _parse_allowed_hosts(value: str) -> set[str]:
    hosts = set()
    for raw_host in filter(None, (part.strip() for part in value.split(","))):
        if (
            HOST_PATTERN.fullmatch(raw_host) is None
            or raw_host.startswith(".")
            or raw_host.endswith(".")
            or ".." in raw_host
        ):
            raise ArtifactError(
                "--allowed-hosts must contain exact DNS names or IPv4 addresses"
            )
        hosts.add(raw_host.lower())
    return hosts


def load_selected_artifacts(
    manifest_path: Path,
    environment: str,
    filter_tags: set[str],
) -> list[ArtifactSpec]:
    try:
        with manifest_path.open(encoding="utf-8") as manifest_file:
            manifest = yaml.safe_load(manifest_file)
    except (OSError, yaml.YAMLError) as error:
        raise ArtifactError(f"cannot read manifest: {error}") from error

    if not isinstance(manifest, dict) or not isinstance(manifest.get("models"), list):
        raise ArtifactError("manifest must contain a models list")

    selected: list[ArtifactSpec] = []
    targets: set[str] = set()
    for index, model in enumerate(manifest["models"]):
        prefix = f"models[{index}]"
        if not isinstance(model, dict):
            raise ArtifactError(f"{prefix} must be an object")

        target = model.get("target")
        if (
            not isinstance(target, str)
            or target in {".", ".."}
            or SAFE_TARGET_PATTERN.fullmatch(target) is None
        ):
            raise ArtifactError(f"{prefix}.target must be a safe non-empty name")
        if target in targets:
            raise ArtifactError(f"duplicate model target: {target}")
        targets.add(target)

        enabled = model.get("enabled", True)
        tags = model.get("tags", [])
        environments = model.get("environments", ["dev", "staging", "prod"])
        artifact_source = model.get("artifact", "local")
        if not isinstance(enabled, bool):
            raise ArtifactError(f"{prefix}.enabled must be boolean")
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            raise ArtifactError(f"{prefix}.tags must be a string list")
        if (
            not isinstance(environments, list)
            or not environments
            or not all(item in {"dev", "staging", "prod"} for item in environments)
        ):
            raise ArtifactError(f"{prefix}.environments contains an invalid value")
        if artifact_source not in {"local", "external"}:
            raise ArtifactError(f"{prefix}.artifact must be local or external")

        is_selected = (
            enabled
            and environment in environments
            and (not filter_tags or bool(filter_tags.intersection(tags)))
        )
        if not is_selected or artifact_source == "local":
            continue

        required_files = model.get("required_files", [])
        artifacts = model.get("artifacts")
        if not isinstance(required_files, list) or not all(
            isinstance(item, str) for item in required_files
        ):
            raise ArtifactError(f"{prefix}.required_files must be a string list")
        normalized_required = {
            _relative_artifact_path(item, f"{prefix}.required_files")
            for item in required_files
        }
        if len(normalized_required) != len(required_files):
            raise ArtifactError(f"{prefix}.required_files must not contain duplicates")
        if not isinstance(artifacts, list) or not artifacts:
            raise ArtifactError(
                f"enabled external model {target} must declare a non-empty artifacts list"
            )

        artifact_paths: set[str] = set()
        for artifact_index, artifact in enumerate(artifacts):
            artifact_prefix = f"{prefix}.artifacts[{artifact_index}]"
            if not isinstance(artifact, dict):
                raise ArtifactError(f"{artifact_prefix} must be an object")
            unknown_fields = set(artifact) - ALLOWED_ARTIFACT_FIELDS
            if unknown_fields:
                raise ArtifactError(
                    f"{artifact_prefix} contains unknown fields: "
                    f"{', '.join(sorted(unknown_fields))}"
                )

            artifact_path = _relative_artifact_path(
                artifact.get("path"), f"{artifact_prefix}.path"
            )
            if artifact_path in artifact_paths:
                raise ArtifactError(f"duplicate artifact path for {target}: {artifact_path}")
            artifact_paths.add(artifact_path)

            uri = artifact.get("uri")
            checksum = artifact.get("sha256")
            size_bytes = artifact.get("size_bytes")
            auth_token_env = artifact.get("auth_token_env")
            if not isinstance(uri, str) or not uri:
                raise ArtifactError(f"{artifact_prefix}.uri is required")
            if not isinstance(checksum, str) or SHA256_PATTERN.fullmatch(checksum) is None:
                raise ArtifactError(
                    f"{artifact_prefix}.sha256 must be 64 lowercase hex characters"
                )
            if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes <= 0:
                raise ArtifactError(f"{artifact_prefix}.size_bytes must be a positive integer")
            if auth_token_env is not None and (
                not isinstance(auth_token_env, str)
                or ENV_NAME_PATTERN.fullmatch(auth_token_env) is None
            ):
                raise ArtifactError(
                    f"{artifact_prefix}.auth_token_env must be a valid environment name"
                )

            selected.append(
                ArtifactSpec(
                    target=target,
                    path=artifact_path,
                    uri=uri,
                    sha256=checksum,
                    size_bytes=size_bytes,
                    auth_token_env=auth_token_env,
                )
            )

        if artifact_paths != normalized_required:
            raise ArtifactError(
                f"external model {target} artifacts paths must exactly match required_files"
            )

    return selected


def _safe_https_uri(uri: str, allowed_hosts: set[str]) -> tuple[str, str]:
    parsed = urlsplit(uri)
    if (
        parsed.scheme.lower() != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or not parsed.path
        or parsed.fragment
    ):
        raise ArtifactError("remote artifact URI must be an HTTPS URL without credentials or fragment")
    hostname = parsed.hostname.lower()
    if hostname not in allowed_hosts:
        raise ArtifactError(f"artifact host is not allowlisted: {hostname}")
    return hostname, urlunsplit(("https", parsed.netloc, parsed.path, "", ""))


class _SafeRedirectHandler(HTTPRedirectHandler):
    def __init__(self, allowed_hosts: set[str]):
        super().__init__()
        self.allowed_hosts = allowed_hosts

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        old_host = (urlsplit(req.full_url).hostname or "").lower()
        new_host, _ = _safe_https_uri(newurl, self.allowed_hosts)
        if req.has_header("Authorization") and new_host != old_host:
            raise ArtifactError("authenticated artifact redirects must stay on the same host")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _verify_local_source(uri: str, local_root: Path | None) -> tuple[BinaryIO, str]:
    if local_root is None:
        raise ArtifactError("file artifacts require --local-root and are development-only")
    parsed = urlsplit(uri)
    if (
        parsed.scheme != "file"
        or parsed.netloc not in {"", "localhost"}
        or parsed.query
        or parsed.fragment
    ):
        raise ArtifactError(
            "local artifact URI must use file:///absolute/path without query or fragment"
        )
    source = Path(unquote(parsed.path))
    if not source.is_absolute():
        raise ArtifactError("local artifact URI must contain an absolute path")
    try:
        root = local_root.resolve(strict=True)
        lexical_source = Path(os.path.abspath(source))
        if not lexical_source.is_relative_to(root):
            raise ArtifactError("local artifact escapes --local-root")
        current = root
        for part in lexical_source.relative_to(root).parts:
            current /= part
            if current.is_symlink():
                raise ArtifactError("local artifact path must not contain symlinks")
        resolved = lexical_source.resolve(strict=True)
    except OSError as error:
        raise ArtifactError(f"cannot resolve local artifact: {error}") from error
    if not resolved.is_relative_to(root):
        raise ArtifactError("local artifact escapes --local-root")
    if not resolved.is_file():
        raise ArtifactError("local artifact must be a regular file")
    return resolved.open("rb"), resolved.as_uri()


def _open_artifact(
    spec: ArtifactSpec,
    allowed_hosts: set[str],
    local_root: Path | None,
    timeout: float,
) -> tuple[BinaryIO, str]:
    parsed = urlsplit(spec.uri)
    if parsed.scheme == "file":
        if spec.auth_token_env is not None:
            raise ArtifactError("file artifacts cannot use auth_token_env")
        return _verify_local_source(spec.uri, local_root)

    _, public_uri = _safe_https_uri(spec.uri, allowed_hosts)
    headers = {"Accept-Encoding": "identity", "User-Agent": "triton-artifact-fetcher/1"}
    if spec.auth_token_env:
        token = os.getenv(spec.auth_token_env)
        if not token:
            raise ArtifactError(f"required token environment is unset: {spec.auth_token_env}")
        if "\r" in token or "\n" in token:
            raise ArtifactError(f"token environment contains line breaks: {spec.auth_token_env}")
        headers["Authorization"] = f"Bearer {token}"

    opener = build_opener(
        _SafeRedirectHandler(allowed_hosts),
        HTTPSHandler(context=ssl.create_default_context()),
    )
    response = opener.open(Request(spec.uri, headers=headers), timeout=timeout)
    content_encoding = response.headers.get("Content-Encoding", "identity").lower()
    if content_encoding not in {"", "identity"}:
        response.close()
        raise ArtifactError("artifact server must return identity content encoding")
    content_length = response.headers.get("Content-Length")
    if content_length is not None:
        try:
            advertised_size = int(content_length)
        except ValueError as error:
            response.close()
            raise ArtifactError("artifact server returned an invalid Content-Length") from error
        if advertised_size != spec.size_bytes:
            response.close()
            raise ArtifactError(
                f"artifact Content-Length mismatch for {spec.target}/{spec.path}"
            )
    return response, public_uri


def _write_verified(source: BinaryIO, destination: Path, spec: ArtifactSpec) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    size = 0
    try:
        with os.fdopen(descriptor, "wb") as output_file:
            while chunk := source.read(CHUNK_SIZE):
                size += len(chunk)
                if size > spec.size_bytes:
                    raise ArtifactError(
                        f"artifact exceeds declared size for {spec.target}/{spec.path}"
                    )
                digest.update(chunk)
                output_file.write(chunk)
            output_file.flush()
            os.fsync(output_file.fileno())
        if size != spec.size_bytes:
            raise ArtifactError(f"artifact size mismatch for {spec.target}/{spec.path}")
        if digest.hexdigest() != spec.sha256:
            raise ArtifactError(f"artifact SHA-256 mismatch for {spec.target}/{spec.path}")
        os.chmod(temporary, 0o644)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_directory(staging: Path, output: Path) -> None:
    if output.is_symlink():
        raise ArtifactError("artifact output directory must not be a symlink")
    if output.exists() and not output.is_dir():
        raise ArtifactError("artifact output path must be a directory")
    if not output.exists():
        os.replace(staging, output)
        return

    backup = output.parent / f".{output.name}.backup-{os.getpid()}"
    if backup.exists():
        raise ArtifactError(f"artifact backup path already exists: {backup}")
    os.replace(output, backup)
    try:
        os.replace(staging, output)
    except BaseException:
        os.replace(backup, output)
        raise
    else:
        shutil.rmtree(backup)


def fetch_artifacts(args: argparse.Namespace) -> int:
    if args.environment not in {"dev", "staging", "prod"}:
        raise ArtifactError("--env must be dev, staging, or prod")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ArtifactError("--timeout must be a finite value greater than zero")
    if args.max_artifact_bytes <= 0 or args.max_total_bytes <= 0:
        raise ArtifactError("artifact byte limits must be positive")

    filter_tags = _parse_tags(args.tags)
    allowed_hosts = _parse_allowed_hosts(args.allowed_hosts)
    manifest_path = Path(args.manifest).expanduser().resolve()
    output = Path(args.output_dir).expanduser().absolute()
    local_root = Path(args.local_root).expanduser() if args.local_root else None
    specs = load_selected_artifacts(manifest_path, args.environment, filter_tags)
    total_size = sum(spec.size_bytes for spec in specs)
    if any(spec.size_bytes > args.max_artifact_bytes for spec in specs):
        raise ArtifactError("an artifact exceeds --max-artifact-bytes")
    if total_size > args.max_total_bytes:
        raise ArtifactError("selected artifacts exceed --max-total-bytes")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.fetch-", dir=output.parent))
    receipt_entries = []
    try:
        for spec in specs:
            print(f"[artifact] fetching {spec.target}/{spec.path}")
            source, public_uri = _open_artifact(
                spec, allowed_hosts, local_root, args.timeout
            )
            try:
                _write_verified(source, staging / spec.target / spec.path, spec)
            finally:
                source.close()
            receipt = asdict(spec)
            receipt["uri"] = public_uri
            receipt.pop("auth_token_env", None)
            receipt_entries.append(receipt)
            print(f"[artifact] verified {spec.target}/{spec.path}")

        receipt = {
            "schema_version": 1,
            "source_revision": os.getenv("GITHUB_SHA", "unknown"),
            "environment": args.environment,
            "artifacts": receipt_entries,
        }
        receipt_path = staging / "receipt.json"
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _publish_directory(staging, output)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    print(
        f"[artifact] complete: {len(specs)} artifact(s), "
        f"{total_size} byte(s), receipt={output / 'receipt.json'}"
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--env", dest="environment", default="dev")
    parser.add_argument("--tags", default="")
    parser.add_argument(
        "--allowed-hosts",
        default=os.getenv("ARTIFACT_ALLOWED_HOSTS", ""),
        help="Comma-separated exact HTTPS host allowlist",
    )
    parser.add_argument(
        "--local-root",
        help="Development-only root for file:// artifacts",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-artifact-bytes", type=int, default=50 * 1024**3)
    parser.add_argument("--max-total-bytes", type=int, default=200 * 1024**3)
    return parser.parse_args()


def main() -> int:
    try:
        return fetch_artifacts(parse_args())
    except (ArtifactError, HTTPError, URLError, OSError) as error:
        print(f"[artifact] error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
