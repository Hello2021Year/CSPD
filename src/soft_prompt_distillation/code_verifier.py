"""Fail-closed bridge to an isolated executable-code verifier.

The model-serving process never imports or executes generated programs.  It
writes a versioned JSONL request and invokes a caller-supplied runner.  The
production runner is expected to enter a locked-down container (Slurm/Pyxis on
the project cluster) before starting the verifier worker.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

from .grading import GradeResult


REQUEST_SCHEMA = "soft-prompt-code-verifier-request-v1"
RESPONSE_SCHEMA = "soft-prompt-code-verifier-response-v1"
PAYLOAD_SCHEMA = "soft-prompt-code-verifier-payload-v1"
LCB_UPSTREAM_COMMIT = "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"
LCB_GRADER_ID = f"livecodebench@{LCB_UPSTREAM_COMMIT}:code_generation"
OPEN_INSTRUCT_UPSTREAM_COMMIT = "e6c3044aa6201c8bfa4573c1b03142af599e730d"
PYTHON_ASSERTIONS_GRADER_ID = (
    f"python_assertions:open-instruct@{OPEN_INSTRUCT_UPSTREAM_COMMIT}:"
    "cpython-3.11:per-test-timeout-1s:pass-rate-0.99"
)
ALLOWED_ISOLATION_BACKENDS = frozenset(
    {"slurm_pyxis", "docker", "podman", "apptainer", "nsjail"}
)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").split("\n")
        rows = [json.loads(line) for line in lines if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"code verifier returned invalid JSONL: {exc}") from exc
    if not all(isinstance(row, dict) for row in rows):
        raise RuntimeError("code verifier response rows must be JSON objects")
    return rows


def _resolved_command(command: Path) -> Path:
    try:
        resolved = command.expanduser().resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"code verifier command is unavailable: {command}") from exc
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise RuntimeError(f"code verifier command is not executable: {resolved}")
    return resolved


def _preserve_failure_bundle(
    root_path: Path,
    *,
    runner: Path,
    returncode: int | None,
    stderr: str,
    failure_kind: str,
) -> Path | None:
    """Persist verifier inputs after a failed isolated run when opted in."""

    raw_directory = os.environ.get("SOFT_PROMPT_CODE_VERIFIER_FAILURE_DIR")
    if not raw_directory:
        return None
    try:
        failure_root = Path(raw_directory).expanduser().resolve()
        failure_root.mkdir(parents=True, exist_ok=True)
        bundle = Path(tempfile.mkdtemp(prefix="failure-", dir=failure_root))
        for name in ("requests.jsonl", "payloads.jsonl", "responses.jsonl"):
            source = root_path / name
            if source.is_file():
                shutil.copy2(source, bundle / name)
        (bundle / "stderr.txt").write_text(stderr, encoding="utf-8")
        (bundle / "failure.json").write_text(
            json.dumps(
                {
                    "failure_kind": failure_kind,
                    "returncode": returncode,
                    "runner": str(runner),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return bundle
    except OSError:
        # Diagnostics must never replace the fail-closed verifier error.
        return None


def run_external_code_verifier(
    requests: Sequence[Mapping[str, Any]],
    *,
    command: Path | None,
    timeout_seconds: int | None,
) -> list[GradeResult]:
    """Run one verifier batch and validate its isolation/audit receipt."""

    if command is None:
        raise RuntimeError(
            "executable-code grading requires --code-verifier-command; "
            "generated code is never executed in the model-serving process"
        )
    if timeout_seconds is None or timeout_seconds <= 0:
        raise RuntimeError("code verifier timeout must be a positive integer")
    runner = _resolved_command(command)
    normalized_requests = []
    payload_rows = []
    payload_index: dict[str, int] = {}
    for index, request in enumerate(requests):
        request_body = dict(request)
        payload = request_body.pop("payload", None)
        payload_identity = request_body.pop("payload_identity", None)
        if not isinstance(payload, Mapping):
            raise ValueError("code verifier request payload must be an object")
        if payload_identity is not None:
            if not isinstance(payload_identity, str) or not payload_identity:
                raise ValueError("code verifier payload identity must be a string")
            payload_key = f"declared:{payload_identity}"
        else:
            payload_serialized = json.dumps(
                payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            payload_key = hashlib.sha256(payload_serialized.encode()).hexdigest()
        payload_ref = payload_index.get(payload_key)
        if payload_ref is None:
            payload_ref = len(payload_rows)
            payload_index[payload_key] = payload_ref
            payload_rows.append(
                {
                    "schema": PAYLOAD_SCHEMA,
                    "payload_ref": payload_ref,
                    "payload": dict(payload),
                }
            )
        elif payload_rows[payload_ref]["payload"] != payload:
            raise ValueError("code verifier payload identity maps to multiple payloads")
        normalized_requests.append(
            {
                "schema": REQUEST_SCHEMA,
                "request_id": index,
                "payload_ref": payload_ref,
                **request_body,
            }
        )

    with tempfile.TemporaryDirectory(prefix="soft-prompt-code-verifier-") as root:
        root_path = Path(root)
        request_path = root_path / "requests.jsonl"
        response_path = root_path / "responses.jsonl"
        payload_path = root_path / "payloads.jsonl"
        _write_jsonl(request_path, normalized_requests)
        _write_jsonl(payload_path, payload_rows)
        try:
            completed = subprocess.run(
                [str(runner), str(request_path), str(response_path)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            bundle = _preserve_failure_bundle(
                root_path,
                runner=runner,
                returncode=None,
                stderr=stderr,
                failure_kind="batch_timeout",
            )
            bundle_suffix = f"; request bundle: {bundle}" if bundle else ""
            raise RuntimeError(
                f"isolated code verifier exceeded {timeout_seconds}s batch timeout"
                f"{bundle_suffix}"
            ) from exc
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            bundle = _preserve_failure_bundle(
                root_path,
                runner=runner,
                returncode=completed.returncode,
                stderr=completed.stderr,
                failure_kind="nonzero_exit",
            )
            if len(stderr) > 2000:
                stderr = stderr[-2000:]
            bundle_suffix = f"; request bundle: {bundle}" if bundle else ""
            raise RuntimeError(
                f"isolated code verifier failed with exit {completed.returncode}: "
                f"{stderr or '<no stderr>'}{bundle_suffix}"
            )
        if not response_path.is_file():
            raise RuntimeError("isolated code verifier produced no response file")
        response_rows = _read_jsonl(response_path)

    if len(response_rows) != len(normalized_requests):
        raise RuntimeError(
            "code verifier response count does not match request count: "
            f"{len(response_rows)} != {len(normalized_requests)}"
        )
    by_id: dict[int, dict[str, Any]] = {}
    for response in response_rows:
        if response.get("schema") != RESPONSE_SCHEMA:
            raise RuntimeError("code verifier response uses an unknown schema")
        request_id = response.get("request_id")
        if (
            isinstance(request_id, bool)
            or not isinstance(request_id, int)
            or request_id in by_id
        ):
            raise RuntimeError("code verifier response has an invalid request_id")
        by_id[request_id] = response

    results = []
    for request_id in range(len(normalized_requests)):
        response = by_id.get(request_id)
        if response is None:
            raise RuntimeError(f"code verifier omitted request {request_id}")
        correct = response.get("correct")
        grader = response.get("grader")
        metadata = response.get("metadata")
        isolation = response.get("isolation")
        if not isinstance(correct, bool):
            raise RuntimeError("code verifier response correct must be boolean")
        if not isinstance(grader, str) or not grader:
            raise RuntimeError("code verifier response grader must be non-empty")
        expected_grader = normalized_requests[request_id].get("grader_requested")
        if grader != expected_grader:
            raise RuntimeError(
                "code verifier response grader does not match the request: "
                f"{grader!r} != {expected_grader!r}"
            )
        if not isinstance(metadata, dict):
            raise RuntimeError("code verifier response metadata must be an object")
        if not isinstance(isolation, dict):
            raise RuntimeError("code verifier response has no isolation receipt")
        backend = isolation.get("backend")
        image = isolation.get("image")
        if backend not in ALLOWED_ISOLATION_BACKENDS:
            raise RuntimeError(
                f"code verifier used unsupported isolation backend: {backend!r}"
            )
        if not isinstance(image, str) or not image:
            raise RuntimeError("code verifier isolation receipt has no image identity")
        results.append(
            GradeResult(
                correct=correct,
                grader=grader,
                metadata={**metadata, "isolation": dict(isolation)},
            )
        )
    return results


__all__ = [
    "ALLOWED_ISOLATION_BACKENDS",
    "LCB_GRADER_ID",
    "LCB_UPSTREAM_COMMIT",
    "PAYLOAD_SCHEMA",
    "PYTHON_ASSERTIONS_GRADER_ID",
    "REQUEST_SCHEMA",
    "RESPONSE_SCHEMA",
    "run_external_code_verifier",
]
