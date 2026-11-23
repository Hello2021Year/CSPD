"""Container-only worker for the executable-code verifier bridge.

Do not invoke this module on a login/model-serving host.  The host-side bridge
requires an isolation receipt, and the supported production launcher enters a
read-only Slurm/Pyxis container with no home mount before starting this worker.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import importlib.metadata
import io
from itertools import repeat
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
from typing import Any
import zlib

from .code_verifier import (
    LCB_GRADER_ID,
    LCB_UPSTREAM_COMMIT,
    OPEN_INSTRUCT_UPSTREAM_COMMIT,
    PAYLOAD_SCHEMA,
    PYTHON_ASSERTIONS_GRADER_ID,
    REQUEST_SCHEMA,
    RESPONSE_SCHEMA,
)


DEFAULT_TEST_TIMEOUT_SECONDS = 6
PYTHON_ASSERTION_TEST_TIMEOUT_SECONDS = 1
PYTHON_ASSERTION_PASS_RATE_THRESHOLD = 0.99
# LiveCodeBench executes each generated program in-process.  Its pinned
# checker leaves the advertised memory guard disabled and, on a global
# timeout, can also leave enough child state behind to exhaust a whole Slurm
# step.  A 32 GiB per-candidate ceiling is deliberately much higher than the
# 4 GiB limit proposed upstream while still containing pathological outputs.
LCB_CANDIDATE_MEMORY_LIMIT_BYTES = 32 * 1024 * 1024 * 1024


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not all(isinstance(row, dict) for row in rows):
        raise RuntimeError("verifier requests must be JSON objects")
    return rows


def _direct_url_commit(distribution: str) -> str | None:
    try:
        raw = importlib.metadata.distribution(distribution).read_text(
            "direct_url.json"
        )
    except importlib.metadata.PackageNotFoundError:
        return None
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    commit = value.get("vcs_info", {}).get("commit_id")
    return commit if isinstance(commit, str) else None


@lru_cache(maxsize=1)
def _validate_lcb_dependency() -> None:
    commit = _direct_url_commit("livecodebench")
    image_marker = Path("/opt/livecodebench-commit")
    installed_marker = (
        image_marker.read_text(encoding="utf-8").strip()
        if image_marker.is_file()
        else None
    )
    if commit != LCB_UPSTREAM_COMMIT and installed_marker != LCB_UPSTREAM_COMMIT:
        raise RuntimeError(
            "LiveCodeBench must be installed from pinned upstream commit "
            f"{LCB_UPSTREAM_COMMIT}; found direct={commit!r}, "
            f"image_marker={installed_marker!r}"
        )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return repr(value)


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        raise pickle.UnpicklingError(
            f"LiveCodeBench private tests requested forbidden global {module}.{name}"
        )


def _decode_lcb_cases(value: str) -> list[dict[str, Any]]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        try:
            compressed = base64.b64decode(value.encode("utf-8"))
            serialized = zlib.decompress(compressed)
            decoded = json.loads(
                _RestrictedUnpickler(io.BytesIO(serialized)).load()
            )
        except (ValueError, TypeError, zlib.error, pickle.UnpicklingError) as exc:
            raise ValueError("cannot decode LiveCodeBench private tests") from exc
    if not isinstance(decoded, list) or not all(
        isinstance(item, dict) for item in decoded
    ):
        raise ValueError("LiveCodeBench tests must decode to a list of objects")
    return decoded


def _prepare_livecodebench_payload(payload: dict[str, Any]) -> dict[str, Any]:
    input_output = payload.get("input_output")
    if isinstance(input_output, dict):
        return payload
    raw = payload.get("livecodebench_raw")
    if not isinstance(raw, dict):
        raise ValueError("LiveCodeBench request has no test payload")
    public = raw.get("public_test_cases")
    private = raw.get("private_test_cases")
    metadata_raw = raw.get("metadata")
    if not all(isinstance(value, str) for value in (public, private, metadata_raw)):
        raise ValueError("LiveCodeBench raw test payload is invalid")
    tests = _decode_lcb_cases(public) + _decode_lcb_cases(private)
    try:
        metadata = json.loads(metadata_raw)
        inputs = [test["input"] for test in tests]
        outputs = [test["output"] for test in tests]
    except (json.JSONDecodeError, KeyError) as exc:
        raise ValueError("LiveCodeBench raw test payload is malformed") from exc
    if not isinstance(metadata, dict) or not all(
        isinstance(value, str) for value in inputs + outputs
    ):
        raise ValueError("LiveCodeBench raw test values are invalid")
    return {
        "input_output": {
            "inputs": inputs,
            "outputs": outputs,
            "fn_name": metadata.get("func_name"),
        }
    }


def _run_livecodebench_candidate(
    sample: dict[str, str],
    code: str,
    timeout: int,
    connection: Any,
) -> None:
    """Run one candidate in a disposable process with bounded address space."""

    try:
        from lcb_runner.evaluation import testing_util

        upstream_guard = testing_util.reliability_guard

        def bounded_guard(_maximum_memory_bytes: int | None = None) -> None:
            upstream_guard(LCB_CANDIDATE_MEMORY_LIMIT_BYTES)

        # run_test resolves this module global at call time.  The mutation is
        # confined to this disposable child and therefore cannot leak between
        # candidates or affect the parent verifier process.
        testing_util.reliability_guard = bounded_guard
        results, metadata = testing_util.run_test(
            sample,
            test=code,
            debug=False,
            timeout=timeout,
        )
        connection.send(("ok", results, metadata))
    except BaseException as exc:  # the candidate process must report and exit
        connection.send(("error", type(exc).__name__, repr(exc)))
    finally:
        connection.close()


def _safe_livecodebench_check(
    sample: dict[str, str], code: str, timeout: int
) -> tuple[list[Any], dict[str, Any]]:
    """Preserve pinned LCB semantics without Manager/zombie accumulation."""

    input_output = json.loads(sample["input_output"])
    inputs = input_output.get("inputs", [])
    global_timeout = (timeout + 1) * len(inputs) + 5
    context = multiprocessing.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_run_livecodebench_candidate,
        args=(sample, code, timeout, sender),
    )
    process.start()
    sender.close()
    # Drain the one-way pipe while the child is alive.  Waiting for the child
    # first can deadlock when LCB returns metadata larger than the OS pipe
    # buffer: the child blocks in ``send`` and the parent blocks in ``join``
    # until the global timeout, at which point a healthy candidate is killed.
    message = None
    response_ready = receiver.poll(global_timeout)
    if response_ready:
        try:
            message = receiver.recv()
        except EOFError:
            message = None
    # A candidate may leave non-daemon threads alive after sending its result.
    # Never wait indefinitely for interpreter shutdown after receiving a verdict.
    process.join(0)
    timed_out = not response_ready and process.is_alive()
    if process.is_alive():
        process.kill()
    # Always reap the exact PID.  This is the missing cleanup in the pinned
    # upstream check_correctness implementation that can accumulate zombies.
    process.join()
    receiver.close()
    if message and message[0] == "ok":
        return message[1], message[2]
    if timed_out:
        outcome = "global_timeout"
    elif process.exitcode is not None and process.exitcode < 0:
        outcome = "resource_limit"
    else:
        outcome = "candidate_process_error"
    metadata: dict[str, Any] = {
        "error_code": -5,
        "error_message": outcome,
        "candidate_exitcode": process.exitcode,
        "memory_limit_bytes": LCB_CANDIDATE_MEMORY_LIMIT_BYTES,
    }
    if message and message[0] == "error":
        metadata.update({"error_type": message[1], "error": message[2]})
    return [-1 for _ in inputs], metadata


def _grade_livecodebench(code: str, payload: dict[str, Any]) -> tuple[bool, dict]:
    _validate_lcb_dependency()

    input_output = payload.get("input_output")
    if not isinstance(input_output, dict):
        raise ValueError("LiveCodeBench request has no input_output object")
    sample = {"input_output": json.dumps(input_output, ensure_ascii=False)}
    try:
        results, metadata = _safe_livecodebench_check(
            sample, code, DEFAULT_TEST_TIMEOUT_SECONDS
        )
    except Exception as exc:  # official harness failures are a failed candidate
        return False, {
            "outcome": "harness_error",
            "error_type": type(exc).__name__,
            "test_count": len(input_output.get("inputs", [])),
        }
    fixed = [value.item() if hasattr(value, "item") else value for value in results]
    return bool(fixed) and all(value is True for value in fixed), {
        "outcome": "pass" if fixed and all(value is True for value in fixed) else "fail",
        "test_count": len(fixed),
        "test_results": fixed,
        "candidate_memory_limit_bytes": LCB_CANDIDATE_MEMORY_LIMIT_BYTES,
        "official_metadata": _json_safe(metadata) if isinstance(metadata, dict) else {},
    }


def _grade_assertions(code: str, payload: dict[str, Any]) -> tuple[bool, dict]:
    runtime_grader = (
        f"python_assertions:open-instruct@{OPEN_INSTRUCT_UPSTREAM_COMMIT}:"
        f"cpython-{sys.version_info.major}.{sys.version_info.minor}:"
        f"per-test-timeout-{PYTHON_ASSERTION_TEST_TIMEOUT_SECONDS}s:"
        f"pass-rate-{PYTHON_ASSERTION_PASS_RATE_THRESHOLD}"
    )
    if runtime_grader != PYTHON_ASSERTIONS_GRADER_ID:
        raise RuntimeError(
            "python assertion verifier runtime does not match its pinned identity: "
            f"{runtime_grader!r} != {PYTHON_ASSERTIONS_GRADER_ID!r}"
        )
    assertions = payload.get("assertions")
    if not isinstance(assertions, list) or not assertions or not all(
        isinstance(assertion, str) for assertion in assertions
    ):
        raise ValueError("python_assertions request has no assertion list")
    test_results: list[bool] = []
    returncodes: list[int | None] = []
    stderr_tails: list[str] = []
    with tempfile.TemporaryDirectory(prefix="code-assertions-") as directory:
        candidate_path = Path(directory) / "candidate.py"
        for assertion in assertions:
            program = code + "\n\n" + assertion + "\n"
            candidate_path.write_bytes(
                program.encode("utf-8", errors="surrogatepass")
            )
            try:
                completed = subprocess.run(
                    [sys.executable, "-I", "-S", str(candidate_path)],
                    cwd=directory,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=PYTHON_ASSERTION_TEST_TIMEOUT_SECONDS,
                    check=False,
                    env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
                )
            except subprocess.TimeoutExpired:
                test_results.append(False)
                returncodes.append(None)
                stderr_tails.append("timeout")
                continue
            test_results.append(completed.returncode == 0)
            returncodes.append(completed.returncode)
            stderr = completed.stderr.strip()
            stderr_tails.append(stderr[-1000:] if len(stderr) > 1000 else stderr)
    passed_test_count = sum(test_results)
    pass_rate = passed_test_count / len(test_results)
    correct = pass_rate >= PYTHON_ASSERTION_PASS_RATE_THRESHOLD
    first_failure = next(
        (index for index, passed in enumerate(test_results) if not passed), None
    )
    return correct, {
        "outcome": "pass" if correct else "fail",
        "test_count": len(assertions),
        "passed_test_count": passed_test_count,
        "pass_rate": pass_rate,
        "test_results": test_results,
        "returncode": returncodes[first_failure] if first_failure is not None else 0,
        "stderr_tail": stderr_tails[first_failure] if first_failure is not None else "",
    }


def _isolation_receipt() -> dict[str, str]:
    backend = os.environ.get("SOFT_PROMPT_VERIFIER_ISOLATION_BACKEND")
    image = os.environ.get("SOFT_PROMPT_VERIFIER_IMAGE_IDENTITY")
    if not backend or not image:
        raise RuntimeError("verifier worker has no production isolation identity")
    return {"backend": backend, "image": image}


def _verify_one(
    request: dict[str, Any], isolation: dict[str, str], payload: dict[str, Any]
) -> dict[str, Any]:
    if request.get("schema") != REQUEST_SCHEMA:
        raise RuntimeError("unknown verifier request schema")
    request_id = request.get("request_id")
    protocol = request.get("protocol")
    code = request.get("code")
    if isinstance(request_id, bool) or not isinstance(request_id, int):
        raise RuntimeError("verifier request_id must be an integer")
    if not isinstance(code, str):
        raise RuntimeError("verifier request has invalid code/payload")
    if protocol == "livecodebench":
        correct, metadata = _grade_livecodebench(code, payload)
        grader = LCB_GRADER_ID
    elif protocol == "python_assertions":
        correct, metadata = _grade_assertions(code, payload)
        grader = PYTHON_ASSERTIONS_GRADER_ID
    else:
        raise RuntimeError(f"unsupported verifier protocol: {protocol!r}")
    return {
        "schema": RESPONSE_SCHEMA,
        "request_id": request_id,
        "correct": correct,
        "grader": grader,
        "metadata": _json_safe(metadata),
        "isolation": isolation,
    }


def _verify_group(
    requests: list[dict[str, Any]],
    isolation: dict[str, str],
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    if requests and requests[0].get("protocol") == "livecodebench":
        payload = _prepare_livecodebench_payload(payload)
    return [_verify_one(request, isolation, payload) for request in requests]


def _payload_bank(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    bank = {}
    for row in _read_jsonl(path):
        if row.get("schema") != PAYLOAD_SCHEMA:
            raise RuntimeError("unknown verifier payload schema")
        payload_ref = row.get("payload_ref")
        payload = row.get("payload")
        if (
            isinstance(payload_ref, bool)
            or not isinstance(payload_ref, int)
            or payload_ref in bank
            or not isinstance(payload, dict)
        ):
            raise RuntimeError("verifier payload sidecar has an invalid row")
        bank[payload_ref] = payload
    return bank


def _resolve_payload(
    request: dict[str, Any], bank: dict[int, dict[str, Any]]
) -> tuple[str, dict[str, Any]]:
    protocol = request.get("protocol")
    if not isinstance(protocol, str) or not protocol:
        raise RuntimeError("verifier request has no protocol")
    embedded = request.get("payload")
    payload_ref = request.get("payload_ref")
    if isinstance(embedded, dict) and payload_ref is None:
        serialized = json.dumps(
            embedded, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        digest = hashlib.sha256(serialized.encode()).hexdigest()
        return f"{protocol}:embedded:{digest}", embedded
    if (
        embedded is None
        and not isinstance(payload_ref, bool)
        and isinstance(payload_ref, int)
        and payload_ref in bank
    ):
        return f"{protocol}:sidecar:{payload_ref}", bank[payload_ref]
    raise RuntimeError("verifier request must reference exactly one valid payload")


def _worker_count(request_count: int) -> int:
    explicit = os.environ.get("SOFT_PROMPT_VERIFIER_WORKERS")
    raw = explicit or os.environ.get("SLURM_CPUS_PER_TASK", "1")
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError("verifier CPU worker count must be an integer") from exc
    if value <= 0:
        raise RuntimeError("verifier CPU worker count must be positive")
    if explicit is not None:
        return max(1, min(request_count, value))
    # Every candidate grader may launch one contained child process, so reserve
    # two allocated CPUs per concurrent request rather than oversubscribing.
    return max(1, min(request_count, max(1, value // 2)))


def verify(
    requests: list[dict[str, Any]], payloads_path: Path | None = None
) -> list[dict[str, Any]]:
    isolation = _isolation_receipt()
    bank = _payload_bank(payloads_path)
    grouped: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for request in requests:
        key, payload = _resolve_payload(request, bank)
        existing_payload, group = grouped.setdefault(key, (payload, []))
        if existing_payload != payload:
            raise RuntimeError("verifier payload identity collision")
        group.append(request)
    groups = list(grouped.values())
    workers = _worker_count(len(groups))
    if workers == 1:
        nested = [
            _verify_group(group, isolation, payload) for payload, group in groups
        ]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            nested = list(
                executor.map(
                    _verify_group,
                    [group for _payload, group in groups],
                    repeat(isolation),
                    [payload for payload, _group in groups],
                )
            )
    responses = [response for group in nested for response in group]
    responses.sort(key=lambda response: response["request_id"])
    if [response["request_id"] for response in responses] != list(
        range(len(requests))
    ):
        raise RuntimeError("verifier request IDs must be contiguous from zero")
    return responses


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("requests", type=Path)
    parser.add_argument("responses", type=Path)
    parser.add_argument("payloads", type=Path, nargs="?")
    args = parser.parse_args()
    requests = _read_jsonl(args.requests)
    responses = verify(requests, args.payloads)
    temporary = args.responses.with_suffix(args.responses.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for response in responses:
            handle.write(
                json.dumps(response, ensure_ascii=False, sort_keys=True) + "\n"
            )
    temporary.replace(args.responses)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
