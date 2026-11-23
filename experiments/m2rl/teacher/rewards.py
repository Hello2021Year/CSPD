"""The M2RL training rewards used to verify teacher responses (math, code, science, IF).

All four rewards run through M2RL's own ``async_rm`` dispatcher, fetched from the pinned
Mosi-AI/M2RL commit and checked against git blob hashes:

* math: ``deepscaler`` (10 s timeout inside ``async_rm``)
* science: ``gpqa`` with the question's valid option letters
* instruction following: ``ifevalg`` (strict, every instruction must be followed). This reward is not
  deterministic for a few instruction types (for example ``english_capital``, which runs language
  detection). Regrading the original responses reproduced 99.3% of the verdicts, and the rest also
  change between repeated calls.
* code: ``unit_test`` with ``pass_all``: at most 20 of the question's tests, 10 s each

The code reward posts each test to a SandboxFusion server. Here the HTTP client is replaced by
local execution with the same limits (the wrapped program from ``get_guarded_code``, 4 GiB
address space, 10 s per test), and the 20-test subset is drawn with a per-question seed.
Generated programs run on this machine, so grade code inside a container or sandbox.
"""
from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import os
import random
import re
import resource
import shutil
import signal
import sys
import tempfile
import urllib.request
from pathlib import Path
from types import ModuleType, SimpleNamespace

M2RL_COMMIT = "c94ce97ce7f2e0f2800e6dd8c20667f00ca223d4"
M2RL_RAW = f"https://raw.githubusercontent.com/Mosi-AI/M2RL/{M2RL_COMMIT}/M2RL/slime/slime"
SOURCES = {  # path under slime/slime -> git blob sha1
    "rollout/rm_hub/__init__.py": "4525a72871c2168d5c6e5731ccd8a7915c95e405",
    "rollout/rm_hub/deepscaler.py": "3c722ed9ec0fa46a2dedddb5f3b9e5133eede743",
    "rollout/rm_hub/gpqa.py": "ca99830f99e6cfe3b67ed6d9c70587b669d3cd3a",
    "rollout/rm_hub/math_utils.py": "94ec98b9205d21a22df6c6315d7d44640aabdc2e",
    "rewards/code.py": "90b958ffa83d26c3da6405efa167d47cbbff6afc",
    "rewards/IFEvalG/__init__.py": "52d2f504169a1987f4c46cee18f24ffa54084181",
    "rewards/IFEvalG/evaluation_main.py": "7eb3cabc6f0a1650a7e9ee39927e92cfb8eb9205",
    "rewards/IFEvalG/instructions.py": "fa77d51730caa86847e5ee471a94247b51c10187",
    "rewards/IFEvalG/instructions_registry.py": "4e1faecead1986e1c6fcb57aa0ab35d546ad4b3f",
    "rewards/IFEvalG/instructions_util.py": "afa118ce61ba98bf013b527275442adc5bdd8887",
}
CACHE = Path.home() / ".cache" / "cspd-graders" / f"m2rl-rewards-{M2RL_COMMIT[:12]}"


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def fetch_sources():
    """Download the pinned reward files into ``CACHE/slime/...`` and verify them.

    Call once in the main process before starting grading workers (``init`` only reads)."""
    for name, blob in SOURCES.items():
        target = CACHE / "slime" / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(f"{M2RL_RAW}/{name}", timeout=60) as response:
                staging = target.with_suffix(f".{os.getpid()}.tmp")
                staging.write_bytes(response.read())
                staging.replace(target)
        content = target.read_bytes()
        if hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest() != blob:
            raise ValueError(f"{name} does not match M2RL@{M2RL_COMMIT[:7]}")
    for package in ("slime", "slime/rewards"):
        (CACHE / package / "__init__.py").touch()
    import nltk  # IFEvalG tokenizes with punkt
    for name in ("punkt", "punkt_tab"):
        nltk.download(name, quiet=True)
    return CACHE


# ------------------------------------------------------------------ local code execution

class _Response:
    """Stands in for the SandboxFusion HTTP response: runs one test in a child process."""
    status = 200

    def __init__(self, payload):
        self.payload = payload

    async def __aenter__(self):
        payload = self.payload
        work = Path(tempfile.mkdtemp(prefix="cspd-code-"))
        try:
            (work / "main.py").write_text(payload["code"])
            limit = payload["memory_limit"]

            def limits():
                resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
                resource.setrlimit(resource.RLIMIT_FSIZE, (8 << 20, 8 << 20))

            process = await asyncio.create_subprocess_exec(
                sys.executable, "-B", str(work / "main.py"), stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL, cwd=work,
                preexec_fn=limits, start_new_session=True)
            try:
                stdout, _ = await asyncio.wait_for(process.communicate(payload["stdin"].encode()),
                                                   payload["run_timeout"])
            except asyncio.TimeoutError:
                os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
                raise  # the M2RL reward counts a timed-out test as failed
            self.stdout = stdout.decode(errors="replace")
        finally:
            shutil.rmtree(work, ignore_errors=True)
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self):
        return {"run_result": {"stdout": self.stdout}}


class _Session:
    def __init__(self, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def post(self, url, json):
        return _Response(json)


_FAKE_AIOHTTP = SimpleNamespace(ClientSession=_Session, TCPConnector=lambda **kw: None,
                                ClientTimeout=lambda **kw: None, ClientError=OSError)

# ------------------------------------------------------------------ reward dispatcher

_RM = {}


def init():
    """Load the rewards once per worker process (after ``fetch_sources`` in the main process)."""
    root = CACHE
    for name, blob in SOURCES.items():
        content = (root / "slime" / name).read_bytes()
        if hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest() != blob:
            raise ValueError(f"{name} does not match M2RL@{M2RL_COMMIT[:7]}; run fetch_sources() first")
    sys.path.insert(0, str(root))
    # Import rm_hub's modules without running its __init__.py (it needs slime's training runtime).
    hub = ModuleType("slime.rollout.rm_hub")
    hub.__path__ = [str(root / "slime/rollout/rm_hub")]
    sys.modules["slime.rollout.rm_hub"] = hub
    from slime.rollout.rm_hub.deepscaler import get_deepscaler_rule_based_reward
    from slime.rollout.rm_hub.gpqa import compute_gpqa_reward

    # rewards/code.py imports aiohttp at module level; load it with the local executor instead.
    source = (root / "slime/rewards/code.py").read_text()
    code_env = dict(re=re, asyncio=asyncio, random=random, aiohttp=_FAKE_AIOHTTP)
    nodes = [n for n in ast.parse(source).body if isinstance(n, (ast.Assign, ast.FunctionDef, ast.AsyncFunctionDef))]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "slime/rewards/code.py", "exec"), code_env)
    _RM["code_env"] = code_env
    code_module = ModuleType("slime.rewards.code")
    code_module.reward_with_unit_test = code_env["reward_with_unit_test"]
    sys.modules["slime.rewards.code"] = code_module

    # async_rm from rm_hub/__init__.py, without its aiohttp/slime.utils imports.
    tree = ast.parse((root / "slime/rollout/rm_hub/__init__.py").read_text())
    function = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "async_rm")
    namespace = {"asyncio": asyncio, "random": random, "Sample": SimpleNamespace,
                 "get_deepscaler_rule_based_reward": get_deepscaler_rule_based_reward,
                 "compute_gpqa_reward": compute_gpqa_reward}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "slime/rollout/rm_hub/__init__.py", "exec"), namespace)
    _RM["async_rm"] = namespace["async_rm"]


def code_test_seed(question_id):
    """Seed of the per-question 20-test subset (fixed so every sample sees the same tests)."""
    return int(digest({"question_id": question_id, "base_seed": 42, "purpose": "author-20-test-subset"})[:16], 16) % 2**31


def score(job):
    """``job``: question_id, text (decoded response, thinking included), label, metadata. Returns 0.0 or 1.0."""
    if not _RM:
        fetch_sources()
        init()
    metadata = job["metadata"]
    if metadata["rm_type"] == "unit_test":
        random.seed(code_test_seed(job["question_id"]))
        # code.py's module-level semaphore binds to one event loop; give each call a fresh one.
        _RM["code_env"]["SEMAPHORE"] = asyncio.Semaphore(_RM["code_env"]["TOOL_CONFIGS"]["tool_concurrency"])
    sample = SimpleNamespace(response=job["text"], label=job["label"], metadata=metadata)
    args = SimpleNamespace(custom_rm_path=None, rm_type=metadata["rm_type"], code_sandbox_url="local")
    return float(asyncio.run(_RM["async_rm"](args, sample)))
