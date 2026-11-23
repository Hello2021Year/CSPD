from __future__ import annotations

import os
import subprocess
import tomllib
from pathlib import Path

import pytest

from soft_prompt_distillation import common
from soft_prompt_distillation.common import RunOptions
from soft_prompt_distillation.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_model_source_resolves_hub_subfolder_without_flattening(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(ROOT / "configs" / "math" / "nemotron-cascade-8b.yaml")
    snapshot = tmp_path / "models--nvidia--nemotron" / "snapshots" / "abc123"
    stage = snapshot / "Nemotron-Cascade-8B-SFT"
    stage.mkdir(parents=True)
    (stage / "config.json").write_text("{}\n", encoding="utf-8")
    calls: list[dict[str, object]] = []

    def fake_snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setattr(common, "snapshot_download", fake_snapshot_download)
    options = RunOptions(Path("config.yaml"), tmp_path / "output", tmp_path / "cache")

    source, revision = common.model_source(options, config, "student")

    assert source == str(stage)
    assert revision is None
    assert calls == [
        {
            "repo_id": "nvidia/Nemotron-Cascade-8B-Intermediate-ckpts",
            "allow_patterns": ["Nemotron-Cascade-8B-SFT/**"],
            "cache_dir": tmp_path / "cache",
        }
    ]


def test_template_source_applies_exact_audited_chat_template_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(ROOT / "configs" / "ablations" / "gpt-oss-prompt-length-256.yaml")

    class FakeTokenizer:
        chat_template = '{{ strftime_now("%Y-%m-%d") }}{{- model_identity + "\\n" }}'
        init_kwargs: dict[str, str] = {}

        @staticmethod
        def get_vocab() -> dict[str, int]:
            return {"token": 0}

    tokenizer = FakeTokenizer()
    monkeypatch.setattr(
        common.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: tokenizer,
    )
    options = RunOptions(Path("config.yaml"), tmp_path / "output", tmp_path / "cache")

    loaded, identity = common.load_tokenizer(options, config, "teacher")

    assert loaded.chat_template == (
        '{{ "2026-08-21" }}{{- "<|soft_prompt|>" + model_identity + "\\n" }}'
    )
    assert identity["template_chat_replacements"] == {
        'strftime_now("%Y-%m-%d")': '"2026-08-21"',
        '{{- model_identity + "\\n" }}': '{{- "<|soft_prompt|>" + model_identity + "\\n" }}',
    }


def test_chat_template_replacement_fails_if_upstream_template_drifted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(ROOT / "configs" / "ablations" / "gpt-oss-prompt-length-256.yaml")

    class FakeTokenizer:
        chat_template = "upstream changed"
        init_kwargs: dict[str, str] = {}

        @staticmethod
        def get_vocab() -> dict[str, int]:
            return {"token": 0}

    monkeypatch.setattr(
        common.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: FakeTokenizer(),
    )
    options = RunOptions(Path("config.yaml"), tmp_path / "output", tmp_path / "cache")

    with pytest.raises(RuntimeError, match="expected exactly one occurrence"):
        common.load_tokenizer(options, config, "teacher")


def test_runtime_setup_declares_and_verifies_ninja() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    sglang_dependencies = project["project"]["optional-dependencies"]["sglang"]
    assert any(dependency.startswith("ninja") for dependency in sglang_dependencies)

    setup = (ROOT / "scripts" / "setup_runtime.sh").read_text(encoding="utf-8")
    assert '"${env_dir}/bin/ninja" --version' in setup
    assert "export PATH=%q:$PATH" in setup


def test_training_runtime_pins_blackwell_flex_attention_stack() -> None:
    setup = (ROOT / "scripts" / "setup_training_runtime.sh").read_text(encoding="utf-8")
    assert '"torch==2.13.0" "transformers==4.57.1"' in setup
    assert "from torch.nn.attention.flex_attention import flex_attention" in setup
    assert "SOFT_PROMPT_TRAIN_PYTHON" in setup


def test_blackwell_inference_runtime_pins_a_native_blackwell_stack() -> None:
    setup = (ROOT / "scripts" / "setup_blackwell_runtime.sh").read_text(
        encoding="utf-8"
    )
    assert '"torch==2.13.0"' in setup
    assert '"sglang==0.5.18"' in setup
    assert '"sglang-kernel==0.4.6.post1"' in setup
    assert '"transformers==5.12.1"' in setup
    assert "SOFT_PROMPT_PYTHON" in setup


def test_pipeline_exposes_selected_runtime_bin_to_subprocesses() -> None:
    pipeline = (ROOT / "scripts" / "run_pipeline.sh").read_text(encoding="utf-8")
    assert 'export PATH="${runtime_bin}:${PATH}"' in pipeline
    assert 'log_progress "stage=sample event=start' in pipeline
    assert 'resume incomplete sample stage:' in pipeline
    assert 'resume incomplete training stage:' in pipeline
    assert '"${directory}/training_state.pt"' in pipeline
    assert 'train_args+=(--resume)' in pipeline
    assert 'log_progress "stage=train event=start' in pipeline
    assert 'log_progress "stage=evaluate_student event=start' in pipeline
    assert 'log_progress "stage=evaluate_teacher event=start' in pipeline
    assert 'log_progress "stage=evaluate_student_soft_prompt event=start' in pipeline
    assert 'log_progress "stage=pipeline event=complete' in pipeline
    assert 'config="${config_snapshot}"' in pipeline
    assert 'train_python_bin="${SOFT_PROMPT_TRAIN_PYTHON:-${python_bin}}"' in pipeline
    assert '"${train_python_bin}" -m torch.distributed.run' in pipeline
    stage_starts = [
        pipeline.index('log_progress "stage=evaluate_student event=start'),
        pipeline.index('log_progress "stage=evaluate_teacher event=start'),
        pipeline.index('log_progress "stage=sample event=start'),
        pipeline.index('log_progress "stage=train event=start'),
        pipeline.index('log_progress "stage=evaluate_student_soft_prompt event=start'),
    ]
    assert stage_starts == sorted(stage_starts)


@pytest.mark.parametrize(
    ("failing_target", "expected_targets"),
    [("student", ["student"]), ("teacher", ["student", "teacher"])],
)
def test_pipeline_stops_before_sampling_when_baseline_evaluation_fails(
    tmp_path: Path,
    failing_target: str,
    expected_targets: list[str],
) -> None:
    invocation_log = tmp_path / "invocations.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$*" >> "$FAKE_PYTHON_INVOCATIONS"\n'
        'if [[ " $* " == *" --target ${FAILING_TARGET} "* ]]; then exit 42; fi\n'
        "exit 0\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    config = tmp_path / "config.yaml"
    config.write_text("name: baseline-first-regression\n", encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "SOFT_PROMPT_PYTHON": str(fake_python),
            "FAKE_PYTHON_INVOCATIONS": str(invocation_log),
            "FAILING_TARGET": failing_target,
            "HF_HOME": str(tmp_path / "cache"),
        }
    )

    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "run_pipeline.sh"),
            str(config),
            str(tmp_path / "run"),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 42
    invocations = invocation_log.read_text(encoding="utf-8").splitlines()
    assert all(str(tmp_path / "run" / "config.yaml") in call for call in invocations)
    assert [
        target
        for invocation in invocations
        for target in ("student", "teacher")
        if f"--target {target}" in invocation
    ] == expected_targets
    assert all(" sample " not in f" {invocation} " for invocation in invocations)


def test_pipeline_can_pause_for_review_after_baseline_evaluations(
    tmp_path: Path,
) -> None:
    invocation_log = tmp_path / "invocations.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$*" >> "$FAKE_PYTHON_INVOCATIONS"\n'
        "exit 0\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    config = tmp_path / "config.yaml"
    config.write_text("name: baseline-review-gate\n", encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "SOFT_PROMPT_PYTHON": str(fake_python),
            "SOFT_PROMPT_PIPELINE_STOP_AFTER_BASELINES": "1",
            "FAKE_PYTHON_INVOCATIONS": str(invocation_log),
            "HF_HOME": str(tmp_path / "cache"),
        }
    )

    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "run_pipeline.sh"),
            str(config),
            str(tmp_path / "run"),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "event=paused_after_baselines" in result.stdout
    invocations = invocation_log.read_text(encoding="utf-8").splitlines()
    assert [
        target
        for invocation in invocations
        for target in ("student", "teacher")
        if f"--target {target}" in invocation
    ] == ["student", "teacher"]
    assert all(" sample " not in f" {invocation} " for invocation in invocations)


def test_pipeline_archives_incomplete_baseline_before_retry(tmp_path: Path) -> None:
    invocation_log = tmp_path / "invocations.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$*" >> "$FAKE_PYTHON_INVOCATIONS"\n'
        "output=\n"
        "while (( $# )); do\n"
        '  if [[ "$1" == "--output" ]]; then output="$2"; shift 2; else shift; fi\n'
        "done\n"
        'if [[ -n "$output" ]]; then\n'
        '  mkdir -p "$output"\n'
        '  touch "$output/results.jsonl" "$output/summary.json" "$output/manifest.json"\n'
        "fi\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    config = tmp_path / "config.yaml"
    config.write_text("name: archive-incomplete-baseline\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    incomplete = run_dir / "evaluate_student"
    incomplete.mkdir(parents=True)
    (incomplete / "partial.txt").write_text("preserve me\n", encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "SOFT_PROMPT_PYTHON": str(fake_python),
            "SOFT_PROMPT_PIPELINE_STOP_AFTER_BASELINES": "1",
            "FAKE_PYTHON_INVOCATIONS": str(invocation_log),
            "HF_HOME": str(tmp_path / "cache"),
            "SLURM_JOB_ID": "12345",
        }
    )

    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "run_pipeline.sh"),
            str(config),
            str(run_dir),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    evidence = run_dir / "evaluate_student_incomplete_12345"
    assert (evidence / "partial.txt").read_text(encoding="utf-8") == "preserve me\n"
    assert (run_dir / "evaluate_student" / "summary.json").is_file()
    assert "archive incomplete evaluation stage" in result.stderr
    invocations = invocation_log.read_text(encoding="utf-8").splitlines()
    assert sum("--target student" in call for call in invocations) == 1
    assert sum("--target teacher" in call for call in invocations) == 1


def test_pipeline_preserves_virtualenv_python_launcher(tmp_path: Path) -> None:
    base_python = tmp_path / "base-python"
    invocation = tmp_path / "invocation.txt"
    base_python.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$0" > "$FAKE_PYTHON_INVOCATION"\nexit 42\n',
        encoding="utf-8",
    )
    base_python.chmod(0o755)

    runtime_bin = tmp_path / "runtime" / "bin"
    runtime_bin.mkdir(parents=True)
    selected_python = runtime_bin / "python"
    selected_python.symlink_to(base_python)

    config = tmp_path / "config.yaml"
    config.write_text("name: runtime-path-regression\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    cache_dir = tmp_path / "cache"
    env = os.environ.copy()
    env.update(
        {
            "SOFT_PROMPT_PYTHON": str(selected_python),
            "FAKE_PYTHON_INVOCATION": str(invocation),
            "HF_HOME": str(cache_dir),
        }
    )

    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "run_pipeline.sh"), str(config), str(run_dir)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 42
    assert invocation.read_text(encoding="utf-8").strip() == str(selected_python)


