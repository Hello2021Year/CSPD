from __future__ import annotations

import json
from pathlib import Path
import shutil
import threading
from types import SimpleNamespace

import pytest

from soft_prompt_distillation import common, evaluation, pipeline, reporting, sampling
from soft_prompt_distillation.config import GenerationSpec, SamplingSpec
from soft_prompt_distillation.grading import GradeResult
from soft_prompt_distillation.pipeline import RunOptions
from soft_prompt_distillation.sglang_backend import GenerationResult, SGLangRuntime
from soft_prompt_distillation.tasks import CompletionParse


class _Tokenizer:
    eos_token_id = 2

    def decode(self, token_ids, *, skip_special_tokens):
        values = list(token_ids)
        if skip_special_tokens and values and values[-1] == self.eos_token_id:
            values.pop()
        return ",".join(str(value) for value in values)


class _Runtime:
    version = "test"

    def __init__(self):
        self.calls = []
        self.data_parallel_rank_calls = []
        self.progress_labels = []
        self.stopped = False

    def generate_token_ids(
        self,
        prompts,
        spec,
        seeds,
        eos_token_id,
        forbidden_token_ids=(),
        progress_label="sglang",
        data_parallel_ranks=None,
    ):
        assert forbidden_token_ids == ()
        self.progress_labels.append(progress_label)
        self.calls.append((prompts, seeds, eos_token_id))
        self.data_parallel_rank_calls.append(data_parallel_ranks)
        return [GenerationResult([seed // 2, eos_token_id], "eos") for seed in seeds]

    def shutdown(self):
        self.stopped = True


def test_sampling_source_prompt_ids_fail_closed_on_template_drift() -> None:
    row = {"question_id": "q0", "source_prompt_ids": [1, 2, 3]}

    sampling._validate_source_prompt_ids(row, SimpleNamespace(token_ids=(1, 2, 3)))
    with pytest.raises(RuntimeError, match="does not match source prompt token IDs"):
        sampling._validate_source_prompt_ids(
            row, SimpleNamespace(token_ids=(1, 9, 3))
        )


def test_code_evaluation_overlaps_cpu_verification_with_gpu_generation(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("SOFT_PROMPT_SGLANG_ATTENTION_BACKEND", "flashinfer")
    generation = GenerationSpec(0.6, 0.95, 32, 1, "test recipe")
    config = SimpleNamespace(
        evaluation=SimpleNamespace(
            arms={"teacher": SimpleNamespace(generation=generation)},
            benchmarks={},
        ),
        teacher=SimpleNamespace(
            trust_remote_code=False,
            context_length=None,
            config_overrides={},
        ),
    )
    rows = [
        {"dataset": "code", "question_id": "q0"},
        {"dataset": "code", "question_id": "q1"},
    ]
    grade_started = threading.Event()
    allow_grade_to_finish = threading.Event()

    class Tokenizer:
        eos_token_id = 2

        @staticmethod
        def decode(token_ids, *, skip_special_tokens):
            del skip_special_tokens
            return ",".join(str(value) for value in token_ids)

    class Runtime(_Runtime):
        def generate_token_ids(self, prompts, spec, seeds, eos_token_id, **kwargs):
            del prompts, spec, eos_token_id, kwargs
            if self.calls:
                assert grade_started.wait(timeout=2)
                allow_grade_to_finish.set()
            self.calls.append(seeds)
            return [GenerationResult([seed, 2], "eos") for seed in seeds]

    class Task:
        requires_external_verifier = True

        @staticmethod
        def parse_completion(completion):
            return CompletionParse("test", None, completion, completion)

        @staticmethod
        def grade_evaluation_batch(completions, normalized_rows, **kwargs):
            del normalized_rows, kwargs
            if not grade_started.is_set():
                grade_started.set()
                assert allow_grade_to_finish.wait(timeout=2)
            return [GradeResult(True, "test", {}) for _ in completions]

    rendered = lambda row: SimpleNamespace(
        messages=({"role": "user", "content": row["question_id"]},),
        text=row["question_id"],
        tokenization_strategy="test",
        token_ids=(1,),
        token_ids_sha256="input",
        canonical_token_count=1,
        canonical_token_ids_sha256="canonical",
        tokenization_matches_canonical=True,
    )
    runtime = Runtime()
    monkeypatch.setattr(evaluation, "model_source", lambda *_args: ("teacher", None))
    monkeypatch.setattr(
        evaluation,
        "render_for_execution",
        lambda _config, *, execution_model, row, tokenizer, execution_tokenizer, template=None: (
            rendered(row)
        ),
    )
    monkeypatch.setattr(
        evaluation.SGLangRuntime,
        "start",
        classmethod(lambda cls, *args, **kwargs: runtime),
    )

    scored, _backend = evaluation._generate_and_score(
        RunOptions(
            config=tmp_path / "config.yaml",
            output=tmp_path / "output",
            cache_dir=tmp_path / "cache",
            sglang_batch_size=1,
            code_verifier_command=tmp_path / "verifier",
        ),
        config,
        Task(),
        rows,
        Tokenizer(),
        Tokenizer(),
        "teacher",
        None,
    )

    assert len(runtime.calls) == 2
    assert allow_grade_to_finish.is_set()
    assert runtime.stopped is True
    assert [row["question_id"] for row in scored] == ["q0", "q1"]


def test_teacher_sampling_uses_batched_sglang_and_preserves_snapshot(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.setenv("SOFT_PROMPT_SGLANG_ATTENTION_BACKEND", "flashinfer")
    generation = GenerationSpec(0.8, 0.9, 32, 2, "test recipe")
    config = SimpleNamespace(
        task="test",
        sampling=SamplingSpec(generation, "correct_only"),
        teacher=SimpleNamespace(
            trust_remote_code=False,
            context_length=40960,
            config_overrides={"max_position_embeddings": 40960},
        ),
    )
    rows = [
        {
            "question_id": "q0",
            "problem": "p0",
            "answer": "a0",
            "token": 10,
            "source_prompt_ids": [10],
        },
        {
            "question_id": "q1",
            "problem": "p1",
            "answer": "a1",
            "token": 11,
            "source_prompt_ids": [11],
        },
    ]
    tokenizer = _Tokenizer()
    grade_started = threading.Event()
    allow_grade_to_finish = threading.Event()
    grade_finished = threading.Event()
    graded_completions = []

    class Runtime(_Runtime):
        def generate_token_ids(self, *args, **kwargs):
            if self.calls:
                assert grade_started.wait(timeout=2)
                allow_grade_to_finish.set()
            return super().generate_token_ids(*args, **kwargs)

        def shutdown(self):
            if len(self.calls) == 2:
                assert grade_finished.wait(timeout=2)
            super().shutdown()

    def grade_sampling_batch(completions, normalized_rows, **kwargs):
        del normalized_rows, kwargs
        graded_completions.extend(completions)
        if not grade_started.is_set():
            grade_started.set()
            assert allow_grade_to_finish.wait(timeout=2)
            grade_finished.set()
        return [GradeResult(True, "test", {}) for _ in completions]

    runtime = Runtime()
    task = SimpleNamespace(
        requires_external_verifier=True,
        grade_sampling_batch=grade_sampling_batch,
    )
    manifest = {}
    start_options = []

    monkeypatch.setattr(sampling, "load_config", lambda _path: config)
    monkeypatch.setattr(sampling, "get_task", lambda _name: task)
    monkeypatch.setattr(
        sampling,
        "load_template_tokenizer",
        lambda _options, _config: (tokenizer, {}),
    )
    monkeypatch.setattr(
        sampling,
        "download_rows",
        lambda _options, _config, _task, _kind: (rows, [{"name": "test"}]),
    )
    monkeypatch.setattr(
        sampling,
        "render_for_execution",
        lambda _config, *, execution_model, row, tokenizer: SimpleNamespace(
            token_ids=(row["token"],)
        ),
    )
    monkeypatch.setattr(
        sampling,
        "resolve_insertion_index",
        lambda _config, rendered: rendered.token_ids[0],
    )
    monkeypatch.setattr(
        sampling,
        "model_source",
        lambda _options, _config, _role: ("teacher", None),
    )
    monkeypatch.setattr(
        SGLangRuntime,
        "start",
        classmethod(
            lambda cls, source, tp_size, dp_size, cache_dir, log_level, **kwargs: (
                start_options.append(kwargs) or runtime
            )
        ),
    )
    monkeypatch.setattr(
        sampling,
        "write_manifest",
        lambda _options, _config, _stage, counts, _files, identity, **_kwargs: (
            manifest.update(counts=counts, identity=identity)
        ),
    )

    output = tmp_path / "sample"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("test: config\n", encoding="utf-8")
    snapshot = sampling.sample(
        RunOptions(
            config=config_path,
            output=output,
            cache_dir=tmp_path / "cache",
            sglang_tp_size=2,
            sglang_dp_size=4,
            sglang_batch_size=3,
            sglang_log_level="warning",
            code_verifier_command=tmp_path / "verifier",
        )
    )

    assert runtime.calls == [
        ([(10,), (10,)], [1000, 1001], 2),
        ([(11,), (11,)], [1002, 1003], 2),
    ]
    assert runtime.data_parallel_rank_calls == [[0, 0], [1, 1]]
    assert runtime.progress_labels == ["sample_part_1", "sample_part_2"]
    assert runtime.stopped is True
    assert grade_finished.is_set()
    assert graded_completions == ["500", "500", "501", "501"]
    assert start_options == [
        {
            "context_length": 40960,
            "model_config_overrides": {"max_position_embeddings": 40960},
        },
        {
            "context_length": 40960,
            "model_config_overrides": {"max_position_embeddings": 40960},
        },
    ]
    progress = capsys.readouterr().out
    assert "generation_progress finished=2 total=4" in progress
    assert "generation_progress finished=4 total=4" in progress
    records = [json.loads(line) for line in snapshot.read_text().splitlines()]
    assert [record["response_ids"] for record in records] == [
        [500, 2],
        [501, 2],
    ]
    assert [record["raw_completion"] for record in records] == ["500,2", "501,2"]
    assert [record["completion"] for record in records] == ["500", "501"]
    assert all(record["finish_reason"] == "eos" for record in records)
    assert all(record["correct"] is True for record in records)
    assert [record["insert_at"] for record in records] == [10, 11]
    assert manifest["counts"] == {
        "source_rows": 2,
        "questions": 2,
        "generation_requests": 4,
        "correct_trajectories_before_deduplication": 4,
        "duplicate_responses_filtered": 2,
        "trajectories": 2,
        "parts": 2,
    }
    assert manifest["identity"]["sampling"]["source_deduplication"] == {
        "key": "exact_problem_text",
        "policy": (
            "exclude_empty_answers_then_keep_first_if_all_exact_answers_agree_"
            "else_exclude_group"
        ),
        "raw_rows": 2,
        "empty_answer_rows_excluded": 0,
        "unique_problem_groups": 2,
        "duplicate_rows": 0,
        "conflicting_problem_groups": 0,
        "conflicting_rows_excluded": 0,
        "rows_after_deduplication": 2,
    }
    assert manifest["identity"]["sampling"]["reply_uniqueness"] == {
        "scope": "within_source_problem",
        "key": "source_row_index,response_ids",
        "enforced": True,
    }
    assert manifest["identity"]["sampling"]["backend"] == {
        "name": "sglang",
        "attention_backend": "flashinfer",
        "version": "test",
        "tensor_parallel_size": 2,
        "data_parallel_size": 4,
        "requested_batch_size": 3,
        "resolved_batch_size": 2,
        "log_level": "warning",
        "context_length": 40960,
        "model_config_overrides": {"max_position_embeddings": 40960},
        "deterministic_inference": False,
        "data_parallel_routing": {
            "mode": "explicit_problem_modulo",
            "formula": "deduplicated_row_index % data_parallel_size",
            "all_samples_for_problem_share_rank": True,
        },
    }
    assert manifest["identity"]["sampling"]["checkpointing"] == {
        "schema_version": 1,
        "unit": "whole_questions",
        "questions_per_part": 1,
        "requests_per_full_part": 2,
        "part_count": 2,
        "engine_lifecycle": "restart_per_part",
        "commit": "atomic_directory_rename",
    }

    shutil.rmtree(output / "parts" / "part-00001")
    snapshot.unlink()
    runtime.calls.clear()
    runtime.data_parallel_rank_calls.clear()
    runtime.progress_labels.clear()

    resumed_snapshot = sampling.sample(
        RunOptions(
            config=config_path,
            output=output,
            cache_dir=tmp_path / "cache",
            sglang_tp_size=2,
            sglang_dp_size=4,
            sglang_batch_size=3,
            sglang_log_level="warning",
        )
    )

    assert runtime.calls == [
        ([(11,), (11,)], [1002, 1003], 2),
    ]
    assert runtime.progress_labels == ["sample_part_2"]
    assert len(resumed_snapshot.read_text().splitlines()) == 2


def test_teacher_sampling_submits_all_trajectories_in_one_batch_by_default(
    monkeypatch, tmp_path
) -> None:
    generation = GenerationSpec(0.8, 0.9, 32, 2, None)
    config = SimpleNamespace(task="test", sampling=SamplingSpec(generation, "all"))
    rows = [
        {
            "question_id": f"q{i}",
            "problem": f"p{i}",
            "answer": "a",
            "token": i,
        }
        for i in range(5)
    ]
    tokenizer = _Tokenizer()
    runtime = _Runtime()
    task = SimpleNamespace(grade=lambda completion, answer: True)

    monkeypatch.setattr(sampling, "load_config", lambda _path: config)
    monkeypatch.setattr(sampling, "get_task", lambda _name: task)
    monkeypatch.setattr(
        sampling,
        "load_template_tokenizer",
        lambda _options, _config: (tokenizer, {}),
    )
    monkeypatch.setattr(
        sampling,
        "download_rows",
        lambda _options, _config, _task, _kind: (rows, []),
    )
    monkeypatch.setattr(
        sampling,
        "render_for_execution",
        lambda _config, *, execution_model, row, tokenizer: SimpleNamespace(
            token_ids=(row["token"],)
        ),
    )
    monkeypatch.setattr(sampling, "resolve_insertion_index", lambda *_args: 0)
    monkeypatch.setattr(sampling, "model_source", lambda *_args: ("teacher", None))
    monkeypatch.setattr(
        SGLangRuntime,
        "start",
        classmethod(lambda cls, *args: runtime),
    )
    monkeypatch.setattr(sampling, "write_manifest", lambda *_args, **_kwargs: None)

    config_path = tmp_path / "config.yaml"
    config_path.write_text("test: config\n", encoding="utf-8")
    sampling.sample(
        RunOptions(
            config=config_path,
            output=tmp_path / "sample-all",
            cache_dir=tmp_path / "cache-all",
        )
    )

    assert len(runtime.calls) == 1
    prompts, seeds, _eos = runtime.calls[0]
    assert len(prompts) == 10
    assert len(seeds) == 10


def test_sampling_source_deduplication_excludes_conflicting_groups() -> None:
    rows = [
        {"question_id": "a0", "problem": "same", "answer": "1"},
        {"question_id": "a1", "problem": "same", "answer": "1"},
        {"question_id": "b0", "problem": "conflict", "answer": "2"},
        {"question_id": "b1", "problem": "conflict", "answer": "3"},
        {"question_id": "c0", "problem": "unique", "answer": "4"},
        {"question_id": "d0", "problem": "unverifiable", "answer": ""},
    ]

    kept, identity = sampling._deduplicate_source_rows(rows)

    assert [row["question_id"] for row in kept] == ["a0", "c0"]
    assert identity == {
        "key": "exact_problem_text",
        "policy": (
            "exclude_empty_answers_then_keep_first_if_all_exact_answers_agree_"
            "else_exclude_group"
        ),
        "raw_rows": 6,
        "empty_answer_rows_excluded": 1,
        "unique_problem_groups": 3,
        "duplicate_rows": 2,
        "conflicting_problem_groups": 1,
        "conflicting_rows_excluded": 2,
        "rows_after_deduplication": 2,
    }


def test_sampling_question_id_cannot_identify_different_problems() -> None:
    rows = [
        {"question_id": "shared", "problem": "first", "answer": "1"},
        {"question_id": "shared", "problem": "second", "answer": "2"},
    ]

    with pytest.raises(RuntimeError, match="multiple different problems"):
        sampling._validate_question_id_mapping(rows, "test-task")


def test_sampling_question_id_mapping_accepts_distinct_ids() -> None:
    sampling._validate_question_id_mapping(
        [
            {"question_id": "q0", "problem": "first", "answer": "1"},
            {"question_id": "q1", "problem": "second", "answer": "2"},
        ],
        "test-task",
    )


def test_sampling_seed_does_not_collide_above_one_hundred_samples() -> None:
    seeds = {
        sampling._sampling_seed(row_index, sample_index, 128)
        for row_index in range(3)
        for sample_index in range(128)
    }
    assert len(seeds) == 3 * 128


def test_reuse_uses_only_exact_nonoverlapping_question_answer_pairs(
    monkeypatch, tmp_path
) -> None:
    generation = GenerationSpec(1.0, 1.0, 32, 4, "new provenance")
    config = SimpleNamespace(
        task="test",
        teacher=SimpleNamespace(checkpoint="teacher"),
        template=SimpleNamespace(source="teacher"),
        sampling=SamplingSpec(generation, "correct_only"),
    )
    previous_config = SimpleNamespace(
        task="test",
        teacher=config.teacher,
        template=config.template,
        sampling=SamplingSpec(
            GenerationSpec(1.0, 1.0, 32, 4, "old provenance"),
            "correct_only",
        ),
    )
    previous_rows = [
        {"question_id": "old-a", "problem": "a", "answer": "1"},
        {"question_id": "old-b", "problem": "b", "answer": "2"},
        {"question_id": "old-only", "problem": "old", "answer": "3"},
    ]
    target_rows = [
        {"question_id": "new-b", "problem": "b", "answer": "changed"},
        {"question_id": "new-a", "problem": "a", "answer": "1"},
        {"question_id": "new-only", "problem": "new", "answer": "4"},
    ]
    reuse_dir = tmp_path / "reuse"
    reuse_dir.mkdir()
    (reuse_dir / "config.yaml").write_text("old: config\n", encoding="utf-8")
    common.write_jsonl(
        reuse_dir / "teacher_trajectories.jsonl",
        [
            {
                "question_id": "old-a",
                "source_row_index": 0,
                "sample_index": 2,
                "problem": "a",
                "answer": "1",
                "response_ids": [10],
            },
            {
                "question_id": "old-b",
                "source_row_index": 1,
                "sample_index": 0,
                "problem": "b",
                "answer": "2",
                "response_ids": [20],
            },
        ],
    )
    snapshot = reuse_dir / "teacher_trajectories.jsonl"
    (reuse_dir / "manifest.json").write_text(
        json.dumps(
            {
                "stage": "sample",
                "content_sha256": {
                    "teacher_trajectories.jsonl": common.sha256_file(snapshot)
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(sampling, "load_config", lambda _path: previous_config)
    monkeypatch.setattr(sampling, "get_task", lambda _name: object())
    monkeypatch.setattr(
        sampling,
        "download_rows",
        lambda *_args: (previous_rows, [{"name": "old"}]),
    )

    records, reused_indices, identity = sampling._load_reused_trajectories(
        RunOptions(
            config=tmp_path / "new.yaml",
            output=tmp_path / "out",
            cache_dir=tmp_path / "cache",
            reuse_sample_dir=reuse_dir,
        ),
        config,
        target_rows,
    )

    assert reused_indices == {1}
    assert len(records) == 1
    assert records[0]["source_row_index"] == 1
    assert records[0]["question_id"] == "new-a"
    assert identity["overlapping_questions_reused"] == 1
    assert identity["nonoverlapping_source_questions"] == 2


def test_pending_sampling_parts_partition_questions_without_overlap() -> None:
    specs = sampling._pending_part_specs([0, 3, 4, 9, 11], 2, 4)

    question_sets = [set(spec["source_row_indices"]) for spec in specs]
    assert set.union(*question_sets) == {0, 3, 4, 9, 11}
    assert all(
        not question_sets[left] & question_sets[right]
        for left in range(len(question_sets))
        for right in range(left + 1, len(question_sets))
    )
    assert [spec["request_count"] for spec in specs] == [8, 8, 4]


def test_evaluation_overall_accuracy_is_macro_average() -> None:
    benchmarks = evaluation._benchmark_summary(
        [
            {"dataset": "small", "question_id": "s", "correct": False},
            {"dataset": "large", "question_id": "l0", "correct": True},
            {"dataset": "large", "question_id": "l1", "correct": True},
            {"dataset": "large", "question_id": "l2", "correct": True},
        ]
    )

    assert evaluation._macro_accuracy(benchmarks) == pytest.approx(0.5)


def test_reporting_overall_metrics_macro_average_unequal_benchmarks(
    tmp_path: Path,
) -> None:
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    rows = [
        {"dataset": "small", "question_id": "s", "correct": False},
        *(
            {"dataset": "large", "question_id": f"l{index}", "correct": True}
            for index in range(9)
        ),
    ]
    (evaluation_dir / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    accuracy = reporting._read_accuracy_by_benchmark(evaluation_dir)
    sample_count, pass_at_k = reporting._read_pass_at_k_by_benchmark(evaluation_dir)

    assert accuracy["small"] == 0.0
    assert accuracy["large"] == 1.0
    assert accuracy["overall"] == 0.5
    assert sample_count == 1
    assert pass_at_k["overall"]["1"] == pytest.approx(0.5)


def test_config_snapshot_is_exact_and_rejects_reuse_with_different_config(
    tmp_path,
) -> None:
    source = tmp_path / "source.yaml"
    source.write_text("name: first\n", encoding="utf-8")
    output = tmp_path / "stage"
    output.mkdir()
    options = RunOptions(source, output, tmp_path / "cache")

    snapshot = common.snapshot_config(options)
    assert snapshot.read_bytes() == source.read_bytes()
    assert common.snapshot_config(options) == snapshot

    source.write_text("name: second\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="different config snapshot"):
        common.snapshot_config(options)


def test_read_dataset_rows_supports_parquet(tmp_path) -> None:
    parquet = pytest.importorskip("pyarrow.parquet")
    pyarrow = pytest.importorskip("pyarrow")
    path = tmp_path / "rows.parquet"
    parquet.write_table(
        pyarrow.Table.from_pylist([{"problem": "q", "answer": 42.0}]), path
    )

    assert common.read_dataset_rows(path) == [{"problem": "q", "answer": 42.0}]


def test_read_dataset_rows_supports_arrow_stream(tmp_path) -> None:
    ipc = pytest.importorskip("pyarrow.ipc")
    pyarrow = pytest.importorskip("pyarrow")
    path = tmp_path / "rows.arrow"
    table = pyarrow.Table.from_pylist([{"question": "q", "gold_answer": "42"}])
    with path.open("wb") as sink:
        with ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)

    assert common.read_dataset_rows(path) == [{"question": "q", "gold_answer": "42"}]


def test_download_rows_combines_multi_artifact_rl_pool(monkeypatch, tmp_path) -> None:
    from soft_prompt_distillation.tasks import DatasetSource

    paths = {}
    for index in range(2):
        path = tmp_path / f"source-{index}.jsonl"
        path.write_text(
            json.dumps({"problem": f"q{index}", "answer": str(index)}) + "\n",
            encoding="utf-8",
        )
        paths[path.name] = path

    class MultiSourceTask:
        def get_sampling_datasets(self):
            return tuple(
                DatasetSource("owner/data", name, "train", "math") for name in paths
            )

        def include_sampling_row(self, row):
            return True

        def normalize_sampling_row(self, row):
            return {**row, "question_id": row["problem"]}

    monkeypatch.setattr(
        common,
        "hf_hub_download",
        lambda repo, filename, **kwargs: str(paths[filename]),
    )
    config = SimpleNamespace(
        data=SimpleNamespace(sampling=SimpleNamespace(num_prompts=1, selection_seed=42))
    )
    options = RunOptions(tmp_path / "config.yaml", tmp_path, tmp_path / "cache")

    rows, identities = common.download_rows(
        options, config, MultiSourceTask(), "sampling"
    )

    assert len(rows) == 1
    assert len(identities) == 2
    assert {identity["filename"] for identity in identities} == set(paths)
    assert sum(identity["record_count"] for identity in identities) == 1
    assert sum(identity["available_record_count"] for identity in identities) == 2


def test_download_rows_pools_multi_artifact_evaluation_as_one_benchmark(
    monkeypatch, tmp_path
) -> None:
    from soft_prompt_distillation.tasks import DatasetSource

    paths = {}
    for index in range(2):
        path = tmp_path / f"benchmark-{index}.jsonl"
        path.write_text(
            json.dumps({"question": f"q{index}", "answer": str(index)}) + "\n",
            encoding="utf-8",
        )
        paths[path.name] = path

    class MultiSourceEvaluationTask:
        evaluation_datasets = {
            "pooled": tuple(
                DatasetSource("owner/data", name, "test", "math") for name in paths
            )
        }

        def get_evaluation_datasets(self, label):
            return self.evaluation_datasets[label]

        def normalize_evaluation_row(self, row, label):
            return {
                "dataset": label,
                "problem": row["question"],
                "answer": row["answer"],
                "question_id": row["question"],
            }

    monkeypatch.setattr(
        common,
        "hf_hub_download",
        lambda repo, filename, **kwargs: str(paths[filename]),
    )
    config = SimpleNamespace(
        task="pooled",
        data=SimpleNamespace(
            evaluation={"pooled": SimpleNamespace(num_prompts="all", selection_seed=42)}
        ),
    )
    options = RunOptions(tmp_path / "config.yaml", tmp_path, tmp_path / "cache")

    rows, identities = common.download_rows(
        options, config, MultiSourceEvaluationTask(), "evaluation"
    )

    assert {row["dataset"] for row in rows} == {"pooled"}
    assert {row["question_id"] for row in rows} == {"q0", "q1"}
    assert len(identities) == 2
    assert sum(identity["record_count"] for identity in identities) == 2
    assert sum(identity["available_record_count"] for identity in identities) == 2


def test_manifest_retains_exact_config_snapshot(tmp_path) -> None:
    config_path = tmp_path / "source.yaml"
    config_path.write_text("schema_version: 1\nname: recorded\n", encoding="utf-8")
    output = tmp_path / "stage"
    output.mkdir()
    result = output / "result.jsonl"
    result.write_text("{}\n", encoding="utf-8")
    stage_source = tmp_path / "stage.py"
    stage_source.write_text("# stage\n", encoding="utf-8")

    common.write_manifest(
        RunOptions(config_path, output, tmp_path / "cache"),
        SimpleNamespace(),
        "test",
        {"rows": 1},
        [result],
        {},
        stage_source=stage_source,
    )

    snapshot = output / "config.yaml"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert snapshot.read_bytes() == config_path.read_bytes()
    assert manifest["content_sha256"]["config.yaml"] == common.sha256_file(snapshot)


def test_evaluate_runs_only_the_selected_plain_arm(monkeypatch, tmp_path) -> None:
    config = SimpleNamespace(
        task="test",
        evaluation=SimpleNamespace(
            arms={
                "student": SimpleNamespace(
                    generation=GenerationSpec(0.0, 1.0, 8, 1, None)
                )
            }
        ),
    )
    task = SimpleNamespace()
    rows = [{"question_id": "q0"}]
    tokenizer = object()
    calls = []

    monkeypatch.setattr(evaluation, "load_config", lambda _path: config)
    monkeypatch.setattr(evaluation, "get_task", lambda _name: task)
    monkeypatch.setattr(
        evaluation,
        "download_rows",
        lambda *_args: (rows, [{"name": "eval"}]),
    )
    monkeypatch.setattr(
        evaluation,
        "load_template_tokenizer",
        lambda _options, _config: (
            calls.append(("template_tokenizer",)) or tokenizer,
            {"models": {}, "tokenizer_vocab_sha256": "same"},
        ),
    )
    monkeypatch.setattr(
        evaluation,
        "load_tokenizer",
        lambda _options, _config, role: (
            calls.append(("execution_tokenizer", role)) or object(),
            {"models": {role: {}}, "tokenizer_vocab_sha256": "same"},
        ),
    )
    monkeypatch.setattr(
        evaluation,
        "_generate_and_score",
        lambda _options, _config, _task, _rows, _template_tokenizer, _execution_tokenizer, arm, path: (
            calls.append((arm, path))
            or [
                {
                    "dataset": "eval",
                    "question_id": "q0",
                    "input_tokenization": "marker_segmented",
                    "input_token_count": 1,
                    "input_token_ids_sha256": "aligned",
                    "canonical_input_token_count": 1,
                    "canonical_input_token_ids_sha256": "canonical",
                    "input_tokenization_matches_canonical": True,
                    "completion": "answer",
                    "finish_reason": "eos",
                    "correct": True,
                }
            ],
            {"name": "sglang", "arm": arm},
        ),
    )
    monkeypatch.setattr(evaluation, "write_manifest", lambda *_args, **_kwargs: None)

    output = tmp_path / "student-eval"
    summary = evaluation.evaluate(
        RunOptions(Path("config.yaml"), output, tmp_path / "cache"),
        "student",
    )

    assert calls == [
        ("template_tokenizer",),
        ("execution_tokenizer", "student"),
        ("student", None),
    ]
    assert (output / "results.jsonl").is_file()
    assert not (output / "teacher_eval.jsonl").exists()
    assert json.loads(summary.read_text())["arm"] == "student"


def test_teacher_tokenizer_loading_does_not_touch_student(
    monkeypatch, tmp_path
) -> None:
    class Tokenizer:
        init_kwargs = {"_commit_hash": "teacher-revision"}
        chat_template = "teacher-template"

        def get_vocab(self):
            return {"token": 0}

    loaded = []
    monkeypatch.setattr(
        common.AutoTokenizer,
        "from_pretrained",
        lambda source, revision=None, trust_remote_code=False: (
            loaded.append((source, revision, trust_remote_code)) or Tokenizer()
        ),
    )
    config = SimpleNamespace(
        teacher=SimpleNamespace(checkpoint="org/teacher"),
        template=SimpleNamespace(source="teacher"),
    )
    options = RunOptions(Path("config.yaml"), tmp_path / "out", tmp_path / "cache")

    _tokenizer, identity = common.load_template_tokenizer(options, config)

    assert loaded == [("org/teacher", None, False)]
    assert set(identity["models"]) == {"teacher"}
    assert identity["template_source"] == "teacher"


def test_matching_vocab_allows_different_chat_templates() -> None:
    template_identity = {
        "tokenizer_vocab_sha256": "a" * 64,
        "template_chat_sha256": "b" * 64,
    }
    student_identity = {
        "tokenizer_vocab_sha256": "a" * 64,
        "template_chat_sha256": "c" * 64,
    }

    common.require_matching_tokenizer_vocabulary(template_identity, student_identity)


def test_different_execution_vocabulary_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="execution-model tokenizer vocabulary"):
        common.require_matching_tokenizer_vocabulary(
            {"tokenizer_vocab_sha256": "a" * 64},
            {"tokenizer_vocab_sha256": "b" * 64},
        )


def test_snapshot_template_identity_includes_chat_template(tmp_path) -> None:
    snapshot_dir = tmp_path / "sample"
    snapshot_dir.mkdir()
    snapshot = snapshot_dir / "teacher_trajectories.jsonl"
    snapshot.write_text("{}\n", encoding="utf-8")
    (snapshot_dir / "manifest.json").write_text(
        json.dumps(
            {
                "source": {
                    "tokenizer_vocab_sha256": "a" * 64,
                    "template_chat_sha256": "b" * 64,
                    "template_source": "teacher",
                }
            }
        ),
        encoding="utf-8",
    )

    assert common.snapshot_template_identity(snapshot) == {
        "tokenizer_vocab_sha256": "a" * 64,
        "template_chat_sha256": "b" * 64,
        "template_source": "teacher",
        "template_render_mode": None,
    }


def test_balanced_model_loading_shards_without_cuda_copy(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    captured = {}

    class FakeModel:
        def __init__(self):
            self.embedding = SimpleNamespace(
                weight=SimpleNamespace(dtype=torch.bfloat16)
            )
            self.cuda_called = False

        def cuda(self):
            self.cuda_called = True
            return self

        def eval(self):
            return self

        def get_input_embeddings(self):
            return self.embedding

    model = FakeModel()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(common, "is_flash_attn_2_available", lambda: True)
    monkeypatch.setattr(
        common.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(model_type="fake"),
    )
    monkeypatch.setattr(
        common.AutoModelForCausalLM,
        "from_pretrained",
        lambda source, **kwargs: captured.update(source=source, **kwargs) or model,
    )

    actual = common.load_model(
        "org/student",
        None,
        placement="balanced",
        attention_implementation="flash_attention_2",
        trust_remote_code=True,
        config_overrides={"max_position_embeddings": 40960},
    )

    assert actual is model
    assert captured["device_map"] == "balanced"
    assert captured["dtype"] == torch.bfloat16
    assert captured["attn_implementation"] == "flash_attention_2"
    assert captured["trust_remote_code"] is True
    assert captured["max_position_embeddings"] == 40960
    assert model.cuda_called is False


def test_gpt_oss_flash_attention_2_fails_before_weight_loading(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(common, "is_flash_attn_2_available", lambda: True)
    monkeypatch.setattr(
        common.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(model_type="gpt_oss"),
    )
    weight_loads = []
    monkeypatch.setattr(
        common.AutoModelForCausalLM,
        "from_pretrained",
        lambda *_args, **_kwargs: weight_loads.append(True),
    )

    with pytest.raises(RuntimeError, match="learned attention sinks"):
        common.load_model(
            "org/gpt-oss",
            None,
            attention_implementation="flash_attention_2",
        )

    assert weight_loads == []


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_evaluation_expands_multiple_samples_with_stable_indices(
    monkeypatch, tmp_path, batch_size
) -> None:
    generation = GenerationSpec(0.8, 0.9, 32, 2, "formal recipe")
    short_generation = GenerationSpec(0.8, 0.9, 32, 1, "short recipe")
    config = SimpleNamespace(
        student=SimpleNamespace(
            trust_remote_code=False,
            context_length=131072,
            config_overrides={"max_position_embeddings": 131072},
        ),
        evaluation=SimpleNamespace(
            arms={"student": SimpleNamespace(generation=generation)},
            benchmarks={"short_eval": short_generation},
        ),
    )
    rows = [
        {"dataset": "eval", "question_id": "q0", "answer": "a", "token": 10},
        {
            "dataset": "short_eval",
            "question_id": "q1",
            "answer": "a",
            "token": 11,
        },
    ]

    class EvalTokenizer:
        eos_token_id = 2

        def decode(self, token_ids, *, skip_special_tokens):
            prefix = "clean:" if skip_special_tokens else "raw:"
            return prefix + ",".join(str(value) for value in token_ids)

    class EvalRuntime(_Runtime):
        def generate_token_ids(
            self, prompts, spec, seeds, eos_token_id, *, forbidden_token_ids=(), data_parallel_ranks=None
        ):
            assert forbidden_token_ids == ()
            return super().generate_token_ids(prompts, spec, seeds, eos_token_id, data_parallel_ranks=data_parallel_ranks)

    runtime = EvalRuntime()
    start_options = []
    monkeypatch.setattr(
        evaluation,
        "render_for_execution",
        lambda _config, *, execution_model, row, tokenizer, execution_tokenizer, template=None: (
            SimpleNamespace(
                token_ids=(row["token"],),
                messages=(
                    {"role": "system", "content": "Solve carefully."},
                    {"role": "user", "content": f"Question: {row['question_id']}"},
                ),
                text=f"system=Solve carefully.|user=Question: {row['question_id']}",
                tokenization_strategy="marker_segmented",
                token_ids_sha256=f"aligned-{row['token']}",
                canonical_token_count=1,
                canonical_token_ids_sha256=f"canonical-{row['token']}",
                tokenization_matches_canonical=True,
            )
        ),
    )
    monkeypatch.setattr(evaluation, "model_source", lambda *_args: ("student", None))
    monkeypatch.setattr(
        SGLangRuntime,
        "start",
        classmethod(
            lambda cls, *args, **kwargs: start_options.append(kwargs) or runtime
        ),
    )

    graded_completions = []
    scored, backend = evaluation._generate_and_score(
        RunOptions(
            Path("config.yaml"),
            tmp_path / "out",
            tmp_path / "cache",
            sglang_batch_size=batch_size,
            sglang_dp_size=2,
        ),
        config,
        SimpleNamespace(
            grade=lambda completion, answer: True,
            grade_evaluation=lambda completion, row: (
                graded_completions.append(completion) or True
            ),
            evaluation_grader=lambda row: "test-evaluator",
            parse_completion=lambda completion: CompletionParse(
                extractor="test",
                reasoning_text="reasoning",
                final_text=completion,
                extracted_answer="answer",
            ),
        ),
        rows,
        EvalTokenizer(),
        EvalTokenizer(),
        "student",
        None,
    )

    assert [p for call in runtime.calls for p in call[0]] == [[10], [10], [11]]
    assert [seed for call in runtime.calls for seed in call[1]] == [2000, 2001, 2002]
    assert [rank for call in runtime.data_parallel_rank_calls for rank in call] == [0, 0, 1]
    assert backend["data_parallel_routing"]["all_samples_for_problem_share_rank"] is True
    assert start_options == [
        {
            "context_length": 131072,
            "model_config_overrides": {"max_position_embeddings": 131072},
        }
    ]
    assert [row["sample_index"] for row in scored] == [0, 1, 0]
    assert scored[0]["input_messages"] == [
        {"role": "system", "content": "Solve carefully."},
        {"role": "user", "content": "Question: q0"},
    ]
    assert scored[0]["input_text"] == ("system=Solve carefully.|user=Question: q0")
    assert scored[0]["input_tokenization"] == "marker_segmented"
    assert scored[0]["input_token_count"] == 1
    assert scored[0]["input_token_ids_sha256"] == "aligned-10"
    assert scored[0]["canonical_input_token_count"] == 1
    assert scored[0]["canonical_input_token_ids_sha256"] == "canonical-10"
    assert scored[0]["input_tokenization_matches_canonical"] is True
    assert evaluation._tokenization_summary(scored) == {
        "strategy": "marker_segmented",
        "questions": 2,
        "matches_canonical": 2,
        "differs_from_canonical": 0,
    }
    assert scored[0]["input_messages"] is not scored[1]["input_messages"]
    assert scored[0]["response_ids"] == [1000, 2]
    assert scored[0]["raw_completion"] == "raw:1000,2"
    assert scored[0]["completion"] == "clean:1000,2"
    assert scored[0]["reasoning_text"] == "reasoning"
    assert scored[0]["final_text"] == "clean:1000,2"
    assert scored[0]["extracted_answer"] == "answer"
    assert scored[0]["extractor"] == "test"
    assert scored[0]["grader"] == "test-evaluator"
    assert graded_completions == [row["completion"] for row in scored]
    assert backend["samples_per_prompt"] == 2
    assert backend["request_count"] == 3
    assert backend["context_length"] == 131072
    assert backend["model_config_overrides"] == {"max_position_embeddings": 131072}
    assert backend["benchmark_generation"]["eval"]["samples_per_prompt"] == 2
    assert backend["benchmark_generation"]["short_eval"]["samples_per_prompt"] == 1


def test_evaluation_seed_is_arm_independent(monkeypatch) -> None:
    monkeypatch.delenv("SOFT_PROMPT_EVALUATION_SEED_OFFSET", raising=False)
    expected = [2000, 2001, 2002, 2003]
    for _arm in ("student", "teacher", "student_soft_prompt"):
        assert [evaluation._evaluation_seed(index) for index in range(4)] == expected


def test_evaluation_seed_offset_selects_an_independent_stream(monkeypatch) -> None:
    monkeypatch.setenv("SOFT_PROMPT_EVALUATION_SEED_OFFSET", "100000")
    assert [evaluation._evaluation_seed(index) for index in range(3)] == [
        102000,
        102001,
        102002,
    ]


def test_evaluation_seed_offset_rejects_non_integer(monkeypatch) -> None:
    monkeypatch.setenv("SOFT_PROMPT_EVALUATION_SEED_OFFSET", "different")
    with pytest.raises(RuntimeError, match="must be an integer"):
        evaluation._evaluation_seed(0)


def test_evaluation_selects_generation_recipe_per_benchmark() -> None:
    default = GenerationSpec(0.6, 0.95, 32768, 64, "AIME avg@64")
    math_500 = GenerationSpec(0.6, 0.95, 32768, 5, "MATH avg@5")
    config = SimpleNamespace(
        evaluation=SimpleNamespace(
            arms={"student": SimpleNamespace(generation=default)},
            benchmarks={"math_500": math_500},
        )
    )

    assert evaluation._generation_for_dataset(config, "student", "aime_2024") is default
    assert evaluation._generation_for_dataset(config, "student", "math_500") is math_500


def test_posthoc_checkpoint_evaluation_supports_all_resume_and_cleanup(
    monkeypatch, tmp_path
) -> None:
    train_dir = tmp_path / "train"
    checkpoint_dir = train_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    for update in (5, 10):
        (checkpoint_dir / f"update_{update:08d}.safetensors").write_bytes(b"prompt")
    exported_paths = []
    evaluated_updates = []

    def fake_export(options, checkpoint):
        options.output.mkdir(parents=True)
        exported_paths.append(options.output)
        (options.output / "manifest.json").write_text("{}\n", encoding="utf-8")
        (options.output / "soft_prompt_tokens.json").write_text(
            "{}\n", encoding="utf-8"
        )
        return options.output

    def fake_evaluate(options, arm):
        assert arm == "student_soft_prompt"
        assert options.soft_prompt_model_path is not None
        name = options.output.name.removeprefix(".").removesuffix(".staging")
        update = int(name.removeprefix("update_"))
        evaluated_updates.append(update)
        options.output.mkdir(parents=True)
        (options.output / "results.jsonl").write_text("{}\n", encoding="utf-8")
        (options.output / "summary.json").write_text(
            json.dumps({"accuracy": update / 100}) + "\n", encoding="utf-8"
        )
        (options.output / "manifest.json").write_text("{}\n", encoding="utf-8")
        return options.output / "summary.json"

    monkeypatch.setattr(pipeline, "export_soft_prompt_model", fake_export)
    monkeypatch.setattr(pipeline, "evaluate", fake_evaluate)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("name: checkpoint-test\n", encoding="utf-8")
    options = RunOptions(config_path, tmp_path / "checkpoint-evals", tmp_path / "cache")

    report_path = pipeline.evaluate_checkpoints(
        options, train_dir, "all", tmp_path / "staging"
    )

    assert evaluated_updates == [5, 10]
    assert (options.output / "config.yaml").read_bytes() == config_path.read_bytes()
    assert all(not path.exists() for path in exported_paths)
    assert all(
        (options.output / f"update_{update:08d}" / "export_manifest.json").is_file()
        for update in (5, 10)
    )
    report = json.loads(report_path.read_text())
    assert [item["update"] for item in report["evaluations"]] == [5, 10]

    pipeline.evaluate_checkpoints(options, train_dir, "5", tmp_path / "staging")
    assert evaluated_updates == [5, 10]


def test_checkpoint_evaluation_retries_after_a_partial_runtime_failure(
    monkeypatch, tmp_path
) -> None:
    train_dir = tmp_path / "train"
    checkpoint_dir = train_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "update_00000005.safetensors").write_bytes(b"prompt")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("name: atomic-retry\n", encoding="utf-8")
    output = tmp_path / "checkpoint-evals"
    options = RunOptions(config_path, output, tmp_path / "cache")

    def fake_export(actual_options, _checkpoint):
        actual_options.output.mkdir(parents=True)
        (actual_options.output / "manifest.json").write_text("{}\n", encoding="utf-8")

    attempts = 0

    def flaky_evaluate(actual_options, _arm):
        nonlocal attempts
        attempts += 1
        actual_options.output.mkdir(parents=True)
        (actual_options.output / "results.jsonl").write_text(
            "partial\n", encoding="utf-8"
        )
        if attempts == 1:
            raise RuntimeError("simulated engine failure")
        (actual_options.output / "summary.json").write_text("{}\n", encoding="utf-8")
        (actual_options.output / "manifest.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(pipeline, "export_soft_prompt_model", fake_export)
    monkeypatch.setattr(pipeline, "evaluate", flaky_evaluate)

    with pytest.raises(RuntimeError, match="simulated engine failure"):
        pipeline.evaluate_checkpoints(options, train_dir, "5")

    final = output / "update_00000005"
    staging = output / ".update_00000005.staging"
    assert not final.exists()
    assert (staging / "results.jsonl").is_file()

    pipeline.evaluate_checkpoints(options, train_dir, "5")

    assert (final / "export_manifest.json").is_file()
    assert not staging.exists()
    preserved = output / ".update_00000005.staging.incomplete-001"
    assert (preserved / "results.jsonl").read_text(encoding="utf-8") == "partial\n"


def test_checkpoint_evaluation_quarantines_a_legacy_partial_final_directory(
    monkeypatch, tmp_path
) -> None:
    train_dir = tmp_path / "train"
    checkpoint_dir = train_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "update_00000005.safetensors").write_bytes(b"prompt")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("name: legacy-partial\n", encoding="utf-8")
    output = tmp_path / "checkpoint-evals"
    partial = output / "update_00000005"
    partial.mkdir(parents=True)
    (partial / "results.jsonl").write_text("old-partial\n", encoding="utf-8")

    def fake_export(actual_options, _checkpoint):
        actual_options.output.mkdir(parents=True)
        (actual_options.output / "manifest.json").write_text("{}\n", encoding="utf-8")

    def fake_evaluate(actual_options, _arm):
        actual_options.output.mkdir(parents=True)
        for name, content in (
            ("results.jsonl", "{}\n"),
            ("summary.json", "{}\n"),
            ("manifest.json", "{}\n"),
        ):
            (actual_options.output / name).write_text(content, encoding="utf-8")

    monkeypatch.setattr(pipeline, "export_soft_prompt_model", fake_export)
    monkeypatch.setattr(pipeline, "evaluate", fake_evaluate)

    pipeline.evaluate_checkpoints(
        RunOptions(config_path, output, tmp_path / "cache"), train_dir, "5"
    )

    assert (partial / "export_manifest.json").is_file()
    preserved = output / "update_00000005.incomplete-001"
    assert (preserved / "results.jsonl").read_text(encoding="utf-8") == (
        "old-partial\n"
    )


@pytest.mark.parametrize(
    ("selector", "expected_refreshes"),
    (("all", [[0], [0, 20]]), ("all-desc", [[20], [0, 20]])),
)
def test_checkpoint_evaluation_refreshes_recovery_after_each_completed_point(
    monkeypatch, tmp_path, selector, expected_refreshes
) -> None:
    run_dir = tmp_path / "run"
    train_dir = run_dir / "train"
    checkpoint_dir = train_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    for update in (0, 20):
        (checkpoint_dir / f"update_{update:08d}.safetensors").write_bytes(b"prompt")

    def write_results(directory: Path, correct: int) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "dataset": "aime_2024",
                "question_id": "q0",
                "sample_index": index,
                "correct": index < correct,
            }
            for index in range(4)
        ]
        (directory / "results.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    write_results(run_dir / "evaluate_student", 1)
    write_results(run_dir / "evaluate_teacher", 3)

    def fake_export(options, checkpoint):
        options.output.mkdir(parents=True)
        (options.output / "manifest.json").write_text("{}\n", encoding="utf-8")
        return options.output

    def fake_evaluate(options, arm):
        assert arm == "student_soft_prompt"
        name = options.output.name.removeprefix(".").removesuffix(".staging")
        update = int(name.removeprefix("update_"))
        write_results(options.output, 1 if update == 0 else 2)
        (options.output / "summary.json").write_text(
            json.dumps({"accuracy": 0.25 if update == 0 else 0.5}) + "\n",
            encoding="utf-8",
        )
        (options.output / "manifest.json").write_text("{}\n", encoding="utf-8")
        return options.output / "summary.json"

    refreshes = []
    real_write_recovery_artifacts = pipeline.write_recovery_artifacts

    def recording_write_recovery_artifacts(experiment_dir, output_dir, evaluations):
        evaluation_list = list(evaluations)
        refreshes.append([item["update"] for item in evaluation_list])
        return real_write_recovery_artifacts(
            experiment_dir, output_dir, evaluation_list
        )

    monkeypatch.setattr(pipeline, "export_soft_prompt_model", fake_export)
    monkeypatch.setattr(pipeline, "evaluate", fake_evaluate)
    monkeypatch.setattr(
        pipeline, "write_recovery_artifacts", recording_write_recovery_artifacts
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("name: incremental-recovery\n", encoding="utf-8")
    output_dir = run_dir / "evaluate_student_soft_prompt"

    report_path = pipeline.evaluate_checkpoints(
        RunOptions(config_path, output_dir, tmp_path / "cache"),
        train_dir,
        selector,
        tmp_path / "staging",
    )

    assert refreshes == expected_refreshes
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert [
        item["recovery"]["recovery_ratio"]["overall"] for item in report["evaluations"]
    ] == pytest.approx([0.0, 0.5])
    for update, expected in ((0, 0.0), (20, 0.5)):
        recovery_path = output_dir / f"update_{update:08d}" / "recovery.json"
        recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
        assert recovery["recovery_ratio"]["overall"] == pytest.approx(expected)
    assert (output_dir / "recovery_curve.svg").is_file()
    assert (output_dir / "recovery_curve_overall.svg").is_file()

    refreshes.clear()
    pipeline.evaluate_checkpoints(
        RunOptions(config_path, output_dir, tmp_path / "cache"),
        train_dir,
        selector,
        tmp_path / "staging",
    )
    assert refreshes == [[0, 20]]


def test_checkpoint_selector_rejects_missing_update(tmp_path) -> None:
    (tmp_path / "train" / "checkpoints").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="does not exist"):
        pipeline.resolve_checkpoint_paths(tmp_path / "train", "7")


def test_soft_prompt_evaluation_requires_derived_model_before_output_creation(
    tmp_path,
) -> None:
    output = tmp_path / "missing-derived-model"
    with pytest.raises(RuntimeError, match="requires --soft-prompt-model-path"):
        evaluation.evaluate(
            RunOptions(Path("config.yaml"), output, tmp_path / "cache"),
            "student_soft_prompt",
        )
    assert not output.exists()


def test_unified_evaluate_target_routes_plain_and_soft_prompt(
    monkeypatch, tmp_path
) -> None:
    seen = []
    options = RunOptions(Path("config.yaml"), tmp_path / "eval", tmp_path / "cache")
    monkeypatch.setattr(
        pipeline,
        "evaluate",
        lambda actual_options, arm: (
            seen.append(("plain", actual_options, arm))
            or actual_options.output / "summary.json"
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "evaluate_checkpoints",
        lambda actual_options, train_dir, checkpoint, temporary_dir: (
            seen.append(
                ("soft-prompt", actual_options, train_dir, checkpoint, temporary_dir)
            )
            or actual_options.output / "checkpoint_summary.json"
        ),
    )

    pipeline.evaluate_target(options, "student")
    pipeline.evaluate_target(
        options,
        "soft-prompt",
        train_dir=tmp_path / "train",
        checkpoint="all",
        temporary_dir=tmp_path / "scratch",
    )

    assert seen == [
        ("plain", options, "student"),
        (
            "soft-prompt",
            options,
            tmp_path / "train",
            "all",
            tmp_path / "scratch",
        ),
    ]


def test_unified_soft_prompt_target_defaults_to_final(monkeypatch, tmp_path) -> None:
    captured = {}
    monkeypatch.setattr(
        pipeline,
        "evaluate_checkpoints",
        lambda options, train_dir, checkpoint, temporary_dir: (
            captured.update(checkpoint=checkpoint)
            or options.output / "checkpoint_summary.json"
        ),
    )

    pipeline.evaluate_target(
        RunOptions(Path("config.yaml"), tmp_path / "eval", tmp_path / "cache"),
        "soft-prompt",
        train_dir=tmp_path / "train",
    )

    assert captured["checkpoint"] == "final"


def test_run_smoke_uses_unified_soft_prompt_evaluation(monkeypatch, tmp_path) -> None:
    seen = {}

    def fake_sample(options):
        seen["sample"] = options.output
        return options.output / "teacher_trajectories.jsonl"

    def fake_train(options, snapshot):
        seen["train"] = (options.output, snapshot)
        return options.output / "soft_prompt.safetensors"

    def fake_evaluate_target(
        options, target, *, train_dir=None, checkpoint=None, temporary_dir=None
    ):
        seen["evaluate"] = (
            options.output,
            target,
            train_dir,
            checkpoint,
            temporary_dir,
        )
        return options.output / "checkpoint_summary.json"

    monkeypatch.setattr(pipeline, "sample", fake_sample)
    monkeypatch.setattr(pipeline, "train", fake_train)
    monkeypatch.setattr(pipeline, "evaluate_target", fake_evaluate_target)

    root = tmp_path / "run"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("name: smoke-test\n", encoding="utf-8")
    result = pipeline.run_smoke(RunOptions(config_path, root, tmp_path / "cache"))

    assert seen == {
        "sample": root / "sample",
        "train": (
            root / "train",
            root / "sample" / "teacher_trajectories.jsonl",
        ),
        "evaluate": (
            root / "evaluate_student_soft_prompt",
            "soft-prompt",
            root / "train",
            None,
            None,
        ),
    }
    assert result == root / "evaluate_student_soft_prompt" / "checkpoint_summary.json"
    assert (root / "config.yaml").read_bytes() == config_path.read_bytes()


def test_checkpoint_selector_includes_and_accepts_step_zero(tmp_path) -> None:
    checkpoint_dir = tmp_path / "train" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    step_zero = checkpoint_dir / "update_00000000.safetensors"
    step_five = checkpoint_dir / "update_00000005.safetensors"
    step_zero.write_bytes(b"initial")
    step_five.write_bytes(b"trained")

    assert pipeline.resolve_checkpoint_paths(tmp_path / "train", "0") == [step_zero]
    assert pipeline.resolve_checkpoint_paths(tmp_path / "train", "all") == [
        step_zero,
        step_five,
    ]
    assert pipeline.resolve_checkpoint_paths(tmp_path / "train", "all-desc") == [
        step_five,
        step_zero,
    ]


def test_periodic_checkpoint_selector_keeps_interval_and_final(tmp_path) -> None:
    checkpoint_dir = tmp_path / "train" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    paths = {}
    for update in (0, 10, 20, 30, 40, 46):
        path = checkpoint_dir / f"update_{update:08d}.safetensors"
        path.write_bytes(b"prompt")
        paths[update] = path
    (tmp_path / "train" / "manifest.json").write_text("{}\n", encoding="utf-8")

    assert pipeline.resolve_checkpoint_paths(tmp_path / "train", "every:20") == [
        paths[0],
        paths[20],
        paths[40],
        paths[46],
    ]


def test_periodic_checkpoint_selector_ignores_nonfinal_latest_during_training(
    tmp_path,
) -> None:
    checkpoint_dir = tmp_path / "train" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    paths = {}
    for update in (0, 20, 40, 46):
        path = checkpoint_dir / f"update_{update:08d}.safetensors"
        path.write_bytes(b"prompt")
        paths[update] = path

    assert pipeline.resolve_checkpoint_paths(tmp_path / "train", "every:20") == [
        paths[0],
        paths[20],
        paths[40],
    ]


def test_recovery_report_writes_per_benchmark_curve_with_step_zero(tmp_path) -> None:
    experiment_dir = tmp_path / "run"

    def write_results(directory: Path, scores: dict[str, tuple[int, int]]) -> None:
        directory.mkdir(parents=True)
        rows = []
        for benchmark, (correct, total) in scores.items():
            rows.extend(
                {
                    "dataset": benchmark,
                    "question_id": f"{benchmark}-question",
                    "sample_index": index,
                    "correct": index < correct,
                }
                for index in range(total)
            )
        (directory / "results.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    write_results(
        experiment_dir / "evaluate_student", {"aime_2024": (1, 4), "aime_2025": (2, 4)}
    )
    write_results(
        experiment_dir / "evaluate_teacher", {"aime_2024": (3, 4), "aime_2025": (4, 4)}
    )
    step_zero_dir = experiment_dir / "evaluate_student_soft_prompt" / "update_00000000"
    step_ten_dir = experiment_dir / "evaluate_student_soft_prompt" / "update_00000010"
    write_results(step_zero_dir, {"aime_2024": (1, 4), "aime_2025": (2, 4)})
    write_results(step_ten_dir, {"aime_2024": (2, 4), "aime_2025": (3, 4)})
    evaluations = [
        {
            "update": 0,
            "checkpoint_name": "update_00000000",
            "evaluation_dir": str(step_zero_dir),
        },
        {
            "update": 10,
            "checkpoint_name": "update_00000010",
            "evaluation_dir": str(step_ten_dir),
        },
    ]

    artifacts = reporting.write_recovery_artifacts(
        experiment_dir, experiment_dir / "evaluate_student_soft_prompt", evaluations
    )

    report = json.loads(Path(artifacts["json"]).read_text(encoding="utf-8"))
    assert report["benchmarks"] == ["overall", "aime_2024", "aime_2025"]
    assert [point["step"] for point in report["points"]] == [0, 10]
    assert report["points"][0]["recovery_ratio"]["overall"] == pytest.approx(0.0)
    assert report["points"][1]["recovery_ratio"]["overall"] == pytest.approx(0.5)
    assert report["denominator_epsilon"] == 1e-6
    assert Path(artifacts["csv"]).is_file()
    svg = Path(artifacts["svg"]).read_text(encoding="utf-8")
    assert "pass@1 recovery by benchmark" in svg
    assert "aime_2024" in svg
    assert 'font-weight="600" fill=' not in svg
    overall_svg = Path(artifacts["overall_svg"]).read_text(encoding="utf-8")
    assert overall_svg.count('font-weight="600" fill=') == 2
    assert ">0.0%</text>" in overall_svg
    assert ">50.0%</text>" in overall_svg
    pass_report = json.loads(
        Path(artifacts["pass_at_k"]["json"]).read_text(encoding="utf-8")
    )
    assert pass_report["max_k"] == 4
    assert pass_report["evaluated_k"] == [1, 2, 4]
    assert pass_report["recovery_denominator_epsilon"] == 1e-6
    assert pass_report["baselines"]["student"]["aime_2024"]["1"] == pytest.approx(0.25)
    assert pass_report["baselines"]["student"]["aime_2024"]["4"] == pytest.approx(1.0)
    assert pass_report["points"][0]["recovery_at_k"]["aime_2024"]["1"] == pytest.approx(
        0.0
    )
    assert pass_report["points"][1]["recovery_at_k"]["aime_2024"]["1"] == pytest.approx(
        0.5
    )
    assert Path(artifacts["pass_at_k"]["csv"]).is_file()
    assert set(artifacts["pass_at_k"]["svg"]) == {"aime_2024", "aime_2025"}
    assert all(Path(path).is_file() for path in artifacts["pass_at_k"]["svg"].values())
    benchmark_svg = Path(artifacts["pass_at_k"]["svg"]["aime_2024"]).read_text(
        encoding="utf-8"
    )
    assert "Optimizer step" in benchmark_svg
    assert "Teacher-gap recovery" in benchmark_svg
    assert "pass@1" in benchmark_svg
    assert "pass@2" in benchmark_svg
    assert "pass@4" in benchmark_svg
    assert Path(artifacts["pass_at_k"]["pass_at_1_by_benchmark_svg"]).is_file()
    assert Path(artifacts["pass_at_k"]["overall_pass_at_1_svg"]).is_file()


def test_recovery_treats_tiny_teacher_gap_as_undefined() -> None:
    assert reporting._recovery(0.7, 0.5, 0.5) is None
    assert reporting._recovery(0.7, 0.5, 0.5000005) is None
    assert reporting._recovery(0.4, 0.5, 0.45) is None
    assert reporting._recovery(0.7, 0.5, 0.500002) == pytest.approx(100000.0)
