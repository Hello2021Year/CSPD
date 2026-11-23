from __future__ import annotations

import pytest

from soft_prompt_distillation.tasks import TaskAdapter, get_task


def test_deepscaler_adapter_normalizes_both_aime_schemas() -> None:
    task = get_task("deepscaler_math")
    assert isinstance(task, TaskAdapter)
    assert task.sampling_dataset.repo == "agentica-org/DeepScaleR-Preview-Dataset"
    assert set(task.evaluation_datasets) == {
        "aime_2024",
        "aime_2025",
        "aime_2025_i",
        "aime_2026",
        "math_500",
        "gsm8k",
        "amc_2022_2023",
        "minerva_math",
        "olympiad_bench",
        "hmmt_feb_2025",
    }
    chat = task.normalize_evaluation_row(
        {"prompt": [{"role": "user", "content": "q24"}], "label": "24"},
        "aime_2024",
    )
    plain = task.normalize_evaluation_row(
        {"problem": "q26", "answer": 26, "id": 7}, "aime_2026"
    )
    assert (chat["problem"], chat["answer"]) == ("q24", "24")
    assert (plain["problem"], plain["answer"], plain["question_id"]) == (
        "q26",
        "26",
        "7",
    )


def test_evaluation_prompt_uses_only_the_single_user_message() -> None:
    task = get_task("deepscaler_math")
    normalized = task.normalize_evaluation_row(
        {
            "prompt": [
                {"role": "system", "content": "do not copy this"},
                {"role": "user", "content": "actual problem"},
                {"role": "assistant", "content": "leaked answer"},
            ],
            "label": "24",
        },
        "aime_2024",
    )
    assert normalized["problem"] == "actual problem"

    with pytest.raises(ValueError, match="exactly one nonempty user"):
        task.normalize_evaluation_row(
            {
                "prompt": [
                    {"role": "user", "content": "first"},
                    {"role": "user", "content": "second"},
                ],
                "label": "24",
            },
            "aime_2024",
        )


@pytest.mark.parametrize("field", ["gold_answer", "final_answer", "clean_answer"])
def test_evaluation_answer_aliases_are_normalized(field: str) -> None:
    task = get_task("deepscaler_math")
    normalized = task.normalize_evaluation_row(
        {"problem": "q", field: [24], "id": "q"}, "aime_2024"
    )
    assert normalized["answer"] == "24"


def test_deepscaler_adapter_normalizes_official_math_and_amc_schemas() -> None:
    task = get_task("deepscaler_math")
    math = task.normalize_evaluation_row(
        {
            "problem": "Find x.",
            "solution": "After simplifying, we get \\boxed{\\frac{14}{3}}.",
            "answer": "\\frac{14}{3}",
            "unique_id": "test/algebra/1.json",
        },
        "math_500",
    )
    amc = task.normalize_evaluation_row(
        {
            "prompt": [
                {
                    "role": "user",
                    "content": (
                        "Find m+n. Let's think step by step and output the final "
                        "answer within \\boxed{}."
                    ),
                }
            ],
            "reward_model": {"ground_truth": 142.0, "style": "rule"},
            "extra_info": {"index": 0, "split": "test"},
        },
        "amc_2022_2023",
    )
    assert (math["problem"], math["answer"], math["question_id"]) == (
        "Find x.",
        "\\frac{14}{3}",
        "test/algebra/1.json",
    )
    assert math["evaluation_reference"] == (
        "After simplifying, we get \\boxed{\\frac{14}{3}}."
    )
    assert math["evaluation_reference_format"] == "latex"
    assert (amc["problem"], amc["answer"], amc["question_id"]) == (
        "Find m+n.",
        "142",
        "0",
    )

    minerva = task.normalize_evaluation_row(
        {"problem": "Integrate.", "solution": "\\boxed{1}", "idx": 9},
        "minerva_math",
    )
    olympiad = task.normalize_evaluation_row(
        {"question": "Prove and compute.", "answer": "\\boxed{2}", "id": "o2"},
        "olympiad_bench",
    )
    assert minerva["answer"] == "\\boxed{1}"
    assert minerva["question_id"] == "9"
    assert olympiad["problem"] == "Prove and compute."
    hmmt = task.normalize_evaluation_row(
        {"problem": "HMMT question", "answer": "5", "problem_idx": 12},
        "hmmt_feb_2025",
    )
    assert hmmt["question_id"] == "12"


def test_gsm8k_uses_only_the_published_final_answer_marker() -> None:
    task = get_task("deepscaler_math")
    normalized = task.normalize_evaluation_row(
        {
            "question": "How many?",
            "answer": "work <<1000+250=1250>>\n#### 1,250",
        },
        "gsm8k",
    )
    assert normalized["problem"] == "How many?"
    assert normalized["answer"] == "1250"

    with pytest.raises(ValueError, match="#### final marker"):
        task.normalize_evaluation_row(
            {"question": "How many?", "answer": "1250"}, "gsm8k"
        )


def test_klear_and_eurus_adapters_use_published_rl_rows() -> None:
    klear = get_task("klear_math")
    assert klear.sampling_dataset.repo == "Kwai-Klear/KlearReasoner-MathSub-30K"
    assert klear.normalize_sampling_row(
        {
            "prompt": [{"role": "user", "content": "Find x."}],
            "reward_model": {"ground_truth": "24"},
            "__index_level_0__": 3,
        }
    ) == {"problem": "Find x.", "answer": "24", "question_id": "3"}

    eurus = get_task("eurus_math")
    assert eurus.include_sampling_row({"ability": "math"})
    assert not eurus.include_sampling_row({"ability": "code"})

    def eurus_row(problem: str, index: int) -> dict:
        return eurus.normalize_sampling_row(
            {
                "prompt": [
                    {"role": "system", "content": "ignored source system"},
                    {
                        "role": "user",
                        "content": problem
                        + "\n\nPresent the answer in LaTex format: "
                        + "\\boxed{Your answer}",
                    },
                ],
                "reward_model": {"ground_truth": "7"},
                "extra_info": {"index": index},
            }
        )

    first = eurus_row("Compute y.", 0)
    assert first["problem"] == "Compute y."
    assert first["answer"] == "7"
    # Eurus-2-RL-Data sets extra_info["index"] to 0 on every row, so the
    # identity must NOT come from it: distinct problems must get distinct ids
    # even when the index is identical, and the same problem must be stable
    # across differing indices. Deriving the id from extra_info collapsed a
    # 17,019-trajectory snapshot to 2 once the per-question cap applied.
    assert first["question_id"] != eurus_row("Compute z.", 0)["question_id"]
    assert first["question_id"] == eurus_row("Compute y.", 99)["question_id"]


def test_eurus_rejects_rows_without_the_published_suffix() -> None:
    task = get_task("eurus_math")
    with pytest.raises(ValueError, match="published user suffix"):
        task.normalize_sampling_row(
            {
                "prompt": [{"role": "user", "content": "Compute y."}],
                "reward_model": {"ground_truth": "7"},
            }
        )


@pytest.mark.parametrize(
    ("task_name", "repo", "filename", "row", "answer"),
    [
        
        
        
    ],
)
def test_pair_specific_simple_rl_schemas(
    task_name: str,
    repo: str,
    filename: str,
    row: dict[str, object],
    answer: str,
) -> None:
    task = get_task(task_name)
    assert task.sampling_dataset.repo == repo
    assert task.sampling_dataset.filename == filename
    assert task.normalize_sampling_row(row)["answer"] == answer


def test_unknown_task_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown task adapter"):
        get_task("missing")


def test_mimo_adapter_uses_explicit_deepscaler_substitute() -> None:
    task = get_task("mimo_math_deepscaler")
    assert {"aime_2024", "aime_2025", "math_500"} <= set(
        task.evaluation_datasets
    )
    assert task.sampling_dataset.repo == "agentica-org/DeepScaleR-Preview-Dataset"
    assert task.normalize_sampling_row(
        {"problem": "Find x.", "answer": "24", "id": "m24"}
    ) == {"problem": "Find x.", "answer": "24", "question_id": "m24"}


def test_olmo_adapter_uses_official_dolci_math_rows() -> None:
    task = get_task("olmo_math")
    sources = task.get_sampling_datasets()
    assert len(sources) == 3
    assert {source.repo for source in sources} == {"allenai/Dolci-Instruct-RL"}
    assert task.include_sampling_row({"dataset": ["math"]})
    assert task.include_sampling_row({"ability": "math"})
    assert not task.include_sampling_row({"dataset": ["code"]})
    assert task.normalize_sampling_row(
        {
            "prompt": "user: Find x.",
            "ground_truth": ["24"],
            "id": "olmo-24",
        }
    ) == {"problem": "Find x.", "answer": "24", "question_id": "olmo-24"}
    assert task.normalize_sampling_row(
        {
            "source_prompt": [{"role": "user", "content": "Compute y."}],
            "reward_model": {"ground_truth": "7"},
            "extra_info": {"index": "dolci-7"},
            "ability": "math",
        }
    ) == {"problem": "Compute y.", "answer": "7", "question_id": "dolci-7"}


@pytest.mark.parametrize(
    ("task_name", "expected_repos"),
    [
        
        ("deepscaler_math", ("agentica-org/DeepScaleR-Preview-Dataset",)),
        ("eurus_math", ("PRIME-RL/Eurus-2-RL-Data",)),
        ("intellect_math", ("PRIME-RL/Eurus-2-RL-Data",)),
        ("klear_math", ("Kwai-Klear/KlearReasoner-MathSub-30K",)),
        ("mimo_math_deepscaler", ("agentica-org/DeepScaleR-Preview-Dataset",)),
        
        (
            "olmo_math",
            (
                "allenai/Dolci-Instruct-RL",
                "allenai/Dolci-Instruct-RL",
                "allenai/Dolci-Instruct-RL",
            ),
        ),
        ("dolci_think_math", ("allenai/Dolci-Think-RL-7B",) * 9),
        ("gpt_oss_swallow_math", ("allenai/Dolci-Think-RL-7B",) * 9),
        ("nemotron_cascade_math", ("nvidia/Nemotron-Cascade-RL-Math",)),
        
        
        ("poro_math", ("Kwai-Klear/KlearReasoner-MathSub-30K",)),
        
        
        
        
    ],
)
def test_each_adapter_resolves_its_own_sampling_artifacts(
    task_name: str, expected_repos: tuple[str, ...]
) -> None:
    assert tuple(
        source.repo for source in get_task(task_name).get_sampling_datasets()
    ) == expected_repos


@pytest.mark.parametrize(
    ("completion", "answer", "correct"),
    [
        ("reasoning</think>Therefore, \\boxed{24}.", "24", True),
        ("###Response The answer is \\boxed{\\frac{1}{2}}.", "0.5", True),
        ("reasoning</think>First \\boxed{1}, then \\boxed{2}.", "2", True),
        ("reasoning</think>Therefore, \\boxed{48/2}.", "24", False),
        ("reasoning</think>Therefore, \\boxed{\\frac{2}{4}}.", "\\frac{1}{2}", False),
        ("reasoning</think>Therefore, \\boxed{(1,2)}.", "(1,2)", True),
        ("reasoning</think>Therefore, \\boxed{p-q}.", "p - q", True),
        ("reasoning</think>Therefore, \\boxed{142}.", "142", True),
        ("reasoning</think>Therefore, \\boxed{25}.", "24", False),
        ("Therefore, \\boxed{24}.", "24", False),
        ("reasoning</think>Therefore, 24.", "24", False),
    ],
)
def test_deepscaler_grader_is_local_and_format_aware(
    completion: str, answer: str, correct: bool
) -> None:
    assert get_task("deepscaler_math").grade(completion, answer) is correct


@pytest.mark.parametrize(
    ("completion", "answer", "correct"),
    [
        ("Therefore, \\boxed{24}.", "24", True),
        ("reasoning</think>Therefore, \\boxed{24}.", "24", True),
        ("###Response Therefore, \\boxed{\\frac{1}{2}}.", "0.5", True),
        ("First \\boxed{1}, then \\boxed{2}.", "2", True),
        ("Therefore, 24.", "24", False),
        ("Therefore, \\boxed{25}.", "24", False),
    ],
)
def test_olmo_grader_accepts_boxed_answers_without_think_tags(
    completion: str, answer: str, correct: bool
) -> None:
    task = get_task("olmo_math")
    assert isinstance(task, TaskAdapter)
    assert task.grade(completion, answer) is correct


def test_dolci_think_adapter_normalizes_released_math_schema() -> None:
    task = get_task("dolci_think_math")
    assert len(task.get_sampling_datasets()) == 9
    assert task.include_sampling_row({"dataset": ["math"]})
    assert not task.include_sampling_row({"dataset": ["code"]})
    assert task.normalize_sampling_row(
        {
            "prompt": "user: Find x.",
            "ground_truth": ["24"],
            "id": "dolci-think-24",
        }
    ) == {
        "problem": "Find x.",
        "answer": "24",
        "question_id": "dolci-think-24",
    }


def test_gpt_oss_harmony_parser_grades_only_the_final_channel() -> None:
    task = get_task("gpt_oss_swallow_math")
    completion = (
        "<|channel|>analysis<|message|>work with \\boxed{1}<|end|>"
        "<|start|>assistant<|channel|>final<|message|>Therefore \\boxed{24}."
        "<|return|>"
    )
    parsed = task.parse_completion(completion)
    assert parsed.reasoning_text == "work with \\boxed{1}"
    assert parsed.final_text == "Therefore \\boxed{24}."
    assert parsed.extracted_answer == "24"
    assert task.grade(completion, "24")

    wrong_final = completion.replace("\\boxed{24}", "\\boxed{25}")
    assert not task.grade(wrong_final, "24")


def test_official_evaluation_grader_is_broader_than_sampling_reward() -> None:
    task = get_task("gpt_oss_swallow_math")
    row = task.normalize_evaluation_row(
        {
            "problem": "What is four plus six?",
            "solution": "Adding gives \\boxed{10}.",
            "answer": "10",
            "unique_id": "math-10",
        },
        "math_500",
    )
    completion = (
        "<|channel|>analysis<|message|>work<|end|>"
        "<|start|>assistant<|channel|>final<|message|>Therefore, 4+6=10."
        "<|return|>"
    )

    # Teacher-trajectory filtering preserves the strict boxed reward contract.
    assert not task.grade(completion, "10")
    # Benchmark evaluation follows Swallow's expression/LaTeX verifier.
    assert task.grade_evaluation(completion, row)
    assert task.evaluation_grader(row).startswith("swallow_lighteval_v202510")


def test_task_completion_parsers_preserve_reasoning_final_and_extracted_answer() -> None:
    strict = get_task("deepscaler_math").parse_completion(
        "work \\boxed{1}</think>final \\boxed{2}."
    )
    assert strict.reasoning_text == "work \\boxed{1}"
    assert strict.final_text == "final \\boxed{2}."
    assert strict.extracted_answer == "2"
    assert strict.extractor == "deepscaler_last_boxed_after_think"

    missing_boundary = get_task("deepscaler_math").parse_completion(
        "final \\boxed{2}."
    )
    assert missing_boundary.extracted_answer is None
    assert missing_boundary.extractor == "deepscaler_missing_response_boundary"

    tagged = get_task("olmo_math").parse_completion(
        "reasoning</think><final>answer \\boxed{3}</final>trailer"
    )
    assert tagged.reasoning_text == "reasoning"
    assert tagged.final_text == "answer \\boxed{3}"
    assert tagged.extracted_answer == "3"
    assert tagged.extractor == "last_boxed_after_think_final_tag"


