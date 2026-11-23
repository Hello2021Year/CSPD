"""Config-bound post-sampling schedule; never edits the frozen sampling config."""
import copy
import json
from pathlib import Path

import yaml

from gptoss_ablation import sha

BENCHMARK_QUESTIONS = {'aime_2024': 30, 'aime_2025': 30, 'math_500': 500}


def checkpoint_steps(config):
    d = yaml.safe_load(Path(config).read_text())['distillation']
    total, every = int(d['max_steps']), int(d['checkpoint_every_updates'])
    if total <= 0 or every <= 0:
        raise ValueError('positive training total and checkpoint interval required')
    return sorted(set([*range(every, total + 1, every), total]))


def expected_counts(config):
    c = yaml.safe_load(Path(config).read_text())
    expected = {}
    for name, questions in BENCHMARK_QUESTIONS.items():
        if c['data']['evaluation'][name]['num_prompts'] != 'all':
            raise ValueError('full benchmark membership required')
        counts = set()
        for arm in ('student', 'teacher', 'student_soft_prompt'):
            generation = c['evaluation'].get('benchmarks', {}).get(name, {}).get(
                'generation', c['evaluation']['arms'][arm]['generation'])
            counts.add(int(generation['samples_per_prompt']))
        if len(counts) != 1 or min(counts) <= 0:
            raise ValueError('matched positive repetition counts required for every arm')
        expected[name] = (questions, questions * counts.pop())
    return expected


def evaluation_root(root, config):
    # Both raw cache and scored outputs are namespaced, including the anchors.
    return Path(root) / 'evaluation-protocols' / sha(config)


def validate_summary(summary, config):
    expected = expected_counts(config)
    if summary['questions'] != sum(q for q, _ in expected.values()) or summary['samples'] != sum(n for _, n in expected.values()):
        raise ValueError('evaluation point has the wrong total denominator')
    if set(summary['benchmarks']) != set(expected):
        raise ValueError('evaluation benchmark membership changed')
    for name, (questions, samples) in expected.items():
        b = summary['benchmarks'][name]
        if b['questions'] != questions or b['samples'] != samples:
            raise ValueError(f'evaluation denominator changed: {name}')
    return summary


def validate_training_config(config, sampling_config):
    """Permit only the authorized total/evaluation revision, not pool drift."""
    new = yaml.safe_load(Path(config).read_text())
    old = yaml.safe_load(Path(sampling_config).read_text())
    if int(new['distillation']['max_steps']) != 200:
        raise ValueError('current training contract requires 200 updates')
    if expected_counts(config) != {'aime_2024': (30, 480), 'aime_2025': (30, 480), 'math_500': (500, 1000)}:
        raise ValueError('current evaluation contract requires AIME16/MATH2')
    normalized = copy.deepcopy(new)
    normalized['name'] = old['name']
    normalized['distillation']['max_steps'] = old['distillation']['max_steps']
    for arm in ('student', 'teacher', 'student_soft_prompt'):
        for field in ('samples_per_prompt', 'provenance'):
            normalized['evaluation']['arms'][arm]['generation'][field] = old['evaluation']['arms'][arm]['generation'][field]
    for field in ('samples_per_prompt', 'provenance'):
        normalized['evaluation']['benchmarks']['math_500']['generation'][field] = old['evaluation']['benchmarks']['math_500']['generation'][field]
    if normalized != old:
        raise ValueError('post-sampling config changes more than schedule/evaluation repetitions')


def training_identity(root, arm, config):
    root = Path(root)
    sampling_config = root / 'datasets' / arm / 'config.yaml'
    validate_training_config(config, sampling_config)
    pool = json.loads((root / 'pool' / 'manifest.json').read_text())
    if pool['config_sha256'] != sha(sampling_config):
        raise ValueError('selected dataset config no longer matches the frozen pool')
    return {
        'config_sha256': sha(config),
        'sampling_config_sha256': sha(sampling_config),
        'dataset_sha256': sha(root / 'datasets' / arm / 'teacher_trajectories.jsonl'),
        'total_updates': checkpoint_steps(config)[-1],
    }
