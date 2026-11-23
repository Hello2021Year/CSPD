"""Post-pool amendments must not invalidate the 64000-request sampling identity."""
import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from gptoss_ablation import sha
from gptoss_ablation_contract import (checkpoint_steps, evaluation_root, expected_counts,
                                     training_identity, validate_summary, validate_training_config)

OLD = ROOT / 'configs/math/gpt-oss-swallow-20b-sampling.yaml'
NEW = ROOT / 'configs/math/gpt-oss-swallow-20b.yaml'


def test_amendment_only_changes_permitted_fields():
    validate_training_config(NEW, OLD)
    old, new = [yaml.safe_load(p.read_text()) for p in (OLD, NEW)]
    assert old['sampling'] == new['sampling']
    assert old['distillation']['max_steps'] == 226
    assert new['distillation']['max_steps'] == 200
    assert checkpoint_steps(NEW) == list(range(20, 201, 20))
    assert checkpoint_steps(OLD) == [*range(20, 221, 20), 226]
    assert expected_counts(NEW) == {'aime_2024': (30, 480), 'aime_2025': (30, 480), 'math_500': (500, 1000)}
    assert sum(n for _, n in expected_counts(NEW).values()) == 1960


def test_old_and_new_evaluations_cannot_share_cache_or_outputs(tmp_path):
    assert evaluation_root(tmp_path, OLD) != evaluation_root(tmp_path, NEW)
    source = (ROOT / 'scripts/gptoss_ablation_eval.py').read_text()
    assert "cache=protocol_root/'eval-raw'/label" in source
    assert "output=protocol_root/'baselines'" in source
    assert "output=protocol_root/a.arm/'evaluate_student_soft_prompt'" in source


def summary(config):
    counts = expected_counts(config)
    return {'questions': 560, 'samples': sum(n for _, n in counts.values()),
            'benchmarks': {k: {'questions': q, 'samples': n} for k, (q, n) in counts.items()}}


def test_summary_rejects_old_denominator_and_partial_or_swapped_counts():
    assert validate_summary(summary(NEW), NEW)['samples'] == 1960
    with pytest.raises(ValueError):
        validate_summary(summary(OLD), NEW)
    bad = summary(NEW)
    bad['benchmarks']['aime_2024']['samples'] -= 1
    bad['benchmarks']['aime_2025']['samples'] += 1
    with pytest.raises(ValueError):
        validate_summary(bad, NEW)
    bad = summary(NEW)
    bad['samples'] -= 1
    with pytest.raises(ValueError):
        validate_summary(bad, NEW)


@pytest.mark.parametrize('change', ['sampling', 'precision_recipe', 'lr', 'checkpoint_interval', 'evaluation_temperature'])
def test_recipe_changes_fail_closed(tmp_path, change):
    c = yaml.safe_load(NEW.read_text())
    if change == 'sampling': c['sampling']['generation']['temperature'] = 0.7
    if change == 'precision_recipe': c['models']['teacher'] = 'different/model'
    if change == 'lr': c['distillation']['learning_rate'] = 0.02
    if change == 'checkpoint_interval': c['distillation']['checkpoint_every_updates'] = 10
    if change == 'evaluation_temperature': c['evaluation']['arms']['teacher']['generation']['temperature'] = 1.0
    p = tmp_path / 'bad.yaml'; p.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError): validate_training_config(p, OLD)


def test_training_identity_links_frozen_selection_and_new_config(tmp_path):
    d = tmp_path / 'datasets/correct'; d.mkdir(parents=True)
    (d / 'config.yaml').write_bytes(OLD.read_bytes())
    (d / 'teacher_trajectories.jsonl').write_text('{}\n')
    pool = tmp_path / 'pool'; pool.mkdir()
    (pool / 'manifest.json').write_text(json.dumps({'config_sha256': sha(OLD)}))
    identity = training_identity(tmp_path, 'correct', NEW)
    assert identity['sampling_config_sha256'] == sha(OLD)
    assert identity['config_sha256'] == sha(NEW)
    assert identity['total_updates'] == 200
    (pool / 'manifest.json').write_text(json.dumps({'config_sha256': 'changed'}))
    with pytest.raises(ValueError): training_identity(tmp_path, 'correct', NEW)


def test_training_validates_identity_before_completed_state_shortcut():
    source = (ROOT / 'scripts/gptoss_ablation_train.py').read_text()
    assert source.index('identity=training_identity') < source.index("torch.load(state")
    assert "total=identity['total_updates']" in source
    assert 'update<total' in source
    assert 'total_updates=226' not in source


def test_curve_uses_config_schedule_and_namespace():
    source = (ROOT / 'scripts/gptoss_ablation_curve.py').read_text()
    assert 'r=evaluation_root(a.root,a.config)' in source
    assert 'steps=checkpoint_steps(a.config)' in source
    assert 'len(x)==len(steps)' in source
    assert 'validate_summary' in source


def test_config_loader_resolves_all_arms_to_new_counts():
    from soft_prompt_distillation.config import load_config
    c = load_config(NEW)
    assert c.distillation.max_steps == 200
    for arm in ('student', 'teacher', 'student_soft_prompt'):
        assert c.evaluation.arms[arm].generation.samples_per_prompt == 16
    assert c.evaluation.benchmarks['math_500'].samples_per_prompt == 2


def test_curve_complete_at_200_and_rejects_legacy_anchor(tmp_path, monkeypatch):
    pytest.importorskip('matplotlib')
    import gptoss_ablation_curve as curve
    r = evaluation_root(tmp_path, NEW)
    def write(path, accuracy, config=NEW):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(summary(config), accuracy=accuracy)))
    write(r / 'baselines/evaluate_student/summary.json', .2)
    write(r / 'baselines/evaluate_teacher/summary.json', .4)
    for arm in ('correct', 'all'):
        for step in checkpoint_steps(NEW):
            write(r / arm / 'evaluate_student_soft_prompt' / f'update_{step:08d}' / 'summary.json', .3)
    monkeypatch.setattr(sys, 'argv', ['curve', str(tmp_path), '--config', str(NEW)])
    curve.main()
    report = json.loads((r / 'curves/recovery.json').read_text())
    assert report['complete'] is True
    assert report['expected_samples'] == 1960
    assert len(report['curves']['correct']) == 10
    assert report['curves']['all'][-1]['update'] == 200
    assert not (tmp_path / 'curves').exists()
    write(r / 'baselines/evaluate_student/summary.json', .2, OLD)
    with pytest.raises(ValueError): curve.main()
