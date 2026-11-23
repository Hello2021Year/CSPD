import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('ablation',Path(__file__).parents[1]/'scripts/gptoss_ablation.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_equal_n_cap_and_repeatable():
    rows=[dict(slot=q*4+j,question_id=str(q),eligible=True,correct=(j==3 or q==0)) for q in range(8) for j in range(4)]
    c,a=m.select_ids(rows)
    assert len(c)==len(a)==9
    assert (c,a)==m.select_ids(rows)
    assert all(rows[i]['correct'] for i in c)
    assert any(not rows[i]['correct'] for i in a)
    assert all(sum(rows[i]['question_id']==str(q) for i in a)<=2 for q in range(8))

def test_all_blind_to_correctness_when_n_fixed():
    rows=[dict(slot=q*4+j,question_id=str(q),eligible=True,correct=j==0) for q in range(8) for j in range(4)]
    _,a=m.select_ids(rows)
    for row in rows:row['correct']=row['slot']%4==2
    _,b=m.select_ids(rows)
    assert a==b

def test_validity_is_common():
    x=dict(prompt_ids=[1],response_ids=[2],finish_reason='eos')
    assert m.valid(x)
    assert not m.valid(dict(x,finish_reason='length'))
    assert not m.valid(dict(x,error='timeout'))
    assert not m.valid(dict(x,response_ids=[]))
    assert not m.valid(dict(x,prompt_ids=[1]*39744))

def test_continuous_pool_limit_and_bounded_override():
    args=['sample','--config','config.yaml','--root','run','--cache','cache']
    assert m.build_parser().parse_args(args).limit == 64000
    assert m.build_parser().parse_args(args+['--dp-size','8']).dp_size == 8
    import pytest
    with pytest.raises(SystemExit):m.build_parser().parse_args(args+['--concurrency','128'])
    assert m.build_parser().parse_args(args+['--limit','64']).limit == 64
    assert m.build_parser().parse_args(args+['--limit','64000']).limit == 64000
    import pytest
    for value in ['0','-1','64001']:
        with pytest.raises(SystemExit):m.build_parser().parse_args(args+['--limit',value])


def test_no_empty_correct():
    import pytest
    with pytest.raises(RuntimeError):m.select_ids([dict(slot=0,question_id='q',eligible=True,correct=False)])


def test_retry_archives_incomplete_and_preserves_identity(tmp_path):
    req={'slot':7,'question_id':'q','sampling_params':{'seed':1007}}
    m.atomic(tmp_path/'request.json',req)
    m.atomic(tmp_path/'error.json',{'error':'allocation expired'})
    assert m.recover_incomplete(tmp_path,req)
    import json
    assert json.loads((tmp_path/'attempts/0001/request.json').read_text())==req
    assert '968195fd' in (tmp_path/'attempts/0001/retry.json').read_text()
    assert not (tmp_path/'request.json').exists()
    assert not m.recover_incomplete(tmp_path,req)
    m.atomic(tmp_path/'request.json',req)
    assert m.recover_incomplete(tmp_path,req)
    assert (tmp_path/'attempts/0002/retry.json').exists()


def test_retry_never_replaces_raw_or_completed(tmp_path):
    m.atomic(tmp_path/'request.json',{'seed':4})
    m.atomic(tmp_path/'raw.json',{'output_ids':[1], 'finish_reason':'length'})
    before=m.sha(tmp_path/'raw.json')
    assert not m.recover_incomplete(tmp_path,{'seed':4})
    assert m.sha(tmp_path/'raw.json')==before
    assert not (tmp_path/'attempts').exists()


def test_retry_changed_identity_fails_closed(tmp_path):
    import pytest
    m.atomic(tmp_path/'request.json',{'seed':4})
    with pytest.raises(RuntimeError):m.recover_incomplete(tmp_path,{'seed':5})
    assert (tmp_path/'request.json').exists()
    assert not (tmp_path/'attempts').exists()


def test_eval_uses_all_requests_without_rank_routing():
    source=(Path(__file__).parents[1]/'scripts/gptoss_ablation_eval.py').read_text()
    assert "sglang_batch_size='all'" in source
    assert "kwargs['data_parallel_rank']" not in source
    assert "--batch" not in source
    assert 'recover_incomplete(d, request)' in source
