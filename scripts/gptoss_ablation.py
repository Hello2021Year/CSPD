"""GPT-OSS equal-N correct/all experiment. Explicit stages; audited restart of incomplete requests.
Run only inside an interactive compute allocation. See the experiment runbook.
"""
from __future__ import annotations
import argparse, asyncio, dataclasses, hashlib, json, os, random, shutil
from pathlib import Path

DATA_REV = '0fb6466d31ef3a9dd16985ef635e6429e05a6491'


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


RETRY_AUTHORITY = 'human DM968195fd 2026-09-17: rerun fault-interrupted incomplete requests'


def recover_incomplete(directory, expected=None):
    """Archive a failed attempt on process restart, never a completed response.

    Copy + durable manifest before unlink makes interruption of archival safe.
    Caller must use the same logical request identity and must not run two writers.
    Raw responses (including length termination) are immutable and reused.
    """
    d = Path(directory)
    if (d/'raw.json').exists() or (d/'response.json').exists(): return False
    request = d/'request.json'
    if not request.exists(): return False
    previous = json.loads(request.read_text())
    if expected is not None and previous != expected:
        raise RuntimeError(f'retry request identity changed: {d}')
    attempts = d/'attempts'; attempts.mkdir(exist_ok=True)
    n = 1
    while (attempts/f'{n:04d}').exists(): n += 1
    archive = attempts/f'{n:04d}'; archive.mkdir()
    hashes = {}
    for name in ('request.json','error.json'):
        source = d/name
        if source.exists():
            atomic(archive/name, json.loads(source.read_text()))
            hashes[name] = sha(archive/name)
    atomic(archive/'retry.json', {'authority':RETRY_AUTHORITY,
           'reason':'previous invocation incomplete; no durable raw response',
           'attempt':n,'files_sha256':hashes,'logical_request':previous})
    # Keep the request in place until all other files are removed. Re-entry
    # after a crash may archive twice but never loses the original evidence.
    (d/'error.json').unlink(missing_ok=True)
    request.unlink()
    return True


def engine_receipt(runtime):
    engine=runtime._engine; tm=engine.tokenizer_manager; mc=tm.model_config; args=engine.server_args
    receipt={'sglang':runtime.version,'model_path':str(args.model_path),
             'model_config_dtype':str(mc.dtype),'model_config_quantization':mc.quantization,
             'resolved_context_length':int(mc.context_len),'reserved_tokens':runtime.reserved_tokens,
             'server_dtype':str(args.dtype),'server_quantization':args.quantization,
             'server_context_length':args.context_length,'attention_backend':args.attention_backend,
             'tp_size':args.tp_size,'dp_size':args.dp_size}
    if receipt['model_config_dtype']!='torch.bfloat16' or receipt['model_config_quantization'] is not None:
        raise RuntimeError('unexpected resolved model precision: '+repr(receipt))
    return receipt


def valid(row):
    return (not row.get('error') and row.get('finish_reason') in ('eos', 'stop')
            and bool(row.get('response_ids')) and bool(row.get('prompt_ids'))
            and len(row['prompt_ids']) + len(row['response_ids']) + 256 <= 40000)


def select_ids(rows, seed=42):
    """Metadata-only deterministic selection. All selection never reads correct."""
    groups = {}
    for x in rows:
        if x['eligible']:
            groups.setdefault(x['question_id'], []).append(x)
    correct = []
    for q in sorted(groups):
        correct += [x['slot'] for x in sorted(groups[q], key=lambda x:x['slot']) if x['correct'] is True][:2]
    rng = random.Random(seed); questions = sorted(groups); rng.shuffle(questions)
    all_ids = []
    for q in questions:
        candidates = sorted(groups[q], key=lambda x:x['slot']); rng.shuffle(candidates)
        all_ids.extend(x['slot'] for x in candidates[:2])
    if not correct or len(all_ids) < len(correct):
        raise RuntimeError('empty correct set or insufficient eligible all pool')
    return correct, all_ids[:len(correct)]


def context(a):
    from soft_prompt_distillation.common import RunOptions
    from soft_prompt_distillation.config import load_config
    c = load_config(a.config)
    o = RunOptions(config=a.config, output=a.root, cache_dir=a.cache)
    return c,o


def prepare(a):
    from soft_prompt_distillation.common import download_rows, load_template_tokenizer
    from soft_prompt_distillation.sampling import _deduplicate_source_rows
    from soft_prompt_distillation.tasks.new_architecture_math import GptOssSwallowMathTask
    from soft_prompt_distillation.rendering import render_for_execution, resolve_insertion_index
    c,o = context(a)
    # Pin all nine source artifacts without changing the shared task adapter.
    class PinnedTask(GptOssSwallowMathTask):
        sampling_datasets = tuple(dataclasses.replace(s, revision=DATA_REV) for s in GptOssSwallowMathTask.sampling_datasets)
    dest = a.root/'pool'
    if dest.exists(): raise RuntimeError('pool already exists; inspect rather than overwrite')
    rows, data = download_rows(o,c,PinnedTask(),'sampling')
    rows, dedup = _deduplicate_source_rows(rows)
    assert len(rows) == 16000, len(rows)
    tokenizer, identity = load_template_tokenizer(o,c)
    temporary = a.root/'pool.building'; temporary.mkdir(parents=True,exist_ok=False)
    # Freeze exactly the rows and rendered prompts before any generation.
    with (temporary/'questions.jsonl').open('w') as f:
        for i,row in enumerate(rows):
            rendered = render_for_execution(c,execution_model='teacher',row=row,tokenizer=tokenizer)
            item = dict(row,source_row_index=i,prompt_ids=list(rendered.token_ids),insert_at=resolve_insertion_index(c,rendered))
            f.write(json.dumps(item,ensure_ascii=False)+'\n')
        f.flush(); os.fsync(f.fileno())
    shutil.copyfile(a.config,temporary/'config.yaml')
    atomic(temporary/'manifest.json',{'source':identity,'datasets':data,'dedup':dedup,'questions':len(rows),'requests':64000,'questions_sha256':sha(temporary/'questions.jsonl'),'config_sha256':sha(a.config),'data_revision':DATA_REV})
    temporary.rename(dest)
    print('POOL_READY',sha(dest/'questions.jsonl'),flush=True)


def sample(a):
    from soft_prompt_distillation.common import load_template_tokenizer, model_source
    from soft_prompt_distillation.sglang_backend import SGLangRuntime, _sampling_params, _parse_results
    c,o = context(a); pool=a.root/'pool'; m=json.loads((pool/'manifest.json').read_text())
    assert m['config_sha256']==sha(a.config)
    assert m['questions_sha256']==sha(pool/'questions.jsonl')
    rows=[json.loads(l) for l in (pool/'questions.jsonl').open()]
    slots=a.root/'slots'; slots.mkdir(exist_ok=True)
    tokenizer,identity=load_template_tokenizer(o,c)
    if identity['tokenizer_vocab_sha256'] != m['source']['tokenizer_vocab_sha256']: raise RuntimeError('tokenizer changed')
    # Parse a saved raw response on resume, never regenerate it.
    def finalize(slot, row, sample_index, raw):
        result=_parse_results(raw,1,tokenizer.eos_token_id)[0]
        item=dict(row,sample_index=sample_index,slot=row['source_row_index']*4+sample_index,
                  seed=1000+row['source_row_index']*4+sample_index,response_ids=result.token_ids,
                  raw_completion=tokenizer.decode(result.token_ids,skip_special_tokens=False),
                  completion=tokenizer.decode(result.token_ids,skip_special_tokens=True),finish_reason=result.finish_reason)
        atomic(slot/'response.json',item)
    pending=[]
    for row in rows:
        for j in range(4):
            slot=slots/f"{row['source_row_index']*4+j:05d}"
            if (slot/'response.json').exists(): continue
            if (slot/'raw.json').exists(): finalize(slot,row,j,json.loads((slot/'raw.json').read_text()));continue
            pending.append((slot,row,j))
    if not pending: print('SAMPLING_COMPLETE');return
    pending=pending[:a.limit]
    source,_=model_source(o,c,'teacher')
    runtime=SGLangRuntime.start(source,1,a.dp_size,a.cache,'warning')
    import time
    atomic(a.root/'runtime-receipts'/f'{time.time_ns()}.json',engine_receipt(runtime))
    if a.engine_receipt_only:
        runtime.shutdown();print('ENGINE_RECEIPT_ONLY_NO_REQUESTS');return
    version_path=a.root/'sampling-runtime.json'
    version={'sglang':runtime.version,'model':source,'config_sha256':sha(a.config)}
    if version_path.exists() and json.loads(version_path.read_text()) != version:
        runtime.shutdown();raise RuntimeError('sampling runtime identity changed')
    atomic(version_path,version)
    async def run():
        started=time.monotonic();completed=0;output_tokens=0
        if (a.root/'stop').exists():
            print('STOP_BEFORE_SUBMISSION',flush=True);return
        async def one(entry):
            nonlocal completed,output_tokens
            slot,row,j=entry;slot.mkdir(exist_ok=True)
            params=_sampling_params(c.sampling.generation,1000+row['source_row_index']*4+j,tokenizer.eos_token_id,())
            request={'slot':row['source_row_index']*4+j,'question_id':row['question_id'],'input_ids':row['prompt_ids'],'sampling_params':params}
            recover_incomplete(slot, request)
            atomic(slot/'request.json',request)
            try:
                raw=await runtime._engine.async_generate(input_ids=request['input_ids'],sampling_params=params)
                atomic(slot/'raw.json',raw)
                finalize(slot,row,j,raw)
                completed+=1;output_tokens+=len(raw.get('output_ids',[]))
                print('RESPONSE_DURABLE',request['slot'],flush=True)
                if completed%32==0:
                    print('THROUGHPUT',json.dumps({'responses':completed,'output_tokens':output_tokens,'elapsed_seconds':time.monotonic()-started,'output_tokens_per_second':output_tokens/max(time.monotonic()-started,1e-9),'submitted_pool_size':len(pending),'dp_size':a.dp_size}),flush=True)
            except Exception as exc:
                atomic(slot/'error.json',{'type':type(exc).__name__,'error':str(exc)})
                raise
        # No client window, grouping, rank routing or custom scheduling.
        # Every remaining ready request is handed to SGLang's default scheduler.
        # Once submitted, stop cannot retract work safely across allocations;
        # archive incomplete attempts on restart under human DM968195fd.
        atomic(a.root/'submission-receipt.json',{'pending':len(pending),'dp_size':a.dp_size,'policy':'all_ready_default_engine_scheduler','time':time.time(),'cross_allocation_resume':'reuse raw; archive and retry incomplete with same identity under DM968195fd'})
        results=await asyncio.gather(*(one(entry) for entry in pending),return_exceptions=True)
        errors=[x for x in results if isinstance(x,BaseException)]
        if errors:raise RuntimeError(f'{len(errors)} preserved generation errors, no retries: {errors[0]}')
    try: runtime._runner.run(run())
    finally: runtime.shutdown()


def select(a):
    from soft_prompt_distillation.tasks import get_task
    c,o=context(a);task=get_task(c.task);metadata=[]; root=a.root/'datasets'
    if root.exists(): raise RuntimeError('datasets already exist; inspect rather than overwrite')
    for slot in range(64000):
        p=a.root/'slots'/f'{slot:05d}'/'response.json'
        if not p.exists(): raise RuntimeError(f'incomplete candidate pool at {slot}')
        x=json.loads(p.read_text()); assert x['slot']==slot
        gradepath=p.parent/'grade.json'
        if gradepath.exists(): grade=json.loads(gradepath.read_text());assert grade['response_sha256']==sha(p)
        else:
            grade={'correct':bool(task.grade(x['completion'],x['answer'])),'response_sha256':sha(p),'grader':type(task).__name__+'.grade'}
            atomic(gradepath,grade)
        metadata.append({'slot':slot,'question_id':x['question_id'],'eligible':valid(x),'correct':grade['correct'],'response_sha256':grade['response_sha256']})
    correct,all_ids=select_ids(metadata);meta={x['slot']:x for x in metadata}
    tmp=a.root/'datasets.building';tmp.mkdir(exist_ok=False)
    manifest=json.loads((a.root/'pool/manifest.json').read_text())
    for arm,ids in [('correct',correct),('all',all_ids)]:
        d=tmp/arm;d.mkdir();counts={'trajectories':len(ids),'correct':0,'tokens':0};questions=set()
        with (d/'teacher_trajectories.jsonl').open('w') as f:
            for slot in ids:
                p=a.root/'slots'/f'{slot:05d}'/'response.json';assert sha(p)==meta[slot]['response_sha256']
                x=json.loads(p.read_text());x['correct']=meta[slot]['correct'];questions.add(x['question_id'])
                counts['correct']+=int(x['correct']);counts['tokens']+=len(x['prompt_ids'])+len(x['response_ids'])
                f.write(json.dumps(x,ensure_ascii=False)+'\n')
            f.flush();os.fsync(f.fileno())
        counts['questions']=len(questions)
        shutil.copyfile(a.config,d/'config.yaml')
        atomic(d/'selection.json',{'seed':42,'slots':ids,'counts':counts,'candidate_count':64000,'completeness':'eos_or_stop_nonempty_context40000_including_sp256','cap':2})
        atomic(d/'manifest.json',{'source':manifest['source'],'record_counts':counts,'content_sha256':{'teacher_trajectories.jsonl':sha(d/'teacher_trajectories.jsonl'),'selection.json':sha(d/'selection.json')}})
    atomic(tmp/'candidate-audit.json',metadata);tmp.rename(root)
    print('EQUAL_N_READY',len(correct),flush=True)


def request_limit(value):
    value = int(value)
    if not 1 <= value <= 64000:
        raise argparse.ArgumentTypeError('request limit must be between 1 and 64000')
    return value


def build_parser():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','sample','select']);p.add_argument('--config',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--dp-size',type=int,default=1,help='authorized GPU data-parallel size, requests use engine default scheduling');p.add_argument('--limit',type=request_limit,default=64000,help='maximum NEW requests this invocation; default continues the complete fixed pool');p.add_argument('--engine-receipt-only',action='store_true')
    return p


def main():
    a=build_parser().parse_args();assert os.environ.get('SLURM_JOB_ID'),'compute allocation required';assert 1<=a.dp_size<=8
    a.config=a.config.resolve();a.root=a.root.resolve();a.cache=a.cache.resolve();a.root.mkdir(parents=True,exist_ok=True)
    globals()[a.stage](a)
if __name__=='__main__':main()
