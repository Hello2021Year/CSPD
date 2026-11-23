"""Sample 4 teacher responses per training issue with SGLang (temperature 1, 32k tokens).

Env: WORK, TEACHER_MODEL (local Nemotron-Cascade-8B), SHARD (0-3).
"""
import os,json,pathlib,gzip,hashlib,urllib.request,socket,subprocess,sys,time,signal,shutil
from concurrent.futures import ThreadPoolExecutor,as_completed
from transformers import AutoTokenizer
w=pathlib.Path(os.environ['WORK']);a=w/'artifacts';source=a/'training-prompts-v1';dest=a/'training-rollouts-v1';dest.mkdir(exist_ok=True);shard=int(os.environ['SHARD']);nshards=4
model=os.environ['TEACHER_MODEL'];revision='276a779fe0c1fe2d63a68fd573fec7b4b0a1907e'
manifest=json.loads((source/'manifest.json').read_text());assert manifest['selected_issues']==8192
# Read exact source inventory once per shard; references are not opened by this program.
rows=[json.loads(s) for s in (source/'prompts.jsonl').open()];assert len(rows)==8192 and len({x['id'] for x in rows})==8192
assert hashlib.sha256((source/'prompts.jsonl').read_bytes()).hexdigest()==manifest['prompts_sha256']
rows=rows[shard::nshards]
limit=int(os.environ.get('MAX_QUESTIONS','0'))
if limit:rows=rows[:limit] # bounded initial real validation; same per-question seeds/requests on resume
assert 0<=shard<4 and limit>=0
tok=AutoTokenizer.from_pretrained(model)
cfg={'max_position_embeddings':65536,'rope_scaling':{'rope_type':'yarn','factor':2.0,'original_max_position_embeddings':32768}}
policy={'teacher_revision':revision,'source_revision':manifest['source_revision'],'source_prompt_sha256':manifest['prompts_sha256'],'shards':4,'attempts':4,'temperature':1.0,'top_p':1.0,'max_new_tokens':32768,'input_cap':32767,'prompt_policy':'unaltered source prompt wrapped in original teacher chat template, enable_thinking=True; no gold reference','correctness':'unverified; correct=null, no format-as-correct conversion','seeds':'1000000 + 4*global_manifest_index + attempt','storage':'gzip raw engine responses; atomic files; no completed attempt retry'}
pp=dest/f'policy-shard{shard}.json'
if pp.exists():assert json.loads(pp.read_text())==policy
else:pp.write_text(json.dumps(policy,indent=2))
with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
url=f'http://127.0.0.1:{port}'
cmd=[sys.executable,'-m','sglang.launch_server','--model-path',model,'--host','127.0.0.1','--port',str(port),'--tp-size','1','--dtype','bfloat16','--context-length','65536','--mem-fraction-static','0.60','--attention-backend','triton','--disable-cuda-graph','--json-model-override-args',json.dumps(cfg)]
def atom_json(p,obj):
 tmp=p.with_suffix(p.suffix+'.tmp')
 with gzip.open(tmp,'wt',encoding='utf-8',compresslevel=6) as f:json.dump(obj,f,ensure_ascii=False)
 tmp.replace(p)
def run(pair):
 localidx,x=pair;globalidx=shard+localidx*nshards;root=dest/x['id'];root.mkdir(exist_ok=True)
 rendered=tok.apply_chat_template([{'role':'user','content':x['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=True);ids=tok.encode(rendered,add_special_tokens=False)
 req={'id':x['id'],'global_index':globalidx,'prompt_sha256':hashlib.sha256(x['prompt'].encode()).hexdigest(),'rendered_sha256':hashlib.sha256(rendered.encode()).hexdigest(),'input_ids_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest(),'input_tokens':len(ids),'source_row_sha256':x['row_sha256'],'teacher_revision':revision,'seeds':[1000000+4*globalidx+i for i in range(4)]}
 rp=root/'request.json'
 if rp.exists():assert json.loads(rp.read_text())==req,'changed request '+x['id']
 else:rp.write_text(json.dumps(req,indent=2))
 if len(ids)>32767:
  (root/'overlength.json').write_text(json.dumps({'input_tokens':len(ids),'policy':'whole input excluded, not truncated','generated':False}));return {'id':x['id'],'status':'input_overlength'}
 missing=[i for i in range(4) if not (root/f'answer-{i}.json.gz').exists()]
 if not missing:return {'id':x['id'],'status':'retained_completed'}
 if shutil.disk_usage(dest).free<1024**3:raise RuntimeError('durable storage reserve below1GiB; do not generate without storage')
 params=[{'max_new_tokens':32768,'temperature':1.0,'top_p':1.0,'sampling_seed':req['seeds'][i],'stop_token_ids':[151645]} for i in missing]
 body={'input_ids':[ids for _ in missing],'sampling_params':params}
 request=urllib.request.Request(url+'/generate',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(request,timeout=14400) as response:values=json.load(response)
 assert isinstance(values,list) and len(values)==len(missing)
 for i,v in zip(missing,values):
  assert 'text' in v and 'finish_reason' in v.get('meta_info',{})
  atom_json(root/f'answer-{i}.json.gz',{'id':x['id'],'attempt':i,'seed':req['seeds'][i],'teacher_revision':revision,'correct':None,'grader':None,'raw_engine_response':v})
 (root/'complete.json').write_text(json.dumps({'attempts':4,'correct':None,'job':os.environ.get('SLURM_JOB_ID', 'local')}));return {'id':x['id'],'status':'sampled'}
log=dest/f'shard{shard}-{os.environ.get("SLURM_JOB_ID", "local")}-server.log'
with log.open('w') as f:
 server=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 try:
  for _ in range(240):
   if server.poll() is not None:raise RuntimeError('server died '+str(log))
   try:
    with urllib.request.urlopen(url+'/get_server_info',timeout=2) as response:info=json.load(response)
    assert info.get('model_path',info.get('server_args',{}).get('model_path'))==model;break
   except (OSError,urllib.error.URLError):time.sleep(5)
  else:raise TimeoutError('server readiness')
  (dest/f'shard{shard}-{os.environ.get("SLURM_JOB_ID", "local")}-server-info.json').write_text(json.dumps(info,indent=2))
  # Admit all four attempts/question, bounded32 questions (128 queued requests) per resident engine.
  with ThreadPoolExecutor(max_workers=32) as pool:
   futures=[pool.submit(run,pair) for pair in enumerate(rows)]
   for future in as_completed(futures):print(json.dumps(future.result()),flush=True)
 finally:
  try:os.killpg(server.pid,signal.SIGTERM)
  except ProcessLookupError:pass
  try:server.wait(timeout=45)
  except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait()
print('SHARD_COMPLETE',shard,flush=True)
