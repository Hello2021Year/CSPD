"""Select 8,192 Nemotron-Cascade-RL-SWE training issues (SWE-bench Verified excluded).

Env: WORK (output root; expects artifacts/benchmark500-v2/plan.json with the 500 evaluation ids).
"""
import urllib.request,json,os,pathlib,hashlib,heapq,collections
w=pathlib.Path(os.environ['WORK']);d=w/'artifacts/nemotron-swe/training-prompts-v1';d.mkdir(exist_ok=True)
rev='2c3c6c7cfcc1424160fe2227ad982c5acca34bc2';url=f'https://huggingface.co/datasets/nvidia/Nemotron-Cascade-RL-SWE/resolve/{rev}/train_16k.jsonl';seen=set();heap=[];total=0;digest=hashlib.sha256();excluded=0
bench=set(json.loads((w/'artifacts/nemotron-swe/benchmark500-v2/plan.json').read_text())['ids'])
with urllib.request.urlopen(url,timeout=180) as f:
 for line in f:
  digest.update(line);total+=1;x=json.loads(line);iid=x['instance_id']
  if iid in bench:excluded+=1;continue
  if iid in seen:continue
  seen.add(iid);rank=int.from_bytes(hashlib.sha256(('nemotron-swe-training-v1:42:'+iid).encode()).digest(),'big');entry={'id':iid,'source':x['source'],'prompt':x['prompt'],'golden_patch':x['golden_patch'],'original_prompt':x['original_prompt'],'source_line':total,'row_sha256':hashlib.sha256(line).hexdigest()}
  if len(heap)<8192:heapq.heappush(heap,(-rank,iid,entry))
  elif rank < -heap[0][0]:heapq.heapreplace(heap,(-rank,iid,entry))
  if total%2000==0:print('scanned',total,'unique',len(seen),flush=True)
assert len(heap)==8192
rows=[x[2] for x in sorted(heap,key=lambda x:-x[0])]
with (d/'prompts.jsonl').open('w') as p,(d/'reference-only.jsonl').open('w') as g:
 for i,x in enumerate(rows):
  p.write(json.dumps({k:v for k,v in x.items() if k not in ['golden_patch','original_prompt']})+'\n');g.write(json.dumps({k:x[k] for k in ['id','golden_patch','original_prompt']})+'\n')
manifest={'source_revision':rev,'source_file':'train_16k.jsonl','source_sha256':digest.hexdigest(),'source_rows':total,'unique_nonbenchmark_ids':len(seen),'exact_benchmark_rows_excluded':excluded,'selected_issues':8192,'selection':'lowest SHA256(nemotron-swe-training-v1:42:instance_id), first source row per ID; no outcome selection','source_counts':dict(collections.Counter(x['source'] for x in rows)),'prompts_sha256':hashlib.sha256((d/'prompts.jsonl').read_bytes()).hexdigest(),'references_sha256':hashlib.sha256((d/'reference-only.jsonl').read_bytes()).hexdigest(),'note':'Source gold-informed localization retained as documented training source policy; reference patch/original_prompt segregated from generation inputs. No claim of verified teacher output; sampling not yet performed.'}
(d/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
