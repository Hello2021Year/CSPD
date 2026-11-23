"""Train the 256-vector SWE soft prompt on the Nemotron-Cascade-8B-SFT student.

Response-only SFT on single-turn teacher trajectories, global batch 64, AdamW LR 0.01, 254 updates.
Env: WORK, STUDENT_MODEL, TEACHER_MODEL, TRAIN_OUTPUT. Launch with torchrun.
"""
import os,json,pathlib,hashlib,gzip,random,math,time,dataclasses,sys
import torch,torch.distributed as dist
from transformers import AutoTokenizer,AutoModelForCausalLM
from safetensors.torch import save_file
w=pathlib.Path(os.environ['WORK']);a=w/'artifacts';# requires `pip install -e .` from the repository root
from soft_prompt_distillation.objective import make_response_sft_example
from soft_prompt_distillation.training import SoftPromptModel,initialize_prompt_embeddings,collate_examples,accumulate_response_loss,finish_prompt_update
from types import SimpleNamespace
rank=int(os.environ.get('RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'));local=int(os.environ.get('LOCAL_RANK','0'));torch.cuda.set_device(local)
if world>1:dist.init_process_group('nccl')
torch.manual_seed(42);torch.cuda.manual_seed_all(42)
torch.backends.cuda.enable_flash_sdp(True);torch.backends.cuda.enable_math_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
root=a/'swe-sp-training-v1';out=pathlib.Path(os.environ['TRAIN_OUTPUT']);out.mkdir(parents=True,exist_ok=True);selected_path=root/'selected.json';selected=json.loads(selected_path.read_text());assert len(selected)==8121
student=os.environ['STUDENT_MODEL']
teacher=os.environ['TEACHER_MODEL'];tok=AutoTokenizer.from_pretrained(teacher,local_files_only=True);st=AutoTokenizer.from_pretrained(student,local_files_only=True)
assert tok.get_vocab()==st.get_vocab(),'teacher/student vocab mismatch'
model=AutoModelForCausalLM.from_pretrained(student,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
model.config.use_cache=False
assert float(getattr(model.config,'attention_dropout',0))==0,'resume contract requires no attention dropout'
assert model.config.max_position_embeddings>=32768
# Max actual sequence is31525 includingSP, so no extrapolation is required for these data.
init=initialize_prompt_embeddings(model,tok,SimpleNamespace(method='sampled_vocabulary',seed=42),256)
backend=os.environ.get('CE_BACKEND','torch')
wrapper=SoftPromptModel(model,init,activation_checkpointing=True,cross_entropy_backend=backend,loss_normalization='sequence');optimizer=torch.optim.AdamW([wrapper.prompt_parameter],lr=.01,weight_decay=0)
def sha(b):return hashlib.sha256(b).hexdigest()
contract={'student_revision':'9b379bd8f8ecb1feb19825efefe319bd30d8a223','student_subfolder':'Nemotron-Cascade-8B-SFT','teacher_revision':'276a779fe0c1fe2d63a68fd573fec7b4b0a1907e','selected_sha256':sha(selected_path.read_bytes()),'runner_sha256':sha(pathlib.Path(__file__).read_bytes()),'samples':len(selected),'prompt_length':256,'insert':'at user content start in exact teacher generation template; marker not fed as text','prompt_init':'seed42 sampled non-special vocabulary rows','loss':'mean per-sequence full response including thinking and EOS; context and prompt masked','ce_backend':backend,'attention':'torch SDPA flash-only','activation_checkpointing':True,'optimizer':'AdamW','lr':.01,'schedule':'constant','weight_decay':0,'max_grad_norm':10,'micro_batch':1,'effective_batch':64,'epochs':2,'keep_epoch_tail':True,'save_every':20,'eval_every':40,'save_eval_final':True,'rng_contract':'zero dropout; deterministic epochshuffle; optimizer/checkpoint resume','world_size':world,'precision':'BF16 frozen backbone, FP32 prompt','total_updates':254}
cp=out/'contract.json'
if rank==0:
 if cp.exists():assert json.loads(cp.read_text())==contract
 else:cp.write_text(json.dumps(contract,indent=2))
if world>1:dist.barrier()
# Lightweight immutable backbone sentinels plus requires_grad/grad assertions every update.
sentinels={n:p.detach().view(-1)[::max(1,p.numel()//32)].clone() for n,p in model.named_parameters()}
source=[json.loads(line) for line in (a/'training-prompts-v1/prompts.jsonl').open()];prompts={r['id']:r['prompt'] for r in source};del source
cache={}
def example(x):
 iid=x['id'];d=a/'training-rollouts-v1'/iid
 if iid not in cache:
  text=tok.apply_chat_template([{'role':'user','content':prompts[iid]}],tokenize=False,add_generation_prompt=True,enable_thinking=True);ids=tok.encode(text,add_special_tokens=False);req=json.loads((d/'request.json').read_text());assert sha(text.encode())==req['rendered_sha256'] and sha(json.dumps(ids).encode())==req['input_ids_sha256']
  marker='<|im_start|>user\n';pos=text.index(marker)+len(marker);prefix=tok.encode(text[:pos],add_special_tokens=False);assert ids[:len(prefix)]==prefix
  cache[iid]=(ids,len(prefix))
 ids,insert=cache[iid];raw=(d/f"answer-{x['attempt']}.json.gz").read_bytes();assert sha(raw)==x['raw_answer_sha256'];r=json.loads(gzip.decompress(raw));response=tok.encode(r['raw_engine_response']['text'],add_special_tokens=False)
 if not response or response[-1]!=151645:response.append(151645)
 assert sha(json.dumps(response).encode())==x['response_ids_sha256'];assert len(ids)+len(response)+256==x['total_tokens_with_sp']<=40000
 return make_response_sft_example(ids,response),insert

def save(update):
 if rank!=0:return
 tensors={'prompt_embeddings':wrapper.prompt_parameter.detach().cpu().contiguous()};path=out/f'update_{update:08d}.safetensors';tmp=path.with_suffix('.tmp');save_file(tensors,tmp);tmp.replace(path)
 state={'update':update,'prompt':tensors['prompt_embeddings'],'optimizer':optimizer.state_dict(),'contract':contract};tmp=out/'training-state.tmp';torch.save(state,tmp);tmp.replace(out/'training-state.pt')
 if update%40==0 or update==254:
  pending=out/'evaluation-pending';pending.mkdir(exist_ok=True);(pending/f'update_{update:08d}.json').write_text(json.dumps({'update':update,'checkpoint':str(path),'checkpoint_sha256':sha(path.read_bytes()),'status':'pending_not_evaluated','scope':'must use same SWE protocol and matched SFT/RL anchors; queue receipt is not evaluation success'}))
start=0
if (out/'training-state.pt').exists():
 state=torch.load(out/'training-state.pt',map_location='cpu',weights_only=True);assert state['contract']==contract;wrapper.prompt_parameter.data.copy_(state['prompt'].to('cuda'));optimizer.load_state_dict(state['optimizer']);start=state['update']
 if rank==0 and (out/'metrics.jsonl').exists():
  mp=out/'metrics.jsonl';lines=mp.read_text().splitlines();kept=[line for line in lines if json.loads(line)['update']<=start]
  if len(kept)!=len(lines):
   (out/f'metrics-interrupted-after-{start}-{os.environ.get("SLURM_JOB_ID", "local")}.jsonl').write_text('\n'.join(lines[len(kept):])+'\n');mp.write_text('\n'.join(kept)+'\n')
else:save(0)
if world>1:dist.barrier()
probe=os.environ.get('TRAIN_PROBE')=='1'
limit=int(os.environ.get('STOP_AFTER_UPDATE','254'))
if probe:
 # Longest selected sample is an explicit mechanics/throughput check in a SEPARATE run root.
 assert os.environ.get('TRAIN_RUN')!='formal';target=max(selected,key=lambda x:x['total_tokens_with_sp']);ex,pos=example(target);cpu=collate_examples([ex],insert_at=[pos],pad_token_id=151645);batch=type(cpu)(*(getattr(cpu,f.name).to('cuda') for f in dataclasses.fields(cpu)));optimizer.zero_grad(set_to_none=True);t=time.monotonic();loss,tokens=accumulate_response_loss(wrapper,batch,global_loss_denominator=world);result=finish_prompt_update(wrapper,optimizer,local_loss_sum=loss,local_response_tokens=tokens,local_loss_denominator=1,learning_rate=.01,max_grad_norm=10)
 for n,p in model.named_parameters():assert p.grad is None and torch.equal(sentinels[n],p.detach().view(-1)[::max(1,p.numel()//32)])
 assert result.gradient_norm>0
 if rank==0:(out/'probe.json').write_text(json.dumps({'sample':target,'metrics':dataclasses.asdict(result),'seconds':time.monotonic()-t,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'world_size':world,'scope':'longest-sequence engineering update; not a formal training checkpoint'},indent=2));print('LONGEST_SEQUENCE_PROBE_PASS',flush=True)
else:
 for update in range(start+1,min(limit,254)+1):
  epoch=(update-1)//127;bi=(update-1)%127;indices=list(range(len(selected)));random.Random(42+epoch).shuffle(indices);global_ids=indices[bi*64:(bi+1)*64];assert global_ids
  optimizer.zero_grad(set_to_none=True);loss_sum=0.;token_sum=0;local_count=0;t=time.monotonic()
  for index in global_ids[rank::world]:
   ex,pos=example(selected[index]);cpu=collate_examples([ex],insert_at=[pos],pad_token_id=151645);batch=type(cpu)(*(getattr(cpu,f.name).to('cuda') for f in dataclasses.fields(cpu)));loss,tokens=accumulate_response_loss(wrapper,batch,global_loss_denominator=len(global_ids));loss_sum+=loss;token_sum+=tokens;local_count+=1
  result=finish_prompt_update(wrapper,optimizer,local_loss_sum=loss_sum,local_response_tokens=token_sum,local_loss_denominator=local_count,learning_rate=.01,max_grad_norm=10)
  for n,p in model.named_parameters():assert p.grad is None and torch.equal(sentinels[n],p.detach().view(-1)[::max(1,p.numel()//32)])
  metric={'update':update,'epoch':epoch+1,'samples':len(global_ids),'seconds':time.monotonic()-t,**dataclasses.asdict(result)}
  if rank==0:
   with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(metric)+'\n')
   print(json.dumps(metric),flush=True)
  if update%20==0 or update==254:save(update)
  if world>1:dist.barrier()
 print('TRAINING_CHUNK_COMPLETE',min(limit,254),'rank',rank,flush=True)
if world>1:dist.destroy_process_group()
