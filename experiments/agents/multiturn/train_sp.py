"""Multi-turn soft-prompt trainer for the SkyRL (SWE-bench) and TMax (Terminal-Bench) agents.

Renders complete resolved teacher episodes with soft_prompt_distillation.multiturn (assistant-only labels),
frozen backbone, global-batch prompt updates. Usage: torchrun ... train_sp.py {skyrl,tmax}
Env: AGENT_ROOT (collected episodes per lane, with plan.json), AGENT_CONFIGS.
"""
import os,sys,json,hashlib,random,time,dataclasses,copy
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.distributed as dist
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoModelForImageTextToText
from safetensors.torch import save_file
from soft_prompt_distillation.multiturn import render_multiturn
from soft_prompt_distillation.training import SoftPromptModel,initialize_prompt_embeddings,collate_examples,accumulate_response_loss,finish_prompt_update
lane=sys.argv[1];root=Path(os.environ['AGENT_ROOT'])/lane;assert (root/'collection-terminal.json').exists()
rank=int(os.environ.get('RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'));local=int(os.environ.get('LOCAL_RANK','0'));torch.cuda.set_device(local)
if world>1:dist.init_process_group('nccl',device_id=torch.device('cuda',local))
out=root/'soft-prompt';out.mkdir(exist_ok=True);plan=json.loads((root/'plan.json').read_text());bs=plan['effective_batch_size'];modelpath=json.loads(Path(os.environ.get('AGENT_CONFIGS', 'artifacts/agent-pilots')+'/'+('openhands' if lane=='skyrl' else 'tmax')+'/model-paths.json').read_text())[0]['path'];tok=AutoTokenizer.from_pretrained(modelpath,local_files_only=True);examples=[];selected=[];rejected=[];counts={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for slot in plan['slots']:
 iid,j=slot['task_id'],str(slot['sample']);d=root/'official-slots-v1'/iid/j;source=d/('session' if lane=='skyrl' else 'episode');v=source/('fresh-score/verdict.json' if lane=='skyrl' else 'receipt.json')
 if not v.exists() or json.loads(v.read_text()).get('resolved') is not True:continue
 if plan['selection_cap'] is not None and counts.get(iid,0)>=plan['selection_cap']:continue
 try:
  requests=[json.loads(x) for x in (source/'requests.jsonl').read_text().splitlines()];responses=[json.loads(x) for x in (source/'responses.jsonl').read_text().splitlines()];assert len(requests)==len(responses)>0
  assert all(x['temperature']==1 and x['top_p']==1 and x['top_k']==-1 and x['seed']==slot['seed'] for x in requests)
  kwargs={}
  if lane=='skyrl':
   for k in range(1,len(requests)):
    prev=requests[k-1]['messages'];current=requests[k]['messages'];assert current[:len(prev)]==prev and current[len(prev)]['role']=='assistant';assert sum(x['role']=='assistant' for x in current)==k
   reply=responses[-1]['choices'][0]['message'];messages=copy.deepcopy(requests[-1]['messages'])+[{'role':'assistant','content':reply.get('content') or ''}]
   for m in messages:
    if isinstance(m.get('content'),list):
     assert all(x.get('type')=='text' for x in m['content']);m['content']=''.join(x['text'] for x in m['content'])
  else:
   messages=json.loads((source/'trajectory.json').read_text());tools=requests[0]['tools']
   assert all(x.get('tools')==tools and messages[:len(x['messages'])]==x['messages'] for x in requests)
   for m in messages:
    for call in m.get('tool_calls') or []:
     if isinstance(call['function'].get('arguments'),str):call['function']['arguments']=json.loads(call['function']['arguments'])
   kwargs={'tools':tools,'enable_thinking':True}
  rendered=render_multiturn(tok,messages,template_kwargs=kwargs,max_length=(32768 if lane=='skyrl' else 65536)-256,insertion='system_body_start');assert len(rendered.assistant_spans)==len(responses)
  examples.append(rendered);counts[iid]=counts.get(iid,0)+1;selected.append({**slot,'requests_sha256':sha(source/'requests.jsonl'),'responses_sha256':sha(source/'responses.jsonl'),'verdict_sha256':sha(v),'tokens':len(rendered.token_offsets)})
 except Exception as exc:rejected.append({**slot,'error':repr(exc)})
if rank==0:(out/'selection.json').write_text(json.dumps({'selected':selected,'rejected':rejected},indent=2))
assert examples,'no usable successful trajectories; do not train empty set'
contract={'plan_sha256':sha(root/'plan.json'),'selection':selected,'student':modelpath,'template_sha256':hashlib.sha256(tok.chat_template.encode()).hexdigest(),'insertion':'system_body_start','sp_tokens':256,'batch':bs,'epochs':2,'lr':.01,'seed':42,'normalization':'sequence','runner_sha256':sha(Path(__file__))}
if rank==0:
 p=out/'contract.json'
 if p.exists():assert json.loads(p.read_text())==contract
 else:p.write_text(json.dumps(contract,indent=2))
if world>1:dist.barrier()
torch.manual_seed(42);cls=AutoModelForCausalLM if lane=='skyrl' else AutoModelForImageTextToText
model=cls.from_pretrained(modelpath,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to(torch.device('cuda',local));model.config.use_cache=False
wrapper=SoftPromptModel(model,initialize_prompt_embeddings(model,tok,SimpleNamespace(method='sampled_vocabulary',seed=42),256),activation_checkpointing=True,cross_entropy_backend='torch',loss_normalization='sequence');opt=torch.optim.AdamW([wrapper.prompt_parameter],lr=.01,weight_decay=0)
sentinels={n:p.detach().flatten()[::max(1,p.numel()//32)].clone() for n,p in model.named_parameters()};start=0
if (out/'training-state.pt').exists():
 state=torch.load(out/'training-state.pt',map_location='cpu',weights_only=True);assert state['contract']==contract;wrapper.prompt_parameter.data.copy_(state['prompt'].to(local));opt.load_state_dict(state['optimizer']);start=state['update']
update=0
for epoch in range(2):
 order=list(range(len(examples)));random.Random(42+epoch).shuffle(order)
 for begin in range(0,len(order),bs):
  update+=1
  if update<=start:continue
  batchids=order[begin:begin+bs];mine=batchids[rank::world];opt.zero_grad(set_to_none=True);loss_sum=0.;tokens=0;timer=time.time()
  for index in mine:
   x=examples[index];cpu=collate_examples([x.example],insert_at=[x.insert_at],pad_token_id=tok.pad_token_id or tok.eos_token_id);batch=type(cpu)(*(getattr(cpu,f.name).to(local) for f in dataclasses.fields(cpu)));loss,n=accumulate_response_loss(wrapper,batch,global_loss_denominator=len(batchids));loss_sum+=loss;tokens+=n
  if not mine:wrapper.prompt_parameter.grad=torch.zeros_like(wrapper.prompt_parameter)
  result=finish_prompt_update(wrapper,opt,local_loss_sum=loss_sum,local_response_tokens=tokens,local_loss_denominator=len(mine),learning_rate=.01,max_grad_norm=10)
  for name,p in model.named_parameters():assert p.grad is None and torch.equal(sentinels[name],p.detach().flatten()[::max(1,p.numel()//32)])
  assert torch.isfinite(wrapper.prompt_parameter).all()
  if rank==0:
   tensor=wrapper.prompt_parameter.detach().cpu().contiguous();torch.save({'update':update,'prompt':tensor,'optimizer':opt.state_dict(),'contract':contract},out/'training-state.tmp');(out/'training-state.tmp').replace(out/'training-state.pt')
   endepoch=begin+bs>=len(order)
   if update%10==0 or endepoch:save_file({'prompt_embeddings':tensor},str(out/f'update_{update:08d}.safetensors'))
   if update%20==0 or epoch==1 and endepoch:(out/f'eval-request-{update:08d}.json').write_text(json.dumps({'checkpoint':str(out/f'update_{update:08d}.safetensors'),'update':update,'status':'pending_same_backend_validation_and_benchmark'}))
   with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps({'update':update,'epoch':epoch,'global_batch':len(batchids),'seconds':time.time()-timer,**dataclasses.asdict(result)})+'\n')
  if world>1:dist.barrier()
if rank==0:(out/'complete.json').write_text(json.dumps({'updates':update,'examples':len(examples),'status':'trained_not_evaluated'}))
if world>1:dist.destroy_process_group()
