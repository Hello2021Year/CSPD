"""Separate H100 engineering probe, never a benchmark or training point.
Uses a deterministic synthetic token sequence to exercise the full 40K budget.
"""
import argparse,copy,json,os
from pathlib import Path
from gptoss_ablation import atomic

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
 a=p.parse_args();assert os.environ.get('SLURM_JOB_ID')
 import torch
 from soft_prompt_distillation.common import RunOptions,load_template_tokenizer,model_source,load_model
 from soft_prompt_distillation.config import load_config
 from soft_prompt_distillation.training import SoftPromptModel,initialize_prompt_embeddings
 out=a.root/'engineering-preflight';out.mkdir(exist_ok=False)
 c=load_config(a.config);o=RunOptions(config=a.config,output=out,cache_dir=a.cache)
 tokenizer,_=load_template_tokenizer(o,c);source,revision=model_source(o,c,'student')
 model=load_model(source,revision,attention_implementation='flex_attention',placement='replicated')
 init=initialize_prompt_embeddings(model,tokenizer,c.distillation.prompt_init,256)
 wrapper=SoftPromptModel(model,init,activation_checkpointing=True,cross_entropy_backend='liger_fused',loss_normalization='token')
 assert all(not p.requires_grad for p in model.parameters());assert wrapper.prompt_parameter.dtype==torch.float32
 assert model.get_input_embeddings().weight.dtype==torch.bfloat16
 # Cheap value sentinels plus authoritative all-parameter gradient/freeze checks.
 sentinels=[p.detach().reshape(-1)[:8].clone() for p in model.parameters()]
 base=tokenizer.encode('Compute one plus one. The answer is two.',add_special_tokens=False)
 def batch(n):
  ids=torch.tensor((base*((n+len(base)-1)//len(base)))[:n],device='cuda').unsqueeze(0)
  labels=ids.clone();labels[:,:n//2]=-100
  return ids,torch.ones_like(ids),torch.tensor([1],device='cuda'),labels
 optimizer=torch.optim.AdamW([wrapper.prompt_parameter],lr=.01,weight_decay=0)
 def update(n):
  optimizer.zero_grad(set_to_none=True);loss,_=wrapper.response_loss(*batch(n));assert torch.isfinite(loss)
  loss.backward();g=wrapper.prompt_parameter.grad;assert g is not None and torch.isfinite(g).all() and g.norm()>0
  assert all(p.grad is None for p in model.parameters())
  norm=torch.nn.utils.clip_grad_norm_([wrapper.prompt_parameter],10);optimizer.step();return float(loss),float(norm)
 first=update(128);saved_prompt=wrapper.prompt_parameter.detach().clone();saved_optimizer=copy.deepcopy(optimizer.state_dict())
 torch.save({'prompt':saved_prompt.cpu(),'optimizer':saved_optimizer},out/'resume-probe.pt')
 second=update(128);reference=wrapper.prompt_parameter.detach().clone()
 saved=torch.load(out/'resume-probe.pt',weights_only=False,map_location='cuda')
 with torch.no_grad():wrapper.prompt_parameter.copy_(saved['prompt'])
 optimizer.load_state_dict(saved['optimizer']);replayed=update(128)
 torch.testing.assert_close(wrapper.prompt_parameter,reference,rtol=1e-5,atol=1e-6)
 # Full original text budget + 256 SP. No truncation of a real trajectory.
 torch.cuda.reset_peak_memory_stats();full=update(39744)
 for old,param in zip(sentinels,model.parameters(),strict=True):assert torch.equal(old,param.detach().reshape(-1)[:8])
 atomic(out/'receipt.json',{'status':'pass','scope':'synthetic engineering only','gpu':torch.cuda.get_device_name(),'text_length':39744,'prompt_length':256,'total_length':40000,'short_first':first,'short_second':second,'short_resume':replayed,'full':full,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'all_backbone_frozen':True,'all_backbone_grads_none':True,'parameter_sentinels_unchanged':True,'resume_prompt_close':True})
 print('ENGINEERING_PREFLIGHT_PASS',flush=True)
if __name__=='__main__':main()
