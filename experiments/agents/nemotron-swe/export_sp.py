"""Export a trained SWE prompt as a virtual-token model for SGLang/vLLM serving.

Env: WORK, ALIGNMENT_ROOT, EVAL_ARM (sp<update>), STUDENT_MODEL, TEACHER_MODEL, TRAIN_OUTPUT, SP_EXPORT.
"""
import os,sys,json,hashlib,gc
from pathlib import Path
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM,AutoTokenizer,AutoConfig
w=Path(os.environ['WORK']);# requires `pip install -e .` from the repository root
from soft_prompt_distillation.virtual_tokens import install_virtual_prompt,write_virtual_prompt_metadata,insert_virtual_token_ids
r=Path(os.environ['ALIGNMENT_ROOT'])/'full500-v1';arm=os.environ['EVAL_ARM'];update=int(arm[2:]);assert update in [40,80,120,140,160,200,240,254]
checkpoint=Path(os.environ['TRAIN_OUTPUT'])/f'update_{update:08d}.safetensors'
base=os.environ['STUDENT_MODEL']
out=Path(os.environ['SP_EXPORT']);assert not out.exists();sp=load_file(checkpoint)['prompt_embeddings'];assert sp.shape==(256,4096) and torch.isfinite(sp).all();sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
t=AutoTokenizer.from_pretrained(base,local_files_only=True);teacher=AutoTokenizer.from_pretrained(os.environ['TEACHER_MODEL'],local_files_only=True);assert t.get_vocab()==teacher.get_vocab()
cfg=AutoConfig.from_pretrained(base);cfg.max_position_embeddings=65536;cfg.rope_scaling={'rope_type':'yarn','factor':2.0,'original_max_position_embeddings':32768}
m=AutoModelForCausalLM.from_pretrained(base,config=cfg,torch_dtype=torch.bfloat16,attn_implementation='sdpa')
m.eval().cuda();text=t.apply_chat_template([{'role':'user','content':'Locate and repair the bug in this short Python example.'}],tokenize=False,add_generation_prompt=True)
marker='<|im_start|>user\n';at=text.index(marker)+len(marker);ids=t.encode(text,add_special_tokens=False);prefix=t.encode(text[:at],add_special_tokens=False);assert ids[:len(prefix)]==prefix
with torch.inference_mode():
 x=torch.tensor([ids],device='cuda');e=m.get_input_embeddings()(x);direct=torch.cat([e[:,:len(prefix)],sp.to(device='cuda',dtype=e.dtype)[None],e[:,len(prefix):]],dim=1);reference=m(inputs_embeds=direct).logits[:,-1].float().cpu()
 spec=install_virtual_prompt(m,t,sp);v=list(spec.token_ids);actual=m(input_ids=torch.tensor([insert_virtual_token_ids(ids,len(prefix),v)],device='cuda')).logits[:,-1].float().cpu();mask=torch.ones(actual.shape[-1],dtype=torch.bool);mask[v]=False;delta=(reference[:,mask]-actual[:,mask]).abs().max().item();assert delta==0,delta
 assert torch.equal(m.get_input_embeddings().weight[v].cpu(),sp.to(torch.bfloat16))
m.cpu();del e,direct,x,reference,actual;gc.collect();torch.cuda.empty_cache();m.generation_config.do_sample=True;m.save_pretrained(out,safe_serialization=True);t.save_pretrained(out);write_virtual_prompt_metadata(out,spec,base_model=base,soft_prompt_sha256=sha)
receipt={'arm':arm,'checkpoint':str(checkpoint),'checkpoint_sha256':sha,'export_path':str(out),'virtual_ids':v,'logit_bias':spec.logit_bias,'placement':'after user role header, both localization and repair','input_budget_includes_sp':True,'input_budget':32768,'output_budget':32768,'context':65536,'rope':'YaRN2 evaluation; original training native RoPE','check':'HF BF16 direct inputs_embeds vs virtual input IDs at evaluation YaRN2; identical nonvirtual final logits','max_abs_nonvirtual_logit_difference':delta,'runtime_caveat':'not a HF-vs-SGLang numerical equality claim; virtual output rows zeroed and logit bias -100','job':os.environ.get('SLURM_JOB_ID', 'local')}
p=r/f'{arm}-export-receipt.json'
if p.exists():
 old=json.loads(p.read_text());assert old['checkpoint_sha256']==sha and old['virtual_ids']==v
else:p.write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt,indent=2),flush=True)
