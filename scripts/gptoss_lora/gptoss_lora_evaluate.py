"""Merge one small K/V adapter temporarily and use the established math evaluator."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import tempfile
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from gptoss_lora_core import atomic, sha


def export_merged(source, adapter, output, scale=1.):
    import torch
    weights=load_file(str(adapter));deltas={}
    assert len(weights)==96
    for name,value in weights.items():
        if name.endswith('.lora_A'):
            prefix=name.removesuffix('.lora_A');b=weights[prefix+'.lora_B']
            assert value.shape==(5,2880) and b.shape==(512,5)
            deltas[prefix+'.weight']=(b@value)*scale
    assert len(deltas)==48
    index=json.loads((source/'model.safetensors.index.json').read_text())
    affected={index['weight_map'][name] for name in deltas};seen=set()
    for filename in sorted(set(index['weight_map'].values())):
        if filename not in affected:
            (output/filename).symlink_to(source/filename);continue
        with safe_open(source/filename,framework='pt',device='cpu') as f:
            tensors={name:f.get_tensor(name) for name in f.keys()};metadata=f.metadata()
            for name in deltas:
                if name in tensors:
                    original=tensors[name];assert original.dtype==torch.bfloat16
                    tensors[name]=(original.float()+deltas[name]).to(original.dtype);seen.add(name)
            save_file(tensors,str(output/filename),metadata=metadata)
            del tensors
    assert seen==set(deltas)
    for name in ['model.safetensors.index.json','config.json','generation_config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','chat_template.jinja','vocab.json','merges.txt']:
        if (source/name).exists():(output/name).symlink_to(source/name)
    atomic(output/'lora_export.json',{'base':str(source),'adapter_sha256':sha(adapter),'scale':scale,'targets':sorted(seen),'merge_dtype':'float32 accumulation, bfloat16 storage'})


def install_cache(raw, identity):
    from soft_prompt_distillation.sglang_backend import SGLangRuntime
    from gptoss_ablation import engine_receipt, recover_incomplete
    raw.mkdir(parents=True,exist_ok=True)
    if (raw/'identity.json').exists():assert json.loads((raw/'identity.json').read_text())==identity
    else:atomic(raw/'identity.json',identity)
    async def cached(self,prompts,params,data_parallel_ranks,*,progress_label):
        receipt=engine_receipt(self);receipt.pop('model_path')
        if (raw/'resolved-engine.json').exists():assert json.loads((raw/'resolved-engine.json').read_text())==receipt
        else:atomic(raw/'resolved-engine.json',receipt)
        async def one(prompt,param,rank):
            request={'input_ids':list(prompt),'sampling_params':param,'data_parallel_rank':None}
            key=hashlib.sha256(json.dumps(request,sort_keys=True,separators=(',',':')).encode()).hexdigest();path=raw/key
            if (path/'raw.json').exists():
                assert json.loads((path/'request.json').read_text())==request
                return json.loads((path/'raw.json').read_text())
            recover_incomplete(path,request);path.mkdir(exist_ok=True);atomic(path/'request.json',request)
            try:
                result=await self._engine.async_generate(input_ids=list(prompt),sampling_params=param)
                atomic(path/'raw.json',result);return result
            except Exception as error:
                atomic(path/'error.json',{'type':type(error).__name__,'error':str(error)});raise
        results=await asyncio.gather(*(one(*x) for x in zip(prompts,params,data_parallel_ranks,strict=True)),return_exceptions=True)
        failures=[x for x in results if isinstance(x,BaseException)]
        if failures:raise RuntimeError(f'Preserved request failure: {failures[0]}')
        return results
    SGLangRuntime._generate_token_ids_concurrently=cached


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--step',type=int,required=True);a=p.parse_args()
    spec=json.loads((a.root/'protocol.json').read_text());assert a.step in spec['evaluation_order']
    adapter=a.root/'train/adapters'/f'update_{a.step:08d}.safetensors'
    identity={'method':'lora_kv','rank':5,'alpha':5,'update':a.step,'adapter_sha256':sha(adapter),'protocol_sha256':sha(a.root/'protocol.json'),'reference_config_sha256':sha(spec['reference_config'])}
    output=a.root/'evaluations'/f'update_{a.step:08d}';raw=a.root/'eval-raw'/f'update_{a.step:08d}'
    install_cache(raw,identity)
    if (output/'summary.json').exists():
        assert json.loads((output/'lora-identity.json').read_text())==identity
        print('REUSE_EVALUATION',a.step,flush=True);return
    if output.exists():output.rename(output.with_name(output.name+f'.incomplete-{__import__("time").time_ns()}'))
    from soft_prompt_distillation.common import RunOptions
    from soft_prompt_distillation import evaluation
    from gptoss_ablation_contract import validate_summary
    source=Path(spec['student_path']);temporary=a.root/'temporary-models';temporary.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'lora-{a.step}-',dir=temporary) as directory:
        model_path=Path(directory);export_merged(source,adapter,model_path,spec['alpha']/spec['rank'])
        original=evaluation.model_source
        def merged_source(options,config,role):
            return (str(model_path),None) if role=='student' else original(options,config,role)
        evaluation.model_source=merged_source
        options=RunOptions(config=Path(spec['reference_config']),output=output,cache_dir=source.parents[2],sglang_tp_size=1,sglang_dp_size=4,sglang_batch_size='all',sglang_log_level='warning')
        evaluation.evaluate(options,'student')
        validate_summary(json.loads((output/'summary.json').read_text()),Path(spec['reference_config']))
        atomic(output/'lora-identity.json',identity)
        atomic(output/'lora-export.json',json.loads((model_path/'lora_export.json').read_text()))
    engine=json.loads((raw/'resolved-engine.json').read_text())
    assert engine['dp_size']==4 and engine['tp_size']==1 and engine['model_config_dtype']=='torch.bfloat16' and engine['model_config_quantization'] is None
    print('LORA_EVALUATION_COMPLETE',a.step,flush=True)


if __name__=='__main__':main()
