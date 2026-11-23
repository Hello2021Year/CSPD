"""Existing corrected evaluator with immutable per-request raw-response cache.
Rerunning this entry resumes saved responses, not regenerated answers. An
interrupted request with no raw is archived and retried under DM968195fd.
"""
import argparse, asyncio, dataclasses, hashlib, json, os
from pathlib import Path
from gptoss_ablation import atomic, sha, engine_receipt, recover_incomplete
from gptoss_ablation_contract import checkpoint_steps, evaluation_root, training_identity, validate_summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--arm',choices=['student','teacher','correct','all'],required=True);p.add_argument('--step',type=int);p.add_argument('--dp-size',type=int,default=1)
    a=p.parse_args();assert os.environ.get('SLURM_JOB_ID');assert 1<=a.dp_size<=8
    from soft_prompt_distillation.common import RunOptions
    from soft_prompt_distillation.pipeline import evaluate_target
    from soft_prompt_distillation.sglang_backend import SGLangRuntime
    label=a.arm if a.arm in ('student','teacher') else f'{a.arm}-{a.step}'
    protocol_root=evaluation_root(a.root,a.config)
    cache=protocol_root/'eval-raw'/label;cache.mkdir(parents=True,exist_ok=True)
    identity={'config_sha256':sha(a.config),'arm':a.arm,'step':a.step}
    train=a.root/a.arm/'train'
    if a.arm in ('correct','all'):
        assert a.step in checkpoint_steps(a.config)
        contract=a.root/a.arm/'training-identity.json'
        expected=training_identity(a.root,a.arm,a.config)
        assert json.loads(contract.read_text())==expected,'training/evaluation config identity changed'
        identity['checkpoint_sha256']=sha(train/'checkpoints'/f'update_{a.step:08d}.safetensors')
    ip=cache/'identity.json'
    if ip.exists():assert json.loads(ip.read_text())==identity,'evaluation identity changed'
    else:atomic(ip,identity)
    new_responses=0
    async def cached(self,prompts,params,data_parallel_ranks,*,progress_label):
        nonlocal new_responses
        # Human c0ed0725: finish the complete evaluation point before remote
        # backup; raw responses are still atomically persisted per request.
        resolved=cache/'resolved-engine.json'
        receipt=engine_receipt(self)
        # Exported SP model lives in a fresh temporary directory on resume;
        # checkpoint bytes are bound by identity.json, not that ephemeral path.
        receipt.pop('model_path')
        if resolved.exists():assert json.loads(resolved.read_text())==receipt,'resolved evaluation engine changed'
        else:atomic(resolved,receipt)
        vp=cache/'runtime.json';v={'sglang':self.version}
        if vp.exists():assert json.loads(vp.read_text())==v,'evaluation runtime changed'
        else:atomic(vp,v)
        async def one(prompt,param,rank):
            nonlocal new_responses
            request={'input_ids':list(prompt),'sampling_params':param,'data_parallel_rank':None}
            key=hashlib.sha256(json.dumps(request,sort_keys=True,separators=(',',':')).encode()).hexdigest();d=cache/key
            if (d/'raw.json').exists():
                assert json.loads((d/'request.json').read_text())==request
                return json.loads((d/'raw.json').read_text())
            recover_incomplete(d, request)
            d.mkdir(exist_ok=True);atomic(d/'request.json',request)
            try:
                kwargs=dict(input_ids=list(prompt),sampling_params=param)
                raw=await self._engine.async_generate(**kwargs);atomic(d/'raw.json',raw);new_responses+=1;return raw
            except Exception as e:atomic(d/'error.json',{'type':type(e).__name__,'error':str(e)});raise
        results=await asyncio.gather(*(one(*x) for x in zip(prompts,params,data_parallel_ranks,strict=True)),return_exceptions=True)
        errors=[x for x in results if isinstance(x,BaseException)]
        if errors:raise RuntimeError(f'preserved evaluation error, no retries: {errors[0]}')
        return results
    SGLangRuntime._generate_token_ids_concurrently=cached
    if a.arm in ('student','teacher'):
        output=protocol_root/'baselines'/('evaluate_'+a.arm)
        if (output/'summary.json').exists():
            validate_summary(json.loads((output/'summary.json').read_text()),a.config)
            print('COMPLETE',output);return
        if output.exists():
            i=1
            while output.with_name(output.name+f'.incomplete-{i}').exists():i+=1
            output.rename(output.with_name(output.name+f'.incomplete-{i}'))
        target=a.arm;kwargs={}
    else:
        output=protocol_root/a.arm/'evaluate_student_soft_prompt';target='soft-prompt';kwargs={'train_dir':train,'checkpoint':str(a.step)}
    options=RunOptions(config=a.config.resolve(),output=output.resolve(),cache_dir=a.cache.resolve(),sglang_tp_size=1,sglang_dp_size=a.dp_size,sglang_batch_size='all',sglang_log_level='warning')
    result=evaluate_target(options,target,**kwargs)
    summary=output/'summary.json' if a.arm in ('student','teacher') else output/f'update_{a.step:08d}'/'summary.json'
    validate_summary(json.loads(summary.read_text()),a.config)
    print(result,flush=True)
if __name__=='__main__':main()
