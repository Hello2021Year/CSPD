"""Train up to one durable checkpoint without changing the config-bound schedule."""
import argparse, os
from pathlib import Path
from gptoss_ablation_contract import checkpoint_steps, training_identity

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--arm',choices=['correct','all'],required=True);p.add_argument('--through',type=int,required=True)
 a=p.parse_args();assert os.environ.get('SLURM_JOB_ID');assert a.through in checkpoint_steps(a.config)
 from soft_prompt_distillation.common import RunOptions, snapshot_config
 from gptoss_ablation import atomic, sha
 import json
 from soft_prompt_distillation import distillation as d
 import torch
 out=a.root/a.arm/'train';state=out/'training_state.pt'
 contract=a.root/a.arm/'training-identity.json'
 identity=training_identity(a.root,a.arm,a.config)
 if contract.exists():assert json.loads(contract.read_text())==identity,'training input changed'
 elif state.exists():raise RuntimeError('resume state exists without training identity')
 else:atomic(contract,identity)
 if state.exists():
  update=int(torch.load(state,map_location='cpu',weights_only=False)['update'])
  if update>=a.through:print('CHECKPOINT_ALREADY_DURABLE',update);return
 elif out.exists():raise RuntimeError('training directory exists without resume state; preserve and investigate')
 total=identity['total_updates']
 original=d._save_training_state
 def save(output,update,optimizer,metrics_count):
  original(output,update,optimizer,metrics_count)
  snapshot_config(RunOptions(config=a.config.resolve(),output=output,cache_dir=a.cache.resolve()))
  atomic(output/'pause-receipt.json',dict(identity,update=update,status='checkpoint_durable'))
  if update==a.through and update<total:
   print('PAUSED_AT_DURABLE_CHECKPOINT',update,flush=True)
   raise SystemExit(0)
 d._save_training_state=save
 options=RunOptions(config=a.config.resolve(),output=out.resolve(),cache_dir=a.cache.resolve(),resume_training=state.exists())
 d.train(options,a.root/'datasets'/a.arm/'teacher_trajectories.jsonl')
if __name__=='__main__':main()
