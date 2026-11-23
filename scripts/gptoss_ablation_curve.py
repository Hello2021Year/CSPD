"""Render both curves from complete fixed-denominator checkpoint summaries."""
import argparse,json
from pathlib import Path
from gptoss_ablation import atomic,sha
from gptoss_ablation_contract import checkpoint_steps,evaluation_root,validate_summary,expected_counts

def main():
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--config',type=Path,required=True);a=p.parse_args();r=evaluation_root(a.root,a.config);steps=checkpoint_steps(a.config)
 def load(p):
  return validate_summary(json.loads(p.read_text()),a.config)
 s=load(r/'baselines/evaluate_student/summary.json');t=load(r/'baselines/evaluate_teacher/summary.json');gap=t['accuracy']-s['accuracy']
 curves={}
 for arm in ['correct','all']:
  points=[]
  for step in steps:
   path=r/arm/'evaluate_student_soft_prompt'/f'update_{step:08d}'/'summary.json'
   if not path.exists():continue
   x=load(path);points.append({'update':step,'accuracy':x['accuracy'],'recovery_percent':100*(x['accuracy']-s['accuracy'])/gap if gap>0 else None,'summary_sha256':sha(path),'benchmarks':x['benchmarks']})
  curves[arm]=points
 report={'config_sha256':sha(a.config),'expected_samples':sum(n for _,n in expected_counts(a.config).values()),'expected_steps':steps,'baseline_student':s['accuracy'],'baseline_teacher':t['accuracy'],'positive_headroom':gap>0,'recovery_formula':'100*(SP-student)/(teacher-student), macro accuracy','curves':curves,'complete':all(len(x)==len(steps) for x in curves.values())}
 out=r/'curves';out.mkdir(exist_ok=True);atomic(out/'recovery.json',report)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 for metric,ylabel in [('accuracy','Macro accuracy'),('recovery_percent','Teacher gap recovery (%)')]:
  fig,ax=plt.subplots()
  for arm,points in curves.items():
   usable=[x for x in points if x[metric] is not None];ax.plot([x['update'] for x in usable],[x[metric] for x in usable],marker='o',label=arm)
  ax.set(xlabel='Optimizer updates',ylabel=ylabel,title='GPT-OSS equal-N correctness-filter ablation');ax.legend();ax.grid(alpha=.3);fig.tight_layout();fig.savefig(out/f'{metric}.png');plt.close(fig)
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
