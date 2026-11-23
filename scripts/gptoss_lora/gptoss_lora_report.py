"""Audit raw scored outputs and write the LoRA/SP comparison without reselection."""
import collections
import json
from pathlib import Path
from gptoss_lora_core import atomic, sha


def audit_point(path):
    summary=json.loads((path/'summary.json').read_text());counts=collections.Counter();correct=collections.Counter();slots=set();questions=collections.defaultdict(set);ends=collections.Counter()
    with (path/'results.jsonl').open() as f:
        for line in f:
            r=json.loads(line);slot=(r['dataset'],r['question_id'],r['sample_index']);assert slot not in slots;slots.add(slot)
            questions[slot[:2]].add(slot[2]);counts[slot[0]]+=1;correct[slot[0]]+=r['correct'];ends[r['finish_reason']]+=1
            assert r['input_tokenization_matches_canonical'] is True and r['response_ids'] and r['raw_completion'].strip()
    assert counts=={'aime_2024':480,'aime_2025':480,'math_500':1000} and len(questions)==560
    for (ds,_),indices in questions.items():assert indices==set(range(2 if ds=='math_500' else 16))
    scores={ds:correct[ds]/counts[ds] for ds in counts};macro=sum(scores.values())/3
    assert abs(macro-summary['accuracy'])<1e-12
    for ds,value in scores.items():assert abs(value-summary['benchmarks'][ds]['accuracy'])<1e-12
    return {'accuracy':macro,'benchmarks':scores,'samples':1960,'finish_reasons':dict(ends),'results_sha256':sha(path/'results.jsonl'),'summary_sha256':sha(path/'summary.json')}


def report(root,complete=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);spec=json.loads((root/'protocol.json').read_text());reference=Path(spec['reference_evaluation'])
    prior=json.loads((reference/'curves/recovery.json').read_text())
    points=[]
    for step in sorted(spec['evaluation_order']):
        path=root/'evaluations'/f'update_{step:08d}'
        if not (path/'lora-identity.json').exists():continue
        identity=json.loads((path/'lora-identity.json').read_text())
        assert identity['adapter_sha256']==sha(root/'train/adapters'/f'update_{step:08d}.safetensors')
        row=audit_point(path);row['update']=step
        row['recovery_percent']=100*(row['accuracy']-prior['baseline_student'])/(prior['baseline_teacher']-prior['baseline_student'])
        manifest=json.loads((path/'manifest.json').read_text());anchor_manifest=json.loads((reference/'baselines/evaluate_student/manifest.json').read_text())
        assert manifest['source']['datasets']==anchor_manifest['source']['datasets']
        points.append(row)
    if complete:assert len(points)==11 and points[-1]['update']==200
    curves=root/'curves';curves.mkdir(exist_ok=True)
    result={'complete':complete,'protocol_sha256':sha(root/'protocol.json'),'baseline_student':prior['baseline_student'],'baseline_teacher':prior['baseline_teacher'],'lora_correct':points,'sp_correct':prior['curves']['correct'],'anchor_source':str(reference),'anchor_curve_sha256':sha(reference/'curves/recovery.json')}
    atomic(curves/'recovery.json',result)
    if points:
        for filename,field,label in [('recovery_percent.png','recovery_percent','Teacher gap recovery (%)'),('accuracy.png','accuracy','Macro accuracy (%)')]:
            fig,ax=plt.subplots(figsize=(7,4.8))
            for name,series in [('LoRA K/V r=5 correct',points),('SP 256 correct',prior['curves']['correct'])]:
                ax.plot([r['update'] for r in series],[r[field]*(100 if field=='accuracy' else 1) for r in series],marker='o',label=name)
            ax.set(xlabel='Optimizer update',ylabel=label,title='GPT-OSS correct: LoRA vs soft prompt');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(curves/filename,dpi=160);plt.close(fig)
    table=['| Update | LoRA accuracy | LoRA recovery | SP accuracy | SP recovery |','|---:|---:|---:|---:|---:|']
    for row in points:
        sp=next(x for x in prior['curves']['correct'] if x['update']==row['update'])
        table.append(f"| {row['update']} | {row['accuracy']*100:.4f}% | {row['recovery_percent']:.4f}% | {sp['accuracy']*100:.4f}% | {sp['recovery_percent']:.4f}% |")
    text='# GPT-OSS correct: K/V LoRA rank 5\n\n'+('Complete' if complete else 'Partial')+' recovery curve.\n\n'
    text+='Same 28,946 correct trajectories and global batch 256 as SP; 200 updates; seed42. LoRA: all 24 K/V projections, rank5/alpha5/dropout0, 814,080 FP32 parameters, BF16 compute, AdamW LR 2e-4 constant, no warmup/weight decay, clip10. Learning rate is prespecified, not tuned. No soft-prompt tokens are inserted.\n\n'
    text+='Four replicated GPUs with token-normalized SUM gradient reduction. Microbatch selected by engineering throughput/memory probes, with only within-update length sorting. Train all200 updates continuously before evaluation. Save every10 plus113; evaluation order113,100,120,80,140,60,160,40,180,20,200. DP4/TP1 evaluation, no client cap, AIME2024/2025 avg@16, MATH500 avg@2, 1960 outputs/point. Anchors are reused from the verified reference protocol.\n\n'
    text+='\n'.join(table)+'\n\n![Recovery](curves/recovery_percent.png)\n\nRecovery = 100*(macro-SFT)/(RL-SFT), equal benchmark weights. Length-limited responses remain in the denominator. Single training seed and learning rate; best checkpoint on this evaluation is descriptive, not independent test selection.\n'
    (root/'README.md').write_text(text)
    return result
