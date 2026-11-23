"""Two-token numerical probes at the first observed greedy divergence.

Each adapter is requested once per server run. This is not a performance run:
input histories are teacher-forced and logprobs are collected. Step 1 exercises
decode; it is compared only if step 0 produced the same token in both runs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reference-root', type=Path, required=True)
    p.add_argument('--arm', choices=['r1', 'r16'], required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--run-index', type=int, required=True)
    a = p.parse_args()
    refdir = a.reference_root / f'{a.arm}-triton-resident'
    ref = {r['index']: r for r in map(json.loads, (refdir/'requests.jsonl').read_text().splitlines())}
    variants = [{r['index']: r for r in map(json.loads, path.read_text().splitlines())}
                for path in sorted(a.reference_root.glob(f'{a.arm}-*/requests.jsonl'))]
    manifest = json.loads((refdir/'adapter-manifest.json').read_text())
    assets = Path(json.loads(Path('experiment.json').read_text())['assets'])
    probes = []
    for i, row in sorted(ref.items()):
        output = row['output_ids']
        differences = [j for variant in variants for j, (x, y) in
                       enumerate(zip(output, variant[i]['output_ids'])) if x != y]
        position = min(differences) if differences else 1
        ids = json.loads((assets/f'input-{row["asset_index"]:06d}.json').read_text())
        if position > 0:
            ids += output[:position-1]
        probes.append(dict(index=i, input_ids=ids, compare_step=int(position > 0),
                           original_position=position,
                           token_ids=sorted({v[i]['output_ids'][position] for v in variants})))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out/'probes.json').write_text(json.dumps(probes, indent=2))

    def post(route, payload):
        request = urllib.request.Request(f'http://127.0.0.1:{a.port}/{route}',
            data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.load(response)
        if result.get('success') is False or 'error' in result:
            raise RuntimeError(str(result))
        return result

    for probe in probes:
        i = probe['index']
        post('load_lora_adapter', dict(lora_name=f'lp-{a.run_index}-{i}',
             lora_path=str(Path(manifest[i]['path']).parent)))
    results = []
    for probe in probes:
        i, ids = probe['index'], probe['input_ids']
        response = post('generate', dict(input_ids=ids,
            lora_path=f'lp-{a.run_index}-{i}', rid=f'lp-{a.run_index}-{i}',
            sampling_params=dict(max_new_tokens=2, min_new_tokens=2, ignore_eos=True, temperature=0.),
            stream=False, return_logprob=True, logprob_start_len=len(ids)-1,
            top_logprobs_num=32, token_ids_logprob=probe['token_ids']))
        assert len(response['output_ids']) == response['meta_info']['completion_tokens'] == 2
        results.append(dict(index=i, compare_step=probe['compare_step'],
            input_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(), response=response))
        (a.out/'logprob-results.json').write_text(json.dumps(results, indent=2))
        print(json.dumps(dict(index=i, original_position=probe['original_position'],
                              output_ids=response['output_ids'])), flush=True)


if __name__ == '__main__':
    main()
