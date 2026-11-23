"""Check that the backed-up formal cells satisfy the registered-adapter protocol."""
import argparse
import csv
from collections import defaultdict
import hashlib
import json
import math
import re
from pathlib import Path
import statistics
import struct


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def audit_cell(cell, arm, rate, winner):
    config = read(cell / 'config.json')
    experiment = read(cell / 'experiment.json')
    summary = read(cell / 'summary.json')
    window = read(cell / 'window.json')
    warm_window = read(cell / 'warmup/window.json')
    assert config['arm'] == arm and config['seed'] == 1 and config['rate'] == rate
    assert config['mode'] == 'e2e' and config['adapter_preparation'] == 'registered'
    assert config['text_input_tokens'] == 20480 and config['output_tokens'] == 256
    assert config['warmup'] == 0 and config['duration'] == 60
    assert experiment['tp_size'] == 1 and experiment['max_total_tokens'] == 262144
    assert experiment['swa_full_tokens_ratio'] == 0.5
    server = read(cell / 'get_server_info.json')
    shared_server = {
        'tp_size': 1, 'context_length': 24576, 'max_running_requests': 32,
        'max_total_tokens': 262144, 'swa_full_tokens_ratio': 0.5,
        'disable_radix_cache': True, 'attention_backend': 'triton',
    }
    assert all(server[key] == value for key, value in shared_server.items())
    assert server['enable_soft_prompt'] == (arm == 'sp')
    assert server['enable_lora'] == (arm == 'r1') or (
        arm != 'r1' and server['enable_lora'] is None
    )
    if arm == 'r1':
        assert server['lora_backend'] == winner and server['max_loras_per_batch'] == 32
    model = read(cell / 'model-provenance.json')
    assert model['dtype'] == 'bfloat16'
    provenance = dict(model_repo=model['repo'], model_revision=model['revision'],
                      engine_version=server['version'],
                      engine_patch_sha256=hashlib.sha256((cell / 'engine.patch').read_bytes()).hexdigest())
    devices = read(cell / 'allocated-gpus.json')
    assert len(devices) == 1 and 'H100' in devices[0]['name']
    device_uuid = devices[0]['uuid'].removeprefix('GPU-').lower()
    assert summary['gpu_budget_observed_pass'] and not summary['missing_records']
    assert summary['failed'] == 0
    if arm == 'r1':
        assert config['kernel'] == winner

    traces = [read(cell / name / 'trace.json') for name in ('warmup', '.')]
    requests = [lines(cell / name / 'requests.jsonl') for name in ('warmup', '.')]
    manifests = [read(cell / name / 'adapter-manifest.json') for name in ('warmup', '.')]
    assert len(traces[0]) == len(requests[0]) == 8
    assert all(r['status'] == 'success' and r['output_tokens'] == 256
               and r['finished'] < window['origin'] and not r.get('cleanup_error')
               for r in requests[0])
    assert warm_window['origin'] < window['origin']
    assert len(traces[1]) == len(requests[1]) == summary['offered']
    assert all(0 <= r['arrival_s'] < 60 for r in traces[1])
    assert all(r['status'] in ('success', 'timeout') for r in requests[1])
    assert all(r['output_tokens'] == 256 for r in requests[1] if r['status'] == 'success')

    if arm != 'base':
        copies = defaultdict(list)
        for event in lines(cell / 'engine.loads.jsonl'):
            names = event['name'] if isinstance(event['name'], list) else [event['name']]
            for name in names:
                copies[name].append(event['start_ns'] / 1e9)
        hashes = []
        names = []
        for trace, rows, manifest in zip(traces, requests, manifests):
            assert len(trace) == len(rows) == len(manifest)
            assert [r['asset_index'] for r in trace] == [m['adapter_id'] for m in manifest]
            hashes.extend(m['sha256'] for m in manifest)
            assert all(m['bytes'] > 0 and m['arm'] == arm for m in manifest)
            for row in rows:
                name = row['adapter_name']
                names.append(name)
                assert 'load_start' not in row, 'client issued an adapter load during replay'
                times = copies[name]
                assert all(t >= row.get('generate_sent', math.inf) for t in times)
                if row.get('first_token_received') is not None:
                    assert times and min(times) <= row['first_token_received']
        assert len(set(names)) == len(names)
        assert len(set(hashes)) == len(hashes)
    else:
        assert not any(manifests) and not any(r.get('adapter_name') for r in requests[1])

    measure_ids = {int(r['request_id'][3:]) for r in traces[1]}
    counts = {}
    generated = 0
    for ns, request_id, count in struct.iter_unpack('<QII', (cell / 'engine.tokens.bin').read_bytes()):
        if request_id not in measure_ids:
            assert ns / 1e9 < window['start'], 'warmup produced tokens in the measurement window'
            continue
        if window['start'] <= ns / 1e9 < window['stop']:
            generated += count - counts.get(request_id, 0)
        counts[request_id] = count
    assert generated == summary['output_tokens_in_window']
    assert math.isclose(summary['output_tokens_s'], generated / 60)
    ttft_samples = []
    unsent = censored = 0
    for row in requests[1]:
        submitted = row.get('generate_sent')
        first_token = row.get('first_token_received')
        if submitted is None:
            unsent += 1
            continue
        end = row['finished'] if first_token is None else first_token
        assert end >= submitted
        censored += first_token is None
        ttft_samples.append(end - submitted)
    if len(ttft_samples) > 1:
        p95 = statistics.quantiles(ttft_samples, n=100, method='inclusive')[94]
    else:
        p95 = ttft_samples[0] if ttft_samples else None
    log_text = (cell / 'server.log').read_text()
    retract_events = re.findall(r'Retract requests\. #retracted_reqs: (\d+)', log_text)
    trace_payload = [[(r['arrival_s'], r['asset_index']) for r in trace] for trace in traces]
    input_hashes = [read(cell / name / 'input-hashes.json') for name in ('warmup', '.')]
    signature = hashlib.sha256(json.dumps([trace_payload, input_hashes]).encode()).hexdigest()
    return dict(arm=arm, rate=rate, offered=summary['offered'], incomplete=summary['incomplete'],
                window_output_tokens=generated, output_tokens_s=generated / 60,
                ttft_p95_lower_bound_s=p95, ttft_censored=censored, ttft_not_submitted=unsent,
                max_gpu_gib=summary['peak_gpu_memory_gib'],
                kv_retractions=sum(map(int, retract_events)), gpu_uuid=device_uuid,
                shared_trace_sha256=signature, provenance=provenance)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--plot-csv', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    selection = read(args.root / 'kernel-selection.json')
    scores = {}
    for kernel, row in selection['candidates'].items():
        if 'rejected' in row:
            continue
        cell = args.root / 'screen' / kernel / 'r1-rate-2'
        assert (args.root / f'screen-{kernel}.exit').read_text().strip() == '0'
        summary = read(cell / 'summary.json')
        assert summary['all_arrivals_reached_output_length'] and not summary['failed']
        assert summary['output_tokens_s'] == row['output_tokens_s']['2']
        assert read(cell / 'config.json')['kernel'] == kernel
        scores[kernel] = (row['output_tokens_s']['2'], -row['generate_to_first_token_p95_s']['2'])
    assert scores and selection['winner'] == max(scores, key=scores.get)
    if len(scores) > 1:
        for rate in ('0.5', '2'):
            cells = [args.root / 'screen' / kernel / f'r1-rate-{rate}' for kernel in scores]
            for filename in ('trace.json', 'input-hashes.json', 'adapter-manifest.json'):
                records = [read(cell / filename) for cell in cells]
                assert all(record == records[0] for record in records[1:]), (rate, filename)
    assert (args.root / 'FORMAL-COMPLETE').exists()
    rows = []
    for rate in (0.25, 0.5, 1, 2, 4):
        matched = [audit_cell(args.root / 'seed-1' / f'{arm}-rate-{rate:g}', arm, rate,
                              selection['winner']) for arm in ('base', 'sp', 'r1')]
        assert len({row['shared_trace_sha256'] for row in matched}) == 1
        rows.extend(matched)
    assert all(row['provenance'] == rows[0]['provenance'] for row in rows)
    if args.plot_csv:
        with args.plot_csv.open() as stream:
            plotted = {(r['arm'], float(r['rate'])): r for r in csv.DictReader(stream)}
        assert len(plotted) == len(rows)
        for row in rows:
            point = plotted[row['arm'], row['rate']]
            assert point['ttft_origin'] == 'generate'
            for key in ('offered', 'ttft_censored', 'ttft_not_submitted'):
                assert int(point[key]) == row[key]
            assert math.isclose(float(point['output_tokens_s']), row['output_tokens_s'])
            p95 = row['ttft_p95_lower_bound_s']
            if p95 is None:
                assert point['ttft_p95_lower_bound_s'] == ''
            else:
                assert math.isclose(float(point['ttft_p95_lower_bound_s']), p95)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(selection=selection, formal_cells=rows), indent=2) + '\n')


if __name__ == '__main__':
    main()
