"""Recompute from all offered requests and engine token events, including failures."""
import argparse
import csv
import json
import math
import struct
from pathlib import Path


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    x = (len(values)-1)*q
    lo, hi = math.floor(x), math.ceil(x)
    return values[lo] + (values[hi]-values[lo])*(x-lo)


def summarize(path):
    path = Path(path)
    cfg = json.loads((path / 'config.json').read_text())
    window = json.loads((path / 'window.json').read_text())
    rows = [json.loads(s) for s in (path / 'requests.jsonl').read_text().splitlines()]
    trace = json.loads((path / 'trace.json').read_text())
    assert len({r['request_id'] for r in trace}) == len(trace)
    assert len({r['request_id'] for r in rows}) == len(rows)
    assert {r['request_id'] for r in rows} <= {r['request_id'] for r in trace}
    start, stop = window['start'], window['stop']
    adapter_names = {r.get('adapter_name') for r in trace if r.get('adapter_name')}
    if cfg.get('adapter_preparation') == 'on-demand':
        for r in rows:
            if 'prepare_start' in r:
                assert r['prepare_start'] >= r['scheduled'], 'Future CPU preparation detected'
        manifest = json.loads((path / 'adapter-manifest.json').read_text())
        schedule = {r['asset_index']: window['origin']+r['arrival_s'] for r in trace}
        for item in manifest:
            if item.get('materialized'):
                assert item['worker_started'] >= schedule[item['adapter_id']], 'Future CPU generation detected'
    if cfg.get('adapter_preparation') == 'bounded-ready':
        for r in rows:
            if 'load_start' in r:
                assert r['prepare_end'] <= r['admitted'] <= r['load_start']
                assert r['load_start'] >= r['scheduled'], 'Future adapter load detected'
    for rank in range(cfg.get('tp_size', 1)):
        load_path = path / ('engine.loads.jsonl' if rank == 0 else f'engine.tp{rank}.loads.jsonl')
        if not load_path.exists():
            if adapter_names:
                raise ValueError(f'Missing GPU load audit for TP rank {rank}')
            continue
        load_events = [json.loads(line) for line in load_path.read_text().splitlines()]
        by_name = {r.get('adapter_name'): window['origin']+r['arrival_s'] for r in trace if r.get('adapter_name')}
        if cfg.get('adapter_preparation') == 'registered':
            by_name = dict.fromkeys(by_name, math.inf)
            by_name.update({r['adapter_name']: r['generate_sent'] for r in rows
                            if r.get('adapter_name') and 'generate_sent' in r})
        seen = set()
        for event in load_events:
            names = event['name'] if isinstance(event['name'], list) else [event['name']]
            for name in names:
                if name not in by_name:
                    continue
                seen.add(name)
                if cfg['mode'] == 'e2e' and event['start_ns']/1e9 < by_name[name]:
                    raise ValueError(f'Future GPU prefetch detected: {name}')
                if cfg['mode'] == 'resident' and event['end_ns']/1e9 > window['origin']:
                    raise ValueError(f'Diagnostic adapter was not resident: {name}')
        for r in rows:
            if r['status'] == 'success' and r.get('adapter_name') and r['adapter_name'] not in seen:
                raise ValueError(f'Missing GPU load audit: {r["adapter_name"]}')
    counts, first_last = {}, {}
    generated = drain = 0
    data = (path / 'engine.tokens.bin').read_bytes()
    if len(data) % 16:
        raise ValueError('Truncated engine token audit')
    for ns, rid, n in struct.iter_unpack('<QII', data):
        t = ns/1e9
        delta = n-counts.get(rid, 0)
        if delta <= 0:
            raise ValueError('Nonmonotonic output count')
        counts[rid] = n
        times = first_last.setdefault(rid, [t, t])
        times[1] = t
        if start <= t < stop:
            generated += delta
        elif t >= stop:
            drain += delta
    for r in rows:
        rid = int(r['request_id'][3:])
        r['server_output_tokens'] = counts.get(rid, 0)
        if rid in first_last:
            first, last = first_last[rid]
            r['server_tpot_s'] = (last-first)/(counts[rid]-1) if counts[rid] > 1 else None
            r['server_first_token_time'], r['server_last_token_time'] = first, last
            received_first, received_last = r.get('first_token_received'), r.get('last_token_received')
            r['first_token_delivery_lag_s'] = received_first-first if received_first is not None else None
            # Only compare timestamps for the same final output token. A timed
            # out client may stop reading while the server continues generating.
            r['last_token_delivery_lag_s'] = (
                received_last-last if received_last is not None and
                r.get('output_tokens') == counts[rid] else None)
        if r['status'] == 'success' and r['server_output_tokens'] != cfg['output_tokens']:
            raise ValueError(f'Missing server events for successful request {rid}')
    (path / 'requests-audited.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    measured = [r for r in rows if r['phase'] == 'measurement']
    offered = sum(r['phase'] == 'measurement' for r in trace)
    successful = [r for r in measured if r['status'] == 'success']
    good = sum(r['slo_met'] for r in measured)
    # A client timeout does not prove the server drained its outstanding work.
    # Include warmup arrivals here, but exclude separate resident-warmup IDs.
    all_ids = {int(r['request_id'][3:]) for r in trace}
    completed_ids = {rid for rid in all_ids if counts.get(rid, 0) == cfg['output_tokens']}
    last_server_token = max((first_last[rid][1] for rid in all_ids if rid in first_last), default=stop)
    # Client percentiles include observed partial/failed requests where defined;
    # undefined values and failures are separately counted, never silently dropped.
    result = dict(arm=cfg['arm'], mode=cfg['mode'], rate=cfg['rate'], seed=cfg['seed'],
                  tp_size=cfg.get('tp_size', 1), kernel=cfg.get('kernel', 'unknown'),
                  attention_backend=cfg.get('attention_backend', 'unknown'),
                  offered=offered, records=len(measured), succeeded=len(successful),
                  failed=sum(r['status']=='failed' for r in measured),
                  timeouts=sum(r['status']=='timeout' for r in measured),
                  missing_records=offered-len(measured),
                  incomplete=offered-len(successful),
                  output_tokens_in_window=generated,
                  output_tokens_s=generated/(stop-start), drain_tokens=drain,
                  drain_s=window['drain_s'],
                  server_token_drain_s=max(0., last_server_token-stop),
                  server_completed_all_arrivals=len(completed_ids),
                  server_incomplete_all_arrivals=len(all_ids-completed_ids),
                  all_arrivals_reached_output_length=completed_ids == all_ids,
                  ttft_p95_s=percentile([r['ttft_s'] for r in measured if r['ttft_s'] is not None], .95),
                  tpot_p95_ms=percentile([1000*r['tpot_s'] for r in measured if r['tpot_s'] is not None], .95),
                  undefined_ttft=offered-len(measured)+sum(r['ttft_s'] is None for r in measured),
                  undefined_tpot=offered-len(measured)+sum(r['tpot_s'] is None for r in measured),
                  slo_goodput_rps=good/(stop-start), slo_met=good,
                  slo_fraction=good/offered if offered else 0,
                  dispatch_lag_p95_s=percentile([r['dispatch_lag_s'] for r in measured], .95))
    first_chunks = [r['first_chunk_tokens'] for r in measured if 'first_chunk_tokens' in r]
    result['first_chunk_tokens_max'] = max(first_chunks, default=None)
    result['multi_token_first_chunks'] = sum(n > 1 for n in first_chunks)
    for kind in ('first', 'last'):
        field = f'{kind}_token_delivery_lag_s'
        result[field.replace('_s', '_p95_s')] = percentile(
            [r[field] for r in measured if r.get(field) is not None], .95)
    shutdown_path = path / 'shutdown.json'
    if shutdown_path.exists():
        shutdown = json.loads(shutdown_path.read_text())
        result['server_shutdown_after_window_s'] = max(0., shutdown['started']-stop)
        result['server_forced_kill'] = shutdown['forced_kill']
    if cfg['mode'] == 'resident':
        result['finite_elapsed_s'] = window['finite_elapsed_s']
        result['finite_output_tokens_s'] = sum(r['server_output_tokens'] for r in measured) / window['finite_elapsed_s']
        result['capacity_interpretation'] = 'finite request diagnostic; not steady-state capacity'
    gpu_ids = path / 'allocated-gpus.json'
    if gpu_ids.exists():
        devices = json.loads(gpu_ids.read_text())
        allowed = {d['uuid'].removeprefix('GPU-') for d in devices}
        peaks = {uuid: 0. for uuid in allowed}
        with open(path / 'gpu.csv') as f:
            for record in csv.DictReader(f, skipinitialspace=True):
                uuid = record['uuid'].strip().removeprefix('GPU-')
                if uuid not in allowed:
                    raise ValueError('GPU telemetry includes a device outside the allocation')
                mem = float(record['memory.used [MiB]'].split()[0]) / 1024
                peaks[uuid] = max(peaks[uuid], mem)
        assert all(peaks.values()), 'Missing GPU telemetry'
        result['peak_gpu_memory_gib'] = max(peaks.values())
        result['sum_per_gpu_peaks_gib'] = sum(peaks.values())
        result['gpu_memory_sampling_interval_s'] = 1
        budget = json.loads((path / 'experiment.json').read_text())['gpu_budget_gib']
        result['gpu_budget_gib_per_device'] = budget
        result['gpu_budget_observed_pass'] = max(peaks.values()) <= budget
    store = path / 'cpu-store-index.json'
    if store.exists():
        result['shared_cpu_catalogue_bytes'] = json.loads(store.read_text())['bytes']
    result['adapter_preparation'] = cfg.get('adapter_preparation', 'preloaded')
    result['text_input_tokens'] = cfg.get('text_input_tokens', 1024)
    result['preparation_p95_s'] = percentile([r['preparation_s'] for r in measured if 'preparation_s' in r], .95)
    result['loading_p95_s'] = percentile([r['load_end']-r['load_start'] for r in measured if 'load_end' in r], .95)
    if cfg.get('adapter_preparation') == 'bounded-ready' and cfg['arm'] != 'base':
        preparation = json.loads((path/'preparation-queue.json').read_text())
        result['preparation_starvation_count'] = preparation['starvation_count']
        result['cpu_preparation_valid'] = preparation['valid']
    live = path / 'live-adapters.json'
    if live.exists():
        stats = json.loads(live.read_text())
        bound = cfg['slots'] + (cfg['preparation_buffer'] if cfg.get('adapter_preparation') == 'bounded-ready' else 0)
        assert stats['peak_live_modules'] <= bound, 'CPU adapter live set exceeded its bound'
        result.update(stats)
    (path / 'summary.json').write_text(json.dumps(result, indent=2))
    with open(path / 'summary.csv', 'w') as f:
        w = csv.DictWriter(f, fieldnames=list(result))
        w.writeheader()
        w.writerow(result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('run')
    a = p.parse_args()
    print(json.dumps(summarize(a.run), indent=2))
