"""Small, all-resident, unique-adapter bursts; never a steady-state load curve."""
import argparse
import asyncio
import json
import shutil
import struct
import time
from pathlib import Path

import bench
from analyze import summarize, percentile
from live_adapters import LiveAdapters


def split_audit(root, out, rows, tp):
    ids = {int(r['request_id'][3:]) for r in rows}
    names = {r['adapter_name'] for r in rows if r.get('adapter_name')}
    data = (root / 'engine.tokens.bin').read_bytes()
    assert len(data) % 16 == 0
    with (out / 'engine.tokens.bin').open('wb') as f:
        for ns, rid, count in struct.iter_unpack('<QII', data):
            if rid in ids:
                f.write(struct.pack('<QII', ns, rid, count))
    for rank in range(tp):
        name = 'engine.loads.jsonl' if rank == 0 else f'engine.tp{rank}.loads.jsonl'
        if not (root / name).exists():
            assert not names
            continue
        events = [json.loads(s) for s in (root / name).read_text().splitlines()]
        events = [e for e in events if names.intersection(e['name'] if isinstance(e['name'], list) else [e['name']])]
        (out / name).write_text(''.join(json.dumps(e) + '\n' for e in events))
    for name in ['allocated-gpus.json', 'gpu.csv', 'experiment.json']:
        shutil.copyfile(root / name, out / name)


async def run(a):
    root = Path(a.out)
    experiment = json.loads((root / 'experiment.json').read_text())
    plan = dict(kind='finite_resident_bursts', tp_size=1, attention='triton',
                kernel='triton-classic' if a.arm in ('r1', 'r16') else 'none',
                workloads={'1024': [1, 2, 4, 8, 16, 32], '20480': [1, 2, 4, 8]},
                repeats=3, output_tokens=1024, warmup_output_tokens=32,
                sp_fixes_enabled=False, steady_state_capacity=False,
                loading_included=False, max_cpu_modules=32)
    (root / 'fixed-plan.json').write_text(json.dumps(plan, indent=2))
    live = LiveAdapters(experiment, workers=2) if a.arm != 'base' else None
    asset_offset = 0
    round_index = 0
    cells = []
    try:
        for text, levels in plan['workloads'].items():
            for concurrency in levels:
                for repeat in range(4):  # warmup at every batch shape, then three repeats
                    warmup = repeat == 0
                    out = root / f'text{text}-c{concurrency}-' / ('warmup' if warmup else f'repeat{repeat}')
                    out.mkdir(parents=True)
                    shutil.copyfile(root / 'experiment.json', out / 'experiment.json')
                    prepared_at = time.monotonic()
                    items = [] if live is None else await asyncio.gather(*(
                        live.prepare(a.arm, i) for i in range(asset_offset, asset_offset + concurrency)))
                    if items:
                        assert len({m['sha256'] for m in items}) == concurrency
                        (out / f'{a.arm}-manifest.json').write_text(json.dumps([None] * asset_offset + items))
                    args = bench.parser().parse_args([
                        '--arm', a.arm, '--mode', 'resident', '--arrival-pattern', 'burst',
                        '--port', str(a.port), '--assets', str(out), '--out', str(out),
                        '--input-assets', 'assets-text-v2', '--text-input-tokens', text,
                        '--rate', '1', '--warmup', '0', '--duration', '0',
                        '--count', str(concurrency), '--slots', '32', '--seed', str(max(1, repeat)),
                        '--asset-offset', str(asset_offset), '--run-index', str(a.run_index + round_index),
                        '--timeout', '300', '--output-tokens', '32' if warmup else '1024',
                        '--incremental-output', '--unload-after-run',
                        '--kernel', plan['kernel'], '--attention-backend', 'triton'])
                    await bench.run(args)
                    rows = [json.loads(s) for s in (out / 'requests.jsonl').read_text().splitlines()]
                    split_audit(root, out, rows, experiment['tp_size'])
                    summary = summarize(out)
                    audited = [json.loads(s) for s in (out / 'requests-audited.jsonl').read_text().splitlines()]
                    assert len(audited) == concurrency and all(r['status'] == 'success' for r in audited)
                    assert summary['all_arrivals_reached_output_length'] and summary['gpu_budget_observed_pass']
                    assert summary['first_chunk_tokens_max'] == 1
                    assert len({r['scheduled'] for r in audited}) == 1
                    assert all(not r.get('cleanup_error') for r in audited)
                    # The API uses the same monotonic clock on this allocated node.
                    # Server audit, rather than stream-chunk spacing, supplies token endpoints.
                    first = min(r['server_first_token_time'] for r in audited)
                    last = max(r['server_last_token_time'] for r in audited)
                    tpots = [r['server_tpot_s'] * 1000 for r in audited]
                    summary.update(concurrency=concurrency, repeat=repeat, warmup=warmup,
                                   arrival_pattern='burst', loading_included=False,
                                   total_output_tokens_s=summary['finite_output_tokens_s'],
                                   decode_span_tokens_s=sum(r['output_tokens']-1 for r in audited) / (last-first),
                                   server_tpot_p50_ms=percentile(tpots, .5),
                                   server_tpot_p95_ms=percentile(tpots, .95),
                                   summary_path=str(out / 'burst-summary.json'))
                    (out / 'burst-summary.json').write_text(json.dumps(summary, indent=2))
                    if not warmup:
                        cells.append(summary)
                        (root / 'fixed-results.json').write_text(json.dumps(cells, indent=2))
                    if live is not None:
                        for item in items:
                            live.release(item)
                    # No module from any prior burst is reused, including warmups.
                    asset_offset += concurrency
                    round_index += 1
                    print(json.dumps(dict(arm=a.arm, text=text, concurrency=concurrency, repeat=repeat,
                                          tokens_s=summary['total_output_tokens_s'], elapsed=time.monotonic()-prepared_at)), flush=True)
        (root / 'fixed-complete.json').write_text(json.dumps(dict(measured_cells=len(cells), passed=True)))
    finally:
        if live is not None:
            (root / 'fixed-cpu-stats.json').write_text(json.dumps(live.stats(), indent=2))
            live.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--arm', required=True)
    p.add_argument('--run-index', type=int, required=True)
    asyncio.run(run(p.parse_args()))
