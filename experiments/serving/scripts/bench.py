"""Open-loop unique-adapter client. Engine audit supplies exact window throughput."""
import argparse
import asyncio
import hashlib
import json
import random
import time
from pathlib import Path


def trace(rate, warmup, duration, seed, count=None):
    rng = random.Random(seed)
    now = 0.
    result = []
    while True:
        now += rng.expovariate(rate)
        if count is None and now >= warmup + duration:
            break
        result.append(dict(index=len(result), arrival_s=now,
                           phase='warmup' if now < warmup else 'measurement'))
        if count is not None and len(result) >= count:
            break
    return result


def make_connector():
    import aiohttp
    # Idle connections can expire server-side while the event loop handles a
    # large completion/load burst. Never reuse such a socket for a POST: retrying
    # generation or adapter mutation would have ambiguous execution semantics.
    return aiohttp.TCPConnector(limit=0, force_close=True)


def make_arrivals(a):
    if a.arrival_pattern == 'closed-loop':
        if a.adapter_preparation != 'registered' or a.warmup or not a.count or a.count < a.slots:
            raise ValueError('Closed-loop measurement needs registered mode and count >= slots')
        return [dict(index=i, arrival_s=0., phase='measurement') for i in range(a.count)]
    if getattr(a, 'arrival_pattern', 'poisson') == 'burst':
        if (a.mode != 'resident' and a.adapter_preparation != 'registered'
                or a.warmup != 0 or not a.count or a.count > a.slots):
            raise ValueError('A finite burst needs resident or registered adapters, zero warmup, and count <= slots')
        return [dict(index=i, arrival_s=0., phase='measurement') for i in range(a.count)]
    return trace(a.rate, a.warmup, a.duration, a.seed, a.count)


async def run(a):
    import aiohttp
    resolved_config = Path(a.out) / 'experiment.json'
    experiment = json.loads((resolved_config if resolved_config.exists() else Path('experiment.json')).read_text())
    registered = a.adapter_preparation == 'registered'
    bounded = a.adapter_preparation == 'bounded-ready'
    on_demand = a.adapter_preparation in ('on-demand', 'bounded-ready')
    if on_demand and a.mode != 'e2e':
        raise ValueError('Arrival-time preparation is only for the E2E experiment')
    root, out = Path(a.assets or experiment['assets']), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    arrivals = make_arrivals(a)
    if getattr(a, 'stop_at_window', False) and (a.count is not None or a.mode != 'e2e'):
        raise ValueError('Window cutoff requires a time-bounded E2E arrival trace')
    if a.mode == 'resident' and a.warmup:
        raise ValueError('Resident diagnostic requires separate warmup adapters; use warmup=0 for smoke only.')
    for row in arrivals:
        row['asset_index'] = a.asset_offset + row['index']
        row['request_id'] = f'ub-{a.run_index * 1000000 + row["index"]}'
        row['adapter_name'] = f'a{a.run_index}-{row["index"]:06d}' if a.arm != 'base' else None
    (out / 'trace.json').write_text(json.dumps(arrivals, indent=2))
    run_config = dict(vars(a), tp_size=experiment['tp_size'], http_connection_reuse=False)
    (out / 'config.json').write_text(json.dumps(run_config, indent=2))
    from workload_inputs import load_inputs
    inputs, corpus = load_inputs(Path(a.input_assets or root), [r['asset_index'] for r in arrivals], a.text_input_tokens)
    if corpus:
        (out/'input-corpus.json').write_text(json.dumps(corpus, indent=2))
    if bounded:
        assert corpus, 'Formal CPU-ready runs require the pinned real-text corpus'
    assert all(len(ids) == a.text_input_tokens for ids in inputs), 'Unexpected text input length'
    live = None
    ready = None
    if on_demand and a.arm != 'base':
        from live_adapters import LiveAdapters
        live = LiveAdapters(experiment, workers=a.generation_workers)
        manifest = [live.descriptor(a.arm, r['asset_index']) for r in arrivals]
        if bounded:
            from prepared_adapters import PreparedAdapters
            ready = PreparedAdapters(live, a.arm, [r['asset_index'] for r in arrivals],
                                     capacity=a.slots+a.preparation_buffer, workers=a.generation_workers)
            try:
                await ready.start()
            except BaseException:
                await ready.close()
                live.close()
                raise
    else:
        manifest = [] if a.arm == 'base' else json.loads((root / f'{a.arm}-manifest.json').read_text())[a.asset_offset:a.asset_offset+len(arrivals)]
    if a.arm != 'base':
        assert len(manifest) == len(arrivals)
        assert len({m['seed'] if on_demand else m['sha256'] for m in manifest}) == len(manifest)
    (out / 'adapter-manifest.json').write_text(json.dumps(manifest, indent=2))
    (out / 'input-hashes.json').write_text(json.dumps([hashlib.sha256(json.dumps(ids).encode()).hexdigest() for ids in inputs]))
    base = f'http://127.0.0.1:{a.port}'
    connector = make_connector()
    async with aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(total=a.timeout)) as session:
        async def post(route, data):
            async with session.post(base + route, json=data) as response:
                body = await response.json()
                if response.status != 200 or body.get('success') is False:
                    raise RuntimeError(f'{route}: {response.status}: {str(body)[:500]}')
                return body

        for row, item in ([] if on_demand else zip(arrivals, manifest)):
            if a.arm == 'sp':
                await post('/load_soft_prompt', dict(name=row['adapter_name'], path=item['path'], cpu_only=a.mode == 'e2e'))
            else:
                await post('/load_lora_adapter', dict(lora_name=row['adapter_name'], lora_path=str(Path(item['path']).parent)))
        # Only the initial CPU buffer precedes the origin in bounded-ready mode.
        # Offers remain open-loop; admission and GPU loading are timed below.
        origin = time.monotonic()
        start, stop = origin + a.warmup, origin + a.warmup + a.duration
        window = dict(origin=origin, start=start, stop=stop, clock='monotonic',
                      throughput_definition='server committed tokens in [start, stop)')
        (out / 'window.json').write_text(json.dumps(window, indent=2))
        slots = asyncio.Semaphore(a.slots)
        counters = dict(offered=0, completed=0, external_waiting=0, active=0)
        log = open(out / 'requests.jsonl', 'w', buffering=1)
        queue = open(out / 'queue.jsonl', 'w', buffering=1)
        done_sampling = asyncio.Event()
        results = []

        async def sample_queue():
            while not done_sampling.is_set():
                sample = dict(t=time.monotonic(), **counters)
                if live is not None:
                    sample.update(live.stats())
                try:
                    async with session.get(base + '/metrics', timeout=aiohttp.ClientTimeout(total=2)) as response:
                        lines = (await response.text()).splitlines()
                    sample['server_gauges'] = [line for line in lines if line.startswith((
                        'sglang:num_running_reqs', 'sglang:num_queue_reqs', 'sglang:token_usage'))]
                except Exception as exc:
                    sample['metrics_error'] = repr(exc)
                queue.write(json.dumps(sample) + '\n')
                try:
                    await asyncio.wait_for(done_sampling.wait(), timeout=1.)
                except asyncio.TimeoutError:
                    pass

        async def request(row):
            idx = row['index']
            scheduled = origin + row['arrival_s']
            result = dict(row, scheduled=scheduled, output_tokens=0, status='unfinished')
            acquired = loaded = waiting = False
            load_attempted = False
            prepared_item = None
            first = last = None
            output_ids = None
            try:
                await asyncio.sleep(max(0., scheduled-time.monotonic()))
                counters['offered'] += 1
                result['dispatch'] = time.monotonic()
                result['dispatch_lag_s'] = result['dispatch'] - scheduled
                remaining = a.timeout - result['dispatch_lag_s']
                if getattr(a, 'stop_at_window', False):
                    remaining = min(remaining, stop-time.monotonic())
                async with asyncio.timeout(max(.001, remaining)):
                    if on_demand or (a.arm == 'sp' and a.mode == 'e2e' and not registered):
                        counters['external_waiting'] += 1
                        waiting = True
                        await slots.acquire()
                        acquired = True
                        counters['external_waiting'] -= 1
                        waiting = False
                        result['admitted'] = time.monotonic()
                    if live is not None:
                        if ready is not None:
                            prepared_item = ready.take(row['asset_index'])
                            result['prepare_start'] = prepared_item['worker_started']
                            result['prepare_end'] = prepared_item['prepared']
                            assert result['prepare_end'] <= result['admitted']
                        else:
                            result['prepare_start'] = time.monotonic()
                            prepared_item = await live.prepare(a.arm, row['asset_index'])
                            result['prepare_end'] = time.monotonic()
                        manifest[idx] = prepared_item
                        result['preparation_s'] = result['prepare_end'] - result['prepare_start']
                    if a.arm != 'base' and (on_demand or (a.arm == 'sp' and a.mode == 'e2e' and not registered)):
                        result['load_start'] = time.monotonic()
                        load_attempted = True
                        if a.arm == 'sp':
                            await post('/load_soft_prompt', dict(name=row['adapter_name'], path=manifest[idx]['path']))
                        else:
                            await post('/load_lora_adapter', dict(lora_name=row['adapter_name'], lora_path=str(Path(manifest[idx]['path']).parent)))
                        loaded = True
                        result['load_end'] = time.monotonic()
                    payload = dict(input_ids=([experiment['placeholder_id']]*256 if a.arm == 'sp' else []) + inputs[idx],
                                   sampling_params=dict(max_new_tokens=a.output_tokens, min_new_tokens=a.output_tokens,
                                                        ignore_eos=True, temperature=0.),
                                   stream=True, rid=row['request_id'])
                    if a.arm == 'sp':
                        payload['soft_prompt'] = row['adapter_name']
                    elif a.arm != 'base':
                        payload['lora_path'] = row['adapter_name']
                    result['generate_sent'] = time.monotonic()
                    counters['active'] += 1
                    async with session.post(base + '/generate', json=payload) as response:
                        if response.status != 200:
                            raise RuntimeError(f'generate {response.status}: {(await response.text())[:500]}')
                        async for line in response.content:
                            if not line.startswith(b'data:'):
                                continue
                            raw = line[5:].strip()
                            if raw == b'[DONE]':
                                break
                            event = json.loads(raw)
                            if 'error' in event:
                                raise RuntimeError(str(event['error']))
                            meta = event.get('meta_info', {})
                            if 'prompt_tokens' in meta:
                                result['server_prompt_tokens'] = meta['prompt_tokens']
                            for key in ('forward_entry_time', 'prefill_finished_time', 'num_retractions'):
                                if key in meta:
                                    result[key] = meta[key]
                            if 'output_ids' in event:
                                if a.incremental_output:
                                    if output_ids is None:
                                        output_ids = []
                                    output_ids.extend(event['output_ids'])
                                else:
                                    output_ids = event['output_ids']
                            count = meta.get('completion_tokens', 0)
                            if count > result['output_tokens']:
                                now = time.monotonic()
                                if first is None:
                                    first = now
                                    result['first_chunk_tokens'] = count
                                last = now
                                result['output_tokens'] = count
                            result['finish_reason'] = meta.get('finish_reason')
                    result['status'] = 'success' if result['output_tokens'] == a.output_tokens else 'incomplete'
                    if result['status'] == 'success':
                        expected_prompt = a.text_input_tokens + (256 if a.arm == 'sp' else 0)
                        if result.get('server_prompt_tokens') != expected_prompt:
                            raise ValueError(f'Server prompt length differs from expected {expected_prompt}')
            except TimeoutError:
                result['status'] = 'timeout'
                if getattr(a, 'stop_at_window', False) and time.monotonic() >= stop:
                    result['censored_at_window_end'] = True
            except Exception as exc:
                result['status'] = 'failed'
                result['error'] = repr(exc)
            finally:
                result['finished'] = time.monotonic()
                if waiting:
                    counters['external_waiting'] -= 1
                if output_ids is not None:
                    result['output_ids_sha256'] = hashlib.sha256(json.dumps(output_ids).encode()).hexdigest()
                    if a.record_token_ids:
                        result['output_ids'] = output_ids
                result['first_token_received'] = first
                result['last_token_received'] = last
                result['ttft_s'] = None if first is None else first - scheduled
                result['tpot_s'] = None if first is None or result['output_tokens'] < 2 else (last-first)/(result['output_tokens']-1)
                result['slo_met'] = (result['status'] == 'success' and result['ttft_s'] <= 2 and result['tpot_s'] <= .1)
                if 'generate_sent' in result:
                    counters['active'] -= 1
                counters['completed'] += 1
                # A timed-out/failed server call may still own its adapter. Keep
                # that slot and CPU pages until shutdown; never reuse it early.
                if loaded and result['status'] == 'success':
                    try:
                        if a.arm == 'sp':
                            await post('/unload_soft_prompt', dict(name=row['adapter_name']))
                        else:
                            await post('/unload_lora_adapter', dict(lora_name=row['adapter_name']))
                        loaded = False
                    except Exception as exc:
                        result['cleanup_error'] = repr(exc)
                safe_release = ((result['status'] == 'success' and not loaded) or
                                (not load_attempted and 'generate_sent' not in result))
                if live is not None and prepared_item is not None and safe_release:
                    if ready is not None:
                        ready.release(prepared_item)
                    else:
                        live.release(prepared_item)
                    result['adapter_released'] = time.monotonic()
                if acquired and safe_release:
                    slots.release()
                if ready is not None:
                    ready.skip(row['asset_index'])
                log.write(json.dumps(result) + '\n')
                results.append(result)
            return result

        sampler = asyncio.create_task(sample_queue())
        if a.arrival_pattern == 'closed-loop':
            pending = iter(arrivals)

            async def worker():
                for row in pending:
                    result = await request(dict(row, arrival_s=time.monotonic() - origin))
                    if result['status'] != 'success':
                        break

            await asyncio.gather(*(worker() for _ in range(a.slots)))
        else:
            await asyncio.gather(*(request(r) for r in arrivals))
        if a.arrival_pattern in ('burst', 'closed-loop'):
            stop = max(r['finished'] for r in results)
            window.update(stop=stop, throughput_definition='committed tokens / request makespan; excludes preload')
        else:
            await asyncio.sleep(max(0., stop-time.monotonic()))
        done_sampling.set()
        await sampler
        log.close()
        queue.close()
        window['drain_s'] = max(0., max((r['finished'] for r in results), default=stop)-stop)
        window['finite_elapsed_s'] = max((r['finished'] for r in results), default=origin)-min((r['scheduled'] for r in results), default=origin)
        window['cleanup_end'] = time.monotonic()
        (out / 'window.json').write_text(json.dumps(window, indent=2))
        if live is not None:
            if ready is not None:
                await ready.close()
                (out/'preparation-queue.json').write_text(json.dumps(dict(
                    capacity=ready.capacity, lookahead_slots=a.preparation_buffer, workers=a.generation_workers,
                    starvation_count=ready.misses, valid=ready.misses == 0), indent=2))
            manifest = [live.records.get((a.arm, r['asset_index']), item) for r, item in zip(arrivals, manifest)]
            (out / 'adapter-manifest.json').write_text(json.dumps(manifest, indent=2))
            (out / 'live-adapters.json').write_text(json.dumps(live.stats(), indent=2))
            live.close()
        if a.unload_after_run and a.arm != 'base':
            if any(r['status'] != 'success' for r in results):
                raise RuntimeError('Warmup has incomplete requests; cannot safely unload/reuse GPU slots')
            for row in arrivals:
                if a.arm == 'sp':
                    if a.mode == 'resident' or registered:
                        await post('/unload_soft_prompt', dict(name=row['adapter_name']))
                else:
                    await post('/unload_lora_adapter', dict(lora_name=row['adapter_name']))


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', choices=['base','sp','r1','r4','r16'], required=True)
    p.add_argument('--kernel', default='none')
    p.add_argument('--attention-backend', default='triton')
    p.add_argument('--mode', choices=['e2e','resident'], default='e2e')
    p.add_argument('--port', type=int, default=31000)
    p.add_argument('--assets')
    p.add_argument('--out', required=True)
    p.add_argument('--rate', type=float, required=True)
    p.add_argument('--warmup', type=float, default=60)
    p.add_argument('--duration', type=float, default=180)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--run-index', type=int, default=0)
    p.add_argument('--asset-offset', type=int, default=0)
    p.add_argument('--unload-after-run', action='store_true')
    p.add_argument('--slots', type=int, default=128)
    p.add_argument('--count', type=int)
    p.add_argument('--timeout', type=float, default=900)
    p.add_argument('--output-tokens', type=int, default=1024)
    p.add_argument('--text-input-tokens', type=int, default=1024)
    p.add_argument('--record-token-ids', action='store_true')
    p.add_argument('--incremental-output', action='store_true')
    p.add_argument('--adapter-preparation', choices=['preloaded', 'on-demand', 'bounded-ready', 'registered'], default='preloaded')
    p.add_argument('--generation-workers', type=int, default=4)
    p.add_argument('--preparation-buffer', type=int, default=128)
    p.add_argument('--input-assets')
    p.add_argument('--arrival-pattern', choices=['poisson', 'burst', 'closed-loop'], default='poisson')
    p.add_argument('--stop-at-window', action='store_true')
    return p


if __name__ == '__main__':
    asyncio.run(run(parser().parse_args()))
