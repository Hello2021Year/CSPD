"""Own exactly one server process group inside an existing Slurm allocation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tarfile
import time
import urllib.request

from analyze import summarize
from bench import trace


def serving_context_length(text_tokens, output_tokens):
    # TP worker reserves one position; scheduler reserves another. Use the
    # longest arm (SP256) for ALL arms and round up to a common context capacity.
    required = text_tokens + 256 + output_tokens + 2
    return max(4096, ((required + 4095) // 4096) * 4096)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', choices=['base', 'sp', 'r1', 'r4', 'r16'], required=True)
    p.add_argument('--kernel', default='triton')
    p.add_argument('--mode', default='e2e')
    p.add_argument('--rate', type=float, default=1.)
    p.add_argument('--warmup', type=float, default=60.)
    p.add_argument('--duration', type=float, default=180.)
    p.add_argument('--count', type=int)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--run-index', type=int, default=0)
    p.add_argument('--slots', type=int, default=128)
    p.add_argument('--output-tokens', type=int, default=1024)
    p.add_argument('--text-input-tokens', type=int, choices=[1024, 20480], default=1024)
    p.add_argument('--input-assets')
    p.add_argument('--out', required=True)
    p.add_argument('--timeout', type=float, default=900)
    p.add_argument('--asset-offset', type=int, default=0)
    p.add_argument('--virtual-experts', action='store_true')
    p.add_argument('--attention-backend')
    p.add_argument('--mem-fraction-static', type=float)
    p.add_argument('--max-total-tokens', type=int)
    p.add_argument('--swa-full-tokens-ratio', type=float)
    p.add_argument('--stream-interval', type=int)
    p.add_argument('--logprob-probe-root')
    p.add_argument('--sp-probe', action='store_true')
    p.add_argument('--sp-concurrent-updates', type=int, choices=[0,1])
    p.add_argument('--sp-prefill-graph', type=int, choices=[0,1])
    p.add_argument('--sp-profile', type=int, choices=[0,1])
    p.add_argument('--diagnostic-disable-cuda-graph', action='store_true')
    p.add_argument('--adapter-preparation', choices=['preloaded', 'on-demand', 'bounded-ready', 'registered'], default='preloaded')
    p.add_argument('--generation-workers', type=int, default=4)
    p.add_argument('--preparation-buffer', type=int, default=128)
    p.add_argument('--development-tp-size', type=int, choices=[1, 2])
    p.add_argument('--fixed-concurrency', action='store_true', help='Finite resident bursts on one server; separate diagnostic output')
    p.add_argument('--stop-at-window', action='store_true', help='Censor unfinished requests at window end; server is then shut down')
    p.add_argument('--drained-warmup-count', type=int, default=0)
    p.add_argument('--registered-assets', type=Path)
    p.add_argument('--registered-bursts', nargs='+', type=int)
    p.add_argument('--registered-concurrency', nargs='+', type=int)
    a = p.parse_args()
    registered = a.adapter_preparation == 'registered'
    assert not (a.registered_bursts and a.registered_concurrency)
    levels = a.registered_concurrency or a.registered_bursts
    if registered:
        assert a.registered_assets and a.mode == 'e2e'
        assert a.count is None
        if levels:
            assert not a.drained_warmup_count and a.warmup == 0
            assert all(1 <= n <= a.slots for n in levels)
            catalog_size = max(levels) * (8 if a.registered_concurrency else 1)
        else:
            assert a.drained_warmup_count
            catalog_size = len(trace(a.rate, 0, a.duration, a.seed)) + a.drained_warmup_count
            if catalog_size == a.drained_warmup_count:
                raise ValueError('Measurement trace is empty')
    else:
        assert not levels
    if a.drained_warmup_count:
        assert a.drained_warmup_count > 0 and a.mode == 'e2e' and a.warmup == 0
    if a.fixed_concurrency:
        assert a.mode == 'resident' and a.development_tp_size == 1 and a.slots == 32
        assert a.text_input_tokens == 20480 and a.adapter_preparation == 'preloaded'
    assert os.environ.get('SLURM_JOB_ID'), 'GPU work requires a Slurm allocation'
    if a.stop_at_window:
        assert a.count is None and a.mode == 'e2e' and not a.fixed_concurrency
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'requests.jsonl').exists():
        raise RuntimeError('Refusing to overwrite a previous run')
    model = Path('model-path.txt').read_text().strip()
    experiment = json.loads(Path('experiment.json').read_text())
    experiment['adapter_preparation'] = a.adapter_preparation
    if a.diagnostic_disable_cuda_graph:
        assert a.logprob_probe_root, 'Graph isolation is only for numerical probes'
        experiment.update(development_only=True, diagnostic_disable_cuda_graph=True)
    if a.adapter_preparation in ('on-demand', 'bounded-ready'):
        assert a.mode == 'e2e' and not a.logprob_probe_root
        assert 'UNIQUE_ADAPTER_CPU_STORE' not in os.environ
    if a.development_tp_size:
        assert a.count is not None or a.logprob_probe_root or a.sp_probe or a.stop_at_window or a.drained_warmup_count or levels, 'Development hardware requires an explicit diagnostic mode'
        experiment.update(tp_size=a.development_tp_size, development_only=True)
    if a.attention_backend:
        experiment['attention_backend'] = a.attention_backend
    if a.mem_fraction_static is not None:
        assert 0 < a.mem_fraction_static < 1
        experiment['mem_fraction_static'] = a.mem_fraction_static
    stream_interval = a.stream_interval if a.stream_interval is not None else experiment.get('stream_interval', 1)
    assert stream_interval >= 1
    experiment['stream_interval'] = stream_interval
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    cmd = [sys.executable, '-m', 'sglang.launch_server', '--model-path', model,
           '--tp-size', str(experiment['tp_size']),
           '--host', '127.0.0.1', '--port', str(port), '--dtype', 'bfloat16',
           '--attention-backend', a.attention_backend or experiment['attention_backend'],
           '--mem-fraction-static', str(experiment.get('mem_fraction_static', .7)),
           '--moe-runner-backend', experiment['moe_runner_backend'],
           '--context-length', str(serving_context_length(a.text_input_tokens,a.output_tokens)), '--max-running-requests', str(a.slots),
           '--cuda-graph-max-bs-decode', str(a.slots), '--chunked-prefill-size', '8192',
           '--disable-radix-cache', '--stream-interval', str(stream_interval), '--random-seed', '0',
           '--enable-metrics', '--incremental-streaming-output']
    if a.diagnostic_disable_cuda_graph:
        cmd += ['--disable-cuda-graph']
    if a.max_total_tokens is not None:
        assert a.max_total_tokens > 0
        cmd += ['--max-total-tokens', str(a.max_total_tokens)]
        experiment['max_total_tokens'] = a.max_total_tokens
    if a.swa_full_tokens_ratio is not None:
        assert 0 < a.swa_full_tokens_ratio <= 1
        cmd += ['--swa-full-tokens-ratio', str(a.swa_full_tokens_ratio)]
        experiment['swa_full_tokens_ratio'] = a.swa_full_tokens_ratio
    if a.arm == 'sp':
        cmd += ['--enable-soft-prompt', '--soft-prompt-placeholder-id', str(experiment['placeholder_id']),
                '--max-soft-prompt-len', '256', '--max-soft-prompts', str(catalog_size if registered else a.slots)]
    elif a.arm != 'base':
        cmd += ['--enable-lora', '--max-lora-rank', a.arm[1:],
                '--max-loras-per-batch', str(a.slots), '--lora-backend', a.kernel,
                '--lora-target-modules', 'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']
        cmd += ['--no-experts-shared-outer-loras', '--lora-strict-loading']
        if registered:
            cmd += ['--max-loaded-loras', str(max(a.slots, catalog_size))]
        if a.virtual_experts:
            cmd += ['--lora-use-virtual-experts']
    env = os.environ.copy()
    if registered:
        env['UNIQUE_ADAPTER_REGISTERED'] = '1'
    # Diagnostic overrides are explicit, recorded, and never silently enabled.
    for key, value in (('UNIQUE_ADAPTER_SP_CONCURRENT_UPDATES', a.sp_concurrent_updates),
                       ('UNIQUE_ADAPTER_SP_PREFILL_GRAPH', a.sp_prefill_graph),
                       ('UNIQUE_ADAPTER_SP_PROFILE', a.sp_profile)):
        env[key] = str(value) if value is not None else os.environ.get(key, '0')
    env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
    env['UNIQUE_ADAPTER_AUDIT'] = str(out / 'engine')
    env['UNIQUE_ADAPTER_RESIDENT'] = '1' if a.mode == 'resident' else '0'
    if a.adapter_preparation in ('on-demand', 'bounded-ready') or a.fixed_concurrency:
        env['UNIQUE_ADAPTER_DYNAMIC_STORE'] = '1'
    env['OMP_NUM_THREADS'] = '8'
    env['HF_HUB_OFFLINE'] = '1'
    (out / 'server-command.json').write_text(json.dumps(cmd, indent=2))
    (out / 'runtime-env.json').write_text(json.dumps({k: env[k] for k in (
        'PATH', 'OMP_NUM_THREADS', 'UNIQUE_ADAPTER_AUDIT', 'UNIQUE_ADAPTER_RESIDENT', 'HF_HUB_OFFLINE',
        'UNIQUE_ADAPTER_CPU_STORE', 'UNIQUE_ADAPTER_DYNAMIC_STORE',
        'UNIQUE_ADAPTER_SP_CONCURRENT_UPDATES', 'UNIQUE_ADAPTER_SP_PREFILL_GRAPH',
        'UNIQUE_ADAPTER_SP_PROFILE') if k in env}, indent=2))
    (out / 'experiment.json').write_text(json.dumps(experiment, indent=2))
    (out / 'model-provenance.json').write_text(Path('model-provenance.json').read_text())
    devices = json.loads(subprocess.check_output([sys.executable, '-c',
        'import torch,json; print(json.dumps([dict(uuid=str(torch.cuda.get_device_properties(i).uuid), name=torch.cuda.get_device_properties(i).name, total_memory_bytes=torch.cuda.get_device_properties(i).total_memory) for i in range(torch.cuda.device_count())]))'], text=True))
    assert len(devices) == experiment['tp_size'], 'Allocated GPU count must equal the serving TP size'
    if a.development_tp_size:
        experiment['gpu_budget_gib'] = min(d['total_memory_bytes'] // 2**30 for d in devices)
        (out / 'experiment.json').write_text(json.dumps(experiment, indent=2))
    else:
        assert all('B200' in d['name'] for d in devices), devices
    gpu_ids = ','.join(d['uuid'] if d['uuid'].startswith('GPU-') else 'GPU-'+d['uuid'] for d in devices)
    (out / 'allocated-gpus.json').write_text(json.dumps(devices, indent=2))
    (out / 'hardware.txt').write_text(subprocess.check_output(['nvidia-smi','-i',gpu_ids,'-q'], text=True))
    (out / 'slurm-job.txt').write_text(subprocess.check_output(['scontrol','show','job',os.environ['SLURM_JOB_ID']], text=True))
    head = subprocess.check_output(['git', '-C', 'engine', 'rev-parse', 'HEAD'], text=True).strip()
    patch = subprocess.check_output(['git', '-C', 'engine', 'diff', '--binary'])
    (out / 'engine.patch').write_bytes(patch)
    modified = subprocess.check_output(['git', '-C', 'engine', 'ls-files', '-m', '-o', '--exclude-standard', '-z']).decode().split('\0')
    sources = [Path('engine')/name for name in modified if name]
    for pattern in ('*.py', '*.slurm', '*.sh'):
        sources += sorted(Path('scripts').glob(pattern))
    sources += [Path('experiment.json'), Path('PROTOCOL.md')]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources if path.is_file()}
    (out / 'source-hashes.json').write_text(json.dumps(dict(engine_commit=head, files=hashes), indent=2))
    with tarfile.open(out / 'source-snapshot.tar.gz', 'w:gz') as archive:
        for path in hashes:
            archive.add(path, arcname=path)
    (out / 'engine-source.txt').write_text(head + ' + engine.patch and untracked files in source-snapshot.tar.gz\n')
    log = open(out / 'server.log', 'w')
    server = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    telem = open(out / 'gpu.csv', 'w')
    monitor = subprocess.Popen(['nvidia-smi','-i',gpu_ids,'--query-gpu=timestamp,uuid,memory.used,utilization.gpu,power.draw,clocks.sm,clocks.mem','--format=csv','-l','1'], stdout=telem)
    process_monitor = subprocess.Popen([sys.executable, 'scripts/monitor_processes.py',
        '--parent', str(os.getpid()), '--out', str(out / 'processes.jsonl')])
    try:
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError(f'Server exited {server.returncode}; inspect {out}/server.log')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            time.sleep(2)
        else:
            raise TimeoutError('Server startup timeout')
        for route in ['get_server_info', 'get_model_info']:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/' + route) as r:
                (out / (route + '.json')).write_bytes(r.read())
        if a.diagnostic_disable_cuda_graph:
            graph = json.loads((out / 'get_server_info.json').read_text())['cuda_graph_config']
            assert all(graph[phase]['backend'] == 'disabled' for phase in ('prefill', 'decode')), graph
        if a.logprob_probe_root:
            subprocess.run([sys.executable, 'scripts/logprob_probe.py',
                '--reference-root', a.logprob_probe_root, '--arm', a.arm,
                '--port', str(port), '--out', str(out),
                '--run-index', str(a.run_index)], check=True)
            return  # Numerical diagnostic only; finally still cleans the server.
        if a.sp_probe:
            assert a.arm == 'sp' and a.text_input_tokens == 20480
            subprocess.run([sys.executable, 'scripts/sp_numerical_probe.py',
                '--port', str(port), '--out', str(out)], check=True)
            return  # Numerical checks never become performance summaries.
        if a.fixed_concurrency:
            subprocess.run([sys.executable, 'scripts/fixed_concurrency.py',
                '--port', str(port), '--out', str(out), '--arm', a.arm,
                '--run-index', str(a.run_index)], check=True)
            return  # The finite-burst driver writes one audited summary per cell.
        measurement_warmup = a.warmup
        asset_offset = a.asset_offset
        if a.mode == 'resident' and a.warmup > 0:
            warm_rate = 1.5
            warm_count = len(trace(warm_rate, 0, a.warmup, a.seed))
            if warm_count > a.slots:
                raise ValueError('Diagnostic warmup must fit all-resident slots')
            warm = [sys.executable, 'scripts/bench.py', '--arm', a.arm, '--mode', 'resident',
                    '--port', str(port), '--rate', str(warm_rate), '--warmup', '0',
                    '--duration', str(a.warmup), '--seed', str(a.seed),
                    '--run-index', str(a.run_index+2000), '--slots', str(a.slots),
                    '--out', str(out / 'warmup'), '--output-tokens', str(a.output_tokens),
                    '--timeout', str(a.timeout), '--asset-offset', str(a.asset_offset),
                    '--unload-after-run', '--incremental-output']
            warm += ['--text-input-tokens', str(a.text_input_tokens)]
            if a.input_assets:
                warm += ['--input-assets', a.input_assets]
            subprocess.run(warm, check=True)
            asset_offset += a.slots
            measurement_warmup = 0
        client = [sys.executable, 'scripts/bench.py', '--arm', a.arm, '--mode', a.mode,
                  '--kernel', (a.kernel + ('+virtual' if a.virtual_experts else '')) if a.arm.startswith('r') else 'none',
                  '--attention-backend', experiment['attention_backend'],
                  '--port', str(port),
                  '--rate', str(a.rate), '--warmup', str(measurement_warmup), '--duration', str(a.duration),
                  '--seed', str(a.seed), '--run-index', str(a.run_index), '--slots', str(a.slots),
                  '--out', str(out), '--output-tokens', str(a.output_tokens), '--timeout', str(a.timeout),
                  '--asset-offset', str(asset_offset), '--incremental-output']
        client += ['--adapter-preparation', a.adapter_preparation,
                   '--generation-workers', str(a.generation_workers),
                   '--preparation-buffer', str(a.preparation_buffer),
                   '--text-input-tokens', str(a.text_input_tokens)]
        if a.input_assets:
            client += ['--input-assets', a.input_assets]
        if registered:
            client += ['--assets', str(a.registered_assets)]
        if a.count is not None:
            client += ['--count', str(a.count), '--record-token-ids']
        if a.stop_at_window:
            client += ['--stop-at-window']
        if levels:
            for index, concurrency in enumerate(levels):
                stages = ('warmup', 'measurement') if a.registered_concurrency else (
                    'warmup', 'repeat-1', 'repeat-2', 'repeat-3')
                for repeat, stage in enumerate(stages):
                    sustained = stage == 'measurement'
                    count = 8 * concurrency if sustained else concurrency
                    burst_out = out / f'concurrency-{concurrency}' / stage
                    burst = client.copy()
                    values = {
                        '--out': str(burst_out), '--asset-offset': str(asset_offset),
                        '--run-index': str(a.run_index + index * 4 + repeat),
                    }
                    for flag, value in values.items():
                        burst[burst.index(flag) + 1] = value
                    if sustained:
                        burst[burst.index('--slots') + 1] = str(concurrency)
                    burst += ['--count', str(count), '--arrival-pattern', 'closed-loop' if sustained else 'burst',
                              '--unload-after-run']
                    subprocess.run(burst, check=True, env=env)
                    rows = [json.loads(line) for line in (burst_out / 'requests.jsonl').read_text().splitlines()]
                    assert len(rows) == count and all(
                        row['status'] == 'success' and row['output_tokens'] == a.output_tokens
                        and not row.get('cleanup_error') and row.get('num_retractions') == 0
                        and row.get('prefill_finished_time', 0) > row.get('forward_entry_time', float('inf'))
                        for row in rows), burst_out
                    asset_offset += count
            return
        if a.drained_warmup_count:
            warm = client.copy()
            warm_out = out / 'warmup'
            for flag, value in {
                '--out': str(warm_out), '--run-index': str(a.run_index + 2000),
                '--asset-offset': str(a.asset_offset + (catalog_size - a.drained_warmup_count if registered else 1000000)), '--rate': '2',
            }.items():
                warm[warm.index(flag) + 1] = value
            warm = [option for option in warm if option != '--stop-at-window']
            warm += ['--count', str(a.drained_warmup_count)]
            if registered:
                warm += ['--unload-after-run']
            subprocess.run(warm, check=True, env=env)
            rows = [json.loads(line) for line in
                    (warm_out / 'requests.jsonl').read_text().splitlines()]
            if len(rows) != a.drained_warmup_count or any(
                row['status'] != 'success' or row['output_tokens'] != a.output_tokens
                or row.get('cleanup_error') for row in rows
            ):
                raise RuntimeError('Warmup did not complete and unload every request')
        subprocess.run(client, check=True, env=env)
        # Successful request completion flushes the token audit. Partial requests
        # may still execute; this delay also records post-timeout drain separately.
        time.sleep(2)
    finally:
        shutdown = dict(started=time.monotonic(), forced_kill=False)
        try:
            os.killpg(server.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            shutdown['forced_kill'] = True
            os.killpg(server.pid, signal.SIGKILL)
            server.wait()
        shutdown.update(finished=time.monotonic(), returncode=server.returncode)
        (out / 'shutdown.json').write_text(json.dumps(shutdown, indent=2))
        monitor.terminate()
        monitor.wait()
        process_monitor.terminate()
        process_monitor.wait()
        log.close()
        telem.close()
    summary = summarize(out)
    print(json.dumps(summary, indent=2), flush=True)
    if a.adapter_preparation == 'bounded-ready':
        rows = [json.loads(line) for line in (out/'requests.jsonl').read_text().splitlines()]
        if (not summary.get('cpu_preparation_valid', True)
                or any(r['status'] not in ('success','timeout') or r.get('cleanup_error') for r in rows)):
            raise RuntimeError('CPU preparation or transport failed; this run is not serving evidence')
    if a.count is not None:
        rows = [json.loads(line) for line in (out/'requests.jsonl').read_text().splitlines()]
        if (len(rows) != a.count or any(r['status'] != 'success' for r in rows)
                or not summary['all_arrivals_reached_output_length']
                or not summary['gpu_budget_observed_pass']):
            raise RuntimeError('Finite diagnostic failed; inspect retained request logs and summary')


if __name__ == '__main__':
    main()
