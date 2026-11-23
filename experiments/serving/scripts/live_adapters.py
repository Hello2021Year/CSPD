"""Bounded, arrival-time synthetic adapters backed by sealed shared CPU pages."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import tempfile
import threading
import time

from cpu_store import normalize
from prepare_assets import adapter_config, adapter_seed, make_adapter


class LiveAdapters:
    def __init__(self, experiment, workers=4):
        import torch
        torch.set_num_threads(1)
        self.experiment = experiment
        self.cfg = json.loads((Path(Path('model-path.txt').read_text().strip()) / 'config.json').read_text())
        self.root = Path(tempfile.mkdtemp(prefix='adapter-live-', dir=os.environ.get('SLURM_TMPDIR')))
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='adapter-generator')
        self.lock = threading.Lock()
        self.live = {}
        self.records = {}
        self.peak_bytes = self.peak_modules = 0

    def descriptor(self, arm, index):
        return dict(adapter_id=index, seed=adapter_seed(arm, index), arm=arm,
                    synthetic=True, serving_only=True, materialized=False,
                    sha256=None, bytes=0, path=None,
                    expert_count=self.cfg.get('num_local_experts', 0),
                    expert_lora_format=self.experiment['expert_lora_format'],
                    model_revision=json.loads(Path('model-provenance.json').read_text())['revision'])

    def _prepare(self, arm, index):
        import torch
        from safetensors.torch import save
        started = time.monotonic()
        tensors = make_adapter(self.cfg, arm, index)
        path = self.root / f'{arm}-{index:06d}'
        path.mkdir()
        libc = ctypes.CDLL(None, use_errno=True)
        create = libc.memfd_create
        create.argtypes, create.restype = [ctypes.c_char_p, ctypes.c_uint], ctypes.c_int
        fd = create(b'live-adapter', 2)
        if fd < 0:
            raise OSError(ctypes.get_errno(), 'memfd_create failed')
        try:
            digest, layout = hashlib.sha256(), []
            with os.fdopen(os.dup(fd), 'wb', buffering=8*1024*1024) as stream:
                if arm == 'sp':
                    data = save(tensors)
                    stream.write(data)
                    digest.update(data)
                else:
                    tensors = normalize(tensors)
                    for name, tensor in sorted(tensors.items()):
                        layout.append(dict(name=name, shape=list(tensor.shape),
                                           numel=tensor.numel(), offset=stream.tell()))
                        data = memoryview(tensor.contiguous().view(torch.uint8).numpy()).cast('B')
                        stream.write(data)
                        digest.update(data)
            size = os.fstat(fd).st_size
            fcntl.fcntl(fd, getattr(fcntl, 'F_ADD_SEALS', 1033), 1 | 2 | 4 | 8)
            memfd_path = f'/proc/{os.getpid()}/fd/{fd}'
            if arm != 'sp':
                (path/'adapter_config.json').write_text(json.dumps(adapter_config(self.experiment, self.cfg, arm)))
                record = dict(ready=True, sealed=True, format='gpt_oss_normalized_bf16_v1',
                              hostname=socket.gethostname(),
                              owner_job=os.environ['SLURM_JOB_ID'],
                              memfd_path=memfd_path, bytes=size, layout=layout,
                              adapters={str(path.resolve()): dict(offset=0)})
                (path/'cpu-store-index.json').write_text(json.dumps(record))
            item = dict(self.descriptor(arm, index), materialized=True, sha256=digest.hexdigest(),
                        sha256_format='safetensors' if arm == 'sp' else 'normalized_bf16_tensor_bytes_v1',
                        bytes=size, path=memfd_path if arm == 'sp' else str(path/'adapter_model.safetensors'),
                        local_directory=str(path), worker_started=started, prepared=time.monotonic())
            with self.lock:
                self.live[(arm, index)] = (fd, size, path)
                self.records[(arm, index)] = item
                self.peak_bytes = max(self.peak_bytes, sum(x[1] for x in self.live.values()))
                self.peak_modules = max(self.peak_modules, len(self.live))
            return item
        except BaseException:
            os.close(fd)
            shutil.rmtree(path, ignore_errors=True)
            raise

    async def prepare(self, arm, index):
        future = asyncio.get_running_loop().run_in_executor(self.executor, self._prepare, arm, index)
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            # A cancelled await does not stop a CPU worker. Retain ownership
            # until it completes, then close its pages before propagating timeout.
            try:
                item = await asyncio.shield(future)
                self.release(item)
            finally:
                raise

    def release(self, item):
        with self.lock:
            value = self.live.pop((item['arm'], item['adapter_id']), None)
        if value is not None:
            fd, _, path = value
            os.close(fd)
            shutil.rmtree(path)

    def stats(self):
        with self.lock:
            return dict(peak_live_modules=self.peak_modules, peak_sealed_cpu_bytes=self.peak_bytes,
                        retained_modules=len(self.live), retained_cpu_bytes=sum(x[1] for x in self.live.values()))

    def close(self):
        self.executor.shutdown(wait=True)
        for arm, index in list(self.live):
            self.release(dict(arm=arm, adapter_id=index))
        shutil.rmtree(self.root)
