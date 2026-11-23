"""Build a derived M2RL-SFT checkpoint whose embedding table carries the soft prompt.

The composed prompt (k x 256 rows) is appended to ``model.embed_tokens.weight`` as new
vocabulary rows. Requests then reference these rows by token id, so any engine that takes
``prompt_token_ids`` (vLLM, SGLang) runs the composition without input-embedding support.
Every original tensor byte is preserved; the appended rows are masked from the output.
"""
import copy
import hashlib
import json
from pathlib import Path
import struct


def extend_embedding_shard(source, destination, appended_bf16, *, rows, hidden=2560):
    """Preserve every source tensor byte; append only to model.embed_tokens.weight.

    Validates contiguous offsets, commits no shared source modification, and
    verifies every copied byte again from the new shard. Caller publishes it
    only after updating and validating a separate derived config and index.
    """
    source, destination = Path(source), Path(destination)
    if destination.exists() or source.resolve() == destination.resolve():
        raise ValueError('Fresh independent destination required')
    if type(rows) is not int or rows <= 0 or len(appended_bf16) != rows*hidden*2:
        raise ValueError('Wrong appended BF16 shape/byte count')
    with source.open('rb') as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise ValueError('Missing safetensors header')
        header_size = struct.unpack('<Q', prefix)[0]
        if not 0 < header_size <= 100_000_000:
            raise ValueError('Invalid header length')
        raw_header = f.read(header_size)
        header = json.loads(raw_header)
        data_start = 8+header_size
        entries = sorted((v['data_offsets'][0], k, v) for k, v in header.items() if k != '__metadata__')
        cursor = 0
        for start, key, value in entries:
            end = value['data_offsets'][1]
            if start != cursor or end < start:
                raise ValueError('Source offsets must be contiguous and non-overlapping')
            cursor = end
        if data_start+cursor != source.stat().st_size:
            raise ValueError('Source tensor data size differs')
        name = 'model.embed_tokens.weight'
        if name not in header:
            raise ValueError('Embedding tensor missing')
        embed = header[name]
        if (embed['dtype'] != 'BF16' or len(embed['shape']) != 2
                or embed['shape'][1] != hidden
                or embed['data_offsets'][1]-embed['data_offsets'][0] != embed['shape'][0]*hidden*2):
            raise ValueError('Embedding format/shape differs')
        original_vocab = embed['shape'][0]
        new = copy.deepcopy(header)
        cursor = 0
        for start, key, value in entries:
            size = value['data_offsets'][1]-start
            extra = len(appended_bf16) if key == name else 0
            new[key]['data_offsets'] = [cursor, cursor+size+extra]
            if key == name:
                new[key]['shape'][0] += rows
            cursor += size+extra
        encoded = json.dumps(new, separators=(',', ':'), ensure_ascii=False).encode()
        encoded += b' '*((-len(encoded)) % 8)
        expected = {}
        with destination.open('xb') as out:
            out.write(struct.pack('<Q', len(encoded)))
            out.write(encoded)
            for start, key, value in entries:
                remaining = value['data_offsets'][1]-start
                f.seek(data_start+start)
                digest = hashlib.sha256()
                while remaining:
                    block = f.read(min(4*1024*1024, remaining))
                    if not block:
                        raise ValueError('Source ended inside tensor')
                    out.write(block)
                    digest.update(block)
                    remaining -= len(block)
                if key == name:
                    out.write(appended_bf16)
                expected[key] = digest.hexdigest()
            out.flush()
            import os
            os.fsync(out.fileno())
    with destination.open('rb') as f:
        new_start = 8+len(encoded)
        for start, key, value in entries:
            f.seek(new_start+new[key]['data_offsets'][0])
            remaining = value['data_offsets'][1]-start
            digest = hashlib.sha256()
            while remaining:
                block = f.read(min(4*1024*1024, remaining))
                if not block:
                    raise ValueError('Destination ended inside tensor')
                digest.update(block)
                remaining -= len(block)
            if digest.hexdigest() != expected[key]:
                raise ValueError('Original tensor bytes changed during extension')
            if key == name and f.read(len(appended_bf16)) != appended_bf16:
                raise ValueError('Appended prompt bytes changed')
        if destination.stat().st_size != new_start+cursor:
            raise ValueError('Extended shard size differs')
    # Use the actual library parser, independently of this header writer.
    from safetensors import safe_open
    with safe_open(str(destination), framework='pt', device='cpu') as handle:
        if set(handle.keys()) != set(expected):
            raise ValueError('Library parser tensor inventory differs')
        for key in handle.keys():
            if handle.get_slice(key).get_shape() != new[key]['shape']:
                raise ValueError('Library parser tensor shape differs')
    return dict(original_vocab_size=original_vocab, extended_vocab_size=original_vocab+rows,
                added_tensor_bytes=len(appended_bf16), original_tensor_sha256=expected,
                appended_rows_sha256=hashlib.sha256(appended_bf16).hexdigest())


ORIGINAL_VOCAB = 151936
HIDDEN = 2560


def build_derived_model(source, destination, prompt):
    """Copy ``source`` (a local M2RL-SFT snapshot) to ``destination`` with ``prompt`` appended.

    Returns the list of virtual token ids, in prompt-row order.
    """
    import os
    import shutil
    import torch

    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    config = json.loads((source / "config.json").read_text())
    index = json.loads((source / "model.safetensors.index.json").read_text())
    shard = index["weight_map"]["model.embed_tokens.weight"]
    rows = prompt.to(torch.bfloat16).contiguous()
    proof = extend_embedding_shard(source / shard, destination / shard,
                                   rows.view(torch.uint8).numpy().tobytes(), rows=rows.shape[0], hidden=HIDDEN)
    config["vocab_size"] = proof["extended_vocab_size"]
    index["metadata"]["total_size"] += proof["added_tensor_bytes"]
    (destination / "config.json").write_text(json.dumps(config, indent=2))
    (destination / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    for item in source.iterdir():
        if item.name in (shard, "config.json", "model.safetensors.index.json"):
            continue
        if item.suffix == ".safetensors":
            os.symlink(item.resolve(), destination / item.name)
        elif item.is_file():
            shutil.copyfile(item, destination / item.name)
    return list(range(proof["original_vocab_size"], proof["extended_vocab_size"]))
