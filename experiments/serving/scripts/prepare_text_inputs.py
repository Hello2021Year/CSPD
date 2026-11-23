"""Tokenize pinned LongBench-v2 contexts into paired exact-length serving inputs.

These are context excerpts for performance measurement, not LongBench quality
evaluation: no answers are scored and no synthetic repetition pads short text.
"""
import argparse
import hashlib
import json
from pathlib import Path

REVISION = '2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9'
REPO = 'zai-org/LongBench-v2'
SOURCE_SHA256 = '15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2'


def main():
    import tokenizers
    from tokenizers import Tokenizer
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, default=Path('data/longbench-v2/data.json'))
    p.add_argument('--out', type=Path, default=Path('assets-text-v2'))
    p.add_argument('--model', default=None)
    a = p.parse_args()
    model = a.model or Path('model-path.txt').read_text().strip()
    tokenizer_path = Path(model)/'tokenizer.json'
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    assert hashlib.sha256(a.source.read_bytes()).hexdigest() == SOURCE_SHA256
    data = json.loads(a.source.read_text())
    corpora = {1024: [], 20480: []}
    sources = []
    seen = set()
    for row in sorted(data, key=lambda r: str(r['_id'])):
        # Only document-based tasks; omit code repositories and synthetic tables.
        if row['domain'] not in ('Single-Document QA', 'Multi-Document QA'):
            continue
        ids = tokenizer.encode(row['context'], add_special_tokens=False).ids
        if len(ids) < 20480:
            continue
        long = ids[:20480]
        assert 200018 not in long
        key = hashlib.sha256(json.dumps(long).encode()).hexdigest()
        if key in seen:
            continue  # Multiple questions can share one document context.
        seen.add(key)
        corpora[20480].append(long)
        corpora[1024].append(long[:1024])
        sources.append(dict(id=row['_id'], domain=row['domain'], sub_domain=row.get('sub_domain'),
                            original_context_tokens=len(ids), token_start=0, token_end=20480))
    assert len(sources) >= 32, f'Too few eligible document contexts: {len(sources)}'
    a.out.mkdir(parents=True, exist_ok=True)
    manifest = dict(dataset=REPO, revision=REVISION, license='apache-2.0',
                    source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),
                    tokenizer_model=model, model_revision=json.loads(Path('model-provenance.json').read_text())['revision'],
                    tokenizer_sha256=hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
                    tokenizer_library_version=tokenizers.__version__,
                    use='Serving performance only; deterministic excerpts, not task-quality evaluation.',
                    assignment='request asset_index modulo corpus size; identical across arms and seeds',
                    sources=sources, files={})
    for length, bank in corpora.items():
        target = a.out/f'inputs-{length}.json'
        target.write_text(json.dumps(bank))
        manifest['files'][str(length)] = dict(sha256=hashlib.sha256(target.read_bytes()).hexdigest(), count=len(bank))
    (a.out/'input-corpus.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(contexts=len(sources), lengths=list(corpora), out=str(a.out))), flush=True)


if __name__ == '__main__':
    main()
