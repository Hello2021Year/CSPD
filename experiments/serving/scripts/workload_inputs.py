"""Load a fixed real-text corpus without replicating long inputs per request."""
import hashlib
import json
from pathlib import Path


def load_inputs(root, indices, length):
    root = Path(root)
    corpus = root / f'inputs-{length}.json'
    if not corpus.exists():
        if length != 1024:
            raise FileNotFoundError(f'Missing real-text corpus: {corpus}')
        return [json.loads((root / f'input-{i:06d}.json').read_text()) for i in indices], None
    provenance = root/'input-corpus.json'
    manifest = json.loads(provenance.read_text())
    assert manifest['files'][str(length)]['sha256'] == hashlib.sha256(corpus.read_bytes()).hexdigest()
    bank = json.loads(corpus.read_text())
    assert bank and all(len(ids) == length for ids in bank)
    # Repeated text is allowed; adapters remain unique. Prefix cache is disabled.
    inputs = [bank[i % len(bank)] for i in indices]
    return inputs, dict(manifest, provenance_sha256=hashlib.sha256(provenance.read_bytes()).hexdigest(),
                        request_text_indices=[i % len(bank) for i in indices])
