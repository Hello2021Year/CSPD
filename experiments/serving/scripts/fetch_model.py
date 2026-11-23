"""Resolve and download the model selected by the experiment configuration."""
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

cfg = json.loads(Path('experiment.json').read_text())
info = HfApi().model_info(cfg['model_repo'], revision=cfg['model_revision'])
provenance = dict(repo=cfg['model_repo'], revision=info.sha, dtype=cfg['weight_dtype'])
Path('model-provenance.json').write_text(json.dumps(provenance, indent=2))
p = snapshot_download(cfg['model_repo'], revision=info.sha,
                      allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja'])
Path('model-path.txt').write_text(p + '\n')
provenance['snapshot_path'] = p
Path('model-provenance.json').write_text(json.dumps(provenance, indent=2))
print(json.dumps(provenance), flush=True)
