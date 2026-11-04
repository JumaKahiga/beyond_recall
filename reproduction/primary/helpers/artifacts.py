"""Small provenance helpers; deliberately independent of model libraries."""
from pathlib import Path
import datetime, hashlib, json, os

ROOT = Path(__file__).resolve().parent

def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(tmp, path)

def load(path):
    return json.loads(Path(path).read_text())

def canonical_hash(value):
    return hashlib.sha256((json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False) + '\n').encode()).hexdigest()

def check_recipe(root=ROOT):
    expected=load(root/'project.json')['study_sha256']
    if sha(root/'study.json')!=expected:raise RuntimeError('Study settings changed')
    return load(root/'study.json')


def savez(path, **arrays):
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)
