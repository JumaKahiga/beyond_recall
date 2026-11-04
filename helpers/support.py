"""Execution provenance, atomic artifacts, and frozen audit reference."""
from pathlib import Path
import hashlib, json, os, platform, sys
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parent
import audit as core
import numpy as np
import scipy, sklearn
from threadpoolctl import threadpool_info, threadpool_limits
threadpool_limits(1)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def array_hash(a):return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()
def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n');os.replace(tmp,path)
def savez(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
    os.replace(tmp,path)
def loadz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def runtime():
    return {'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
            'scikit_learn':sklearn.__version__,'threadpools':[{k:p.get(k) for k in ['user_api','internal_api','version','num_threads','threading_layer']} for p in threadpool_info()]}
def validate_runtime():
    r=runtime();expected=json.loads((ROOT/'STUDY_MANIFEST.json').read_text())['runtime']
    if any(p['num_threads']!=1 for p in r['threadpools']):raise RuntimeError('Thread limit mismatch')
    return r

def load_inputs(target,data_root=None):
    from dataclasses import dataclass
    @dataclass
    class Dataset:
        window_ids: list
        embeddings: np.ndarray
        labels: np.ndarray
    m=json.loads((ROOT/'STUDY_MANIFEST.json').read_text())
    for name,digest in m['inputs'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Instantiated input changed: '+name)
    pool=loadz(ROOT/'data/constructed/pool.npz')
    data=loadz(ROOT/'data/embeddings/SSW_EMBEDDINGS.npz')
    ids=pool['window_ids'].tolist()
    if data['window_ids'].tolist()!=ids:raise RuntimeError('Embedding ID alignment mismatch')
    codes=pool['target_codes'].tolist()
    if target not in codes:raise RuntimeError('Target not frozen')
    labels=pool['labels'][:,codes.index(target)].astype(np.int8)
    X=data['embeddings']
    if X.shape!=(m['pool']['N'],1536) or X.dtype!=np.float32 or not np.isfinite(X).all():raise RuntimeError('Invalid embeddings')
    if len(set(ids))!=len(ids) or not set(labels.tolist())<={0,1}:raise RuntimeError('Invalid label authority')
    return Dataset(ids,X,labels),{'target':target,'N':len(ids),'P':int(labels.sum()),'embedding_sha256':sha(ROOT/'data/embeddings/SSW_EMBEDDINGS.npz')}
