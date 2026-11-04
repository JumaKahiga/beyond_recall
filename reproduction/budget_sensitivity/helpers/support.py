"""Scientific hashes and atomic output; runtime paths stay in the local session."""
from pathlib import Path
import hashlib, json, os, platform
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parent
import numpy as np
import scipy, sklearn
from threadpoolctl import threadpool_info, threadpool_limits
threadpool_limits(1)
import audit as core

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def load(path):return json.loads(Path(path).read_text())
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    os.replace(temp,path)
def savez(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    with temp.open('wb') as f:np.savez_compressed(f,**arrays)
    os.replace(temp,path)
def loadz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def array_hash(a):return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()
def runtime():
    return {'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
            'scikit_learn':sklearn.__version__,
            'threadpools':[{k:p.get(k) for k in ['internal_api','version','num_threads']} for p in threadpool_info()]}
def validate_runtime():
    expected=load(ROOT/'experiment.json')['runtime'];actual=runtime()
    for key in ['python','numpy','scipy','scikit_learn']:
        if actual[key]!=expected[key]:raise RuntimeError('Pinned analysis runtime differs: '+key)
    if any(p['num_threads']!=1 for p in actual['threadpools']):raise RuntimeError('Expected one numerical thread')
    return actual
def check_sources():
    project=load(ROOT/'project.json')
    for name,digest in project['execution_files'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Project source changed: '+name)
    return sha(ROOT/'project.json')
def config():return load(ROOT/'session.json')
def store():return Path(config()['output_directory'])/sha(ROOT/'experiment.json')[:16]/config()['mode']

