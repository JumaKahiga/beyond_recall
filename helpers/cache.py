"""Persistent scientific inputs, with no machine paths in saved metadata."""
from pathlib import Path
import argparse, json, os, shutil, tempfile, zipfile
from artifacts import ROOT, load, dump, sha, canonical_hash

CONSTRUCTED=['RECORDING_PERMUTATION.json','RECORDING_SELECTION.json','ANNOTATION_PREFLIGHT.json','CONSTRUCTION_MANIFEST.json',
             'CONSTRUCTION_ARTIFACT_HASHES.json','pool.npz','label_construction.npz']
OPTIONAL_CONSTRUCTED=['AUDIO_RECOVERY_PROVENANCE.json']
SOURCE=['annotations.csv','species.csv','recording_location.txt']

def cache_key(root):
    return canonical_hash({name:sha(root/name) for name in ['study.json','models.json','construction.py','embedding.py']})

def files(root, complete):
    selected=[root/'data/constructed'/name for name in CONSTRUCTED]
    selected += [root/'data/constructed'/name for name in OPTIONAL_CONSTRUCTED if (root/'data/constructed'/name).exists()]
    selected += [root/'data/source'/name for name in SOURCE]
    for name in ['EMBEDDING_EXECUTION_MANIFEST.json','CONSTRUCTION_REPORT.md']:
        if (root/name).exists():selected.append(root/name)
    for record in load(root/'data/constructed/CONSTRUCTION_MANIFEST.json')['recordings']:
        p=root/'data/embeddings'/(record['filename']+'.json')
        if p.exists():selected += [p,p.with_suffix('.npz')]
    if complete:selected += [root/'data/embeddings/SSW_EMBEDDINGS.npz',root/'data/embeddings/EMBEDDING_RECEIPT.json']
    return {str(p.relative_to(root)):sha(p) for p in selected}

def valid_name(name):
    p=Path(name)
    return (not p.is_absolute() and '..' not in p.parts and
            (name in ['EMBEDDING_EXECUTION_MANIFEST.json','CONSTRUCTION_REPORT.md'] or
             (len(p.parts)==3 and ((p.parts[:2]==('data','constructed') and p.name in CONSTRUCTED+OPTIONAL_CONSTRUCTED) or
              (p.parts[:2]==('data','source') and p.name in SOURCE) or
              (p.parts[:2]==('data','embeddings') and (p.name in ['SSW_EMBEDDINGS.npz','EMBEDDING_RECEIPT.json'] or
               (p.name.startswith('SSW_') and (p.name.endswith('.flac.npz') or p.name.endswith('.flac.json')))))))))

def verify(root,complete=True,arrays=False):
    c=load(root/'data/constructed/CONSTRUCTION_MANIFEST.json')
    if c['status']!='TASKS_CONSTRUCTED' or c['recipe_freeze_sha256']!=sha(root/'study.json'):
        raise RuntimeError('Cached dataset construction differs from study settings')
    for name,digest in load(root/'data/constructed/CONSTRUCTION_ARTIFACT_HASHES.json')['files'].items():
        if Path(name).name!=name or sha(root/'data/constructed'/name)!=digest:raise RuntimeError('Construction checksum mismatch')
    for name,digest in c['source_sha256'].items():
        if not valid_name(name) or sha(root/name)!=digest:raise RuntimeError('Construction source checksum mismatch')
    m=load(root/'EMBEDDING_EXECUTION_MANIFEST.json');spec=load(root/'models.json')[m['device']]
    if m['model_id']!=spec['id'] or m['model_files']!=spec['model_files'] or m['source_sha256']!=sha(root/'embedding.py'):
        raise RuntimeError('Embedding model or preprocessing differs')
    provenance={'embedding_manifest_sha256':sha(root/'EMBEDDING_EXECUTION_MANIFEST.json'),
                'pool_sha256':sha(root/'data/constructed/pool.npz'),'runtime':m['runtime']}
    for record in c['recordings']:
        name=record['filename']
        if Path(name).name!=name:raise RuntimeError('Unsafe recording name')
        p=root/'data/embeddings'/(name+'.json')
        if not p.exists() and not complete:continue
        r=load(p)
        if r['provenance']!=provenance or sha(p.with_suffix('.npz'))!=r['sha256'] or r['audio_sha256']!=record['audio']['sha256']:
            raise RuntimeError('Recording embedding checksum mismatch')
    if complete:
        r=load(root/'data/embeddings/EMBEDDING_RECEIPT.json')
        if r['provenance']!=provenance or sha(root/'data/embeddings/SSW_EMBEDDINGS.npz')!=r['sha256']:
            raise RuntimeError('Complete embedding checksum mismatch')
    if arrays:
        import numpy as np
        with np.load(root/'data/constructed/pool.npz',allow_pickle=False) as z:
            ids=z['window_ids'];labels=z['labels'];codes=z['target_codes'];ri=z['recording_index']
        with np.load(root/'data/constructed/label_construction.npz',allow_pickle=False) as z:
            raw=z['raw_ids'];unknown=z['unknown_exclusion'];zero=z['zero_duration_exclusion'];common=z['common_exclusion']
            if any(a.shape!=raw.shape or a.dtype!=np.bool_ for a in [unknown,zero,common]):raise RuntimeError('Invalid exclusion masks')
            if not np.array_equal(common,unknown|zero) or not np.array_equal(ids,raw[~common]):raise RuntimeError('Common exclusion mask differs from pool')
        counts={'raw_complete_windows':len(raw),'unknown_windows_removed':int(unknown.sum()),
                'zero_duration_timestamp_windows':int(zero.sum()),'zero_duration_windows_already_unknown':int((zero&unknown).sum()),
                'zero_duration_additional_windows_removed':int((zero&~unknown).sum()),'common_windows_removed':int(common.sum()),
                'N':len(ids),'B':len(ids)//4,'K':len(ids)//10}
        if any(c[k]!=v for k,v in counts.items()):raise RuntimeError('Window exclusion accounting differs')
        with np.load(root/'data/embeddings/SSW_EMBEDDINGS.npz',allow_pickle=False) as z:
            X=z['embeddings'];same=np.array_equal(ids,z['window_ids'])
        if not same or X.shape!=(c['N'],1536) or X.dtype!=np.float32 or not np.isfinite(X).all():raise RuntimeError('Invalid embedding array')
        if canonical_hash(ids.tolist())!=c['ordered_pool_id_sha256']:raise RuntimeError('Pool ID order mismatch')
        if codes.tolist()!=[t['code'] for t in c['selected_targets']] or labels.shape!=(c['N'],c['J']) or not np.isin(labels,[0,1]).all():
            raise RuntimeError('Target labels differ')
        for j,t in enumerate(c['selected_targets']):
            if int(labels[:,j].sum())!=t['positive_windows']:raise RuntimeError('Target positive count differs')
        for i,record in enumerate(c['recordings']):
            with np.load(root/'data/embeddings'/(record['filename']+'.npz'),allow_pickle=False) as z:
                if not np.array_equal(z['window_ids'],ids[ri==i]) or not np.array_equal(z['embeddings'],X[ri==i]):
                    raise RuntimeError('Recording cache and assembled embeddings differ')

def copy_checked(source,target,digest):
    if target.exists() and sha(target)==digest:return
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_name(target.name+'.tmp')
    shutil.copyfile(source,temporary)
    if sha(temporary)!=digest:raise RuntimeError('Storage transfer checksum mismatch')
    os.replace(temporary,target)

def save(root,directory):
    complete=(root/'data/embeddings/EMBEDDING_RECEIPT.json').exists()
    verify(root,complete);folder=directory/cache_key(root)
    inventory=files(root,complete)
    for name,digest in inventory.items():copy_checked(root/name,folder/'objects'/digest,digest)
    record={'files':inventory,'complete':complete,'key':cache_key(root)}
    name=canonical_hash(record)+'.json';dump(folder/'snapshots'/name,record)
    dump(folder/('complete.json' if complete else 'partial.json'),{'snapshot':name})
    print('CACHE_SAVED', 'complete' if complete else 'partial',flush=True)

def restore(root,directory,force=False):
    request=canonical_hash({'directory':str(directory.resolve()),'force':force,'key':cache_key(root)})
    if (root/'CACHE_STATUS.json').exists():
        previous=load(root/'CACHE_STATUS.json')
        if previous.get('request_sha256')!=request:raise RuntimeError('Cache settings changed in an existing run; use a new run directory')
        if previous['status']=='complete':verify(root,True)
        return previous
    result={'status':'bypassed' if force else 'missing'};folder=directory/cache_key(root)
    pointer=folder/'complete.json'
    if not pointer.exists():pointer=folder/'partial.json'
    if not force and pointer.exists():
        name=load(pointer)['snapshot']
        if Path(name).name!=name:raise RuntimeError('Unsafe snapshot name')
        record=load(folder/'snapshots'/name)
        if canonical_hash(record)+'.json'!=name or record['key']!=cache_key(root):raise RuntimeError('Cache manifest differs')
        for path,digest in record['files'].items():
            if not valid_name(path) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise RuntimeError('Unsafe cached file')
            if sha(folder/'objects'/digest)!=digest:raise RuntimeError('Cached file checksum mismatch')
        for path,digest in record['files'].items():copy_checked(folder/'objects'/digest,root/path,digest)
        verify(root,record['complete'])
        result={'status':'complete' if record['complete'] else 'partial','snapshot_sha256':name[:-5]}
    result['request_sha256']=request
    dump(root/'CACHE_STATUS.json',result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['restore','verify']);p.add_argument('--directory',type=Path)
    p.add_argument('--regenerate',action='store_true');a=p.parse_args()
    if a.action=='restore':print(restore(ROOT,a.directory,a.regenerate))
    else:verify(ROOT,True,True);print('INPUTS_VERIFIED')
if __name__=='__main__':main()
