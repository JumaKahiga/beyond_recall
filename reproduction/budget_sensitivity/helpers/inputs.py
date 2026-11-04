"""Read the existing Drive snapshot or original results archive, without downloads."""
from pathlib import Path, PurePosixPath
import io, json, os, shutil, zipfile
from support import ROOT, np, core, sha, load, dump, config, check_sources

def safe_name(name):
    p=PurePosixPath(name)
    if not name or '\\' in name or p.is_absolute() or '..' in p.parts or str(p)!=name:
        raise ValueError('Unsafe relative artifact name')
    return name

class Packet:
    """Read data only. Never execute code from an uploaded results packet."""
    def __init__(self,path):
        self.path=Path(path);self.zip=None;ref=load(ROOT/'reference_25.json')
        if self.path.is_dir():
            if (self.path/'ssw_results/manifest.json').is_file():self.path=self.path/'ssw_results'
            manifest=self.path/'manifest.json'
            if not manifest.is_file() or sha(manifest)!=ref['packet_manifest_sha256']:
                raise RuntimeError('Unpacked results manifest does not match the completed comparison')
            self.files=load(manifest)['files']
        else:
            if not self.path.is_file() or sha(self.path)!=ref['archive_sha256']:
                raise RuntimeError('Original results ZIP is missing or its checksum differs')
            self.zip=zipfile.ZipFile(self.path)
            names=self.zip.namelist()
            if len(names)!=len(set(names)):raise RuntimeError('Duplicate archive entries')
            for name in names:safe_name(name.rstrip('/'))
            raw=self.zip.read('ssw_results/manifest.json')
            import hashlib
            if hashlib.sha256(raw).hexdigest()!=ref['packet_manifest_sha256']:raise RuntimeError('Archive manifest mismatch')
            self.files=json.loads(raw)['files']
    def read(self,name):
        import hashlib
        safe_name(name)
        if name not in self.files:raise RuntimeError('Unlisted results artifact: '+name)
        if self.zip is not None:raw=self.zip.read('ssw_results/'+name)
        else:
            path=(self.path/name).resolve()
            if self.path.resolve() not in path.parents:raise RuntimeError('Results link escapes its folder')
            raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=self.files[name]:raise RuntimeError('Results artifact checksum mismatch: '+name)
        return raw
    def json(self,name):return json.loads(self.read(name))
    def arrays(self,name):
        with np.load(io.BytesIO(self.read(name)),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
    def copy(self,name,destination):
        # Stream large embeddings; avoid holding compressed bytes alongside the array.
        safe_name(name);destination=Path(destination);digest=self.files[name]
        if destination.exists() and sha(destination)==digest:return
        destination.parent.mkdir(parents=True,exist_ok=True);temp=destination.with_name(destination.name+'.tmp')
        if not self.zip and self.path.resolve() not in (self.path/name).resolve().parents:
            raise RuntimeError('Results link escapes its folder')
        source=self.zip.open('ssw_results/'+name) if self.zip else (self.path/name).open('rb')
        with source as incoming, temp.open('wb') as outgoing:shutil.copyfileobj(incoming,outgoing,8*1024*1024)
        if sha(temp)!=digest:raise RuntimeError('Results copy checksum mismatch')
        os.replace(temp,destination)
    def close(self):
        if self.zip:self.zip.close()

def find_packet(required=False):
    settings=config();supplied=settings['results_path'].strip()
    if supplied:return Packet(supplied)
    digest=load(ROOT/'reference_25.json')['archive_sha256']
    candidate=Path(settings['drive_cache_path'])/'result_exports'/('ssw_results_'+digest+'.zip')
    if candidate.is_file():return Packet(candidate)
    if required:raise RuntimeError('Saved-history mode needs the original results ZIP. Set RESULTS_PATH to its Drive path, or choose rebuild_from_embeddings. No audio download was attempted.')
    return None

def prepare():
    check_sources();ref=load(ROOT/'reference_25.json');settings=config();folder=Path(settings['drive_cache_path'])/ref['cache_key']
    pointer=folder/'complete.json';record=None;packet=None
    if pointer.is_file():
        name=load(pointer)['snapshot'];safe_name(name)
        if '/' in name:raise RuntimeError('Invalid cache snapshot pointer')
        record=load(folder/'snapshots'/name)
        if core.canonical_hash(record)+'.json'!=name or record['key']!=ref['cache_key'] or not record['complete']:
            raise RuntimeError('Complete cache snapshot is inconsistent')
        for relative,digest in ref['input_sha256'].items():
            if record['files'].get(relative)!=digest:raise RuntimeError('Cached scientific input differs: '+relative)
            origin=folder/'objects'/digest
            if not origin.is_file() or sha(origin)!=digest:raise RuntimeError('Missing or damaged cached object for '+relative)
            destination=ROOT/relative;destination.parent.mkdir(parents=True,exist_ok=True)
            if not destination.exists() or sha(destination)!=digest:
                temp=destination.with_name(destination.name+'.tmp');shutil.copyfile(origin,temp)
                if sha(temp)!=digest:raise RuntimeError('Input copy failed verification')
                os.replace(temp,destination)
        scope='Verified original Drive embedding snapshot'
    else:
        packet=find_packet(required=False)
        if packet is None:
            raise RuntimeError('The complete embedding snapshot is missing. Check DRIVE_CACHE_PATH or supply the original results ZIP in RESULTS_PATH. This notebook never downloads audio or regenerates embeddings.')
        for relative,digest in ref['input_sha256'].items():
            if packet.files.get(relative)!=digest:raise RuntimeError('Results input differs: '+relative)
            packet.copy(relative,ROOT/relative)
        packet.close();scope='Verified embeddings and labels restored from the original results packet'
    ids,X,labels,codes=verified_arrays()
    if settings['mode']=='saved_histories':
        packet=find_packet(required=True);packet.close()
    receipt={'source':scope,'input_sha256':ref['input_sha256'],'N':len(ids),'dimension':int(X.shape[1]),
             'targets':codes,'embeddings_regenerated':False,'audio_downloaded':False}
    dump(ROOT/'INPUTS_VERIFIED.json',receipt)
    print('INPUTS_VERIFIED',len(ids),'windows;',len(codes),'targets; no audio or Perch stage',flush=True)
    return receipt

def verified_arrays():
    ref=load(ROOT/'reference_25.json');exp=load(ROOT/'experiment.json')
    for relative,digest in ref['input_sha256'].items():
        if sha(ROOT/relative)!=digest:raise RuntimeError('Input changed: '+relative)
    with np.load(ROOT/'data/constructed/pool.npz',allow_pickle=False) as p:
        ids=p['window_ids'].copy();labels=p['labels'].copy();codes=p['target_codes'].tolist()
    with np.load(ROOT/'data/embeddings/SSW_EMBEDDINGS.npz',allow_pickle=False) as p:
        if not np.array_equal(ids,p['window_ids']):raise RuntimeError('Embedding IDs differ')
        X=p['embeddings'].copy()
    with np.load(ROOT/'data/constructed/label_construction.npz',allow_pickle=False) as p:
        mask=p['unknown_exclusion']|p['zero_duration_exclusion']
        if not np.array_equal(mask,p['common_exclusion']) or not np.array_equal(ids,p['raw_ids'][~mask]):
            raise RuntimeError('Original common exclusions differ')
    if len(set(ids.tolist()))!=len(ids) or X.shape!=(exp['N'],1536) or X.dtype!=np.float32 or not np.isfinite(X).all():
        raise RuntimeError('Invalid original embeddings')
    if codes!=[t['code'] for t in exp['targets']] or labels.shape!=(exp['N'],5) or not np.isin(labels,[0,1]).all():
        raise RuntimeError('Invalid original target labels')
    if labels.sum(axis=0).tolist()!=[t['positive_windows'] for t in exp['targets']]:raise RuntimeError('Positive counts differ')
    return ids.tolist(),X,labels,codes

if __name__=='__main__':prepare()
