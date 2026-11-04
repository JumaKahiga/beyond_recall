"""Verified recovery backups and persistent completed-recording downloads."""
from pathlib import Path
import argparse, copy, hashlib, json, re, shutil, os
from artifacts import ROOT, load, dump, sha, canonical_hash

SMALL=['study.json','dataset.json','data/constructed/RECORDING_PERMUTATION.json',
       'data/source/annotations.csv','data/source/species.csv','data/source/recording_location.txt','data/source/zip_tail.bin']
CONFIG=['project.json','models.json','dependencies.json']
SELECTION='data/constructed/RECORDING_SELECTION.json'
AUDIO_PATTERN=re.compile(r'SSW_\d{3}_\d{8}_\d{6}Z\.flac')


def same_protocol_except_label_exception(old,new):
    def stable(s):
        s=copy.deepcopy(s);s.pop('protocol_version',None)
        s['construction'].pop('zero_duration_annotation',None)
        s['construction'].pop('malformed_annotation',None)
        return s
    return stable(old)==stable(new)


def recording_key(study):
    c=study['construction']
    return canonical_hash({'dataset':study['dataset'],'construction':{k:c[k] for k in
        ['recording_seed','recordings','generator','numpy','inventory_order','sample_rate','complete_duration_seconds','upper_duration_exclusive']}})


def allowed(name):
    p=Path(name)
    if p.is_absolute() or '..' in p.parts:return False
    if name in SMALL+CONFIG+[SELECTION]:return True
    if len(p.parts)!=3 or p.parts[:2]!=('data','audio'):return False
    filename=p.name[:-12] if p.name.endswith('.source.json') else p.name
    return bool(AUDIO_PATTERN.fullmatch(filename))


def checked_copy(source,target,digest):
    if target.exists():
        if sha(target)!=digest:raise RuntimeError('Existing audio/source artifact differs: '+target.name)
        return
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_name(target.name+'.partial')
    shutil.copyfile(source,temporary)
    if sha(temporary)!=digest:raise RuntimeError('Audio/source transfer checksum mismatch: '+source.name)
    os.replace(temporary,target)


def validate_selection(selection,permutation):
    selected=selection['selected'];failures=selection.get('technical_failures',[])
    if selection['status']!='RECORDINGS_COMPLETE' or len(selected)!=33:raise RuntimeError('Backup lacks 33 selected recordings')
    positions=permutation['permutation_filenames'];accepted=[];rejected=[]
    for row,target in [(r,accepted) for r in selected]+[(r,rejected) for r in failures]:
        i=row['permutation_position']
        if not isinstance(i,int) or not 0<=i<len(positions) or positions[i]!=row['filename']:
            raise RuntimeError('Backup changed the recording permutation')
        target.append(i)
    if accepted!=sorted(set(accepted)) or len(set(rejected))!=len(rejected) or set(accepted)&set(rejected):
        raise RuntimeError('Duplicate or inconsistent recording selection')
    if sorted(accepted+rejected)!=list(range(accepted[-1]+1)):
        raise RuntimeError('Backup omitted a recording from the selected prefix')
    for record in selected:
        a=record['audio'];frames=a['frames'];rate=a['sample_rate']
        if rate!=32000 or a['channels']<1 or not 3600*rate<=frames<3605*rate or a['complete_windows']!=720:
            raise RuntimeError('Backup contains invalid audio geometry')


def read_inventory(folder,manifest_name):
    m=load(folder/manifest_name)
    expected=m['files']
    if any(not allowed(n) or not isinstance(d,str) or not re.fullmatch('[0-9a-f]{64}',d) for n,d in expected.items()):
        raise RuntimeError('Unexpected file in audio backup manifest')
    if not set(SMALL)<=set(expected):raise RuntimeError('Audio backup missing source metadata')
    for name in SMALL:
        if sha(folder/name)!=expected[name]:raise RuntimeError('Audio backup source checksum mismatch: '+name)
    if manifest_name=='BACKUP_MANIFEST.json':
        if m['status']!='COMPLETE' or m['selected_recordings']!=33 or SELECTION not in expected:
            raise RuntimeError('Incomplete recovery backup')
        if sha(folder/SELECTION)!=expected[SELECTION]:raise RuntimeError('Backup selection checksum mismatch')
        selection=load(folder/SELECTION);permutation=load(folder/'data/constructed/RECORDING_PERMUTATION.json')
        validate_selection(selection,permutation)
        if selection['recipe_freeze_sha256']!=expected['study.json'] or permutation['recipe_freeze_sha256']!=expected['study.json']:
            raise RuntimeError('Backup study and selection hashes differ')
        for record in selection['selected']:
            name='data/audio/'+record['filename']
            if expected.get(name)!=record['audio']['sha256'] or name+'.source.json' not in expected:
                raise RuntimeError('Backup missing selected audio or receipt')
    else:
        expected=dict(expected)
        for commit in sorted((folder/'commits').glob('*.json')):
            r=load(commit);name=r['filename']
            if not AUDIO_PATTERN.fullmatch(name):raise RuntimeError('Invalid cached recording name')
            if any(not isinstance(r[k],str) or not re.fullmatch('[0-9a-f]{64}',r[k]) for k in ['audio_sha256','receipt_sha256']):
                raise RuntimeError('Invalid cached recording checksum')
            expected['data/audio/'+name]=r['audio_sha256']
            expected['data/audio/'+name+'.source.json']=r['receipt_sha256']
    return m,expected


def choose(root,directory,required=False):
    directory=Path(directory)
    candidates=[directory]
    if directory.is_dir():
        candidates+=sorted(p for p in directory.iterdir() if p.is_dir())
        if (directory/'downloads').is_dir():candidates+=sorted(p for p in (directory/'downloads').iterdir() if p.is_dir())
    found=[]
    for folder in candidates:
        kind='BACKUP_MANIFEST.json' if (folder/'BACKUP_MANIFEST.json').exists() else 'SOURCE_CACHE_MANIFEST.json'
        if not (folder/kind).exists():continue
        m,expected=read_inventory(folder,kind)
        if same_protocol_except_label_exception(load(folder/'study.json'),load(root/'study.json')):
            found.append((folder,kind,m,expected))
    complete=[x for x in found if x[1]=='BACKUP_MANIFEST.json']
    candidates=complete or found
    if len(candidates)>1:raise RuntimeError('Multiple matching audio backups; specify one exact backup folder')
    if not candidates and required:raise RuntimeError('Requested audio backup is missing or incompatible; no download fallback was attempted')
    return candidates[0] if candidates else None


def restore(root,directory,required=False,metadata_only=False):
    chosen=choose(root,directory,required)
    if chosen is None:
        dump(root/'AUDIO_RESTORE_STATUS.json',{'status':'missing'});print('NO_AUDIO_BACKUP');return
    folder,kind,m,expected=chosen
    small=list(SMALL)+([SELECTION] if kind=='BACKUP_MANIFEST.json' else [])
    # Source study metadata is verified but cannot overwrite the current protocol.
    for name in small:
        if name=='study.json':continue
        if name.endswith('RECORDING_PERMUTATION.json') or name==SELECTION:
            value=load(folder/name);value['recipe_freeze_sha256']=sha(root/'study.json')
            target=root/name
            if target.exists() and load(target)!=value:raise RuntimeError('Existing recording selection differs from backup')
            dump(target,value)
        else:checked_copy(folder/name,root/name,expected[name])
    if not metadata_only:
        names=[name for name in expected if name.startswith('data/audio/')]
        if kind=='BACKUP_MANIFEST.json':
            # Rejected candidates are already accounted for in the verified selection.
            # Only the 33 selected recordings are needed for construction/inference.
            selected={'data/audio/'+r['filename'] for r in load(folder/SELECTION)['selected']}
            names=[name for name in names if name in selected or name.removesuffix('.source.json') in selected]
        for i,name in enumerate(names,1):
            checked_copy(folder/name,root/name,expected[name])
            print('AUDIO_RESTORED',i,'of',len(names),Path(name).name,flush=True)
        # Check every copied FLAC against its release-entry receipt too.
        for name in names:
            if not name.endswith('.flac'):continue
            receipt=load(root/(name+'.source.json'))
            if receipt['audio_sha256']!=expected[name] or not receipt['zip_crc32_verified']:
                raise RuntimeError('Audio release receipt mismatch')
        if kind=='BACKUP_MANIFEST.json':
            proof={'backup_manifest_sha256':sha(folder/kind),'original_study_sha256':expected['study.json'],
                   'original_selection_sha256':expected[SELECTION],
                   'current_study_sha256':sha(root/'study.json'),
                   'scope':'Restored original audio and recording selection; window labels and targets are constructed under the current protocol'}
            dump(root/'data/constructed/AUDIO_RECOVERY_PROVENANCE.json',proof)
    status='complete' if kind=='BACKUP_MANIFEST.json' else 'partial'
    dump(root/'AUDIO_RESTORE_STATUS.json',{'status':status,'metadata_only':metadata_only,'backup_manifest_sha256':sha(folder/kind)})
    print('AUDIO_BACKUP_METADATA_READY' if metadata_only else 'AUDIO_BACKUP_RESTORED',status,flush=True)


def save_source(root,base):
    folder=Path(base)/'downloads'/recording_key(load(root/'study.json'))[:16]
    expected={name:sha(root/name) for name in SMALL}
    # A new label-policy version can reuse the same release recording cache.
    if (folder/'SOURCE_CACHE_MANIFEST.json').exists():
        read_inventory(folder,'SOURCE_CACHE_MANIFEST.json')
        if not same_protocol_except_label_exception(load(folder/'study.json'),load(root/'study.json')):
            raise RuntimeError('Cached recording source uses different study settings')
        return folder
    for name,digest in expected.items():checked_copy(root/name,folder/name,digest)
    dump(folder/'SOURCE_CACHE_MANIFEST.json',{'status':'PARTIAL','files':expected})
    return folder


def save_record(root,base,filename):
    folder=save_source(root,base)
    if not AUDIO_PATTERN.fullmatch(filename):raise RuntimeError('Invalid recording filename')
    audio='data/audio/'+filename;receipt=audio+'.source.json';record=load(root/receipt)
    if not record['zip_crc32_verified']:raise RuntimeError('Unverified recording cannot be cached')
    digest=record['audio_sha256'];receipt_digest=sha(root/receipt)
    checked_copy(root/audio,folder/audio,digest);checked_copy(root/receipt,folder/receipt,receipt_digest)
    dump(folder/'commits'/(filename+'.json'),{'filename':filename,'audio_sha256':digest,'receipt_sha256':receipt_digest})
    print('AUDIO_SAVED_TO_DRIVE',filename,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['restore']);p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--required',action='store_true');p.add_argument('--metadata-only',action='store_true');a=p.parse_args()
    restore(ROOT,a.directory,a.required,a.metadata_only)
if __name__=='__main__':main()
