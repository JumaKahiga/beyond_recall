"""Export an explicit scientific allowlist; omit session paths and installation logs."""
from support import *
import zipfile, argparse

def main():
    check_sources();folder=store();out=folder/'results';verified=load(out/'VERIFIED.json')
    if verified['status']!='PASS':raise RuntimeError('Independent verification required before export')
    for name,digest in verified['files'].items():assert sha(out/name)==digest
    from budget_experiment import check_baseline_gate
    check_baseline_gate();sensitivity=load(folder/'SENSITIVITY_VERIFIED.json')
    for name,digest in sensitivity['files'].items():assert sha(folder/name)==digest
    selected={}
    for name in load(ROOT/'project.json')['execution_files']:selected['code/'+name]=ROOT/name
    selected['code/project.json']=ROOT/'project.json'
    for name in ['rows_25.json','rows_30.json','traces_25.json','traces_30.json','EXECUTION.json','BASELINE_VERIFIED.json','SENSITIVITY_VERIFIED.json']:
        selected[name]=folder/name
    for p in sorted(out.iterdir()):
        if p.suffix in ['.json','.npz','.csv','.md']:selected['results/'+p.name]=p
    for p in sorted((folder/'histories').glob('*/*/history.npz')):selected[str(p.relative_to(folder))]=p
    selected['data/pool.npz']=ROOT/'data/constructed/pool.npz'
    manifest={'files':{n:sha(p) for n,p in selected.items()},
              'scope':'Post hoc 25% versus 30% fixed-history analysis; labels and final scores included, embeddings identified by original input hashes.',
              'mode':config()['mode']}
    destination=folder/'exports';destination.mkdir(parents=True,exist_ok=True);temp=destination/'ssw2_results.tmp.zip'
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
        for name,p in sorted(selected.items()):z.write(p,'ssw2_results/'+name)
        z.writestr('ssw2_results/manifest.json',json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    digest=sha(temp);final=destination/('ssw2_results_'+digest+'.zip');os.replace(temp,final)
    if sha(final)!=digest:raise RuntimeError('Export checksum mismatch')
    dump(folder/'EXPORT.json',{'filename':final.name,'sha256':digest,'files':len(selected)+1})
    print('RESULTS_EXPORTED',final.name,flush=True)

if __name__=='__main__':main()
