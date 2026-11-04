"""Export only declared reproducibility inputs, executable code and scientific outputs."""
from pathlib import Path
import argparse, json, os, re, tempfile, zipfile
from artifacts import ROOT, load, sha, dump
import cache

RESULTS=['verification.json','history_results.json','audit_traces.json','bootstrap_indices.npz',
         'summary_by_target.json','summary_aggregate.json','paired_results.json','trigger_summary.json',
         'mechanisms.json','decision.json','history_results.csv','independent_statistics.json']


def selected_files(root):
    study=load(root/'STUDY_MANIFEST.json');project=load(root/'project.json')
    names=set(['study.json','models.json','dataset.json','dependencies.json','project.json','STUDY_MANIFEST.json',
               'EXECUTION_MANIFEST.json','EXECUTION_MANIFEST.json.sha256','REPORT.md'])
    names.update(cache.files(root,True))
    names.update(project['scientific_source_sha256']);names.update(project['helper_source_sha256'])
    names.update('results/'+n for n in RESULTS)
    for target in study['targets']:
        code=target['code']
        if not re.fullmatch(r'[a-zA-Z0-9_]+',code):raise RuntimeError('Unsafe target code')
        names.add('results/input_receipt_'+code+'.json')
        for seed in target['seeds']:
            prefix='runs/'+code+'/'+str(int(seed))+'/'
            # Derive the permitted snapshot files from the paid checkpoint counts.
            checkpoint=load(root/prefix/'checkpoint.json')
            names.update(prefix+n for n in ['COMPLETE.json','checkpoint.json','endpoint_selection.json','identities.npz','rows.json','traces.json','run_receipt.json'])
            names.update(prefix+f'lr_{j:02d}.npz' for j in range(len(checkpoint['snapshot_files'])))
            rows=load(root/prefix/'rows.json')
            for count in {r['training_count'] for r in rows if r['candidate_available']}:
                names.update(prefix+f'svm_{int(count)}'+suffix for suffix in ['.npz','.json'])
    for name in names:
        p=Path(name)
        if p.is_absolute() or '..' in p.parts:raise RuntimeError('Unsafe export name')
        if not (root/p).is_file():raise RuntimeError('Missing scientific artifact: '+name)
    return sorted(names)


def export(root,directory):
    if load(root/'results/verification.json')['status']!='PASS':raise RuntimeError('Verify scientific outputs before export')
    if load(root/'results/independent_statistics.json')['status']!='PASS':raise RuntimeError('Independent statistics must pass before export')
    names=selected_files(root)
    # Allowlisting excludes diagnostics, environment files, download headers,
    # logs and saved notebook outputs. Scan selected text and string arrays too.
    forbidden=re.compile(r'/(?:Users|home)/[^/\s]+/|/private/(?:var)/|/opt/(?:homebrew)/|C:\\Users\\|[mM]acOS-')
    import numpy as np
    for name in names:
        path=root/name
        if path.suffix=='.npz':
            with np.load(path,allow_pickle=False) as arrays:
                for key in arrays.files:
                    if arrays[key].dtype.kind in 'US' and any(forbidden.search(str(v)) for v in arrays[key].flat):
                        raise RuntimeError('Local identifying information detected in scientific arrays')
        elif forbidden.search(path.read_text()):
            raise RuntimeError('Local identifying information detected in export candidate '+name)
    manifest={'scope':load(root/'STUDY_MANIFEST.json')['reproduction_scope'],
              'files':{n:sha(root/n) for n in names}}
    directory.mkdir(parents=True,exist_ok=True)
    descriptor,filename=tempfile.mkstemp(prefix='.ssw_export_',suffix='.zip',dir=directory);os.close(descriptor)
    temporary=Path(filename)
    try:
        with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as archive:
            def write(name,body):
                info=zipfile.ZipInfo('ssw_results/'+name);info.external_attr=0o644<<16;info.compress_type=zipfile.ZIP_DEFLATED
                archive.writestr(info,body)
            for name in names:write(name,(root/name).read_bytes())
            write('manifest.json',json.dumps(manifest,indent=2)+'\n')
        digest=sha(temporary);output=directory/('ssw_results_'+digest+'.zip')
        if output.exists():
            if sha(output)!=digest:raise RuntimeError('Existing results archive checksum differs')
        else:os.replace(temporary,output)
        dump(root/'EXPORT_RECEIPT.json',{'filename':output.name,'sha256':digest,'files':len(names)+1})
        print('PUBLIC_RESULTS_EXPORTED',output.name)
        return output
    finally:
        if temporary.exists():temporary.unlink()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True);args=parser.parse_args()
    export(ROOT,args.directory)
if __name__=='__main__':main()
