"""Lock scientific inputs and executable source before classifier fitting."""
from artifacts import ROOT, load, dump, sha, check_recipe
import cache

def main():
    if (ROOT/'EXECUTION_MANIFEST.json').exists():
        from run_experiment import check_execution
        check_execution();print('REUSING_EXECUTION');return
    cache.verify(ROOT,True,True)
    project=load(ROOT/'project.json')
    for name,digest in dict(project['scientific_source_sha256'],**project['helper_source_sha256']).items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Scientific implementation changed: '+name)
    study=check_recipe();c=load(ROOT/'data/constructed/CONSTRUCTION_MANIFEST.json')
    study.update(targets=c['selected_targets'],pool={'N':c['N'],'embedding_dim':1536,'embedding_dtype':'float32'})
    study['constants'].update(unique_label_cap=c['B'],flagged_count=c['K'])
    paths=['study.json','models.json','data/constructed/pool.npz','data/constructed/CONSTRUCTION_MANIFEST.json',
           'data/constructed/CONSTRUCTION_ARTIFACT_HASHES.json','data/embeddings/SSW_EMBEDDINGS.npz',
           'data/embeddings/EMBEDDING_RECEIPT.json','EMBEDDING_EXECUTION_MANIFEST.json']
    study['inputs']={p:sha(ROOT/p) for p in paths}
    status=load(ROOT/'CACHE_STATUS.json')['status']
    description={'complete':'reused verified arrays','partial':'resumed partial cache; generated remaining recordings from original audio'}
    study['reproduction_scope']={'embeddings':description.get(status,'generated from original audio'),
                                  'classifier_training':'performed in this run'}
    dump(ROOT/'STUDY_MANIFEST.json',study)
    from support import validate_runtime
    from run_experiment import source_hashes
    m={'source_sha256':source_hashes(),'runtime':validate_runtime(),'study_sha256':sha(ROOT/'STUDY_MANIFEST.json')}
    dump(ROOT/'EXECUTION_MANIFEST.json',m)
    (ROOT/'EXECUTION_MANIFEST.json.sha256').write_text(sha(ROOT/'EXECUTION_MANIFEST.json')+'\n')
    print('SCIENTIFIC_INPUTS_AND_SOURCE_FROZEN')
if __name__=='__main__':main()
