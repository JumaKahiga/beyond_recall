"""Independent saved-row statistics: no learner, auditor or analysis import."""
from pathlib import Path
import hashlib,json,sys,platform
import numpy as np
ROOT=Path(__file__).resolve().parent

def close(a,b):
    if a is None or b is None:
        assert a is b
    else:
        assert abs(float(a)-float(b))<=1e-10,(a,b)

def main(root=ROOT):
    def read(name):return json.loads((root/name).read_text())
    study=read('STUDY_MANIFEST.json');targets=study['targets'];J=len(targets)
    rows=read('results/history_results.json')
    by={(r['target'],r['seed'],r['workflow']):r for r in rows}
    assert len(rows)==len(by)==160*J
    rng=np.random.Generator(np.random.PCG64(991240000))
    indices=[rng.integers(0,20,(20000,20)) for _ in targets]
    digest=hashlib.sha256(np.asarray(indices).tobytes()).hexdigest()
    decision=read('results/decision.json')
    assert digest==decision['bootstrap_indices_sha256']
    comparisons=[]
    for saved in read('results/paired_results.json'):
        target_boot_y=[];target_boot_cost=[];target_means_y=[];target_means_cost=[]
        for ti,target in enumerate(targets):
            left=[by[(target['code'],seed,saved['left'])] for seed in target['seeds']]
            right=[by[(target['code'],seed,saved['right'])] for seed in target['seeds']]
            # Reconstruct z from p and adequacy, including zero for unavailable.
            def z(row):
                return row['conditional_accept_probability']*bool(row['candidate_available'] and row['evaluator_only_adequate'])
            effect=np.asarray([z(a)-z(b) for a,b in zip(left,right)])
            attention=np.asarray([a['expected_total_attention']-b['expected_total_attention'] for a,b in zip(left,right)])
            ys=effect[indices[ti]].sum(axis=1)/20
            cs=attention[indices[ti]].sum(axis=1)/20
            target_boot_y.append(ys);target_boot_cost.append(cs)
            target_means_y.append(float(effect.sum()/20));target_means_cost.append(float(attention.sum()/20))
            ref=saved['by_target'][ti];assert ref['target']==target['code']
            close(target_means_y[-1],ref['mean_delta_Y']);close(target_means_cost[-1],ref['mean_delta_cost'])
            for a,b in zip(np.percentile(ys,[2.5,97.5],method='linear'),ref['Y_percentile_95']):close(a,b)
            for a,b in zip(np.percentile(cs,[2.5,97.5],method='linear'),ref['cost_percentile_95']):close(a,b)
            assert int((effect>1e-10).sum())==ref['wins']
            assert int((np.abs(effect)<=1e-10).sum())==ref['ties']
            assert int((effect< -1e-10).sum())==ref['losses']
        y=float(sum(target_means_y)/J);cost=float(sum(target_means_cost)/J)
        ys=sum(target_boot_y)/J;cs=sum(target_boot_cost)/J
        ci=np.percentile(ys,[2.5,97.5],method='linear');cci=np.percentile(cs,[2.5,97.5],method='linear')
        ref=saved['aggregate'];close(y,ref['mean_delta_Y']);close(cost,ref['mean_delta_cost'])
        for a,b in zip(ci,ref['Y_percentile_95']):close(a,b)
        for a,b in zip(cci,ref['cost_percentile_95']):close(a,b)
        comparisons.append({'left':saved['left'],'right':saved['right'],'mean_delta_Y':y,
                            'Y_percentile_95':ci.tolist(),'mean_delta_cost':cost,'cost_percentile_95':cci.tolist()})
    per_target=read('results/summary_by_target.json');aggregate=read('results/summary_aggregate.json')
    target_lookup={t['code']:t for t in targets}
    for reference in per_target:
        target=target_lookup[reference['target']]
        group=[by[(target['code'],seed,reference['workflow'])] for seed in target['seeds']]
        for source,key in [('conditional_accept_probability','mean_conditional_accept_probability'),
                           ('correct_accept_probability','mean_correct_accept_probability'),
                           ('false_accept_probability','mean_false_accept_probability'),
                           ('expected_total_attention','mean_expected_attention'),
                           ('realized_total_attention','mean_realized_attention')]:
            close(sum(r[source] for r in group)/20,reference[key])
        adequate=sum(bool(r['candidate_available'] and r['evaluator_only_adequate']) for r in group)
        assert adequate==reference['adequate'];close(adequate/20,reference['adequacy_frequency'])
    for reference in aggregate:
        subs=[r for r in per_target if r['workflow']==reference['workflow']]
        for key in ['mean_correct_accept_probability','mean_false_accept_probability','mean_conditional_accept_probability',
                    'mean_expected_attention','mean_realized_attention','adequacy_frequency']:
            close(sum(r[key] for r in subs)/J,reference[key])
    primary=comparisons[0]
    assert (primary['mean_delta_Y']>=.10 and primary['Y_percentile_95'][0]>0 and all(decision['gates'].values()))==decision['primary_practical_criterion_met']
    result={'status':'PASS','python':platform.python_version(),'numpy':np.__version__,'targets':J,'histories':20*J,'rows':160*J,
            'bootstrap_indices_sha256':digest,'comparisons':comparisons,
            'scope':'Saved-row conditional outcomes, per-target/aggregate means and independent stratified bootstrap; no training or audio inference.'}
    (root/'results/independent_statistics.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
