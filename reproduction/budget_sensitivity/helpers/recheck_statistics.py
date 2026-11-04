"""Check saved statistics without importing the analysis implementation."""
from support import *

def main():
    check_sources();folder=store();exp=load(ROOT/'experiment.json');out=folder/'results'
    rows=load(folder/'rows_25.json')+load(folder/'rows_30.json')
    by={(r['budget_percent'],r['target'],r['seed'],r['workflow']):r for r in rows}
    assert len(by)==len(rows)==800
    rng=np.random.Generator(np.random.PCG64(exp['analysis']['bootstrap_seed']))
    indices=np.stack([rng.integers(0,20,(20000,20)) for _ in exp['targets']])
    assert np.array_equal(indices,loadz(out/'bootstrap_indices.npz')['indices'])
    def vectors(b,w,key):
        if key=='y':return np.asarray([[by[(b,t['code'],seed,w)]['conditional_accept_probability']*bool(by[(b,t['code'],seed,w)]['evaluator_only_adequate']) for seed in t['seeds']] for t in exp['targets']])
        return np.asarray([[by[(b,t['code'],seed,w)]['expected_total_attention'] for seed in t['seeds']] for t in exp['targets']])
    def verify(array,mean,interval):
        boot=sum(array[i][indices[i]].sum(axis=1)/20 for i in range(5))/5
        assert abs(float(array.sum()/100)-mean)<=1e-10
        assert np.allclose(np.quantile(boot,[.025,.975],method='linear'),interval,atol=1e-10,rtol=0)
    summaries=load(out/'summary.json')
    for saved in summaries:
        verify(vectors(saved['budget_percent'],saved['workflow'],'y'),saved['mean_acceptance'],saved['acceptance_interval_95'])
        verify(vectors(saved['budget_percent'],saved['workflow'],'cost'),saved['mean_expected_windows'],saved['expected_windows_interval_95'])
        assert saved['point_estimate_above_80']==(saved['mean_acceptance']>.8)
        assert saved['interval_lower_endpoint_above_80']==(saved['acceptance_interval_95'][0]>.8)
    for saved in load(out/'contrasts.json'):
        w=saved['workflow']
        verify(vectors(30,w,'y')-vectors(25,w,'y'),saved['mean_acceptance_difference'],saved['difference_interval_95'])
        verify(vectors(30,w,'cost')-vectors(25,w,'cost'),saved['mean_expected_windows_difference'],saved['windows_difference_interval_95'])
    primary=next(s for s in summaries if s['budget_percent']==30 and s['workflow']==exp['primary_workflow'])
    decision=load(out/'decision.json')
    for k,v in primary.items():assert decision[k]==v
    assert decision['bootstrap_indices_sha256']==array_hash(indices)
    for saved in load(out/'by_target.json'):
        i=[t['code'] for t in exp['targets']].index(saved['target']);a=vectors(saved['budget_percent'],saved['workflow'],'y')[i]
        assert abs(float(a.mean())-saved['mean_acceptance'])<=1e-10
        assert np.allclose(np.quantile(a[indices[i]].mean(axis=1),[.025,.975],method='linear'),saved['acceptance_interval_95'],atol=1e-10,rtol=0)
    paths=[out/n for n in ['summary.json','contrasts.json','by_target.json','decision.json','bootstrap_indices.npz','REPORT.md','history_results.csv']]
    dump(out/'VERIFIED.json',{'status':'PASS','rows':800,'bootstrap_resamples':20000,
         'files':{p.name:sha(p) for p in paths},'scope':'Independent saved-row probability-weighted aggregation and paired bootstrap; per-history exact audit checks precede aggregation.'})
    print('INDEPENDENT_STATISTICS_PASS',flush=True)

if __name__=='__main__':main()
