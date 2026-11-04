"""Independent saved-output verification. No learner, stopping, or auditor import."""
from pathlib import Path
from fractions import Fraction
from functools import lru_cache
import hashlib,json,math,random
import numpy as np
from scipy.stats import hypergeom
ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(x):return hashlib.sha256((json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False)+'\n').encode()).hexdigest()
def loadz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
@lru_cache(None)
def cdf(d,M,n,c):
    if not 0<=M<=d or not 0<=n<=d:raise ValueError('Invalid hypergeometric counts')
    if c<0:return Fraction(0)
    if c>=min(n,M):return Fraction(1)
    return Fraction(sum(math.comb(M,j)*math.comb(d-M,n-j) for j in range(max(0,n-d+M),min(c,n,M)+1)),math.comb(d,n))
@lru_cache(None)
def independent_cost(d,M,n,c):
    # Forward probability recurrence on surviving hit states. It shares neither
    # the reference CDF implementation nor its vectorized survival-sum code.
    state=np.zeros(c+1,dtype=np.longdouble);state[0]=1;total=np.longdouble(0)
    for t in range(n):
        total+=sum(state)
        nxt=np.zeros_like(state)
        for j in range(c+1):
            if state[j]:
                nxt[j]+=state[j]*max(0,d-M-t+j)/(d-t)
                if j<c:nxt[j+1]+=state[j]*max(0,M-j)/(d-t)
        state=nxt
    return float(total)
def ordinal(values):
    ranks=[0]*len(values)
    for rank,i in enumerate(sorted(range(len(values)),key=lambda i:(-float(values[i]),i)),1):ranks[i]=rank
    return ranks

def verify_row(row,trace,ids,truth,order,flagged,cap):
    L=[ids[i] for i in order];F=set(flagged);U=set(ids);pre=len(set(L)|F);D=U-set(L)-F
    assert trace['training_charged_ids']==L
    assert row['training_count']==len(L) and len(set(L))==len(L)
    assert row['flagged_id_hash']==canonical(flagged)
    assert row['overlap']==len(set(L)&F) and row['pre_attention']==pre
    T=sum(truth[w] for w in F);b=sum(truth[w] for w in set(L)-F);M=sum(truth[w] for w in D)
    assert (T,b,M)==(row['evaluator_only_T'],row['evaluator_only_b'],row['evaluator_only_M'])
    total=T+b+M;adequate=bool(total>0 and 10*T>=9*total)
    assert row['evaluator_only_adequate']==adequate
    assert abs(row['evaluator_only_recall']-(T/total if total else 0))<=1e-15
    n=0;c=-1;p=0.;expected=0.;status=row['design_status'];d=len(D);a=T//9-b
    if pre>cap:
        assert status=='CENSUS_UNAFFORDABLE' and not trace['census_charged_ids']
        planned=len(L);expected=len(L)
    else:
        design=row['design'];nmax=min(d,cap-pre)
        assert {k:design[k] for k in ('d','T','b_known','pre_attention','cap','a','n_max')}==dict(d=d,T=T,b_known=b,pre_attention=pre,cap=cap,a=a,n_max=nmax)
        assert trace['census_charged_ids']==[w for w in flagged if w not in set(L)]
        if T==0:assert status=='NO_FLAGGED_POSITIVES'
        elif a<0:assert status=='KNOWN_MISSES_PRECLUDE'
        elif a>=d:assert status=='COUNT_ACCEPT';p=1.
        elif status=='AUDIT_UNAFFORDABLE':assert cdf(d,a+1,nmax,0)>Fraction(1,20)
        else:
            assert status=='AUDIT';n=design['n'];c=design['c']
            assert cdf(d,a+1,n,c)<=Fraction(1,20)<cdf(d,a+1,n-1,c)
            assert cdf(d,a+1,nmax,c)<=Fraction(1,20)<cdf(d,a+1,nmax,c+1)
            p=float(cdf(d,M,n,c));expected=independent_cost(d,M,n,c)
            assert abs(p-float(hypergeom.cdf(c,d,M,n)))<=1e-10
        planned=pre+n;expected+=pre
    assert abs(p-row['conditional_accept_probability'])<=1e-10
    assert abs(p*adequate-row['correct_accept_probability'])<=1e-10
    assert abs(p*(not adequate)-row['false_accept_probability'])<=1e-10
    assert abs(expected-row['expected_total_attention'])<=1e-10,(expected,row['expected_total_attention'])
    assert abs(adequate*(1-p)-row['unaccepted_adequacy'])<=1e-10
    assert planned==row['planned_total_attention']<=cap
    assert trace['training_count']==len(L) and trace['planned_pre_audit_attention']==pre
    assert trace['actual_pre_audit_attention']==(len(L) if pre>cap else pre)
    if pre<=cap:
        assert (trace['T'],trace['b_known'],trace['unseen_d'],trace['allowance_a'])==(T,b,d,a)
        assert trace['planned_n']==n and trace['cutoff_c']==(c if status=='AUDIT' else None)
    assert abs(row.get('expected_audit_inspected',0)-(expected-(len(L) if pre>cap else pre)))<=1e-10
    rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([991230000,row['seed']])))
    while True:
        keys=rng.random(len(ids))
        if len(np.unique(keys))==len(ids):break
    restricted=[ids[i] for i in np.argsort(keys) if ids[i] in D]
    S=trace['audit_charged_ids'];hits=[truth[w] for w in S]
    assert S==restricted[:len(S)] and len(S)<=n
    if status=='AUDIT':
        if sum(hits)>c:
            assert sum(hits)==c+1 and sum(hits[:-1])<=c and trace['terminal_state']=='STOP_UNACCEPTED'
        else:assert len(S)==n and trace['terminal_state']=='ACCEPT'
    else:
        assert S==[] and (trace['terminal_state']=='ACCEPT')==(status=='COUNT_ACCEPT')
    assert trace['audit_inspected']==len(S)
    assert hits==trace.get('audit_observed_labels',[]) and sum(hits)==trace['observed_hits']
    charged=L+trace['census_charged_ids']+S
    assert [e['window_id'] for e in trace['ledger']]==charged
    assert all(e['charge']==1 for e in trace['ledger'])
    purposes=['paid_search_startup' if i<row['startup_paid'] else 'training' for i in range(len(L))]+['flagged_census']*len(trace['census_charged_ids'])+['audit']*len(S)
    assert [e['purpose'] for e in trace['ledger']]==purposes
    assert len(charged)==len(set(charged))==trace['actual_total_attention']==row['realized_total_attention']<=cap
    assert row['realized_accepted']==(trace['terminal_state']=='ACCEPT')
    assert row['realized_correct_accept']==(row['realized_accepted'] and adequate)
    assert row['realized_false_accept']==(row['realized_accepted'] and not adequate)
    return {'p_absolute_error':abs(p-row['conditional_accept_probability']),'cost_absolute_error':abs(expected-row['expected_total_attention'])}

def independent_endpoints(snaps,monitor):
    kappas=[];conf=[];sp=None;cd=None;run=0
    for j,s in enumerate(snaps):
        vals=s['probability'][monitor];h=0.
        for p0 in vals:
            p=float(p0)
            if 0<p<1:h+=-p*math.log2(p)-(1-p)*math.log2(1-p)
        conf.append(1-h/len(vals))
        if j:
            x=snaps[j-1]['prediction'][monitor];y=s['prediction'][monitor];m=len(x)
            x1=int(x.sum());y1=int(y.sum());pe=x1*y1+(m-x1)*(m-y1)
            k=Fraction(m*int((x==y).sum())-pe,m*m-pe) if pe!=m*m else None
            kappas.append(k)
            window=kappas[-3:]
            if sp is None and len(window)==3 and None not in window and sum(window)/3>=Fraction(99,100):sp=j
            run=run+1 if conf[j]<conf[j-1] else 0
            if cd is None and run==3:cd=j
    return sp,cd,conf,kappas

def main():
    study=json.loads((ROOT/'STUDY_MANIFEST.json').read_text());execution=json.loads((ROOT/'EXECUTION_MANIFEST.json').read_text())
    for name,digest in execution['source_sha256'].items():assert sha(ROOT/name)==digest
    pool=loadz(ROOT/'data/constructed/pool.npz');ids=pool['window_ids'].tolist()
    cap=study['constants']['unique_label_cap'];K=study['constants']['flagged_count']
    allrows=[];alltraces=[];errors=[];fit_count=0;warnings_count=0
    assignments=[(t,seed) for t in study['targets'] for seed in t['seeds']]
    for target,seed in assignments:
        code=target['code'];truth=dict(zip(ids,pool['labels'][:,pool['target_codes'].tolist().index(code)].tolist()))
        folder=ROOT/'runs'/code/str(seed);marker=json.loads((folder/'COMPLETE.json').read_text())
        assert marker['execution_sha256']==sha(ROOT/'EXECUTION_MANIFEST.json')
        for name,digest in marker['files'].items():assert sha(folder/name)==digest
        data=loadz(folder/'identities.npz');assert data['pool_ids'].tolist()==ids
        checkpoint=json.loads((folder/'checkpoint.json').read_text());order=data['purchased_indices'].tolist();startup=checkpoint['startup']
        assert order==checkpoint['order'] and data['purchased_ids'].tolist()==[ids[i] for i in order]
        assert data['purchased_labels'].tolist()==[truth[ids[i]] for i in order]
        shuffled=list(range(len(ids)));random.Random(seed).shuffle(shuffled)
        assert order[:startup]==shuffled[:startup]
        monitor=np.sort(np.random.Generator(np.random.PCG64(np.random.SeedSequence([991220000,seed]))).choice(len(ids),size=2000,replace=False))
        assert np.array_equal(monitor,data['monitor_indices'])
        rows=json.loads((folder/'rows.json').read_text());traces=json.loads((folder/'traces.json').read_text())
        assert all(r['target']==code and r['seed']==seed for r in rows)
        assert all(t['target']==code and t['seed']==seed for t in traces)
        assert len(rows)==len(traces)==8 and [r['workflow'] for r in rows]==[w['id'] for w in study['workflows']]
        snaps=[loadz(folder/s['path']) for s in checkpoint['snapshot_files']]
        if not snaps:
            assert len(order)==3000 and len({truth[ids[i]] for i in order})==1
            assert all(not r['candidate_available'] and r['correct_accept_probability']==0 and r['expected_total_attention']==3000 for r in rows)
            assert all(t['actual_total_attention']==3000 and not t['audit_charged_ids'] for t in traces)
        else:
            assert len({truth[ids[i]] for i in order[:startup]})==2 and len({truth[ids[i]] for i in order[:startup-1]})==1
            counts=[int(s['count']) for s in snaps]
            assert counts==list(range(startup,3000,500))+[3000]
            for j,s in enumerate(snaps):
                assert np.isfinite(s['probability']).all() and np.isfinite(s['decision']).all()
                assert np.array_equal(s['prediction'],(s['decision']>0).astype(np.int8))
                if j<len(snaps)-1:
                    seen=set(order[:counts[j]])
                    def ent(i):
                        p=min(1-1e-12,max(1e-12,float(s['probability'][i])))
                        return -p*math.log(p)-(1-p)*math.log(1-p)
                    selected=sorted((i for i in range(len(ids)) if i not in seen),key=lambda i:(-ent(i),ids[i]))[:counts[j+1]-counts[j]]
                    assert selected==order[counts[j]:counts[j+1]]==checkpoint['queries'][j]['indices']
            sp,cd,conf,ks=independent_endpoints(snaps,monitor)
            detail=json.loads((folder/'endpoint_selection.json').read_text())
            assert detail['first_sp_index']==sp and detail['first_confidence_index']==cd
            assert detail['confidence']==conf and detail['adjacent_kappas']==[str(k) if k is not None else None for k in ks]
            for row,trace,w in zip(rows,traces,study['workflows']):
                trigger={'fixed':len(snaps)-1,'sp':sp,'confidence_decline':cd}[w['stop']]
                index=len(snaps)-1 if trigger is None else trigger;count=counts[index]
                status='fixed_horizon' if w['stop']=='fixed' else 'horizon_without_trigger' if trigger is None else 'triggered_at_horizon' if count==3000 else 'triggered_before_horizon'
                assert row['stop_status']==status and row['training_count']==count
                lr=ordinal(snaps[index]['probability']);svm=ordinal(loadz(folder/f'svm_{count}.npz')['scores'])
                weights={'lr':(1,0),'svm':(0,1),'equal_blend':(1,1),'3to1_blend':(3,1)}[w['ranker']]
                selected=sorted(range(len(ids)),key=lambda i:(weights[0]*lr[i]+weights[1]*svm[i],lr[i],i))[:K]
                flagged=[ids[i] for i in selected];assert data['flagged_'+w['id']].tolist()==flagged
                errors.append(verify_row(row,trace,ids,truth,order[:count],flagged,cap))
            for count in set(r['training_count'] for r in rows):
                meta=json.loads((folder/f'svm_{count}.json').read_text())
                idx=np.asarray(sorted(order[:count]),dtype=np.int64)
                assert meta['training_indices_sha256']==hashlib.sha256(idx.tobytes()).hexdigest() and meta['scaler_fit_count']==count
        receipt=json.loads((folder/'run_receipt.json').read_text());fit_count+=receipt['new_model_fit_count'];warnings_count+=len(receipt['warnings'])
        allrows.extend(rows);alltraces.extend(traces)
        print('VERIFIED_SEED',seed,flush=True)
    expected_rows=len(assignments)*8
    assert len(allrows)==len(alltraces)==expected_rows and len({(r['target'],r['seed'],r['workflow']) for r in allrows})==expected_rows
    out=ROOT/'results'
    for name,data in [('history_results.json',allrows),('audit_traces.json',alltraces)]:
        (out/name).write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n')
    result={'status':'PASS','targets':len(study['targets']),'histories':len(assignments),'rows':expected_rows,'traces':expected_rows,'new_model_fit_count':fit_count,'fit_warning_count':warnings_count,
            'source_sha256':sha(ROOT/'verify.py'),'execution_sha256':sha(ROOT/'EXECUTION_MANIFEST.json'),
            'max_probability_absolute_error':max((e['p_absolute_error'] for e in errors),default=0),
            'max_expected_cost_absolute_error':max((e['cost_absolute_error'] for e in errors),default=0),
            'integer_boundaries':'PASS','one_attempt_ledgers':'PASS','prefixes_triggers_rankings':'PASS',
            'method':'Independent integer combinatorial CDF and long-double forward stopping-cost recurrence; no fitting, stopping, or auditor import'}
    (out/'verification.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
