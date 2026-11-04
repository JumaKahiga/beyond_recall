"""Independent finite-population checks, using the dual combinatorial formula."""
from fractions import Fraction
from math import comb

def C(n,k):return comb(n,k) if 0<=k<=n else 0
def pmf(d,m,n,k):return Fraction(C(n,k)*C(d-n,m-k),C(d,m))
def cdf(d,m,n,c):
    return sum((pmf(d,m,n,k) for k in range(max(0,m-(d-n)),min(m,n,c)+1)),Fraction(0))
def expected_inspected(d,m,n,c):
    # Given k positives in the complete sample, the (c+1)th order statistic
    # has mean (c+1)*(n+1)/(k+1). For k<=c all n draws are required.
    return sum((pmf(d,m,n,k)*(n if k<=c else Fraction((c+1)*(n+1),k+1))
                for k in range(max(0,m-(d-n)),min(m,n)+1)),Fraction(0))
def check(row,trace,ids,truth,purchased,flags,audit_order):
    U,L,F=set(ids),set(purchased),set(flags);D=U-(L|F);pre=len(L|F);cap=row['attention_cap']
    if len(purchased)!=len(L) or len(flags)!=len(F):raise AssertionError('Duplicate history or flag')
    events=trace['ledger'];event_ids=[e['window_id'] for e in events]
    assert len(event_ids)==len(set(event_ids))==trace['actual_total_attention']<=cap
    assert event_ids[:len(purchased)]==list(purchased)
    assert all(e['charge']==1 for e in events)
    design=row['design'];status=row['design_status'];p=Fraction(0);expected=Fraction(0)
    if pre>cap:
        assert status=='CENSUS_UNAFFORDABLE' and len(event_ids)==len(L)
        cost=len(L)
    else:
        assert trace['census_charged_ids']==[w for w in flags if w not in L]
        assert set(event_ids[:pre])==L|F
        T=sum(truth[w] for w in F);b=sum(truth[w] for w in L-F);m=sum(truth[w] for w in D)
        assert (design['d'],design['T'],design['b_known'],design['a'],design['pre_attention'])==(len(D),T,b,T//9-b,pre)
        if status=='AUDIT':
            d,a,n,c=design['d'],design['a'],design['n'],design['c'];nmax=min(d,cap-pre)
            assert 0<=n<=nmax and c>=0
            assert cdf(d,a+1,n,c)<=Fraction(1,20)<cdf(d,a+1,n-1,c)
            assert cdf(d,a+1,nmax,c)<=Fraction(1,20)<cdf(d,a+1,nmax,c+1)
            p=cdf(d,m,n,c);expected=expected_inspected(d,m,n,c)
            sample=[w for w in audit_order if w in D][:n];hits=0;seen=[]
            for w in sample:
                seen.append(w);hits+=truth[w]
                if hits>c:break
            assert seen==trace['audit_charged_ids']
            assert [truth[w] for w in seen]==trace['audit_observed_labels']
            assert trace['observed_hits']==hits
            assert (trace['terminal_state']=='ACCEPT')==(len(seen)==n and hits<=c)
        elif status=='AUDIT_UNAFFORDABLE':
            assert cdf(len(D),T//9-b+1,min(len(D),cap-pre),0)>Fraction(1,20)
            assert not trace['audit_charged_ids']
        elif status=='COUNT_ACCEPT':
            assert T>0 and T//9-b>=len(D);p=Fraction(1)
        elif status=='NO_FLAGGED_POSITIVES':assert T==0
        elif status=='KNOWN_MISSES_PRECLUDE':assert T//9-b<0
        else:raise AssertionError('Unexpected status')
        cost=pre+expected
    total=sum(truth.values());adequate=bool(total and 10*sum(truth[w] for w in F)>=9*total)
    assert row['evaluator_only_adequate']==adequate
    for actual,target in [(row['conditional_accept_probability'],float(p)),
                          (row['correct_accept_probability'],float(p) if adequate else 0.),
                          (row['expected_total_attention'],float(cost))]:
        assert abs(actual-target)<=1e-10,(actual,target)
    assert row['planned_total_attention']<=cap and row['realized_total_attention']==len(event_ids)
    return {'exact_boundary_and_adjacent_choices':True,'independent_probability_and_cost':True,'paid_trace':True}
