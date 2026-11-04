"""Frozen ranking, permitted planning and one-attempt audit execution.

No acquisition code. The planner accepts counts obtainable after a paid census,
never a retrospective result row. Numerical planning is checked with exact integer arithmetic.
"""
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys
import numpy as np
from scipy.stats import hypergeom

ROOT = Path(__file__).resolve().parent

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def canonical_hash(value):
    return hashlib.sha256((json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)+'\n').encode()).hexdigest()

import audit_numeric as AUDITOR
import hypergeom_exact as RATIONAL
ALPHA = Fraction(1, 20)

def align_scores(pool_ids, cache_ids, scores):
    """Join by stable ID; surplus scores cannot enlarge the included pool."""
    ids, incoming = tuple(pool_ids), tuple(cache_ids)
    values = np.asarray(scores, dtype=float)
    if len(set(ids)) != len(ids) or len(set(incoming)) != len(incoming):
        raise ValueError('duplicate stable IDs')
    if values.shape != (len(incoming),) or not np.isfinite(values).all():
        raise ValueError('invalid score vector')
    lookup = dict(zip(incoming, values))
    if not set(ids) <= set(incoming):
        raise ValueError('missing included score IDs')
    return np.asarray([lookup[w] for w in ids])

def ordinal_ranks(scores):
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError('scores must be finite and already aligned to frozen pool IDs')
    order = np.argsort(-values, kind='stable')
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(1, len(values)+1)
    return ranks

@dataclass(frozen=True)
class Candidate:
    pool_ids: tuple
    flagged_ids: tuple
    method_id: str
    score_hash: str
    flagged_id_hash: str

def freeze_candidate(pool_ids, lr_scores, svm_scores, method_id, flagged_count):
    ids = tuple(pool_ids)
    if len(ids) != len(set(ids)) or not 0 <= flagged_count <= len(ids):
        raise ValueError('invalid pool or flagged count')
    lr, svm = ordinal_ranks(lr_scores), ordinal_ranks(svm_scores)
    if len(lr) != len(ids) or len(svm) != len(ids):
        raise ValueError('score length mismatch')
    if method_id == 'LR-RAW-FA': key = lr
    elif method_id == 'SVM-STD-FA': key = svm
    elif method_id == 'LR75-SVM25-FA': key = 3*lr + svm
    elif method_id == 'LR50-SVM50-FA': key = lr + svm
    else: raise ValueError('method outside frozen candidate family')
    order = np.lexsort((np.arange(len(ids)), lr, key))
    flagged = tuple(ids[int(i)] for i in order[:flagged_count])
    return Candidate(ids, flagged, method_id,
        canonical_hash({'ids':ids,'lr':np.asarray(lr_scores).tolist(),'svm':np.asarray(svm_scores).tolist(),'key':key.tolist()}),
        canonical_hash(flagged))

@dataclass(frozen=True)
class Design:
    status: str
    d: int
    T: int
    b_known: int
    pre_attention: int
    cap: int
    a: int
    n_max: int
    n: int = 0
    c: int = -1

@lru_cache(maxsize=None)
def exact_cdf(d, M, n, c):
    if min(d,M,n)<0 or M>d or n>d: raise ValueError('invalid hypergeometric parameters')
    # The retained rational API validates x<=n; extend only its support
    # convention for survival sums at t=0, c=1 and adjacent-cutoff checks.
    if c < 0: return Fraction(0)
    if c >= min(M,n): return Fraction(1)
    return RATIONAL.cdf(int(d), int(M), int(n), int(c))

@lru_cache(maxsize=None)
def verify_design(design):
    """Exact integers, independently of the numerical auditor's log-gamma numerical search."""
    if design.status != 'AUDIT': return {'status':design.status}
    d, m, n, c = design.d, design.a+1, design.n, design.c
    p = exact_cdf(d,m,n,c)
    previous = exact_cdf(d,m,n-1,c) if n else Fraction(1)
    at_max = exact_cdf(d,m,design.n_max,c)
    next_c = exact_cdf(d,m,design.n_max,c+1)
    if not (p <= ALPHA and previous > ALPHA and at_max <= ALPHA and next_c > ALPHA):
        raise ArithmeticError('Existing audit allocation failed the exact boundary/adjacency check')
    return {'boundary_probability':float(p),'previous_n_probability':float(previous),
        'at_n_max_probability':float(at_max),'next_c_at_n_max_probability':float(next_c),
        'boundary_le_1_over_20':True,'n_minimal':True,'c_maximal':True,
        'exact_boundary_fraction_sha256':hashlib.sha256((str(p.numerator)+'/'+str(p.denominator)).encode()).hexdigest()}

def plan_audit(*, d, T, b_known, pre_attention, cap):
    """Permitted fields only: neither unseen M nor evaluator truth is accepted."""
    values = (d,T,b_known,pre_attention,cap)
    if any(type(x) is not int or x < 0 for x in values) or pre_attention > cap:
        raise ValueError('invalid permitted counts or unaffordable census')
    a = T//9-b_known
    n_max = min(d,cap-pre_attention)
    base = dict(d=d,T=T,b_known=b_known,pre_attention=pre_attention,cap=cap,a=a,n_max=n_max)
    if T == 0: return Design('NO_FLAGGED_POSITIVES',**base)
    if a < 0: return Design('KNOWN_MISSES_PRECLUDE',**base)
    if a >= d: return Design('COUNT_ACCEPT',**base)
    result = AUDITOR.affordable_design(d,T,b_known,pre_attention,cap,float(ALPHA))
    # Keep the existing allocator, then resolve its floating-point boundary
    # comparisons using the retained rational implementation. In particular,
    # exact equality to 1/20 is feasible even if log-gamma rounds above .05.
    # Counts alone determine this correction; evaluator labels are unavailable.
    c = int(result['cutoff']) if result is not None else -1
    while exact_cdf(d,a+1,n_max,c+1) <= ALPHA:
        c += 1
    while c >= 0 and exact_cdf(d,a+1,n_max,c) > ALPHA:
        c -= 1
    if c < 0: return Design('AUDIT_UNAFFORDABLE',**base)
    n = int(result['n']) if result is not None else n_max
    while exact_cdf(d,a+1,n,c) > ALPHA:
        n += 1
    while n > 0 and exact_cdf(d,a+1,n-1,c) <= ALPHA:
        n -= 1
    design = Design('AUDIT',n=n,c=c,**base)
    verify_design(design)
    return design

def retrospective_metrics(design, *, evaluator_only_M):
    """Evaluator-only entry point, downstream of a frozen permitted plan."""
    M = evaluator_only_M
    if type(M) is not int or not 0 <= M <= design.d:
        raise ValueError('invalid evaluator tail count')
    total = design.T + design.b_known + M
    recall = design.T/total if total else 0.0
    adequate = bool(total > 0 and 10*design.T >= 9*total)
    p = 1.0 if design.status == 'COUNT_ACCEPT' else 0.0
    expected = 0.0
    if design.status == 'AUDIT':
        p = float(hypergeom.cdf(design.c,design.d,M,design.n))
        expected = float(np.sum(hypergeom.cdf(design.c,design.d,M,np.arange(design.n))))
    return {'evaluator_only_M':M,'evaluator_only_recall':recall,'evaluator_only_adequate':adequate,
        'conditional_accept_probability':p,'correct_accept_probability':p if adequate else 0.0,
        'false_accept_probability':p if not adequate else 0.0,'expected_audit_inspected':expected,
        'expected_total_attention':design.pre_attention+expected,
        'planned_total_attention':design.pre_attention+design.n}

def common_order(pool_ids, acquisition_seed, audit_base=971000000):
    """Label-independent common random keys; each restricted tail is uniform."""
    ids = tuple(pool_ids)
    if len(set(ids)) != len(ids): raise ValueError('duplicate IDs')
    rng = np.random.default_rng(np.random.SeedSequence([audit_base,acquisition_seed]))
    while True:
        keys = rng.random(len(ids))
        if len(np.unique(keys)) == len(ids): break
    return tuple(ids[int(i)] for i in np.argsort(keys))

class PaidOracle:
    """Simulation boundary. Every returned label is charged before access."""
    def __init__(self, truth, cap):
        self.__truth = dict(truth)
        if any(y not in (0,1) for y in self.__truth.values()): raise ValueError('invalid binary oracle label')
        self.cap = cap
        self.observed = {}
        self.events = []
        self.audit_attempt_started = False

    def purchase(self, window_id, purpose):
        if window_id in self.observed: return self.observed[window_id]
        if len(self.observed) >= self.cap: raise RuntimeError('attention cap exceeded')
        if window_id not in self.__truth: raise KeyError(window_id)
        # Append first-purchase attribution before returning label.
        self.events.append({'window_id':window_id,'purpose':purpose,'charge':1})
        label = self.__truth[window_id]
        if label not in (0,1): raise ValueError('invalid binary oracle label')
        self.observed[window_id] = int(label)
        return int(label)

class OneAttempt:
    def __init__(self): self.__spent = False

    def run(self, candidate, purchased_order, oracle, audit_order):
        if self.__spent: raise RuntimeError('one-audit risk already consumed')
        if oracle.audit_attempt_started: raise RuntimeError('this workflow already started an audit attempt')
        self.__spent = True
        oracle.audit_attempt_started = True
        L = tuple(purchased_order)
        U, F = set(candidate.pool_ids),set(candidate.flagged_ids)
        if len(set(L)) != len(L) or not set(L) <= U: raise ValueError('invalid purchased history')
        if set(oracle.observed) != set(L): raise ValueError('arm must start with exactly its own paid L')
        if len(oracle.events) != len(L): raise ValueError('startup/training charge missing')
        if set(audit_order) != U or len(audit_order) != len(U): raise ValueError('invalid common permutation')
        pre = len(set(L)|F)
        trace = {'method_id':candidate.method_id,'flagged_id_hash':candidate.flagged_id_hash,
            'training_charged_ids':list(L),'training_count':len(L),'planned_pre_audit_attention':pre,
            'T':None,'b_known':None,'unseen_d':None,'allowance_a':None,'cutoff_c':None,'planned_n':None}
        if pre > oracle.cap:
            trace.update(terminal_state='STOP_UNACCEPTED',terminal_reason='CENSUS_UNAFFORDABLE',
                actual_pre_audit_attention=len(L),actual_total_attention=len(L),audit_inspected=0,
                census_charged_ids=[],audit_charged_ids=[],observed_hits=0,ledger=list(oracle.events))
            return trace
        census = [w for w in candidate.flagged_ids if w not in oracle.observed]
        for w in census: oracle.purchase(w,'flagged_census')
        T = sum(oracle.observed[w] for w in F)
        b_known = sum(oracle.observed[w] for w in set(L)-F)
        D = U-set(oracle.observed)
        design = plan_audit(d=len(D),T=T,b_known=b_known,pre_attention=pre,cap=oracle.cap)
        trace.update(T=T,b_known=b_known,unseen_d=len(D),allowance_a=design.a,
            cutoff_c=design.c if design.status=='AUDIT' else None,planned_n=design.n,
            actual_pre_audit_attention=pre,census_charged_ids=census)
        if design.status == 'AUDIT':
            class RestrictedSampler:
                def sample(self, population, n):
                    if set(population) != D or n != design.n: raise RuntimeError('frozen sample changed')
                    return [w for w in audit_order if w in D][:n]
            before = len(oracle.events)
            result = AUDITOR.execute({'n':design.n,'cutoff':design.c},
                [w for w in candidate.pool_ids if w in D],oracle,RestrictedSampler(),early_stop=True)
            inspected = [e['window_id'] for e in oracle.events[before:]]
            observed = [oracle.observed[w] for w in inspected]
            accepted = result['accept']
            reason = 'FIXED_SAMPLE_PASS' if accepted else 'EARLY_FAILURE' if result['stopped_early'] else 'FIXED_SAMPLE_FAIL'
            if accepted and len(inspected) != design.n: raise RuntimeError('early acceptance')
        else:
            inspected,observed=[],[]
            accepted=design.status=='COUNT_ACCEPT'
            reason=design.status
        trace.update(audit_charged_ids=inspected,audit_observed_labels=observed,observed_hits=sum(observed),
            audit_inspected=len(inspected),actual_total_attention=len(oracle.observed),
            terminal_state='ACCEPT' if accepted else 'STOP_UNACCEPTED',terminal_reason=reason,
            ledger=list(oracle.events))
        if len(oracle.events) != len(oracle.observed) or len(oracle.observed)>oracle.cap:
            raise RuntimeError('ledger invariant violated')
        # Frozen classifier positives never change when tail positives are found.
        if trace['T'] != T: raise RuntimeError('audit positives reclassified')
        return trace
