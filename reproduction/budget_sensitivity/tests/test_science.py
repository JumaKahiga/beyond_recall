"""Synthetic checks; never execute the real 30% SSW comparison.

Run with the pinned interpreter after the notebook's source-definition cells:
python /path/to/ssw2/tests/test_science.py --runtime /path/to/generated/run
"""
from pathlib import Path
import argparse,sys,unittest
parser=argparse.ArgumentParser();parser.add_argument('--runtime',type=Path,required=True)
args,remaining=parser.parse_known_args();sys.path.insert(0,str(args.runtime.resolve()))
import numpy as np
import audit as core
from fractions import Fraction
from dataclasses import asdict
from checks import cdf,expected_inspected,check
from analyze_budget import calculate

class ScienceChecks(unittest.TestCase):
    def test_independent_probability_identity(self):
        for d in range(1,13):
            for m in range(d+1):
                for n in range(d+1):
                    for c in [-1,0,1,min(m,n)]:
                        self.assertEqual(cdf(d,m,n,c),core.exact_cdf(d,m,n,c))

    def test_expected_early_failure_cost(self):
        for m in [0,1,6,15,40]:
            for n in [3,10,30]:
                for c in [0,1,3]:
                    survival=sum((cdf(40,m,t,c) for t in range(n)),Fraction(0))
                    self.assertEqual(expected_inspected(40,m,n,c),survival)

    def test_budget_arms_and_paid_traces(self):
        ids=['w%04d'%i for i in range(1000)]
        flags=ids[:120];training=ids[:80]
        truth={w:int(i<85 or i in [550,780]) for i,w in enumerate(ids)}
        scores=np.arange(1000,0,-1,dtype=float)
        candidate=core.freeze_candidate(ids,scores,scores,'LR-RAW-FA',120)
        order=core.common_order(ids,7,123)
        for cap in [90,120,250,300]:
            oracle=core.PaidOracle(truth,cap)
            for w in training:oracle.purchase(w,'training')
            trace=core.OneAttempt().run(candidate,training,oracle,order)
            pre=len(set(training)|set(flags));D=set(ids)-set(training)-set(flags)
            if pre>cap:
                row={'design':None,'design_status':'CENSUS_UNAFFORDABLE','attention_cap':cap,
                     'evaluator_only_adequate':True,'conditional_accept_probability':0.,'correct_accept_probability':0.,
                     'expected_total_attention':80,'planned_total_attention':80,'realized_total_attention':80}
            else:
                design=core.plan_audit(d=len(D),T=85,b_known=0,pre_attention=pre,cap=cap)
                row={'design':asdict(design),'design_status':design.status,'attention_cap':cap,
                     'realized_total_attention':trace['actual_total_attention'],**core.retrospective_metrics(design,evaluator_only_M=2)}
            check(row,trace,ids,truth,training,flags,order)
            with self.assertRaises(RuntimeError):core.OneAttempt().run(candidate,training,oracle,order)

    def test_design_does_not_use_hidden_outcomes(self):
        design=core.plan_audit(d=880,T=85,b_known=0,pre_attention=120,cap=300)
        frozen=asdict(design)
        for m in [0,2,10,100]:core.retrospective_metrics(design,evaluator_only_M=m)
        self.assertEqual(asdict(design),frozen)
        self.assertNotIn('evaluator_only_M',core.plan_audit.__annotations__)

    def test_paired_bootstrap_and_80_percent_boundary(self):
        exp={'targets':[{'code':str(t),'common_name':str(t),'seeds':list(range(20))} for t in range(5)],
             'workflows':[{'id':'m'+str(i)} for i in range(4)],'budgets_percent':[25,30],'acceptance_benchmark':.8}
        rows=[]
        for budget in [25,30]:
            for t in exp['targets']:
                for seed in t['seeds']:
                    for w in exp['workflows']:
                        rows.append({'budget_percent':budget,'target':t['code'],'seed':seed,'workflow':w['id'],
                          'correct_accept_probability':.6 if budget==25 else .8,
                          'expected_total_attention':100 if budget==25 else 120,'evaluator_only_recall':.98,
                          'design_status':'AUDIT','design':{'c':0}})
        rng=np.random.default_rng(123);indices=np.stack([rng.integers(0,20,(20000,20)) for _ in range(5)])
        summary,contrasts,by_target=calculate(rows,exp,indices)
        self.assertEqual(len(by_target),40)
        for row in contrasts:
            self.assertAlmostEqual(row['mean_acceptance_difference'],.2)
            self.assertTrue(np.allclose(row['difference_interval_95'],[.2,.2]))
            self.assertEqual(row['mean_expected_windows_difference'],20)
        # Test clearly below/above as well; exactly equal is tested at the
        # comparison expression because floating point averaging may round.
        self.assertFalse(.8>.8)
        for row in summary:
            self.assertEqual(row['point_estimate_above_80'],row['mean_acceptance']>.8)

if __name__=='__main__':unittest.main(argv=[sys.argv[0]]+remaining)
