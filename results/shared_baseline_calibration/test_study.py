"""Independent targets, literal kernels and cost checks before Monte Carlo."""
from fractions import Fraction as F
from itertools import permutations, product
import importlib.util
import math
import os
from pathlib import Path
import sys
import unittest

import numpy as np
import study as s

def package_path():
    explicit = os.environ.get("CERTIFYING_PACKAGE_ROOT")
    choices = [Path(explicit)] if explicit else [Path(__file__).resolve().parents[2],
               Path(__file__).resolve().parents[3]/"release/certifying_forecasting_skill"]
    for path in choices:
        if (path/"results/trajectory_budget/reference_kernels/trajectory_kernel.py").is_file():
            return path
    raise FileNotFoundError("Set CERTIFYING_PACKAGE_ROOT to the released code package")

PACKAGE = package_path()
sys.path.insert(0,str(PACKAGE/"results/trajectory_budget/reference_kernels"))
from trajectory_kernel import shared_scores, validation_scores, full_u_scores
from pooled_trajectory_kernel import pooled_u_score


def literal_cmp(x,y):return F(int(x>y)-int(x<y),2)


def literal_kernels(h,y,b,ix):
    def p(i,j):return literal_cmp(h[i],h[j])-literal_cmp(b[i],b[j])
    def q(i,j):return literal_cmp(y[i],y[j])-literal_cmp(b[i],b[j])
    i,j,k=ix
    shared=p(i,j)*q(i,j)
    correction=(p(i,j)-p(i,k))*(q(i,j)-q(i,k))/2
    u=sum((p(a,bb)*q(a,c) for a,bb,c in permutations(ix)),F())/6
    return shared,correction,u


class StudyTests(unittest.TestCase):
    def test_exact_targets_against_closed_forms(self):
        for name,cells in s.laws():
            targets={r["candidate"]:F(r["theta_exact"]) for r in s.exact_law(cells)}
            self.assertEqual(targets["baseline_identity"],0)
            if name=="informative_nonlinear_baseline":
                self.assertEqual(targets["candidate"],F(7,54))
                self.assertEqual(targets["reverse_candidate"],F(1,54))
                self.assertEqual(targets["constant"],F(2,27))
            else:
                rho=F(1,10) if name=="tied_weak" else F(1) if name.endswith("positive") else F(0)
                probs=[F(1,100),F(49,50),F(1,100)] if name.startswith("near") else [F(1,3)]*3
                variance=(1-sum(p**3 for p in probs))/12
                self.assertEqual(targets["candidate"],rho*variance)
                self.assertEqual(targets["reverse_candidate"],-rho*variance)
                self.assertEqual(targets["constant"],0)

    def test_exact_expectations_and_range_from_literal_role_enumeration(self):
        for _,cells in s.laws():
            hm,y,b=s.candidate_maps(cells)
            targets=s.exact_law(cells)
            for col in range(4):
                eq,eu=F(),F()
                for ix in product(range(len(cells)),repeat=3):
                    weight=math.prod(cells[i][3] for i in ix)
                    ss,qq,uu=literal_kernels(hm[:,col],y,b,ix)
                    self.assertTrue(F(-1,4)<=ss<=1)
                    self.assertTrue(F(-1,2)<=qq<=2)
                    self.assertTrue(F(-1,6)<=uu<=F(1,3))
                    eq+=weight*qq;eu+=weight*uu
                self.assertEqual(eu,F(targets[col]["theta_exact"]))
                self.assertEqual(eq,F(targets[col]["interaction_exact"]))

    def test_complete_center_against_literal_distinct_indices_and_public_rank_sweep(self):
        C=np.asarray([[1.],[-1.]])
        for _,cells in s.laws():
            maps,y,b=s.candidate_maps(cells)
            stream=np.arange(8)%len(cells)
            for col,(p,q) in enumerate(s.kernels(cells)):
                observed=s.complete_center(p,q,stream)
                ref=sum(literal_kernels(maps[:,col],y,b,tuple(stream[j] for j in ix))[2]
                        for ix in permutations(range(8),3))/math.perm(8,3)
                self.assertAlmostEqual(observed,float(ref),places=15)
                self.assertAlmostEqual(observed,pooled_u_score(np.column_stack([maps[stream,col],b[stream]]),
                                        np.column_stack([y[stream],b[stream]]),C,C),places=15)

    def test_independent_public_block_kernel_parity_with_ties(self):
        C=np.asarray([[1.],[-1.]])
        for _,cells in s.laws():
            maps,y,b=s.candidate_maps(cells)
            ix=np.asarray(list(product(range(len(cells)),repeat=3)))
            for col,(p,q) in enumerate(s.kernels(cells)):
                f=np.column_stack([maps[:,col],b]);g=np.column_stack([y,b])
                actual_s=p[ix[:,0],ix[:,1]]*q[ix[:,0],ix[:,1]]
                actual_q=.5*(p[ix[:,0],ix[:,1]]-p[ix[:,0],ix[:,2]])*(q[ix[:,0],ix[:,1]]-q[ix[:,0],ix[:,2]])
                actual_u=s.triple_values(p,q,ix.reshape(-1))
                np.testing.assert_allclose(actual_s,shared_scores(f[ix[:,:2]],g[ix[:,:2]],C,C),atol=1e-16,rtol=0)
                np.testing.assert_allclose(actual_q,validation_scores(f[ix],g[ix],C,C),atol=1e-16,rtol=0)
                np.testing.assert_allclose(actual_u,full_u_scores(f[ix],g[ix],C,C),atol=1e-16,rtol=0)

    def test_budget_allocation_and_complete_comparator_access(self):
        for budget in s.BUDGETS:
            plan=s.allocation(budget)
            self.assertLessEqual(2*plan["M"]+3*plan["n"],budget)
            self.assertLess(plan["unused_budget"],2)
            self.assertEqual(budget%3,0)
        for budget in range(5,41):
            got=s.allocation(budget)
            possibilities=[]
            for m in range(1,budget//2+1):
                for n in range(1,budget//3+1):
                    if 2*m+3*n<=budget:
                        possibilities.append(1.25*math.sqrt(-math.log(s.ALPHA-s.DELTA)/(2*m))+
                                             2.5*math.sqrt(-math.log(s.DELTA)/(2*n)))
            self.assertAlmostEqual(got["range_radius"],min(possibilities),places=14)

    def test_declared_identity_zero_but_observed_ties_do_not_set_width_zero(self):
        cells=s.laws()[0][1]
        stream=np.zeros(30,dtype=int) # Accidental identical states do not change declared widths.
        rows=s.evaluate(cells,stream,s.allocation(30))
        for row in rows:
            if row["candidate"]=="baseline_identity":
                self.assertEqual((row["center"],row["radius"],row["lower"]),(0,0,0))
            else:
                self.assertGreater(row["radius"],0)

    def test_final_radius_formulas_against_released_sharp_rule(self):
        spec=importlib.util.spec_from_file_location("released_sharp",PACKAGE/"results/shared_baseline_ranges/reproduce.py")
        ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
        for n,var,width,alpha in product((2,31,4096),(0.,.001,.125),(0.,.5,1.25,2.5),(s.ALPHA,s.DELTA)):
            for mode in s.RULES:
                old="hoeffding" if mode=="range" else "hybrid"
                self.assertAlmostEqual(s.mean_radius(n,var,width,alpha,mode),ref.mean_radius(n,var,width,alpha,old),places=13)
                self.assertAlmostEqual(s.pooled_radius(n,var,width,alpha,mode),ref.complete_radius(n,var,width,alpha,old),places=13)

    def test_monte_carlo_precision_and_null_definition(self):
        self.assertAlmostEqual(s.interval(0,300)[1],1-.025**(1/300),places=14)
        self.assertAlmostEqual(s.interval(300,300)[0],.025**(1/300),places=14)
        for _,cells in s.laws():
            for target in s.exact_law(cells):
                self.assertEqual(target["null"],F(target["theta_exact"])<=0)
        # Nonlinear repackaging can have a positive declared rank target.
        info=dict(s.laws())["informative_nonlinear_baseline"]
        self.assertGreater(s.exact_law(info)[1]["theta"],0)


if __name__=="__main__":unittest.main(verbosity=2)
