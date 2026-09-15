from fractions import Fraction as F
from itertools import product
import math
import unittest

import numpy as np

import exact_checks as oracle
import pooled_trajectory_kernel as p
import trajectory_kernel as k


class PooledTrajectoryKernelTests(unittest.TestCase):
    def test_all_729_weak_order_pairs(self):
        one = [[1]]
        for f,g in product(product(range(3),repeat=3),repeat=2):
            expected = oracle.direct(tuple(zip(f,g)),[lambda z:z[0]],[lambda z:z[1]],[[F(1)]],[[F(1)]])
            self.assertEqual(p.pooled_u_score(np.array(f)[:,None],np.array(g)[:,None],one,one),float(expected))

    def test_signed_maps_against_complete_array(self):
        rng = np.random.default_rng(2026091407)
        C = [[.5,1/3,1/6],[-1,0,0]]
        D = [[0,1,0],[.5,-.5,0]]
        for n in (3,4,7,31,100,501):
            for ties in (False,True):
                f = rng.integers(-5,6,(n,2)) if ties else rng.normal(size=(n,2))
                g = rng.integers(-3,4,(n,2)) if ties else rng.normal(size=(n,2))
                expected = k.full_u_scores(f[None],g[None],C,D)[0]
                self.assertAlmostEqual(p.pooled_u_score(f,g,C,D),expected,places=14)

    def test_concordances_match_literal_pairs(self):
        x = np.array([2,1,2,3,1,2])
        y = np.array([0,2,1,1,2,3])
        xcode,_,_ = p._rank_codes_and_centered_sums(x)
        ycode,m,_ = p._rank_codes_and_centered_sums(y)
        expected = sum(int(np.sign(x[i]-x[j]))*int(np.sign(y[i]-y[j])) for i in range(len(x)) for j in range(i))
        self.assertEqual(p._concordance_minus_discordance(xcode,ycode,m),expected)

    def test_large_exact_integers_and_ties(self):
        for dtype,base in ((np.int64,2**63-4),(np.uint64,2**64-4),(object,2**100)):
            f = np.array([[base],[base+1],[base+2],[base+2]],dtype=dtype)
            actual = p.pooled_u_score(f,f,[[1]],[[1]])
            expected = k.full_u_scores(f[None],f[None],[[1]],[[1]])[0]
            self.assertEqual(actual,expected)

    def test_tiny_alpha_and_effective_size(self):
        f = np.arange(7)[:,None]
        out = p.pooled_u_hoeffding_lower(f,f,[[1]],[[1]],float(np.nextafter(0.,1.)))
        self.assertEqual(out['independent_triples'],2)
        self.assertTrue(math.isfinite(out['radius']))
        expected = (1/6)*math.sqrt(-math.log(float(np.nextafter(0.,1.)))/4)
        self.assertEqual(out['radius'],expected)

    def test_bad_inputs(self):
        for f,g in ((np.zeros((2,1)),np.zeros((2,1))),(np.zeros((3,1)),np.zeros((4,1))),(np.full((3,1),np.nan),np.zeros((3,1)))):
            with self.assertRaises(ValueError):
                p.pooled_u_score(f,g,[[1]],[[1]])

    def test_joint_radius_matches_independent_formula(self):
        f = np.array([0,1,1,1,0,0,2,1,0,1])[:,None]
        g = np.array([1,0,1,2,0,1,1,1,0,0])[:,None]
        out = p.pooled_u_joint_lower(f,g,[[1]],[[1]],.013)
        blocks = [float(oracle.direct(tuple(zip(f[i:i+3,0],g[i:i+3,0])),[lambda z:z[0]],[lambda z:z[1]],[[F(1)]],[[F(1)]])) for i in (0,3,6)]
        variance = sum((z-sum(blocks)/3)**2 for z in blocks)/2
        x, width = math.log(2/.013), 1/6
        expected = min(width*math.sqrt(x/6), math.sqrt(variance)*math.sqrt(2*x/3)+(2*width/math.sqrt(6)+width/9)*x)
        self.assertAlmostEqual(out['radius'],expected)
        self.assertAlmostEqual(out['triple_sample_variance'],variance)
        self.assertEqual(out['variance_draws'],9)

    def test_joint_degenerate_and_tiny_alpha(self):
        f = np.zeros((7,1))
        out = p.pooled_u_joint_lower(f,f,[[1]],[[1]],float(np.nextafter(0.,1.)))
        self.assertTrue(math.isfinite(out['radius']))
        self.assertEqual(out['triple_sample_variance'],0)
        zero = p.pooled_u_joint_lower(f,f,[[1,-1]],[[1,0]])
        self.assertEqual(zero['lower'],0)
        short = p.pooled_u_joint_lower(f[:4],f[:4],[[1]],[[1]])
        self.assertEqual(short['variance_radius'],None)


if __name__ == '__main__':
    unittest.main()
