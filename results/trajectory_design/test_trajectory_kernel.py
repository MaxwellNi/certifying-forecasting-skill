"""Independent literal-rank oracle checks and numerical boundary regressions."""

from fractions import Fraction as F
from itertools import product
import math
import unittest

import numpy as np

import exact_checks as oracle
import trajectory_kernel as k


class TrajectoryKernelTests(unittest.TestCase):
    def test_all_roles_against_literal_oracle(self):
        fs = [lambda z: z[0] ^ z[1], lambda z: 2*z[0] + z[1]]
        gs = [lambda z: z[0]*z[1], lambda z: z[0]-z[1]]
        C = [[F(1,2), F(1,3), F(1,6)], [F(-1), F(0), F(0)]]
        D = [[F(0), F(1), F(0)], [F(1,2), F(-1,2), F(0)]]
        points = [(0,0), (1,0), (1,1)]
        draws = list(product(points, repeat=4))
        f = np.array([[[fn(z) for fn in fs] for z in draw] for draw in draws])
        g = np.array([[[fn(z) for fn in gs] for z in draw] for draw in draws])
        for fn, full in ((k.distinct_focal_scores, False), (k.full_u_scores, True)):
            expected = [float(oracle.direct(z, fs, gs, C, D, full)) for z in draws]
            np.testing.assert_allclose(fn(f,g,C,D), expected, atol=2e-16)
        expected = [float(oracle.shared(z,fs,gs,C,D)) for z in draws]
        np.testing.assert_allclose(k.shared_scores(f,g,C,D), expected, atol=2e-16)
        omega = [[oracle.dot(x,y) for y in D] for x in C]
        expected = [float(oracle.validator(z[:3],fs,gs,omega)) for z in draws]
        np.testing.assert_allclose(k.validation_scores(f[:,:3],g[:,:3],C,D), expected, atol=2e-16)

    def test_integer_order_and_ties_preserved(self):
        for dtype, base in ((np.int64, 2**63-4), (np.uint64, 2**64-4), (object, 2**100)):
            maps = np.array([[[base], [base+1], [base+2]]], dtype=dtype)
            C, D = [[1,0]], [[1,0]]
            np.testing.assert_equal(k.shared_scores(maps,maps,C,D), [0.25])
            np.testing.assert_allclose(k.full_u_scores(maps,maps,C,D), [1/12])
            maps[0,2,0] = base+1
            np.testing.assert_allclose(k.full_u_scores(maps,maps,C,D), [1/12])

    def test_support_widths(self):
        C, D = [[1,-1],[0,1]], [[0,2],[-1,0]]
        widths = k.design_widths(C,D)
        self.assertEqual(widths['score_width'], 4.5)
        self.assertEqual(widths['validation_width'], 5)
        self.assertEqual(widths['direct_width'], 1.5)
        self.assertEqual(widths['full_u_width'], .5)

    def test_full_u_sharp_map_pair_width_including_ties(self):
        orders = list(product(range(3), repeat=3))
        pairs = list(product(orders, repeat=2))
        f = np.array([[list(row) for row in zip(x)] for x, _ in pairs])
        g = np.array([[list(row) for row in zip(y)] for _, y in pairs])
        scores = k.full_u_scores(f,g,[[1]],[[1]])
        self.assertEqual(float(scores.min()), -1/12)
        self.assertEqual(float(scores.max()), 1/12)

    def test_hybrid_uses_union_budget(self):
        values = [0., .2, -.1, .4]
        out = k.mean_bound(values, 1, .05)
        variance = sum((x-sum(values)/4)**2 for x in values)/3
        h = math.sqrt(math.log(2/.05)/8)
        eb = math.sqrt(2*variance*math.log(4/.05)/4) + 7*math.log(4/.05)/9
        self.assertAlmostEqual(out['radius'], min(h, eb))
        self.assertAlmostEqual(out['lower'], sum(values)/4-min(h,eb))

    def test_tiny_alpha_stays_finite(self):
        alpha = float(np.nextafter(0.,1.))
        for rule in ('hoeffding','empirical_bernstein','hybrid'):
            out = k.mean_bound([0.,0.,.25], .5, alpha, rule=rule)
            self.assertTrue(math.isfinite(out['radius']))

    def test_one_value_and_zero_width(self):
        out = k.mean_bound([.1], .5, .05)
        self.assertAlmostEqual(out['radius'], .5*math.sqrt(math.log(20)/2))
        self.assertEqual(k.mean_bound([.1,.1],0)['lower'], .1)
        with self.assertRaises(ValueError):
            k.mean_bound([.1], .5, rule='empirical_bernstein')

    def test_signed_correction_and_cost(self):
        out = k.corrected_lower([.1,.2],[-.1,-.1],[[1,0]],[[1,0]],alpha=.05,delta=.01)
        self.assertAlmostEqual(out['estimate'], .25)
        self.assertEqual(out['trajectory_draw_calls'], 12)
        self.assertEqual(out['validation_count'],2)
        direct = k.corrected_lower([.1,.2],None,[[1,0]],[[0,1]])
        self.assertEqual(direct['delta'],0)
        self.assertEqual(direct['trajectory_draw_calls'],6)

    def test_nonfinite_and_shape_rejections(self):
        maps = np.zeros((1,3,1))
        for fn in (k.shared_scores,k.validation_scores,k.full_u_scores):
            with self.assertRaises(ValueError):
                fn(maps+np.nan,maps,[[1,0]],[[1,0]])
        with self.assertRaises(ValueError):
            k.shared_scores(maps,maps,[[1]],[[1]])
        with self.assertRaises(ValueError):
            k.design_widths([[1e308]],[[1e308]])
        for values,width in (([0,2],1),([0,1e-17],0),([],1)):
            with self.assertRaises(ValueError):
                k.mean_bound(values,width)
        for alpha in (0,1,float('nan')):
            with self.assertRaises(ValueError):
                k.mean_bound([0,1],1,alpha)


if __name__ == '__main__':
    unittest.main()
