"""Independent small-law and signed-allowance checks for the rank design."""
from fractions import Fraction
from itertools import product
import unittest
import numpy as np
from reference_time_audit import audit, panel_scores


class CompositionTests(unittest.TestCase):
    def setUp(self):
        self.c = np.array([[1.,0.],[0.,0.],[-1.,0.]])
        self.d = np.array([[0.,-.5],[1.,0.],[-.5,0.]])

    def test_positive_zero_negative_overlap_exact_ternary(self):
        panels = np.array(list(product([-2,0,5], repeat=9))).reshape(-1,3,3)
        cases = [self.d, np.array([[-.5,0.],[1.,0.],[0.,-.5]]), np.array([[-.5,0.],[1.,0.],[-.5,0.]])]
        for d, expected in zip(cases,[Fraction(5,108),-Fraction(5,108),Fraction(0)]):
            self.assertAlmostEqual(float(panel_scores(panels,self.c,d).mean()),float(expected),places=15)

    def test_zero_population_expectation_exact(self):
        total=Fraction(0)
        for a,b,c in product(range(3), repeat=3):
            r=[Fraction(2*x+1,6) for x in [a,b,c]]
            total+=(r[0]-r[2])*(r[1]-(r[0]+r[2])/2)
        self.assertEqual(total,0)

    def test_large_integer_order_and_validation_are_preserved(self):
        x=2**54
        panels=np.array([[[x+1,x]]],dtype=np.int64)
        self.assertEqual(panel_scores(panels,[[1]],[[1]])[0],.25)
        triples=np.array([[x+1,x,x+2]],dtype=np.int64)
        r=audit(panels,[[1]],[[1]],triples)
        self.assertEqual(r['omega_hat'],.5)
        self.assertEqual(r['score_mean'],.25)

    def test_positive_overlap_uses_upper_variance(self):
        panels=np.zeros((4,3,3));val=np.zeros((8,3))
        r=audit(panels,self.c,self.d,val)
        self.assertEqual(r['upper_interaction_allowance'],.5*r['omega_interval'][1])
        self.assertLess(r['lower_bound'],0)
        self.assertEqual(r['total_observation_count'],60)

    def test_negative_overlap_uses_lower_variance(self):
        rng=np.random.default_rng(243)
        r=audit(rng.integers(0,3,(40,3,3)),self.c,-self.d,rng.integers(0,3,(200,3)))
        self.assertEqual(r['upper_interaction_allowance'],-.5*r['omega_interval'][0])

    def test_zero_design_is_zero(self):
        r=audit(np.ones((2,3,3)),np.zeros((3,2)),self.d,np.ones((3,3)))
        self.assertEqual(r['lower_bound'],0)
        self.assertFalse(r['positive_lower_bound'])

    def test_all_ties_and_input_validation(self):
        self.assertTrue(np.all(panel_scores(np.ones((3,3,3)),self.c,self.d)==0))
        for args in [(np.ones((2,2,3)),self.c,self.d,np.ones((3,3))),
                     (np.ones((2,3,3)),self.c,self.d,np.empty((0,3)))]:
            with self.assertRaises(ValueError):audit(*args)
        with self.assertRaises(ValueError):audit(np.ones((2,3,3)),self.c,self.d,np.ones((3,3)),alpha=.01,delta=.01)


if __name__=='__main__':unittest.main()
