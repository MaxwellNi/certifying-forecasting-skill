"""Boundary regressions for finite reference certificates and raw comparisons."""
from pathlib import Path
import math
import sys
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/analysis'))
sys.path.insert(0,str(ROOT/'results/aggregate_bias'))
from reference_certificate_efficiency import one_sided_certificate, planned_triples, triple_scores
from aggregate_bias_bound import aggregate_upper_bound


class ReferenceBoundaryPrecisionTests(unittest.TestCase):
    def test_representable_tiny_probability_levels(self):
        for alpha,delta in ((1e-310,1e-312),(1e-320,1e-322),(np.nextafter(1e-310,1.),1e-310)):
            for kind in ('range','variance','independent_bernstein'):
                result=one_sided_certificate(.25,np.full(50000,.25),50000,[.5],[.5],0.,
                                             alpha=alpha,delta=delta,kind=kind)
                self.assertTrue(math.isfinite(result['lower_bound']))
                self.assertEqual(result['reject'],result['lower_bound']>0)
                self.assertTrue(delta<=result['p']<=1.)
        self.assertGreater(planned_triples(.1,0,[.5],[.5],alpha=1e-310,delta=1e-312,beta=1e-310),0)

    def test_aggregate_tiny_probability_and_missing_categories(self):
        for delta in (1e-310,1e-320,np.nextafter(0.,1.)):
            result=aggregate_upper_bound([1.],[0.],[0.],[100],[100],delta)
            x=math.log(3)-math.log(delta)
            self.assertAlmostEqual(result['upper'],math.sqrt(2*x)/400+x/400,places=13)
            absent=aggregate_upper_bound([1.],[0.],[0.],[0],[0],delta)
            self.assertEqual(absent['upper'],1.)

    def test_strict_decisions_near_a_range_boundary(self):
        for alpha,delta in ((.05,.0001),(.5,.01),(1e-310,1e-312)):
            result=one_sided_certificate(.2,None,50000,[.5],[.5],0.,alpha=alpha,delta=delta,kind='range')
            middle=.2-result['radius']
            for bias in (np.nextafter(middle,-np.inf),middle,np.nextafter(middle,np.inf)):
                result=one_sided_certificate(.2,None,50000,[.5],[.5],bias,alpha=alpha,delta=delta,kind='range')
                self.assertEqual(result['reject'],result['p']<alpha)
                self.assertEqual(result['reject'],result['lower_bound']>0)
        # This test does not claim that binary64 provides globally directed
        # rounding at every mathematically equal threshold.
        for alpha,delta in ((.05,.05),(.04,.05)):
            with self.assertRaises(ValueError):
                one_sided_certificate(0,[0,0],2,[.5],[.5],0.,alpha=alpha,delta=delta)

    def test_integer_and_mixed_numeric_ordering(self):
        base=np.array([[0,1,2]],dtype=np.int64)
        f=np.full((1,3),.5)
        expected=triple_scores(base,base,f,f)
        examples=(base.astype(object)+2**90,
                  np.array([[2**63-3,2**63-2,2**63-1]],dtype=np.int64),
                  np.array([[2**64-3,2**64-2,2**64-1]],dtype=np.uint64),
                  np.array([[2**53,2**53+1,float(2**53+2)]],dtype=object))
        for values in examples:
            np.testing.assert_array_equal(expected,triple_scores(values,values,f,f))
        for invalid in ([[0,float('inf'),1]],[[0,'1',2]],[[0,complex(1),2]]):
            with self.assertRaises(ValueError):triple_scores(invalid,invalid,f,f)


if __name__=='__main__':
    unittest.main()
