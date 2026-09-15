"""Exact finite-support full-U summaries for binary V,W and one category.

State order is (0,0),(0,1),(1,0),(1,1), coded 2*V+W. See BINARY_BENCHMARK.md.
The constant-time covariance formula is independently checked against all 256
ordered type quadruples and the original quadratic row algorithm.
"""

from fractions import Fraction
from itertools import permutations, product
import math

import numpy as np


STATES = ((0,0),(0,1),(1,0),(1,1))


def _counts(values, minimum=4):
    if len(values) != 4:
        raise ValueError("four state counts in 00,01,10,11 order are required")
    if any(isinstance(x,(bool,np.bool_)) or not isinstance(x,(int,np.integer)) or x < 0 for x in values):
        raise ValueError("state counts must be nonnegative integers")
    counts=tuple(int(x) for x in values)
    if sum(counts) < minimum:
        raise ValueError(f"at least {minimum} sample positions are required")
    return counts


def exact_u_counts(counts):
    """Exact Fraction equal to full U3-U4, with every duplicate position kept."""
    counts=_counts(counts)
    n=sum(counts)
    sum_v=counts[2]+counts[3]
    sum_w=counts[1]+counts[3]
    return Fraction(n*counts[3]-sum_v*sum_w,4*n*(n-1))


def full_u_binary_counts(counts):
    """Constant-time exact center and jackknife summary from four counts."""
    counts=_counts(counts,minimum=5)
    n=sum(counts)
    center=exact_u_counts(counts)
    deleted=[]
    for s,count in enumerate(counts):
        if count:
            after=list(counts)
            after[s]-=1
            deleted.append(exact_u_counts(after))
        else:
            deleted.append(None)
    deletion_mean=sum((counts[s]*deleted[s] for s in range(4) if counts[s]),Fraction())/n
    assert deletion_mean == center
    variance=Fraction(n-1,n)*sum((counts[s]*(deleted[s]-center)**2
                                for s in range(4) if counts[s]),Fraction())
    return {"n":n,"counts":list(counts),"estimate":float(center),
            "estimate_exact":str(center),"first_term":float(center+Fraction(1,4)),
            "second_term":.25,"jackknife_variance":float(variance),
            "jackknife_variance_exact":str(variance),
            "delete_one_by_state":[None if x is None else float(x) for x in deleted],
            "delete_one_exact_by_state":[None if x is None else str(x) for x in deleted],
            "delete_average_identity_error_exact":str(deletion_mean-center)}


def ordered_quadruple_oracle(counts):
    """Independent literal 4^4 type enumeration with falling multiplicities."""
    counts=_counts(counts)
    numerator_scaled=0
    for types in product(range(4),repeat=4):
        remaining=list(counts)
        multiplicity=1
        for s in types:
            multiplicity*=remaining[s]
            remaining[s]-=1
        if not multiplicity:
            continue
        i,j,k,l=(STATES[s] for s in types)
        aij=1+i[0]-j[0]
        bik=1+i[1]-k[1]
        aik=1+i[0]-k[0]
        bjl=1+j[1]-l[1]
        numerator_scaled += multiplicity*(aij*bik-aik*bjl)
    return Fraction(numerator_scaled,4*math.perm(sum(counts),4))


def exact_symmetrized_block(types):
    """Literal 24-permutation h_s, used only to construct a lookup table."""
    total=0
    for i,j,k,l in permutations(range(4)):
        vi,wi=STATES[int(types[i])]
        vj,wj=STATES[int(types[j])]
        vk,wk=STATES[int(types[k])]
        vl,wl=STATES[int(types[l])]
        total += (1+vi-vj)*(1+wi-wk)-(1+vi-vk)*(1+wj-wl)
    return Fraction(total,96)


BLOCK_VALUES_EXACT=tuple(exact_symmetrized_block(types)
                         for types in product(range(4),repeat=4))
BLOCK_VALUES=np.asarray([float(x) for x in BLOCK_VALUES_EXACT])


def binary_block_values(states):
    """Disjoint consecutive four-row blocks; every block consumes four rows."""
    states=np.asarray(states)
    if states.ndim != 1 or states.dtype.kind not in "iu" or np.any((states<0)|(states>3)):
        raise ValueError("states must be an integer vector of codes 0..3")
    q=len(states)//4
    if q == 0:
        raise ValueError("at least four positions are required")
    rows=states[:4*q].reshape(q,4)
    codes=rows[:,0]*64+rows[:,1]*16+rows[:,2]*4+rows[:,3]
    return BLOCK_VALUES[codes]


def binary_pair_values(states):
    """All-raw disjoint pair covariance scores: (V1-V2)(W1-W2)/8."""
    states=np.asarray(states)
    if states.ndim != 1 or states.dtype.kind not in "iu" or np.any((states<0)|(states>3)):
        raise ValueError("states must be an integer vector of codes 0..3")
    q=len(states)//2
    if q == 0:
        raise ValueError("at least two positions are required")
    rows=states[:2*q].reshape(q,2)
    dv=(rows[:,0]//2).astype(int)-(rows[:,1]//2).astype(int)
    dw=(rows[:,0]%2).astype(int)-(rows[:,1]%2).astype(int)
    return dv*dw/8.0


def full_u_binary_bound(summary,alpha=.05,rule="joint",binary_interactions=False):
    """Frozen generic or binary-specific EB, Hoeffding, or joint calibration."""
    if not 0<alpha<1:
        raise ValueError("alpha must lie in (0,1)")
    if rule not in ("hoeffding","eb","joint"):
        raise ValueError("rule must be hoeffding, eb or joint")
    n=summary["n"]
    if n<5:
        raise ValueError("at least five positions required")
    alpha_component=alpha/2 if rule=="joint" else alpha
    h_radius=(1/6)*math.sqrt(-math.log(alpha_component)/(2*(n//4)))
    A=.5
    B=1.0 if binary_interactions else 2.0
    M,J=A/n,B/(n-1)
    Mp,Jp=A/(n-1),B/(n-2)
    x=math.log(2)-math.log(alpha_component)
    eb_radius=(math.sqrt(2*summary["jackknife_variance"]*x)
               +x*math.sqrt((n-1)/n*(4*Mp*Mp+16*Jp*Jp))
               +(2*M/3+J)*x)
    radius=h_radius if rule=="hoeffding" else eb_radius if rule=="eb" else min(h_radius,eb_radius)
    return {"estimate":summary["estimate"],"lower":summary["estimate"]-radius,
            "radius":radius,"alpha":alpha,"rule":rule,
            "binary_interactions":binary_interactions,
            "hoeffding_radius":h_radius,"eb_radius":eb_radius}
