"""Certified fixed-fraction betting for independent finite-support scores.

Initialize once per predeclared score support and lower bound, then pass support
counts for each independent replicate. No optimizer, outcome-selected fraction,
or outcome-dependent support restriction is used. See FINITE_SUPPORT_BETTING.md.
"""

from decimal import Decimal
from fractions import Fraction
import math
import operator
import time

import numpy as np

from profiled_betting import (_contexts, _fraction, _log_fraction_interval,
                             _to_float_down, _to_float_up)


DEFAULT_FRACTIONS = tuple(float(v) for v in np.geomspace(1e-4, .99, 64))


class FiniteSupportBetting:
    """Mixture of products ``prod(1 + fraction * score / (-lower))``.

    All score observations must be independent, with conditional means <= 0
    under the null and values in the predeclared support. The binary experiment
    uses IID disjoint block scores. Counts preserve their fixed-sample product.
    Float inputs mean their exact represented dyadic values. Use Fraction for
    exact non-dyadic score values and lower bounds, e.g. Fraction(-1, 12).
    """

    def __init__(self, support, lower, fractions=None, *, precision=40):
        started = time.perf_counter()
        self.support = tuple(_fraction(v) for v in support)
        self.lower = _fraction(lower)
        self.fractions = tuple(_fraction(v) for v in
                               (DEFAULT_FRACTIONS if fractions is None else fractions))
        if not self.support or len(set(self.support)) != len(self.support):
            raise ValueError("support must be nonempty and have distinct values")
        if self.lower >= 0 or any(v < self.lower for v in self.support):
            raise ValueError("negative lower bound must bound the entire support")
        if not self.fractions or any(not 0 < v < 1 for v in self.fractions):
            raise ValueError("predeclared fractions must lie strictly between 0 and 1")
        self.precision = operator.index(precision)
        self.down, self.up = _contexts(self.precision)
        self.log_lower, self.log_upper = [], []
        for fraction in self.fractions:
            bounds = [_log_fraction_interval(1+fraction*v/(-self.lower),
                                            self.down, self.up) for v in self.support]
            self.log_lower.append(tuple(v[0] for v in bounds))
            self.log_upper.append(tuple(v[1] for v in bounds))
        self.denominator = Decimal(len(self.fractions))
        self.precompute_seconds = time.perf_counter()-started

    def _counts(self, counts):
        counts = tuple(operator.index(v) for v in counts)
        if len(counts) != len(self.support) or any(v < 0 for v in counts):
            raise ValueError("one nonnegative integer count per support value required")
        return counts

    def _lower_logs(self, counts):
        # Multiplication and accumulation both round down, including negative
        # terms. Log intervals were computed once, independently of these counts.
        active = [(k, Decimal(n)) for k, n in enumerate(counts) if n]
        tolerance = self.down.multiply(Decimal("1e-12"), Decimal(1+sum(counts)))
        lower = []
        for row in self.log_lower:
            value = Decimal(0)
            for k, count in active:
                value = self.down.add(value, self.down.multiply(count, row[k]))
            lower.append(self.down.subtract(value, tolerance))
        return lower

    def _lower_evidence(self, logs):
        evidence = Decimal(0)
        for value in logs:
            # Decimal exp, like ln, is correctly rounded to nearest. Stepping
            # one representable number outward supplies the directed bound.
            term = max(Decimal(0), self.down.next_minus(self.down.exp(value)))
            evidence = self.down.add(evidence, term)
        return self.down.divide(evidence, self.denominator)

    def pvalue(self, counts):
        """Conservative scalar p-value; the inexpensive primary benchmark API."""
        counts = self._counts(counts)
        if not sum(counts):
            return 1.0
        logs = self._lower_logs(counts)
        # Replacing any weak evidence by zero is conservative and saves 64 exp
        # evaluations on common null draws. This is only an arithmetic shortcut.
        if max(logs) <= 0:
            return 1.0
        evidence = self._lower_evidence(logs)
        if evidence <= 1:
            return 1.0
        return min(1.0, _to_float_up(self.up.divide(Decimal(1), evidence)))

    def evaluate(self, counts):
        """Scalar p-value plus outward log-evidence bounds and cost metadata."""
        started = time.perf_counter()
        counts = self._counts(counts)
        lower = self._lower_evidence(self._lower_logs(counts))
        upper = Decimal(0)
        for row in self.log_upper:
            value = Decimal(0)
            for count, log_factor in zip(counts, row):
                value = self.up.add(value, self.up.multiply(Decimal(count), log_factor))
            upper = self.up.add(upper, self.up.next_plus(self.up.exp(value)))
        upper = self.up.divide(upper, self.denominator)
        log_lower = self.down.next_minus(self.down.ln(lower))
        log_upper = self.up.next_plus(self.up.ln(upper))
        p = 1.0 if lower <= 1 else min(1.0, _to_float_up(self.up.divide(Decimal(1), lower)))
        return {"p": p, "p_value": p,
                "log_e_lower": _to_float_down(log_lower),
                "log_e_upper": _to_float_up(log_upper),
                "log_e_gap_upper": _to_float_up(self.up.subtract(log_upper, log_lower)),
                "scores": sum(counts), "support_size": len(self.support),
                "fraction_count": len(self.fractions), "decimal_precision": self.precision,
                "precompute_seconds": self.precompute_seconds,
                "evaluate_seconds": time.perf_counter()-started,
                "certificate": "outward Decimal finite-support log products and mixture"}
