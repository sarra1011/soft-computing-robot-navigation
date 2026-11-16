"""Fuzzy inference engine written from scratch (session 2).

Chain: fuzzification -> rule activation (AND = min) -> implication -> aggregation
-> crisp output.

* Mamdani: lambda_j = min of premise memberships; implication min(lambda_j, mu_Bj(z));
  aggregation max_j; discrete centre of gravity on a uniform grid.
* Sugeno order 0: each rule gives a constant c_j; z* = sum(lambda_j c_j) / sum(lambda_j).

If no rule fires (sum = 0) the engine returns None and counts a diagnostic; the
controller then applies (v, omega) = (0, 0).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- #
# Membership functions
# --------------------------------------------------------------------------- #
class MembershipFunction:
    """Trapezoid trap(a, b, c, d) with a <= b <= c <= d.

    A triangle is trap(a, b, b, c). Shoulders have a == b or c == d: the
    plateau is handled explicitly, so there is never a division by zero.
    """

    def __init__(self, name: str, a, b, c, d, kind: str = "trap"):
        a, b, c, d = (float(t) for t in (a, b, c, d))
        if not (a <= b <= c <= d):
            raise ValueError(f"{name}: need a <= b <= c <= d, got {(a, b, c, d)}")
        self.name, self.kind = name, kind
        self.a, self.b, self.c, self.d = a, b, c, d

    def __call__(self, z: float) -> float:
        a, b, c, d = self.a, self.b, self.c, self.d
        if z < a or z > d:
            return 0.0
        if b <= z <= c:                 # plateau (includes shoulders and the peak)
            return 1.0
        if z < b:                       # here a <= z < b, hence b > a
            return (z - a) / (b - a)
        return (d - z) / (d - c)        # here c < z <= d, hence d > c

    def eval(self, z) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        a, b, c, d = self.a, self.b, self.c, self.d
        mu = np.zeros_like(z)
        mu[(z >= b) & (z <= c)] = 1.0
        if b > a:
            m = (z >= a) & (z < b)
            mu[m] = (z[m] - a) / (b - a)
        if d > c:
            m = (z > c) & (z <= d)
            mu[m] = (d - z[m]) / (d - c)
        return mu

    @property
    def params(self):
        if self.kind == "tri":
            return (self.a, self.b, self.d)
        return (self.a, self.b, self.c, self.d)

    def __repr__(self):
        return f"{self.name}={self.kind}{tuple(round(p, 4) for p in self.params)}"


def tri(name, a, b, c):
    return MembershipFunction(name, a, b, b, c, "tri")


def trap(name, a, b, c, d):
    return MembershipFunction(name, a, b, c, d, "trap")


# --------------------------------------------------------------------------- #
# Linguistic variables
# --------------------------------------------------------------------------- #
class LinguisticVariable:
    def __init__(self, name: str, lo: float, hi: float, sets, unit: str = ""):
        self.name, self.lo, self.hi, self.unit = name, float(lo), float(hi), unit
        self.sets = list(sets)
        self.names = [s.name for s in self.sets]
        self.index = {n: i for i, n in enumerate(self.names)}
        self.n_out_of_domain = 0          # input saturation counter (logged)

    def saturate(self, x: float):
        """Clip to the domain; returns (value, was_out_of_domain)."""
        if x < self.lo:
            self.n_out_of_domain += 1
            return self.lo, True
        if x > self.hi:
            self.n_out_of_domain += 1
            return self.hi, True
        return x, False

    def fuzzify(self, x: float) -> np.ndarray:
        x, _ = self.saturate(x)
        return np.array([s(x) for s in self.sets])

    def coverage(self, n: int = 4001) -> float:
        """min over the domain of max_i mu_i (> 0 means full coverage)."""
        z = np.linspace(self.lo, self.hi, n)
        return float(np.vstack([s.eval(z) for s in self.sets]).max(axis=0).min())

    def __getitem__(self, name):
        return self.sets[self.index[name]]

    def __repr__(self):
        return f"{self.name}[{self.lo:.3g}, {self.hi:.3g}] {self.sets}"


# --------------------------------------------------------------------------- #
# Rules
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Rule:
    antecedent: tuple          # one set name per input variable (AND = min)
    consequent: tuple          # ((output_name, set_name), ...)

    def cons(self, output: str) -> str:
        return dict(self.consequent)[output]

    def text(self, inputs, outputs) -> str:
        prem = " AND ".join(f"{v.name} is {s}" for v, s in zip(inputs, self.antecedent))
        con = ", ".join(f"{o} is {s}" for o, s in self.consequent)
        return f"IF {prem} THEN {con}"


def rules_from_table(table: dict, output_names=("v", "omega")):
    """table[(row_set, col_set)] = (v_set, omega_set) -> list of Rule."""
    return [Rule(tuple(k), tuple(zip(output_names, val))) for k, val in table.items()]


# --------------------------------------------------------------------------- #
# Engines
# --------------------------------------------------------------------------- #
class FuzzyEngine:
    kind = "base"

    def __init__(self, inputs, outputs, rules):
        self.inputs = list(inputs)
        self.outputs = dict(outputs)                 # name -> LinguisticVariable
        self.rules = list(rules)
        self.ante = np.array([[inp.index[s] for inp, s in zip(self.inputs, r.antecedent)]
                              for r in self.rules])
        self.cons = {o: np.array([var.index[r.cons(o)] for r in self.rules])
                     for o, var in self.outputs.items()}
        self.n_inactive = 0                          # "no rule fired" diagnostics

    def activations(self, x):
        mus = [inp.fuzzify(xi) for inp, xi in zip(self.inputs, x)]
        lam = mus[0][self.ante[:, 0]]
        for j in range(1, len(mus)):
            lam = np.minimum(lam, mus[j][self.ante[:, j]])
        return lam, mus

    def infer(self, x):                              # pragma: no cover
        raise NotImplementedError


class MamdaniEngine(FuzzyEngine):
    kind = "mamdani"

    def __init__(self, inputs, outputs, rules, n_points: int = 301):
        super().__init__(inputs, outputs, rules)
        self.n_points = n_points
        self.grid, self.rule_mf = {}, {}
        for o, var in self.outputs.items():
            z = np.linspace(var.lo, var.hi, n_points)
            table = np.vstack([s.eval(z) for s in var.sets])
            self.grid[o] = z
            self.rule_mf[o] = table[self.cons[o]]    # (n_rules, n_points)

    def aggregate(self, lam, o):
        active = lam > 0
        if not active.any():
            return np.zeros(self.n_points)
        return np.minimum(lam[active, None], self.rule_mf[o][active]).max(axis=0)

    def infer(self, x):
        lam, _ = self.activations(x)
        out = {}
        for o in self.outputs:
            agg = self.aggregate(lam, o)
            s = agg.sum()
            if s <= 0.0:
                self.n_inactive += 1
                return None
            out[o] = float(agg @ self.grid[o] / s)
        return out

    def explain(self, x):
        lam, mus = self.activations(x)
        aggs = {o: self.aggregate(lam, o) for o in self.outputs}
        return {"mus": mus, "lambda": lam, "agg": aggs, "grid": self.grid,
                "out": self.infer(x)}


class SugenoEngine(FuzzyEngine):
    kind = "sugeno"

    def __init__(self, inputs, outputs, rules, constants: dict):
        super().__init__(inputs, outputs, rules)
        self.constants = constants
        self.c = {o: np.array([constants[o][r.cons(o)] for r in self.rules], dtype=float)
                  for o in self.outputs}

    def infer(self, x):
        lam, _ = self.activations(x)
        s = lam.sum()
        if s <= 0.0:
            self.n_inactive += 1
            return None
        return {o: float(lam @ self.c[o] / s) for o in self.outputs}

    def explain(self, x):
        lam, mus = self.activations(x)
        return {"mus": mus, "lambda": lam, "out": self.infer(x)}


def centroid(mf: MembershipFunction, lo: float, hi: float, n_points: int = 301) -> float:
    z = np.linspace(lo, hi, n_points)
    mu = mf.eval(z)
    return float(mu @ z / mu.sum())
