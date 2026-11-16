"""Partitions, rule tables and engine factories (sessions 3 and 4).

All values come from the handout. Rule tables are versioned: any change made
after looking at results must get a new version string.
"""
from __future__ import annotations

import math

from fuzzy import (LinguisticVariable, MamdaniEngine, SugenoEngine, centroid,
                   rules_from_table, trap, tri)

PI = math.pi
RHO_MAX = 14.2


# --------------------------------------------------------------------------- #
# Goal seeking: inputs rho, alpha
# --------------------------------------------------------------------------- #
def goal_inputs():
    rho = LinguisticVariable("rho", 0.0, RHO_MAX, [
        trap("Proche", 0, 0, 0.3, 1.2),
        tri("Moyenne", 0.5, 2.5, 5),
        trap("Loin", 3, 6, RHO_MAX, RHO_MAX),
    ], "m")
    alpha = LinguisticVariable("alpha", -PI, PI, [
        trap("GN", -PI, -PI, -1.2, -0.4),
        tri("N", -1.2, -0.5, 0),
        tri("Z", -0.25, 0, 0.25),
        tri("P", 0, 0.5, 1.2),                 # symmetric of N
        trap("GP", 0.4, 1.2, PI, PI),          # symmetric of GN
    ], "rad")
    return rho, alpha


def motion_outputs():
    v = LinguisticVariable("v", 0.0, 0.60, [
        trap("S", 0, 0, 0.02, 0.06),
        tri("L", 0.02, 0.12, 0.25),
        tri("M", 0.15, 0.30, 0.45),
        trap("R", 0.35, 0.50, 0.60, 0.60),
    ], "m/s")
    omega = LinguisticVariable("omega", -1.50, 1.50, [
        trap("TD", -1.50, -1.50, -1.20, -0.70),
        tri("D", -1.20, -0.60, 0),
        tri("Z", -0.15, 0, 0.15),
        tri("G", 0, 0.60, 1.20),               # symmetric of D
        trap("TG", 0.70, 1.20, 1.50, 1.50),    # symmetric of TD
    ], "rad/s")
    return v, omega


SUGENO_CONSTANTS = {
    "v": {"S": 0.0, "L": 0.12, "M": 0.30, "R": 0.55},
    "omega": {"TD": -1.30, "D": -0.60, "Z": 0.0, "G": 0.60, "TG": 1.30},
}

_COLS = ("GN", "N", "Z", "P", "GP")
GOAL_TABLE = {}
for _row, _zv in (("Proche", "L"), ("Moyenne", "M"), ("Loin", "R")):
    for _col, (_v, _w) in zip(_COLS, (("S", "TD"), ("L", "D"), (_zv, "Z"), ("L", "G"), ("S", "TG"))):
        GOAL_TABLE[(_row, _col)] = (_v, _w)


# --------------------------------------------------------------------------- #
# Obstacle avoidance: inputs d_front, beta
# --------------------------------------------------------------------------- #
def avoid_inputs():
    d = LinguisticVariable("d_front", 0.0, 3.0, [
        trap("Danger", 0, 0, 0.25, 0.65),
        tri("Vigilance", 0.35, 0.9, 1.5),
        trap("Libre", 1.1, 1.8, 3, 3),
    ], "m")
    beta = LinguisticVariable("beta", -PI / 2, PI / 2, [
        trap("Droite", -PI / 2, -PI / 2, -0.3, 0),
        tri("Face", -0.3, 0, 0.3),
        trap("Gauche", 0, 0.3, PI / 2, PI / 2),   # symmetric of Droite
    ], "rad")
    return d, beta


AVOID_TABLES = {
    # v1 = handout table, unchanged
    "v1": {
        ("Danger", "Droite"): ("S", "TG"), ("Danger", "Face"): ("S", "TG"), ("Danger", "Gauche"): ("S", "TD"),
        ("Vigilance", "Droite"): ("L", "G"), ("Vigilance", "Face"): ("L", "G"), ("Vigilance", "Gauche"): ("L", "D"),
        ("Libre", "Droite"): ("M", "Z"), ("Libre", "Face"): ("M", "Z"), ("Libre", "Gauche"): ("M", "Z"),
    },
}


# --------------------------------------------------------------------------- #
# Factories
# --------------------------------------------------------------------------- #
def mamdani_centroid_constants(n_points: int = 301):
    v, omega = motion_outputs()
    return {var.name: {s.name: centroid(s, var.lo, var.hi, n_points) for s in var.sets}
            for var in (v, omega)}


def build_engine(kind: str, inputs, table, n_points: int = 301, constants=None):
    v, omega = motion_outputs()
    rules = rules_from_table(table)
    outputs = {"v": v, "omega": omega}
    if kind == "mamdani":
        return MamdaniEngine(inputs, outputs, rules, n_points)
    if kind == "sugeno":
        return SugenoEngine(inputs, outputs, rules, constants or SUGENO_CONSTANTS)
    raise ValueError(kind)


def build_goal_engine(kind: str, n_points: int = 301, constants=None):
    return build_engine(kind, goal_inputs(), GOAL_TABLE, n_points, constants)


def build_avoid_engine(kind: str, version: str = "v1", n_points: int = 301):
    return build_engine(kind, avoid_inputs(), AVOID_TABLES[version], n_points)
