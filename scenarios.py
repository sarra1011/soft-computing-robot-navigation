"""Scenario definitions (section 8 of the handout) + JSON import/export.

Decimal point everywhere, explicit numeric arrays (as required for config files).
Run `python scenarios.py` to check every scenario (start/goal free, gaps).
"""
from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from model import Params, goal_geometry, point_guard, wall_clearance


@dataclass(frozen=True)
class Scenario:
    id: str
    role: str            # reglage | validation | test | demo
    start: tuple         # (x, y, theta)
    goal: tuple          # (xg, yg)
    obstacles: tuple     # ((cx, cy, ro), ...)
    description: str = ""

    def obstacles_array(self) -> np.ndarray:
        return np.asarray(self.obstacles, dtype=float).reshape(-1, 3)

    def to_dict(self, params_version: str = "params-v1") -> dict:
        return {
            "id": self.id,
            "role": self.role,
            "description": self.description,
            "params_version": params_version,
            "start": list(self.start),
            "goal": list(self.goal),
            "obstacles": [list(o) for o in self.obstacles],
        }

    def save_json(self, path, params_version: str = "params-v1") -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(params_version), f, indent=2)

    @staticmethod
    def load_json(path) -> "Scenario":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return Scenario(d["id"], d["role"], tuple(d["start"]), tuple(d["goal"]),
                        tuple(tuple(o) for o in d["obstacles"]),
                        d.get("description", ""))


# A concave "cup" made of overlapping discs, opening towards the robot.
_CUP = tuple((round(6.0 - 1.3 * (1 - math.cos(a)), 3), round(5.0 + 1.3 * math.sin(a), 3), 0.35)
             for a in np.linspace(-1.75, 1.75, 9))

# NB: "(1,1,-2,8)" in the French handout means theta = -2.8 rad.
#     "(5,3,8,0,60)" means centre (5, 3.8), radius 0.60.
SCENARIOS = {
    s.id: s
    for s in [
        Scenario("A", "reglage", (1.0, 1.0, 0.0), (9.0, 9.0), (),
                 "Empty arena, goal on the diagonal"),
        Scenario("A_back", "demo", (5.0, 5.0, math.pi), (9.0, 9.0), (),
                 "Goal initially BEHIND the robot (session 1, item 4)"),
        Scenario("B", "reglage", (1.0, 5.0, 0.0), (9.0, 5.0),
                 ((5.0, 5.0, 0.70),), "One obstacle on the straight line"),
        Scenario("C", "validation", (1.0, 1.0, 0.0), (9.0, 9.0),
                 ((4.0, 4.0, 0.60), (6.0, 6.0, 0.60)), "Two obstacles on the diagonal"),
        Scenario("D", "test", (1.0, 2.0, 0.0), (9.0, 8.0),
                 ((3.0, 3.0, 0.60), (5.0, 5.0, 0.70), (7.0, 7.0, 0.60)),
                 "Three obstacles"),
        Scenario("E", "test", (1.0, 5.0, 0.0), (9.0, 5.0),
                 ((5.0, 3.8, 0.60), (5.0, 6.2, 0.60)), "Narrow passage (1.20 m)"),
        Scenario("F", "test", (1.0, 1.0, -2.8), (9.0, 9.0),
                 ((4.0, 4.0, 0.60), (6.0, 5.0, 0.60)), "Start facing away from the goal"),
        Scenario("U", "demo", (1.0, 5.0, 0.0), (9.0, 5.0), _CUP,
                 "Concave cup (blocking analysis, not used for evaluation)"),
    ]
}


def validate(sc: Scenario, p: Params) -> dict:
    """Sanity checks; raises ValueError if the scenario is unusable."""
    obs = sc.obstacles_array()
    x, y, th = sc.start
    if wall_clearance(x, y, p) <= 0 or point_guard(x, y, obs, p.robot_radius) <= 0:
        raise ValueError(f"{sc.id}: start pose in collision")
    gx, gy = sc.goal
    if not (0 <= gx <= p.arena and 0 <= gy <= p.arena):
        raise ValueError(f"{sc.id}: goal outside the arena")
    if point_guard(gx, gy, obs, 0.0) <= 0:
        raise ValueError(f"{sc.id}: goal inside an obstacle")
    gaps = []
    for (a, b) in itertools.combinations(range(len(obs)), 2):
        gap = math.hypot(*(obs[a, :2] - obs[b, :2])) - obs[a, 2] - obs[b, 2]
        gaps.append((a, b, gap, gap - 2 * p.robot_radius))
    rho0, alpha0 = goal_geometry(x, y, th, sc.goal)
    return {"rho0": rho0, "alpha0": alpha0, "gaps": gaps}


def save_all(directory, p: Params) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for sid, sc in SCENARIOS.items():
        sc.save_json(directory / f"scenario_{sid}.json", p.version)


if __name__ == "__main__":
    p = Params()
    for sid, sc in SCENARIOS.items():
        info = validate(sc, p)
        print(f"{sid:7s} [{sc.role:10s}] start={sc.start} goal={sc.goal} "
              f"rho0={info['rho0']:.2f} m alpha0={math.degrees(info['alpha0']):+.1f} deg "
              f"n_obs={len(sc.obstacles)}")
        if sc.role == "demo":
            continue
        for a, b, gap, free in info["gaps"]:
            print(f"        gap obstacle {a}-{b}: {gap:.2f} m between surfaces, "
                  f"{free:.2f} m beyond the robot diameter "
                  f"(needs >= 2*margin = {2 * p.margin:.2f} m to keep the margin)")
