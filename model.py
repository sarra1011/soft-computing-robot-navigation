"""Kinematic model of the differential-drive robot and geometry helpers.

Conventions (see docs/note_conventions.md):
  * state q = (x, y, theta), metres / radians, theta counter-clockwise positive
  * command u = (v, omega): forward speed [m/s], angular speed [rad/s]
  * alpha = wrap(bearing_to_goal - theta); alpha > 0  <=>  goal is on the LEFT
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

import numpy as np


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Params:
    """Simulation parameters (section 2 of the handout)."""

    dt: float = 0.05            # simulation step [s]
    t_max: float = 120.0        # maximum duration [s]
    wheel_radius: float = 0.05  # r [m]
    track: float = 0.30         # b [m] (distance between wheels)
    v_max: float = 0.60         # [m/s]
    w_max: float = 1.50         # [rad/s]
    robot_radius: float = 0.20  # R [m]
    goal_tol: float = 0.20      # arrival when rho <= goal_tol [m]
    arena: float = 10.0         # square side [m]
    margin: float = 0.15        # desired margin between surfaces [m]
    version: str = "params-v1"

    @property
    def n_steps(self) -> int:
        return int(round(self.t_max / self.dt))

    def to_json(self, path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def from_json(path) -> "Params":
        with open(path, encoding="utf-8") as f:
            return Params(**json.load(f))


# --------------------------------------------------------------------------- #
# Angles, saturation, kinematics
# --------------------------------------------------------------------------- #
def wrap(a: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def saturate(v: float, w: float, p: Params):
    """Apply the physical limits 0 <= v <= v_max and |omega| <= w_max."""
    return clip(v, 0.0, p.v_max), clip(w, -p.w_max, p.w_max)


def step(state, v: float, w: float, p: Params):
    """One explicit-Euler step of the unicycle model.

    The position moves along the *current* heading theta_k, so the displacement
    during one step is an exact straight segment (used for collision checks).
    """
    x, y, th = state
    return (
        x + p.dt * v * math.cos(th),
        y + p.dt * v * math.sin(th),
        wrap(th + p.dt * w),
    )


def wheel_speeds(v: float, w: float, p: Params):
    """(v, omega) -> (Omega_right, Omega_left) in rad/s."""
    od = (v + p.track * w / 2.0) / p.wheel_radius
    og = (v - p.track * w / 2.0) / p.wheel_radius
    return od, og


def body_speeds(od: float, og: float, p: Params):
    """Inverse of wheel_speeds: (Omega_right, Omega_left) -> (v, omega)."""
    v = p.wheel_radius * (od + og) / 2.0
    w = p.wheel_radius * (od - og) / p.track
    return v, w


def goal_geometry(x: float, y: float, th: float, goal):
    """Distance rho and relative bearing alpha to the goal."""
    dx, dy = goal[0] - x, goal[1] - y
    return math.hypot(dx, dy), wrap(math.atan2(dy, dx) - th)


# --------------------------------------------------------------------------- #
# Collision geometry
# --------------------------------------------------------------------------- #
def segment_guard(p0, p1, obstacles, robot_radius: float) -> float:
    """Minimum guard d = dist(segment, centre_i) - r_i - R over all obstacles.

    obstacles: array-like of shape (n, 3) with rows (cx, cy, r_o).
    With p0 == p1 this is the guard of a single point. Returns +inf if there
    is no obstacle. Collision <=> guard <= 0.
    """
    obs = np.asarray(obstacles, dtype=float).reshape(-1, 3)
    if len(obs) == 0:
        return math.inf
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    d = p1 - p0
    l2 = float(d @ d)
    c = obs[:, :2]
    if l2 == 0.0:
        t = np.zeros(len(obs))
    else:
        t = np.clip(((c - p0) @ d) / l2, 0.0, 1.0)
    closest = p0 + t[:, None] * d
    dist = np.linalg.norm(c - closest, axis=1)
    return float(np.min(dist - obs[:, 2] - robot_radius))


def point_guard(x: float, y: float, obstacles, robot_radius: float) -> float:
    return segment_guard((x, y), (x, y), obstacles, robot_radius)


def wall_clearance(x: float, y: float, p: Params) -> float:
    """Distance between the robot disc and the nearest arena wall.

    <= 0 means the disc touches/crosses the boundary. The arena is convex, so
    checking the end point of a straight segment is enough.
    """
    return min(x, y, p.arena - x, p.arena - y) - p.robot_radius
