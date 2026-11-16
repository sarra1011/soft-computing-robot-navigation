"""Observation layer: the simulator keeps the TRUE state, controllers only see an
Observation built here from the (possibly noisy) estimated pose and the known map.

Obstacle perception (session 4) is ideal and geometric, not a lidar model:
for each obstacle, estimated guard g_i = |c_i - p_hat| - r_i - R and bearing
beta_i = wrap(atan2(cy - y, cx - x) - theta). The frontal obstacle is the
closest one with |beta_i| <= pi/2; without one, d_front = 3 m and beta = 0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from model import goal_geometry, wrap

SIGMA_XY = 0.02      # [m]
SIGMA_THETA = 0.02   # [rad]
D_FRONT_MAX = 3.0    # [m]


def make_noise(seed: int, n_steps: int, sigma_xy: float = SIGMA_XY,
               sigma_theta: float = SIGMA_THETA) -> np.ndarray:
    """Pre-generate n_steps triplets (eps_x, eps_y, eps_theta) for one seed, so
    every controller receives exactly the same time sequence."""
    rng = np.random.default_rng(seed)
    out = np.empty((n_steps, 3))
    out[:, 0] = rng.normal(0.0, sigma_xy, n_steps)
    out[:, 1] = rng.normal(0.0, sigma_xy, n_steps)
    out[:, 2] = rng.normal(0.0, sigma_theta, n_steps)
    return out


@dataclass(frozen=True)
class Observation:
    k: int
    t: float
    x: float              # estimated pose
    y: float
    theta: float
    rho: float            # all derived quantities use the ESTIMATED pose
    alpha: float
    goal: tuple
    d_front: float = D_FRONT_MAX   # guard of the frontal obstacle, clipped to 3 m
    beta: float = 0.0              # bearing of the frontal obstacle
    guard_est: float = math.inf    # estimated global guard (all obstacles)
    obstacles_rel: tuple = ()      # ((guard_i, beta_i), ...) for every obstacle


def perceive_obstacles(x, y, th, obstacles, robot_radius):
    d_front, beta, guard, rel = math.inf, 0.0, math.inf, []
    for cx, cy, ro in obstacles:
        g = math.hypot(cx - x, cy - y) - ro - robot_radius
        guard = min(guard, g)
        b = wrap(math.atan2(cy - y, cx - x) - th)
        rel.append((g, b))
        if abs(b) <= math.pi / 2 and g < d_front:
            d_front, beta = g, b
    if d_front == math.inf:
        d_front, beta = D_FRONT_MAX, 0.0
    return min(d_front, D_FRONT_MAX), beta, guard, tuple(rel)


class PoseSensor:
    """Noise-free if noise is None, otherwise adds the pre-generated triplets."""

    def __init__(self, goal, obstacles=(), robot_radius: float = 0.20,
                 noise: np.ndarray | None = None):
        self.goal = (float(goal[0]), float(goal[1]))
        self.obstacles = [tuple(map(float, o)) for o in np.asarray(obstacles, float).reshape(-1, 3)]
        self.R = robot_radius
        self.noise = noise

    def observe(self, state, k: int, dt: float) -> Observation:
        x, y, th = state
        if self.noise is not None:
            ex, ey, eth = self.noise[k]
            x, y, th = x + ex, y + ey, wrap(th + eth)
        rho, alpha = goal_geometry(x, y, th, self.goal)
        d_front, beta, guard, rel = perceive_obstacles(x, y, th, self.obstacles, self.R)
        return Observation(k, k * dt, x, y, th, rho, alpha, self.goal,
                           d_front, beta, guard, rel)
