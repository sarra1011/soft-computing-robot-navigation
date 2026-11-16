"""Simulation loop (handout, session 1):

    for each step: test arrival -> observe -> compute command -> saturate
                   -> advance -> check collision on the displacement -> log

TIME CONVENTION OF THE LOG
  Row k holds the TRUE state q_k at time t_k = k * dt, the observation made at
  t_k, and the command u_k = (v_k, omega_k) applied during [t_k, t_k + dt).
  The last row is the terminal state (arrived / collision / timeout) with
  u = (0, 0). Arrival time is T = K * dt, K being the index of the first row
  where the TRUE rho <= goal_tol.

BLOCKING MONITOR (session 4)
  'blocked' is declared when the TRUE distance to the goal decreased by less
  than 0.05 m over the last 10 s. The episode continues (it may recover); the
  event and its time are logged, and it is never counted as an arrival.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from model import (Params, goal_geometry, point_guard, saturate, segment_guard,
                   step, wall_clearance)
from scenarios import Scenario
from sensors import PoseSensor

NAN = float("nan")
BLOCK_WINDOW_S = 10.0
BLOCK_PROGRESS = 0.05


@dataclass
class EpisodeResult:
    scenario_id: str
    controller: str
    version: str
    condition: str          # "nominal" | "noisy"
    seed: object            # int, or "" for nominal
    status: str             # arrived | collision | timeout
    rows: list = field(default_factory=list)
    min_guard: float = math.inf     # min over all displacement segments (obstacles)
    min_wall: float = math.inf      # min clearance to the arena walls
    blocked: bool = False
    t_blocked: float = NAN


def simulate(scenario: Scenario, controller, params: Params, noise=None,
             seed="", keep_rows: bool = True) -> EpisodeResult:
    """Run one episode. `noise` is a pre-generated (n_steps, 3) array or None."""
    obstacles = scenario.obstacles_array()
    R = params.robot_radius
    sensor = PoseSensor(scenario.goal, obstacles, R, noise)
    controller.reset()

    state = tuple(float(s) for s in scenario.start)
    if point_guard(state[0], state[1], obstacles, R) <= 0 or \
            wall_clearance(state[0], state[1], params) <= 0:
        raise ValueError(f"{scenario.id}: initial pose is in collision")

    res = EpisodeResult(scenario.id, controller.name, controller.version,
                        "noisy" if noise is not None else "nominal", seed, "timeout")
    window = int(round(BLOCK_WINDOW_S / params.dt))
    rho_hist = []

    def log(k, st, obs, v, w, status, t_comp, event=""):
        rho, alpha = goal_geometry(st[0], st[1], st[2], scenario.goal)
        res.rows.append({
            "scenario": scenario.id, "controller": controller.name,
            "version": controller.version, "condition": res.condition, "seed": seed,
            "k": k, "t": round(k * params.dt, 10),
            "x": st[0], "y": st[1], "theta": st[2],
            "x_obs": obs.x if obs else NAN, "y_obs": obs.y if obs else NAN,
            "theta_obs": obs.theta if obs else NAN,
            "d_front_obs": obs.d_front if obs else NAN,
            "beta_obs": obs.beta if obs else NAN,
            "rho": rho, "alpha": alpha, "v": v, "omega": w,
            "guard": point_guard(st[0], st[1], obstacles, R),
            "mode": controller.mode, "status": status, "event": event,
            "flags": getattr(controller, "flags", ""), "t_compute": t_comp,
        })

    for k in range(params.n_steps + 1):
        rho, _ = goal_geometry(state[0], state[1], state[2], scenario.goal)
        if rho <= params.goal_tol:                       # 1) arrival (TRUE state)
            controller.mode = "arrived"
            log(k, state, None, 0.0, 0.0, "arrived", NAN)
            res.status = "arrived"
            break
        if k == params.n_steps:                          # time is up
            log(k, state, None, 0.0, 0.0, "timeout", NAN)
            res.status = "timeout"
            break

        event = ""
        rho_hist.append(rho)                             # blocking monitor
        if not res.blocked and k >= window and rho_hist[k - window] - rho < BLOCK_PROGRESS:
            res.blocked, res.t_blocked, event = True, k * params.dt, "blocked"

        obs = sensor.observe(state, k, params.dt)        # 2) observe
        t0 = time.perf_counter()
        v, w = controller.compute(obs)                   # 3) command
        t_comp = time.perf_counter() - t0
        v, w = saturate(v, w, params)                    # 4) saturate
        log(k, state, obs, v, w, "running", t_comp, event)

        new_state = step(state, v, w, params)            # 5) advance
        seg = segment_guard(state[:2], new_state[:2], obstacles, R)
        wall = wall_clearance(new_state[0], new_state[1], params)
        res.min_guard = min(res.min_guard, seg)
        res.min_wall = min(res.min_wall, wall)
        state = new_state
        if seg <= 0.0 or wall <= 0.0:                    # 6) collision on the segment
            log(k + 1, state, None, 0.0, 0.0, "collision", NAN,
                "hit_obstacle" if seg <= 0.0 else "hit_wall")
            res.status = "collision"
            break
    if not keep_rows:
        res.rows = res.rows[-1:]
    return res
