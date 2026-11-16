"""Controllers. ALL of them share the same interface:

    controller.reset()
    v, omega = controller.compute(observation)
    controller.mode            # logged by the simulator at every step

Building blocks
  * goal controllers  : ClassicController, FuzzyGoalController, MLPGoalController
  * avoid controller  : FuzzyAvoidController (Mamdani or Sugeno engine)
  * SupervisedController = goal + avoid + common supervisor (hysteresis,
    preventive stop). make_controller() builds the three compared systems.
"""
from __future__ import annotations

import math

import numpy as np

from fuzzy_design import RHO_MAX, build_avoid_engine, build_goal_engine
from model import Params, clip


class Controller:
    name = "base"
    version = "v0"

    def __init__(self):
        self.mode = "goal"
        self.flags = ""                 # diagnostics for the current step

    def reset(self) -> None:
        self.mode = "goal"
        self.flags = ""

    def compute(self, obs):                      # pragma: no cover
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Goal controllers
# --------------------------------------------------------------------------- #
class ClassicController(Controller):
    """v = min(v_max, k_rho rho) max(0, cos alpha) ; omega = clip(k_alpha alpha)."""

    name, version = "classic", "v1.0"

    def __init__(self, params: Params, k_rho: float = 0.8, k_alpha: float = 2.0,
                 use_cos: bool = True, tag: str = "", sv: float = 1.0, sw: float = 1.0):
        super().__init__()
        self.p, self.k_rho, self.k_alpha, self.use_cos = params, k_rho, k_alpha, use_cos
        self.sv, self.sw = sv, sw
        if tag:
            self.name = f"classic{tag}"

    def compute(self, obs):
        p = self.p
        if obs.rho <= p.goal_tol:
            return 0.0, 0.0
        gate = max(0.0, math.cos(obs.alpha)) if self.use_cos else 1.0
        v = min(p.v_max, self.k_rho * obs.rho) * gate
        w = clip(self.k_alpha * obs.alpha, -p.w_max, p.w_max)
        return self.sv * v, self.sw * w


class FuzzyGoalController(Controller):
    """Fuzzy goal seeking, inputs (rho, alpha) -> (v, omega).

    sv, sw: optional output factors (session 6, option B). Physical saturation is
    applied afterwards by the simulator.
    """

    def __init__(self, params: Params, engine, name: str, version: str = "v1.0",
                 sv: float = 1.0, sw: float = 1.0):
        super().__init__()
        self.p, self.engine, self.name, self.version = params, engine, name, version
        self.sv, self.sw = sv, sw

    def compute(self, obs):
        if obs.rho <= self.p.goal_tol:
            return 0.0, 0.0
        out = self.engine.infer((obs.rho, obs.alpha))
        if out is None:                       # no rule fired: diagnostic + stop
            self.flags = "no_rule"
            return 0.0, 0.0
        return self.sv * out["v"], self.sw * out["omega"]


class MLPGoalController(Controller):
    """Imitation network (session 6, option A). Pure-numpy forward pass of a
    trained sklearn MLPRegressor (ReLU hidden layers, identity output)."""

    def __init__(self, params: Params, mlp, scaler, name="mlp", version="v1.0"):
        super().__init__()
        self.p, self.name, self.version = params, name, version
        self.W = [np.asarray(w) for w in mlp.coefs_]
        self.b = [np.asarray(b) for b in mlp.intercepts_]
        self.mu, self.sd = scaler.mean_, scaler.scale_

    @staticmethod
    def features(rho, alpha):
        return np.array([rho / RHO_MAX, math.sin(alpha), math.cos(alpha)])

    def forward(self, f):
        h = (f - self.mu) / self.sd
        for W, b in zip(self.W[:-1], self.b[:-1]):
            h = np.maximum(h @ W + b, 0.0)
        return h @ self.W[-1] + self.b[-1]

    def compute(self, obs):
        if obs.rho <= self.p.goal_tol:
            return 0.0, 0.0
        y = self.forward(self.features(min(obs.rho, RHO_MAX), obs.alpha))
        return float(y[0]) * self.p.v_max, float(y[1]) * self.p.w_max


# --------------------------------------------------------------------------- #
# Obstacle avoidance
# --------------------------------------------------------------------------- #
class FuzzyAvoidController(Controller):
    """Fuzzy avoidance, inputs (d_front, beta) -> (v, omega).

    Persistence: the side is chosen ONCE when avoidance starts (obstacle on the
    right or dead ahead -> pass on the left, the handout convention; obstacle on
    the left -> pass on the right) and kept until the supervisor leaves the
    avoidance mode. Implementation by mirror symmetry: for side = -1 the rule
    base is evaluated on -beta and omega is negated, so the 'Face' rule always
    turns towards the committed side. The table being symmetric otherwise, the
    Droite / Gauche rules are unchanged.
    """

    def __init__(self, engine, name: str, persistent: bool = True, tie_eps: float = 0.0):
        super().__init__()
        self.engine, self.name, self.persistent = engine, name, persistent
        self.tie_eps = tie_eps
        self.side = None

    def reset(self):
        super().reset()
        self.side = None

    def compute(self, obs):
        if not self.persistent:
            out = self.engine.infer((obs.d_front, obs.beta))
            return (0.0, 0.0) if out is None else (out["v"], out["omega"])
        if self.side is None:
            # obstacle on the right or ahead -> left (handout convention).
            # tie_eps > 0 stops 1e-17 rounding noise in beta from choosing the side.
            self.side = 1 if obs.beta <= self.tie_eps else -1
        out = self.engine.infer((obs.d_front, self.side * obs.beta))
        if out is None:
            self.flags = "no_rule"
            return 0.0, 0.0
        return out["v"], self.side * out["omega"]


# --------------------------------------------------------------------------- #
# Supervisor common to the three methods
# --------------------------------------------------------------------------- #
class SupervisedController(Controller):
    """goal mode <-> avoid mode with hysteresis, plus a preventive stop.

    * goal -> avoid  if d_front < d_in   (1.0 m)
    * avoid -> goal  if d_front > d_out  (1.4 m)
    * preventive stop for the current step if the estimated global guard
      <= stop_guard (0.15 m): (v, omega) = (0, 0), mode 'stop' in the log.
    * arrival (estimated rho <= tol): exact stop, mode 'arrived'.
    """

    def __init__(self, name, version, params: Params, goal: Controller,
                 avoid: Controller, d_in=1.0, d_out=None, stop_guard=0.15,
                 track_obstacle: bool = False):
        super().__init__()
        self.name, self.version, self.p = name, version, params
        self.goal, self.avoid = goal, avoid
        self.d_in = d_in
        self.d_out = d_in + 0.4 if d_out is None else d_out
        self.stop_guard = stop_guard
        self.track = track_obstacle
        self.nav = "goal"
        self.locked = None

    def reset(self):
        super().reset()
        self.goal.reset()
        self.avoid.reset()
        self.nav = "goal"
        self.locked = None

    @staticmethod
    def _frontal_index(obs):
        best, idx = math.inf, None
        for i, (g, b) in enumerate(obs.obstacles_rel):
            if abs(b) <= math.pi / 2 and g < best:
                best, idx = g, i
        return idx

    def _avoid_inputs(self, obs):
        """v1: handout perception. v2: keep tracking the obstacle that triggered
        the avoidance even when it leaves the frontal half-plane (|beta| > pi/2),
        so that d_front stays continuous and the hysteresis really works."""
        if not self.track or self.locked is None:
            return obs.d_front, obs.beta
        i = self._frontal_index(obs)
        if i is not None and obs.obstacles_rel[i][0] < obs.obstacles_rel[self.locked][0]:
            self.locked = i                    # a closer frontal obstacle takes over
        g, b = obs.obstacles_rel[self.locked]
        return min(g, 3.0), clip(b, -math.pi / 2, math.pi / 2)

    def compute(self, obs):
        self.flags = ""
        if obs.rho <= self.p.goal_tol:
            self.mode = "arrived"
            return 0.0, 0.0
        # mode switching with hysteresis
        if self.nav == "goal" and obs.d_front < self.d_in:
            self.nav = "avoid"
            self.avoid.reset()
            self.locked = self._frontal_index(obs)
        d_front, beta = self._avoid_inputs(obs)
        if self.nav == "avoid" and d_front > self.d_out and obs.d_front > self.d_out:
            self.nav, self.locked = "goal", None
        if obs.guard_est <= self.stop_guard:          # preventive stop (this step)
            self.mode = "stop"
            return 0.0, 0.0
        self.mode = self.nav
        if self.nav == "goal":
            ctrl, o = self.goal, obs
        else:
            ctrl, o = self.avoid, (obs if not self.track else
                                   _AvoidObs(d_front, beta))
        ctrl.flags = ""
        v, w = ctrl.compute(o)
        self.flags = ctrl.flags
        return v, w


class _AvoidObs:
    """Minimal observation handed to the avoidance controller in v2."""
    __slots__ = ("d_front", "beta")

    def __init__(self, d_front, beta):
        self.d_front, self.beta = d_front, beta


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
METHODS = ("classic", "mamdani", "sugeno")

# System versions (frozen BEFORE the test campaign on D, E, F):
#   sys-v1 : handout as written (frontal perception, tie at beta = 0 -> left)
#   sys-v2 : tuned on B/C only -> the avoided obstacle stays tracked during the
#            avoidance (continuous d_front, working hysteresis) + |beta| <= 0.05
#            counts as 'Face' when choosing the side (left convention)
SYSTEM_VERSIONS = {
    "sys-v1": dict(track_obstacle=False, tie_eps=0.0),
    "sys-v2": dict(track_obstacle=True, tie_eps=0.05),
}


def make_controller(method: str, params: Params, system: str = "sys-v1",
                    n_points: int = 301, sv=1.0, sw=1.0, d_in=1.0, goal=None,
                    name=None, avoid_version: str = "v1"):
    """Build one of the compared systems.

    classic : classic goal law + the SAME Sugeno avoidance as the Sugeno system
    mamdani : Mamdani goal + Mamdani avoidance
    sugeno  : Sugeno goal + Sugeno avoidance
    `goal` can replace the goal controller (MLP extension).
    """
    cfg = SYSTEM_VERSIONS[system]
    if method == "classic":
        g = ClassicController(params, sv=sv, sw=sw)
        a = FuzzyAvoidController(build_avoid_engine("sugeno", avoid_version), "avoid_sugeno",
                                 tie_eps=cfg["tie_eps"])
    elif method in ("mamdani", "sugeno"):
        g = FuzzyGoalController(params, build_goal_engine(method, n_points), f"goal_{method}",
                                sv=sv, sw=sw)
        a = FuzzyAvoidController(build_avoid_engine(method, avoid_version, n_points),
                                 f"avoid_{method}", tie_eps=cfg["tie_eps"])
    else:
        raise ValueError(method)
    if goal is not None:
        g = goal
    return SupervisedController(name or method, system, params, g, a, d_in=d_in,
                                track_obstacle=cfg["track_obstacle"])
