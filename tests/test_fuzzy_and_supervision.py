"""Sessions 2-4 checks:  python -m unittest discover -s tests -v"""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controllers import FuzzyAvoidController, make_controller   # noqa: E402
from fuzzy import trap, tri                                      # noqa: E402
from fuzzy_design import (avoid_inputs, build_avoid_engine,      # noqa: E402
                          build_goal_engine, goal_inputs, motion_outputs)
from model import Params                                         # noqa: E402
from scenarios import SCENARIOS                                  # noqa: E402
from sensors import Observation                                  # noqa: E402
from simulator import simulate                                   # noqa: E402

P = Params()


class TestMembership(unittest.TestCase):
    def test_handout_manual_check(self):
        bas, haut = trap("Bas", 0, 0, 0.2, 0.6), trap("Haut", 0.2, 0.6, 1, 1)
        self.assertAlmostEqual(bas(0.25), 0.875)
        self.assertAlmostEqual(haut(0.25), 0.125)
        self.assertAlmostEqual((0.875 * 0.2 + 0.125 * 0.8) / 1.0, 0.275)

    def test_shapes_bounds_vertices(self):
        t = tri("t", 0, 0.5, 1)
        self.assertEqual(t(0.5), 1.0)
        self.assertEqual(t(0), 0.0)
        self.assertEqual(t(1), 0.0)
        self.assertEqual(t(-3), 0.0)
        sh = trap("s", 0, 0, 0.3, 0.7)          # shoulder: no division by zero
        self.assertEqual(sh(0.0), 1.0)
        z = np.linspace(-1, 2, 3001)
        for f in (t, sh, trap("r", 0.3, 0.7, 1, 1)):
            mu = f.eval(z)
            self.assertTrue((mu >= 0).all() and (mu <= 1).all())
            self.assertTrue(np.allclose(mu, [f(x) for x in z]))

    def test_coverage(self):
        for v in (*goal_inputs(), *motion_outputs(), *avoid_inputs()):
            self.assertGreater(v.coverage(), 0.0, v.name)


class TestEngines(unittest.TestCase):
    def test_symmetry(self):
        for k in ("mamdani", "sugeno"):
            e = build_goal_engine(k)
            for r, a in ((3, 0.7), (0.6, 2.0), (9, 0.1)):
                o1, o2 = e.infer((r, a)), e.infer((r, -a))
                self.assertAlmostEqual(o1["v"], o2["v"], places=9)
                self.assertAlmostEqual(o1["omega"], -o2["omega"], places=9)

    def test_ranges(self):
        for k in ("mamdani", "sugeno"):
            e = build_goal_engine(k)
            for r in np.linspace(0, 14.2, 15):
                for a in np.linspace(-math.pi, math.pi, 25):
                    o = e.infer((r, a))
                    self.assertTrue(0 <= o["v"] <= 0.6 and -1.5 <= o["omega"] <= 1.5)

    def test_left_turn_when_goal_left(self):
        for k in ("mamdani", "sugeno"):
            self.assertGreater(build_goal_engine(k).infer((5, 0.8))["omega"], 0)

    def test_inactive_rule_policy(self):
        e = build_goal_engine("sugeno")
        e.c = {o: c for o, c in e.c.items()}
        lam_zero = e.infer
        e.inputs[0].sets = [trap("x", 100, 101, 102, 103)] * 3   # nothing can fire
        self.assertIsNone(lam_zero((5, 0)))
        self.assertEqual(e.n_inactive, 1)

    def test_avoid_persistent_side(self):
        a = FuzzyAvoidController(build_avoid_engine("sugeno"), "a")
        obs = lambda b: Observation(0, 0, 0, 0, 0, 5, 0, (9, 9), 0.5, b, 0.5)
        a.reset()
        _, w1 = a.compute(obs(-0.1))            # obstacle slightly right -> go left
        _, w2 = a.compute(obs(+0.1))            # noise flips beta: keep going left
        self.assertGreater(w1, 0)
        self.assertGreater(w2, 0)


class TestSupervision(unittest.TestCase):
    def test_all_methods_reach_B_without_collision(self):
        for sysv in ("sys-v1", "sys-v2"):
            for m in ("classic", "mamdani", "sugeno"):
                r = simulate(SCENARIOS["B"], make_controller(m, P, sysv), P)
                self.assertEqual(r.status, "arrived", f"{m} {sysv}")
                self.assertGreater(r.min_guard, P.margin)

    def test_v2_removes_chattering(self):
        def switches(sysv):
            r = simulate(SCENARIOS["B"], make_controller("sugeno", P, sysv), P)
            m = [q["mode"] for q in r.rows if q["mode"] in ("goal", "avoid")]
            return sum(a != b for a, b in zip(m, m[1:]))
        self.assertGreater(switches("sys-v1"), 100)
        self.assertLess(switches("sys-v2"), 20)


if __name__ == "__main__":
    unittest.main()
