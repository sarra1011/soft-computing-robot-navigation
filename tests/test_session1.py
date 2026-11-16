"""Session 1 checks:  python -m unittest discover -s tests -v"""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controllers import ClassicController          # noqa: E402
from model import (Params, body_speeds, goal_geometry, segment_guard, step,  # noqa: E402
                   wall_clearance, wheel_speeds, wrap)
from scenarios import SCENARIOS, Scenario, validate  # noqa: E402
from sensors import PoseSensor, make_noise           # noqa: E402
from simulator import simulate                       # noqa: E402

P = Params()


class TestWrap(unittest.TestCase):
    def test_range_and_equivalence(self):
        for a in np.linspace(-20, 20, 801):
            w = wrap(a)
            self.assertLessEqual(abs(w), math.pi + 1e-12)
            self.assertAlmostEqual(math.cos(w), math.cos(a), places=9)
            self.assertAlmostEqual(math.sin(w), math.sin(a), places=9)

    def test_near_pi(self):
        eps = 1e-6
        self.assertAlmostEqual(wrap(math.pi + eps), -math.pi + eps, places=9)
        self.assertAlmostEqual(wrap(-math.pi - eps), math.pi - eps, places=9)
        self.assertAlmostEqual(abs(wrap(3 * math.pi)), math.pi, places=9)
        self.assertAlmostEqual(wrap(2 * math.pi), 0.0, places=9)


class TestKinematics(unittest.TestCase):
    def test_immobility(self):
        q = (2.0, 3.0, 0.7)
        for _ in range(100):
            q = step(q, 0.0, 0.0, P)
        self.assertEqual(q, (2.0, 3.0, 0.7))

    def test_pure_rotation(self):                 # v = 0  -> rotation on the spot
        q = (2.0, 3.0, 0.0)
        for _ in range(20):
            q = step(q, 0.0, 1.0, P)
        self.assertAlmostEqual(q[0], 2.0)
        self.assertAlmostEqual(q[1], 3.0)
        self.assertAlmostEqual(q[2], 20 * P.dt * 1.0)

    def test_straight_line(self):                 # omega = 0 -> straight line
        th = 0.5
        q = (1.0, 1.0, th)
        for _ in range(40):
            q = step(q, 0.4, 0.0, P)
        d = 40 * P.dt * 0.4
        self.assertAlmostEqual(q[0], 1.0 + d * math.cos(th))
        self.assertAlmostEqual(q[1], 1.0 + d * math.sin(th))
        self.assertAlmostEqual(q[2], th)

    def test_alpha_positive_turns_left(self):
        rho, alpha = goal_geometry(0.0, 0.0, 0.0, (0.0, 1.0))   # goal on the left
        self.assertGreater(alpha, 0)
        ctrl = ClassicController(P)
        obs = PoseSensor((0.0, 1.0)).observe((0.0, 0.0, 0.0), 0, P.dt)
        v, w = ctrl.compute(obs)
        self.assertGreater(w, 0)                                 # omega > 0 = left
        q = step((0.0, 0.0, 0.0), v, w, P)
        self.assertGreater(q[2], 0)

    def test_wheel_speeds(self):
        od, og = wheel_speeds(0.0, 1.0, P)                       # on the spot
        self.assertAlmostEqual(od, -og)
        self.assertGreater(od, 0)
        od, og = wheel_speeds(0.6, 0.0, P)                       # straight
        self.assertAlmostEqual(od, og)
        self.assertAlmostEqual(od, 12.0)
        v, w = body_speeds(*wheel_speeds(0.3, -0.8, P), P)
        self.assertAlmostEqual(v, 0.3)
        self.assertAlmostEqual(w, -0.8)


class TestGeometry(unittest.TestCase):
    def test_segment_guard(self):
        # segment along x-axis, obstacle centre (5, 0.5), radius 0.1, R = 0.2
        g = segment_guard((0, 0), (10, 0), [(5.0, 0.5, 0.1)], 0.2)
        self.assertAlmostEqual(g, 0.5 - 0.1 - 0.2)
        # an end-point check alone would MISS this: obstacle in the middle of a long jump
        g = segment_guard((0, 0), (10, 0), [(5.0, 0.0, 0.1)], 0.2)
        self.assertLess(g, 0)
        self.assertEqual(segment_guard((0, 0), (1, 1), [], 0.2), math.inf)

    def test_wall(self):
        self.assertGreater(wall_clearance(5, 5, P), 0)
        self.assertLessEqual(wall_clearance(0.2, 5, P), 0)
        self.assertLessEqual(wall_clearance(5, 9.9, P), 0)

    def test_scenarios_valid(self):
        for sc in SCENARIOS.values():
            validate(sc, P)
        gaps = validate(SCENARIOS["E"], P)["gaps"]
        self.assertAlmostEqual(gaps[0][2], 1.20)                 # opening between obstacles
        self.assertGreater(gaps[0][3], 2 * P.margin)             # robot + margins fit


class TestSimulation(unittest.TestCase):
    def run_ctrl(self, sc, p=P, noise=None):
        return simulate(sc, ClassicController(p), p, noise)

    def test_scenario_A_arrives(self):
        r = self.run_ctrl(SCENARIOS["A"])
        self.assertEqual(r.status, "arrived")
        last = r.rows[-1]
        self.assertLessEqual(last["rho"], P.goal_tol)
        self.assertEqual((last["v"], last["omega"]), (0.0, 0.0))

    def test_target_behind_arrives(self):
        r = self.run_ctrl(SCENARIOS["A_back"])
        self.assertEqual(r.status, "arrived")
        self.assertEqual(r.rows[0]["v"], 0.0)                    # turns on the spot first
        self.assertNotEqual(r.rows[0]["omega"], 0.0)

    def test_left_right_symmetry(self):
        a = Scenario("S1", "t", (1.0, 1.0, 0.0), (9.0, 9.0), ())
        b = Scenario("S2", "t", (1.0, 9.0, 0.0), (9.0, 1.0), ())   # mirror about y = 5
        ra, rb = self.run_ctrl(a), self.run_ctrl(b)
        self.assertEqual(len(ra.rows), len(rb.rows))
        xa = np.array([[r["x"], r["y"], r["theta"]] for r in ra.rows])
        xb = np.array([[r["x"], 10.0 - r["y"], -r["theta"]] for r in rb.rows])
        self.assertTrue(np.allclose(xa, xb, atol=1e-9))

    def test_reproducible_same_seed(self):
        n = make_noise(3, P.n_steps)
        r1 = self.run_ctrl(SCENARIOS["A"], noise=n)
        r2 = self.run_ctrl(SCENARIOS["A"], noise=make_noise(3, P.n_steps))
        f = lambda r: [(x["x"], x["y"], x["theta"], x["v"], x["omega"]) for x in r.rows]
        self.assertEqual(f(r1), f(r2))
        r3 = self.run_ctrl(SCENARIOS["A"], noise=make_noise(4, P.n_steps))
        self.assertNotEqual(f(r1), f(r3))

    def test_noise_shape(self):
        n = make_noise(0, P.n_steps)
        self.assertEqual(n.shape, (2400, 3))

    def test_collision_detected(self):
        sc = Scenario("X", "t", (1.0, 5.0, 0.0), (9.0, 5.0), ((5.0, 5.0, 0.7),))
        r = self.run_ctrl(sc)                                    # no avoidance yet
        self.assertEqual(r.status, "collision")
        self.assertLessEqual(r.min_guard, 0.0)

    def test_dt_halved_close(self):
        p2 = Params(dt=0.025)
        r1 = self.run_ctrl(SCENARIOS["A"])
        r2 = self.run_ctrl(SCENARIOS["A"], p=p2)
        t1, t2 = r1.rows[-1]["t"], r2.rows[-1]["t"]
        self.assertLess(abs(t1 - t2), 0.5)


if __name__ == "__main__":
    unittest.main()
