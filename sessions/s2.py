"""Session 2 - fuzzy engine built from scratch, verified by hand and against
scikit-fuzzy (if installed)."""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from fuzzy import LinguisticVariable, MamdaniEngine, Rule, SugenoEngine, trap, tri
from fuzzy_design import avoid_inputs, build_goal_engine, goal_inputs, motion_outputs
from metrics import write_csv
from sessions.common import banner, done


# ------------------------------------------------------------------ exercise
def exercise_engines(n_points=101):
    u = LinguisticVariable("u", 0, 1, [trap("Bas", 0, 0, 0.2, 0.6), trap("Haut", 0.2, 0.6, 1, 1)])
    out = LinguisticVariable("y", 0, 1, [tri("Faible", 0.0, 0.2, 0.5), tri("Fort", 0.5, 0.8, 1.0)])
    rules = [Rule(("Bas",), (("y", "Faible"),)), Rule(("Haut",), (("y", "Fort"),))]
    sug = SugenoEngine([u], {"y": out}, rules, {"y": {"Faible": 0.2, "Fort": 0.8}})
    mam = MamdaniEngine([u], {"y": out}, rules, n_points)
    return u, out, sug, mam


def manual_mamdani(lb, lh, n_points):
    """Same computation written as explicit sums (no engine), for the comparison."""
    num = den = 0.0
    for i in range(n_points):
        z = i / (n_points - 1)
        f = max(0.0, min((z - 0.0) / 0.2, (0.5 - z) / 0.3))     # Faible
        g = max(0.0, min((z - 0.5) / 0.3, (1.0 - z) / 0.2))     # Fort
        mu = max(min(lb, f), min(lh, g))
        num += z * mu
        den += mu
    return num / den


def run(ctx):
    t0 = banner("SESSION 2 - fuzzy engine")
    lines = []
    u, out, sug, mam = exercise_engines()
    lb, lh = u.fuzzify(0.25)
    ys = sug.infer((0.25,))["y"]
    lines += ["Manual check (handout): u = 0.25",
              f"  lambda_Bas  = {lb:.6f}   (expected 0.875)",
              f"  lambda_Haut = {lh:.6f}   (expected 0.125)",
              f"  Sugeno out  = {ys:.6f}   (expected 0.275)"]
    assert abs(lb - 0.875) < 1e-12 and abs(lh - 0.125) < 1e-12 and abs(ys - 0.275) < 1e-12
    lines.append("Mamdani with Faible = tri(0, 0.2, 0.5), Fort = tri(0.5, 0.8, 1.0):")
    for n in (11, 101, 1001):
        _, _, _, m = exercise_engines(n)
        lines.append(f"  grid {n:5d} pts : engine = {m.infer((0.25,))['y']:.6f}   "
                     f"explicit sums = {manual_mamdani(lb, lh, n):.6f}")
    for l in lines:
        print("  " + l)
    plot_exercise(ctx, u, out, mam, lb, lh)
    plot_shapes(ctx)
    plot_partitions(ctx)
    lines += domain_tests()
    lines += crosscheck_skfuzzy(ctx)
    (ctx.results / "s2_engine_checks.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    done(t0)


def domain_tests():
    out = ["Membership bounds and coverage (min over domain of max_i mu_i, must be > 0):"]
    for var in (*goal_inputs(), *motion_outputs(), *avoid_inputs()):
        z = np.linspace(var.lo - 1, var.hi + 1, 5001)
        vals = np.vstack([s.eval(z) for s in var.sets])
        assert vals.min() >= 0 and vals.max() <= 1
        out.append(f"  {var.name:8s} sets {var.names}  in [0,1]: yes   coverage = {var.coverage():.3f}")
    rho, _ = goal_inputs()
    before = rho.n_out_of_domain
    rho.fuzzify(-1.0); rho.fuzzify(20.0)
    out.append(f"  out-of-domain inputs saturated and counted: {rho.n_out_of_domain - before} events")
    for l in out:
        print("  " + l)
    return out


def crosscheck_skfuzzy(ctx):
    """Same Mamdani surface with scikit-fuzzy (min/max, centroid)."""
    try:
        import skfuzzy as fz
    except ImportError:
        msg = "scikit-fuzzy not installed: cross-check skipped (pip install scikit-fuzzy)"
        print("  " + msg)
        return [msg]
    eng = build_goal_engine("mamdani", 301)
    rho_v, alpha_v = eng.inputs
    rows, worst = [], 0.0
    rng = np.random.default_rng(0)
    pts = np.c_[rng.uniform(0, 14.2, 300), rng.uniform(-math.pi, math.pi, 300)]
    for r, a in pts:
        lam, _ = eng.activations((r, a))
        res = {}
        for o, var in eng.outputs.items():
            z = eng.grid[o]
            agg = np.zeros_like(z)
            for j, rule in enumerate(eng.rules):
                mf = var[rule.cons(o)]
                shape = (fz.trimf(z, [mf.a, mf.b, mf.d]) if mf.kind == "tri"
                         else fz.trapmf(z, [mf.a, mf.b, mf.c, mf.d]))
                agg = np.fmax(agg, np.fmin(lam[j], shape))
            res[o] = fz.defuzz(z, agg, "centroid")
        mine = eng.infer((r, a))
        dv, dw = abs(mine["v"] - res["v"]), abs(mine["omega"] - res["omega"])
        worst = max(worst, dv, dw)
        rows.append({"rho": r, "alpha": a, "v_ours": mine["v"], "v_skfuzzy": res["v"],
                     "omega_ours": mine["omega"], "omega_skfuzzy": res["omega"]})
    write_csv(rows, ctx.results / "s2_skfuzzy_crosscheck.csv")
    msg = (f"scikit-fuzzy {fz.__version__} cross-check on 300 random (rho, alpha): "
           f"max |difference| = {worst:.2e}  (skfuzzy centroid integrates the piecewise-linear "
           f"curve, ours is the discrete sum of the handout; differences come from that)")
    print("  " + msg)
    return [msg]


# ------------------------------------------------------------------ figures
def plot_exercise(ctx, u, out, mam, lb, lh):
    ex = mam.explain((0.25,))
    z = ex["grid"]["y"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    zz = np.linspace(0, 1, 501)
    for s, c in zip(u.sets, ("tab:blue", "tab:orange")):
        axs[0].plot(zz, s.eval(zz), color=c, label=repr(s))
    axs[0].axvline(0.25, color="k", ls="--", lw=0.8)
    axs[0].plot([0.25, 0.25], [lb, lh], "ko")
    axs[0].annotate(f"0.875", (0.25, lb), xytext=(0.32, 0.9))
    axs[0].annotate(f"0.125", (0.25, lh), xytext=(0.32, 0.15))
    axs[0].set_title("Fuzzification of u = 0.25"); axs[0].set_xlabel("u"); axs[0].set_ylabel("mu")
    axs[0].legend(fontsize=8)
    for s, c, lam in zip(out.sets, ("tab:blue", "tab:orange"), (lb, lh)):
        axs[1].plot(z, s.eval(z), color=c, ls=":", label=f"{repr(s)}")
        axs[1].plot(z, np.minimum(lam, s.eval(z)), color=c, lw=1, label=f"min({lam:.3f}, mu)")
    axs[1].fill_between(z, ex["agg"]["y"], color="0.7", alpha=0.6, label="aggregation (max)")
    zs = ex["out"]["y"]
    axs[1].axvline(zs, color="k", lw=2, label=f"centre of gravity = {zs:.4f}")
    axs[1].axvline(0.275, color="tab:green", ls="--", label="Sugeno = 0.275")
    axs[1].set_title(f"Mamdani inference, grid {len(z)} pts"); axs[1].set_xlabel("y")
    axs[1].legend(fontsize=7)
    fig.suptitle("Session 2 - verification exercise (handout section 4)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(ctx.figs / "s2_exercise_inference.png", dpi=150)
    plt.close(fig)


def plot_shapes(ctx):
    z = np.linspace(-0.5, 1.5, 2001)
    shapes = [tri("triangle (0, 0.5, 1)", 0, 0.5, 1),
              trap("trapezoid (0, 0.3, 0.7, 1)", 0, 0.3, 0.7, 1),
              trap("left shoulder (0, 0, 0.3, 0.7)", 0, 0, 0.3, 0.7),
              trap("right shoulder (0.3, 0.7, 1, 1)", 0.3, 0.7, 1, 1)]
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.4), sharey=True)
    for a, s in zip(axs, shapes):
        mu = s.eval(z)
        a.plot(z, mu, lw=2)
        a.axvspan(-0.5, 0, color="0.9"); a.axvspan(1, 1.5, color="0.9")
        a.set_title(s.name, fontsize=9); a.set_xlabel("z"); a.grid(alpha=0.3)
        a.text(0.02, 0.9, f"min {mu.min():.2f} / max {mu.max():.2f}", transform=a.transAxes, fontsize=8)
    axs[0].set_ylabel("mu(z)")
    fig.suptitle("Session 2 - basic shapes (grey = outside the domain [0, 1]); no division by zero on plateaus")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(ctx.figs / "s2_basic_shapes.png", dpi=150)
    plt.close(fig)


def plot_partitions(ctx):
    vars_ = [*goal_inputs(), *motion_outputs(), *avoid_inputs()]
    fig, axs = plt.subplots(3, 2, figsize=(13, 10))
    for a, var in zip(axs.ravel(), vars_):
        z = np.linspace(var.lo, var.hi, 2001)
        for s in var.sets:
            mu = s.eval(z)
            a.plot(z, mu, lw=2, label=repr(s))
            a.fill_between(z, mu, alpha=0.08)
        a.set_title(f"{var.name} [{var.unit}]  domain [{var.lo:.3g}, {var.hi:.3g}]  "
                    f"coverage = {var.coverage():.2f}", fontsize=9)
        a.set_xlabel(f"{var.name} [{var.unit}]"); a.set_ylabel("mu")
        a.legend(fontsize=6, loc="center right"); a.grid(alpha=0.3); a.set_ylim(-0.03, 1.1)
    fig.suptitle("Fuzzy partitions - goal inputs (rho, alpha), outputs (v, omega), avoidance inputs (d_front, beta)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(ctx.figs / "s2_s3_s4_partitions.png", dpi=150)
    plt.close(fig)
