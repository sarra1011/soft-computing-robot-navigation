"""Session 3 - fuzzy goal seeking (no obstacles): Mamdani vs Sugeno."""
from __future__ import annotations

import math
import time

import matplotlib.pyplot as plt
import numpy as np

from controllers import ClassicController, FuzzyGoalController
from fuzzy_design import RHO_MAX, build_goal_engine, mamdani_centroid_constants
from metrics import write_csv
from plots import plot_comparison
from scenarios import SCENARIOS, Scenario
from sessions.common import banner, done

RHO = np.linspace(0, RHO_MAX, 143)
ALPHA = np.linspace(-math.pi, math.pi, 181)


def surface(engine):
    V = np.zeros((len(ALPHA), len(RHO)))
    W = np.zeros_like(V)
    for i, a in enumerate(ALPHA):
        for j, r in enumerate(RHO):
            o = engine.infer((r, a))
            V[i, j], W[i, j] = o["v"], o["omega"]
    return V, W


def goal_controllers(p):
    return {
        "classic": ClassicController(p),
        "mamdani": FuzzyGoalController(p, build_goal_engine("mamdani"), "mamdani"),
        "sugeno": FuzzyGoalController(p, build_goal_engine("sugeno"), "sugeno"),
        "sugeno_centroid": FuzzyGoalController(
            p, build_goal_engine("sugeno", constants=mamdani_centroid_constants()), "sugeno_centroid"),
    }


def run(ctx):
    t0 = banner("SESSION 3 - fuzzy navigation towards the goal")
    p = ctx.params
    surfaces = {k: surface(build_goal_engine(k)) for k in ("mamdani", "sugeno")}
    plot_surfaces(ctx, surfaces)
    plot_surfaces_3d(ctx, surfaces)
    plot_pi_boundary(ctx)
    grid_resolution(ctx)

    extra = Scenario("G", "demo", (8.0, 2.0, math.pi / 2), (2.0, 3.0), (),
                     "Goal on the left-behind side")
    rows = []
    for sc in (SCENARIOS["A"], SCENARIOS["A_back"], extra):
        res = {}
        for name, ctrl in goal_controllers(p).items():
            r, m = ctx.run(sc, ctrl, log_dir=ctx.logs / "session3")
            res[name] = r
            rows.append(m)
            print(f"  [{sc.id:6s}] {name:16s} {m['status']:8s} T = {m['arrival_time']:6.2f} s  "
                  f"L = {m['path_length']:5.2f} m  sum|dw| = {m['sum_domega']:.2f}")
        plot_comparison(res, sc, p, ctx.figs / f"s3_trajectories_{sc.id}.png",
                        f"Session 3 - scenario {sc.id} - goal controllers only (no avoidance) - "
                        f"{p.version}, 301-pt Mamdani grid")
    write_csv(rows, ctx.results / "s3_goal_runs.csv")
    done(t0)


def plot_surfaces(ctx, surfaces):
    fig, axs = plt.subplots(2, 3, figsize=(17, 9))
    ext = [RHO[0], RHO[-1], ALPHA[0], ALPHA[-1]]
    for i, (out, lab, cmap, lim) in enumerate((("v", "v [m/s]", "viridis", (0, 0.6)),
                                               ("omega", "omega [rad/s]", "RdBu_r", (-1.5, 1.5)))):
        for j, k in enumerate(("mamdani", "sugeno")):
            Z = surfaces[k][i]
            im = axs[i, j].imshow(Z, origin="lower", aspect="auto", extent=ext, cmap=cmap,
                                  vmin=lim[0], vmax=lim[1])
            cs = axs[i, j].contour(RHO, ALPHA, Z, levels=8, colors="k", linewidths=0.4)
            axs[i, j].clabel(cs, fontsize=6)
            axs[i, j].set_title(f"{k.capitalize()} : {lab}")
            plt.colorbar(im, ax=axs[i, j])
        D = surfaces["mamdani"][i] - surfaces["sugeno"][i]
        m = np.abs(D).max()
        im = axs[i, 2].imshow(D, origin="lower", aspect="auto", extent=ext, cmap="PuOr",
                              vmin=-m, vmax=m)
        axs[i, 2].set_title(f"Mamdani - Sugeno : {lab} (max |diff| = {m:.3f})")
        plt.colorbar(im, ax=axs[i, 2])
    for a in axs.ravel():
        a.set_xlabel("rho [m]"); a.set_ylabel("alpha [rad]")
        for y in (-1.2, -0.4, -0.25, 0.25, 0.4, 1.2):
            a.axhline(y, color="w", lw=0.3, ls=":")
    fig.suptitle("Session 3 - control surfaces v(rho, alpha) and omega(rho, alpha) - 15 rules, "
                 "Mamdani 301-pt grid, Sugeno constants of the handout")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(ctx.figs / "s3_surfaces.png", dpi=140)
    plt.close(fig)


def plot_surfaces_3d(ctx, surfaces):
    R, A = np.meshgrid(RHO, ALPHA)
    fig = plt.figure(figsize=(14, 10))
    k = 1
    for i, lab in enumerate(("v [m/s]", "omega [rad/s]")):
        for name in ("mamdani", "sugeno"):
            ax = fig.add_subplot(2, 2, k, projection="3d")
            ax.plot_surface(R, A, surfaces[name][i], cmap="viridis" if i == 0 else "RdBu_r",
                            linewidth=0, antialiased=True, rcount=70, ccount=70)
            ax.set_xlabel("rho [m]"); ax.set_ylabel("alpha [rad]"); ax.set_zlabel(lab)
            ax.set_title(f"{name.capitalize()} - {lab}")
            ax.view_init(28, -135)
            k += 1
    fig.suptitle("Session 3 - control surfaces (3D view)")
    fig.tight_layout()
    fig.savefig(ctx.figs / "s3_surfaces_3d.png", dpi=130)
    plt.close(fig)


def plot_pi_boundary(ctx):
    """alpha = +pi and alpha = -pi are the same physical direction."""
    a = np.linspace(-math.pi, math.pi, 2001)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    for k, ls in (("mamdani", "-"), ("sugeno", "--")):
        e = build_goal_engine(k)
        for r, c in ((0.8, "tab:blue"), (3.0, "tab:orange"), (8.0, "tab:green")):
            o = [e.infer((r, x)) for x in a]
            axs[0].plot(a, [q["v"] for q in o], ls=ls, color=c, label=f"{k}, rho = {r} m")
            axs[1].plot(a, [q["omega"] for q in o], ls=ls, color=c, label=f"{k}, rho = {r} m")
    for ax, lab in zip(axs, ("v [m/s]", "omega [rad/s]")):
        ax.axvspan(-math.pi, -1.2, color="0.92"); ax.axvspan(1.2, math.pi, color="0.92")
        ax.set_xlabel("alpha [rad]"); ax.set_ylabel(lab); ax.grid(alpha=0.3)
    axs[1].legend(fontsize=7, ncol=2)
    axs[1].annotate("jump: TD at -pi, TG at +pi\n(same physical direction)",
                    xy=(math.pi, 1.2), xytext=(0.2, -0.6), fontsize=8,
                    arrowprops=dict(arrowstyle="->"))
    fig.suptitle("Session 3 - behaviour at the alpha = +/-pi boundary (grey = GN / GP, robot stops "
                 "translating and turns on the spot)")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(ctx.figs / "s3_alpha_pi_boundary.png", dpi=150)
    plt.close(fig)


def grid_resolution(ctx):
    """151 vs 301 vs 601 points: accuracy (vs a 4001-pt reference) and cost."""
    ref = surface(build_goal_engine("mamdani", 4001))
    rows = []
    rng = np.random.default_rng(1)
    pts = np.c_[rng.uniform(0, RHO_MAX, 2000), rng.uniform(-math.pi, math.pi, 2000)]
    for n in (151, 301, 601):
        e = build_goal_engine("mamdani", n)
        V, W = surface(e)
        t0 = time.perf_counter()
        for r, a in pts:
            e.infer((r, a))
        us = (time.perf_counter() - t0) / len(pts) * 1e6
        rows.append({"n_points": n, "max_err_v": np.abs(V - ref[0]).max(),
                     "max_err_omega": np.abs(W - ref[1]).max(),
                     "mean_err_v": np.abs(V - ref[0]).mean(),
                     "mean_err_omega": np.abs(W - ref[1]).mean(), "us_per_inference": us})
    e = build_goal_engine("sugeno")
    t0 = time.perf_counter()
    for r, a in pts:
        e.infer((r, a))
    rows.append({"n_points": "sugeno", "max_err_v": "", "max_err_omega": "", "mean_err_v": "",
                 "mean_err_omega": "", "us_per_inference": (time.perf_counter() - t0) / len(pts) * 1e6})
    write_csv(rows, ctx.results / "s3_grid_resolution.csv")
    print("  Mamdani grid resolution (errors vs 4001-pt reference):")
    for r in rows:
        if r["n_points"] == "sugeno":
            print(f"    Sugeno (no grid)        {r['us_per_inference']:.1f} us/inference")
        else:
            print(f"    {r['n_points']:4d} pts: max err v = {r['max_err_v']:.2e} m/s, "
                  f"omega = {r['max_err_omega']:.2e} rad/s, {r['us_per_inference']:.1f} us/inference")
    fig, ax = plt.subplots(figsize=(7, 4))
    ns = [r["n_points"] for r in rows[:3]]
    ax.semilogy(ns, [r["max_err_v"] for r in rows[:3]], "o-", label="max error v [m/s]")
    ax.semilogy(ns, [r["max_err_omega"] for r in rows[:3]], "s-", label="max error omega [rad/s]")
    ax2 = ax.twinx()
    ax2.plot(ns, [r["us_per_inference"] for r in rows[:3]], "k^--", label="cost [us / inference]")
    ax2.set_ylabel("cost [us / inference]")
    ax.set_xlabel("points of the output grid"); ax.set_ylabel("error vs 4001-pt grid")
    ax.set_xticks(ns); ax.grid(alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8)
    ax.set_title("Session 3 - Mamdani defuzzification grid: accuracy vs cost")
    fig.tight_layout()
    fig.savefig(ctx.figs / "s3_grid_resolution.png", dpi=150)
    plt.close(fig)
