"""Session 1 - simulator and classic reference (no avoidance)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from controllers import ClassicController
from metrics import write_summary_csv
from model import Params
from plots import arr, draw_arena, plot_episode
from scenarios import SCENARIOS, save_all, validate
from sessions.common import banner, done


def run(ctx):
    t0 = banner("SESSION 1 - simulator + classic reference")
    p = ctx.params
    p.to_json(ctx.configs / "params.json")
    save_all(ctx.configs, p)
    summary = []
    for sid in ("A", "A_back"):
        sc = SCENARIOS[sid]
        validate(sc, p)
        res, m = ctx.run(sc, ClassicController(p), log_dir=ctx.logs / "session1")
        plot_episode(res, sc, p, ctx.figs / f"s1_reference_{sid}.png")
        summary.append(m)
        print(f"  [{sid:6s}] {m['status']:8s} T = {m['arrival_time']:.2f} s, "
              f"L = {m['path_length']:.2f} m")
    write_summary_csv(summary, ctx.results / "s1_summary.csv")
    ablation(ctx)
    dt_comparison(ctx)
    done(t0)


def ablation(ctx):
    """Role of cos(alpha) and of the saturations (target behind the robot)."""
    p = ctx.params
    sc = SCENARIOS["A_back"]
    big = Params(v_max=1e3, w_max=1e3, version=p.version + "-nosat")
    variants = [("reference", p, ClassicController(p)),
                ("without cos(alpha)", p, ClassicController(p, use_cos=False, tag="_nocos")),
                ("without saturations", big, ClassicController(big, tag="_nosat"))]
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    draw_arena(axs[0, 0], sc, p, margin=False)
    print("  Ablation on A_back:")
    lines = []
    for (label, pp, ctrl), c in zip(variants, ("tab:blue", "tab:orange", "tab:green")):
        res, m = ctx.run(sc, ctrl, params=pp)
        t = arr(res, "t")
        axs[0, 0].plot(arr(res, "x"), arr(res, "y"), color=c, label=label)
        axs[0, 1].plot(t, arr(res, "rho"), color=c, label=label)
        axs[1, 0].plot(t, arr(res, "v"), color=c, label=label)
        axs[1, 1].plot(t, arr(res, "omega"), color=c, label=label)
        line = (f"{label:20s} {m['status']:8s} T = {m['arrival_time']:.2f} s  "
                f"L = {m['path_length']:.2f} m  max v = {arr(res, 'v').max():.2f} m/s  "
                f"max |omega| = {np.abs(arr(res, 'omega')).max():.2f} rad/s  "
                f"max rho = {arr(res, 'rho').max():.2f} m")
        lines.append(line)
        print("    " + line)
    (ctx.results / "s1_ablation.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    axs[0, 0].legend(fontsize=7, loc="lower right"); axs[0, 0].set_title("Trajectory")
    for a, yl in ((axs[0, 1], "rho [m]"), (axs[1, 0], "v [m/s]"), (axs[1, 1], "omega [rad/s]")):
        a.set_xlabel("t [s]"); a.set_ylabel(yl); a.grid(alpha=0.3); a.legend(fontsize=7)
        a.set_xlim(0, 12)
    fig.suptitle(f"Scenario A_back - role of cos(alpha) and saturations - classic v1.0 - {p.version}")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(ctx.figs / "s1_ablation_cos_saturation.png", dpi=150)
    plt.close(fig)


def dt_comparison(ctx):
    p = ctx.params
    sc = SCENARIOS["A"]
    p2 = Params(dt=0.025, version=p.version + "-dt0.025")
    r1, _ = ctx.run(sc, ClassicController(p))
    r2, _ = ctx.run(sc, ClassicController(p2), params=p2)
    t1, t2 = arr(r1, "t"), arr(r2, "t")
    xy1, xy2 = np.c_[arr(r1, "x"), arr(r1, "y")], np.c_[arr(r2, "x"), arr(r2, "y")]
    n = min(len(t1), (len(t2) + 1) // 2)
    dev = np.hypot(*(xy1[:n] - xy2[0:2 * n:2]).T)
    txt = (f"dt = 0.050 s : T = {t1[-1]:.3f} s, final ({xy1[-1, 0]:.4f}, {xy1[-1, 1]:.4f})\n"
           f"dt = 0.025 s : T = {t2[-1]:.3f} s, final ({xy2[-1, 0]:.4f}, {xy2[-1, 1]:.4f})\n"
           f"|dT| = {abs(t1[-1] - t2[-1]):.3f} s ; max gap = {dev.max() * 100:.2f} cm ; "
           f"mean gap = {dev.mean() * 100:.2f} cm\n")
    print("  Step size:\n    " + txt.replace("\n", "\n    ").rstrip())
    (ctx.results / "s1_dt_comparison.txt").write_text(txt, encoding="utf-8")
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    draw_arena(axs[0], sc, p, margin=False)
    axs[0].plot(xy1[:, 0], xy1[:, 1], label="dt = 0.05 s", lw=2)
    axs[0].plot(xy2[:, 0], xy2[:, 1], "--", label="dt = 0.025 s", lw=1.5)
    axs[0].legend(fontsize=8, loc="lower right"); axs[0].set_title("Trajectories")
    axs[1].plot(t1[:n], dev * 100)
    axs[1].set_xlabel("t [s]"); axs[1].set_ylabel("position gap [cm]"); axs[1].grid(alpha=0.3)
    axs[1].set_title("Gap between the two runs at common instants")
    fig.suptitle(f"Scenario A - classic v1.0 - effect of the time step - {p.version}")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(ctx.figs / "s1_dt_comparison.png", dpi=150)
    plt.close(fig)
