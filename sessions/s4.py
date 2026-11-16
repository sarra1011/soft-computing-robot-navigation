"""Session 4 - obstacle avoidance and common supervision (tuning on B / C only)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from controllers import METHODS, make_controller
from metrics import write_csv
from plots import MODE_COLORS, arr, col, dedup_legend, draw_arena, mode_band, plot_comparison
from scenarios import SCENARIOS
from sessions.common import banner, done

SYSTEMS = ("sys-v1", "sys-v2")


def run(ctx):
    t0 = banner("SESSION 4 - obstacles and supervision (B = one obstacle, C = several)")
    p = ctx.params
    rows = []
    nominal = {}
    for sysv in SYSTEMS:
        for sid in ("B", "C"):
            sc = SCENARIOS[sid]
            res, noisy = {}, {}
            for meth in METHODS:
                r, m = ctx.run(sc, make_controller(meth, p, sysv), log_dir=ctx.logs / "session4")
                res[meth] = r
                rows.append(m)
                noisy[meth] = []
                for s in ctx.seeds[:5]:
                    rn, mn = ctx.run(sc, make_controller(meth, p, sysv), seed=s)
                    noisy[meth].append(rn)
                    rows.append(mn)
                print(f"  [{sysv} {sid}] {meth:8s} {m['status']:8s} T = {m['arrival_time']:6.2f} s "
                      f"L = {m['path_length']:5.2f} m  guard = {m['min_guard']:.2f} m  "
                      f"switches = {m['n_mode_switches']:4d}  sum|dw| = {m['sum_domega']:6.1f}  "
                      f"stops = {m['n_preventive_stops']}")
            nominal[(sysv, sid)] = res
            plot_comparison(res, sc, p, ctx.figs / f"s4_{sid}_{sysv}.png",
                            f"Session 4 - scenario {sid} ({sc.description}) - {sysv} - "
                            f"{p.version} - bold nominal, thin 5 noisy seeds", noisy)
    write_csv(rows, ctx.results / "s4_tuning_runs.csv")
    plot_chattering(ctx, nominal)
    blocking_analysis(ctx)
    done(t0)


def plot_chattering(ctx, nominal):
    """Why sys-v2: omega and mode chronology on B, v1 vs v2."""
    p = ctx.params
    fig, axs = plt.subplots(3, 2, figsize=(15, 9), gridspec_kw={"height_ratios": [2.2, 1, 0.5]})
    for j, sysv in enumerate(SYSTEMS):
        r = nominal[(sysv, "B")]["sugeno"]
        t = arr(r, "t")
        draw_arena(axs[0, j], SCENARIOS["B"], p, zoom=((2.5, 9.5), (2.8, 7.8)))
        for meth in METHODS:
            rr = nominal[(sysv, "B")][meth]
            axs[0, j].plot(arr(rr, "x"), arr(rr, "y"), color=col(meth), label=meth)
        axs[0, j].legend(fontsize=7, loc="lower right")
        axs[0, j].set_title(f"{sysv}: trajectories on B")
        axs[1, j].plot(t, arr(r, "omega"), color=col("sugeno"), lw=0.6)
        axs[1, j].plot(t, arr(r, "d_front_obs"), color="k", lw=0.8, label="d_front seen [m]")
        axs[1, j].axhline(1.0, color="tab:red", ls="--", lw=0.7, label="d_in = 1.0 m")
        axs[1, j].axhline(1.4, color="tab:purple", ls="--", lw=0.7, label="d_out = 1.4 m")
        axs[1, j].set_ylabel("omega [rad/s] / d [m]"); axs[1, j].legend(fontsize=6, ncol=3)
        sw = sum(1 for a, b in zip(r.rows, r.rows[1:]) if a["mode"] != b["mode"]
                 and a["mode"] in ("goal", "avoid") and b["mode"] in ("goal", "avoid"))
        axs[1, j].set_title(f"Sugeno on B: omega (green) and perceived d_front - {sw} mode switches",
                            fontsize=9)
        axs[1, j].set_xlim(0, t[-1]); axs[1, j].grid(alpha=0.3)
        mode_band(axs[2, j], r)
        axs[2, j].set_yticks([]); axs[2, j].set_xlim(0, t[-1]); axs[2, j].set_xlabel("t [s]")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in MODE_COLORS.values()]
    axs[2, 1].legend(handles, MODE_COLORS.keys(), fontsize=7, ncol=4, loc="lower right")
    fig.suptitle("Session 4 - sys-v1 (handout perception): d_front jumps to 3 m as soon as |beta| > pi/2, "
                 "the hysteresis is bypassed and the mode chatters at every step.\n"
                 "sys-v2: the avoided obstacle stays tracked -> continuous d_front, working hysteresis.",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(ctx.figs / "s4_chattering_v1_vs_v2.png", dpi=140)
    plt.close(fig)


def blocking_analysis(ctx):
    """Concave cup: a purely reactive controller gets trapped (local minimum)."""
    p = ctx.params
    sc = SCENARIOS["U"]
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.5))
    lines = []
    for j, sysv in enumerate(SYSTEMS):
        draw_arena(axs[j], sc, p, margin=False, zoom=((0.5, 9.5), (1.5, 8.5)))
        for meth in METHODS:
            r, m = ctx.run(sc, make_controller(meth, p, sysv), log_dir=ctx.logs / "session4")
            axs[j].plot(arr(r, "x"), arr(r, "y"), color=col(meth), lw=1,
                        label=f"{meth}: {m['status']}, blocked at {m['t_blocked']:.1f} s"
                        if m["blocked"] else f"{meth}: {m['status']}, no blocking")
            if m["blocked"]:
                k = int(round(m["t_blocked"] / p.dt))
                axs[j].plot(r.rows[k]["x"], r.rows[k]["y"], "o", color=col(meth), ms=8, mfc="none", mew=2)
            line = (f"{sysv} {meth:8s} status = {m['status']:8s} blocked = {bool(m['blocked'])} "
                    f"t_blocked = {m['t_blocked']:.2f} s  final rho = {r.rows[-1]['rho']:.2f} m  "
                    f"switches = {m['n_mode_switches']}")
            lines.append(line)
            print("  [U] " + line)
        dedup_legend(axs[j], loc="lower left")
        axs[j].set_title(f"{sysv} - circle = instant where 'blocked' is declared")
    fig.suptitle("Session 4 - blocking analysis in a concave cup (scenario U, demo only): "
                 "reactive control has no memory of the map -> local minimum. "
                 "A global planner (A*, visibility graph) would fix it.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(ctx.figs / "s4_blocking_cup.png", dpi=140)
    plt.close(fig)
    (ctx.results / "s4_blocking.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
