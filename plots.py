"""Shared plotting helpers. Every figure carries a legend, units, the scenario
id and the parameter / system version (handout requirement)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402

COLORS = {"classic": "#1f77b4", "mamdani": "#d62728", "sugeno": "#2ca02c",
          "mlp": "#9467bd", "sugeno_ga": "#ff7f0e", "sugeno_centroid": "#17becf"}
MODE_COLORS = {"goal": "#4c72b0", "avoid": "#dd8452", "stop": "#c44e52",
               "arrived": "#55a868"}
MODE_LEVEL = {"goal": 0, "avoid": 1, "stop": 2, "arrived": 3}


def col(name):
    for k, c in COLORS.items():
        if name == k:
            return c
    return COLORS.get(name.split("_")[0], "k")


def arr(res, key):
    return np.array([r[key] for r in res.rows], dtype=float)


def draw_arena(ax, sc, p, margin=True, zoom=None):
    ax.add_patch(plt.Rectangle((0, 0), p.arena, p.arena, fill=False, ec="k", lw=1.5))
    for (cx, cy, ro) in sc.obstacles:
        ax.add_patch(plt.Circle((cx, cy), ro, fc="0.6", ec="k", zorder=3, label="obstacle"))
        if margin:
            ax.add_patch(plt.Circle((cx, cy), ro + p.robot_radius, fill=False, ec="k", ls=":",
                                    lw=0.8, label="contact (centre at r_o + R)"))
            ax.add_patch(plt.Circle((cx, cy), ro + p.robot_radius + p.margin, fill=False,
                                    ls="--", ec="tab:red", lw=0.8,
                                    label=f"desired margin (+{p.margin} m)"))
    gx, gy = sc.goal
    ax.add_patch(plt.Circle((gx, gy), p.goal_tol, fc="none", ec="tab:green", lw=1.2))
    ax.plot(gx, gy, "*", color="tab:green", ms=13, zorder=5, label="goal")
    ax.plot(sc.start[0], sc.start[1], "ks", ms=6, zorder=5, label="start")
    th = sc.start[2]
    ax.annotate("", xy=(sc.start[0] + 0.6 * np.cos(th), sc.start[1] + 0.6 * np.sin(th)),
                xytext=sc.start[:2], arrowprops=dict(arrowstyle="->", lw=1.2))
    if zoom:
        ax.set_xlim(*zoom[0]); ax.set_ylim(*zoom[1])
    else:
        ax.set_xlim(-0.2, p.arena + 0.2); ax.set_ylim(-0.2, p.arena + 0.2)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")


def dedup_legend(ax, loc="lower right", fs=7):
    h, l = ax.get_legend_handles_labels()
    d = dict(zip(l, h))
    ax.legend(d.values(), d.keys(), loc=loc, fontsize=fs, framealpha=0.85)


def status_label(res):
    if res.status == "arrived":
        return f"T = {res.rows[-1]['t']:.2f} s"
    return res.status.upper()


def plot_episode(res, sc, p, path, title_extra=""):
    """x-y, rho(t), alpha(t), v(t), omega(t) (+ mode band) in one figure."""
    t, x, y, th = arr(res, "t"), arr(res, "x"), arr(res, "y"), arr(res, "theta")
    fig = plt.figure(figsize=(13, 7.5))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.15, 1])
    ax = fig.add_subplot(gs[:, 0])
    draw_arena(ax, sc, p)
    ax.plot(x, y, "-", color=col(res.controller), lw=2, label=f"{res.controller} trajectory")
    idx = np.arange(0, len(t), 40)
    ax.quiver(x[idx], y[idx], np.cos(th[idx]), np.sin(th[idx]), color=col(res.controller),
              scale=30, width=0.004, alpha=0.7)
    ax.add_patch(plt.Circle((x[-1], y[-1]), p.robot_radius, fc="none", ec=col(res.controller)))
    dedup_legend(ax)
    ax.set_title("Trajectory (arrows = heading every 2 s)")

    def panel(row, key, ylabel, lines=()):
        a = fig.add_subplot(gs[row, 1])
        a.plot(t, arr(res, key), color=col(res.controller))
        for val, lab in lines:
            a.axhline(val, color="tab:red", ls="--", lw=0.8, label=lab)
        a.set_ylabel(ylabel); a.grid(alpha=0.3)
        if lines:
            a.legend(fontsize=7, loc="best")
        return a

    panel(0, "rho", "rho [m]", [(p.goal_tol, "arrival radius")])
    panel(1, "alpha", "alpha [rad]", [(0, "alpha = 0")])
    panel(2, "v", "v [m/s]", [(p.v_max, "v_max")])
    a = panel(3, "omega", "omega [rad/s]", [(p.w_max, "+w_max"), (-p.w_max, "-w_max")])
    a.set_xlabel("t [s]")
    fig.suptitle(f"Scenario {sc.id} - {res.controller} ({res.version}) - {res.condition} - "
                 f"{p.version}, dt = {p.dt} s - {status_label(res)} {title_extra}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def mode_band(ax, res, y0=0.0, h=1.0):
    """Coloured band of the supervision mode against time."""
    t = arr(res, "t")
    modes = [r["mode"] for r in res.rows]
    start = 0
    for i in range(1, len(modes) + 1):
        if i == len(modes) or modes[i] != modes[start]:
            t1 = t[i] if i < len(t) else t[-1] + 0.05
            ax.axvspan(t[start], t1, ymin=y0, ymax=y0 + h,
                       color=MODE_COLORS.get(modes[start], "0.5"), lw=0)
            start = i


def plot_comparison(results, sc, p, path, title, noisy=None, zoom=None):
    """results: dict name -> nominal EpisodeResult; noisy: dict name -> list."""
    names = list(results)
    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.1, 1], height_ratios=[1, 1, 1, 0.55])
    ax = fig.add_subplot(gs[:, 0])
    draw_arena(ax, sc, p, zoom=zoom)
    for n in names:
        for r in (noisy or {}).get(n, []):
            ax.plot(arr(r, "x"), arr(r, "y"), color=col(n), lw=0.5, alpha=0.25)
        r = results[n]
        ax.plot(arr(r, "x"), arr(r, "y"), color=col(n), lw=2,
                label=f"{n}: {status_label(r)}, L = {np.sum(np.hypot(*np.diff(np.c_[arr(r,'x'), arr(r,'y')], axis=0).T)):.2f} m")
        if r.status == "collision":
            ax.plot(arr(r, "x")[-1], arr(r, "y")[-1], "x", color=col(n), ms=12, mew=3)
    dedup_legend(ax, fs=7)
    ax.set_title("Trajectories (bold = nominal" + (", thin = 20 noisy seeds)" if noisy else ")"))
    for i, (key, lab) in enumerate((("rho", "rho [m]"), ("v", "v [m/s]"), ("omega", "omega [rad/s]"))):
        a = fig.add_subplot(gs[i, 1])
        for n in names:
            a.plot(arr(results[n], "t"), arr(results[n], key), color=col(n), lw=1, label=n)
        a.set_ylabel(lab); a.grid(alpha=0.3)
        if i == 0:
            a.legend(fontsize=7)
    a = fig.add_subplot(gs[3, 1])
    for j, n in enumerate(names):
        mode_band(a, results[n], y0=1 - (j + 1) / len(names), h=1 / len(names) * 0.9)
    a.set_yticks([1 - (j + 0.55) / len(names) for j in range(len(names))])
    a.set_yticklabels(names, fontsize=7)
    a.set_ylim(0, 1); a.set_xlabel("t [s]")
    a.set_xlim(0, max(arr(results[n], "t")[-1] for n in names))
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in MODE_COLORS.values()]
    a.legend(handles, MODE_COLORS.keys(), fontsize=6, ncol=4, loc="upper right",
             bbox_to_anchor=(1, 1.45))
    a.set_title("supervision mode", fontsize=8, loc="left")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=140)
    plt.close(fig)
