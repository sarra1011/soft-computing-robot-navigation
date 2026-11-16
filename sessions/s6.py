"""Session 6 - extensions (both options are implemented; the handout asks the
report to present ONE of them).

Option A: MLP imitating the Sugeno goal controller, closed loop on D, E, F.
Option B: real-coded genetic algorithm on (s_v, s_omega, d_in) of the Sugeno
system, compared with random search at equal number of evaluations.
"""
from __future__ import annotations

import copy
import math
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np

from controllers import MLPGoalController, make_controller
from fuzzy_design import RHO_MAX, build_goal_engine
from metrics import aggregate, write_csv, write_summary_csv
from plots import arr, col, draw_arena
from scenarios import SCENARIOS
from sessions.common import banner, done, fmt_ms
from sessions.s5 import TEST, run_campaign

EXT_SYSTEM = "sys-v1"      # fixed before the test campaign: the handout system


# =========================================================================== #
# Option A - imitation network
# =========================================================================== #
def make_dataset(n=10_000, seed=42):
    rng = np.random.default_rng(seed)
    rho = rng.uniform(0, RHO_MAX, n)
    alpha = rng.uniform(-math.pi, math.pi, n)
    teacher = build_goal_engine("sugeno")
    Y = np.array([[o["v"], o["omega"]] for o in (teacher.infer((r, a)) for r, a in zip(rho, alpha))])
    X = np.c_[rho / RHO_MAX, np.sin(alpha), np.cos(alpha)]
    idx = rng.permutation(n)                     # fixed split, seed 42
    n_tr, n_va = int(0.70 * n), int(0.15 * n)
    return X, Y, np.c_[rho, alpha], idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]


def train_mlp(Xtr, Ytr, Xva, Yva, init, max_epochs, patience=40):
    from sklearn.neural_network import MLPRegressor
    net = MLPRegressor(hidden_layer_sizes=(32, 16), activation="relu", solver="adam",
                       learning_rate_init=1e-3, batch_size=64, random_state=init)
    hist, best, best_net, wait = [], math.inf, None, 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ep in range(max_epochs):
            net.partial_fit(Xtr, Ytr)
            tr = float(np.mean((net.predict(Xtr) - Ytr) ** 2))
            va = float(np.mean((net.predict(Xva) - Yva) ** 2))
            hist.append((ep + 1, tr, va))
            if va < best - 1e-7:
                best, best_net, wait = va, copy.deepcopy(net), 0
            else:
                wait += 1
                if wait >= patience:
                    break
    best_epoch = min(hist, key=lambda h: h[2])[0]
    return best_net, hist, best_epoch


def option_a(ctx):
    from sklearn.preprocessing import StandardScaler
    p = ctx.params
    print("  Option A - MLP imitation of the Sugeno goal controller")
    X, Y, RA, itr, iva, ite = make_dataset()
    scaler = StandardScaler().fit(X[itr])            # fitted on the training set ONLY
    Yn = Y / np.array([p.v_max, p.w_max])            # v / 0.60, omega / 1.50
    Xs = scaler.transform(X)
    nets, rows, hists = [], [], []
    max_ep = 60 if ctx.quick else 400
    for init in (0, 1, 2):
        t0 = time.perf_counter()
        net, hist, best_ep = train_mlp(Xs[itr], Yn[itr], Xs[iva], Yn[iva], init, max_ep)
        pred = net.predict(Xs[ite]) * np.array([p.v_max, p.w_max])
        err = np.abs(pred - Y[ite])
        row = {"init": init, "best_epoch": best_ep, "epochs_run": len(hist),
               "val_mse_norm": min(h[2] for h in hist),
               "test_mae_v_m_s": err[:, 0].mean(), "test_mae_omega_rad_s": err[:, 1].mean(),
               "test_max_v": err[:, 0].max(), "test_max_omega": err[:, 1].max(),
               "train_seconds": time.perf_counter() - t0}
        rows.append(row); nets.append(net); hists.append(hist)
        print(f"    init {init}: best epoch {best_ep:3d}/{len(hist)}  test MAE v = "
              f"{row['test_mae_v_m_s']:.4f} m/s, omega = {row['test_mae_omega_rad_s']:.4f} rad/s "
              f"(max {row['test_max_omega']:.3f})")
    write_csv(rows, ctx.results / "s6a_mlp_errors.csv")
    plot_learning(ctx, hists, rows)
    plot_error_map(ctx, nets[0], scaler)

    # closed loop: only the goal controller is replaced; avoidance + supervisor kept
    builders = {f"sugeno|{EXT_SYSTEM}": lambda: make_controller("sugeno", p, EXT_SYSTEM)}
    for i, net in enumerate(nets):
        builders[f"mlp{i}|{EXT_SYSTEM}"] = (lambda net=net, i=i: make_controller(
            "sugeno", p, EXT_SYSTEM, goal=MLPGoalController(p, net, scaler), name=f"mlp{i}"))
    metrics, nominal, _ = run_campaign(ctx, builders=builders, label="s6a", log=False)
    write_summary_csv(metrics, ctx.results / "s6a_episodes.csv")
    agg = aggregate(metrics)
    write_csv(agg, ctx.results / "s6a_aggregate.csv")
    plot_closed_loop(ctx, nominal, ["sugeno"] + [f"mlp{i}" for i in range(3)], "s6a",
                     "Option A - Sugeno teacher vs 3 MLP students (goal law only)")
    return agg


def plot_learning(ctx, hists, rows):
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    for h, r, c in zip(hists, rows, ("tab:blue", "tab:orange", "tab:green")):
        h = np.array(h)
        axs[0].semilogy(h[:, 0], h[:, 1], color=c, ls="--", lw=1, label=f"init {r['init']} train")
        axs[0].semilogy(h[:, 0], h[:, 2], color=c, lw=1.5, label=f"init {r['init']} validation")
        axs[0].axvline(r["best_epoch"], color=c, lw=0.6, ls=":")
    axs[0].set_xlabel("epoch"); axs[0].set_ylabel("MSE (normalised outputs)")
    axs[0].legend(fontsize=7, ncol=2); axs[0].grid(alpha=0.3)
    axs[0].set_title("Learning curves (dotted = epoch kept, chosen on validation)")
    x = np.arange(3)
    axs[1].bar(x - 0.2, [r["test_mae_v_m_s"] for r in rows], 0.4, label="MAE v [m/s]")
    axs[1].bar(x + 0.2, [r["test_mae_omega_rad_s"] for r in rows], 0.4, label="MAE omega [rad/s]")
    for xi, r in zip(x, rows):
        axs[1].text(xi - 0.2, r["test_mae_v_m_s"], f"{r['test_mae_v_m_s']:.4f}", ha="center", va="bottom", fontsize=7)
        axs[1].text(xi + 0.2, r["test_mae_omega_rad_s"], f"{r['test_mae_omega_rad_s']:.4f}", ha="center", va="bottom", fontsize=7)
    axs[1].set_xticks(x); axs[1].set_xticklabels([f"init {r['init']}" for r in rows])
    axs[1].set_title("Test errors (15 % test set, 1 500 samples)"); axs[1].legend(fontsize=8)
    fig.suptitle("Session 6 option A - MLP (32, 16) imitating Sugeno, inputs (rho/14.2, sin alpha, cos alpha), "
                 "split 70/15/15 seed 42")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(ctx.figs / "s6a_learning.png", dpi=150)
    plt.close(fig)


def plot_error_map(ctx, net, scaler):
    p = ctx.params
    rho = np.linspace(0, RHO_MAX, 120)
    alpha = np.linspace(-math.pi, math.pi, 160)
    R, A = np.meshgrid(rho, alpha)
    X = scaler.transform(np.c_[R.ravel() / RHO_MAX, np.sin(A.ravel()), np.cos(A.ravel())])
    P = net.predict(X) * np.array([p.v_max, p.w_max])
    teacher = build_goal_engine("sugeno")
    T = np.array([[o["v"], o["omega"]] for o in (teacher.infer((r, a)) for r, a in zip(R.ravel(), A.ravel()))])
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    for k, lab in enumerate(("v [m/s]", "omega [rad/s]")):
        E = np.abs(P[:, k] - T[:, k]).reshape(R.shape)
        im = axs[k].imshow(E, origin="lower", aspect="auto", extent=[0, RHO_MAX, -math.pi, math.pi],
                           cmap="magma")
        plt.colorbar(im, ax=axs[k])
        axs[k].set_title(f"|MLP - Sugeno| on {lab}  (max {E.max():.3f})")
        axs[k].set_xlabel("rho [m]"); axs[k].set_ylabel("alpha [rad]")
    fig.suptitle("Session 6 option A - where the network deviates from its teacher (init 0): "
                 "errors concentrate on the kinks of the fuzzy surface")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(ctx.figs / "s6a_error_map.png", dpi=150)
    plt.close(fig)


def plot_closed_loop(ctx, nominal, names, tag, title):
    p = ctx.params
    fig, axs = plt.subplots(1, len(TEST), figsize=(6 * len(TEST), 6))
    for ax, sid in zip(axs, TEST):
        draw_arena(ax, SCENARIOS[sid], p)
        for n in names:
            r = nominal[(f"{n}|{EXT_SYSTEM}", sid)]
            st = f"T = {r.rows[-1]['t']:.1f} s" if r.status == "arrived" else r.status
            ax.plot(arr(r, "x"), arr(r, "y"), color=col(n) if n in ("sugeno", "sugeno_ga") else None,
                    lw=2 if n.startswith("sugeno") else 1, label=f"{n}: {st}")
        ax.legend(fontsize=7, loc="lower right"); ax.set_title(f"scenario {sid} (nominal)")
    fig.suptitle(f"Session 6 - {title} - {EXT_SYSTEM} - {p.version}")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(ctx.figs / f"{tag}_trajectories.png", dpi=140)
    plt.close(fig)


# =========================================================================== #
# Option B - genetic algorithm
# =========================================================================== #
BOUNDS = np.array([[0.5, 1.2],     # s_v
                   [0.6, 1.4],     # s_omega
                   [0.7, 1.2]])    # d_in [m]   (d_out = d_in + 0.4)
GENES = ("s_v", "s_omega", "d_in")


class Objective:
    """J = mean over A, B of [100 Ic + 30 Ie + min(T,120)/120 + L/20
                              + 10 max(0, 0.15 - d_min)/0.15]"""

    def __init__(self, ctx, scenarios=("A", "B")):
        self.ctx, self.scenarios = ctx, scenarios
        self.cache, self.log = {}, []

    def controller(self, x):
        return make_controller("sugeno", self.ctx.params, EXT_SYSTEM, sv=x[0], sw=x[1], d_in=x[2],
                               name="sugeno_ga")

    def __call__(self, x, tag):
        key = tuple(np.round(x, 10))
        if key in self.cache:
            return self.cache[key]
        terms = []
        for sid in self.scenarios:
            _, m = self.ctx.run(SCENARIOS[sid], self.controller(x))
            ic, ie = m["collision"], 1 - m["success"]
            T = m["arrival_time"] if m["success"] else 120.0
            dmin = m["min_guard"]
            j = (100 * ic + 30 * ie + min(T, 120) / 120 + m["path_length"] / 20
                 + 10 * max(0.0, 0.15 - dmin) / 0.15)
            terms.append(j)
        J = float(np.mean(terms))
        self.cache[key] = J
        self.log.append({**tag, "eval": len(self.cache), **dict(zip(GENES, x)), "J": J,
                         **{f"J_{s}": t for s, t in zip(self.scenarios, terms)}})
        return J


def ga(obj, seed, pop_size=12, generations=10, p_mut=0.2):
    rng = np.random.default_rng(seed)
    lo, hi = BOUNDS[:, 0], BOUNDS[:, 1]
    pop = lo + rng.random((pop_size, 3)) * (hi - lo)
    fit = np.array([obj(x, {"algo": "GA", "seed": seed, "gen": 0}) for x in pop])
    best_curve = []
    for g in range(1, generations):
        elite = pop[np.argmin(fit)].copy()
        children = [elite]                                  # elitism
        while len(children) < pop_size:
            parents = []
            for _ in range(2):                              # tournament of size 3
                c = rng.choice(pop_size, 3, replace=False)
                parents.append(pop[c[np.argmin(fit[c])]])
            a = rng.random()                                # arithmetic crossover
            child = a * parents[0] + (1 - a) * parents[1]
            mut = rng.random(3) < p_mut                     # gaussian mutation, 10 % of range
            child = child + mut * rng.normal(0, 0.1 * (hi - lo))
            children.append(np.clip(child, lo, hi))         # projection into bounds
        pop = np.array(children)
        fit = np.array([obj(x, {"algo": "GA", "seed": seed, "gen": g}) for x in pop])
    i = int(np.argmin(fit))
    return pop[i], float(fit[i])


def random_search(obj, seed, n_evals):
    rng = np.random.default_rng(1000 + seed)
    lo, hi = BOUNDS[:, 0], BOUNDS[:, 1]
    best, bx = math.inf, None
    for _ in range(n_evals):
        x = lo + rng.random(3) * (hi - lo)
        j = obj(x, {"algo": "random", "seed": seed, "gen": -1})
        if j < best:
            best, bx = j, x
    return bx, best


def option_b(ctx):
    p = ctx.params
    print("  Option B - genetic optimisation of (s_v, s_omega, d_in), J on A and B")
    gens = 3 if ctx.quick else 10
    results, logs = [], []
    default = np.array([1.0, 1.0, 1.0])
    j_default = Objective(ctx)(default, {"algo": "default", "seed": -1, "gen": -1})
    for seed in (0, 1, 2):
        o_ga = Objective(ctx)
        x_ga, j_ga = ga(o_ga, seed, generations=gens)
        n_eff = len(o_ga.cache)                             # effective evaluations
        o_rs = Objective(ctx)
        x_rs, j_rs = random_search(o_rs, seed, n_eff)
        logs += o_ga.log + o_rs.log
        results.append({"seed": seed, "n_evals": n_eff, "ga_J": j_ga, "rs_J": j_rs,
                        **{f"ga_{g}": v for g, v in zip(GENES, x_ga)},
                        **{f"rs_{g}": v for g, v in zip(GENES, x_rs)}})
        print(f"    seed {seed}: {n_eff} evaluations | GA J = {j_ga:.4f} at "
              f"(s_v {x_ga[0]:.3f}, s_w {x_ga[1]:.3f}, d_in {x_ga[2]:.3f}) | random J = {j_rs:.4f} "
              f"| default J = {j_default:.4f}")
    write_csv(logs, ctx.results / "s6b_all_evaluations.csv")
    write_csv(results, ctx.results / "s6b_ga_vs_random.csv")
    plot_convergence(ctx, logs, j_default)

    best = min(results, key=lambda r: r["ga_J"])
    x_best = np.array([best[f"ga_{g}"] for g in GENES])
    print(f"    retained: GA seed {best['seed']} -> s_v = {x_best[0]:.3f}, s_omega = {x_best[1]:.3f}, "
          f"d_in = {x_best[2]:.3f} m (d_out = {x_best[2] + 0.4:.3f} m)")

    def build_ga():
        return make_controller("sugeno", p, EXT_SYSTEM, sv=x_best[0], sw=x_best[1],
                               d_in=x_best[2], name="sugeno_ga")

    def build_def():
        return make_controller("sugeno", p, EXT_SYSTEM)

    # validation on C, then a single evaluation on D, E, F
    val, _, _ = run_campaign(ctx, builders={f"sugeno|{EXT_SYSTEM}": build_def,
                                            f"sugeno_ga|{EXT_SYSTEM}": build_ga},
                             scenarios=("C",), label="s6b", log=False)
    write_csv(aggregate(val), ctx.results / "s6b_validation_C.csv")
    metrics, nominal, _ = run_campaign(ctx, builders={f"sugeno|{EXT_SYSTEM}": build_def,
                                                      f"sugeno_ga|{EXT_SYSTEM}": build_ga},
                                       label="s6b", log=False)
    write_summary_csv(metrics, ctx.results / "s6b_episodes.csv")
    agg = aggregate(metrics)
    write_csv(agg, ctx.results / "s6b_aggregate.csv")
    plot_closed_loop(ctx, nominal, ["sugeno", "sugeno_ga"], "s6b",
                     f"Option B - Sugeno default vs GA-tuned (s_v {x_best[0]:.2f}, s_w {x_best[1]:.2f}, "
                     f"d_in {x_best[2]:.2f} m)")
    ctx.save_json({"default_J": j_default, "retained": dict(zip(GENES, x_best)),
                   "runs": results}, "s6b_summary.json")
    return agg, aggregate(val)


def plot_convergence(ctx, logs, j_default):
    fig, ax = plt.subplots(figsize=(9, 5))
    for algo, c in (("GA", "tab:orange"), ("random", "tab:gray")):
        for seed in (0, 1, 2):
            js = [l["J"] for l in logs if l["algo"] == algo and l["seed"] == seed]
            ax.plot(np.arange(1, len(js) + 1), np.minimum.accumulate(js), color=c,
                    lw=1.8 if algo == "GA" else 1.2, ls="-" if algo == "GA" else "--",
                    label=f"{algo} seed {seed}")
    ax.axhline(j_default, color="tab:green", ls=":", label=f"default parameters (J = {j_default:.3f})")
    ax.set_xlabel("number of effective evaluations"); ax.set_ylabel("best J so far (A + B)")
    ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
    ax.set_title("Session 6 option B - GA (12 individuals, tournament 3, arithmetic crossover, "
                 "mutation 0.2) vs random search", fontsize=9)
    fig.tight_layout()
    fig.savefig(ctx.figs / "s6b_convergence.png", dpi=150)
    plt.close(fig)


# =========================================================================== #
def run(ctx):
    t0 = banner("SESSION 6 - extensions (option A: MLP, option B: genetic algorithm)")
    agg_a = option_a(ctx)
    agg_b, val_b = option_b(ctx)
    md = ["# Session 6 - extension results (generated)\n",
          f"System for both extensions: {EXT_SYSTEM}. Noisy seeds 0..{len(ctx.seeds) - 1}.\n"]
    for title, agg, names in (("Option A - MLP imitation", agg_a, ["sugeno", "mlp0", "mlp1", "mlp2"]),
                              ("Option B - validation on C", val_b, ["sugeno", "sugeno_ga"]),
                              ("Option B - test D, E, F", agg_b, ["sugeno", "sugeno_ga"])):
        md.append(f"\n## {title}\n")
        md.append("| Scenario | Method | Nominal | Success / N | Collisions / N | Mean time [s] | "
                  "Mean L [m] | Min guard [m] (worst) |")
        md.append("|---|---|---|---|---|---|---|---|")
        for sid in sorted({a["scenario"] for a in agg}):
            for n in names:
                noi = [a for a in agg if a["scenario"] == sid and a["controller"] == n and a["condition"] == "noisy"]
                nom = [a for a in agg if a["scenario"] == sid and a["controller"] == n and a["condition"] == "nominal"]
                if not noi:
                    continue
                a, b = noi[0], nom[0]
                ns = f"{b['T_mean']:.2f} s" if b["success"] else ("collision" if b["collision"] else "timeout")
                md.append(f"| {sid} | {n} | {ns} | {a['success']}/{a['N']} | {a['collision']}/{a['N']} | "
                          f"{fmt_ms(a['T_mean'], a['T_std'], a['n_used'])} | "
                          f"{fmt_ms(a['L_mean'], a['L_std'], a['n_used'])} | "
                          f"{a['min_guard_mean']:.3f} ± {a['min_guard_std']:.3f} ({a['min_guard_worst']:.3f}) |")
    text = "\n".join(md) + "\n"
    (ctx.results / "s6_tables.md").write_text(text, encoding="utf-8")
    print(text)
    done(t0)
