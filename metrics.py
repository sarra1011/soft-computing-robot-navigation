"""Episode metrics (section 7) and aggregation over the noisy runs."""
from __future__ import annotations

import csv
import gzip
import math

import numpy as np

from model import Params
from simulator import EpisodeResult

NAN = float("nan")

SUMMARY_FIELDS = [
    "scenario", "controller", "version", "condition", "seed", "status",
    "success", "collision", "timeout", "arrival_time", "path_length", "min_guard",
    "min_wall", "sum_dv", "sum_domega", "t_compute_median", "t_compute_p95",
    "n_decisions", "n_preventive_stops", "n_avoid_steps", "n_mode_switches",
    "blocked", "t_blocked", "n_no_rule",
]


def episode_metrics(res: EpisodeResult, p: Params) -> dict:
    rows = res.rows
    xy = np.array([[r["x"], r["y"]] for r in rows])
    length = float(np.sum(np.hypot(*np.diff(xy, axis=0).T))) if len(xy) > 1 else 0.0
    cmd = [r for r in rows if r["status"] == "running"]
    v = np.array([r["v"] for r in cmd])
    w = np.array([r["omega"] for r in cmd])
    tc = np.array([r["t_compute"] for r in cmd])
    modes = [r["mode"] for r in cmd]
    nav = [m for m in modes if m in ("goal", "avoid")]
    success = res.status == "arrived"
    return {
        "scenario": res.scenario_id, "controller": res.controller,
        "version": res.version, "condition": res.condition, "seed": res.seed,
        "status": res.status,
        "success": int(success),
        "collision": int(res.status == "collision"),
        "timeout": int(res.status == "timeout"),
        "arrival_time": round(rows[-1]["k"] * p.dt, 6) if success else NAN,
        "path_length": length,
        "min_guard": res.min_guard,
        "min_wall": res.min_wall,
        "sum_dv": float(np.sum(np.abs(np.diff(v)))) if len(v) > 1 else 0.0,
        "sum_domega": float(np.sum(np.abs(np.diff(w)))) if len(w) > 1 else 0.0,
        "t_compute_median": float(np.median(tc)) if len(tc) else NAN,
        "t_compute_p95": float(np.percentile(tc, 95)) if len(tc) else NAN,
        "n_decisions": len(cmd),
        "n_preventive_stops": modes.count("stop"),
        "n_avoid_steps": modes.count("avoid"),
        "n_mode_switches": sum(1 for a, b in zip(nav, nav[1:]) if a != b),
        "blocked": int(res.blocked),
        "t_blocked": res.t_blocked,
        "n_no_rule": sum(1 for r in cmd if r.get("flags") == "no_rule"),
    }


def _fmt(x):
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return str(x)
        return f"{x:.6g}"
    return x


def write_log_csv(res: EpisodeResult, path) -> None:
    """Writes .csv, or gzip-compressed CSV if the path ends with .gz."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(res.rows[0].keys()))
        wr.writeheader()
        for r in res.rows:
            wr.writerow({k: _fmt(v) for k, v in r.items()})


def write_csv(rows, path, fields=None) -> None:
    fields = fields or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow({k: _fmt(r.get(k, "")) for k in fields})


def write_summary_csv(metrics_list, path) -> None:
    write_csv(metrics_list, path, SUMMARY_FIELDS)


def _ms(vals):
    vals = [v for v in vals if not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))]
    if not vals:
        return NAN, NAN, 0
    a = np.asarray(vals, float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0, len(a)


def aggregate(metrics_list) -> list[dict]:
    """One line per (scenario, controller, condition). Time and length are
    averaged over SUCCESSFUL runs only and the number of successes is reported."""
    groups = {}
    for m in metrics_list:
        groups.setdefault((m["scenario"], m["controller"], m["version"], m["condition"]),
                          []).append(m)
    out = []
    for (sc, ctrl, ver, cond), ms in groups.items():
        ok = [m for m in ms if m["success"]]
        T = _ms([m["arrival_time"] for m in ok])
        L = _ms([m["path_length"] for m in ok])
        G = _ms([m["min_guard"] for m in ms])
        tc = np.array([m["t_compute_median"] for m in ms], float)
        tp = np.array([m["t_compute_p95"] for m in ms], float)
        out.append({
            "scenario": sc, "controller": ctrl, "system": ver, "condition": cond, "N": len(ms),
            "success": len(ok), "collision": sum(m["collision"] for m in ms),
            "timeout": sum(m["timeout"] for m in ms),
            "blocked": sum(m["blocked"] for m in ms),
            "T_mean": T[0], "T_std": T[1], "L_mean": L[0], "L_std": L[1],
            "n_used": T[2], "min_guard_mean": G[0], "min_guard_std": G[1],
            "min_guard_worst": float(np.min([m["min_guard"] for m in ms])),
            "sum_dv_mean": float(np.mean([m["sum_dv"] for m in ms])),
            "sum_domega_mean": float(np.mean([m["sum_domega"] for m in ms])),
            "t_compute_median_us": float(np.nanmedian(tc)) * 1e6,
            "t_compute_p95_us": float(np.nanmedian(tp)) * 1e6,
            "preventive_stops": int(sum(m["n_preventive_stops"] for m in ms)),
            "mode_switches_mean": float(np.mean([m["n_mode_switches"] for m in ms])),
        })
    return out
