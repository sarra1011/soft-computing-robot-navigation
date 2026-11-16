"""Shared context for the session runners."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from metrics import episode_metrics, write_log_csv
from model import Params
from sensors import make_noise
from simulator import simulate

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Context:
    params: Params = field(default_factory=Params)
    quick: bool = False                       # fewer seeds / generations / epochs
    root: Path = ROOT

    def __post_init__(self):
        for d in (self.configs, self.logs, self.figs, self.results):
            d.mkdir(parents=True, exist_ok=True)
        self._noise = {}

    @property
    def configs(self): return self.root / "configs"
    @property
    def logs(self): return self.root / "logs"
    @property
    def figs(self): return self.root / "figures"
    @property
    def results(self): return self.root / "results"

    @property
    def seeds(self):
        return list(range(5 if self.quick else 20))   # handout: seeds 0..19

    def noise(self, seed: int):
        if seed not in self._noise:
            self._noise[seed] = make_noise(seed, self.params.n_steps)
        return self._noise[seed]

    def run(self, scenario, controller, seed=None, log_dir=None, params=None):
        """One episode (nominal if seed is None); optional gzip CSV log."""
        p = params or self.params
        noise = None if seed is None else self.noise(seed)
        res = simulate(scenario, controller, p, noise, "" if seed is None else seed)
        if log_dir is not None:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            tag = "nominal" if seed is None else f"seed{seed:02d}"
            write_log_csv(res, log_dir / f"{scenario.id}_{res.controller}_{res.version}_{tag}.csv.gz")
        return res, episode_metrics(res, p)

    def save_json(self, obj, name):
        with open(self.results / name, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=float)


def banner(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    return time.perf_counter()


def done(t0: float):
    print(f"-> done in {time.perf_counter() - t0:.1f} s")


def fmt_ms(mean, std, n=None, digits=2):
    import math
    if n == 0 or (isinstance(mean, float) and math.isnan(mean)):
        return "- (0)" if n is not None else "-"
    s = f"{mean:.{digits}f} ± {std:.{digits}f}"
    return s + (f" ({n})" if n is not None else "")
