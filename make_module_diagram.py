"""Draws the module diagram required in session 1 -> docs/module_diagram.png

    python make_module_diagram.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

W, H = 0.27, 0.15
# key: (x_centre, y_centre, width, height, text, colour)
BOXES = {
    "run": (0.50, 0.88, W, H, "run_experiments.py\ncampaign, figures, CSV", "#cfe2f3"),
    "out": (0.86, 0.88, 0.24, H, "OUTPUTS\nconfigs/  logs/\nresults/  figures/", "#eeeeee"),
    "scn": (0.16, 0.60, W, H, "scenarios.py\nmaps A-F, JSON configs", "#fff2cc"),
    "sim": (0.50, 0.60, W, H, "simulator.py\narrival > observe > command >\nsaturate > step > collision > log", "#d9ead3"),
    "met": (0.84, 0.60, W, H, "metrics.py\nsuccess, T, L, guard,\ncompute time, CSV", "#fff2cc"),
    "sen": (0.33, 0.33, W, H, "sensors.py\nobservation (noisy pose),\nnoise pre-generated per seed", "#fce5cd"),
    "ctl": (0.67, 0.33, W, H, "controllers.py\nClassic | Mamdani | Sugeno\ncompute(obs) -> (v, omega)", "#f4cccc"),
    "mod": (0.50, 0.09, 0.92, 0.10, "model.py  -  shared by all modules:  Params, wrap, kinematics, "
            "wheel speeds, collision geometry", "#d9d2e9"),
}
ARROWS = [("run", "bottom", "scn", "top"), ("run", "bottom", "sim", "top"),
          ("run", "bottom", "met", "top"), ("run", "right", "out", "left"),
          ("sim", "bottom", "sen", "top"), ("sim", "bottom", "ctl", "top")]


def anchor(key, side):
    x, y, w, h, *_ = BOXES[key]
    return {"top": (x, y + h / 2), "bottom": (x, y - h / 2),
            "left": (x - w / 2, y), "right": (x + w / 2, y)}[side]


def main():
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for x, y, w, h, txt, col in BOXES.values():
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0.005", fc=col, ec="k"))
        ax.text(x, y, txt, ha="center", va="center", fontsize=9)
    for a, sa, b, sb in ARROWS:
        ax.annotate("", xy=anchor(b, sb), xytext=anchor(a, sa),
                    arrowprops=dict(arrowstyle="->", lw=1.3))
    ax.text(0.5, 0.20, "same interface for every controller:   observation  ->  (v, omega)",
            ha="center", fontsize=10, style="italic", color="tab:red")
    ax.set_title("Software architecture - Soft Computing mini-project (modules and data flow)")
    out = Path(__file__).resolve().parent / "docs"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "module_diagram.png", dpi=150, bbox_inches="tight")
    print("Saved", out / "module_diagram.png")


if __name__ == "__main__":
    main()
