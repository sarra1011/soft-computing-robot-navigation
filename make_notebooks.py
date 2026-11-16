"""Builds the notebooks:

  notebooks/session2_fuzzy_engine.ipynb   handout deliverable (one complete inference)
  kaggle/soft_computing_kaggle.ipynb       self-contained: embeds every source file,
                                           runs tests + campaign, shows figures, zips outputs

    python make_notebooks.py
"""
from __future__ import annotations

import base64
import zlib
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent
SOURCES = sorted([*ROOT.glob("*.py"), *ROOT.glob("sessions/*.py"), *ROOT.glob("tests/*.py"),
                  ROOT / "README.md", ROOT / "docs" / "note_conventions.md"])


def session2():
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Session 2 - a complete fuzzy inference, step by step\n"
           "Chain: **fuzzification -> rule activation (AND = min) -> implication -> aggregation -> "
           "crisp output**. Engine: `fuzzy.py` (written from scratch, no library)."),
        code("import sys, math, numpy as np, matplotlib.pyplot as plt\nsys.path.insert(0, '..')\n"
             "from fuzzy import trap, tri, LinguisticVariable, Rule, MamdaniEngine, SugenoEngine\n"
             "from fuzzy_design import build_goal_engine"),
        md("## 1. Handout exercise: u = 0.25\n"
           "Bas = trap(0, 0, 0.2, 0.6), Haut = trap(0.2, 0.6, 1, 1).\n\n"
           "By hand: 0.25 lies on the falling edge of Bas: (0.6 - 0.25) / (0.6 - 0.2) = **0.875**; "
           "on the rising edge of Haut: (0.25 - 0.2) / (0.6 - 0.2) = **0.125**.\n\n"
           "Sugeno (constants 0.2 and 0.8): z* = (0.875 x 0.2 + 0.125 x 0.8) / (0.875 + 0.125) = **0.275**."),
        code("u = LinguisticVariable('u', 0, 1, [trap('Bas', 0, 0, 0.2, 0.6), trap('Haut', 0.2, 0.6, 1, 1)])\n"
             "y = LinguisticVariable('y', 0, 1, [tri('Faible', 0, 0.2, 0.5), tri('Fort', 0.5, 0.8, 1.0)])\n"
             "rules = [Rule(('Bas',), (('y', 'Faible'),)), Rule(('Haut',), (('y', 'Fort'),))]\n"
             "sug = SugenoEngine([u], {'y': y}, rules, {'y': {'Faible': 0.2, 'Fort': 0.8}})\n"
             "lam_bas, lam_haut = u.fuzzify(0.25)\n"
             "print('lambda_Bas =', lam_bas, ' lambda_Haut =', lam_haut)\n"
             "print('Sugeno output =', sug.infer((0.25,))['y'])\n"
             "assert abs(sug.infer((0.25,))['y'] - 0.275) < 1e-12"),
        md("## 2. Mamdani version\nConstants replaced by fuzzy sets on [0, 1]: "
           "**Faible = tri(0, 0.2, 0.5)**, **Fort = tri(0.5, 0.8, 1.0)**.\n\n"
           "Implication min(lambda_j, mu_Bj(z)), aggregation max, discrete centre of gravity "
           "z* = sum z mu(z) / sum mu(z) on a uniform grid. The result depends on the shapes and the grid."),
        code("def by_hand(n):\n"
             "    num = den = 0.0\n"
             "    for i in range(n):\n"
             "        z = i / (n - 1)\n"
             "        f = max(0, min(z / 0.2, (0.5 - z) / 0.3))\n"
             "        g = max(0, min((z - 0.5) / 0.3, (1 - z) / 0.2))\n"
             "        mu = max(min(lam_bas, f), min(lam_haut, g))\n"
             "        num += z * mu; den += mu\n"
             "    return num / den\n\n"
             "for n in (11, 101, 1001):\n"
             "    mam = MamdaniEngine([u], {'y': y}, rules, n)\n"
             "    print(f'{n:5d} points: engine = {mam.infer((0.25,))[\"y\"]:.6f}   explicit sums = {by_hand(n):.6f}')"),
        md("With 11 points the hand computation fits on paper:"),
        code("z = np.linspace(0, 1, 11)\n"
             "mam = MamdaniEngine([u], {'y': y}, rules, 11)\n"
             "agg = mam.explain((0.25,))['agg']['y']\n"
             "for zi, m in zip(z, agg): print(f'z = {zi:.1f}  mu = {m:.4f}  z*mu = {zi*m:.4f}')\n"
             "print('sum mu =', agg.sum().round(4), ' sum z*mu =', (z*agg).sum().round(4), ' z* =', ((z*agg).sum()/agg.sum()).round(6))"),
        code("ex = MamdaniEngine([u], {'y': y}, rules, 301).explain((0.25,))\n"
             "zz = ex['grid']['y']\n"
             "fig, ax = plt.subplots(figsize=(8, 3.5))\n"
             "for s, lam in zip(y.sets, (lam_bas, lam_haut)):\n"
             "    ax.plot(zz, s.eval(zz), ':', label=repr(s))\n"
             "    ax.plot(zz, np.minimum(lam, s.eval(zz)), label=f'min({lam:.3f}, mu)')\n"
             "ax.fill_between(zz, ex['agg']['y'], alpha=.3, label='aggregation')\n"
             "ax.axvline(ex['out']['y'], color='k', label=f\"centroid {ex['out']['y']:.4f}\")\n"
             "ax.axvline(0.275, color='g', ls='--', label='Sugeno 0.275'); ax.legend(fontsize=7); plt.show()"),
        md("## 3. A full inference of the navigation controller\n"
           "rho = 2.0 m, alpha = 0.9 rad (goal on the left)."),
        code("for kind in ('mamdani', 'sugeno'):\n"
             "    e = build_goal_engine(kind)\n"
             "    ex = e.explain((2.0, 0.9))\n"
             "    print(kind.upper())\n"
             "    for var, mu in zip(e.inputs, ex['mus']):\n"
             "        print('  ', var.name, dict(zip(var.names, np.round(mu, 3))))\n"
             "    for r, lam in zip(e.rules, ex['lambda']):\n"
             "        if lam > 0: print(f'   lambda = {lam:.3f}  ', r.text(e.inputs, e.outputs))\n"
             "    print('   output:', {k: round(v, 4) for k, v in ex['out'].items()})"),
        md("## 4. Policy when no rule fires\n"
           "The partitions cover their whole domain (coverage > 0), so this cannot happen with the "
           "handout sets. The engine still handles it: `infer` returns `None`, increments "
           "`n_inactive` (diagnostic), and the controller applies **(v, omega) = (0, 0)**. "
           "Out-of-domain inputs are saturated to the domain and counted (`n_out_of_domain`)."),
        code("from fuzzy_design import goal_inputs, motion_outputs, avoid_inputs\n"
             "for v in (*goal_inputs(), *motion_outputs(), *avoid_inputs()):\n"
             "    print(f'{v.name:8s} coverage = {v.coverage():.3f}')\n"
             "rho, _ = goal_inputs(); rho.fuzzify(-2); rho.fuzzify(30)\n"
             "print('out-of-domain events counted:', rho.n_out_of_domain)"),
    ]
    out = ROOT / "notebooks"
    out.mkdir(exist_ok=True)
    nbf.write(nb, out / "session2_fuzzy_engine.ipynb")
    return out / "session2_fuzzy_engine.ipynb"


def kaggle():
    files = {str(p.relative_to(ROOT)).replace("\\", "/"): p.read_text(encoding="utf-8") for p in SOURCES}
    blob = base64.b64encode(zlib.compress(repr(files).encode("utf-8"), 9)).decode()
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.cells = [
        md("# Soft computing - fuzzy navigation of a mobile robot (full TP, sessions 1-6)\n"
           "Run all cells (**Run All**). The first cell writes the project to `/kaggle/working/soft-computing-nav`; "
           "outputs (figures, CSV logs, results, `viewer.html`) are zipped at the end for download.\n\n"
           "Internet is only needed for the optional `scikit-fuzzy` cross-check."),
        code("%pip install -q scikit-fuzzy  # optional: cross-check of the hand-written engine"),
        code("# --- write every source file of the project ---\n"
             "import os, zlib, base64, ast\n"
             "ROOT = '/kaggle/working/soft-computing-nav' if os.path.isdir('/kaggle/working') else os.path.abspath('soft-computing-nav')\n"
             f"BLOB = '{blob}'\n"
             "files = ast.literal_eval(zlib.decompress(base64.b64decode(BLOB)).decode('utf-8'))\n"
             "for rel, text in files.items():\n"
             "    path = os.path.join(ROOT, rel)\n"
             "    os.makedirs(os.path.dirname(path), exist_ok=True)\n"
             "    with open(path, 'w', encoding='utf-8') as f: f.write(text)\n"
             "os.chdir(ROOT)\n"
             "print(len(files), 'files written to', ROOT)"),
        code("!python env_info.py   # machine description for the report (Kaggle CPU)"),
        code("!python -m unittest discover -s tests -v"),
        code("!python scenarios.py"),
        code("# full campaign (about 5-8 min on a Kaggle CPU). Add --quick for a 2-min smoke test.\n"
             "!python run_experiments.py"),
        code("from IPython.display import Markdown, display\n"
             "display(Markdown(open('results/RESULTS.md', encoding='utf-8').read()))"),
        code("from IPython.display import Image, display\n"
             "import glob\n"
             "for f in sorted(glob.glob('figures/*.png')) + sorted(glob.glob('docs/*.png')):\n"
             "    print(f); display(Image(f, width=1000))"),
        code("display(Image('figures/demo_D_sys-v1.gif')); display(Image('figures/demo_D_sys-v2.gif'))"),
        code("# --- download everything: Output panel (right) -> soft_computing_outputs.zip ---\n"
             "# the zip is written NEXT TO the project folder, never inside it\n"
             "import shutil\n"
             "PARENT = os.path.dirname(ROOT)\n"
             "zip_path = shutil.make_archive(os.path.join(PARENT, 'soft_computing_outputs'), 'zip',\n"
             "                               PARENT, os.path.basename(ROOT))\n"
             "print(zip_path, round(os.path.getsize(zip_path) / 1e6, 1), 'MB')\n"
             "from IPython.display import FileLink\n"
             "FileLink(os.path.relpath(zip_path))"),
    ]
    out = ROOT / "kaggle"
    out.mkdir(exist_ok=True)
    nbf.write(nb, out / "soft_computing_kaggle.ipynb")
    return out / "soft_computing_kaggle.ipynb"


if __name__ == "__main__":
    print("wrote", session2())
    print("wrote", kaggle())
