"""Rebuilds the whole campaign with no manual input:

    python run_experiments.py              # all sessions, full protocol (~ a few minutes)
    python run_experiments.py --quick      # 5 seeds, short trainings: smoke test
    python run_experiments.py --sessions 1 2 3

Outputs: configs/ logs/ figures/ results/ (+ viewer.html, figures/demo_D.gif)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")

from sessions import s1, s2, s3, s4, s5, s6  # noqa: E402
from sessions.common import Context        # noqa: E402

SESSIONS = {1: s1.run, 2: s2.run, 3: s3.run, 4: s4.run, 5: s5.run, 6: s6.run}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", type=int, nargs="*", default=list(SESSIONS))
    ap.add_argument("--quick", action="store_true", help="5 seeds, short trainings (smoke test)")
    ap.add_argument("--no-extras", action="store_true", help="skip the GIF and the HTML viewer")
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    ctx = Context(quick=args.quick)
    import env_info
    env_info.main()
    for s in args.sessions:
        SESSIONS[s](ctx)
    if not args.no_extras and 5 in args.sessions:
        import extras
        extras.run(ctx)
    import report_tables
    report_tables.main(ctx)
    print(f"\nTotal: {time.perf_counter() - t0:.1f} s. See results/RESULTS.md and viewer.html")


if __name__ == "__main__":
    main()
