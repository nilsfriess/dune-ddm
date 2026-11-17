#!/usr/bin/env python3
"""Compares the same runs in different out_dirs, e.g. a cluster or toolchain against the Helix runs: per run the
iterations, the coarse dimension and the time of each phase (see summarize.py) relative to the reference.

Usage: compare_runs.py <reference dirs, comma separated> <dir>...   (runs are matched by the name of their JSON file)
  e.g. compare_runs.py raw/weak3d_rep_P8,raw/ringmethod_weak3d_P8 raw/kit_check_P8_gcc14

Flags what would be unexpected: iterations that differ by more than 1, coarse dimensions that differ by more than
2 %, and a phase whose time ratio to the reference is more than 25 % away from the median ratio of that phase in the
same directory (a different CPU or compiler should speed up all runs of a phase by roughly the same factor). Phases
shorter than 1 s in the reference are not flagged (noise).
"""
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize import summarize  # noqa: E402

PHASES = ["fine", "space", "coarse", "solve", "total"]


def load(dirs):
    return {f.stem: summarize(f) for d in dirs.split(",") for f in sorted(Path(d).glob("*.json"))}


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    ref_dir, dirs = sys.argv[1], sys.argv[2:]
    ref = load(ref_dir)
    for d in dirs:
        runs = load(d)
        common = [n for n in runs if n in ref]
        print(f"\n=== {d}  vs  {ref_dir}: {len(common)} common runs"
              + (f", only here: {sorted(set(runs) - set(ref))}" if set(runs) - set(ref) else ""))
        ratios = {p: [runs[n][p] / ref[n][p] for n in common if ref[n][p] > 1.0] for p in PHASES}
        median = {p: statistics.median(v) if v else float("nan") for p, v in ratios.items()}
        print("median time ratio (here / reference): " + ", ".join(f"{p} {median[p]:.2f}" for p in PHASES))
        print(f"{'run':<38} {'its':>9} {'dim':>13} " + " ".join(f"{p:>14}" for p in PHASES))
        for n in common:
            a, b = ref[n], runs[n]
            flags = []
            if abs(a["its"] - b["its"]) > 1:
                flags.append("its")
            if a["dim"] and abs(a["dim"] - b["dim"]) > 0.02 * a["dim"]:
                flags.append("dim")
            cells = []
            for p in PHASES:
                r = b[p] / a[p] if a[p] > 0 else float("nan")
                if a[p] > 1.0 and abs(r / median[p] - 1) > 0.25:
                    flags.append(p)
                cells.append(f"{b[p]:6.1f} ({r:4.2f})" if a[p] > 0 else f"{b[p]:6.1f}       ")
            print(f"{n:<38} {a['its']:>4}/{b['its']:<4} {a['dim']:>6}/{b['dim']:<6} " + " ".join(f"{c:>14}" for c in cells)
                  + ("   <-- " + ", ".join(flags) if flags else ""))


if __name__ == "__main__":
    main()
