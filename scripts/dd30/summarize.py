#!/usr/bin/env python3
"""Summarises the JSON files that ex2/ex6 write with output.json (see examples/run.hh): one row per run with the
iteration count and the time to solution split into phases.

Phases (the times of the setup phases are wall-clock times if the runs used timing.barriers, which put a barrier
between them; they are then the maximum over the ranks and add up to the setup time):
  fine     the fine level: overlapping subdomains and the factorisation of the subdomain matrices (needed by every method)
  space    the coarse space (MS-GFEM: the local eigenproblems), split on the rank where it took longest into
             neumann   assembly of the Neumann matrices
             eig_fact  factorisation for the shift-invert solves (saddle point matrix, Neumann matrix or ring matrix)
             eig_iter  the rest of the eigensolver (Lanczos iterations: shift-invert solves, projections, ...)
             other     everything else (e.g. assembly of the saddle point matrix)
  coarse   the coarse correction: Galerkin matrix, gathered and factorised on rank 0
  solve    GMRES
Also: the setup time of the whole run (Run/setup), the coarse dimension, the size of the largest overlapping subdomain
(sub_dofs) and of the largest ring of the ring variant (ring_dofs), and the peak memory (max over the ranks).

Usage: summarize.py [--csv] <json files or directories>...
"""
import json
import sys
from pathlib import Path


def event(d, name):
    """Times of an event on all ranks, zeros if no rank has it."""
    e = d["events"].get(name)
    return e["time"] if e else [0.0] * d["ranks"]


def summarize(path):
    d = json.load(open(path))
    meta = d["meta"]
    res = meta["results"]
    cfg = meta["config"]
    P = d["ranks"]

    fine = max(event(d, "Schwarz/setup"))
    space_t = event(d, "Two-level Schwarz/coarse space")
    space = max(space_t)
    # Breakdown of the coarse space on its critical rank
    r = max(range(P), key=lambda p: space_t[p])
    neumann = event(d, "Neumann/matrix")[r]
    eig_fact = event(d, "Eigensolver/factorize")[r]
    eig_iter = event(d, "Eigensolver/solve")[r] - eig_fact
    other = space_t[r] - neumann - eig_fact - eig_iter
    coarse = max(event(d, "Coarse correction/setup"))
    solve = max(event(d, "Run/solve"))
    setup = max(event(d, "Run/setup"))
    basis = d["stats"].get("Coarse space/basis size")
    dim = int(sum(v for v in basis if v)) if basis else 0
    rss = max(v for v in d["stats"]["memory/max rss [MB]"] if v is not None)
    sub = d["stats"].get("Schwarz/subdomain dofs")
    ring = d["stats"].get("Coarse space/ring dofs")
    sub_dofs = int(max(v for v in sub if v is not None)) if sub else 0
    ring_dofs = int(max(v for v in ring if v is not None)) if ring else 0

    prec = cfg.get("prec.type", "")
    if prec == "schwarz":
        method = "ras"
    elif cfg.get("prec.coarse_space_type") == "msgfem":
        method = {"saddle_point": "saddle_point", "two_sided": "two_sided"}.get(cfg.get("prec.coarse_space.method", "saddle_point"), "?")
        if cfg.get("prec.coarse_space.inner_product") == "ring":
            method += "_ring"
    else:
        method = cfg.get("prec.coarse_space_type", "?")

    return {
        "file": Path(path).stem, "method": method, "ranks": P, "dofs": res.get("dofs", 0), "its": res["iterations"],
        "conv": res["converged"], "dim": dim, "sub_dofs": sub_dofs, "ring_dofs": ring_dofs, "setup": setup, "fine": fine, "space": space, "neumann": neumann,
        "eig_fact": eig_fact, "eig_iter": eig_iter, "other": other, "coarse": coarse, "solve": solve,
        "total": setup + solve, "rss_mb": rss,
    }


def main():
    args = sys.argv[1:]
    csv = "--csv" in args
    args = [a for a in args if a != "--csv"]
    files = []
    for a in args:
        p = Path(a)
        files += sorted(p.glob("*.json")) if p.is_dir() else [p]
    rows = [summarize(f) for f in files]
    cols = ["file", "method", "ranks", "dofs", "its", "conv", "dim", "sub_dofs", "ring_dofs", "setup", "fine", "space", "neumann", "eig_fact",
            "eig_iter", "other", "coarse", "solve", "total", "rss_mb"]
    if csv:
        print(",".join(cols))
        for row in rows:
            print(",".join(str(row[c]) for c in cols))
        return
    fmt = {"file": "{:<36}", "method": "{:<18}", "ranks": "{:>5}", "dofs": "{:>9}", "its": "{:>4}", "conv": "{!s:>5}",
           "dim": "{:>6}", "sub_dofs": "{:>8}", "ring_dofs": "{:>9}", "rss_mb": "{:>8.0f}"}
    print(" ".join((fmt.get(c, "{:>8}").replace(".3f", "").replace(".0f", "").replace("!s", "")).format(c) for c in cols))
    for row in rows:
        print(" ".join(fmt.get(c, "{:>8.3f}").format(row[c]) for c in cols))


if __name__ == "__main__":
    main()
