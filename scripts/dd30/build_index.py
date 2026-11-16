#!/usr/bin/env python3
"""Builds an overview of the DD30 results: <results>/index.html (open it in a browser) and <results>/runs.csv (one row
per run of every experiment, for notebooks and the plotting scripts).

<results>/raw/<experiment>/ are copies of the out_dirs of submit_helix.sh (see sync_results.sh). catalog.json (next to
this script) says what each experiment is for, whether it is current or superseded, and defines series: the rows of
several experiments that together make one figure (e.g. a weak scaling), with one canonical run per point.

Flags are derived from the stored config of each run, not from the catalog:
  shift-1e-3  (or another value) a variant with the projection outside of the shifted solve (two_sided, projected,
              ring) with a shift below -1e-8: its eigenproblem is perturbed, by up to 5x more GMRES iterations at -1e-3
              and still 4x for the ring on the channels at -1e-6 (shift_P64); -1e-9 is the setting since ebe6cb1
  not-conv    GMRES did not converge
Failed runs (no JSON) are taken from the job logs: timeout (exit 124, mostly a hang), killed (137), OOM, error.

Usage: build_index.py [results dir]   (default ~/Dokumente/PhD/Conferences/Bogota/Talk/results)
"""
import csv
import html
import json
import re
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from summarize import summarize  # noqa: E402

METHOD_ORDER = ["RAS", "Nicolaides", "GenEO", "saddle point", "new η=ω", "new η=ring", "projected", "basis", "msgfem_ring", "geneo_ring", "?"]
NUM_COLS = ["setup", "fine", "space", "neumann", "eig_fact", "eig_iter", "other", "coarse", "solve", "total"]
CSV_COLS = ["experiment", "run", "status", "method", "ranks", "dofs", "cells", "refine", "coefficient", "contrast", "velocity",
            "nev", "threshold", "ncv", "nev_max", "tolerance", "blocksize", "shift", "mtype", "its", "conv", "dim", "sub_dofs",
            "ring_dofs"] + NUM_COLS + ["rss_mb", "flags", "reason", "commit"]


def method_label(cfg):
    if cfg.get("prec.type") == "schwarz":
        return "RAS"
    t = cfg.get("prec.coarse_space_type", "nicolaides")
    if t == "nicolaides":
        return "Nicolaides"
    if t == "geneo":
        return "GenEO"
    if t == "msgfem":
        m = cfg.get("prec.coarse_space.method", "saddle_point")
        if m == "saddle_point":
            return "saddle point"
        if m == "two_sided":
            return "new η=ring" if cfg.get("prec.coarse_space.inner_product") == "ring" else "new η=ω"
        return m
    return t


def num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def contrast(cfg):
    if cfg.get("problem.coefficient") == "file":
        m = re.search(r"_k(1e\d+)", cfg.get("problem.kappa_file", ""))
        return m.group(1) if m else "1e6"  # the default file has channels with kappa 1e6
    if cfg.get("problem.coefficient") in ("channels",):
        return cfg.get("problem.kappa_channel", "")
    if cfg.get("problem.coefficient") in ("skyscrapers", "random"):
        return cfg.get("problem.kappa_high", "")
    return cfg.get("problem.kappa_light", "")


def params(cfg, ranks):
    es = "prec.coarse_space.eigensolver."
    method = method_label(cfg)
    if method == "new η=ring":
        mtype = cfg.get("prec.coarse_space.ring_prec.mtype", "")
    elif method in ("RAS", "Nicolaides"):
        mtype = ""
    else:
        mtype = cfg.get(es + "prec.mtype", "11")
    shift = cfg.get(es + "shift", "-1e-3") if method not in ("RAS", "Nicolaides") else ""
    cells = cfg.get("grid.cells", "").split()
    return {
        "method": method, "ranks": ranks, "cells": int(cells[0]) if cells else "", "refine": cfg.get("grid.refine", ""),
        "coefficient": cfg.get("problem.coefficient", "checkerboard"), "contrast": contrast(cfg),
        "velocity": cfg.get("problem.velocity_scale", ""),
        "nev": cfg.get(es + "nev", "") if method not in ("RAS", "Nicolaides") else "",
        "threshold": cfg.get(es + "threshold", ""), "ncv": cfg.get(es + "ncv", ""), "nev_max": cfg.get(es + "nev_max", ""),
        "tolerance": cfg.get(es + "tolerance", "") if method not in ("RAS", "Nicolaides") else "",
        "blocksize": cfg.get(es + "blocksize", "") if method not in ("RAS", "Nicolaides") else "",
        "shift": shift, "mtype": mtype,
    }


def flags(row):
    f = []
    s = num(row.get("shift"))
    if row["method"] in ("new η=ω", "new η=ring", "projected") and s is not None and s < -1e-8:
        f.append(f"shift{row['shift']}")
    if row.get("status") == "ok" and row.get("conv") is False:
        f.append("not-conv")
    return f


def job_outcomes(exp_dir):
    """name -> (exit code, seconds) from the job logs (the last one wins), for runs that left no JSON."""
    out = {}
    for job in sorted(exp_dir.glob("job_*.out")):
        for line in job.read_text(errors="replace").splitlines():
            m = re.match(r"^(\S+): exit (\d+), (?:no|\d+) iterations, (\d+) s$", line)
            if m:
                out[m.group(1)] = (int(m.group(2)), int(m.group(3)))
    return out


def failure_reason(exp_dir, name, code):
    log = exp_dir / f"{name}.log"
    text = log.read_text(errors="replace") if log.exists() else ""
    errors = sorted({re.sub(r"^rank \d+: ", "", l) for l in text.splitlines() if re.match(r"^rank \d+: ", l)})
    if "oom_kill" in text or "Out Of Memory" in text:
        return "OOM", "out of memory (killed by Slurm)"
    if errors:
        return "error", "; ".join(errors)[:300]
    if code == 124:
        return "timeout", "hit the time limit of a single run (mostly a hang, see the notes)"
    if code == 137:
        return "killed", "killed (cancelled by hand or by the job's time limit)"
    return "error", f"exit {code}"


def runs_file(exp_dir):
    files = sorted(exp_dir.glob("runs_*.txt"))
    return files[-1] if files else None


def planned_runs(exp_dir):
    rf = runs_file(exp_dir)
    if not rf:
        return [], ""
    names, header, in_header = [], [], True
    for line in rf.read_text(errors="replace").splitlines():
        if line.startswith("#"):
            if in_header:
                header.append(line[1:].strip())
            continue
        in_header = False
        if line.strip():
            names.append(line.split()[0])
    return names, " ".join(header)


def load_experiment(exp_dir):
    commit = (exp_dir / "commit").read_text(errors="replace").splitlines() if (exp_dir / "commit").exists() else []
    planned, purpose = planned_runs(exp_dir)
    outcomes = job_outcomes(exp_dir)
    rows, seen = [], set()
    for js in sorted(exp_dir.glob("*.json")):
        try:
            d = json.load(open(js))
            s = summarize(js)
        except Exception as e:  # a run still being written, or an old format
            rows.append({"experiment": exp_dir.name, "run": js.stem, "status": "unreadable", "method": "?", "reason": str(e)[:200]})
            seen.add(js.stem)
            continue
        cfg = d["meta"]["config"]
        row = {"experiment": exp_dir.name, "run": js.stem, "status": "ok", **params(cfg, d["ranks"])}
        for k in ["dofs", "its", "conv", "dim", "sub_dofs", "ring_dofs", "rss_mb"] + NUM_COLS:
            row[k] = s.get(k)
        rows.append(row)
        seen.add(js.stem)
    for name in planned:
        if name in seen:
            continue
        if name in outcomes:
            code, sec = outcomes[name]
            status, reason = failure_reason(exp_dir, name, code)
            rows.append({"experiment": exp_dir.name, "run": name, "status": status, "method": guess_method(name), "reason": f"{reason} ({sec} s)"})
        else:
            rows.append({"experiment": exp_dir.name, "run": name, "status": "not run", "method": guess_method(name), "reason": "not run (yet)"})
    for r in rows:
        r["commit"] = commit[0][:10] if commit else ""
        r["flags"] = " ".join(flags(r))
    notes = []
    for extra in sorted(exp_dir.iterdir()):
        if extra.is_file() and extra.name.upper().startswith("NOTE"):
            notes.append(extra.read_text(errors="replace").strip())
    notes += [l for l in commit[1:] if l.strip()]
    return {"name": exp_dir.name, "rows": rows, "purpose": purpose, "notes": notes, "runs_file": runs_file(exp_dir),
            "jobs": sorted(exp_dir.glob("job_*.out")), "commit": commit[0][:10] if commit else ""}


def guess_method(name):
    for prefix, label in [("ras", "RAS"), ("nicolaides", "Nicolaides"), ("geneo", "GenEO"), ("msgfem_saddle_point", "saddle point"),
                          ("msgfem_ring", "new η=ring"), ("msgfem_projected", "projected"), ("msgfem", "new η=ω")]:
        if name.startswith(prefix):
            return label
    return "?"


def x_value(row, x):
    if x == "contrast":
        return row.get("contrast")
    return row.get(x)


def series_table(series, exps):
    """Pivot method x parameter: the first experiment in series['dirs'] that has an ok run for a point wins."""
    name_re = re.compile(series.get("name_re", ".*"))
    flt = series.get("filter", {})
    points, xs = {}, set()
    for exp_name in series["dirs"]:
        exp = exps.get(exp_name)
        if not exp:
            continue
        for r in exp["rows"]:
            if not name_re.search(r["run"]) or any(str(r.get(k)) != str(v) for k, v in flt.items()):
                continue
            if r["status"] != "ok":
                key = (r["method"], None)
                continue
            key = (r["method"], x_value(r, series["x"]))
            xs.add(key[1])
            if key not in points:
                points[key] = r
    xs = sorted(xs, key=lambda v: (num(v) is None, num(v) if num(v) is not None else str(v)))
    methods = sorted({m for m, _ in points}, key=lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99)
    return methods, xs, points


def fmt(v, digits=1):
    if v is None or v == "":
        return ""
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def cell(r):
    if not r:
        return "<td class='empty'>–</td>"
    tip = (f"{r['experiment']}/{r['run']}: setup {fmt(r.get('setup'))} s (fine {fmt(r.get('fine'))}, space {fmt(r.get('space'))}, "
           f"coarse {fmt(r.get('coarse'))}), solve {fmt(r.get('solve'))} s, coarse dim {r.get('dim')}, peak {fmt(r.get('rss_mb'), 0)} MB, "
           f"mtype {r.get('mtype')}, shift {r.get('shift')}")
    fl = f" <span class='flag'>{html.escape(r['flags'])}</span>" if r.get("flags") else ""
    return (f"<td title='{html.escape(tip)}'><a href='raw/{r['experiment']}/{r['run']}.json'><b>{r.get('its')}</b> it · "
            f"{fmt(r.get('total'))} s</a>{fl}</td>")


STATUS_CLASS = {"current": "ok", "partly superseded": "old", "superseded": "old", "abandoned": "bad", "check": "info", "pilot": "old", "smoke": "info", "pending": "info"}


def badge(status):
    return f"<span class='badge {STATUS_CLASS.get(status, 'info')}'>{html.escape(status)}</span>"


def run_table(exp):
    cols = ["run", "status", "method", "ranks", "cells", "contrast", "velocity", "nev", "threshold", "tolerance", "shift", "mtype", "its",
            "dim", "fine", "space", "eig_fact", "eig_iter", "coarse", "solve", "total", "rss_mb", "flags"]
    h = ["<table class='runs sortable'><thead><tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr></thead><tbody>"]
    for r in exp["rows"]:
        tds = []
        for c in cols:
            v = r.get(c, "")
            if c == "run":
                links = [f"<a href='raw/{exp['name']}/{r['run']}.{ext}'>{ext}</a>" for ext in ("json", "log")
                         if (RESULTS / "raw" / exp["name"] / f"{r['run']}.{ext}").exists()]
                tds.append(f"<td class='name'>{html.escape(r['run'])} <small>{' '.join(links)}</small></td>")
            elif c == "status":
                cls = "ok" if v == "ok" else ("info" if v == "not run" else "bad")
                tip = html.escape(r.get("reason", "") or "")
                tds.append(f"<td><span class='badge {cls}' title='{tip}'>{html.escape(str(v))}</span></td>")
            elif c == "flags":
                tds.append(f"<td><span class='flag'>{html.escape(str(v))}</span></td>")
            else:
                tds.append(f"<td>{html.escape(fmt(v, 0 if c == 'rss_mb' else 1))}</td>")
        h.append("<tr>" + "".join(tds) + "</tr>")
    h.append("</tbody></table>")
    failed = [r for r in exp["rows"] if r["status"] not in ("ok", "not run")]
    if failed:
        h.append("<ul class='failed'>" + "".join(f"<li><b>{html.escape(r['run'])}</b>: {html.escape(r['status'])}, {html.escape(r.get('reason', ''))}</li>" for r in failed) + "</ul>")
    return "\n".join(h)


CSS = """
:root { --bg:#fcfbf8; --fg:#1f1d1a; --muted:#6b665e; --line:#e4e0d8; --card:#fff; --accent:#C50D29;
        --ok:#2f7d32; --old:#8a7f6a; --bad:#b3261e; --info:#3b5f8a; --flag:#9a5b00; }
@media (prefers-color-scheme: dark) { :root { --bg:#161513; --fg:#ece9e3; --muted:#a39d92; --line:#34312c; --card:#1e1c19;
        --accent:#ff6b7f; --ok:#7cc47f; --old:#a89f8c; --bad:#ff8a80; --info:#8fb3e0; --flag:#f0b35a; } }
* { box-sizing:border-box; }
body { margin:0; padding:24px 16px 64px; background:var(--bg); color:var(--fg); font:14px/1.45 system-ui, sans-serif; }
main { max-width:1400px; margin:0 auto; }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:18px; margin:32px 0 8px; border-bottom:2px solid var(--accent); padding-bottom:4px; }
h3 { font-size:15px; margin:20px 0 6px; }
p, li { color:var(--fg); } .muted, small { color:var(--muted); }
a { color:var(--info); text-decoration:none; } a:hover { text-decoration:underline; }
details { background:var(--card); border:1px solid var(--line); border-radius:6px; margin:6px 0; padding:6px 10px; }
summary { cursor:pointer; } summary b { font-family:ui-monospace, monospace; }
.badge { display:inline-block; padding:0 6px; border-radius:9px; font-size:12px; border:1px solid currentColor; }
.badge.ok { color:var(--ok); } .badge.old { color:var(--old); } .badge.bad { color:var(--bad); } .badge.info { color:var(--info); }
.flag { color:var(--flag); font-size:12px; }
.scroll { overflow-x:auto; }
table { border-collapse:collapse; margin:6px 0; font-size:13px; }
th, td { border-bottom:1px solid var(--line); padding:3px 8px; text-align:right; white-space:nowrap; }
th { color:var(--muted); font-weight:600; cursor:pointer; position:sticky; top:0; background:var(--card); }
td.name, th:first-child, td:first-child { text-align:left; } td.empty { color:var(--muted); text-align:center; }
.series td:first-child { font-weight:600; }
ul.failed { color:var(--bad); font-size:13px; margin:4px 0 8px; }
.purpose { margin:6px 0; } .notes { white-space:pre-wrap; font-family:ui-monospace, monospace; font-size:12px; color:var(--muted); }
"""

JS = """
document.querySelectorAll('table.sortable th').forEach((th, i) => th.addEventListener('click', () => {
  const tb = th.closest('table').tBodies[0], rows = [...tb.rows], dir = th.dataset.dir = th.dataset.dir === 'a' ? 'd' : 'a';
  const val = r => { const t = r.cells[i].innerText.trim(), n = parseFloat(t); return isNaN(n) ? t : n; };
  rows.sort((a, b) => { const x = val(a), y = val(b); const c = (typeof x === typeof y) ? (x > y) - (x < y) : (typeof x === 'number' ? -1 : 1); return dir === 'a' ? c : -c; });
  rows.forEach(r => tb.appendChild(r));
}));
"""


def build(results):
    global RESULTS
    RESULTS = results
    catalog = json.load(open(HERE / "catalog.json"))
    exps = {}
    for d in sorted((results / "raw").iterdir()):
        if d.is_dir() and (runs_file(d) or list(d.glob("*.json"))):
            exps[d.name] = load_experiment(d)

    # runs.csv
    with open(results / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for e in exps.values():
            for r in e["rows"]:
                w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in CSV_COLS})

    cat_exps = catalog["experiments"]
    n_ok = sum(r["status"] == "ok" for e in exps.values() for r in e["rows"])
    n_fail = sum(r["status"] not in ("ok", "not run") for e in exps.values() for r in e["rows"])
    n_todo = sum(r["status"] == "not run" for e in exps.values() for r in e["rows"])
    out = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
           f"<title>DD30 results</title><style>{CSS}</style></head><body><main>",
           "<h1>DD30 results</h1>",
           f"<p class='muted'>Built {datetime.now():%Y-%m-%d %H:%M} from {len(exps)} experiments in <code>raw/</code>: {n_ok} runs ok, "
           f"{n_fail} failed, {n_todo} not run (yet). All runs in one table: <a href='runs.csv'>runs.csv</a>; quick plots: <code>explore.ipynb</code> "
           f"(VS Code with the kernel Python (dd30), or <code>~/.venvs/dd30/bin/jupyter lab</code> in this folder). "
           f"Rebuild: <code>scripts/dd30/sync_results.sh</code> in dune-ddm.</p>",
           "<p class='muted'>Cells: GMRES iterations · time to solution (setup + solve); hover for the breakdown, click for the JSON. "
           "Flags: <span class='flag'>shift-1e-3</span> (or another value) eigenproblem perturbed by the shift: two-sided/projected/ring "
           "with a shift below −1e-8 (−1e-9 since ebe6cb1); their iteration counts are not trustworthy. "
           "<span class='flag'>not-conv</span> GMRES did not converge. Methods: new η=ω / η=ring = two_sided with inner_product "
           "subdomain / ring; saddle point = the baseline MS-GFEM eigenproblem.</p>"]

    # Series, grouped like the experiments
    groups = catalog["groups"]
    out.append("<h2>Series (figures of the talk)</h2>")
    for g in groups:
        ss = [s for s in catalog.get("series", []) if s["group"] == g["id"]]
        if not ss:
            continue
        out.append(f"<h3>{html.escape(g['title'])}</h3>")
        for s in ss:
            methods, xs, points = series_table(s, exps)
            out.append(f"<p class='purpose'><b>{html.escape(s['title'])}</b><br><span class='muted'>{html.escape(s.get('note', ''))} "
                       f"Sources, first wins: {', '.join(s['dirs'])}.</span></p>")
            if not methods:
                out.append("<p class='muted'>No runs yet.</p>")
                continue
            t = ["<div class='scroll'><table class='series'><thead><tr><th>" + html.escape(s.get("x_label", s["x"])) + "</th>" +
                 "".join(f"<th>{html.escape(fmt(x, 0) if isinstance(x, float) else str(x))}</th>" for x in xs) + "</tr></thead><tbody>"]
            for m in methods:
                t.append(f"<tr><td>{html.escape(m)}</td>" + "".join(cell(points.get((m, x))) for x in xs) + "</tr>")
            t.append("</tbody></table></div>")
            out.append("".join(t))

    # Experiments by group
    out.append("<h2>Experiments</h2>")
    listed = set()
    for g in groups + [{"id": None, "title": "Uncatalogued (add to catalog.json)"}]:
        if g["id"] is None:
            names = [n for n in exps if n not in cat_exps]
        else:  # in the order of the catalog
            names = [n for n, c in cat_exps.items() if c.get("group") == g["id"] and n in exps]
        if not names:
            continue
        out.append(f"<h3>{html.escape(g['title'])}</h3>")
        if g.get("note"):
            out.append(f"<p class='muted'>{html.escape(g['note'])}</p>")
        for n in names:
            listed.add(n)
            e, c = exps[n], cat_exps.get(n, {})
            ok = sum(r["status"] == "ok" for r in e["rows"])
            bad = sum(r["status"] not in ("ok", "not run") for r in e["rows"])
            todo = sum(r["status"] == "not run" for r in e["rows"])
            counts = f"{ok} ok" + (f", <span class='flag'>{bad} failed</span>" if bad else "") + (f", {todo} not run" if todo else "")
            out.append(f"<details><summary><b>{html.escape(n)}</b> {badge(c.get('status', '?'))} <small>{counts}</small> — "
                       f"{html.escape(c.get('note', ''))}</summary>")
            links = [f"<a href='raw/{n}/'>folder</a>"]
            if e["runs_file"]:
                links.append(f"<a href='raw/{n}/{e['runs_file'].name}'>runs file</a>")
            links += [f"<a href='raw/{n}/{j.name}'>{j.name}</a>" for j in e["jobs"]]
            out.append(f"<p class='purpose'>{html.escape(e['purpose'])}</p><p><small>{' · '.join(links)} · commit {e['commit']}</small></p>")
            if e["notes"]:
                out.append(f"<div class='notes'>{html.escape(chr(10).join(e['notes']))}</div>")
            out.append(f"<div class='scroll'>{run_table(e)}</div></details>")
    out.append(f"<script>{JS}</script></main></body></html>")
    (results / "index.html").write_text("\n".join(out))
    missing = [n for n in cat_exps if n not in exps]
    print(f"{results / 'index.html'}: {len(exps)} experiments, {n_ok} ok, {n_fail} failed, {n_todo} not run; "
          f"uncatalogued: {[n for n in exps if n not in cat_exps]}; catalogued but missing: {missing}")


if __name__ == "__main__":
    build(Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else Path.home() / "Dokumente/PhD/Conferences/Bogota/Talk/results")
