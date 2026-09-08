#!/usr/bin/env python3
"""Turn a CVFEM benchmark CSV into a single self-contained HTML summary page.

    python3 report_cvfem_bench.py bench.csv -o report.html
    python3 report_cvfem_bench.py bench.csv --perf perf.csv -o report.html

Standard library only, so it runs inside a uenv on Alps with nothing installed.
Charts are inline SVG driven by CSS custom properties, so the page follows the
reader's light/dark theme, and every chart is backed by the table it came from.

The CSV is whatever `cvfem_hex8_ns_upwind_bench --csv FILE` appended. Repeated
runs of the same configuration are reduced to the best observed rate, which is
the right estimator on a shared machine: background load can only ever make a
run slower.
"""

import argparse
import csv
import datetime
import html
import math
import os
import sys
from collections import OrderedDict, defaultdict

# --------------------------------------------------------------------------- data

# Fixed series order. Colours are assigned by identity, never by rank, so a layout
# keeps its colour across every chart on the page.
LAYOUT_ORDER = ["atomic", "packed", "colored", "store"]
OP_ORDER = ["residual", "jac_action", "assemble", "bsr_apply"]
OP_LABEL = {
    "residual": "Residual",
    "jac_action": "Jacobian action",
    "assemble": "Jacobian assemble",
    "bsr_apply": "BSR SpMV",
}
PHASE_LABEL = OrderedDict(
    [
        ("ms_zero_global", "Zero global"),
        ("ms_zero_local", "Zero local"),
        ("ms_gather_u", "Gather"),
        ("ms_element_kernel", "Element kernel"),
        ("ms_local_to_global", "Local to global"),
        ("ms_ghost_reduce", "Ghost reduce"),
    ]
)


def fnum(row, key, default=float("nan")):
    v = row.get(key, "")
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def inum(row, key, default=0):
    v = row.get(key, "")
    if v is None or v == "":
        return default
    try:
        return int(float(v))
    except ValueError:
        return default


class Bench:
    """Best-of runs, indexed by configuration."""

    KEY = ("sweep", "operation", "layout", "kernel", "geom", "threads", "pack_size", "cube_n")

    def __init__(self, rows):
        self.rows = rows
        self.best = {}
        for r in rows:
            k = tuple(r[f] for f in self.KEY)
            prev = self.best.get(k)
            if prev is None or r["MDOF_s"] > prev["MDOF_s"]:
                self.best[k] = r

    @classmethod
    def load(cls, path):
        rows = []
        with open(path, newline="") as fh:
            for raw in csv.DictReader(fh):
                if not raw.get("operation"):
                    continue
                tag = raw.get("tag", "") or ""
                sweep = tag.split(":", 1)[1] if ":" in tag else "all"
                row = dict(raw)
                row["sweep"] = sweep
                row["tag_base"] = tag.split(":", 1)[0]
                row["threads"] = inum(raw, "threads")
                row["pack_size"] = inum(raw, "pack_size")
                row["cube_n"] = inum(raw, "cube_n")
                row["dofs"] = inum(raw, "dofs")
                row["elements"] = inum(raw, "elements")
                row["nodes"] = inum(raw, "nodes")
                row["bsr_nnz"] = inum(raw, "bsr_nnz")
                row["n_colors"] = inum(raw, "n_colors")
                row["packs_per_color_min"] = inum(raw, "packs_per_color_min")
                row["MDOF_s"] = fnum(raw, "MDOF_s")
                row["MDOF_s_element_visits"] = fnum(raw, "MDOF_s_element_visits")
                row["MELEM_s"] = fnum(raw, "MELEM_s")
                row["seconds_per_call"] = fnum(raw, "seconds_per_call")
                row["bsr_values_MiB"] = fnum(raw, "bsr_values_MiB", 0.0)
                row["GFLOP_s_model"] = fnum(raw, "GFLOP_s_model", 0.0)
                for ph in PHASE_LABEL:
                    row[ph] = fnum(raw, ph, float("nan"))
                if math.isnan(row["MDOF_s"]):
                    continue
                rows.append(row)
        if not rows:
            sys.exit("no usable rows in %s" % path)
        return cls(rows)

    def select(self, **kw):
        """Rows matching every given field; None means 'any'."""
        out = []
        for r in self.best.values():
            if all(v is None or r.get(k) == v for k, v in kw.items()):
                out.append(r)
        return out

    def pick(self, **kw):
        rs = self.select(**kw)
        return max(rs, key=lambda r: r["MDOF_s"]) if rs else None

    def best_over(self, over, **kw):
        """Best row per distinct value of `over`, as an ordered dict."""
        buckets = defaultdict(list)
        for r in self.select(**kw):
            buckets[r[over]].append(r)
        return OrderedDict(
            (k, max(v, key=lambda r: r["MDOF_s"])) for k, v in sorted(buckets.items(), key=lambda kv: kv[0])
        )


# ----------------------------------------------------------------------- svg bits

# Layouts are categorical: four fixed hues, assigned by identity and never cycled.
# The set is Okabe-Ito derived and clears the all-pairs colour-vision check in light
# mode; the dark steps sit inside the dark lightness band, where the blue/pink pair
# lands just under the separation floor -- so every chart also carries a secondary
# encoding (distinct markers and dashes on lines, direct value labels on bars) and
# the table underneath it. Never add a fifth layout hue: fold it into "other" or
# facet instead.
SERIES_VAR = ["--s1", "--s2", "--s3", "--s4"]

# Assembly phases are ordinal, not categorical -- they are the pipeline in order --
# so they get a single-hue sequential ramp, which stays readable at six steps and is
# colour-vision-safe by construction.
PHASE_VAR = ["--p1", "--p2", "--p3", "--p4", "--p5", "--p6"]

# Line dash / marker per layout, so identity survives without colour.
SERIES_DASH = ["", "6 3", "2 3", "8 3 2 3"]


def series_color(name, order):
    try:
        i = order.index(name)
    except ValueError:
        i = len(order)
    return "var(%s)" % SERIES_VAR[i % len(SERIES_VAR)]


def series_dash(name, order):
    try:
        i = order.index(name)
    except ValueError:
        i = len(order)
    return SERIES_DASH[i % len(SERIES_DASH)]


def phase_color_var(i):
    return "var(%s)" % PHASE_VAR[i % len(PHASE_VAR)]


def esc(s):
    return html.escape(str(s), quote=True)


def fmt(v, nd=1):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "—"
    return ("%%.%df" % nd) % v


def svg_hbars(rows, unit="MDOF/s", height_per=26, label_w=170, width=760, color_of=None):
    """rows: list of (label, sublabel, value, series_name). Horizontal bars."""
    rows = [r for r in rows if r[2] == r[2]]
    if not rows:
        return "<p class='empty'>no data for this chart</p>"
    vmax = max(r[2] for r in rows) * 1.16 or 1.0
    plot_w = width - label_w - 8
    h = height_per * len(rows) + 26
    out = ['<svg viewBox="0 0 %d %d" width="100%%" height="%d" role="img">' % (width, h, h)]
    for i, (label, sub, val, series) in enumerate(rows):
        y = i * height_per
        bw = max(2.0, plot_w * val / vmax)
        col = color_of(series) if color_of else "var(--s1)"
        out.append(
            '<text x="%d" y="%.1f" class="cl" text-anchor="end">%s</text>'
            % (label_w - 10, y + 12, esc(label))
        )
        if sub:
            out.append(
                '<text x="%d" y="%.1f" class="cs" text-anchor="end">%s</text>'
                % (label_w - 10, y + 22, esc(sub))
            )
        out.append(
            '<rect x="%d" y="%.1f" width="%.1f" height="%d" rx="3" fill="%s">'
            '<title>%s%s: %s %s</title></rect>'
            % (label_w, y + 3, bw, height_per - 10, col, esc(label),
               (" " + esc(sub)) if sub else "", fmt(val, 1), unit)
        )
        out.append(
            '<text x="%.1f" y="%.1f" class="cv">%s</text>'
            % (label_w + bw + 6, y + 15, fmt(val, 1))
        )
    out.append(
        '<text x="%d" y="%d" class="cs">%s — higher is better</text>' % (label_w, h - 4, esc(unit))
    )
    out.append("</svg>")
    return "".join(out)


def svg_lines(series, xlabel, ylabel, logx=False, width=760, height=290, color_of=None,
              dash_of=None):
    """series: OrderedDict name -> list of (x, y). Multi-series line chart."""
    pts = [(x, y) for pts_ in series.values() for (x, y) in pts_ if y == y]
    if not pts:
        return "<p class='empty'>no data for this chart</p>"
    xs = sorted({p[0] for p in pts})
    ymax = max(p[1] for p in pts) * 1.12 or 1.0
    L, R, T, B = 56, 74, 14, 40
    pw, ph = width - L - R, height - T - B

    def tx(x):
        if logx:
            lo, hi = math.log(min(xs)), math.log(max(xs))
            return L + (pw if hi == lo else pw * (math.log(x) - lo) / (hi - lo))
        lo, hi = min(xs), max(xs)
        return L + (pw if hi == lo else pw * (x - lo) / (hi - lo))

    def ty(y):
        return T + ph - ph * y / ymax

    out = ['<svg viewBox="0 0 %d %d" width="100%%" height="%d" role="img">' % (width, height, height)]
    # y grid
    for i in range(5):
        v = ymax * i / 4.0
        y = ty(v)
        out.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" class="grid"/>' % (L, y, L + pw, y))
        out.append('<text x="%d" y="%.1f" class="cs" text-anchor="end">%s</text>' % (L - 8, y + 3, fmt(v, 0)))
    # x ticks
    for x in xs:
        out.append(
            '<text x="%.1f" y="%d" class="cs" text-anchor="middle">%s</text>'
            % (tx(x), T + ph + 15, esc(x))
        )
    out.append(
        '<text x="%.1f" y="%d" class="cs" text-anchor="middle">%s</text>'
        % (L + pw / 2.0, height - 4, esc(xlabel))
    )
    out.append(
        '<text transform="translate(12,%.1f) rotate(-90)" class="cs" text-anchor="middle">%s</text>'
        % (T + ph / 2.0, esc(ylabel))
    )
    for name, data in series.items():
        data = sorted([(x, y) for x, y in data if y == y])
        if not data:
            continue
        col = color_of(name) if color_of else "var(--s1)"
        dash = dash_of(name) if dash_of else ""
        d = " ".join(
            ("M" if i == 0 else "L") + "%.1f %.1f" % (tx(x), ty(y)) for i, (x, y) in enumerate(data)
        )
        out.append(
            '<path d="%s" fill="none" stroke="%s" stroke-width="2"%s/>'
            % (d, col, (' stroke-dasharray="%s"' % dash) if dash else "")
        )
        for x, y in data:
            out.append(
                '<circle cx="%.1f" cy="%.1f" r="4" fill="%s" stroke="var(--surface)" stroke-width="2">'
                '<title>%s — %s %s: %s</title></circle>'
                % (tx(x), ty(y), col, esc(name), esc(xlabel), esc(x), fmt(y, 1))
            )
        # end-of-line label: identity without relying on colour
        lx, ly = data[-1]
        out.append(
            '<text x="%.1f" y="%.1f" class="cv" fill="%s">%s</text>'
            % (min(tx(lx) + 7, width - 4), ty(ly) + 3, col, esc(name))
        )
    out.append("</svg>")
    return "".join(out)


def svg_stacked(rows, keys, labels, unit="ms", width=760, label_w=170, height_per=34, color_of=None):
    """rows: list of (label, sublabel, {key: value}). Stacked horizontal bars."""
    rows = [r for r in rows if any(r[2].get(k, float("nan")) == r[2].get(k, float("nan")) for k in keys)]
    if not rows:
        return "<p class='empty'>no data for this chart</p>"
    totals = [sum(r[2].get(k, 0.0) or 0.0 for k in keys) for r in rows]
    vmax = max(totals) * 1.06 or 1.0
    plot_w = width - label_w - 60
    h = height_per * len(rows) + 8
    out = ['<svg viewBox="0 0 %d %d" width="100%%" height="%d" role="img">' % (width, h, h)]
    for i, ((label, sub, vals), tot) in enumerate(zip(rows, totals)):
        y = i * height_per
        out.append('<text x="%d" y="%.1f" class="cl" text-anchor="end">%s</text>' % (label_w - 10, y + 15, esc(label)))
        if sub:
            out.append('<text x="%d" y="%.1f" class="cs" text-anchor="end">%s</text>' % (label_w - 10, y + 25, esc(sub)))
        x = float(label_w)
        for k in keys:
            v = vals.get(k, 0.0) or 0.0
            if v <= 0 or v != v:
                continue
            w = plot_w * v / vmax
            col = color_of(k) if color_of else "var(--s1)"
            out.append(
                '<rect x="%.1f" y="%.1f" width="%.1f" height="%d" fill="%s" stroke="var(--surface)" '
                'stroke-width="1"><title>%s — %s: %s %s (%.0f%%)</title></rect>'
                % (x, y + 5, max(w - 1, 0.6), height_per - 16, col, esc(label), esc(labels[k]),
                   fmt(v, 2), unit, 100.0 * v / tot if tot else 0.0)
            )
            x += w
        out.append('<text x="%.1f" y="%.1f" class="cv">%s %s</text>' % (x + 6, y + 18, fmt(tot, 1), unit))
    out.append("</svg>")
    return "".join(out)


def legend(names, color_of, dash_of=None):
    def swatch(n):
        if dash_of is None:
            return '<i class="sw" style="background:%s"></i>' % color_of(n)
        d = dash_of(n)
        return ('<svg class="swl" viewBox="0 0 22 8" width="22" height="8">'
                '<line x1="1" y1="4" x2="21" y2="4" stroke="%s" stroke-width="2"%s/></svg>'
                % (color_of(n), (' stroke-dasharray="%s"' % d) if d else ""))
    return '<div class="legend">%s</div>' % "".join(
        "<span>%s%s</span>" % (swatch(n), esc(n)) for n in names
    )


def table(headers, rows, highlight=None):
    out = ['<div class="tablewrap"><table><thead><tr>']
    out += ["<th>%s</th>" % esc(h) for h in headers]
    out.append("</tr></thead><tbody>")
    for i, r in enumerate(rows):
        cls = ' class="best"' if highlight and highlight(i, r) else ""
        out.append("<tr%s>" % cls)
        out += ["<td>%s</td>" % (c if isinstance(c, str) and c.startswith("<") else esc(c)) for c in r]
        out.append("</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


# ------------------------------------------------------------------------- perf

PERF_ALIAS = {
    "cycles": ["cycles", "cpu-cycles", "armv8_pmuv3_0/CPU_CYCLES/"],
    "instructions": ["instructions", "armv8_pmuv3_0/INST_RETIRED/"],
    "stall_backend": ["armv8_pmuv3_0/STALL_BACKEND/", "stalled-cycles-backend", "STALL_BACKEND"],
    "stall_backend_mem": ["armv8_pmuv3_0/STALL_BACKEND_MEM/", "STALL_BACKEND_MEM"],
    "stall_frontend": ["armv8_pmuv3_0/STALL_FRONTEND/", "stalled-cycles-frontend", "STALL_FRONTEND"],
    "l1d_loads": ["L1-dcache-loads", "armv8_pmuv3_0/L1D_CACHE/", "L1D_CACHE"],
    "l1d_misses": ["L1-dcache-load-misses", "armv8_pmuv3_0/L1D_CACHE_REFILL/", "L1D_CACHE_REFILL"],
    "l2d": ["armv8_pmuv3_0/L2D_CACHE/", "L2D_CACHE"],
    "l2d_refill": ["armv8_pmuv3_0/L2D_CACHE_REFILL/", "L2D_CACHE_REFILL"],
    "ll_rd": ["armv8_pmuv3_0/LL_CACHE_RD/", "LL_CACHE_RD"],
    "ll_miss_rd": ["armv8_pmuv3_0/LL_CACHE_MISS_RD/", "LL_CACHE_MISS_RD"],
    "bus_access": ["armv8_pmuv3_0/BUS_ACCESS/", "BUS_ACCESS"],
    "inst_spec": ["armv8_pmuv3_0/INST_SPEC/", "INST_SPEC"],
    "ase_spec": ["armv8_pmuv3_0/ASE_SPEC/", "ASE_SPEC"],
    "vfp_spec": ["armv8_pmuv3_0/VFP_SPEC/", "VFP_SPEC"],
    "ld_spec": ["armv8_pmuv3_0/LD_SPEC/", "LD_SPEC"],
    "st_spec": ["armv8_pmuv3_0/ST_SPEC/", "ST_SPEC"],
    "scf_rd": ["nvidia_scf_pmu_0/cmem_rd_data/", "cmem_rd_data"],
    "scf_wr": ["nvidia_scf_pmu_0/cmem_wr_total_bytes/", "cmem_wr_total_bytes"],
}


def load_perf(path):
    per_cfg = defaultdict(dict)
    meta = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            cfg = r.get("config")
            ev = (r.get("event") or "").strip()
            if not cfg or not ev:
                continue
            try:
                per_cfg[cfg][ev] = float(r["count"])
            except (TypeError, ValueError):
                continue
            meta.setdefault(
                cfg,
                {
                    "operation": r.get("operation", ""),
                    "layout": r.get("layout", ""),
                    "threads": r.get("threads", ""),
                    "cube_n": r.get("cube_n", ""),
                    "seconds": fnum(r, "run_seconds", float("nan")),
                },
            )
    return per_cfg, meta


def perf_get(counts, name):
    for alias in PERF_ALIAS.get(name, []):
        if alias in counts:
            return counts[alias]
        for k, v in counts.items():
            if k.strip("/").endswith(alias.strip("/")):
                return v
    return None


def perf_metrics(counts, seconds, flit_bytes):
    """Derived, with every ratio guarded so a missing counter drops the row not the page."""
    g = lambda n: perf_get(counts, n)
    m = OrderedDict()
    cyc, ins = g("cycles"), g("instructions")
    if cyc and ins:
        m["IPC"] = ins / cyc
    for key, label in (("stall_backend", "Backend stall %"),
                       ("stall_backend_mem", "Memory stall %"),
                       ("stall_frontend", "Frontend stall %")):
        v = g(key)
        if v is not None and cyc:
            m[label] = 100.0 * v / cyc
    ld, ms = g("l1d_loads"), g("l1d_misses")
    if ld and ms is not None:
        m["L1D miss %"] = 100.0 * ms / ld
    l2, l2r = g("l2d"), g("l2d_refill")
    if l2 and l2r is not None:
        m["L2 refill %"] = 100.0 * l2r / l2
    llr, llm = g("ll_rd"), g("ll_miss_rd")
    if llr and llm is not None:
        m["LL read miss %"] = 100.0 * llm / llr
    isp, ase, vfp = g("inst_spec"), g("ase_spec"), g("vfp_spec")
    if isp:
        if ase is not None:
            m["SIMD (ASE) % of ops"] = 100.0 * ase / isp
        if vfp is not None:
            m["FP (VFP) % of ops"] = 100.0 * vfp / isp
    rd, wr = g("scf_rd"), g("scf_wr")
    if seconds == seconds and seconds > 0:
        if rd is not None and wr is not None:
            m["DRAM GB/s (SCF)"] = (rd + wr) * flit_bytes / seconds / 1e9
        bus = g("bus_access")
        if bus is not None:
            m["Bus GB/s (est.)"] = bus * 64.0 / seconds / 1e9
    return m


# ------------------------------------------------------------------------- page

CSS = """
:root{--paper:#F6F6F3;--surface:#FCFCFB;--sunk:#EEEFEA;--ink:#191B1A;--ink-2:#525854;
--muted:#858B86;--rule:#DDDFD8;--rule-2:#C9CCC3;--accent:#0B7A9E;--accent-soft:#E2EEF3;
--s1:#0072B2;--s2:#D55E00;--s3:#009E73;--s4:#CC79A7;--p1:#83B6D6;--p2:#63A0C9;--p3:#4283B2;--p4:#276492;--p5:#144870;--p6:#092E4E;--measure:66ch}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
--paper:#131614;--surface:#1A1D1B;--sunk:#212523;--ink:#E7E9E4;--ink-2:#AEB4AE;
--muted:#838A84;--rule:#2B302D;--rule-2:#3B413D;--accent:#4FB0D2;--accent-soft:#16303A;
--s1:#2A8CC8;--s2:#D46A1C;--s3:#08A176;--s4:#C874A4;--p1:#1E5580;--p2:#2E6E9C;--p3:#4189B8;--p4:#5EA4CE;--p5:#87BFDF;--p6:#B4D8EE}}
:root[data-theme="dark"]{--paper:#131614;--surface:#1A1D1B;--sunk:#212523;--ink:#E7E9E4;
--ink-2:#AEB4AE;--muted:#838A84;--rule:#2B302D;--rule-2:#3B413D;--accent:#4FB0D2;
--accent-soft:#16303A;--s1:#2A8CC8;--s2:#D46A1C;--s3:#08A176;--s4:#C874A4;--p1:#1E5580;--p2:#2E6E9C;--p3:#4189B8;--p4:#5EA4CE;--p5:#87BFDF;--p6:#B4D8EE}
*{box-sizing:border-box}
body{background:var(--paper);color:var(--ink);margin:0;
font-family:"Source Serif 4",Georgia,"Times New Roman",serif;font-size:17px;line-height:1.62;
-webkit-font-smoothing:antialiased}
.wrap{max-width:940px;margin:0 auto;padding:clamp(2rem,6vw,4rem) clamp(1.1rem,4vw,2.5rem) 5rem;
display:flex;flex-direction:column;gap:3rem}
h1,h2,h3,.eyebrow,th,.legend,.cl,.cs,.cv,.tag{font-family:"IBM Plex Sans Condensed","Helvetica Neue",Arial,sans-serif}
h1{font-size:clamp(2rem,5.5vw,2.9rem);font-weight:700;line-height:1.05;letter-spacing:-.015em;
text-wrap:balance;margin:0}
h2{font-size:clamp(1.3rem,3vw,1.65rem);font-weight:600;line-height:1.15;text-wrap:balance;margin:0}
h3{font-size:1.02rem;font-weight:600;margin:0}
p{margin:0;max-width:var(--measure)}
.eyebrow{font-size:.72rem;font-weight:600;letter-spacing:.13em;text-transform:uppercase;
color:var(--accent);margin:0}
code,.mono{font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace;font-size:.85em}
code{background:var(--sunk);border:1px solid var(--rule);border-radius:3px;padding:.08em .36em}
section{display:flex;flex-direction:column;gap:1.2rem}
header{display:flex;flex-direction:column;gap:1rem}
.lede{font-size:1.1rem;color:var(--ink-2);max-width:62ch}
.meta{font-family:"IBM Plex Mono",monospace;font-size:.75rem;color:var(--muted);display:flex;
flex-wrap:wrap;gap:.35rem 1.4rem;padding-top:.9rem;border-top:1px solid var(--rule)}
.figures{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1px;
background:var(--rule);border:1px solid var(--rule)}
.figure{background:var(--surface);padding:1.05rem 1.15rem 1.2rem;display:flex;flex-direction:column;gap:.28rem}
.figure .k{font-family:"IBM Plex Sans Condensed",sans-serif;font-size:.67rem;font-weight:600;
letter-spacing:.11em;text-transform:uppercase;color:var(--muted)}
.figure .v{font-family:"IBM Plex Mono",monospace;font-size:1.8rem;font-weight:500;
letter-spacing:-.02em;font-variant-numeric:tabular-nums;line-height:1}
.figure .n{font-size:.83rem;color:var(--ink-2);line-height:1.4}
.panel{background:var(--surface);border:1px solid var(--rule);padding:clamp(1rem,3vw,1.5rem);
display:flex;flex-direction:column;gap:1rem;overflow-x:auto}
.panel-head{display:flex;flex-direction:column;gap:.25rem}
.panel-head .sub{font-size:.85rem;color:var(--ink-2);max-width:62ch}
.legend{display:flex;flex-wrap:wrap;gap:.3rem 1.1rem;font-size:.78rem;color:var(--ink-2);font-weight:500}
.legend span{display:inline-flex;align-items:center;gap:.4rem}
.sw{width:11px;height:11px;border-radius:2px;flex:none;display:inline-block}
.swl{flex:none;display:inline-block;min-width:22px}
svg{display:block;min-width:560px}
.cl{font-size:11px;font-weight:600;fill:var(--ink)}
.cs{font-size:10px;fill:var(--muted)}
.cv{font-size:10px;font-family:"IBM Plex Mono",monospace;fill:var(--ink-2)}
.grid{stroke:var(--rule);stroke-width:1}
.tablewrap{overflow-x:auto;border:1px solid var(--rule);background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:.84rem}
th,td{text-align:right;padding:.45rem .8rem;border-bottom:1px solid var(--rule);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
thead th{font-size:.69rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;
color:var(--muted);background:var(--sunk)}
tbody td{font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}
tbody td:first-child{font-family:"IBM Plex Sans Condensed",sans-serif;font-weight:500}
tbody tr:last-child td{border-bottom:0}
tbody tr.best td{background:var(--accent-soft)}
.empty{font-size:.85rem;color:var(--muted);font-style:italic}
details{background:var(--surface);border:1px solid var(--rule);padding:.7rem 1rem}
summary{cursor:pointer;font-family:"IBM Plex Sans Condensed",sans-serif;font-weight:600;font-size:.9rem}
details[open] summary{margin-bottom:.7rem}
footer{font-family:"IBM Plex Mono",monospace;font-size:.73rem;color:var(--muted);line-height:1.7;
border-top:1px solid var(--rule);padding-top:1.1rem}
"""

FONTS = ('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
         'family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600;700&'
         'family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap">')


def build(bench, perf, perf_meta, args):
    lay_color = lambda n: series_color(n, LAYOUT_ORDER)
    lay_dash = lambda n: series_dash(n, LAYOUT_ORDER)
    op_color = lambda n: series_color(n, OP_ORDER)
    phase_keys_all = list(PHASE_LABEL.keys())
    phase_color = lambda k: phase_color_var(phase_keys_all.index(k) if k in phase_keys_all else 0)

    host = bench.rows[0].get("host", "unknown")
    tag = bench.rows[0].get("tag_base", "run")
    max_threads = max(r["threads"] for r in bench.rows)
    S = []

    # ---- headline -------------------------------------------------------
    asm_best = bench.pick(operation="assemble", threads=max_threads)
    asm_packed = bench.pick(operation="assemble", layout="packed", threads=max_threads)
    asm_atomic = bench.pick(operation="assemble", layout="atomic", threads=max_threads)
    res_best = bench.pick(operation="residual", threads=max_threads)
    spmv = bench.pick(operation="bsr_apply", threads=max_threads)

    figs = []
    if asm_best:
        figs.append(("Assembly, best", "%s" % fmt(asm_best["MDOF_s"], 1),
                     "MDOF/s — %s, pack %s, n=%s" % (asm_best["layout"], asm_best["pack_size"], asm_best["cube_n"]), True))
    if asm_packed and asm_best and asm_packed["MDOF_s"] > 0:
        figs.append(("vs packed baseline", "%.2fx" % (asm_best["MDOF_s"] / asm_packed["MDOF_s"]),
                     "packed at %s MDOF/s" % fmt(asm_packed["MDOF_s"], 1), False))
    if asm_atomic and asm_best and asm_atomic["MDOF_s"] > 0:
        figs.append(("vs atomic", "%.2fx" % (asm_best["MDOF_s"] / asm_atomic["MDOF_s"]),
                     "atomic at %s MDOF/s" % fmt(asm_atomic["MDOF_s"], 1), False))
    if res_best and asm_best and asm_best["MDOF_s"] > 0:
        figs.append(("Residual / assembly", "%.0fx" % (res_best["MDOF_s"] / asm_best["MDOF_s"]),
                     "matrix-free residual is %s MDOF/s" % fmt(res_best["MDOF_s"], 0), False))

    S.append(
        "<header><p class='eyebrow'>SFEM · spikes/cvfem · HEX8 Navier–Stokes</p>"
        "<h1>%s</h1><p class='lede'>%s</p>"
        "<div class='meta'><span>%s</span><span>%s threads</span><span>%s configurations, "
        "%s runs</span><span>%s</span></div></header>"
        % (esc(args.title), esc(args.subtitle), esc(host), max_threads,
           len(bench.best), len(bench.rows),
           datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    )
    if figs:
        S.append(
            "<div class='figures'>%s</div>"
            % "".join(
                "<div class='figure'><span class='k'>%s</span>"
                "<span class='v'%s>%s</span><span class='n'>%s</span></div>"
                % (esc(k), " style='color:var(--accent)'" if hi else "", esc(v), esc(n))
                for k, v, n, hi in figs
            )
        )
    S.append(
        "<p style='font-size:.88rem;color:var(--ink-2);max-width:74ch'><strong>Metric.</strong> "
        "MDOF/s counts unique mesh degrees of freedom — three velocity components and one pressure "
        "per node — divided by the time for one sweep. Repeated runs of a configuration are reduced "
        "to the best observed rate, which is the right estimator on a shared machine: contention can "
        "only make a run slower.</p>"
    )

    # ---- layout x operation --------------------------------------------
    rows = []
    for op in OP_ORDER:
        for lay in LAYOUT_ORDER:
            r = bench.pick(operation=op, layout=lay, threads=max_threads, sweep="layout")
            if r:
                rows.append((OP_LABEL.get(op, op), "%s · pack %s" % (lay, r["pack_size"]), r["MDOF_s"], lay))
    if rows:
        S.append(
            "<section><p class='eyebrow'>Layouts</p><h2>Every layout, every operation</h2>"
            "<div class='panel'><div class='panel-head'><h3>Throughput by layout</h3>"
            "<p class='sub'>n=%s, %s threads, sumfact kernel. Each bar is the best of the repeated "
            "runs for that configuration.</p></div>%s%s</div>%s</section>"
            % (esc(asm_best["cube_n"] if asm_best else "?"), max_threads,
               legend(LAYOUT_ORDER, lay_color),
               svg_hbars(rows, color_of=lay_color),
               table(["Operation", "Layout", "Pack", "MDOF/s", "s/call"],
                     [(OP_LABEL.get(r[0], r[0]), r[1], r[2], fmt(r[3], 2), "%.4e" % r[4])
                      for r in sorted(
                          [(x["operation"], x["layout"], x["pack_size"], x["MDOF_s"], x["seconds_per_call"])
                           for x in bench.select(threads=max_threads, sweep="layout")],
                          key=lambda t: (OP_ORDER.index(t[0]) if t[0] in OP_ORDER else 9, -t[3]))]))
        )

    # ---- pack size ------------------------------------------------------
    packs = OrderedDict()
    for op in ("assemble", "residual"):
        for lay in ("packed", "colored", "store"):
            pts = [(r["pack_size"], r["MDOF_s"])
                   for r in bench.select(sweep="packsize", operation=op, layout=lay, threads=max_threads)]
            if pts:
                packs["%s / %s" % (OP_LABEL.get(op, op), lay)] = pts
    if packs:
        asm_only = OrderedDict((k, v) for k, v in packs.items() if k.startswith("Jacobian assemble"))
        names = list(asm_only.keys())
        cf = lambda n: series_color(n.split("/")[-1].strip(), LAYOUT_ORDER)
        df = lambda n: series_dash(n.split("/")[-1].strip(), LAYOUT_ORDER)
        S.append(
            "<section><p class='eyebrow'>Pack size</p><h2>How big should a pack be?</h2>"
            "<p>Assembly wants a pack-local working set that fits in cache; the colored sweep "
            "additionally needs each colour to hold at least as many packs as there are threads, "
            "or the barrier ending the colour idles most of them.</p>"
            "<div class='panel'><div class='panel-head'><h3>Assembly vs pack size</h3></div>"
            "%s%s</div></section>"
            % (legend(names, cf),
               svg_lines(asm_only, "pack size (elements)", "MDOF/s", logx=True, color_of=cf))
        )

    # ---- thread scaling -------------------------------------------------
    for op in ("assemble", "residual"):
        ser = OrderedDict()
        for lay in LAYOUT_ORDER:
            pts = [(r["threads"], r["MDOF_s"]) for r in bench.select(sweep="threads", operation=op, layout=lay)]
            if pts:
                ser[lay] = pts
        if not ser:
            continue
        base = {lay: min(pts, key=lambda p: p[0]) for lay, pts in ser.items()}
        eff_rows = []
        for lay, pts in ser.items():
            for t, v in sorted(pts):
                b = base[lay]
                eff = 100.0 * (v / b[1]) / (t / b[0]) if b[1] else float("nan")
                eff_rows.append((lay, t, fmt(v, 1), "%.0f%%" % eff))
        S.append(
            "<section><p class='eyebrow'>Thread scaling</p>"
            "<h2>%s across the socket</h2>"
            "<div class='panel'><div class='panel-head'><h3>%s</h3>"
            "<p class='sub'>Parallel efficiency is relative to the smallest thread count measured "
            "for that layout. A layout that flattens early is bandwidth-bound, not compute-bound.</p>"
            "</div>%s%s</div>%s</section>"
            % (OP_LABEL.get(op, op), OP_LABEL.get(op, op),
               legend(list(ser.keys()), lay_color, dash_of=lay_dash),
               svg_lines(ser, "threads", "MDOF/s", logx=True, color_of=lay_color, dash_of=lay_dash),
               table(["Layout", "Threads", "MDOF/s", "Parallel eff."], eff_rows))
        )

    # ---- problem size ---------------------------------------------------
    ser = OrderedDict()
    size_rows = []
    for lay in LAYOUT_ORDER:
        pts = [(r["dofs"], r["MDOF_s"]) for r in bench.select(sweep="size", operation="assemble", layout=lay,
                                                              threads=max_threads)]
        if pts:
            ser[lay] = pts
    for r in sorted(bench.select(sweep="size", operation="assemble", threads=max_threads),
                    key=lambda r: (r["cube_n"], r["layout"])):
        size_rows.append((r["layout"], r["cube_n"], "%d" % r["dofs"], fmt(r["bsr_values_MiB"], 0) + " MiB",
                          fmt(r["MDOF_s"], 1)))
    if ser:
        S.append(
            "<section><p class='eyebrow'>Problem size</p><h2>The win needs a matrix bigger than cache</h2>"
            "<p>Assembly writes about 3.4&nbsp;KB of matrix per element. While that fits in cache there "
            "is no DRAM traffic to save and the restructuring only costs barriers; once it does not, the "
            "ratio settles.</p>"
            "<div class='panel'><div class='panel-head'><h3>Assembly vs problem size</h3></div>%s%s</div>%s</section>"
            % (legend(list(ser.keys()), lay_color),
               svg_lines(ser, "dofs", "MDOF/s", logx=True, color_of=lay_color),
               table(["Layout", "cube n", "Dofs", "BSR values", "MDOF/s"], size_rows))
        )

    # ---- kernels --------------------------------------------------------
    krows = []
    for r in sorted(bench.select(sweep="kernel", threads=max_threads),
                    key=lambda r: (r["operation"], -r["MDOF_s"])):
        krows.append((OP_LABEL.get(r["operation"], r["operation"]), r["kernel"], r["layout"],
                      fmt(r["MDOF_s"], 1), fmt(r["GFLOP_s_model"], 1)))
    if krows:
        S.append(
            "<section><p class='eyebrow'>Element kernels</p><h2>Kernel variants</h2>"
            "<p>The GFLOP/s column uses the benchmark's idealised flop model, which understates the "
            "SymPy assembly kernel by roughly 2.4&times; — compare kernels by MDOF/s, not by that "
            "column.</p>%s</section>"
            % table(["Operation", "Kernel", "Layout", "MDOF/s", "GFLOP/s (model)"], krows)
        )

    # ---- phase breakdown ------------------------------------------------
    brk = bench.select(sweep="breakdown", operation="assemble", threads=max_threads)
    if brk:
        keys = [k for k in PHASE_LABEL if any(r[k] == r[k] and r[k] > 0 for r in brk)]
        ordered = sorted(brk, key=lambda r: LAYOUT_ORDER.index(r["layout"])
                         if r["layout"] in LAYOUT_ORDER else 9)
        rows = [("%s" % r["layout"], "pack %s" % r["pack_size"], {k: r[k] for k in keys})
                for r in ordered]
        brk_table = table(
            ["Layout", "Pack"] + [PHASE_LABEL[k] for k in keys] + ["Total"],
            [[r["layout"], r["pack_size"]]
             + [fmt(r[k], 2) for k in keys]
             + [fmt(sum((r[k] if r[k] == r[k] else 0.0) for k in keys), 1)]
             for r in ordered])
        S.append(
            "<section><p class='eyebrow'>Where the time goes</p><h2>Phase breakdown</h2>"
            "<p>Milliseconds per assemble, summed over all threads. This is the measurement that "
            "says whether the element kernel or the data movement around it is the cost.</p>"
            "<div class='panel'><div class='panel-head'><h3>Assembly phases</h3>"
            "<p class='sub'>Thread-summed; divide by the thread count for wall time. The timers "
            "perturb the run slightly, so totals sit a few percent above the untimed wall clock.</p>"
            "</div>%s%s</div>%s</section>"
            % (legend([PHASE_LABEL[k] for k in keys], lambda n: phase_color(
                   next(k for k in keys if PHASE_LABEL[k] == n))),
               svg_stacked(rows, keys, PHASE_LABEL, color_of=phase_color),
               brk_table)
        )

    # ---- perf -----------------------------------------------------------
    if perf:
        cfgs = sorted(perf.keys())
        metric_names = OrderedDict()
        per_cfg_metrics = {}
        for cfg in cfgs:
            m = perf_metrics(perf[cfg], perf_meta.get(cfg, {}).get("seconds", float("nan")), args.scf_flit_bytes)
            per_cfg_metrics[cfg] = m
            for k in m:
                metric_names[k] = True
        prows = []
        for cfg in cfgs:
            meta = perf_meta.get(cfg, {})
            m = per_cfg_metrics[cfg]
            prows.append(
                [OP_LABEL.get(meta.get("operation", ""), meta.get("operation", cfg)), meta.get("layout", "")]
                + [fmt(m.get(k), 2) if k == "IPC" else fmt(m.get(k), 1) for k in metric_names]
            )
        S.append(
            "<section><p class='eyebrow'>Hardware counters</p><h2>What the machine says it is doing</h2>"
            "<p>Counters are collected in separate runs per group rather than multiplexed, because "
            "multiplexed ratios are not trustworthy for a roofline argument. A high memory-stall "
            "fraction next to a low IPC is the signature of a bandwidth-bound kernel; a low stall "
            "fraction with low IPC points at dependency chains or spills instead.</p>%s"
            "<p style='font-size:.85rem;color:var(--muted)'>DRAM GB/s is derived from the Grace SCF "
            "counters assuming %d bytes per counted flit — calibrate that against a STREAM run before "
            "quoting it. Bus GB/s assumes 64-byte bus beats and is an estimate.</p></section>"
            % (table(["Operation", "Layout"] + list(metric_names), prows), args.scf_flit_bytes)
        )

    # ---- raw ------------------------------------------------------------
    S.append(
        "<details><summary>All measurements (%d configurations)</summary>%s</details>"
        % (len(bench.best),
           table(["Sweep", "Op", "Layout", "Kernel", "Geom", "Thr", "Pack", "n", "Dofs", "MDOF/s", "s/call"],
                 [(r["sweep"], r["operation"], r["layout"], r["kernel"], r["geom"], r["threads"],
                   r["pack_size"], r["cube_n"], r["dofs"], fmt(r["MDOF_s"], 2), "%.4e" % r["seconds_per_call"])
                  for r in sorted(bench.best.values(),
                                  key=lambda r: (r["sweep"], r["operation"], r["layout"], r["cube_n"],
                                                 r["threads"], r["pack_size"]))]))
    )

    S.append(
        "<footer>Generated by report_cvfem_bench.py from %s%s.<br>"
        "Tag %s · host %s · best of the repeated runs per configuration.</footer>"
        % (esc(os.path.basename(args.csv)),
           esc(" and " + os.path.basename(args.perf)) if args.perf else "",
           esc(tag), esc(host))
    )

    body = "<div class='wrap'>%s</div>" % "".join(S)
    if args.fragment:
        # Body-only form for `Artifact`/Claude Code publishing, which supplies its
        # own document skeleton. Keep the title tag: the tool reads it for the name.
        return "<title>%s</title>%s<style>%s</style>%s" % (esc(args.title), FONTS, CSS, body)
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>%s</title>%s<style>%s</style></head><body>%s</body></html>"
            % (esc(args.title), FONTS, CSS, body))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="benchmark csv written by --csv")
    ap.add_argument("--perf", help="perf counter csv written by perf_hex8_alps.sbatch")
    ap.add_argument("-o", "--out", default="cvfem_report.html")
    ap.add_argument("--title", default="CVFEM HEX8 on Grace")
    ap.add_argument("--subtitle",
                    default="Residual, matrix-free Jacobian action and BSR assembly for the HEX8 "
                            "CVFEM Navier-Stokes operator, measured across assembly layouts.")
    ap.add_argument("--fragment", action="store_true",
                    help="emit body-only html for publishing as a Claude Code artifact")
    ap.add_argument("--scf-flit-bytes", type=int, default=32,
                    help="bytes per Grace SCF cmem flit used for the DRAM bandwidth estimate")
    args = ap.parse_args()

    bench = Bench.load(args.csv)
    perf, perf_meta = load_perf(args.perf) if args.perf else ({}, {})
    with open(args.out, "w") as fh:
        fh.write(build(bench, perf, perf_meta, args))
    print("wrote %s  (%d configurations from %d runs%s)"
          % (args.out, len(bench.best), len(bench.rows),
             ", %d perf configs" % len(perf) if perf else ""))


if __name__ == "__main__":
    main()
