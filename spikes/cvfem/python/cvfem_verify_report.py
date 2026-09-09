#!/usr/bin/env python3
"""Turn a directory of CVFEM verification runs into a Markdown report, with plots.

    python3 cvfem_verify_report.py RUNDIR -o docs/CVFEM_Verification_Report.md --html

RUNDIR is what scripts/verify_report.sh (or jobs/verify_report.sbatch) produced: one log
per run plus a manifest.json describing them. Nothing here launches a solve, so a report
can be regenerated from an old run directory without the machine that made it.

Standard library only, and that is a requirement rather than a preference: numpy,
matplotlib and markdown are all absent from the Alps uenv's python3 and from the default
python3 on the development laptop. So the convergence fit is a hand-rolled log-log least
squares -- deliberately the same estimator as
verification_and_validation/common/convergence.py::fit_convergence_rate, which the plan
asked to reuse and which cannot be imported here because it needs numpy -- and the plots
are hand-emitted inline SVG, following python/report_cvfem_bench.py, which made the same
call for the same reason.

HTML comes from `markdown_py`, per --html. It is not bundled either, so a missing one is
reported as a skipped step and never as a failure: the Markdown is the artifact, and the
HTML is a convenience rendered from it.
"""

import argparse
import datetime
import html
import json
import math
import os
import re
import shutil
import subprocess
import sys

# --------------------------------------------------------------------------- parsing

# One pattern per FIELD, not per line, and each anchored on the literal label the driver
# prints beside it. Per line would be shorter and is what this started as; it broke on the
# first real log, because the driver writes to stdout and stderr and a run that redirects
# both into one file gets them interleaved -- "newton_converged: 0 newton_it: 8 (last
# stage) newton_total: 8 oververification failed (...)" is an actual line from one. A
# per-field pattern reads what survived and leaves the rest absent, rather than losing
# every field on the line to one interruption.
FIELDS = {
    "nnodes":       (r"\bnnodes:\s+(\d+)", int),
    "nelements":    (r"\bnelements:\s+(\d+)", int),
    "ndof":         (r"\bndof:\s+(\d+)", int),
    "u_l2":         (r"\bu_l2\s+(\S+)", float),
    "p_l2":         (r"\bp_l2\s+(\S+)", float),
    "p_l2_shifted": (r"\bp_l2_shifted\s+(\S+)", float),
    "u_linf":       (r"\bu_linf:\s+(\S+)", float),
    "p_linf":       (r"\bp_linf:\s+(\S+)", float),
    "mass_sum":     (r"sum of continuity residual\s+(\S+)", float),
    "mass_abs":     (r"sum \|\.\|\s+(\S+),", float),
    "newton_it":    (r"\bnewton_it:\s+(\d+)", int),
    "lin_it_total": (r"\blin_it_total:\s+(\d+)", int),
    "t_operator":   (r"\bt_operator:\s+(\S+) s", float),
    "t_precond":    (r"\bt_precond:\s+(\S+) s", float),
    "t_solve":      (r"\bt_solve:\s+(\S+) s", float),
    "re_reached":   (r"highest Re SOLVED = (\S+) of", float),
    "re_target":    (r"highest Re SOLVED = \S+ of (\S+) target", float),
    "bc_faces":     (r"^(?:pressure|traction|natural outflow) \(flat\):\s+(\d+) faces", int),
    "p_bar":        (r"p_bar = (\S+)", float),
    "skin_nodes":   (r"skin nodes (\d+) of", int),
    # The driver's own flux oracle, which is a better conservation statement than the
    # residual sum: it compares what entered against what left, both measured, against the
    # closed-form inflow. The residual sum can be small because nothing is happening.
    "inflow":       (r"inflow flux\s+(\S+)", float),
    "inflow_exact": (r"inflow flux\s+\S+\s+\(exact\s+(\S+),", float),
    "inflow_err":   (r"inflow flux\s+\S+\s+\(exact\s+\S+, err\s+(\S+)\)", float),
    "outflow":      (r"outflow flux\s+(\S+)", float),
    "imbalance":    (r"imbalance \(out-in\)\s+(\S+)", float),
    "imbalance_rel": (r"imbalance \(out-in\)\s+\S+\s+relative\s+(\S+)", float),
    # The diaphragm pump. swept is what the prescribed normal velocity displaces; the port
    # is what actually left. They are the same number or transpiration is not doing what it
    # claims. The last v_diaphragm on the log is the waveform at the instant measured.
    "pump_swept":    (r"^pump: swept\s+(\S+)", float),
    "pump_port":     (r"^pump: swept\s+\S+\s+port\s+(\S+)", float),
    "pump_diaph":    (r"^pump: swept\s+\S+\s+port\s+\S+\s+diaphragm\s+(\S+)", float),
    "pump_err":      (r"^pump: \|port - swept\|\s+(\S+)", float),
    "pump_balance":  (r"^pump: \|port - swept\|\s+\S+\s+\|port \+ diaphragm\|\s+(\S+)", float),
    "pump_area":     (r"^pump: chamber \S+\s+diaphragm area\s+(\S+)", float),
    "pump_faces":    (r"^pump: chamber .*port\s+(\d+) faces", int),
}

# Fields whose value is text rather than a number.
TEXT_FIELDS = {
    "gauge":      r"pressure gauge:\s+(.*?)\s+\(",
    "bc_sideset": r"faces from sideset '([^']+)'",
    "re_note":    r"highest Re SOLVED = \S+ of \S+ target\s+\((.*)\)",
}


def parse_log(path):
    """Pull every recognised quantity out of one driver log."""
    out = {"log": os.path.basename(path)}
    if not os.path.exists(path):
        out["missing"] = True
        return out
    with open(path, "r", errors="replace") as fh:
        text = fh.read()
    for key, (pat, cast) in FIELDS.items():
        m = re.search(pat, text, re.MULTILINE)
        if m:
            try:
                out[key] = cast(m.group(1))
            except ValueError:
                pass
    for key, pat in TEXT_FIELDS.items():
        m = re.search(pat, text, re.MULTILINE)
        if m:
            out[key] = m.group(1).strip()
    # `newton_converged: 1` is the only place the driver states convergence, and it is the
    # field most often clipped by interleaving, so it is read on its own and left absent
    # rather than defaulted -- "did not converge" and "did not say" are different claims.
    vs = re.findall(r"^pump: t = (\S+)\s+v_diaphragm = (\S+)", text, re.MULTILINE)
    if vs:
        out["pump_t"], out["pump_v"] = float(vs[-1][0]), float(vs[-1][1])
    m = re.search(r"\bnewton_converged:\s+(\d+)", text)
    if m:
        out["converged"] = m.group(1) == "1"
    if re.search(r"^traction \(flat\):", text, re.MULTILINE):
        out["bc"] = "traction"
    elif re.search(r"^pressure \(flat\):", text, re.MULTILINE):
        out["bc"] = "pressure"
    elif re.search(r"^natural outflow \(flat\):", text, re.MULTILINE):
        out["bc"] = "natural"
    return out


# --------------------------------------------------------------------------- fitting

def fit_convergence_rate(scales, errors):
    """Log-log least squares, returning (rate, intercept, r_squared, n).

    Deliberately the same estimator as the parent repository's
    verification_and_validation/common/convergence.py::fit_convergence_rate -- ordinary
    least squares on (log h, log e) with the same R-squared definition -- reimplemented
    only because that one imports numpy, which neither the Alps uenv nor the development
    laptop has. It is the closed-form OLS solution, which is what numpy's lstsq on the
    same [x, 1] design matrix returns, so the two agree by construction rather than by
    coincidence. `--selftest` checks it against power laws whose exponent is known.
    """
    pairs = [(s, e) for s, e in zip(scales, errors)
             if s and e and s > 0 and e > 0 and math.isfinite(s) and math.isfinite(e)]
    if len(pairs) < 2:
        return None
    xs = [math.log(s) for s, _ in pairs]
    ys = [math.log(e) for _, e in pairs]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    rate = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - rate * mx
    resid = sum((y - (rate * x + intercept)) ** 2 for x, y in zip(xs, ys))
    total = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 if total <= 1e-300 else 1.0 - resid / total
    return rate, intercept, r2, n


# --------------------------------------------------------------------------- SVG

# Figures carry their palette as CSS custom properties so they follow docs/style.cscs.css,
# and every one of them names a literal fallback -- var(--s1, #d97757) -- so a figure still
# draws correctly in a Markdown viewer, or in any HTML that was produced without the
# stylesheet beside it. There is no inline <style> block: presentation belongs in the
# stylesheet, and a raw <style> at the top of a .md file is noise to every Markdown reader.
SERIES_FALLBACK = ["#d97757", "#8c8880", "#4a6fa5", "#a8a29a"]
SERIES_COLORS = ["var(--s%d, %s)" % (i + 1, c) for i, c in enumerate(SERIES_FALLBACK)]


def _nice_ticks(lo, hi, count=5):
    if hi <= lo:
        return [lo]
    span = hi - lo
    step = 10 ** math.floor(math.log10(span / max(count - 1, 1)))
    for mult in (1, 2, 2.5, 5, 10):
        if span / (step * mult) <= count:
            step *= mult
            break
    first = math.ceil(lo / step) * step
    ticks, t = [], first
    while t <= hi + step * 1e-9:
        ticks.append(t)
        t += step
    return ticks


def _decade_ticks(lo, hi, want=5):
    """Ticks for a log axis, in exponent space.

    _nice_ticks on the exponent produces fractional exponents, and labelling those as
    "1e%d" rounds several of them to the same string -- an axis reading 1e-2, 1e-1, 1e-1,
    1e-1 across four gridlines, which is worse than no labels. A log axis gets whole
    decades, thinned when the range spans more than a handful.
    """
    first, last = math.floor(lo), math.ceil(hi)
    step = max(1, int(math.ceil((last - first) / float(want))))
    return [v for v in range(first, last + 1, step) if lo <= v <= hi]


def svg_xy(series, xlabel, ylabel, logx=False, logy=False, width=720, height=330,
           fmt_x="%.3g", fmt_y="%.3g", caption=None):
    """A scatter/line figure. `series` is [(name, [(x, y), ...], style), ...].

    style is "line", "dash" (a fitted or reference line, drawn without markers) or
    "points". Log axes take the log before scaling, which is what makes a power law a
    straight line and its slope readable as the convergence rate.
    """
    pad_l, pad_r, pad_t, pad_b = 68, 16, 16, 46
    pts = [(x, y) for _, data, _ in series for x, y in data]
    if not pts:
        return ""
    tx = (lambda v: math.log10(v)) if logx else (lambda v: v)
    ty = (lambda v: math.log10(v)) if logy else (lambda v: v)
    xs = [tx(x) for x, _ in pts]
    ys = [ty(y) for _, y in pts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if x1 - x0 < 1e-12:
        x0, x1 = x0 - 0.5, x1 + 0.5
    if y1 - y0 < 1e-12:
        y0, y1 = y0 - 0.5, y1 + 0.5
    mx, my = (x1 - x0) * 0.06, (y1 - y0) * 0.10
    x0, x1, y0, y1 = x0 - mx, x1 + mx, y0 - my, y1 + my
    px = lambda v: pad_l + (tx(v) - x0) / (x1 - x0) * (width - pad_l - pad_r)
    py = lambda v: height - pad_b - (ty(v) - y0) / (y1 - y0) * (height - pad_t - pad_b)

    o = ['<svg class="cvfig" viewBox="0 0 %d %d" width="100%%" role="img">' % (width, height)]
    if caption:
        o.append('<title>%s</title>' % html.escape(caption))

    for v in _decade_ticks(y0, y1) if logy else _nice_ticks(y0, y1):
        val = 10 ** v if logy else v
        yy = py(val)
        if not (pad_t - 1 <= yy <= height - pad_b + 1):
            continue
        o.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="var(--grid, #e4e1d8)" stroke-width="1"/>'
                 % (pad_l, yy, width - pad_r, yy))
        lab = ("1e%d" % round(v)) if logy else (fmt_y % val)
        o.append('<text class="mut" x="%d" y="%.1f" text-anchor="end">%s</text>'
                 % (pad_l - 8, yy + 4, html.escape(lab)))
    for v in _decade_ticks(x0, x1) if logx else _nice_ticks(x0, x1):
        val = 10 ** v if logx else v
        xx = px(val)
        if not (pad_l - 1 <= xx <= width - pad_r + 1):
            continue
        o.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="var(--grid, #e4e1d8)" stroke-width="1"/>'
                 % (xx, pad_t, xx, height - pad_b))
        lab = ("1e%d" % round(v)) if logx else (fmt_x % val)
        o.append('<text class="mut" x="%.1f" y="%d" text-anchor="middle">%s</text>'
                 % (xx, height - pad_b + 18, html.escape(lab)))

    o.append('<text class="mut" x="%.1f" y="%d" text-anchor="middle">%s</text>'
             % ((pad_l + width - pad_r) / 2, height - 6, html.escape(xlabel)))
    o.append('<text class="mut" x="%d" y="%.1f" text-anchor="middle" transform="rotate(-90 14 %.1f)">%s</text>'
             % (14, (pad_t + height - pad_b) / 2, (pad_t + height - pad_b) / 2, html.escape(ylabel)))

    for i, (name, data, style) in enumerate(series):
        if not data:
            continue
        col = SERIES_COLORS[i % len(SERIES_COLORS)]
        pathd = " ".join(("M" if k == 0 else "L") + "%.1f %.1f" % (px(x), py(y))
                         for k, (x, y) in enumerate(sorted(data)))
        dash = ' stroke-dasharray="6 4"' if style == "dash" else ""
        if style != "points":
            o.append('<path d="%s" fill="none" stroke="%s" stroke-width="2"%s/>' % (pathd, col, dash))
        if style != "dash":
            for x, y in data:
                o.append('<circle cx="%.1f" cy="%.1f" r="3.5" fill="%s" stroke="var(--surface, #ffffff)" '
                         'stroke-width="1.5"/>' % (px(x), py(y), col))
        o.append('<text x="%d" y="%d" fill="%s">%s</text>'
                 % (pad_l + 8, pad_t + 14 + i * 16, col, html.escape(name)))
    o.append("</svg>")
    return "\n".join(o)


# --------------------------------------------------------------------------- report

def fmt(v, spec="%.3e", dash="--"):
    return dash if v is None else (spec % v)


def status(ok):
    # A span with a class rather than bold text: the stylesheet turns these into badges, and
    # Markdown passes inline HTML through, so the same string is readable either way.
    if ok is True:
        return '<span class="st-pass">pass</span>'
    if ok is False:
        return '<span class="st-fail">FAIL</span>'
    return '<span class="st-idle">n/a</span>'


def note(text):
    """A verdict that is not a verdict -- "not converged" and the like."""
    return '<span class="st-idle">%s</span>' % html.escape(text)


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out) + "\n"


def section_mms(runs, checks):
    """Spatial order of accuracy. The rate is the claim; the errors are the evidence."""
    rows = [r for r in runs if r["group"] == "mms" and "u_l2" in r]
    if len(rows) < 2:
        return ""
    rows.sort(key=lambda r: r["ndof"])
    # h ~ ndof^(-1/3) in 3D, which is the scale the rate is quoted against.
    for r in rows:
        r["h"] = r["ndof"] ** (-1.0 / 3.0)

    body = ["## Spatial order of accuracy", "",
            "Manufactured solution, volume-weighted L2 against the exact field, with the",
            "pressure gauge offset removed (`p_l2_shifted`). The rate is fitted by log-log",
            "least squares against `h ~ ndof^(-1/3)`.", ""]
    body.append(table(
        ["ndof", "h", "u L2", "p L2", "p L2 shifted"],
        [[r["ndof"], "%.4f" % r["h"], fmt(r.get("u_l2")), fmt(r.get("p_l2")),
          fmt(r.get("p_l2_shifted"))] for r in rows]))

    fits, series = [], []
    for key, label, expect in (("u_l2", "u L2", 2.0), ("p_l2_shifted", "p L2 shifted", 1.3)):
        data = [(r["h"], r[key]) for r in rows if r.get(key)]
        f = fit_convergence_rate([d[0] for d in data], [d[1] for d in data])
        if not f:
            continue
        rate, inter, r2, n = f
        ok = rate >= expect - 0.35
        fits.append([label, "%.3f" % rate, "%.4f" % r2, n, "&ge; %.2f" % (expect - 0.35), status(ok)])
        checks.append(("Spatial order, %s" % label, "rate %.3f (expect >= %.2f)" % (rate, expect - 0.35), ok))
        series.append((("%s: rate %.2f" % (label, rate)), data, "points"))
        lo, hi = min(d[0] for d in data), max(d[0] for d in data)
        series.append((("  fit h^%.2f" % rate),
                       [(lo, math.exp(inter) * lo ** rate), (hi, math.exp(inter) * hi ** rate)], "dash"))
    if fits:
        body.append(table(["quantity", "fitted rate", "R^2", "levels", "threshold", "status"], fits))
        body.append(svg_xy(series, "h  (~ ndof^-1/3)", "L2 error", logx=True, logy=True,
                           caption="MMS convergence") + "\n")
    return "\n".join(body) + "\n"


def section_boundary(runs, checks):
    """The boundary conditions, judged on identities rather than on eyeballed plots."""
    body = ["## Boundary conditions", ""]

    nat = next((r for r in runs if r["group"] == "bc" and r.get("label") == "natural"), None)
    tr0 = next((r for r in runs if r["group"] == "bc" and r.get("label") == "traction0"), None)
    if nat and tr0 and "u_linf" in nat and "u_linf" in tr0:
        du = abs(nat["u_linf"] - tr0["u_linf"])
        dp = abs(nat["p_linf"] - tr0["p_linf"])
        ok = du <= 1e-12 and dp <= 1e-12
        body += ["### Traction generalises the do-nothing outflow", "",
                 "A prescribed traction of zero must reproduce the do-nothing outflow exactly,",
                 "not approximately: the two take the same kernel branch and add a term that is",
                 "identically zero. Anything else means they are separate mechanisms that happen",
                 "to agree.", "",
                 table(["quantity", "do-nothing", "traction t = 0", "difference"],
                       [["u_linf", fmt(nat["u_linf"]), fmt(tr0["u_linf"]), fmt(du)],
                        ["p_linf", fmt(nat["p_linf"]), fmt(tr0["p_linf"]), fmt(dp)]])]
        checks.append(("Traction t=0 equals the do-nothing outflow",
                       "max diff %.2e" % max(du, dp), ok))

    ports = sorted((r for r in runs if r["group"] == "port" and "p_bar" in r and "p_linf" in r),
                   key=lambda r: r["p_bar"])
    if len(ports) >= 2:
        p_exact = ports[0].get("p_exact_outlet")
        body += ["", "### A pressure port fixes the level and nothing else", "",
                 "Holding a port at `p_bar` should shift the whole pressure field by",
                 "`p_bar - p_exact(outlet)` and leave the velocity alone. Both halves are",
                 "checked: the shift against that closed form, and the velocity against the",
                 "exact solution.", ""]
        rows, resid, bad, unconverged = [], [], 0, 0
        for r in ports:
            pred = None if p_exact is None else r["p_bar"] - p_exact
            err = None if pred is None else abs(r["p_linf"] - abs(pred))
            conv = r.get("converged")
            if conv is False:
                # Reported, never scored. A run that did not converge says nothing about
                # the boundary condition, and counting it against one would be the same
                # mistake as counting it for one. The solver table below carries the
                # detail, and the evidence line says how many landed here.
                unconverged += 1
                st = note("not converged")
            else:
                ok = (err is not None and err <= 1e-6) and r.get("u_linf", 1) <= 1e-6
                st = status(ok)
                if not ok:
                    bad += 1
            rows.append(["%g" % r["p_bar"], fmt(r.get("u_linf")), fmt(r["p_linf"]),
                         fmt(abs(pred)) if pred is not None else "--", fmt(err), st])
            if pred is not None and conv is not False:
                resid.append((r["p_bar"], r["p_linf"]))
        body.append(table(["p_bar", "u_linf", "p_linf", "predicted shift", "error", "status"], rows))
        n_scored = len(ports) - unconverged
        # A sweep in which nothing converged is not evidence of anything, so it fails.
        all_ok = bad == 0 and n_scored >= 2
        checks.append(("Pressure port shifts the level by p_bar - p_exact",
                       "%d of %d values verified%s" % (
                           n_scored - bad, len(ports),
                           "" if not unconverged else ", %d did not converge" % unconverged),
                       all_ok))
        if resid:
            pred_line = [(r["p_bar"], abs(r["p_bar"] - p_exact)) for r in ports]
            body.append(svg_xy([("measured p_linf", resid, "points"),
                                ("predicted |p_bar - p_exact|", pred_line, "dash")],
                               "prescribed p_bar", "pressure offset",
                               caption="Pressure port linearity") + "\n")
    return "\n".join(body) + "\n" if len(body) > 2 else ""


def section_pump(runs, checks):
    """The diaphragm pump, judged on an identity rather than on a solution."""
    rows = [r for r in runs if r["group"] == "pump" and "pump_swept" in r]
    if not rows:
        return ""
    rows.sort(key=lambda r: r.get("pump_t", 0))
    body = ["## Diaphragm pump", "",
            "A closed chamber with a diaphragm that moves and one port that lets fluid in or",
            "out. The diaphragm is transpiration on a fixed mesh: the wall does not move, its",
            "normal velocity is prescribed through it. That is exact for the mass it carries",
            "and silent about the geometric nonlinearity a real diaphragm has, which an ALE",
            "formulation would capture and this deliberately does not.", "",
            "It is checked on an identity, not against a solution, because it has none. The",
            "chamber is fixed and the flow incompressible, so the flux through its closed",
            "boundary is zero; the walls carry none and the diaphragm's velocity is",
            "prescribed. So the port must carry exactly what the diaphragm sweeps,",
            "`rho V Lx Lz`, and if transpiration is moving the wrong mass the identity fails",
            "by exactly that much. Both fluxes are integrated on the operator's own boundary",
            "sub-control surfaces, so this measures the discretisation rather than a second",
            "quadrature's opinion of it.", "",
            "There is no valve, so the pump does not rectify: over a full cycle it moves",
            "fluid back and forth and nets nothing. That is the scope, not an oversight.", ""]
    out, bad, unconverged = [], 0, 0
    for r in rows:
        err, bal = r.get("pump_err"), r.get("pump_balance")
        if r.get("converged") is False:
            unconverged += 1
            st = note("not converged")
        else:
            ok = err is not None and err <= 1e-12 and bal is not None and bal <= 1e-12
            st = status(ok)
            if not ok:
                bad += 1
        out.append([r.get("label", r["log"]), fmt(r.get("pump_t"), "%.3f"), fmt(r.get("pump_v"), "%+.4f"),
                    fmt(r.get("pump_swept"), "%+.9f"), fmt(r.get("pump_port"), "%+.9f"),
                    fmt(err), fmt(bal), st])
    # Column names without absolute-value bars: a "|" inside a Markdown table cell is a
    # column separator, and "|port - swept|" silently split the header into two extra empty
    # columns while the body rows kept eight, which renders as a table with its headings
    # shifted one place left of the numbers they name.
    body.append(table(["run", "t", "v diaphragm", "swept", "port flux",
                       "abs(port - swept)", "abs(port + diaphragm)", "status"], out))
    n_scored = len(rows) - unconverged
    checks.append(("Pump: the port carries what the diaphragm sweeps",
                   "%d of %d instant(s) verified%s" % (
                       n_scored - bad, len(rows),
                       "" if not unconverged else ", %d did not converge" % unconverged),
                   bad == 0 and n_scored >= 1))

    pts = [(r["pump_v"], r["pump_port"]) for r in rows
           if r.get("pump_v") is not None and r.get("pump_port") is not None]
    if len(pts) >= 2:
        area = rows[0].get("pump_area") or 1.0
        line = [(v, v * area) for v, _ in pts]
        body.append(svg_xy([("port flux", pts, "points"), ("rho V Lx Lz", line, "dash")],
                           "prescribed diaphragm velocity", "flux through the port",
                           caption="Pump: swept volume against port flux") + "\n")
    return "\n".join(body) + "\n"


def section_conservation(runs, checks):
    rows = [r for r in runs if "mass_sum" in r or "imbalance_rel" in r]
    if not rows:
        return ""
    body = ["## Global mass conservation", "",
            "Summed over every node, the interior sub-control-surface fluxes telescope away",
            "and the boundary closure is all that is left, so the total continuity residual",
            "is the net mass imbalance of the domain and should be zero. Zero needs a scale",
            "to be meaningful, and the scale is the exact volumetric inflow -- 1/9 for the",
            "backward-facing step -- so what is checked is the ratio.", "",
            "**The plane-integrated flux imbalance is reported and deliberately not scored.**",
            "docs/CVFEM_Verification_Farrell.md measured it at 17% on a case whose residual",
            "sum was 7e-14, and the 17% is trapezoidal error rather than lost mass: the inlet",
            "is a smooth parabola and the outlet profile is developing and recirculating. The",
            "residual sum is quadrature-free and is the conservation test; the inflow figure",
            "is kept because it does check the prescribed profile, against its own tolerance.", ""]
    out, bad, unconverged = [], 0, 0
    for r in rows:
        exact = r.get("mass_exact") or r.get("inflow_exact")
        # Two independent statements, and both must hold. The residual sum says the
        # discrete equations balance; the measured out-minus-in says the flow the solution
        # actually carries balances. A scheme can satisfy the first and violate the second.
        rel_res = None if not exact or "mass_sum" not in r else abs(r["mass_sum"]) / exact
        # The prescribed inlet profile, against a tolerance of its own: it is a trapezoidal
        # integral of a parabola, which the reference measured 3% low against the exact 1/9,
        # so holding it to round-off would fail a correct inlet.
        inflow_ok = r.get("inflow_err") is None or r["inflow_err"] <= 0.05
        if r.get("converged") is False:
            unconverged += 1
            st = note("not converged")
        else:
            ok = rel_res is not None and rel_res <= 1e-9 and inflow_ok
            st = status(ok)
            if not ok:
                bad += 1
        out.append([r.get("label", r["log"]), r.get("ndof", "--"),
                    fmt(r.get("mass_sum")), fmt(rel_res),
                    fmt(r.get("inflow")), fmt(r.get("inflow_err")),
                    fmt(r.get("imbalance_rel")), st])
    body.append(table(["case", "ndof", "sum of continuity", "relative to inflow", "inflow",
                       "inflow err", "flux imbalance (not scored)", "status"], out))
    if unconverged:
        body += ["", "%d of these run(s) did not reach the nonlinear tolerance, so nothing "
                     "above is a statement about conservation for them -- the residual sum of "
                     "an unconverged iterate measures where the solver stopped, not whether "
                     "the scheme conserves mass. The solver table below carries the detail." % unconverged, ""]
    n_scored = len(rows) - unconverged
    checks.append(("Global mass balance",
                   "%d of %d case(s) verified%s" % (
                       n_scored - bad, len(rows),
                       "" if not unconverged else ", %d did not converge" % unconverged),
                   bad == 0 and n_scored >= 1))
    return "\n".join(body) + "\n"


def section_solver(runs):
    rows = [r for r in runs if "converged" in r]
    if not rows:
        return ""
    body = ["## Solver behaviour", "",
            "Reported for context, not asserted here: a verification result from a run that",
            "did not converge is not a verification result. Timings carry the dof count they",
            "were measured on, and the machine is named in Provenance.", ""]
    return "\n".join(body) + "\n" + table(
        ["case", "ndof", "converged", "Newton", "linear its", "t_solve (s)", "Re reached", "gauge"],
        [[r.get("label", r["log"]), r.get("ndof", "--"), "yes" if r["converged"] else "no",
          r.get("newton_it", "--"), r.get("lin_it_total", "--"),
          fmt(r.get("t_solve"), "%.3f"), fmt(r.get("re_reached"), "%g"),
          r.get("gauge", "--")] for r in rows])


def section_unit(manifest, checks):
    ct = manifest.get("ctest")
    if not ct:
        return ""
    ok = ct.get("failed", 1) == 0
    checks.append(("Unit tests (ctest)", "%d of %d passed" % (ct.get("passed", 0), ct.get("total", 0)), ok))
    body = ["## Unit tests", "",
            "The kernel- and operator-level checks that do not need a solve. These are the",
            "layer that names a line when it fails, so they run first and the rest of this",
            "report is only meaningful if they pass.", "",
            table(["total", "passed", "failed", "status"],
                  [[ct.get("total", "--"), ct.get("passed", "--"), ct.get("failed", "--"), status(ok)]])]
    if ct.get("failing"):
        body.append("Failing: " + ", ".join("`%s`" % f for f in ct["failing"]) + "\n")
    return "\n".join(body) + "\n"


def build_report(manifest, rundir):
    runs = []
    for spec in manifest.get("runs", []):
        rec = parse_log(os.path.join(rundir, spec["log"]))
        rec.update({k: v for k, v in spec.items() if k != "log"})
        runs.append(rec)

    checks = []
    parts = [section_unit(manifest, checks),
             section_mms(runs, checks),
             section_boundary(runs, checks),
             section_pump(runs, checks),
             section_conservation(runs, checks)]

    head = ["# CVFEM verification report", "",
            manifest.get("subtitle",
                         "Every claim below is an identity or a fitted rate checked against a "
                         "threshold, so this page states whether the code is right rather than "
                         "inviting a reader to judge a plot."), "",
            "## Summary", ""]
    n_fail = sum(1 for _, _, ok in checks if ok is False)
    head.append("**%d of %d checks pass.**%s\n" % (
        len(checks) - n_fail, len(checks),
        "" if n_fail == 0 else "  %d FAILING -- see the sections below." % n_fail))
    head.append(table(["check", "evidence", "status"],
                      [[c, e, status(ok)] for c, e, ok in checks]) if checks else
                "_No checks were produced; the run directory has no recognised logs._\n")

    tail = [section_solver(runs), "## Provenance", "",
            table(["field", "value"],
                  [["generated", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                   ["machine", manifest.get("machine", "--")],
                   ["threads", manifest.get("threads", "--")],
                   ["commit", manifest.get("commit", "--")],
                   ["linear solver", manifest.get("solver", "--")],
                   ["element refine level", manifest.get("level", "--")],
                   ["run directory", os.path.abspath(rundir)],
                   ["runs parsed", "%d of %d" % (sum(1 for r in runs if not r.get("missing")),
                                                 len(runs))]]),
            "Regenerate with `python3 python/cvfem_verify_report.py %s`.\n" % rundir]

    # Sections are joined with a blank line between them and never filtered on
    # truthiness: the separators ARE empty strings, and dropping them ran every heading
    # into the paragraph above it, which Markdown then refused to treat as a heading.
    doc = "\n".join(head) + "\n" + "\n".join(p for p in parts if p.strip()) + "\n" + \
          "\n".join(t for t in tail if t is not None)
    return doc, n_fail


def selftest():
    """Check the machinery this report's conclusions rest on, since a report generator
    that quietly fits the wrong slope is worse than no report at all."""
    fails = []

    def ck(ok, what):
        print("%-58s %s" % (what, "OK" if ok else "FAIL"))
        if not ok:
            fails.append(what)

    # Exact power laws: the fit must return the exponent it was built from.
    for expect in (1.0, 1.3, 2.0, 3.0):
        hs = [1.0, 0.5, 0.25, 0.125]
        es = [7.3 * h ** expect for h in hs]
        rate, inter, r2, n = fit_convergence_rate(hs, es)
        ck(abs(rate - expect) < 1e-10 and abs(r2 - 1.0) < 1e-10 and n == 4,
           "fit recovers h^%.1f exactly" % expect)
        ck(abs(math.exp(inter) - 7.3) < 1e-9, "fit recovers the h^%.1f prefactor" % expect)

    # Noise must lower R-squared without moving the rate much, which is what makes
    # R-squared worth printing beside the rate rather than instead of it.
    hs = [1.0, 0.5, 0.25, 0.125]
    es = [1.0, 0.26, 0.062, 0.0158]
    rate, _, r2, _ = fit_convergence_rate(hs, es)
    ck(1.9 < rate < 2.1 and 0.99 < r2 < 1.0, "a noisy second-order ladder reads as ~2")

    # Degenerate input must be refused rather than fitted.
    ck(fit_convergence_rate([1.0], [1.0]) is None, "a single point is not a rate")
    ck(fit_convergence_rate([1.0, 2.0], [1.0, 0.0]) is None, "a zero error is refused")
    ck(fit_convergence_rate([1.0, 1.0], [1.0, 2.0]) is None, "repeated scales are refused")

    # The parser must survive the interleaving that broke the first version.
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "x.log")
        with open(log, "w") as fh:
            fh.write("nnodes: 8281  nelements: 6912  ndof: 33124\n"
                     "mms_err: u_l2 1.5e-03  p_l2 2.5e-02  p_l2_shifted 3.5e-03  (dp_mean 1e-9, vol 4.0)\n"
                     "newton_converged: 0  newton_it: 8 (last stage)  newton_total: 8 "
                     "oververification failed (converged=0, u_linf=1.7e-01, tol=0.01)\n"
                     "u_linf: 1.764292e-01  p_linf: 2.140081e-01\n"
                     "pressure (flat): 144 faces from sideset 'outlet', p_bar = 1.5\n")
        r = parse_log(log)
    ck(r.get("ndof") == 33124 and r.get("u_l2") == 1.5e-03 and r.get("p_l2") == 2.5e-02
       and r.get("p_l2_shifted") == 3.5e-03, "every labelled field is read")
    ck(r.get("newton_it") == 8 and r.get("converged") is False,
       "a line clipped by stdout/stderr interleaving still parses")
    ck(r.get("p_bar") == 1.5 and r.get("bc") == "pressure" and r.get("bc_sideset") == "outlet",
       "the boundary condition is identified")

    # A figure must be emitted and be well-formed enough to survive into HTML.
    fig = svg_xy([("s", [(1.0, 1.0), (0.5, 0.25)], "points")], "h", "e", logx=True, logy=True)
    ck(fig.startswith("<svg") and fig.rstrip().endswith("</svg>") and "circle" in fig,
       "svg_xy emits a closed figure")
    ck(svg_xy([("s", [], "points")], "h", "e") == "", "an empty series draws nothing")

    print("\n%d check(s) failed" % len(fails) if fails else "\nall self-tests passed")
    return 1 if fails else 0


DOC = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>%(title)s</title>
<link rel="stylesheet" href="style.cscs.css">
</head>
<body>
<div class="report-head">
  <span class="mark">CVFEM</span>
  <span>verification</span>
  %(facts)s
</div>
%(body)s
</body>
</html>
"""


def write_html(target, fragment, manifest):
    """Wrap markdown_py's fragment in a document that links the stylesheet.

    The masthead carries what a reader needs to know before believing a number -- which
    machine, how many threads, which commit -- rather than leaving it to the Provenance
    table at the very bottom.
    """
    facts = []
    for label, key in (("machine", "machine"), ("threads", "threads"), ("commit", "commit")):
        v = manifest.get(key)
        if v not in (None, "", "--"):
            facts.append("<span>%s <dfn>%s</dfn></span>" % (label, html.escape(str(v))))
    facts.append("<span>%s</span>" % datetime.datetime.now().strftime("%Y-%m-%d"))
    with open(target, "w") as fh:
        fh.write(DOC % {"title": "CVFEM verification report",
                        "facts": "\n  ".join(facts),
                        "body": fragment})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rundir", nargs="?", help="directory holding manifest.json and the run logs")
    ap.add_argument("--selftest", action="store_true",
                    help="check the fitter, the parser and the plotter, and exit")
    ap.add_argument("-o", "--output", default=None, help="Markdown file to write")
    ap.add_argument("--html", action="store_true", help="also render HTML with markdown_py")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any check fails, for use as a gate")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    if not args.rundir:
        ap.error("a run directory is required (or --selftest)")

    mpath = os.path.join(args.rundir, "manifest.json")
    if not os.path.exists(mpath):
        sys.exit("no manifest.json in %s -- run scripts/verify_report.sh first" % args.rundir)
    with open(mpath) as fh:
        manifest = json.load(fh)

    md, n_fail = build_report(manifest, args.rundir)
    out = args.output or os.path.join(args.rundir, "report.md")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as fh:
        fh.write(md)
    print("wrote %s" % out)

    if args.html:
        target = os.path.splitext(out)[0] + ".html"
        exe = shutil.which("markdown_py") or shutil.which("markdown_py3")
        if not exe:
            # Not a failure: markdown_py is not bundled and is absent from the Alps uenv.
            # The Markdown is the artifact; the HTML is a rendering of it.
            print("markdown_py not found; wrote Markdown only. To render:\n"
                  "  pip install --user markdown && markdown_py -x tables -o html %s -f %s"
                  % (out, target))
        else:
            # `tables` is the one that matters: every result on this page is a table, and
            # without it they render as paragraphs of pipes. The inline SVG needs no
            # extension -- Markdown passes block-level HTML through untouched.
            #
            # Extension sets are tried strongest first and the first that loads wins,
            # because markdown_py versions differ in what they ship: md_in_html arrived in
            # Markdown 3.3, and the markdown_py on this laptop is a Python 2.7 build that
            # fails to import it outright.
            frag, rc = None, 1
            for exts in (["tables", "md_in_html"], ["tables"], []):
                argv = [exe]
                for e in exts:
                    argv += ["-x", e]
                argv += [out]
                try:
                    frag = subprocess.check_output(argv, stderr=subprocess.DEVNULL).decode("utf-8")
                    rc = 0
                    break
                except subprocess.CalledProcessError:
                    continue
            if rc != 0:
                print("markdown_py failed for every extension set; the Markdown stands on its own")
            else:
                # markdown_py emits a bare fragment. Wrap it in a real document so the page
                # has a title, a charset, a viewport and the stylesheet -- without which the
                # tables are unreadable and the figures have no palette.
                css_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs",
                                       "style.cscs.css")
                css_dst = os.path.join(os.path.dirname(os.path.abspath(target)), "style.cscs.css")
                if os.path.exists(css_src) and os.path.abspath(css_src) != os.path.abspath(css_dst):
                    shutil.copyfile(css_src, css_dst)
                write_html(target, frag, manifest)
                print("wrote %s%s" % (target, "" if exts else "  (no extensions)"))

    if args.strict and n_fail:
        sys.exit("%d verification check(s) failed" % n_fail)


if __name__ == "__main__":
    main()
