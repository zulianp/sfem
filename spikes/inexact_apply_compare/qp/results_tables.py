"""The paper's results tables, from the Grace logs in measured/ and the H100 logs in measured_gpu/.

    python results_tables.py measured/ ../../../docs/compressed_partial_assembly/tables \
        measured_qsym/ measured_qbsr/
    python results_tables.py --machine gpu measured_gpu/ ../../../docs/compressed_partial_assembly/tables

Reads the `bench_qp` logs (one per element; a packed-only log may be split off) and the
`newton.sbatch` logs, and writes LaTeX table bodies and macros, so no measured number is typed
into the article.  The further directories hold the quantized-BSR campaigns: their benchmarks add
the quantized BSR rows and their Newton logs the quantized BSR operators (in the order given, the
first one wins); the references (BSR, the stores, the element matrices) stay those of measured/.
"""

import glob
import os
import re
import sys

ELEMENTS = ("TET4", "TET10", "HEX8", "PROTEUS_HEX27")
# Macro suffixes, and the names the article prints.
TAGS = {"TET4": "TetFour", "TET10": "TetTen", "HEX8": "HexEight", "PROTEUS_HEX27": "HexTwentySeven"}
NAMES = {"PROTEUS_HEX27": "HEX27"}


def parse_bench(paths):
    """rows[element][operator] = dict(setup_ms, mdofs, bytes_per_dof, l2, energy); meta[element]."""
    rows, meta = {}, {}
    for path in paths:
        el = None
        for line in open(path):
            # Grace: "<k> threads"; the GPU: the device's name.
            m = re.match(r"neohookean_ogden (\w+): n=(\d+), (\d+) elements, (\d+) nodes, (\d+) dof, (?:(\d+) threads|([^,]+))", line)
            if m:
                el = m.group(1)
                meta[el] = dict(n=int(m.group(2)), elements=int(m.group(3)), dof=int(m.group(5)),
                                threads=int(m.group(6)) if m.group(6) else None, device=m.group(7))
                rows.setdefault(el, {})
                continue
            m = re.match(r"\s{2}(\S.*?)\s{2,}([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\S+)\s+(\S+)\s*$", line)
            if el and m and m.group(1) != "operator":
                rows[el][m.group(1).strip()] = dict(setup_ms=float(m.group(2)), mdofs=float(m.group(3)),
                                                    bytes_per_dof=float(m.group(4)), l2=float(m.group(5)),
                                                    energy=float(m.group(6)))
    return rows, meta


def parse_newton(path):
    """runs[label] = dict(newton, linear, seconds, status, store_mb, peak_mb, update_seconds,
    applies, apply_seconds);
    meta = (element, dof)."""
    runs, label, meta = {}, None, {}
    for line in open(path):
        if line.startswith("node "):
            meta["element"] = line.split()[2]
        m = re.match(r"Playground: .*?, (\d+) elements, (\d+) dofs", line)
        if m:
            meta["dof"] = int(m.group(2))
            meta["elements"] = int(m.group(1))
        if line.startswith("== "):
            label = line[3:].strip()
            runs[label] = dict(newton=0, linear=0, seconds=None, status=None, store_mb=None, peak_mb=None,
                               device_peak_mb=None)
            continue
        if label is None:
            continue
        m = re.match(r"step \d+ .*?(\d+) Newton iterations, (\d+) linear", line)
        if m:
            runs[label]["newton"] += int(m.group(1))
            runs[label]["linear"] += int(m.group(2))
        m = re.match(r"newton_qp: run ([\d.]+) s, status (\d+)", line)
        if m:
            runs[label]["seconds"] = float(m.group(1))
            runs[label]["status"] = int(m.group(2))
        m = re.match(r"newton_qp: peak resident ([\d.]+) MB", line)
        if m:
            runs[label]["peak_mb"] = float(m.group(1))
        m = re.match(r"newton_qp: device peak ([\d.]+) MB", line)
        if m:
            runs[label]["device_peak_mb"] = float(m.group(1))
        m = re.match(r"qp: \w+ store ([\d.]+) MB, \d+ updates ([\d.]+) s, (\d+) applies ([\d.]+) s", line)
        if m:
            runs[label]["store_mb"] = float(m.group(1))
            runs[label]["update_seconds"] = float(m.group(2))
            runs[label]["applies"] = int(m.group(3))
            runs[label]["apply_seconds"] = float(m.group(4))
    return runs, meta


# The logs of each kind in measured/, and in the quantized-BSR campaigns' directories.
MAIN_GLOBS = {"bench": ["bench_*.out"], "1M": ["newton_*_1M_*.out"], "twist": ["twist_*.out"],
              "10M": ["newton_*_10M_*.out"]}
EXTRA_GLOBS = {"bench": ["ubench_*.out", "qbench_*.out"], "1M": ["[qu]newton_*_1M_*.out"],
               "twist": ["[qu]twist_*.out"], "10M": ["[qu]newton_*_10M*.out"], "peak": ["ipnewton_*.out"]}
# The quantized BSR rows a further directory may add: every BSR row but the two references.
REFERENCE_BSR = ("BSR f64", "BSR f32")


def logs(directory, patterns):
    return sorted(p for pat in patterns for p in glob.glob(os.path.join(directory, pat)))


def bench_rows(src, extras=()):
    """parse_bench over measured/, with the quantized BSR rows of the further directories."""
    rows, meta = parse_bench(logs(src, MAIN_GLOBS["bench"]))
    for d in extras:
        for path in logs(d, EXTRA_GLOBS["bench"]):
            more, _ = parse_bench([path])
            for el, rr in more.items():
                for k, v in rr.items():
                    if k.startswith("BSR ") and k not in REFERENCE_BSR:
                        rows.setdefault(el, {}).setdefault(k, v)
    return rows, meta


def newton_case(src, extras, case):
    """{element: ({operator: run}, meta)} for one case ("1M", "twist", "10M"), merged over its logs;
    runs that failed (non-zero status) are left out.  The further directories add their quantized
    BSR operators only."""
    out = {}
    for path in logs(src, MAIN_GLOBS[case]):
        runs, m = parse_newton(path)
        e = out.setdefault(m["element"], ({}, m))
        e[0].update({operator_name(l): r for l, r in runs.items() if r["seconds"] is not None and r["status"] == 0})
    for d in extras:
        for path in logs(d, EXTRA_GLOBS[case]):
            runs, m = parse_newton(path)
            e = out.setdefault(m["element"], ({}, m))
            for l, r in runs.items():
                op = operator_name(l)
                if op.startswith("qp bsr") and r["seconds"] is not None and r["status"] == 0:
                    e[0].setdefault(op, r)
    return out


def operator_name(label):
    if "--qp" in label:
        return "qp " + re.search(r"--qp (\S+)", label).group(1)
    if "storage_bytes=4" in label:
        return "BSR f32"
    fmt = re.search(r"format=(\S+)", label).group(1)
    return {"BSR": "BSR f64", "MF": "matrix-free", "INEXACT": "P0 projected"}[fmt]


def fmt(x, nd=0):
    return f"{x:,.{nd}f}".replace(",", "{,}")


def sci(x):
    m, e = f"{x:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


# The operators the throughput table reports, and their names in the paper.
THROUGHPUT_ROWS = [
    ("BSR f64", "BSR, \\texttt{fp64}"),
    ("BSR f32", "BSR, \\texttt{fp32}"),
    ("projected P0 f32 packed", "projected (approximate)"),
    ("qp f32 blk a32 packed", "exact, \\texttt{fp32}"),
    ("qp f16 blk a32 lanes packed", "exact, \\texttt{fp16}"),
    ("qp f16 mean+dev blk a32 lanes packed", "exact, \\texttt{fp16} mean+dev"),
    ("qp i8 mean+dev blk a32 lanes packed", "exact, \\texttt{int8} mean+dev"),
    ("qp i8/el blk a32 lanes packed", "exact, \\texttt{int8} per element"),
    ("BSR f16 upper u16", "quantized BSR, \\texttt{fp16}"),
    ("BSR i8/blk upper u16", "quantized BSR, \\texttt{int8}"),
]


def write(rows, meta, newton, outdir):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "grace_throughput.tex"), "w") as f:
        for key, label in THROUGHPUT_ROWS:
            cells = []
            for el in ELEMENTS:
                r = rows.get(el, {}).get(key)
                cells.append(fmt(r["mdofs"]) if r else "---")
            for el in ELEMENTS:
                r = rows.get(el, {}).get(key)
                cells.append(fmt(r["bytes_per_dof"]) if r else "---")
            f.write(f"{label} & " + " & ".join(cells) + " \\\\\n")
    with open(os.path.join(outdir, "grace_errors.tex"), "w") as f:
        for key, label in THROUGHPUT_ROWS[2:]:
            cells = []
            for el in ELEMENTS:
                r = rows.get(el, {}).get(key)
                cells.append(sci(r["energy"]) if r else "---")
            f.write(f"{label} & " + " & ".join(cells) + " \\\\\n")
    with open(os.path.join(outdir, "grace_setup.tex"), "w") as f:
        for key, label in (("BSR f64", "BSR, \\texttt{fp64}"), ("BSR f32", "BSR, \\texttt{fp32}"),
                           ("projected P0 f32", "projected"), ("qp f16 blk a32 lanes", "exact, \\texttt{fp16}"),
                           ("qp i8 mean+dev blk a32 lanes", "exact, \\texttt{int8} mean+dev"),
                           ("BSR f16 upper u16", "quantized BSR, \\texttt{fp16}"),
                           ("BSR i8/blk upper u16", "quantized BSR, \\texttt{int8}")):
            cells = [fmt(rows[el][key]["setup_ms"]) if key in rows.get(el, {}) else "---" for el in ELEMENTS]
            f.write(f"{label} & " + " & ".join(cells) + " \\\\\n")
    with open(os.path.join(outdir, "grace_meta.tex"), "w") as f:
        for el, tag in TAGS.items():
            if el in meta:
                f.write(f"\\newcommand{{\\benchdof{tag}}}{{{fmt(meta[el]['dof'])}}}\n")
                f.write(f"\\newcommand{{\\benchn{tag}}}{{{meta[el]['n']}}}\n")
    order = ["BSR f64", "BSR f32", "matrix-free", "P0 projected", "qp f64", "qp f16", "qp f16md", "qp i8md", "qp i8",
             "qp bsro16", "qp bsroi8"]
    names = {"BSR f64": "BSR, \\texttt{fp64}", "BSR f32": "BSR, \\texttt{fp32}", "matrix-free": "matrix-free",
             "P0 projected": "projected", "qp f64": "exact, \\texttt{fp64}", "qp f16": "exact, \\texttt{fp16}",
             "qp f16md": "exact, \\texttt{fp16} mean+dev", "qp i8md": "exact, \\texttt{int8} mean+dev",
             "qp i8": "exact, \\texttt{int8} per element", "qp bsro16": "quantized BSR, \\texttt{fp16}",
             "qp bsroi8": "quantized BSR, \\texttt{int8}"}
    with open(os.path.join(outdir, "grace_newton.tex"), "w") as f:
        for op in order:
            cells = []
            for el in ELEMENTS:
                run = newton.get(el, ({}, {}))[0].get(op)
                if run and run["seconds"] is not None:
                    cells += [f"{run['newton']}", fmt(run["linear"]), fmt(run["seconds"], 1)]
                else:
                    cells += ["---", "---", "---"]
            f.write(f"{names[op]} & " + " & ".join(cells) + " \\\\\n")
    with open(os.path.join(outdir, "grace_newton_meta.tex"), "w") as f:
        for el, tag in TAGS.items():
            if el in newton:
                f.write(f"\\newcommand{{\\newtondof{tag}}}{{{fmt(newton[el][1]['dof'])}}}\n")


def write_macros(src, outdir, extras=()):
    """The numbers the Results prose quotes, as macros."""
    rows, meta = bench_rows(src, extras)
    m = {}
    tags = TAGS
    keys = {"QpHalf": "qp f16 blk a32 lanes packed", "QpEightMd": "qp i8 mean+dev blk a32 lanes packed",
            "QpHalfMd": "qp f16 mean+dev blk a32 lanes packed", "EaDiffHalf": "ea diff f16 packed",
            "EaDiffEight": "ea diff i8 packed", "Pzero": "projected P0 f32 packed",
            "EaFullHalf": "ea full f16 packed", "Mf": "exact matrix-free packed two-pass",
            "EaPairEight": "ea diff i8 pair packed", "QbsrHalf": "BSR f16 upper u16",
            "QbsrEight": "BSR i8/blk upper u16", "QbsrEightDiff": "BSR i8/blk diff", "QbsrEightFull": "BSR i8/blk full",
            "QbsrEightUpperIdx": "BSR i8/blk upper"}
    for el, tag in tags.items():
        r = rows.get(el, {})
        b64, b32 = r.get("BSR f64"), r.get("BSR f32")
        if not (b64 and b32):
            continue
        for k, key in (("SetupBsrThirtyTwo", "BSR f32"), ("SetupBsrSixtyFour", "BSR f64"),
                       ("SetupQpHalf", "qp f16 blk a32 lanes"), ("SetupEaDiffHalf", "ea diff f16 atomic"),
                       ("SetupPzero", "projected P0 f32"), ("SetupQbsrEight", "BSR i8/blk upper u16"),
                       ("SetupQbsrEightDiff", "BSR i8/blk diff")):
            if key in r:
                m[f"{k}{tag}"] = fmt(r[key]["setup_ms"])
                m[f"bsrSixtyFourRate{tag}"] = fmt(b64["mdofs"])
        m[f"bsrThirtyTwoRate{tag}"] = fmt(b32["mdofs"])
        for k, key in keys.items():
            if key in r:
                m[f"{k}Rate{tag}"] = fmt(r[key]["mdofs"])
                m[f"{k}VsThirtyTwo{tag}"] = f"{r[key]['mdofs'] / b32['mdofs']:.2f}"
                m[f"{k}VsSixtyFour{tag}"] = f"{r[key]['mdofs'] / b64['mdofs']:.2f}"
                if r[key]["bytes_per_dof"] > 0:  # the matrix-free operator stores nothing
                    m[f"{k}MemVsThirtyTwo{tag}"] = f"{b32['bytes_per_dof'] / r[key]['bytes_per_dof']:.1f}"
                    m[f"{k}MemVsSixtyFour{tag}"] = f"{b64['bytes_per_dof'] / r[key]['bytes_per_dof']:.1f}"
    # Quantized BSR: the bandwidth its application moves against BSR's, what storing each pair once
    # gains over the difference form, and what the column offsets save over column indices.
    for el, tag in tags.items():
        r = rows.get(el, {})
        for k, key in (("BsrThirtyTwo", "BSR f32"), ("QbsrEight", "BSR i8/blk upper u16")):
            if key in r:
                m[f"gbs{k}{tag}"] = fmt(r[key]["mdofs"] * r[key]["bytes_per_dof"] / 1e3)
        up, diff, u16 = r.get("BSR i8/blk upper"), r.get("BSR i8/blk diff"), r.get("BSR i8/blk upper u16")
        if up and diff:
            m[f"upperGain{tag}"] = f"{up['mdofs'] / diff['mdofs']:.2f}"
            m[f"upperMem{tag}"] = f"{diff['bytes_per_dof'] / up['bytes_per_dof']:.1f}"
        if up and u16:
            m[f"offsetSaving{tag}"] = f"{100 * (1 - u16['bytes_per_dof'] / up['bytes_per_dof']):.0f}"
    # Operator errors the prose quotes: the double-precision store against BSR (relative l2), and the
    # energy error of the int8 element matrices.  Element-matrix rows missing at the benchmark size
    # come from the smaller membench_* runs.
    small = {}
    for path in sorted(glob.glob(os.path.join(src, "membench_*.out"))):
        for el, dof, rr in parse_bench_sweep(path):
            small.setdefault(el, {}).update(rr)
    for el, tag in tags.items():
        r = dict(small.get(el, {}))
        r.update(rows.get(el, {}))
        if "qp f64 soa a64" in r:
            m[f"corrQpSixtyFour{tag}"] = sci(r["qp f64 soa a64"]["l2"])
        for k, key in (("EaDiffEight", "ea diff i8 packed"), ("EaPairEight", "ea diff i8 pair packed"),
                       ("EaDiffHalf", "ea diff f16 packed"), ("EaFullHalf", "ea full f16 packed")):
            if key in r:
                m[f"err{k}{tag}"] = sci(r[key]["energy"])
        a, b = r.get("ea diff i8 packed"), r.get("ea diff i8 pair packed")
        if a and b:
            m[f"pairSlowdown{tag}"] = f"{100 * (1 - b['mdofs'] / a['mdofs']):.0f}"
            m[f"pairErrGain{tag}"] = f"{a['energy'] / b['energy']:.0f}"
    for el, (runs, mm) in newton_case(src, extras, "1M").items():
        tag = tags[el]
        ex, p0 = runs.get("BSR f64"), runs.get("P0 projected")
        if ex:
            m[f"newtonExact{tag}"] = str(ex["newton"])
        if ex and p0:
            m[f"newtonPzero{tag}"] = str(p0["newton"])
            m[f"newtonPzeroExtra{tag}"] = f"{100 * (p0['newton'] / ex['newton'] - 1):.0f}"
        for k, op in (("EaDiffEight", "qp eadi8"), ("EaPairEight", "qp eadi8p"), ("EaDiffHalf", "qp ead16"),
                      ("QbsrHalf", "qp bsro16"), ("QbsrEight", "qp bsroi8"), ("QpEightEl", "qp i8"),
                      ("QpEightMd", "qp i8md"), ("EaFullHalf", "qp eaf16")):
            if op in runs:
                m[f"newton{k}{tag}"] = str(runs[op]["newton"])
        m[f"newtonDof{tag}"] = fmt(mm["dof"])
    # The twist, merged per element over its logs: Newton iterations of the references and the
    # int8 element matrices.
    for el, (runs, mm) in newton_case(src, extras, "twist").items():
        tag = tags[el]
        for k, op in (("Exact", "BSR f32"), ("EaDiffEight", "qp eadi8"), ("EaPairEight", "qp eadi8p"),
                      ("EaDiffHalf", "qp ead16"), ("QbsrHalf", "qp bsro16"), ("QbsrEight", "qp bsroi8"),
                      ("QpEightEl", "qp i8"), ("QpEightMd", "qp i8md"), ("EaFullHalf", "qp eaf16")):
            if op in runs:
                m[f"twist{k}{tag}"] = str(runs[op]["newton"])
        ref = runs.get("BSR f32")
        for k, op in (("QbsrHalf", "qp bsro16"), ("QbsrEight", "qp bsroi8"), ("EaPairEight", "qp eadi8p"),
                      ("EaDiffHalf", "qp ead16")):
            if ref and op in runs:
                m[f"twistTime{k}{tag}"] = f"{runs[op]['seconds'] / ref['seconds']:.2f}"
    # A long case may be split over several logs (BSR and MF in one, the stores in another).
    for el, (runs, mm) in newton_case(src, extras, "10M").items():
        tag = tags[el]
        ref = runs.get("BSR f32")
        for k, op in (("QpHalf", "qp f16"), ("QpEightMd", "qp i8md"), ("BsrSixtyFour", "BSR f64"),
                      ("EaDiffHalf", "qp ead16"), ("EaPairEight", "qp eadi8p"), ("QbsrHalf", "qp bsro16"),
                      ("QbsrEight", "qp bsroi8"), ("Pzero", "P0 projected")):
            if ref and op in runs:
                m[f"time{k}{tag}"] = f"{runs[op]['seconds'] / ref['seconds']:.2f}"
        for k, op in (("Exact", "BSR f32"), ("QbsrHalf", "qp bsro16"), ("QbsrEight", "qp bsroi8")):
            if op in runs:
                m[f"tenmNewton{k}{tag}"] = str(runs[op]["newton"])
        # The share of the run spent setting the store up (assembly and encoding, for the matrices).
        for k, op in (("QpHalf", "qp f16"), ("QpEightMd", "qp i8md"), ("EaDiffHalf", "qp ead16"),
                      ("EaPairEight", "qp eadi8p"), ("QbsrHalf", "qp bsro16"), ("QbsrEight", "qp bsroi8")):
            if runs.get(op, {}).get("update_seconds") is not None:
                m[f"tenmSetupShare{k}{tag}"] = f"{100 * runs[op]['update_seconds'] / runs[op]['seconds']:.0f}"
        m[f"timeDof{tag}"] = fmt(mm["dof"])
    write_summary(rows, newton_case(src, extras, "10M"), outdir)
    m.update(qbsr_macros(src, extras))
    with open(os.path.join(outdir, "results_macros.tex"), "w") as f:
        for k, v in sorted(m.items()):
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    return m


# The evaluation's summary: time to solution at about 10M dof relative to single-precision BSR, and
# the operator memory it saves, per format and element; the fastest format of each element in bold.
SUMMARY_COLUMNS = [("qp bsro16", "BSR f16 upper u16"), ("qp bsroi8", "BSR i8/blk upper u16"),
                   ("qp ead16", "ea diff f16 packed"), ("qp eadi8p", "ea diff i8 pair packed"),
                   ("qp f16", "qp f16 blk a32 lanes packed"), ("qp i8md", "qp i8 mean+dev blk a32 lanes packed")]


def write_summary(rows, tenm, outdir):
    with open(os.path.join(outdir, "summary.tex"), "w") as f:
        for el in ELEMENTS:
            runs = tenm.get(el, ({}, {}))[0]
            ref, b32 = runs.get("BSR f32"), rows.get(el, {}).get("BSR f32")
            cells = []
            for op, key in SUMMARY_COLUMNS:
                r, b = runs.get(op), rows.get(el, {}).get(key)
                cells.append((r["seconds"] / ref["seconds"], b32["bytes_per_dof"] / b["bytes_per_dof"])
                             if r and ref and b and b32 else None)
            best = min((c[0] for c in cells if c), default=None)
            out = []
            for c in cells:
                if not c:
                    out.append("---")
                    continue
                t = f"{c[0]:.2f}"
                out.append((f"\\textbf{{{t}}}" if c[0] == best else t) + f" ({c[1]:.1f})")
            f.write(f"{NAMES.get(el, el)} & " + " & ".join(out) + " \\\\\n")


def qbsr_macros(src, extras):
    """What the quantized-BSR section quotes beyond the rates: the full int8 form's Newton
    failures, the remainder's share of the upper-only store, and the peak memory of the runs."""
    m = {}
    # Newton runs of the full int8 form, every case of every campaign directory.
    tried = failed = 0
    for d in extras:
        for case in ("1M", "twist", "10M"):
            for path in logs(d, EXTRA_GLOBS[case]):
                for l, r in parse_newton(path)[0].items():
                    if operator_name(l) == "qp bsrfi8" and r["status"] is not None:
                        tried += 1
                        failed += r["status"] != 0
    if tried:
        m["fullEightTried"], m["fullEightFailed"] = str(tried), str(failed)
    # The remainder: blocks outside the SELL part (columns beyond the thread's rows or the offsets'
    # reach), as a share of all stored blocks, padding included (the store's diagnostic line).
    shares = []
    for d in extras:
        for path in logs(d, ["ubench_*.out"]):
            for line in open(path):
                mm = re.match(r"upper store: (\d+) SELL steps x (\d+) lanes, (\d+) remainder blocks", line)
                if mm:
                    steps, lanes, rem = (int(x) for x in mm.groups())
                    shares.append(100 * rem / (rem + steps * lanes))
    if shares:
        m["remainderShareLo"], m["remainderShareHi"] = f"{min(shares):.0f}", f"{max(shares):.0f}"
    # Peak resident memory of whole Newton runs at about 10M dof (GB).
    for d in extras:
        for path in logs(d, EXTRA_GLOBS["peak"]):
            runs, mm = parse_newton(path)
            tag = TAGS[mm["element"]]
            for l, r in runs.items():
                k = {"BSR f32": "BsrThirtyTwo", "P0 projected": "Pzero", "qp bsroi8": "QbsrEight",
                     "qp bsro16": "QbsrHalf", "qp ead16": "EaDiffHalf", "qp eadi8p": "EaPairEight"}.get(operator_name(l))
                if k and r["peak_mb"]:
                    m[f"peak{k}{tag}"] = f"{r['peak_mb'] / 1024:.1f}"
                if k and r["store_mb"]:
                    m[f"store{k}{tag}"] = f"{r['store_mb'] / 1024:.2f}"
            m[f"peakDof{tag}"] = fmt(mm["dof"])
    return m


# ------------------------------------------------------------------------------------- the GPU
#
# measured_gpu/ holds the H100 campaign with the CPU campaign's log stems (bench_qp_cuda,
# newton_qp_cuda).  Its rows name the GPU's layouts: the quantized BSR is the full difference form in
# packed records, with int16 offsets or, on HEX27, int32 columns.

# The bench rows behind each format on the GPU (the first present; the set-up is the same row).
GPU_BENCH = {"BSR f64": ["BSR f64"], "BSR f32": ["BSR f32"],
             "qbsr f16": ["BSR f16 full off16 packed", "BSR f16 full i32 packed"],
             "qbsr i8": ["BSR i8/blk full off16 packed", "BSR i8/blk full i32 packed"],
             "ea diff f16": ["ea diff f16"], "ea pair i8": ["ea diff i8 pair"],
             "qp f16": ["qp f16"], "qp i8md": ["qp i8 mean+dev"]}
GPU_NEWTON = {"BSR f64": "BSR f64", "BSR f32": "BSR f32", "qbsr f16": "qp bsro16", "qbsr i8": "qp bsroi8",
              "ea diff f16": "qp ead16", "ea pair i8": "qp eadi8p", "qp f16": "qp f16", "qp i8md": "qp i8md"}
GPU_NAMES = {"BSR f64": "BSR, \\texttt{fp64}", "BSR f32": "BSR, \\texttt{fp32}",
             "qbsr f16": "quantized BSR, \\texttt{fp16}", "qbsr i8": "quantized BSR, \\texttt{int8}",
             "ea diff f16": "element matrix, \\texttt{fp16}", "ea pair i8": "element matrix, \\texttt{int8} (pair)",
             "qp f16": "quadrature points, \\texttt{fp16}", "qp i8md": "quadrature points, \\texttt{int8} mean+dev"}
GPU_FORMATS = list(GPU_NAMES)
# Macro stems of the formats, as the CPU macros name them.
GPU_TAGS = {"BSR f64": "BsrSixtyFour", "BSR f32": "BsrThirtyTwo", "qbsr f16": "QbsrHalf", "qbsr i8": "QbsrEight",
            "ea diff f16": "EaDiffHalf", "ea pair i8": "EaPairEight", "qp f16": "QpHalf", "qp i8md": "QpEightMd"}
# cuSPARSE's BSR product on the library's values, a reference only (its f32 row has f32 vectors): the
# bench row, its name in the tables, its macro stem and the library row it is compared with.
GPU_CUSPARSE = {"cusparse BSR f64": ("cuSPARSE BSR, \\texttt{fp64}", "CusparseSixtyFour", "BSR f64"),
                "cusparse BSR f32": ("cuSPARSE BSR, \\texttt{fp32}", "CusparseThirtyTwo", "BSR f32")}
# cuSPARSE's generic product on the matrix expanded to scalar CSR, its fastest path (its generic
# product does not take BSR): from the cusparse_*.out logs, compared with those runs' own BSR rows.
GPU_CUSPARSE_CSR = {"cusparse SpMV CSR f64": ("cuSPARSE CSR, \\texttt{fp64}", "CsrSixtyFour", "BSR f64"),
                    "cusparse SpMV CSR f32": ("cuSPARSE CSR, \\texttt{fp32}", "CsrThirtyTwo", "BSR f32")}


def gpu_row(rows, el, fmt_):
    for key in GPU_BENCH[fmt_]:
        if key in rows.get(el, {}):
            return rows[el][key]
    return None


def write_gpu(src, outdir):
    """The GPU's tables (h100_*.tex, summary_gpu.tex) and macros (results_macros_gpu.tex, \\gpu...)."""
    os.makedirs(outdir, exist_ok=True)
    rows, meta = parse_bench(logs(src, MAIN_GLOBS["bench"]))
    crows, _ = parse_bench(logs(src, ["cusparse_*.out"]))
    newton = {c: newton_case(src, (), c) for c in ("1M", "twist", "10M")}
    with open(os.path.join(outdir, "h100_throughput.tex"), "w") as f:
        for o in GPU_FORMATS:
            rs = [gpu_row(rows, el, o) for el in ELEMENTS]
            cells = [fmt(r["mdofs"]) if r else "---" for r in rs] + [fmt(r["bytes_per_dof"]) if r else "---" for r in rs]
            f.write(f"{GPU_NAMES[o]} & " + " & ".join(cells) + " \\\\\n")
    with open(os.path.join(outdir, "h100_cusparse.tex"), "w") as f:
        for key, (name, _, _) in GPU_CUSPARSE.items():
            rs = [rows.get(el, {}).get(key) for el in ELEMENTS]
            cells = [fmt(r["mdofs"]) if r else "---" for r in rs] + [fmt(r["bytes_per_dof"]) if r else "---" for r in rs]
            f.write(f"{name} & " + " & ".join(cells) + " \\\\\n")
        for key, (name, _, _) in GPU_CUSPARSE_CSR.items():
            rs = [crows.get(el, {}).get(key) for el in ELEMENTS]
            cells = [fmt(r["mdofs"]) if r else "---" for r in rs] + [fmt(r["bytes_per_dof"]) if r else "---" for r in rs]
            f.write(f"{name} & " + " & ".join(cells) + " \\\\\n")
    with open(os.path.join(outdir, "h100_setup.tex"), "w") as f:
        for o in GPU_FORMATS:
            rs = [gpu_row(rows, el, o) for el in ELEMENTS]
            f.write(f"{GPU_NAMES[o]} & " + " & ".join(fmt(r["setup_ms"], 1) if r else "---" for r in rs) + " \\\\\n")
    with open(os.path.join(outdir, "h100_newton.tex"), "w") as f:
        for o in GPU_FORMATS:
            cells = []
            for el in ELEMENTS:
                run = newton["1M"].get(el, ({}, {}))[0].get(GPU_NEWTON[o])
                cells += [f"{run['newton']}", fmt(run["linear"]), fmt(run["seconds"], 1)] if run else ["---"] * 3
            f.write(f"{GPU_NAMES[o]} & " + " & ".join(cells) + " \\\\\n")
    # The summary: time to solution at about 10M dof relative to BSR fp32, with the memory factor.
    with open(os.path.join(outdir, "summary_gpu.tex"), "w") as f:
        for el in ELEMENTS:
            runs = newton["10M"].get(el, ({}, {}))[0]
            ref, b32 = runs.get("BSR f32"), gpu_row(rows, el, "BSR f32")
            cells = []
            for o in GPU_FORMATS[2:]:
                r, b = runs.get(GPU_NEWTON[o]), gpu_row(rows, el, o)
                cells.append((r["seconds"] / ref["seconds"], b32["bytes_per_dof"] / b["bytes_per_dof"])
                             if r and ref and b and b32 else None)
            best = min((c[0] for c in cells if c), default=None)
            out = ["---" if not c else (f"\\textbf{{{c[0]:.2f}}}" if c[0] == best else f"{c[0]:.2f}") + f" ({c[1]:.1f})"
                   for c in cells]
            f.write(f"{NAMES.get(el, el)} & " + " & ".join(out) + " \\\\\n")
    m = {}
    for el, tag in TAGS.items():
        if el in meta:
            m[f"gpuBenchDof{tag}"] = fmt(meta[el]["dof"])
            m["gpuDevice"] = meta[el]["device"] or ""
        b32, b64 = gpu_row(rows, el, "BSR f32"), gpu_row(rows, el, "BSR f64")
        for o in GPU_FORMATS:
            r = gpu_row(rows, el, o)
            if not r:
                continue
            k = GPU_TAGS[o]
            m[f"gpu{k}Rate{tag}"] = fmt(r["mdofs"])
            m[f"gpu{k}Setup{tag}"] = fmt(r["setup_ms"], 1)
            m[f"gpu{k}Gbs{tag}"] = fmt(r["mdofs"] * r["bytes_per_dof"] / 1e3)
            if b32:
                m[f"gpu{k}VsThirtyTwo{tag}"] = f"{r['mdofs'] / b32['mdofs']:.2f}"
            if b64:
                m[f"gpu{k}VsSixtyFour{tag}"] = f"{r['mdofs'] / b64['mdofs']:.2f}"
        for key, (_, k, lib) in GPU_CUSPARSE.items():
            r, b = rows.get(el, {}).get(key), gpu_row(rows, el, lib)
            if not r:
                continue
            m[f"gpu{k}Rate{tag}"] = fmt(r["mdofs"])
            if b:
                m[f"gpu{GPU_TAGS[lib]}VsCusparse{tag}"] = f"{b['mdofs'] / r['mdofs']:.2f}"
        for key, (_, k, lib) in GPU_CUSPARSE_CSR.items():
            r, b = crows.get(el, {}).get(key), crows.get(el, {}).get(lib)
            if not r:
                continue
            m[f"gpu{k}Rate{tag}"] = fmt(r["mdofs"])
            m[f"gpu{k}Gbs{tag}"] = fmt(r["mdofs"] * r["bytes_per_dof"] / 1e3)
            if b:
                m[f"gpu{GPU_TAGS[lib]}VsCsr{tag}"] = f"{b['mdofs'] / r['mdofs']:.2f}"
                m[f"gpu{k}BytesVsBsr{tag}"] = f"{r['bytes_per_dof'] / b['bytes_per_dof']:.2f}"
        for case, stem in (("1M", "Newton"), ("twist", "Twist"), ("10M", "Tenm")):
            runs, mm = newton[case].get(el, ({}, {}))
            ref = runs.get("BSR f32")
            for o in GPU_FORMATS:
                r = runs.get(GPU_NEWTON[o])
                if not r:
                    continue
                k = GPU_TAGS[o]
                m[f"gpu{stem}It{k}{tag}"] = str(r["newton"])
                m[f"gpu{stem}Seconds{k}{tag}"] = fmt(r["seconds"], 1)
                if ref:
                    m[f"gpu{stem}Time{k}{tag}"] = f"{r['seconds'] / ref['seconds']:.2f}"
                if r["device_peak_mb"]:
                    m[f"gpu{stem}Peak{k}{tag}"] = f"{r['device_peak_mb'] / 1024:.1f}"
                if r.get("update_seconds") is not None and case == "10M":
                    m[f"gpu{stem}SetupShare{k}{tag}"] = f"{100 * r['update_seconds'] / r['seconds']:.0f}"
                # The rate of the applications inside the Newton run, and against the bench's.
                if r.get("applies") and case == "10M":
                    rate = mm["dof"] * r["applies"] / r["apply_seconds"] / 1e6
                    m[f"gpu{stem}Apply{k}{tag}"] = fmt(rate)
                    b = gpu_row(rows, el, o)
                    if b:
                        m[f"gpu{stem}ApplyVsBench{k}{tag}"] = f"{rate / b['mdofs']:.2f}"
            if mm.get("dof"):
                m[f"gpu{stem}Dof{tag}"] = fmt(mm["dof"])
    # The bench on the Newton study's TET4 mesh (twelve tetrahedra per cell), against the box's.
    mrows, mmeta = parse_bench(logs(src, ["meshbench_tet4_*.out"]))
    for el, rr in mrows.items():
        tag = TAGS[el]
        for o in GPU_FORMATS:
            for key in GPU_BENCH[o]:
                if key in rr:
                    m[f"gpuMesh{GPU_TAGS[o]}Rate{tag}"] = fmt(rr[key]["mdofs"])
                    b = gpu_row(rows, el, o)
                    if b:
                        m[f"gpuMesh{GPU_TAGS[o]}VsBox{tag}"] = f"{rr[key]['mdofs'] / b['mdofs']:.2f}"
                    break
        b32 = rr.get("BSR f32")
        for o in GPU_FORMATS:
            r = next((rr[k] for k in GPU_BENCH[o] if k in rr), None)
            if r and b32:
                m[f"gpuMesh{GPU_TAGS[o]}VsThirtyTwo{tag}"] = f"{r['mdofs'] / b32['mdofs']:.2f}"
        m[f"gpuMeshDof{tag}"] = fmt(mmeta[el]["dof"])
    # The largest size of each element's application sweep.
    for path in logs(src, ["bsweepauto_*.out"]):
        for line in open(path):
            mt = re.match(r"\S+ (\w+): n=\d+, \d+ elements, \d+ nodes, (\d+) dof", line)
            if mt and mt.group(1) in TAGS:
                key = f"gpuSweepMaxDof{TAGS[mt.group(1)]}"
                if int(mt.group(2)) > int(m.get(key, "0").replace("{,}", "")):
                    m[key] = fmt(int(mt.group(2)))
    # Ranges over the elements of the CSR comparison: <stem>Min and <stem>Max.
    for stem in ("gpuBsrSixtyFourVsCsr", "gpuBsrThirtyTwoVsCsr", "gpuCsrSixtyFourGbs", "gpuCsrThirtyTwoGbs",
                 "gpuCsrSixtyFourBytesVsBsr", "gpuCsrThirtyTwoBytesVsBsr"):
        vs = [m[stem + tag] for tag in TAGS.values() if stem + tag in m]
        if vs:
            num = lambda v: float(v.replace("{,}", ""))
            m[stem + "Min"], m[stem + "Max"] = min(vs, key=num), max(vs, key=num)
    with open(os.path.join(outdir, "results_macros_gpu.tex"), "w") as f:
        for k, v in sorted(m.items()):
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    for el in ELEMENTS:
        print(el, len(rows.get(el, {})), "bench rows,", {c: len(newton[c].get(el, ({}, {}))[0]) for c in newton}, "newton runs")
    return m


def main():
    if sys.argv[1] == "--machine":
        if sys.argv[2] != "gpu":
            raise SystemExit("--machine gpu, or the Grace arguments")
        write_gpu(sys.argv[3], sys.argv[4])
        return
    src, outdir, extras = sys.argv[1], sys.argv[2], sys.argv[3:]
    rows, meta = bench_rows(src, extras)
    write(rows, meta, {el: r for el, r in newton_case(src, extras, "1M").items()}, outdir)
    write_macros(src, outdir, extras)
    newton = newton_case(src, extras, "1M")
    for el in ELEMENTS:
        print(el, len(rows.get(el, {})), "bench rows,", len(newton.get(el, ({}, {}))[0]), "newton runs")



# ------------------------------------------------------------------------- resolution sweeps


def parse_bench_sweep(path):
    """[(element, dof, rows)] for a log holding several bench_qp runs."""
    out = []
    el = dof = None
    rows = None
    for line in open(path):
        m = re.match(r"neohookean_ogden (\w+): n=(\d+), (\d+) elements, (\d+) nodes, (\d+) dof", line)
        if m:
            if rows is not None:
                out.append((el, dof, rows))
            el, dof, rows = m.group(1), int(m.group(5)), {}
            continue
        m = re.match(r"\s{2}(\S.*?)\s{2,}([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\S+)\s+(\S+)\s*$", line)
        if rows is not None and m and m.group(1) != "operator":
            rows[m.group(1).strip()] = dict(setup_ms=float(m.group(2)), mdofs=float(m.group(3)),
                                            bytes_per_dof=float(m.group(4)), l2=float(m.group(5)),
                                            energy=float(m.group(6)))
    if rows is not None:
        out.append((el, dof, rows))
    return out


def parse_newton_sweep(path):
    """[(dof, {operator: run})] for a sweep_newton log, one entry per size."""
    out = []
    runs = label = None
    dof = None
    for line in open(path):
        if line.startswith("#### size"):
            if runs is not None:
                out.append((dof, runs))
            runs, label, dof = {}, None, None
            continue
        if runs is None:
            continue
        if line.startswith("== "):
            label = operator_name(line[3:].strip())
            runs[label] = dict(newton=0, linear=0, seconds=None)
            continue
        m = re.match(r"Playground: .*?, (\d+) elements, (\d+) dofs", line)
        if m:
            dof = int(m.group(2))
        if label is None:
            continue
        m = re.match(r"step \d+ .*?(\d+) Newton iterations, (\d+) linear", line)
        if m:
            runs[label]["newton"] += int(m.group(1))
            runs[label]["linear"] += int(m.group(2))
        m = re.match(r"newton_qp: run ([\d.]+) s", line)
        if m:
            runs[label]["seconds"] = float(m.group(1))
    if runs is not None:
        out.append((dof, runs))
    return out


if __name__ == "__main__":
    main()
