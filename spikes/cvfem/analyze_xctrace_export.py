#!/usr/bin/env python3
"""Analyze CVFEM xctrace traces (Time Profiler or CPU Profiler).

Supports:
  - Time Profiler  -> schema time-profile, weight in nanoseconds
  - CPU Profiler   -> schema cpu-profile,   weight in CPU cycles

Examples:
  ../../venv/bin/python analyze_xctrace_export.py traces/cvfem_*.trace
  ../../venv/bin/python analyze_xctrace_export.py traces/export/*_cpu-profile.xml
  ./profile_xctrace.sh --template 'CPU Profiler' --mode jacobian --analyze
"""

from __future__ import annotations

import argparse
import html
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence


PROFILE_SCHEMAS = ("time-profile", "cpu-profile")


def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(1)


def short(name: str, n: int = 96) -> str:
    return name if len(name) <= n else name[: n - 3] + "..."


def classify_leaf(name: str) -> str:
    """Map a leaf symbol to a stable performance bucket."""
    l = name.lower()
    if "jacobian_dense" in l or "jacobian_element" in l:
        return "jacobian_kernel"
    if "simd_microkernel" in l or "run_microkernel" in l:
        return "residual_kernel"
    if (
        "tet4_local_to_global" in l
        or "tet4_local_slots_to_bsr4" in l
        or "bsr4_accum" in l
        or "bsr4_add" in l
        or "find_cols" in l
    ):
        return "bsr_scatter"
    if "assemble_bsr" in l:
        return "assemble_glue"
    if "apply_packed" in l or "apply_atomic" in l:
        return "apply_glue"
    if "memset" in l or "bzero" in l or "cvfem_zero" in l:
        return "zero_memset"
    if "memcpy" in l or "memmove" in l:
        return "memcpy"
    if "gather" in l:
        return "gather"
    if "scatter" in l:
        return "scatter"
    if "omp" in l or "gomp" in l or "libomp" in l or "kmp_" in l:
        return "openmp_runtime"
    if any(
        k in l
        for k in (
            "packedmesh",
            "create_tet",
            "precompute",
            "sfc",
            "reorder",
            "node_to_node",
            "create_n2n",
            "build_pack",
        )
    ):
        return "setup_mesh"
    if (
        name.startswith("0x")
        or "dyld" in l
        or name.startswith("__")
        or "pthread" in l
        or "malloc" in l
        or "nanov2" in l
    ):
        return "system_runtime"
    return "other"


def hot_keys_for_label(label: str) -> tuple[str, ...]:
    l = label.lower()
    if "jac" in l or "assemble" in l:
        return ("assemble_bsr4", "jacobian_dense", "tet4_local_to_global", "tet4_local_slots_to_bsr4")
    if "res" in l or "apply" in l:
        return ("apply_packed", "apply_atomic", "simd_microkernel", "run_microkernel")
    return ("cvfem", "assemble_bsr", "apply_packed", "jacobian", "simd_microkernel")


def guess_label(path: Path) -> str:
    name = path.name.lower()
    if "jacobian" in name or "assemble" in name:
        return "jacobian"
    if "residual" in name or "apply" in name:
        return "residual"
    return path.stem


def run_xctrace(cmd: list[str], what: str) -> None:
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        die("xcrun/xctrace not found (need Xcode / Command Line Tools)")
    except subprocess.CalledProcessError as e:
        die(f"{what} failed ({e.returncode})")


def detect_profile_schema(trace: Path, export_dir: Path) -> str:
    """Return 'time-profile' or 'cpu-profile' from the trace TOC."""
    toc = export_dir / f"{trace.stem}_toc.xml"
    run_xctrace(
        [
            "xcrun",
            "xctrace",
            "export",
            "--input",
            str(trace),
            "--toc",
            "--output",
            str(toc),
        ],
        f"xctrace toc export for {trace.name}",
    )
    text = toc.read_text(errors="ignore")
    schemas = set(re.findall(r'schema="([^"]+)"', text))
    for schema in PROFILE_SCHEMAS:
        if schema in schemas:
            return schema
    die(
        f"no supported profile schema in {trace.name}; found: "
        f"{', '.join(sorted(schemas)) or '(none)'}. "
        f"Supported: {', '.join(PROFILE_SCHEMAS)}"
    )


def export_profile(trace: Path, schema: str, xml_out: Path) -> None:
    xml_out.parent.mkdir(parents=True, exist_ok=True)
    print(f"==> exporting {schema}: {trace.name}")
    run_xctrace(
        [
            "xcrun",
            "xctrace",
            "export",
            "--input",
            str(trace),
            "--xpath",
            f'/trace-toc/run[@number="1"]/data/table[@schema="{schema}"]',
            "--output",
            str(xml_out),
        ],
        f"xctrace {schema} export for {trace.name}",
    )
    if not xml_out.exists():
        die(f"export did not create {xml_out}")


def infer_schema_from_xml(path: Path) -> str:
    name = path.name.lower()
    if "cpu-profile" in name or "cpu_profile" in name:
        return "cpu-profile"
    if "time-profile" in name or "time_profile" in name:
        return "time-profile"
    # Peek at schema attribute without full parse.
    with path.open("r", errors="ignore") as f:
        head = f.read(4096)
    m = re.search(r'<schema name="([^"]+)"', head)
    if m and m.group(1) in PROFILE_SCHEMAS:
        return m.group(1)
    return "time-profile"


def parse_profile_rows(path: Path) -> tuple[list[tuple[int, list[str]]], str]:
    """Return (rows, weight_kind) where weight_kind is 'ns' or 'cycles'."""
    frame_name: dict[str, str] = {}
    backtrace_frames: dict[str, list[str]] = {}
    state_fmt: dict[str, str] = {}
    weight_of: dict[str, int] = {}
    rows: list[tuple[int, list[str]]] = []
    weight_kind = "ns"

    for _event, elem in ET.iterparse(path, events=("end",)):
        tag = elem.tag
        if tag == "frame":
            fid, name = elem.get("id"), elem.get("name")
            if fid and name:
                frame_name[fid] = html.unescape(name)
        elif tag == "thread-state":
            sid, fmt = elem.get("id"), elem.get("fmt")
            if sid and fmt:
                state_fmt[sid] = fmt
        elif tag in ("weight", "cycle-weight"):
            wid = elem.get("id")
            if wid and elem.text:
                weight_of[wid] = int(elem.text)
            if tag == "cycle-weight":
                weight_kind = "cycles"
        elif tag == "backtrace":
            bid = elem.get("id")
            if bid is not None:
                names: list[str] = []
                for fr in elem.findall("frame"):
                    if fr.get("name"):
                        n = html.unescape(fr.get("name") or "")
                        names.append(n)
                        if fr.get("id"):
                            frame_name[fr.get("id") or ""] = n
                    elif fr.get("ref"):
                        names.append(frame_name.get(fr.get("ref") or "", f"<unresolved {fr.get('ref')}>"))
                backtrace_frames[bid] = names
        elif tag == "row":
            st_el = elem.find("thread-state")
            st = ""
            if st_el is not None:
                st = st_el.get("fmt") or state_fmt.get(st_el.get("ref") or "", "")
            w_el = elem.find("cycle-weight")
            if w_el is None:
                w_el = elem.find("weight")
            default_w = 1 if weight_kind == "cycles" else 1_000_000
            w = default_w
            if w_el is not None:
                if w_el.tag == "cycle-weight":
                    weight_kind = "cycles"
                if w_el.text:
                    w = int(w_el.text)
                elif w_el.get("ref"):
                    w = weight_of.get(w_el.get("ref") or "", default_w)
            if st and st != "Running":
                elem.clear()
                continue
            bt_el = elem.find("backtrace")
            frames: list[str] = []
            if bt_el is not None:
                bid = bt_el.get("id")
                ref = bt_el.get("ref")
                if bid and bid in backtrace_frames:
                    frames = backtrace_frames[bid]
                elif ref:
                    frames = backtrace_frames.get(ref, [])
            if frames:
                rows.append((w, frames))
            elem.clear()
    return rows, weight_kind


def in_hot_path(frames: Sequence[str], keys: Sequence[str]) -> bool:
    return any(any(k in f for k in keys) for f in frames)


def format_weight(w: int, total: int, kind: str) -> str:
    pct = 100.0 * w / total
    if kind == "cycles":
        return f"{pct:6.2f}%  {w:12,}"
    return f"{pct:6.2f}%  {w / 1e9:7.3f}s"


def report(label: str, rows: list[tuple[int, list[str]]], top_n: int, weight_kind: str) -> None:
    total = sum(w for w, _ in rows) or 1
    leaf = Counter()
    inclusive = Counter()
    for w, frames in rows:
        leaf[frames[0]] += w
        for n in set(frames):
            inclusive[n] += w

    keys = hot_keys_for_label(label)
    hot = [(w, f) for w, f in rows if in_hot_path(f, keys)]
    hot_w = sum(w for w, _ in hot) or 1
    hot_leaf = Counter()
    for w, f in hot:
        hot_leaf[f[0]] += w

    unit = "cycles" if weight_kind == "cycles" else "ns (wall-sample weight)"
    print(f"\n{'=' * 72}")
    print(f"{label}")
    print(f"{'=' * 72}")
    print(f"samples: {len(rows)}  total_{weight_kind}: {total:,}  ({unit})")
    print(f"hot keys: {', '.join(keys)}")
    print(f"hot fraction: {100.0 * hot_w / total:.1f}%")

    print("\nTop SELF (all Running samples):")
    for name, w in leaf.most_common(top_n):
        print(f"  {format_weight(w, total, weight_kind)}  {short(name)}")

    print("\nCategory SELF (all samples):")
    cat = Counter()
    for name, w in leaf.items():
        cat[classify_leaf(name)] += w
    for name, w in cat.most_common():
        print(f"  {100.0 * w / total:6.2f}%  {name}")

    print("\nHot-path SELF (filtered):")
    for name, w in hot_leaf.most_common(top_n):
        print(f"  {format_weight(w, hot_w, weight_kind)}  {short(name)}")

    print("\nHot-path collapsed:")
    hot_cat = Counter()
    for name, w in hot_leaf.items():
        hot_cat[classify_leaf(name)] += w
    for name, w in hot_cat.most_common():
        print(f"  {100.0 * w / hot_w:6.2f}%  {name}")

    print("\nInteresting INCLUSIVE (all samples):")
    interesting_keys = (
        "cvfem",
        "assemble",
        "jacobian",
        "apply",
        "microkernel",
        "tet4_local",
        "memset",
        "bzero",
        "main",
        "packed",
        "precompute",
        "sfc",
        "memmove",
        "memcpy",
    )
    shown = 0
    for name, w in inclusive.most_common():
        l = name.lower()
        if any(k in l for k in interesting_keys):
            print(f"  {format_weight(w, total, weight_kind)}  {short(name)}")
            shown += 1
            if shown >= top_n:
                break


def resolve_inputs(
    paths: Iterable[Path],
    export_dir: Path,
    force_export: bool,
    schema_override: str | None,
) -> list[tuple[str, Path, str]]:
    """Return (label, xml_path, schema) triples."""
    out: list[tuple[str, Path, str]] = []
    for p in paths:
        if not p.exists():
            die(f"path not found: {p}")
        if p.suffix == ".xml":
            schema = schema_override or infer_schema_from_xml(p)
            out.append((guess_label(p), p, schema))
            continue
        if p.suffix == ".trace" or (p.is_dir() and p.name.endswith(".trace")):
            schema = schema_override or detect_profile_schema(p, export_dir)
            xml_out = export_dir / f"{p.stem}_{schema}.xml"
            if force_export or not xml_out.exists():
                export_profile(p, schema, xml_out)
            else:
                print(f"==> reusing export: {xml_out}")
            out.append((guess_label(p), xml_out, schema))
            continue
        die(f"unsupported input (want .trace or profile .xml): {p}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Summarize CVFEM residual/Jacobian xctrace profiles "
        "(Time Profiler or CPU Profiler)."
    )
    ap.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help=".trace files and/or exported time-profile / cpu-profile .xml files",
    )
    ap.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="Directory for exported XML (default: <first-input-dir>/export)",
    )
    ap.add_argument("--force-export", action="store_true", help="Re-run xctrace export even if XML exists")
    ap.add_argument(
        "--schema",
        choices=PROFILE_SCHEMAS,
        default=None,
        help="Force profile schema (default: auto-detect from TOC / filename)",
    )
    ap.add_argument("--top", type=int, default=20, help="Rows to show in top lists (default: 20)")
    ap.add_argument(
        "--label",
        action="append",
        default=[],
        help="Override label for inputs in order (repeatable)",
    )
    args = ap.parse_args()

    inputs = [p.expanduser().resolve() for p in args.inputs]
    export_dir = args.export_dir
    if export_dir is None:
        export_dir = inputs[0].parent / "export"
    export_dir = export_dir.expanduser().resolve()
    export_dir.mkdir(parents=True, exist_ok=True)

    resolved = resolve_inputs(inputs, export_dir, args.force_export, args.schema)
    if args.label:
        if len(args.label) != len(resolved):
            die(f"--label count ({len(args.label)}) must match inputs ({len(resolved)})")
        resolved = [(lab, xml, schema) for lab, (_, xml, schema) in zip(args.label, resolved)]

    print("CVFEM xctrace profile analysis")
    print(f"  export_dir: {export_dir}")
    for label, xml_path, schema in resolved:
        print(f"\n==> parsing {xml_path}  (schema={schema})")
        rows, weight_kind = parse_profile_rows(xml_path)
        if schema == "cpu-profile":
            weight_kind = "cycles"
        if not rows:
            print(f"warning: no Running samples in {xml_path}", file=sys.stderr)
            continue
        report(f"{label} [{schema}]", rows, top_n=args.top, weight_kind=weight_kind)

    print(
        "\nNote: weights sum over OpenMP threads; compare percentages, not absolute totals.\n"
        "Time Profiler uses sample-time weights (ns); CPU Profiler uses cycle counts."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
