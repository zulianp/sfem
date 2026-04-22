#!/usr/bin/env python3

"""
Regression: Mooney–Rivlin hyperelasticity fit.

This script fits a 2-parameter incompressible Mooney–Rivlin model (C10, C01)
to the *measurement* curves in `70EPDM281_verification.xlsx`:
- Uniaxial
- Equibiax
- Pure shear

We intentionally avoid external Excel dependencies (pandas/openpyxl). The .xlsx
file is parsed via stdlib `zipfile` + XML.

Model (nominal/engineering stress P vs engineering strain ε, with λ = 1 + ε):
- Uniaxial:
    P = 2*C10*(λ - λ^-2) + 2*C01*(1 - λ^-3)
- Equibiaxial:
    P = 2*C10*(λ - λ^-5) + 2*C01*(λ^3 - λ^-3)
- Pure shear (plane strain / traction-free thickness):
    P = 2*(C10 + C01)*(λ - λ^-3)

The fit is linear in (C10, C01), solved by a 2x2 normal-equation system.
"""

from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

_XML_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _xlsx_col_to_idx(col: str) -> int:
    idx = 0
    for c in col:
        idx = idx * 26 + (ord(c) - ord("A") + 1)
    return idx - 1


def _xlsx_load_shared_strings(z: zipfile.ZipFile) -> List[str]:
    # shared strings are optional; this workbook has them.
    try:
        root = ET.fromstring(z.read("xl/sharedStrings.xml"))
    except KeyError:
        return []

    out: List[str] = []
    for si in root.findall("m:si", _XML_NS):
        # Shared strings can be in <t> or split across multiple <r><t>.
        t = si.find("m:t", _XML_NS)
        if t is not None and t.text is not None:
            out.append(t.text)
            continue

        parts: List[str] = []
        for rt in si.findall(".//m:r/m:t", _XML_NS):
            if rt.text:
                parts.append(rt.text)
        out.append("".join(parts))

    return out


def xlsx_load_shared_strings(z: zipfile.ZipFile) -> List[str]:
    return _xlsx_load_shared_strings(z)


def _xlsx_read_cells(
    z: zipfile.ZipFile, sheet_path: str, shared_strings: List[str]
) -> Dict[Tuple[int, int], str]:
    """
    Return (row_idx_1based, col_idx_0based) -> raw string value.
    Only cells with cached values are returned (formulas without cached values are ignored).
    """
    root = ET.fromstring(z.read(sheet_path))
    cells: Dict[Tuple[int, int], str] = {}

    for c in root.findall(".//m:sheetData/m:row/m:c", _XML_NS):
        r = c.attrib.get("r", "")
        if not r:
            continue

        col = "".join([ch for ch in r if ch.isalpha()])
        row = "".join([ch for ch in r if ch.isdigit()])
        if not col or not row:
            continue

        ri = int(row)
        ci = _xlsx_col_to_idx(col)

        v = c.find("m:v", _XML_NS)
        if v is None or v.text is None:
            continue

        t = c.attrib.get("t")
        if t == "s":
            # shared string index
            try:
                cells[(ri, ci)] = shared_strings[int(v.text)]
            except (ValueError, IndexError):
                cells[(ri, ci)] = v.text
        else:
            cells[(ri, ci)] = v.text

    return cells


def xlsx_read_cells(z: zipfile.ZipFile, sheet_path: str, shared_strings: List[str]) -> Dict[Tuple[int, int], str]:
    return _xlsx_read_cells(z, sheet_path, shared_strings)


def _find_header_row_and_cols(
    cells: Dict[Tuple[int, int], str],
    *,
    strain_header_substr: str = "Engineering Strain",
    stress_header_substr: str = "Engineering Stress",
    max_scan_rows: int = 40,
    max_scan_cols: int = 40,
) -> Tuple[int, int, int]:
    header_row: Optional[int] = None
    strain_col: Optional[int] = None
    stress_col: Optional[int] = None

    for r in range(1, max_scan_rows + 1):
        for c in range(0, max_scan_cols):
            v = cells.get((r, c))
            if not v:
                continue
            if strain_header_substr in v:
                strain_col = c
            if stress_header_substr in v:
                stress_col = c

        if strain_col is not None and stress_col is not None:
            header_row = r
            break

    if header_row is None or strain_col is None or stress_col is None:
        raise RuntimeError(
            f"Failed to locate measurement headers '{strain_header_substr}' and "
            f"'{stress_header_substr}' in sheet."
        )

    return header_row, strain_col, stress_col


def find_header_row_and_cols(
    cells: Dict[Tuple[int, int], str],
    *,
    strain_header_substr: str = "Engineering Strain",
    stress_header_substr: str = "Engineering Stress",
    max_scan_rows: int = 40,
    max_scan_cols: int = 40,
) -> Tuple[int, int, int]:
    return _find_header_row_and_cols(
        cells,
        strain_header_substr=strain_header_substr,
        stress_header_substr=stress_header_substr,
        max_scan_rows=max_scan_rows,
        max_scan_cols=max_scan_cols,
    )


def _extract_xy_from_cols(
    cells: Dict[Tuple[int, int], str],
    header_row: int,
    strain_col: int,
    stress_col: int,
    *,
    max_rows: int = 20000,
) -> Tuple[List[float], List[float]]:
    strains: List[float] = []
    stresses: List[float] = []

    started = False
    for r in range(header_row + 1, max_rows + 1):
        sv = cells.get((r, strain_col))
        tv = cells.get((r, stress_col))

        if sv is None or tv is None:
            if started:
                break
            continue

        try:
            s = float(sv)
            t = float(tv)
        except (TypeError, ValueError):
            continue

        started = True
        # Skip all-zero row (present in workbook).
        if abs(s) < 1e-14 and abs(t) < 1e-14:
            continue

        strains.append(s)
        stresses.append(t)

    if not strains:
        raise RuntimeError("No measurement data extracted (strain/stress arrays are empty).")

    return strains, stresses


def extract_xy_from_cols(
    cells: Dict[Tuple[int, int], str],
    header_row: int,
    strain_col: int,
    stress_col: int,
    *,
    max_rows: int = 20000,
) -> Tuple[List[float], List[float]]:
    return _extract_xy_from_cols(
        cells,
        header_row,
        strain_col,
        stress_col,
        max_rows=max_rows,
    )


def _mr_coeffs(mode: str, engineering_strain: float) -> Tuple[float, float]:
    lam = 1.0 + engineering_strain
    if lam <= 0:
        raise ValueError(f"Nonpositive stretch computed: lambda={lam} from strain={engineering_strain}")

    if mode == "uniax":
        a = lam - lam**-2
        b = 1.0 - lam**-3
    elif mode == "equibiax":
        a = lam - lam**-5
        b = lam**3 - lam**-3
    elif mode == "pureshear":
        a = lam - lam**-3
        b = lam - lam**-3
    else:
        raise KeyError(mode)

    # P = 2*a*C10 + 2*b*C01
    return 2.0 * a, 2.0 * b


def _solve_2x2_normal_eq(points: Iterable[Tuple[float, float, float, float]]) -> Tuple[float, float]:
    # Solve min Σ w_i (a1_i*C10 + a2_i*C01 - y_i)^2 with x=(C10,C01),
    # where rows are (a1,a2) and weights are w_i.
    s11 = s12 = s22 = 0.0
    b1 = b2 = 0.0

    n = 0
    for a1, a2, y, w in points:
        n += 1
        s11 += w * a1 * a1
        s12 += w * a1 * a2
        s22 += w * a2 * a2
        b1 += w * a1 * y
        b2 += w * a2 * y

    if n < 2:
        raise RuntimeError("Not enough points to fit (need at least 2).")

    det = s11 * s22 - s12 * s12
    if det == 0.0:
        raise RuntimeError("Singular normal equations (det == 0).")

    c10 = (b1 * s22 - b2 * s12) / det
    c01 = (s11 * b2 - s12 * b1) / det
    return c10, c01


def _rmse(mode: str, strains: List[float], stresses: List[float], c10: float, c01: float) -> float:
    se = 0.0
    for s, y in zip(strains, stresses):
        a1, a2 = _mr_coeffs(mode, s)
        pred = a1 * c10 + a2 * c01
        e = pred - y
        se += e * e
    return math.sqrt(se / len(strains))


@dataclass(frozen=True)
class FitResult:
    c10: float
    c01: float
    rmse_uniax: float
    rmse_equibiax: float
    rmse_pureshear: float


def _predict_stress(mode: str, strains: List[float], c10: float, c01: float) -> List[float]:
    out: List[float] = []
    for s in strains:
        a1, a2 = _mr_coeffs(mode, s)
        out.append(a1 * c10 + a2 * c01)
    return out


def _plot_measurements_vs_fit(
    *,
    strains_by_mode: Dict[str, List[float]],
    stresses_by_mode: Dict[str, List[float]],
    c10: float,
    c01: float,
    out_path: Optional[Path],
) -> None:
    try:
        import importlib

        matplotlib = importlib.import_module("matplotlib")
    except ImportError as e:
        raise RuntimeError(
            "Plotting requires matplotlib. It is listed in python/requirements.txt; "
            "ensure your Python environment has it installed."
        ) from e

    if out_path is not None:
        # Headless-friendly default.
        matplotlib.use("Agg", force=True)  # type: ignore[attr-defined]

    plt = importlib.import_module("matplotlib.pyplot")

    modes = [
        ("uniax", "Uniaxial"),
        ("equibiax", "Equibiaxial"),
        ("pureshear", "Pure shear"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
    for ax, (mode, title) in zip(axes, modes):
        strains = strains_by_mode[mode]
        stresses = stresses_by_mode[mode]
        pred = _predict_stress(mode, strains, c10, c01)

        ax.plot(strains, stresses, "o", ms=3, alpha=0.8, label="measurement")
        ax.plot(strains, pred, "-", lw=2, alpha=0.9, label="Mooney–Rivlin fit")
        ax.set_title(title)
        ax.set_xlabel("Engineering strain")
        ax.set_ylabel("Engineering stress [MPa]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    fig.suptitle(f"Mooney–Rivlin fit: C10={c10:.6g} MPa, C01={c01:.6g} MPa")

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
    else:
        plt.show()

    plt.close(fig)


def fit_mooney_rivlin_from_xlsx(xlsx_path: Path) -> FitResult:
    with zipfile.ZipFile(xlsx_path) as z:
        shared = _xlsx_load_shared_strings(z)

        sheet_paths = {
            "uniax": "xl/worksheets/sheet2.xml",  # Uniaxial
            "equibiax": "xl/worksheets/sheet3.xml",  # Equibiax
            "pureshear": "xl/worksheets/sheet4.xml",  # Pure shear
        }

        per_mode_xy: Dict[str, Tuple[List[float], List[float]]] = {}
        points: List[Tuple[float, float, float, float]] = []

        for mode, sheet_path in sheet_paths.items():
            cells = _xlsx_read_cells(z, sheet_path, shared)
            header_row, strain_col, stress_col = _find_header_row_and_cols(cells)
            strains, stresses = _extract_xy_from_cols(cells, header_row, strain_col, stress_col)
            print(f"Mode: {mode}, Strains: {strains}, Stresses: {stresses}")
            per_mode_xy[mode] = (strains, stresses)

            for s, y in zip(strains, stresses):
                a1, a2 = _mr_coeffs(mode, s)
                points.append((a1, a2, y, 1.0))

    c10, c01 = _solve_2x2_normal_eq(points)

    rmse_uniax = _rmse("uniax", *per_mode_xy["uniax"], c10, c01)
    rmse_equibiax = _rmse("equibiax", *per_mode_xy["equibiax"], c10, c01)
    rmse_pureshear = _rmse("pureshear", *per_mode_xy["pureshear"], c10, c01)

    return FitResult(
        c10=c10,
        c01=c01,
        rmse_uniax=rmse_uniax,
        rmse_equibiax=rmse_equibiax,
        rmse_pureshear=rmse_pureshear,
    )


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Mooney–Rivlin regression fit from bundled verification workbook.")
    ap.add_argument(
        "--xlsx",
        type=Path,
        default=Path(__file__).with_name("70EPDM281_verification.xlsx"),
        help="Path to 70EPDM281_verification.xlsx",
    )
    ap.add_argument("--plot", action="store_true", help="Plot measurement vs fit (interactive if --save-plot not set)")
    ap.add_argument(
        "--save-plot",
        type=Path,
        default=None,
        help="If set, save plot to this file (e.g., out.png). Implies --plot and uses headless backend.",
    )
    ap.add_argument(
        "--setup-weights",
        nargs=3,
        type=float,
        default=(3.0, 3.0, 1.0),
        metavar=("W_UNIAX", "W_EQUIBIAX", "W_PURESHEAR"),
        help="Weights for uniax/equibiax/pure-shear setups in the fit (default: 3 3 1).",
    )
    ap.add_argument(
        "--first-half",
        action="store_true",
        help="Fit using only the first half of each measurement curve (per setup).",
    )
    ap.add_argument(
        "--force-expected-check",
        action="store_true",
        help="Force checking --expected-c10/--expected-c01 even when --first-half is enabled.",
    )
    ap.add_argument("--expected-c10", type=float, default=0.31949710763216704, help="Expected C10 [MPa]")
    ap.add_argument("--expected-c01", type=float, default=0.6145828400186, help="Expected C01 [MPa]")
    ap.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance for (C10,C01) checks")
    ap.add_argument("--rmse-uniax-max", type=float, default=0.25, help="Max allowed uniax RMSE [MPa]")
    ap.add_argument("--rmse-equibiax-max", type=float, default=0.35, help="Max allowed equibiax RMSE [MPa]")
    ap.add_argument("--rmse-pureshear-max", type=float, default=0.25, help="Max allowed pure shear RMSE [MPa]")
    ap.add_argument("--verbose", action="store_true", help="Print fit details")
    args = ap.parse_args(argv)

    if not args.xlsx.exists():
        print(f"[sfem] ERROR: workbook not found: {args.xlsx}", file=sys.stderr)
        return 2

    # Load per-mode measurement curves (needed for fit and optional plotting).
    with zipfile.ZipFile(args.xlsx) as z:
        shared = _xlsx_load_shared_strings(z)
        sheet_paths = {
            "uniax": "xl/worksheets/sheet2.xml",  # Uniaxial
            "equibiax": "xl/worksheets/sheet3.xml",  # Equibiax
            "pureshear": "xl/worksheets/sheet4.xml",  # Pure shear
        }

        strains_by_mode: Dict[str, List[float]] = {}
        stresses_by_mode: Dict[str, List[float]] = {}
        points: List[Tuple[float, float, float, float]] = []

        weights = {
            "uniax": float(args.setup_weights[0]),
            "equibiax": float(args.setup_weights[1]),
            "pureshear": float(args.setup_weights[2]),
        }

        if args.verbose:
            print(f"[sfem] Weights: {weights}")

        strains_fit_by_mode: Dict[str, List[float]] = {}
        stresses_fit_by_mode: Dict[str, List[float]] = {}

        for mode, sheet_path in sheet_paths.items():
            cells = _xlsx_read_cells(z, sheet_path, shared)
            header_row, strain_col, stress_col = _find_header_row_and_cols(cells)
            strains, stresses = _extract_xy_from_cols(cells, header_row, strain_col, stress_col)

            if args.first_half:
                n = len(strains)
                n_fit = max(1, n // 2)
                strains_fit = strains[:n_fit]
                stresses_fit = stresses[:n_fit]
            else:
                strains_fit = strains
                stresses_fit = stresses

            strains_by_mode[mode] = strains
            stresses_by_mode[mode] = stresses
            strains_fit_by_mode[mode] = strains_fit
            stresses_fit_by_mode[mode] = stresses_fit

            if args.verbose:
                print(f"[sfem] Mode: {mode} n={len(strains)} n_fit={len(strains_fit)}")

            w = weights[mode]
            for s, y in zip(strains_fit, stresses_fit):
                a1, a2 = _mr_coeffs(mode, s)
                points.append((a1, a2, y, w))

    c10, c01 = _solve_2x2_normal_eq(points)
    res = FitResult(
        c10=c10,
        c01=c01,
        rmse_uniax=_rmse("uniax", strains_by_mode["uniax"], stresses_by_mode["uniax"], c10, c01),
        rmse_equibiax=_rmse("equibiax", strains_by_mode["equibiax"], stresses_by_mode["equibiax"], c10, c01),
        rmse_pureshear=_rmse("pureshear", strains_by_mode["pureshear"], stresses_by_mode["pureshear"], c10, c01),
    )

    rmse_fit_uniax = _rmse("uniax", strains_fit_by_mode["uniax"], stresses_fit_by_mode["uniax"], c10, c01)
    rmse_fit_equibiax = _rmse("equibiax", strains_fit_by_mode["equibiax"], stresses_fit_by_mode["equibiax"], c10, c01)
    rmse_fit_pureshear = _rmse("pureshear", strains_fit_by_mode["pureshear"], stresses_fit_by_mode["pureshear"], c10, c01)

    ok = True
    if args.verbose:
        print(f"[sfem] Fit: C10={res.c10:.16g} MPa, C01={res.c01:.16g} MPa")
        if args.first_half:
            print(
                f"[sfem] RMSE (fit subset): uniax={rmse_fit_uniax:.6g} MPa, "
                f"equibiax={rmse_fit_equibiax:.6g} MPa, "
                f"pureshear={rmse_fit_pureshear:.6g} MPa"
            )
            print(
                f"[sfem] RMSE (full curve): uniax={res.rmse_uniax:.6g} MPa, "
                f"equibiax={res.rmse_equibiax:.6g} MPa, "
                f"pureshear={res.rmse_pureshear:.6g} MPa"
            )
        else:
            print(
                f"[sfem] RMSE: uniax={res.rmse_uniax:.6g} MPa, "
                f"equibiax={res.rmse_equibiax:.6g} MPa, "
                f"pureshear={res.rmse_pureshear:.6g} MPa"
            )

    check_expected = (not args.first_half) or args.force_expected_check
    if check_expected:
        if abs(res.c10 - args.expected_c10) > args.atol:
            print(
                f"[sfem] FAIL: C10 mismatch: got {res.c10:.16g}, expected {args.expected_c10:.16g} ± {args.atol:g}",
                file=sys.stderr,
            )
            ok = False

        if abs(res.c01 - args.expected_c01) > args.atol:
            print(
                f"[sfem] FAIL: C01 mismatch: got {res.c01:.16g}, expected {args.expected_c01:.16g} ± {args.atol:g}",
                file=sys.stderr,
            )
            ok = False
    else:
        if args.verbose:
            print("[sfem] NOTE: skipping expected (C10,C01) checks because --first-half is enabled")

    rmse_chk_uniax = rmse_fit_uniax if args.first_half else res.rmse_uniax
    rmse_chk_equibiax = rmse_fit_equibiax if args.first_half else res.rmse_equibiax
    rmse_chk_pureshear = rmse_fit_pureshear if args.first_half else res.rmse_pureshear

    if rmse_chk_uniax > args.rmse_uniax_max:
        print(f"[sfem] FAIL: uniax RMSE {rmse_chk_uniax:.6g} > {args.rmse_uniax_max:.6g}", file=sys.stderr)
        ok = False

    if rmse_chk_equibiax > args.rmse_equibiax_max:
        print(
            f"[sfem] FAIL: equibiax RMSE {rmse_chk_equibiax:.6g} > {args.rmse_equibiax_max:.6g}",
            file=sys.stderr,
        )
        ok = False

    if rmse_chk_pureshear > args.rmse_pureshear_max:
        print(
            f"[sfem] FAIL: pure shear RMSE {rmse_chk_pureshear:.6g} > {args.rmse_pureshear_max:.6g}",
            file=sys.stderr,
        )
        ok = False

    if args.plot or args.save_plot is not None:
        plot_strains_by_mode = strains_fit_by_mode if args.first_half else strains_by_mode
        plot_stresses_by_mode = stresses_fit_by_mode if args.first_half else stresses_by_mode
        _plot_measurements_vs_fit(
            strains_by_mode=plot_strains_by_mode,
            stresses_by_mode=plot_stresses_by_mode,
            c10=res.c10,
            c01=res.c01,
            out_path=args.save_plot,
        )

    if ok:
        if args.verbose:
            print("[sfem] PASS")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())