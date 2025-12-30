#!/usr/bin/env python3

"""
Regression: generalized incompressible Mooney–Rivlin with LASSO sparsity.

Goal
----
Fit a sparse generalized Mooney–Rivlin strain-energy density of the form:

    W = Σ c_{i,j} (I1 - 3)^i (I2 - 3)^j,   with i+j <= degree, excluding (0,0)

Assumptions
-----------
- Incompressible, isotropic hyperelasticity (3D model).
- Reduction to standard experimental modes (same reductions as in
  `regression_hyperelasticity.py`):
  - Uniaxial:    P = 2*(λ - λ^-2)*W1 + 2*(1 - λ^-3)*W2
  - Equibiaxial: P = 2*(λ - λ^-5)*W1 + 2*(λ^3 - λ^-3)*W2
  - Pure shear:  P = 2*(λ - λ^-3)*(W1 + W2)
  where W1 = ∂W/∂I1, W2 = ∂W/∂I2 evaluated at the mode's invariants.

Optimization
------------
Weighted LASSO (coordinate descent):

    min_c  0.5 * Σ w_k ( (Xc)_k - y_k )^2  + alpha * ||c||_1

Setup weights let you emphasize modes differently (default 3,3,1).

Notes
-----
This implementation is stdlib-only (no numpy/scipy dependency required).
"""

from __future__ import annotations

import argparse
import math
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import regression_hyperelasticity as rh


def _mode_invariants(mode: str, lam: float) -> Tuple[float, float]:
    if lam <= 0:
        raise ValueError(f"lambda must be > 0, got {lam}")

    if mode == "uniax":
        l2 = lam ** (-0.5)
        l3 = l2
    elif mode == "equibiax":
        l2 = lam
        l3 = lam ** (-2.0)
    elif mode == "pureshear":
        # Chosen to be consistent with the pure shear reduction used in
        # regression_hyperelasticity.py (P depends on (W1+W2)).
        l2 = 1.0
        l3 = lam ** (-1.0)
    else:
        raise KeyError(mode)

    l1 = lam
    I1 = l1 * l1 + l2 * l2 + l3 * l3
    I2 = (l1 * l1) * (l2 * l2) + (l2 * l2) * (l3 * l3) + (l3 * l3) * (l1 * l1)
    return I1, I2


def _mode_factors(mode: str, lam: float) -> Tuple[float, float]:
    # Return the scalar factors multiplying W1 and W2 in the nominal stress P:
    # P = 2*f1*W1 + 2*f2*W2 (uniax/equibiax), and pure shear uses (W1+W2).
    if mode == "uniax":
        return (lam - lam**-2, 1.0 - lam**-3)
    if mode == "equibiax":
        return (lam - lam**-5, lam**3 - lam**-3)
    if mode == "pureshear":
        f = lam - lam**-3
        return (f, f)
    raise KeyError(mode)


@dataclass(frozen=True)
class Term:
    i: int
    j: int

    def name(self) -> str:
        return f"(I1-3)^{self.i} (I2-3)^{self.j}"


def _build_terms(degree: int) -> List[Term]:
    if degree < 1:
        raise ValueError("degree must be >= 1")

    terms: List[Term] = []
    for i in range(0, degree + 1):
        for j in range(0, degree + 1 - i):
            if i == 0 and j == 0:
                continue
            terms.append(Term(i=i, j=j))
    return terms


def _basis_value(term: Term, mode: str, lam: float) -> float:
    # Contribution of coefficient c_{i,j} to nominal stress P at this point.
    I1, I2 = _mode_invariants(mode, lam)
    x = I1 - 3.0
    y = I2 - 3.0

    f1, f2 = _mode_factors(mode, lam)

    v = 0.0
    if term.i > 0:
        v += f1 * term.i * (x ** (term.i - 1)) * (y ** term.j)
    if term.j > 0:
        v += f2 * term.j * (x ** term.i) * (y ** (term.j - 1))

    # Nominal stress has a leading factor 2.
    return 2.0 * v


def _soft_threshold(z: float, alpha: float) -> float:
    if z > alpha:
        return z - alpha
    if z < -alpha:
        return z + alpha
    return 0.0


def _weighted_lasso_coordinate_descent(
    X: List[List[float]],
    y: List[float],
    w: List[float],
    alpha: float,
    *,
    max_iter: int = 5000,
    tol: float = 1e-10,
    standardize: bool = True,
) -> List[float]:
    """
    Solve weighted LASSO with coordinate descent.

    X: n x p (row-major), y: n, w: n nonnegative weights
    """
    n = len(y)
    if n == 0:
        raise ValueError("empty dataset")
    if len(X) != n or len(w) != n:
        raise ValueError("dimension mismatch")
    p = len(X[0])
    if p == 0:
        raise ValueError("no features")

    for wi in w:
        if wi < 0:
            raise ValueError("weights must be nonnegative")

    # Standardize columns with weighted mean/std for better LASSO behavior.
    means = [0.0] * p
    scales = [1.0] * p

    if standardize:
        wsum = sum(w)
        if wsum <= 0:
            raise ValueError("sum of weights must be > 0")

        for j in range(p):
            m = 0.0
            for i in range(n):
                m += w[i] * X[i][j]
            m /= wsum
            means[j] = m

        for j in range(p):
            s2 = 0.0
            for i in range(n):
                d = X[i][j] - means[j]
                s2 += w[i] * d * d
            s2 /= wsum
            s = math.sqrt(s2) if s2 > 0 else 1.0
            scales[j] = s

        for i in range(n):
            row = X[i]
            for j in range(p):
                row[j] = (row[j] - means[j]) / scales[j]

    # Precompute weighted column norms.
    col_norm = [0.0] * p
    for j in range(p):
        s = 0.0
        for i in range(n):
            s += w[i] * X[i][j] * X[i][j]
        col_norm[j] = s

    beta = [0.0] * p
    r = y[:]  # residual = y - X beta (beta starts at 0)

    for _it in range(max_iter):
        max_delta = 0.0

        for j in range(p):
            denom = col_norm[j]
            if denom == 0.0:
                continue

            # rho = Σ w_i * x_ij * (r_i + x_ij*beta_j)
            rho = 0.0
            for i in range(n):
                xij = X[i][j]
                rho += w[i] * xij * (r[i] + xij * beta[j])

            new_bj = _soft_threshold(rho, alpha) / denom
            delta = new_bj - beta[j]
            if delta != 0.0:
                beta[j] = new_bj
                for i in range(n):
                    r[i] -= X[i][j] * delta
                ad = abs(delta)
                if ad > max_delta:
                    max_delta = ad

        if max_delta < tol:
            break

    # Unstandardize: original-space coefficients
    if standardize:
        for j in range(p):
            beta[j] = beta[j] / scales[j]

    return beta


def _weighted_rmse(y: List[float], yhat: List[float], w: List[float]) -> float:
    sw = 0.0
    se = 0.0
    for yi, yhi, wi in zip(y, yhat, w):
        sw += wi
        d = yhi - yi
        se += wi * d * d
    return math.sqrt(se / sw) if sw > 0 else float("nan")


def _plot_measurements_vs_fit(
    *,
    strains_by_mode: Dict[str, List[float]],
    stresses_by_mode: Dict[str, List[float]],
    preds_by_mode: Dict[str, List[float]],
    title: str,
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
        matplotlib.use("Agg", force=True)  # type: ignore[attr-defined]

    plt = importlib.import_module("matplotlib.pyplot")

    modes = [("uniax", "Uniaxial"), ("equibiax", "Equibiaxial"), ("pureshear", "Pure shear")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
    for ax, (mode, label) in zip(axes, modes):
        strains = strains_by_mode[mode]
        stresses = stresses_by_mode[mode]
        pred = preds_by_mode[mode]
        ax.plot(strains, stresses, "o", ms=3, alpha=0.8, label="measurement")
        ax.plot(strains, pred, "-", lw=2, alpha=0.9, label="fit")
        ax.set_title(label)
        ax.set_xlabel("Engineering strain")
        ax.set_ylabel("Engineering stress [MPa]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    fig.suptitle(title)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
    else:
        plt.show()

    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Generalized Mooney–Rivlin regression (LASSO sparsity).")
    ap.add_argument(
        "--xlsx",
        type=Path,
        default=Path(__file__).with_name("70EPDM281_verification.xlsx"),
        help="Path to 70EPDM281_verification.xlsx",
    )
    ap.add_argument("--degree", type=int, default=3, help="Max total degree i+j (default: 3)")
    ap.add_argument("--alpha", type=float, default=1e-3, help="LASSO alpha (default: 1e-3)")
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
    ap.add_argument("--plot", action="store_true", help="Plot measurement vs fit (interactive if --save-plot not set)")
    ap.add_argument(
        "--save-plot",
        type=Path,
        default=None,
        help="If set, save plot to this file (e.g., out.svg or out.png).",
    )
    ap.add_argument("--max-iter", type=int, default=5000, help="Max coordinate descent iterations")
    ap.add_argument("--tol", type=float, default=1e-10, help="Convergence tolerance on max coefficient update")
    ap.add_argument("--no-standardize", action="store_true", help="Disable weighted column standardization")
    ap.add_argument("--top-k", type=int, default=20, help="Print top-K coefficients by magnitude")
    ap.add_argument("--verbose", action="store_true", help="Print extra fit diagnostics")
    args = ap.parse_args(argv)

    if not args.xlsx.exists():
        print(f"[sfem] ERROR: workbook not found: {args.xlsx}", file=sys.stderr)
        return 2

    weights = {"uniax": float(args.setup_weights[0]), "equibiax": float(args.setup_weights[1]), "pureshear": float(args.setup_weights[2])}
    for k, v in weights.items():
        if v < 0:
            print(f"[sfem] ERROR: negative setup weight for {k}: {v}", file=sys.stderr)
            return 2

    terms = _build_terms(args.degree)

    # Reuse extraction logic from regression_hyperelasticity.py (xlsx parsing + headers).
    with zipfile.ZipFile(args.xlsx) as z:
        shared = rh.xlsx_load_shared_strings(z)
        sheet_paths = {
            "uniax": "xl/worksheets/sheet2.xml",
            "equibiax": "xl/worksheets/sheet3.xml",
            "pureshear": "xl/worksheets/sheet4.xml",
        }

        X: List[List[float]] = []
        y: List[float] = []
        w: List[float] = []
        mode_of_row: List[str] = []

        strains_by_mode: Dict[str, List[float]] = {}
        stresses_by_mode: Dict[str, List[float]] = {}
        strains_fit_by_mode: Dict[str, List[float]] = {}
        stresses_fit_by_mode: Dict[str, List[float]] = {}

        for mode, sheet_path in sheet_paths.items():
            cells = rh.xlsx_read_cells(z, sheet_path, shared)
            header_row, strain_col, stress_col = rh.find_header_row_and_cols(cells)
            strains, stresses = rh.extract_xy_from_cols(cells, header_row, strain_col, stress_col)

            strains_by_mode[mode] = strains
            stresses_by_mode[mode] = stresses

            if args.first_half:
                n = len(strains)
                n_fit = max(1, n // 2)
                strains_fit = strains[:n_fit]
                stresses_fit = stresses[:n_fit]
            else:
                strains_fit = strains
                stresses_fit = stresses

            strains_fit_by_mode[mode] = strains_fit
            stresses_fit_by_mode[mode] = stresses_fit

            w_mode = weights[mode]
            for eps, sig in zip(strains_fit, stresses_fit):
                lam = 1.0 + eps
                row = [_basis_value(t, mode, lam) for t in terms]
                X.append(row)
                y.append(sig)
                w.append(w_mode)
                mode_of_row.append(mode)

    beta = _weighted_lasso_coordinate_descent(
        X,
        y,
        w,
        args.alpha,
        max_iter=args.max_iter,
        tol=args.tol,
        standardize=(not args.no_standardize),
    )

    # Predictions + per-mode diagnostics.
    yhat = [sum(xj * bj for xj, bj in zip(row, beta)) for row in X]
    rmse_all = _weighted_rmse(y, yhat, w)

    if args.plot or args.save_plot is not None:
        preds_by_mode: Dict[str, List[float]] = {}
        for mode in ("uniax", "equibiax", "pureshear"):
            strains = strains_fit_by_mode[mode] if args.first_half else strains_by_mode[mode]
            preds_by_mode[mode] = [
                sum(_basis_value(t, mode, 1.0 + eps) * bj for t, bj in zip(terms, beta)) for eps in strains
            ]

        _plot_measurements_vs_fit(
            strains_by_mode=strains_fit_by_mode if args.first_half else strains_by_mode,
            stresses_by_mode=stresses_fit_by_mode if args.first_half else stresses_by_mode,
            preds_by_mode=preds_by_mode,
            title=f"Generalized Mooney–Rivlin (deg={args.degree}, alpha={args.alpha:g})",
            out_path=args.save_plot,
        )

    if args.verbose:
        print(f"[sfem] degree={args.degree} alpha={args.alpha:g} setup_weights={args.setup_weights}")
        print(f"[sfem] weighted RMSE (all) = {rmse_all:.6g} MPa")

    # Summarize coefficients (sparsity).
    idx_sorted = sorted(range(len(beta)), key=lambda j: abs(beta[j]), reverse=True)
    k = min(args.top_k, len(idx_sorted))
    print(f"[sfem] Top {k} coefficients (by |value|):")
    for j in idx_sorted[:k]:
        bj = beta[j]
        if bj == 0.0:
            continue
        print(f"  c[{terms[j].i},{terms[j].j}] = {bj:.16g}   # {terms[j].name()}")

    nnz = sum(1 for bj in beta if bj != 0.0)
    print(f"[sfem] nnz = {nnz}/{len(beta)}  weighted_RMSE_all={rmse_all:.6g} MPa")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())