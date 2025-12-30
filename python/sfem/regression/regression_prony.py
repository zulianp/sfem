#!/usr/bin/env python3

"""
Regression: Prony series (generalized Maxwell) from DMA storage/loss data.

This script fits a sparse Prony series to the *measurement* columns in the `DMA`
worksheet of `70EPDM281_verification.xlsx`.

Model (complex modulus, frequency-domain)
-----------------------------------------
We fit a generalized Maxwell model:

    G*(ω) = G_inf + Σ_k g_k * (i ω τ_k) / (1 + i ω τ_k)

which implies:

    G'(ω)  = G_inf + Σ_k g_k * (ω^2 τ_k^2) / (1 + ω^2 τ_k^2)
    G''(ω) =           Σ_k g_k * (ω τ_k)     / (1 + ω^2 τ_k^2)

This is linear in the unknown coefficients (G_inf and g_k) if τ_k are fixed.

Optimization
------------
Weighted LASSO (coordinate descent) on g_k to enforce sparsity:

    min 0.5 Σ_i w_i ( (Xβ)_i - y_i )^2 + alpha * ||g||_1

The intercept G_inf is NOT penalized.

Notes / limitations
-------------------
- The workbook contains temperature-dependent data, but this script does not
  model time-temperature superposition. Instead it fits *independently per
  temperature* (each temperature group is its own small regression problem).
- Units: the sheet uses MPa; we treat storage/loss as the same modulus units.

Additional mode: viscoelastic Mooney–Rivlin (QLV-style) on uniax/equibiax/pure-shear
----------------------------------------------------------------------------------
When `--dataset=qlv`, we fit a Prony relaxation function G(t) to the mechanical
test curves (Uniaxial / Equibiax / Pure shear) using a simple quasi-linear
viscoelasticity (QLV) ansatz:

    σ(t) = ∫_0^t G(t-s) dσ_el(ε(s))/ds ds

with a Prony series:

    G(t) = g_inf + Σ_k g_k exp(-t/τ_k), with G(0)=1 (i.e. g_inf = 1 - Σ g_k)

This keeps the problem linear in g_k when τ_k are fixed.
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


@dataclass(frozen=True)
class DMARow:
    freq_hz: float
    temp_c: float
    storage_mpa: float
    loss_mpa: float


def _find_cell_with_substr(
    cells: Dict[Tuple[int, int], str],
    substr: str,
    *,
    max_rows: int = 50,
    max_cols: int = 40,
) -> Tuple[int, int]:
    for r in range(1, max_rows + 1):
        for c in range(0, max_cols):
            v = cells.get((r, c))
            if v and substr in v:
                return r, c
    raise RuntimeError(f"Failed to find header containing '{substr}'")


def _extract_dma_rows(xlsx_path: Path) -> List[DMARow]:
    # DMA is workbook sheet8.xml (from earlier workbook inspection).
    sheet_path = "xl/worksheets/sheet8.xml"

    with zipfile.ZipFile(xlsx_path) as z:
        shared = rh.xlsx_load_shared_strings(z)
        cells = rh.xlsx_read_cells(z, sheet_path, shared)

    # Headers are in row 2:
    # Frequenz [Hz], Temp [°C], Storage [MPa], Loss [MPa], Storage [MPa] (sim), Loss [MPa] (sim)
    # We locate by substring to be robust.
    hdr_r, freq_c = _find_cell_with_substr(cells, "Frequenz")
    _, temp_c = _find_cell_with_substr(cells, "Temp")
    _, stor_c = _find_cell_with_substr(cells, "Storage [MPa]")
    _, loss_c = _find_cell_with_substr(cells, "Loss [MPa]")

    # Data starts below header row.
    rows: List[DMARow] = []
    started = False
    for r in range(hdr_r + 1, 5000):
        fv = cells.get((r, freq_c))
        tv = cells.get((r, temp_c))
        sv = cells.get((r, stor_c))
        lv = cells.get((r, loss_c))
        if fv is None or tv is None or sv is None or lv is None:
            if started:
                break
            continue

        try:
            f = float(fv)
            t = float(tv)
            s = float(sv)
            l = float(lv)
        except (TypeError, ValueError):
            continue

        started = True
        rows.append(DMARow(freq_hz=f, temp_c=t, storage_mpa=s, loss_mpa=l))

    if not rows:
        raise RuntimeError("No DMA rows extracted")

    return rows


def _tau_grid(tau_min: float, tau_max: float, n: int) -> List[float]:
    if tau_min <= 0 or tau_max <= 0:
        raise ValueError("tau_min/tau_max must be > 0")
    if tau_max <= tau_min:
        raise ValueError("tau_max must be > tau_min")
    if n < 1:
        raise ValueError("n must be >= 1")
    if n == 1:
        return [tau_min]

    l0 = math.log10(tau_min)
    l1 = math.log10(tau_max)
    out: List[float] = []
    for i in range(n):
        a = i / (n - 1)
        out.append(10 ** (l0 * (1.0 - a) + l1 * a))
    return out


def _soft_threshold(z: float, alpha: float) -> float:
    if z > alpha:
        return z - alpha
    if z < -alpha:
        return z + alpha
    return 0.0


def _weighted_lasso_with_intercept(
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
    Weighted LASSO with an unpenalized intercept in column 0.
    X: n x p, where X[i][0] must be 1.0 for all i.
    """
    n = len(y)
    if n == 0:
        raise ValueError("empty dataset")
    if len(X) != n or len(w) != n:
        raise ValueError("dimension mismatch")
    p = len(X[0])
    if p < 2:
        raise ValueError("need at least intercept + one feature")

    for wi in w:
        if wi < 0:
            raise ValueError("weights must be nonnegative")

    wsum = sum(w)
    if wsum <= 0:
        raise ValueError("sum of weights must be > 0")

    # Standardize columns 1..p-1 (leave intercept as-is).
    means = [0.0] * p
    scales = [1.0] * p
    if standardize:
        for j in range(1, p):
            m = 0.0
            for i in range(n):
                m += w[i] * X[i][j]
            m /= wsum
            means[j] = m

        for j in range(1, p):
            s2 = 0.0
            for i in range(n):
                d = X[i][j] - means[j]
                s2 += w[i] * d * d
            s2 /= wsum
            s = math.sqrt(s2) if s2 > 0 else 1.0
            scales[j] = s

        for i in range(n):
            row = X[i]
            for j in range(1, p):
                row[j] = (row[j] - means[j]) / scales[j]

    # Precompute weighted column norms for j>=1.
    col_norm = [0.0] * p
    for j in range(1, p):
        s = 0.0
        for i in range(n):
            s += w[i] * X[i][j] * X[i][j]
        col_norm[j] = s

    beta = [0.0] * p

    # Initialize intercept to weighted mean of y.
    beta[0] = sum(w[i] * y[i] for i in range(n)) / wsum

    # residual r = y - X beta
    r = [y[i] - beta[0] for i in range(n)]

    for _it in range(max_iter):
        max_delta = 0.0

        # Update intercept (unpenalized): beta0 = argmin Σ w (r + beta0_old - beta0_new)^2
        # Equivalent: beta0_new = weighted mean of (y - Σ_{j>=1} x_j beta_j).
        # Current residual r = y - beta0 - Σ x_j beta_j => y - Σ x_j beta_j = r + beta0.
        num = 0.0
        for i in range(n):
            num += w[i] * (r[i] + beta[0])
        new_b0 = num / wsum
        d0 = new_b0 - beta[0]
        if d0 != 0.0:
            beta[0] = new_b0
            for i in range(n):
                r[i] -= d0
            max_delta = max(max_delta, abs(d0))

        # Update penalized coefficients
        for j in range(1, p):
            denom = col_norm[j]
            if denom == 0.0:
                continue

            rho = 0.0
            for i in range(n):
                xij = X[i][j]
                rho += w[i] * xij * (r[i] + xij * beta[j])

            new_bj = _soft_threshold(rho, alpha) / denom
            dj = new_bj - beta[j]
            if dj != 0.0:
                beta[j] = new_bj
                for i in range(n):
                    r[i] -= X[i][j] * dj
                max_delta = max(max_delta, abs(dj))

        if max_delta < tol:
            break

    # Unstandardize penalized coefficients.
    if standardize:
        for j in range(1, p):
            beta[j] = beta[j] / scales[j]
        # Adjust intercept back: original x_j = (xj_std * scale + mean)
        # y ≈ b0 + Σ b_j ( (x_std*scale + mean) ) = (b0 + Σ b_j*mean) + Σ (b_j*scale) x_std
        # But we unscaled b_j already, so intercept needs subtract Σ b_j * mean.
        beta[0] = beta[0] - sum(beta[j] * means[j] for j in range(1, p))

    return beta


def _weighted_lasso_with_intercept_alpha_vec(
    X: List[List[float]],
    y: List[float],
    w: List[float],
    alpha_vec: List[float],
    *,
    max_iter: int = 5000,
    tol: float = 1e-10,
    standardize: bool = True,
) -> List[float]:
    """
    Weighted LASSO with an unpenalized intercept in column 0, and per-feature penalties for columns 1..p-1.

    - X: n x p, with X[i][0] = 1.0
    - alpha_vec: length (p-1), alpha_vec[j-1] is the L1 penalty for column j
      (use 0.0 for unpenalized features).
    """
    n = len(y)
    if n == 0:
        raise ValueError("empty dataset")
    if len(X) != n or len(w) != n:
        raise ValueError("dimension mismatch")
    p = len(X[0])
    if p < 2:
        raise ValueError("need at least intercept + one feature")
    if len(alpha_vec) != p - 1:
        raise ValueError("alpha_vec must have length p-1")

    for wi in w:
        if wi < 0:
            raise ValueError("weights must be nonnegative")

    wsum = sum(w)
    if wsum <= 0:
        raise ValueError("sum of weights must be > 0")

    # Standardize columns 1..p-1 (leave intercept as-is).
    means = [0.0] * p
    scales = [1.0] * p
    if standardize:
        for j in range(1, p):
            m = 0.0
            for i in range(n):
                m += w[i] * X[i][j]
            m /= wsum
            means[j] = m

        for j in range(1, p):
            s2 = 0.0
            for i in range(n):
                d = X[i][j] - means[j]
                s2 += w[i] * d * d
            s2 /= wsum
            s = math.sqrt(s2) if s2 > 0 else 1.0
            scales[j] = s

        for i in range(n):
            row = X[i]
            for j in range(1, p):
                row[j] = (row[j] - means[j]) / scales[j]

    # Precompute weighted column norms for j>=1.
    col_norm = [0.0] * p
    for j in range(1, p):
        s = 0.0
        for i in range(n):
            s += w[i] * X[i][j] * X[i][j]
        col_norm[j] = s

    beta = [0.0] * p
    beta[0] = sum(w[i] * y[i] for i in range(n)) / wsum

    r = [y[i] - beta[0] for i in range(n)]

    for _it in range(max_iter):
        max_delta = 0.0

        # intercept update
        num = 0.0
        for i in range(n):
            num += w[i] * (r[i] + beta[0])
        new_b0 = num / wsum
        d0 = new_b0 - beta[0]
        if d0 != 0.0:
            beta[0] = new_b0
            for i in range(n):
                r[i] -= d0
            max_delta = max(max_delta, abs(d0))

        for j in range(1, p):
            denom = col_norm[j]
            if denom == 0.0:
                continue

            rho = 0.0
            for i in range(n):
                xij = X[i][j]
                rho += w[i] * xij * (r[i] + xij * beta[j])

            aj = alpha_vec[j - 1]
            new_bj = _soft_threshold(rho, aj) / denom
            dj = new_bj - beta[j]
            if dj != 0.0:
                beta[j] = new_bj
                for i in range(n):
                    r[i] -= X[i][j] * dj
                max_delta = max(max_delta, abs(dj))

        if max_delta < tol:
            break

    if standardize:
        for j in range(1, p):
            beta[j] = beta[j] / scales[j]
        beta[0] = beta[0] - sum(beta[j] * means[j] for j in range(1, p))

    return beta


def _rmse(y: List[float], yhat: List[float]) -> float:
    if not y:
        return float("nan")
    se = 0.0
    for yi, yhi in zip(y, yhat):
        d = yhi - yi
        se += d * d
    return math.sqrt(se / len(y))


def _predict_storage_loss(
    *,
    freq_hz: float,
    tau: List[float],
    g_inf: float,
    g: List[float],
) -> Tuple[float, float]:
    omega = 2.0 * math.pi * freq_hz
    gp = g_inf
    gpp = 0.0
    for gi, tk in zip(g, tau):
        a = omega * tk
        denom = 1.0 + a * a
        gp += gi * (a * a) / denom
        gpp += gi * a / denom
    return gp, gpp


def _plot_measurements_vs_fit(
    *,
    temp_c: float,
    rows: List[DMARow],
    tau: List[float],
    g_inf: float,
    g: List[float],
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

    freqs = [r.freq_hz for r in rows]
    meas_storage = [r.storage_mpa for r in rows]
    meas_loss = [r.loss_mpa for r in rows]

    pred_storage: List[float] = []
    pred_loss: List[float] = []
    for f in freqs:
        gp, gpp = _predict_storage_loss(freq_hz=f, tau=tau, g_inf=g_inf, g=g)
        pred_storage.append(gp)
        pred_loss.append(gpp)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    ax = axes[0]
    ax.set_title("Storage modulus")
    ax.plot(freqs, meas_storage, "o", ms=4, alpha=0.8, label="measurement")
    ax.plot(freqs, pred_storage, "-", lw=2, alpha=0.9, label="fit")
    ax.set_xscale("log")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Storage [MPa]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")

    ax = axes[1]
    ax.set_title("Loss modulus")
    ax.plot(freqs, meas_loss, "o", ms=4, alpha=0.8, label="measurement")
    ax.plot(freqs, pred_loss, "-", lw=2, alpha=0.9, label="fit")
    ax.set_xscale("log")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Loss [MPa]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")

    fig.suptitle(f"Prony fit @ {temp_c:g}°C: G_inf={g_inf:.6g} MPa")

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
    else:
        plt.show()

    plt.close(fig)


def _mr_nominal_stress(mode: str, engineering_strain: float, c10: float, c01: float) -> float:
    # Same nominal stress formulas used in regression_hyperelasticity.py.
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

    return 2.0 * a * c10 + 2.0 * b * c01


def _mr_coeffs(mode: str, engineering_strain: float) -> Tuple[float, float]:
    # Returns (a1, a2) such that sigma = a1*C10 + a2*C01 for the chosen mode.
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

    return 2.0 * a, 2.0 * b


def _fit_mr_from_mechanical_curves(
    *,
    xlsx_path: Path,
    first_half: bool,
    setup_weights: Dict[str, float],
) -> Tuple[float, float]:
    """
    Weighted least squares fit of (C10,C01) using the Measurement curves on:
      - Uniaxial / Equibiax / Pure shear

    Weighting: per-setup weight distributed uniformly over its samples.
    """
    sheet_paths = {
        "uniax": "xl/worksheets/sheet2.xml",
        "equibiax": "xl/worksheets/sheet3.xml",
        "pureshear": "xl/worksheets/sheet4.xml",
    }

    s11 = s12 = s22 = 0.0
    b1 = b2 = 0.0

    for mode, sp in sheet_paths.items():
        _, e, s_meas = _extract_mechanical_rows(xlsx_path, sp)
        if first_half:
            n = len(e)
            n_fit = max(2, n // 2)
            e = e[:n_fit]
            s_meas = s_meas[:n_fit]

        w_mode = float(setup_weights[mode])
        w_per = w_mode / max(1, len(e))
        for eps, sig in zip(e, s_meas):
            a1, a2 = _mr_coeffs(mode, eps)
            s11 += w_per * a1 * a1
            s12 += w_per * a1 * a2
            s22 += w_per * a2 * a2
            b1 += w_per * a1 * sig
            b2 += w_per * a2 * sig

    det = s11 * s22 - s12 * s12
    if det == 0.0:
        raise RuntimeError("Singular MR normal equations (det == 0)")

    c10 = (b1 * s22 - b2 * s12) / det
    c01 = (s11 * b2 - s12 * b1) / det
    return c10, c01


def _extract_mechanical_rows(
    xlsx_path: Path, sheet_path: str
) -> Tuple[List[float], List[float], List[float]]:
    """
    Extract (time[s], engineering_strain, engineering_stress[MPa]) from the *Measurement* columns
    in a mechanical sheet (Uniaxial / Equibiax / Pure shear).
    """
    with zipfile.ZipFile(xlsx_path) as z:
        shared = rh.xlsx_load_shared_strings(z)
        cells = rh.xlsx_read_cells(z, sheet_path, shared)

    # Find a row that contains all three headers; choose the left-most occurrences.
    header_row = None
    time_col = strain_col = stress_col = None
    for r in range(1, 50):
        cols_time = [c for (rr, c), v in cells.items() if rr == r and isinstance(v, str) and "Time [s]" in v]
        cols_strain = [c for (rr, c), v in cells.items() if rr == r and isinstance(v, str) and "Engineering Strain" in v]
        cols_stress = [c for (rr, c), v in cells.items() if rr == r and isinstance(v, str) and "Engineering Stress" in v]
        if cols_time and cols_strain and cols_stress:
            header_row = r
            time_col = min(cols_time)
            strain_col = min(cols_strain)
            stress_col = min(cols_stress)
            break

    if header_row is None or time_col is None or strain_col is None or stress_col is None:
        raise RuntimeError(f"Failed to find Measurement headers in sheet {sheet_path}")

    t: List[float] = []
    e: List[float] = []
    s: List[float] = []
    started = False
    for r in range(header_row + 1, 50000):
        tv = cells.get((r, time_col))
        ev = cells.get((r, strain_col))
        sv = cells.get((r, stress_col))
        if tv is None or ev is None or sv is None:
            if started:
                break
            continue

        try:
            tt = float(tv)
            ee = float(ev)
            ss = float(sv)
        except (TypeError, ValueError):
            continue

        started = True
        if abs(tt) < 1e-14 and abs(ee) < 1e-14 and abs(ss) < 1e-14:
            continue
        t.append(tt)
        e.append(ee)
        s.append(ss)

    if not t:
        raise RuntimeError(f"No measurement rows extracted in sheet {sheet_path}")

    return t, e, s


def _compute_qk_features(
    t: List[float],
    sigma_el: List[float],
    tau: List[float],
) -> List[List[float]]:
    """
    Compute features phi_k(t_i) = q_k(t_i) - sigma_el(t_i), where q_k is the internal variable:

        dq_k/dt = (sigma_el - q_k)/tau_k,  q_k(0) = sigma_el(0)

    We discretize with an exact exponential step using sigma_el at the new time.
    """
    n = len(t)
    if n != len(sigma_el):
        raise ValueError("t and sigma_el length mismatch")
    if n < 2:
        raise ValueError("need at least 2 points")

    q = [sigma_el[0]] * len(tau)
    phi: List[List[float]] = [[0.0] * len(tau) for _ in range(n)]
    for i in range(n):
        for k in range(len(tau)):
            phi[i][k] = q[k] - sigma_el[i]
        if i + 1 == n:
            break
        dt = t[i + 1] - t[i]
        if dt <= 0:
            dt = 0.0
        for k, tk in enumerate(tau):
            if tk <= 0 or dt == 0.0:
                q[k] = sigma_el[i + 1]
                continue
            a = math.exp(-dt / tk)
            q[k] = a * q[k] + (1.0 - a) * sigma_el[i + 1]
    return phi


def _compute_qk(
    t: List[float],
    sigma_el: List[float],
    tau: List[float],
) -> List[List[float]]:
    """
    Compute q_k(t_i) internal variables for:
        dq_k/dt = (sigma_el - q_k)/tau_k, q_k(0)=sigma_el(0)
    """
    n = len(t)
    if n != len(sigma_el):
        raise ValueError("t and sigma_el length mismatch")
    if n < 2:
        raise ValueError("need at least 2 points")

    q = [sigma_el[0]] * len(tau)
    out: List[List[float]] = [[0.0] * len(tau) for _ in range(n)]
    for i in range(n):
        for k in range(len(tau)):
            out[i][k] = q[k]
        if i + 1 == n:
            break
        dt = t[i + 1] - t[i]
        if dt <= 0:
            dt = 0.0
        for k, tk in enumerate(tau):
            if tk <= 0 or dt == 0.0:
                q[k] = sigma_el[i + 1]
                continue
            a = math.exp(-dt / tk)
            q[k] = a * q[k] + (1.0 - a) * sigma_el[i + 1]
    return out


def _weighted_lasso_no_intercept(
    X: List[List[float]],
    y: List[float],
    w: List[float],
    alpha: float,
    *,
    max_iter: int = 8000,
    tol: float = 1e-10,
    standardize: bool = True,
) -> List[float]:
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

    wsum = sum(w)
    if wsum <= 0:
        raise ValueError("sum of weights must be > 0")

    means = [0.0] * p
    scales = [1.0] * p
    if standardize:
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

    col_norm = [0.0] * p
    for j in range(p):
        s = 0.0
        for i in range(n):
            s += w[i] * X[i][j] * X[i][j]
        col_norm[j] = s

    beta = [0.0] * p
    r = y[:]  # y - X beta

    for _it in range(max_iter):
        max_delta = 0.0
        for j in range(p):
            denom = col_norm[j]
            if denom == 0.0:
                continue

            rho = 0.0
            for i in range(n):
                xij = X[i][j]
                rho += w[i] * xij * (r[i] + xij * beta[j])

            new_bj = _soft_threshold(rho, alpha) / denom
            dj = new_bj - beta[j]
            if dj != 0.0:
                beta[j] = new_bj
                for i in range(n):
                    r[i] -= X[i][j] * dj
                max_delta = max(max_delta, abs(dj))
        if max_delta < tol:
            break

    if standardize:
        for j in range(p):
            beta[j] = beta[j] / scales[j]

    return beta


def _plot_mechanical_measurements_vs_fit(
    *,
    strains_by_mode: Dict[str, List[float]],
    stresses_by_mode: Dict[str, List[float]],
    pred_by_mode: Dict[str, List[float]],
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
        e = strains_by_mode[mode]
        s = stresses_by_mode[mode]
        sp = pred_by_mode[mode]
        ax.plot(e, s, "o", ms=3, alpha=0.8, label="measurement")
        ax.plot(e, sp, "-", lw=2, alpha=0.9, label="Prony(QLV) fit")
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


def _select_tau_from_dma(
    *,
    xlsx_path: Path,
    tau_grid: List[float],
    alpha: float,
    temps: str,
    top_k: int,
    weight_storage: float,
    weight_loss: float,
    max_iter: int,
    tol: float,
    standardize: bool,
    verbose: bool,
) -> List[int]:
    rows = _extract_dma_rows(xlsx_path)
    by_temp: Dict[float, List[DMARow]] = {}
    for r in rows:
        by_temp.setdefault(r.temp_c, []).append(r)
    for t in by_temp:
        by_temp[t].sort(key=lambda rr: rr.freq_hz)

    if temps.strip().lower() != "all":
        want = set(float(x.strip()) for x in temps.split(",") if x.strip())
        by_temp = {t: v for t, v in by_temp.items() if t in want}
        if not by_temp:
            raise RuntimeError(f"No matching DMA temps found for temps={temps!r}")

    agg = [0.0] * len(tau_grid)
    for temp_c, tr in sorted(by_temp.items(), key=lambda kv: kv[0]):
        X: List[List[float]] = []
        y: List[float] = []
        w: List[float] = []

        for rr in tr:
            omega = 2.0 * math.pi * rr.freq_hz

            row_s = [1.0]
            for tk in tau_grid:
                a = omega * tk
                row_s.append((a * a) / (1.0 + a * a))
            X.append(row_s)
            y.append(rr.storage_mpa)
            w.append(weight_storage)

            row_l = [0.0]
            for tk in tau_grid:
                a = omega * tk
                row_l.append(a / (1.0 + a * a))
            X.append(row_l)
            y.append(rr.loss_mpa)
            w.append(weight_loss)

        beta = _weighted_lasso_with_intercept(
            X,
            y,
            w,
            alpha,
            max_iter=max_iter,
            tol=tol,
            standardize=standardize,
        )

        g = beta[1:]
        for k, gi in enumerate(g):
            agg[k] += abs(gi)

        if verbose:
            nnz = sum(1 for gi in g if gi != 0.0)
            print(f"[sfem] DMA prefit @ {temp_c:g}C: nnz={nnz}/{len(g)}")

    idx_sorted = sorted(range(len(agg)), key=lambda k: agg[k], reverse=True)
    idx = [k for k in idx_sorted if agg[k] > 0.0][: max(1, top_k)]
    return idx


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Prony-series regression (DMA or mechanical QLV).")
    ap.add_argument(
        "--xlsx",
        type=Path,
        default=Path(__file__).with_name("70EPDM281_verification.xlsx"),
        help="Path to 70EPDM281_verification.xlsx",
    )
    ap.add_argument(
        "--dataset",
        type=str,
        default="dma",
        choices=("dma", "qlv"),
        help="Dataset to fit: 'dma' (storage/loss vs frequency) or 'qlv' (uniax/equibiax/pure-shear stress history).",
    )
    ap.add_argument("--tau-min", type=float, default=1e-4, help="Min relaxation time tau [s] (log grid)")
    ap.add_argument("--tau-max", type=float, default=1e2, help="Max relaxation time tau [s] (log grid)")
    ap.add_argument("--n-tau", type=int, default=12, help="Number of tau grid points")
    ap.add_argument("--alpha", type=float, default=1e-2, help="LASSO alpha applied to g_k (intercept not penalized)")
    ap.add_argument("--max-iter", type=int, default=8000, help="Max coordinate descent iterations")
    ap.add_argument("--tol", type=float, default=1e-10, help="Convergence tolerance")
    ap.add_argument("--no-standardize", action="store_true", help="Disable feature standardization")
    ap.add_argument("--weight-storage", type=float, default=1.0, help="Relative weight of storage equations")
    ap.add_argument("--weight-loss", type=float, default=1.0, help="Relative weight of loss equations")
    ap.add_argument("--plot", action="store_true", help="Plot measurement vs fit (interactive if --save-plot not set)")
    ap.add_argument(
        "--save-plot",
        type=Path,
        default=None,
        help="If set, save plots to this path. If it is a directory, one PNG per temperature is written there.",
    )
    ap.add_argument(
        "--setup-weights",
        nargs=3,
        type=float,
        default=(1.0, 1.0, 1.0),
        metavar=("W_UNIAX", "W_EQUIBIAX", "W_PURESHEAR"),
        help="Weights for uniax/equibiax/pure-shear in QLV fit (default: 1 1 1). Ignored for DMA.",
    )
    ap.add_argument(
        "--first-half",
        action="store_true",
        help="In QLV mode, fit/plot using only the first half of each mechanical measurement curve.",
    )
    ap.add_argument(
        "--tau-from-dma",
        action="store_true",
        help="In QLV mode, prefit DMA and keep only the strongest tau terms (helps conditioning).",
    )
    ap.add_argument(
        "--tau-from-dma-temps",
        type=str,
        default="all",
        help="Temps used for DMA prefit selection (comma-separated or 'all').",
    )
    ap.add_argument(
        "--tau-from-dma-top-k",
        type=int,
        default=6,
        help="Number of tau terms to keep from DMA prefit (default: 6).",
    )
    ap.add_argument(
        "--tau-from-dma-alpha",
        type=float,
        default=None,
        help="Alpha for DMA prefit selection. Defaults to --alpha if not set.",
    )
    ap.add_argument(
        "--c10",
        type=float,
        default=0.31949710763216704,
        help="Mooney–Rivlin C10 [MPa] backbone for QLV mode (default matches regression_hyperelasticity weighted fit).",
    )
    ap.add_argument(
        "--c01",
        type=float,
        default=0.6145828400186,
        help="Mooney–Rivlin C01 [MPa] backbone for QLV mode (default matches regression_hyperelasticity weighted fit).",
    )
    ap.add_argument(
        "--fit-mr-first",
        action="store_true",
        default=True,
        help="In QLV mode, fit Mooney–Rivlin (C10,C01) from the mechanical measurements first (default: on).",
    )
    ap.add_argument(
        "--no-fit-mr-first",
        action="store_true",
        help="In QLV mode, do NOT refit MR; instead use --c10/--c01 as provided.",
    )
    ap.add_argument(
        "--temps",
        type=str,
        default="all",
        help="Temperatures to fit as comma-separated list (e.g. '-90,-60'), or 'all' (default).",
    )
    ap.add_argument("--top-k", type=int, default=10, help="Print top-K g_k by magnitude per temperature")
    ap.add_argument("--verbose", action="store_true", help="Print extra diagnostics")
    args = ap.parse_args(argv)

    if not args.xlsx.exists():
        print(f"[sfem] ERROR: workbook not found: {args.xlsx}", file=sys.stderr)
        return 2

    tau = _tau_grid(args.tau_min, args.tau_max, args.n_tau)

    if args.dataset == "dma":
        rows = _extract_dma_rows(args.xlsx)
        by_temp: Dict[float, List[DMARow]] = {}
        for r in rows:
            by_temp.setdefault(r.temp_c, []).append(r)
        for t in by_temp:
            by_temp[t].sort(key=lambda rr: rr.freq_hz)

        if args.temps.strip().lower() != "all":
            want = set(float(x.strip()) for x in args.temps.split(",") if x.strip())
            by_temp = {t: v for t, v in by_temp.items() if t in want}
            if not by_temp:
                print(f"[sfem] ERROR: no matching temps found for --temps={args.temps!r}", file=sys.stderr)
                return 2

        # Fit independently per temperature.
        for temp_c, tr in sorted(by_temp.items(), key=lambda kv: kv[0]):
            # Build stacked equations for storage and loss.
            X: List[List[float]] = []
            y: List[float] = []
            w: List[float] = []

            for rr in tr:
                omega = 2.0 * math.pi * rr.freq_hz

                # Storage equation
                row_s = [1.0]
                for tk in tau:
                    a = omega * tk
                    row_s.append((a * a) / (1.0 + a * a))
                X.append(row_s)
                y.append(rr.storage_mpa)
                w.append(args.weight_storage)

                # Loss equation
                row_l = [0.0]
                for tk in tau:
                    a = omega * tk
                    row_l.append(a / (1.0 + a * a))
                X.append(row_l)
                y.append(rr.loss_mpa)
                w.append(args.weight_loss)

            beta = _weighted_lasso_with_intercept(
                X,
                y,
                w,
                args.alpha,
                max_iter=args.max_iter,
                tol=args.tol,
                standardize=(not args.no_standardize),
            )

            g_inf = beta[0]
            g = beta[1:]
            yhat = [sum(xj * bj for xj, bj in zip(row, beta)) for row in X]
            rmse = _rmse(y, yhat)

            # Report
            nnz = sum(1 for gi in g if gi != 0.0)
            print(f"[sfem] Temp={temp_c:g}C  G_inf={g_inf:.6g} MPa  nnz={nnz}/{len(g)}  RMSE={rmse:.6g} MPa")

            idx_sorted = [k for k, _ in sorted(enumerate(g), key=lambda kv: abs(kv[1]), reverse=True)]
            kmax = min(args.top_k, len(idx_sorted))
            if args.verbose:
                print(f"[sfem]  tau grid: [{tau[0]:.3g} .. {tau[-1]:.3g}] s  n_tau={len(tau)}  alpha={args.alpha:g}")
            print(f"[sfem]  Top {kmax} g_k:")
            shown = 0
            for k in idx_sorted:
                if shown >= kmax:
                    break
                if g[k] == 0.0:
                    continue
                print(f"    g[{k}]={g[k]:.16g}  tau={tau[k]:.6g} s")
                shown += 1

            if args.plot or args.save_plot is not None:
                out_path: Optional[Path] = None
                if args.save_plot is not None:
                    sp = args.save_plot
                    if sp.exists() and sp.is_dir():
                        out_path = sp / f"prony_fit_T{temp_c:g}C.png"
                    else:
                        if len(by_temp) > 1:
                            out_path = sp.with_name(f"{sp.stem}_T{temp_c:g}C{sp.suffix or '.png'}")
                        else:
                            out_path = sp if sp.suffix else sp.with_suffix(".png")

                _plot_measurements_vs_fit(
                    temp_c=temp_c,
                    rows=tr,
                    tau=tau,
                    g_inf=g_inf,
                    g=g,
                    out_path=out_path,
                )
    else:
        # QLV on mechanical sheets.
        if args.tau_from_dma:
            dma_alpha = args.alpha if args.tau_from_dma_alpha is None else float(args.tau_from_dma_alpha)
            keep_idx = _select_tau_from_dma(
                xlsx_path=args.xlsx,
                tau_grid=tau,
                alpha=dma_alpha,
                temps=args.tau_from_dma_temps,
                top_k=args.tau_from_dma_top_k,
                weight_storage=args.weight_storage,
                weight_loss=args.weight_loss,
                max_iter=args.max_iter,
                tol=args.tol,
                standardize=(not args.no_standardize),
                verbose=args.verbose,
            )
            tau = [tau[i] for i in keep_idx]
            if args.verbose:
                print(f"[sfem] QLV: using tau subset from DMA (k={len(tau)}): {tau}")

        sheet_paths = {
            "uniax": "xl/worksheets/sheet2.xml",
            "equibiax": "xl/worksheets/sheet3.xml",
            "pureshear": "xl/worksheets/sheet4.xml",
        }

        setup_weights = {
            "uniax": float(args.setup_weights[0]),
            "equibiax": float(args.setup_weights[1]),
            "pureshear": float(args.setup_weights[2]),
        }

        fit_mr_first = args.fit_mr_first and (not args.no_fit_mr_first)
        if fit_mr_first:
            c10_fit, c01_fit = _fit_mr_from_mechanical_curves(
                xlsx_path=args.xlsx,
                first_half=args.first_half,
                setup_weights=setup_weights,
            )
            c10 = c10_fit
            c01 = c01_fit
            print(f"[sfem] MR fit first: C10={c10:.16g} MPa  C01={c01:.16g} MPa")
        else:
            c10 = float(args.c10)
            c01 = float(args.c01)
            if args.verbose:
                print(f"[sfem] MR fixed: C10={c10:.16g} MPa  C01={c01:.16g} MPa")

        plot_strain: Dict[str, List[float]] = {}
        plot_stress: Dict[str, List[float]] = {}
        plot_pred: Dict[str, List[float]] = {}

        # We fit the linear form:
        #   sigma(t) = b0 + g_inf * sigma_el(t) + Σ_k g_k * q_k(t)
        # where g_inf is unpenalized and g_k are LASSO-penalized.
        X: List[List[float]] = []
        y: List[float] = []
        w: List[float] = []

        for mode, sp in sheet_paths.items():
            t, e, s_meas = _extract_mechanical_rows(args.xlsx, sp)
            if args.first_half:
                n = len(t)
                n_fit = max(2, n // 2)
                t = t[:n_fit]
                e = e[:n_fit]
                s_meas = s_meas[:n_fit]

            sigma_el = [_mr_nominal_stress(mode, ee, c10, c01) for ee in e]
            qk = _compute_qk(t, sigma_el, tau)

            # per-sample weight scaled by 1/N so setups weights behave as intended
            base_w = setup_weights[mode] / max(1, len(t))
            for i in range(len(t)):
                # X row: [1, sigma_el, q1..qK]
                X.append([1.0, sigma_el[i], *qk[i]])
                y.append(s_meas[i])
                w.append(base_w)

            plot_strain[mode] = e
            plot_stress[mode] = s_meas

        # alpha for columns: sigma_el unpenalized, qk penalized.
        alpha_vec = [0.0] + [args.alpha] * len(tau)
        beta = _weighted_lasso_with_intercept_alpha_vec(
            X,
            y,
            w,
            alpha_vec,
            max_iter=args.max_iter,
            tol=args.tol,
            standardize=(not args.no_standardize),
        )

        b0 = beta[0]
        g_inf = beta[1]
        g = beta[2:]
        nnz = sum(1 for gi in g if gi != 0.0)
        print(f"[sfem] QLV fit: b0={b0:.6g}  g_inf={g_inf:.6g}  nnz={nnz}/{len(g)}")

        idx_sorted = [k for k, _ in sorted(enumerate(g), key=lambda kv: abs(kv[1]), reverse=True)]
        kmax = min(args.top_k, len(idx_sorted))
        print(f"[sfem] Top {kmax} g_k:")
        shown = 0
        for k in idx_sorted:
            if shown >= kmax:
                break
            if g[k] == 0.0:
                continue
            print(f"    g[{k}]={g[k]:.16g}  tau={tau[k]:.6g} s")
            shown += 1

        # Build predictions per mode for plotting / reporting.
        for mode, sp in sheet_paths.items():
            t, e, s_meas = _extract_mechanical_rows(args.xlsx, sp)
            if args.first_half:
                n = len(t)
                n_fit = max(2, n // 2)
                t = t[:n_fit]
                e = e[:n_fit]
                s_meas = s_meas[:n_fit]

            sigma_el = [_mr_nominal_stress(mode, ee, c10, c01) for ee in e]
            qk = _compute_qk(t, sigma_el, tau)
            s_pred = [b0 + g_inf * se + sum(gi * qki for gi, qki in zip(g, qrow)) for se, qrow in zip(sigma_el, qk)]
            plot_pred[mode] = s_pred

            rmse = _rmse(s_meas, s_pred)
            print(f"[sfem] Mode={mode} RMSE={rmse:.6g} MPa")

        if args.plot or args.save_plot is not None:
            out_path = args.save_plot
            _plot_mechanical_measurements_vs_fit(
                strains_by_mode=plot_strain,
                stresses_by_mode=plot_stress,
                pred_by_mode=plot_pred,
                title=f"Prony(QLV) fit (alpha={args.alpha:g}, tau=[{tau[0]:.3g}..{tau[-1]:.3g}]s)",
                out_path=out_path,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())