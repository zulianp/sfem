#!/usr/bin/env python3
"""Measure the viscoelastic lag from a cyclic torsion run, and check it against the Prony series.

A relaxation test shows viscoelasticity as a torque that decays at fixed twist. A cyclic test
shows it as a *lag*: the torque peaks before the twist does, the torque-angle curve opens into a
hysteresis loop instead of retracing itself, and the area of that loop is the energy dissipated
per cycle. This reads all three off a `history.csv` written with `torsion.profile: cyclic`.

Method. Over the last whole cycle, fit the fundamental harmonic

    T(t) = a0 + a1 cos(w t) + b1 sin(w t),   w = 2 pi / period

by ordinary least squares. The drive is theta(t) = A sin(w t), so b1 is the component of the
torque in phase with the twist (storage) and a1 the component in quadrature, leading it (loss).
Then

    delta   = atan2(a1, b1)          the loss angle
    tan d   = a1 / b1                the loss tangent
    area    = pi * A * a1            energy dissipated per cycle

For comparison the Prony series predicts these in closed form. With
G(t) = g_inf + sum_i g_i exp(-t/tau_i),

    G'(w)  = g_inf + sum_i g_i (w tau_i)^2 / (1 + (w tau_i)^2)
    G''(w) =         sum_i g_i (w tau_i)   / (1 + (w tau_i)^2)

and tan delta = G'' / G'. This is the same complex-modulus model that
python/sfem/regression/regression_prony.py fits to measured DMA data, so a run analysed here is
directly comparable with a measurement analysed there.

Two systematic biases to expect, both downward on the measured delta:

  * The series multiplies only the DEVIATORIC stress. The torque also carries an unrelaxed
    volumetric part, which is exactly in phase with the twist and so inflates the storage term
    without touching the loss term. The stiffer the bulk modulus, the more delta is understated
    -- the same mechanism that biases the weights in the relaxation fit.
  * At large amplitude the elastic response is geometrically nonlinear, so the loop is not an
    ellipse and the fundamental does not capture all of it. The residual reported below is the
    guard: a small residual means the fundamental describes the loop and the phase is
    meaningful.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_torsion_release import solve_normal_equations  # noqa: E402


def prony_complex_modulus(g: List[float], tau: List[float], omega: float):
    """Storage and loss parts of G*(w) for a normalised Prony series."""
    g_inf = 1.0 - sum(g)
    storage = g_inf + sum(gi * (omega * t) ** 2 / (1 + (omega * t) ** 2) for gi, t in zip(g, tau))
    loss = sum(gi * (omega * t) / (1 + (omega * t) ** 2) for gi, t in zip(g, tau))
    return storage, loss


def analyse(rows, period: float, amplitude: float):
    omega = 2 * math.pi / period

    t_end = rows[-1][0]
    start = t_end - period
    cycle = [r for r in rows if r[0] >= start - 1e-12]
    if len(cycle) < 8:
        raise SystemExit(f"[hysteresis] only {len(cycle)} samples in the last cycle; too few to fit")

    basis = lambda t: [1.0, math.cos(omega * t), math.sin(omega * t)]
    ata = [[0.0] * 3 for _ in range(3)]
    atb = [0.0] * 3
    for t, _, torque in cycle:
        phi = basis(t)
        for i in range(3):
            atb[i] += phi[i] * torque
            for j in range(3):
                ata[i][j] += phi[i] * phi[j]

    a0, a1, b1 = solve_normal_equations(ata, atb)

    residual = scale = 0.0
    for t, _, torque in cycle:
        model = sum(c * p for c, p in zip((a0, a1, b1), basis(t)))
        residual += (model - torque) ** 2
        scale += torque ** 2
    rms = math.sqrt(residual / scale) if scale > 0 else float("inf")

    delta = math.degrees(math.atan2(a1, b1))
    loop_area = math.pi * amplitude * a1
    return {
        "n": len(cycle),
        "mean": a0,
        "quadrature": a1,
        "in_phase": b1,
        "tan_delta": a1 / b1 if b1 else float("nan"),
        "delta_deg": delta,
        "loop_area": loop_area,
        "rms": rms,
        "omega": omega,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("history", help="history.csv from a run with torsion.profile: cyclic")
    ap.add_argument("--case", help="the case YAML, for the period, amplitude and Prony terms")
    ap.add_argument("--period", type=float)
    ap.add_argument("--amplitude", type=float)
    ap.add_argument("--min-delta", type=float, default=1.0,
                    help="fail if the measured loss angle is below this many degrees")
    ap.add_argument("--max-rms", type=float, default=0.05,
                    help="fail if the fundamental does not describe the loop this well")
    args = ap.parse_args()

    period, amplitude, g, tau = args.period, args.amplitude, [], []
    if args.case:
        import yaml

        case = yaml.safe_load(Path(args.case).read_text())
        torsion = case.get("torsion") or {}
        period = period or float(torsion.get("period", 0))
        amplitude = amplitude or float(torsion.get("angle", 0))
        terms = (case.get("material") or {}).get("prony") or []
        g = [float(t["g"]) for t in terms]
        tau = [float(t["tau"]) for t in terms]

    if not period or not amplitude:
        print("[hysteresis] need a period and amplitude (--case or --period/--amplitude)", file=sys.stderr)
        return 1

    rows = [
        (float(r["time"]), float(r["angle"]), float(r["torque"]))
        for r in csv.DictReader(open(args.history))
    ]
    m = analyse(rows, period, amplitude)

    print(f"[hysteresis] last cycle: {m['n']} samples, omega = {m['omega']:.4f} rad/s, "
          f"period = {period:g} s, amplitude = {amplitude:g} rad")
    print(f"  torque in phase with twist (storage): {m['in_phase']:.6e}")
    print(f"  torque in quadrature, leading (loss): {m['quadrature']:.6e}")
    print(f"  loss angle delta:                     {m['delta_deg']:.3f} deg")
    print(f"  loss tangent tan(delta):              {m['tan_delta']:.4f}")
    print(f"  dissipated per cycle (loop area):     {m['loop_area']:.6e}")
    print(f"  fundamental fit residual (relative):  {m['rms']:.3e}")

    failures = []
    if m["rms"] > args.max_rms:
        failures.append(
            f"the fundamental describes the loop only to {m['rms']:.3e}; the phase is not meaningful"
        )
    if m["delta_deg"] < args.min_delta:
        failures.append(
            f"loss angle {m['delta_deg']:.3f} deg is below {args.min_delta} deg -- no lag was observed"
        )

    if g:
        storage, loss = prony_complex_modulus(g, tau, m["omega"])
        predicted = math.degrees(math.atan2(loss, storage))
        print()
        print(f"[hysteresis] Prony prediction at this frequency (deviatoric only):")
        print(f"  G'  = {storage:.4f}   G'' = {loss:.4f}   tan(delta) = {loss/storage:.4f}   "
              f"delta = {predicted:.3f} deg")
        print(f"  measured / predicted tan(delta): {m['tan_delta'] / (loss / storage):.4f}")
        print("  (measured is expected to come in BELOW the prediction: the torque carries an")
        print("   unrelaxed volumetric part that adds to storage but not to loss)")
        if m["delta_deg"] > predicted * 1.05:
            failures.append(
                f"measured loss angle {m['delta_deg']:.3f} deg exceeds the deviatoric prediction "
                f"{predicted:.3f} deg. The volumetric term can only push the measurement DOWN, so "
                "the excess is not the material. The usual cause is algorithmic damping: Newmark "
                "with gamma > 0.5 dissipates by construction and contributes its own phase lag on "
                "top of the material's. Measure this quasi-statically -- at these frequencies "
                "inertia is not what is being measured"
            )

    if failures:
        print("\n[hysteresis] FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print("\n[hysteresis] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
