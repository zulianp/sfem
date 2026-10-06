#!/usr/bin/env python3
"""Check that a torsion-with-release history behaves like a Prony-series viscoelastic solid.

The run has three phases and each one has a signature that a purely elastic material would
not produce:

  ramp     the twist grows, so the torque grows;
  hold     the twist is constant but the torque decays through the relaxation spectrum,
           towards the long-term fraction g_inf of its peak;
  release  the constraint is gone, so the torque falls to the residual floor of the
           nonlinear solve, and the twist keeps recovering afterwards rather than being
           finished the instant the grip lets go, which is what a purely elastic body
           would do.

Only the last of these needs a reference value; the rest are relations within the run
itself, which keeps the check meaningful when the material parameters in the case change.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from typing import List


class Row:
    __slots__ = ("time", "angle", "released", "torque", "ux", "uy", "uz", "gnorm")

    def __init__(self, d):
        self.time = float(d["time"])
        self.angle = float(d["angle"])
        self.released = int(d["released"]) != 0
        self.torque = float(d["torque"])
        self.ux = float(d["ux"])
        self.uy = float(d["uy"])
        self.uz = float(d["uz"])
        self.gnorm = float(d["gnorm"])

    @property
    def twist_magnitude(self) -> float:
        """Displacement of the control point in the plane normal to the twist axis."""
        return math.hypot(self.uy, self.uz)


def read_history(path: str) -> List[Row]:
    with open(path, newline="") as fh:
        rows = [Row(d) for d in csv.DictReader(fh)]
    if not rows:
        raise SystemExit(f"[validate] {path} has no rows")
    return rows


def solve_normal_equations(a: List[List[float]], b: List[float]) -> List[float]:
    """Gaussian elimination with partial pivoting. The system is (n_terms + 1) square -- three
    or four unknowns in practice -- so there is no reason to reach for a library."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-300:
            raise ValueError("singular normal equations")
        m[col], m[pivot] = m[pivot], m[col]
        for r in range(n):
            if r == col:
                continue
            f = m[r][col] / m[col][col]
            for c in range(col, n + 1):
                m[r][c] -= f * m[col][c]

    return [m[i][n] / m[i][i] for i in range(n)]


def ramp_factor(tau: float, t_ramp: float) -> float:
    """How much of a Prony term survives a linear ramp of duration t_ramp, relative to a step.

    R = (tau / t_ramp) * (exp(t_ramp / tau) - 1), which tends to 1 as t_ramp -> 0.

    This factor is the whole reason a naive fit misreads the series. The torque decay after a
    ramp is NOT the relaxation function translated to the end of the ramp: a term whose tau is
    comparable to the ramp has already done much of its relaxing before the hold begins, so its
    apparent weight is suppressed and the slower terms absorb the difference. In this spike's
    default case tau_fast and the ramp are both 1.0, and ignoring this reads g[0] as 0.30
    instead of 0.40.
    """
    if t_ramp <= 0:
        return 1.0
    x = t_ramp / tau
    # expm1 keeps the small-x case accurate, which is where tau >> t_ramp and R is near 1.
    return math.expm1(x) / x


def fit_relaxation(rows: List[Row], taus: List[float], t_ramp: float):
    """Recover the Prony weights from the torque decay at constant twist.

    Under quasi-linear viscoelasticity the stress is the convolution
    sigma(t) = int_0^t G(t - s) d sigma_el(s), and with sigma_el ramped linearly to sigma_e over
    [0, t_ramp] this integrates, for t >= t_ramp, to

        T(t) = sigma_e * ( g_inf + sum_i g_i R_i exp(-t / tau_i) )

    with R_i = ramp_factor(tau_i, t_ramp). So the model is still linear in the unknowns and the
    fit is still ordinary least squares in 1 + len(taus) coefficients -- but the exponentials are
    measured from t = 0, not from the end of the ramp, and each coefficient carries R_i.

    sigma_e is not known independently, so it is recovered from the series' own normalisation,
    G(0) = 1, i.e. g_inf + sum_i g_i = 1. That makes the returned weights a genuine measurement
    rather than a ratio against an arbitrary reference torque.

    (python/sfem/regression/regression_prony.py carries the machinery for the harder problem
    where the tau_i are unknown and have to be selected by a sparse fit; it is not needed here.)
    """
    basis = lambda t: [1.0] + [math.exp(-t / tau) for tau in taus]
    n = 1 + len(taus)

    ata = [[0.0] * n for _ in range(n)]
    atb = [0.0] * n
    for r in rows:
        phi = basis(r.time)
        for i in range(n):
            atb[i] += phi[i] * r.torque
            for j in range(n):
                ata[i][j] += phi[i] * phi[j]

    coeffs = solve_normal_equations(ata, atb)

    residual = 0.0
    scale = 0.0
    for r in rows:
        phi = basis(r.time)
        model = sum(c * p for c, p in zip(coeffs, phi))
        residual += (model - r.torque) ** 2
        scale += r.torque ** 2

    rms = math.sqrt(residual / scale) if scale > 0 else float("inf")

    # Undo the ramp suppression, then normalise with G(0) = 1.
    unramped = [coeffs[0]] + [coeffs[i + 1] / ramp_factor(tau, t_ramp) for i, tau in enumerate(taus)]
    total = sum(unramped)
    weights = [c / total for c in unramped] if total != 0 else unramped

    return weights, rms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("history", help="history.csv written by prony_visco_torsion")
    ap.add_argument("--control", help="history.csv of the never-released control run")
    ap.add_argument(
        "--case",
        help=(
            "the case YAML the run used. When given, the Prony weights are recovered from the "
            "torque decay and checked against the ones the case asked for"
        ),
    )
    ap.add_argument(
        "--fit-tol",
        type=float,
        default=0.05,
        help="tolerance on each recovered weight, absolute, in units of the relaxation function",
    )
    ap.add_argument(
        "--min-relaxation",
        type=float,
        default=0.05,
        help="the torque must lose at least this fraction of its peak while held",
    )
    ap.add_argument(
        "--min-recovery",
        type=float,
        default=0.05,
        help="the control point must recover at least this fraction of its twist after release",
    )
    ap.add_argument(
        "--released-torque-tol",
        type=float,
        default=1e-4,
        help=(
            "torque tolerated on the released face, relative to the peak. This is a "
            "Newton-tolerance-limited zero rather than a round-off one: the torque is a "
            "lever-weighted sum of residual entries, so it is bounded by the nonlinear "
            "tolerance the step converged to, not by machine precision"
        ),
    )
    args = ap.parse_args()

    rows = read_history(args.history)
    failures: List[str] = []

    # Whether inertia is in play, and whether the twist or the torque is the prescribed one,
    # both change what counts as correct.
    dynamic_run = False
    creep_run = False
    gate_recovery = True
    case_terms: List[dict] = []
    ramp_time = 0.0
    if args.case:
        try:
            import yaml as _yaml
        except ImportError:
            pass
        else:
            with open(args.case) as fh:
                _case = _yaml.safe_load(fh)
            dynamic_run = (_case.get("dynamics") or {}).get("type", "quasi_static") != "quasi_static"
            _torsion = _case.get("torsion") or {}
            creep_run = _torsion.get("control", "angle") == "torque"
            ramp_time = float(_torsion.get("ramp_time", 0.0))
            case_terms = (_case.get("material") or {}).get("prony") or []
            # A case can opt out of the Prony recovery gate. The fit assumes the reported moment
            # is proportional to the deviatoric relaxation function, which is a good assumption
            # for torsion of this beam and a poor one for anything that stretches fibres --
            # bending carries a much larger unrelaxed share, and the fit hands it to g_inf. The
            # residual does not catch that: the exponentials still describe the decay, it is the
            # decomposition that is wrong.
            gate_recovery = bool((_case.get("validate") or {}).get("prony_recovery", True))
            if dynamic_run:
                print("[validate] dynamic run: the released end may ring through the undeformed state")

    for r in rows:
        if not math.isfinite(r.gnorm) or not math.isfinite(r.torque):
            failures.append(f"non-finite output at t = {r.time}")
            break

    held = [r for r in rows if not r.released]
    freed = [r for r in rows if r.released]

    if not held:
        failures.append("no held phase in the history")
        return report(failures, rows)

    peak = max(abs(r.torque) for r in held)
    if peak <= 0:
        failures.append("the torque never left zero while the twist was applied")
        return report(failures, rows)

    # --- hold: the torque must relax at constant twist -------------------------------------
    final_angle = held[-1].angle
    plateau = [r for r in held if abs(r.angle - final_angle) <= 1e-12 * max(1.0, abs(final_angle))]

    if creep_run:
        # Torque control: the twist is the unknown and the torque is held, so there is no
        # plateau in the angle to read a relaxation off. What has to be true instead is that
        # the control worked and that the structure kept turning under it.
        after_ramp = [r for r in held if r.time >= ramp_time - 1e-12]
        if len(after_ramp) < 3:
            failures.append("no held-torque phase in the history")
        else:
            target = after_ramp[0].torque
            worst = max(abs(r.torque - target) for r in after_ramp)
            theta_0, theta_end = after_ramp[0].angle, after_ramp[-1].angle
            factor = theta_end / theta_0 if theta_0 else float("nan")

            print(f"[validate] creep: torque held at {target:.6e}, largest excursion {worst:.2e}")
            print(
                f"[validate] creep: twist {theta_0:.5f} -> {theta_end:.5f} rad over "
                f"t = {after_ramp[0].time:g} .. {after_ramp[-1].time:g}  (x{factor:.3f})"
            )

            if abs(target) > 0 and worst > 1e-4 * abs(target):
                failures.append(
                    f"the torque was not held: it moved by {worst:.2e} against a target of {target:.2e}"
                )

            if factor <= 1.0:
                failures.append(
                    f"the twist did not creep (factor {factor:.3f}); under a held torque a "
                    "viscoelastic solid must keep turning"
                )

            for a, b in zip(after_ramp, after_ramp[1:]):
                if abs(b.angle) < abs(a.angle) - 1e-9:
                    failures.append(
                        f"the twist went backwards between t = {a.time:g} and {b.time:g} "
                        "while the torque was held"
                    )
                    break

            if case_terms:
                # Creep is the inverse of relaxation: the twist grows towards 1/g_inf times its
                # instantaneous value. Overshooting that is not something the material can do.
                g_inf = 1.0 - sum(float(t["g"]) for t in case_terms)
                limit = 1.0 / g_inf if g_inf > 0 else float("inf")
                print(f"[validate] creep: limit 1/g_inf = {limit:.4f}, reached {100*factor/limit:.1f}% of it")
                if factor > limit * 1.02:
                    failures.append(
                        f"creep factor {factor:.3f} exceeds the limit 1/g_inf = {limit:.3f}"
                    )
    elif dynamic_run:
        # The measured torque is the total reaction, inertia included, so while the body is
        # still ringing it is not the relaxation function and nothing below applies to it. The
        # release and recovery checks further down are unaffected and still run.
        print(
            "[validate] dynamic run: skipping the relaxation checks -- the reaction torque "
            "carries the inertial term, so it is neither monotone nor proportional to G(t)"
        )
    elif len(plateau) < 3:
        failures.append("the twist was never held long enough to see it relax")
    else:
        first, last = plateau[0], plateau[-1]
        lost = (abs(first.torque) - abs(last.torque)) / peak
        print(
            f"[validate] hold: |T| {abs(first.torque):.6e} -> {abs(last.torque):.6e} "
            f"over t = {first.time:g} .. {last.time:g}  ({100 * lost:.2f}% of peak)"
        )
        if lost < args.min_relaxation:
            failures.append(
                f"the torque lost only {100 * lost:.2f}% of its peak at constant twist, "
                f"expected at least {100 * args.min_relaxation:.2f}%"
            )

        # Relaxation is monotone for a Prony series held at fixed strain. A small tolerance
        # absorbs the Newton tolerance rather than any real rise.
        tol = 1e-6 * peak
        for a, b in zip(plateau, plateau[1:]):
            if abs(b.torque) > abs(a.torque) + tol:
                failures.append(
                    f"the torque rose from {abs(a.torque):.6e} to {abs(b.torque):.6e} "
                    f"between t = {a.time:g} and t = {b.time:g} at constant twist"
                )
                break

    # --- release: no torque, and the twist recovers ----------------------------------------
    if freed:
        worst = max(abs(r.torque) for r in freed)
        print(f"[validate] release: largest |T| after release {worst:.6e} (peak while held {peak:.6e})")
        if worst > args.released_torque_tol * peak:
            failures.append(
                f"the released face still carries a torque of {worst:.6e}, "
                f"more than {args.released_torque_tol:g} of the peak {peak:.6e}"
            )

        at_release = held[-1].twist_magnitude
        at_end = freed[-1].twist_magnitude
        if at_release <= 0:
            failures.append("the control point was not displaced at release")
        else:
            recovered = (at_release - at_end) / at_release
            print(
                f"[validate] recovery: control point |u_yz| {at_release:.6e} -> {at_end:.6e} "
                f"({100 * recovered:.2f}% recovered over {freed[-1].time - freed[0].time:g} s)"
            )
            if recovered < args.min_recovery:
                failures.append(
                    f"the control point recovered only {100 * recovered:.2f}% of its twist, "
                    f"expected at least {100 * args.min_recovery:.2f}%"
                )
            # Overshoot is a failure for a quasi-static run, where the recovery is a monotone
            # creep, and expected for a dynamic one, where the released end rings through the
            # undeformed state and back. So the check needs to know which it is looking at.
            if recovered > 1.0 + 1e-9 and not dynamic_run:
                failures.append(
                    "the control point overshot through the undeformed state in a quasi-static run"
                )
    else:
        print("[validate] no release phase in this history (control run)")

    # --- recover the Prony weights from the decay --------------------------------------------
    if args.case and creep_run:
        print(
            "[validate] creep run: skipping the Prony recovery -- it reads the relaxation "
            "function off a decaying torque at fixed twist, and here neither is fixed"
        )
    elif args.case and dynamic_run:
        print(
            "[validate] dynamic run: skipping the Prony recovery -- the fit assumes the torque is "
            "proportional to the relaxation function, which it is not once inertia contributes"
        )
    elif args.case:
        try:
            import yaml
        except ImportError:
            print("[validate] PyYAML not available; skipping the Prony recovery check")
        else:
            with open(args.case) as fh:
                case = yaml.safe_load(fh)

            material = case.get("material") or {}
            terms = material.get("prony") or []
            taus = [float(t["tau"]) for t in terms]
            g_in = [float(t["g"]) for t in terms]
            ramp = float((case.get("torsion") or {}).get("ramp_time", 0.0))

            # WLF shifts the relaxation times, so the basis to fit in is tau/a_T, not the
            # nominal tau. Fitting the nominal ones against a shifted response is fitting the
            # wrong exponentials and returns weights that mean nothing.
            wlf = material.get("wlf") or {}
            if wlf.get("enabled"):
                dT = float(wlf.get("temperature", 20.0)) - float(wlf.get("T_ref", -54.29))
                denom = float(wlf.get("C2", 47.4781)) + dT
                a_T = 10.0 ** (float(wlf.get("C1", 16.6253)) * dT / denom) if abs(denom) > 1e-10 else 1.0
                taus = [t / a_T for t in taus]
                print(f"[validate] WLF is on: a_T = {a_T:.4g}, fitting shifted tau = "
                      + ", ".join(f"{t:.4g}" for t in taus))

            if not taus:
                print("[validate] the case has no Prony terms; nothing to recover")
            elif len(plateau) < 2 * (len(taus) + 1):
                failures.append(
                    f"the hold phase has {len(plateau)} samples, too few to recover "
                    f"{len(taus) + 1} coefficients"
                )
            else:
                g_out, rms = fit_relaxation(plateau, taus, ramp)
                if abs(plateau[0].torque) <= 0:
                    failures.append("the torque at the start of the hold is zero")
                else:
                    g_inf_in = 1.0 - sum(g_in)

                    print(
                        f"[validate] recovered relaxation function from the hold "
                        f"({len(plateau)} samples, relative rms {rms:.3e}):"
                    )
                    print(f"    g_inf: asked {g_inf_in:.4f}  recovered {g_out[0]:.4f}")
                    for i, tau in enumerate(taus):
                        print(
                            f"    g[{i}] (tau={tau:g}): asked {g_in[i]:.4f}  "
                            f"recovered {g_out[i + 1]:.4f}"
                        )

                    # The recovered weights are the measurement; the torque is proportional to
                    # the relaxation function only to the extent that the response is quasi
                    # linear, so a large twist will not reproduce the inputs exactly and the
                    # tolerance is a statement about how far from linear the case is.
                    if not gate_recovery:
                        print("[validate] the case opts out of gating on the recovered weights; "
                              "reported above for information only")
                    elif abs(g_out[0] - g_inf_in) > args.fit_tol:
                        failures.append(
                            f"recovered g_inf {g_out[0]:.4f} differs from the case's "
                            f"{g_inf_in:.4f} by more than {args.fit_tol}"
                        )
                    for i, tau in enumerate(taus):
                        if not gate_recovery:
                            break
                        if abs(g_out[i + 1] - g_in[i]) > args.fit_tol:
                            failures.append(
                                f"recovered g[{i}] (tau={tau:g}) {g_out[i + 1]:.4f} differs "
                                f"from the case's {g_in[i]:.4f} by more than {args.fit_tol}"
                            )

    # --- against the control run -----------------------------------------------------------
    if args.control:
        control = read_history(args.control)
        by_time = {round(r.time, 9): r for r in control}
        release_time = freed[0].time if freed else None

        worst_before = 0.0
        for r in held:
            other = by_time.get(round(r.time, 9))
            if other is None:
                continue
            worst_before = max(worst_before, abs(r.torque - other.torque))

        print(f"[validate] control: largest |T_release - T_hold| before release {worst_before:.6e}")
        if worst_before > 1e-6 * peak:
            failures.append(
                "the released run and the held control disagree before the release, "
                "so the difference after it cannot be attributed to the release"
            )

        if release_time is not None:
            after = [r for r in control if r.time > release_time]
            if after and abs(after[-1].torque) <= args.released_torque_tol * peak:
                failures.append("the control run lost its torque too, so it was not actually held")

    return report(failures, rows)


def report(failures: List[str], rows: List[Row]) -> int:
    if failures:
        print(f"\n[validate] FAILED after {len(rows)} steps:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print(f"\n[validate] OK ({len(rows)} steps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
