#!/usr/bin/env python3
"""Fail the build when a lane-blocked kernel's object holds no vector arithmetic.

WHY THIS READS THE OBJECT INSTEAD OF ASKING THE COMPILER. The obvious gate is
-Rpass-missed=loop-vectorize with -Werror=pass-failed, and it does not work on these kernels.
Measured on this tree: with the limiter-stats atomic still inside the lane loop,
-Rpass-analysis=loop-vectorize did name the three causes, but -Rpass-missed alone produced one
remark for the whole translation unit; once that barrier was removed the loop vectoriser stopped
reporting anything at all, because a lane loop of CVFEM_HEX8_VEC_SIZE iterations is a
compile-time constant trip count -- it is unrolled and the straight-line body is vectorised by
the SLP pass, which emitted 661 remarks for a kernel that produced zero vector instructions. A
gate built on remarks is therefore silent exactly when it matters. The emitted object is not.

This is also what the tree already did by hand. The generator's notes record judging a kernel by
"6000 instructions and not one NEON register", and the note on cvfem_hex8_conv_face_lane records
diagnosing an out-of-line call that left the whole lane loop scalar. The second of those was
written, fixed with __attribute__((flatten)), and the loop was still scalar afterwards for a
different reason nobody measured again. A comment is not a gate.

Only one thing is enforced: a kernel declared vectorised must contain vector floating-point
arithmetic. The share is reported, never enforced -- it moves with the compiler and with how much
scalar setup surrounds the loop, and a threshold on it would be a number nobody could defend.
"""
import argparse
import re
import subprocess
import sys

# AArch64 writes NEON as fmla.2d / fadd.4s; x86 as vfmadd213pd %zmm.., and its scalar forms are
# the ...sd/...ss suffixes. Both families are matched so the gate means the same thing on the
# Grace nodes that decide performance and on the laptop that compiles.
VEC_FP = re.compile(r"\b(?:f[a-z]+|[a-z]+)\.[0-9]+[dsh]\b" r"|\bv[a-z]+p[ds]\b")
SCALAR_FP = re.compile(
    r"\b(?:fmul|fadd|fsub|fmadd|fmsub|fnmadd|fnmsub|fmla|fmls|fdiv|fsqrt|fabs|fmax|fmin|fneg)"
    r"\s+[dsh][0-9]"
    r"|\bv?(?:mul|add|sub|div|sqrt|max|min)s[ds]\b"
)


def disassemble(obj, objdump):
    for tool in (objdump, "llvm-objdump", "objdump"):
        if not tool:
            continue
        try:
            r = subprocess.run([tool, "-d", obj], capture_output=True, text=True)
        except OSError:
            continue
        if r.returncode == 0 and r.stdout:
            return r.stdout
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objdump", default="")
    # name=path, one per kernel. The build compiles one object per kernel precisely so that a
    # scalar kernel cannot hide behind its neighbours' vector instructions: three wrappers in one
    # object were tail-merged into a single shared body the first time this was tried.
    ap.add_argument("kernels", nargs="+", metavar="NAME=OBJECT")
    # Kernels known to emit no vector arithmetic, which the build still compiles and still
    # measures. An exemption is a debt, so it is enforced in BOTH directions: an exempt kernel
    # that starts vectorising also fails, with instructions to delete its entry. That is what
    # stops the list outliving the problem, which is the usual fate of a skip list.
    ap.add_argument("--expect-scalar", action="append", default=[], metavar="NAME")
    a = ap.parse_args()

    rows, failed, unreadable, fixed = [], [], [], []
    expect_scalar = set(a.expect_scalar)
    for spec in a.kernels:
        name, _, obj = spec.partition("=")
        text = disassemble(obj, a.objdump)
        if text is None:
            unreadable.append(name)
            continue
        insns = sum(1 for ln in text.splitlines() if re.match(r"\s+[0-9a-f]+:\s", ln))
        vec = len(VEC_FP.findall(text))
        sca = len(SCALAR_FP.findall(text))
        rows.append((name, insns, vec, sca))
        if vec == 0 and name not in expect_scalar:
            failed.append(name)
        if vec > 0 and name in expect_scalar:
            fixed.append(name)

    w = max((len(r[0]) for r in rows), default=6)
    print(f"{'kernel'.ljust(w)}  {'insns':>8} {'vector_fp':>10} {'scalar_fp':>10}  share  note")
    for name, insns, vec, sca in rows:
        share = f"{100 * vec // (vec + sca)}%" if vec + sca else "n/a"
        note = "KNOWN SCALAR -- owed" if name in expect_scalar else ""
        print(f"{name.ljust(w)}  {insns:>8} {vec:>10} {sca:>10}  {share:>5}  {note}")

    # An object the gate could not read is not a pass. Silently skipping is how a check becomes
    # decoration: it would go green on a machine with no objdump and nobody would notice.
    sys.stdout.flush()
    if unreadable:
        print(f"\nFAILED: could not disassemble: {', '.join(unreadable)}", file=sys.stderr)
        return 2
    if fixed:
        print(
            f"\nFAILED: these kernels now vectorise and are still listed as known-scalar: "
            f"{', '.join(fixed)}\n"
            "Remove them from CVFEM_VEC_GATE_KNOWN_SCALAR in CMakeLists.txt so the gate starts\n"
            "enforcing what the code now does.",
            file=sys.stderr,
        )
        return 3
    if failed:
        print(
            f"\nFAILED: no vector floating-point arithmetic in: {', '.join(failed)}\n"
            "The lane loop is running scalar. Compile that kernel with\n"
            "  -Rpass-analysis=loop-vectorize -Rpass-analysis=slp-vectorizer\n"
            "to see what the compiler refused, and note that a loop containing a call, an\n"
            "`#pragma omp atomic`, or control flow it cannot turn into a select will not\n"
            "vectorise however it is annotated.",
            file=sys.stderr,
        )
        return 1
    if expect_scalar:
        print(
            f"\nPASSED, with {len(expect_scalar)} kernel(s) exempt and owed: "
            f"{', '.join(sorted(expect_scalar))}"
        )
    else:
        print("\nPASSED: every gated kernel emits vector floating-point arithmetic")
    return 0


if __name__ == "__main__":
    sys.exit(main())
