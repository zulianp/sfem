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

# COUNTING IS PER INSTRUCTION LINE, AND THE ASSEMBLER SYNTAX IS NOT ASSUMED. The first version
# of this matched LLVM's Apple-style mnemonic suffix -- "fmla.2d v2, v0, v3" -- and read a flat
# zero for every kernel on the Grace nodes, where GNU objdump writes the same instruction as
# "fmla v2.2d, v0.2d, v3.2d" with the lane suffix on the REGISTERS. A gate that reports no vector
# arithmetic on the machine whose performance is the point is worse than no gate, so the forms are
# enumerated explicitly and anything unrecognised counts as neither.
#
#   AArch64, GNU:    fmla  v2.2d, v0.2d, v3.2d        NEON lanes on the operands
#   AArch64, LLVM:   fmla.2d v2, v0, v3               lanes on the mnemonic
#   AArch64, SVE:    fmla  z1.d, p0/m, z2.d, z3.d     scalable, predicated
#   x86-64:          vfmadd213pd %zmm1, %zmm2, %zmm3  packed suffix on the mnemonic
#
# Scalar floating point is the counterpart: a d/s/h register on AArch64, an ...sd/...ss mnemonic
# on x86. It is reported for context and never enforced.
INSN = re.compile(r"^\s*[0-9a-f]+:\s+(?:[0-9a-f]{2,8}\s+)*\t?\s*([a-z][a-z0-9._]*)\s*(.*)$")

# NEON/SVE vector registers carrying a lane specifier, in either operand or mnemonic position.
VEC_OPERAND = re.compile(r"\b[vz][0-9]+\.(?:[0-9]+)?[bhsdq]\b")
VEC_MNEMONIC = re.compile(r"^[a-z][a-z0-9]*\.[0-9]+[bhsdq]$")
# x86 packed: the p[sd] suffix distinguishes packed from the scalar s[sd] forms.
VEC_X86 = re.compile(r"^v?[a-z][a-z0-9]*p[sd]$")
VEC_X86_REG = re.compile(r"%[xyz]mm[0-9]+")

FP_MNEMONIC = frozenset(
    "fmul fadd fsub fdiv fsqrt fabs fmax fmin fneg fmadd fmsub fnmadd fnmsub fmla fmls "
    "fcmp fccmp fcsel fmaxnm fminnm frinta frintm frintp frintz fcvt scvtf ucvtf".split()
)
SCALAR_REG = re.compile(r"\b[dsh][0-9]+\b")
SCALAR_X86 = re.compile(r"^v?[a-z][a-z0-9]*s[sd]$")


def classify(text):
    """Return (instruction count, vector FP count, scalar FP count) for a disassembly."""
    insns = vec = sca = 0
    for ln in text.splitlines():
        m = INSN.match(ln)
        if not m:
            continue
        mnemonic, operands = m.group(1), m.group(2)
        insns += 1
        if (
            VEC_OPERAND.search(operands)
            or VEC_MNEMONIC.match(mnemonic)
            or (VEC_X86.match(mnemonic) and VEC_X86_REG.search(operands))
        ):
            vec += 1
        elif mnemonic in FP_MNEMONIC and SCALAR_REG.search(operands):
            sca += 1
        elif SCALAR_X86.match(mnemonic) and VEC_X86_REG.search(operands):
            sca += 1
    return insns, vec, sca


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
        insns, vec, sca = classify(text)
        # A disassembly this parser recognised no instructions in means the output format is not
        # one of the four above, not that the kernel is empty. Saying so beats reporting zeros.
        if insns == 0:
            unreadable.append(name)
            continue
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
