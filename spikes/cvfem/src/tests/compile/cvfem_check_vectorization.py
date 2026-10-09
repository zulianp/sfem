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


# A SCALAR OP'S VECTOR TWIN. The share column alone cannot tell two very different things apart,
# and that mattered: `jv_lane_defcor_unlimited` read 58% and looked half-scalar, while the loop
# vectoriser reported every one of its twelve lane loops vectorised at width 2. Pairing the
# mnemonics showed why -- fmadd 672 against fmla.2d 672, fmul 505 against 504, fnmsub 12 against
# fmls.2d 12, fneg 12 against 12. EVERY scalar op had an exactly matching vector one, which is
# not scalarised arithmetic but a complete duplicate of the body: the vectoriser's scalar
# remainder loop. The lane count is a constexpr 16 and the width is 2, so that remainder can
# never execute; it costs object size and nothing else.
#
# What a shortfall actually looks like is the other case, and this tree had it: the two Jacobian
# limiter arms emitted ZERO vector instructions because a lambda inside cvfem_hex8_scs_defcor_jv
# was called 36 times from inside the lane loop. There the scalar ops have no vector twin at all.
#
# So the number to look at is UNPAIRED scalar arithmetic -- a scalar op whose vector form is
# absent from the object. A dead remainder is fully paired and reads zero here; a kernel that
# really went scalar reads all of it. Forcing the remainder away is not worth it: simdlen(16)
# removed the shortfall on paper, 58% to 92%, by growing the kernel from 13,631 instructions to
# 104,228, which trades I-cache for a cosmetic number.
TWIN = {
    "fmadd": ("fmla",), "fmsub": ("fmls",), "fnmsub": ("fmls", "fmla"), "fnmadd": ("fmla", "fmls"),
    "fmul": ("fmul",), "fadd": ("fadd",), "fsub": ("fsub",), "fdiv": ("fdiv",),
    "fneg": ("fneg",), "fabs": ("fabs",), "fsqrt": ("fsqrt",),
    "fmax": ("fmax",), "fmin": ("fmin",), "fmaxnm": ("fmaxnm",), "fminnm": ("fminnm",),
    # A compare-and-select becomes a vector compare plus a bitwise select.
    "fcmp": ("fcmgt", "fcmlt", "fcmge", "fcmle", "fcmeq"),
    "fccmp": ("fcmgt", "fcmlt", "fcmge", "fcmle", "fcmeq"),
    "fcsel": ("bsl", "bit", "bif", "fmaxnm", "fminnm"),
    "fmov": ("mov", "dup", "fmov"),
    "fcvt": ("fcvt",), "scvtf": ("scvtf",), "ucvtf": ("ucvtf",),
    "frinta": ("frinta",), "frintm": ("frintm",), "frintp": ("frintp",), "frintz": ("frintz",),
}


def classify(text):
    """Return (instructions, vector FP, scalar FP, unpaired scalar FP) for a disassembly."""
    insns = vec = 0
    scalar_ops, vector_ops = {}, set()
    for ln in text.splitlines():
        m = INSN.match(ln)
        if not m:
            continue
        mnemonic, operands = m.group(1), m.group(2)
        insns += 1
        base = mnemonic.split(".")[0]
        if (
            VEC_OPERAND.search(operands)
            or VEC_MNEMONIC.match(mnemonic)
            or (VEC_X86.match(mnemonic) and VEC_X86_REG.search(operands))
        ):
            vec += 1
            vector_ops.add(base)
        elif mnemonic in FP_MNEMONIC and SCALAR_REG.search(operands):
            scalar_ops[mnemonic] = scalar_ops.get(mnemonic, 0) + 1
        elif SCALAR_X86.match(mnemonic) and VEC_X86_REG.search(operands):
            scalar_ops[mnemonic] = scalar_ops.get(mnemonic, 0) + 1
    sca = sum(scalar_ops.values())
    # Unpaired: a scalar op none of whose vector forms appears anywhere in this object. On x86 the
    # twin is the same mnemonic with a packed suffix, which the VEC_X86 branch has already put in
    # vector_ops under its own base, so an unknown mnemonic is treated as unpaired rather than
    # silently excused.
    unpaired = sum(n for m, n in scalar_ops.items()
                   if not (set(TWIN.get(m, ())) & vector_ops))
    return insns, vec, sca, unpaired


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
        insns, vec, sca, unp = classify(text)
        # A disassembly this parser recognised no instructions in means the output format is not
        # one of the four above, not that the kernel is empty. Saying so beats reporting zeros.
        if insns == 0:
            unreadable.append(name)
            continue
        rows.append((name, insns, vec, sca, unp))
        # TWO RULES, both scale-free, because the gate's own note is right that a threshold on the
        # share is a number nobody could defend. The first is the original one. The second is what
        # the unpaired column buys: more scalar arithmetic with no vector form than there is
        # vector arithmetic at all means the kernel is mostly running scalar, which the share
        # alone could not distinguish from a dead remainder loop. One stray scalar op against
        # fifteen hundred vector ones is noise; sixteen hundred against none is the defect this
        # gate exists for, and it sat behind an exemption for as long as only the first rule ran.
        if name not in expect_scalar and (vec == 0 or unp > vec):
            failed.append(name)
        if vec > 0 and name in expect_scalar:
            fixed.append(name)

    w = max((len(r[0]) for r in rows), default=6)
    # `unpaired` is the column that means something. `scalar_fp` is reported beside it because the
    # difference between them is the vectoriser's dead remainder, which is worth seeing but is not
    # a shortfall: see the note on TWIN above.
    print(f"{'kernel'.ljust(w)}  {'insns':>8} {'vector_fp':>10} {'scalar_fp':>10} "
          f"{'unpaired':>9}  share  note")
    for name, insns, vec, sca, unp in rows:
        share = f"{100 * vec // (vec + sca)}%" if vec + sca else "n/a"
        note = "KNOWN SCALAR -- owed" if name in expect_scalar else ""
        if unp and not note:
            note = f"{unp} scalar op(s) with NO vector form in the object"
        print(f"{name.ljust(w)}  {insns:>8} {vec:>10} {sca:>10} {unp:>9}  {share:>5}  {note}")

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
            f"\nFAILED: scalar lane loop in: {', '.join(failed)}\n"
            "Either the object holds no vector floating-point arithmetic at all, or more of its\n"
            "scalar arithmetic has no vector counterpart than it has vector arithmetic.\n"
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
