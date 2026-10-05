#!/usr/bin/env python3
"""Every kernel that names a computation type must take it as a template parameter.

DESIGN.md's third correction: "the kernels should be templated as well. They should support
different types for the computation, template scalar_t, geom_t, idx_t, etc... (in a short time
we would like to try single precision kernels as well)".

The three mixed-precision gates show that the templated kernels RUN at single precision and
agree. They cannot show that nothing was missed: a definition still spelling the build's
`scalar_t` compiles perfectly, and it is only reached at the other precision if some f32 test
happens to instantiate it. Most of src/kernels/ is not reached by those three gates. So this
walks the tree instead and fails on any definition whose signature names a computation type
without declaring it.

WHAT IT DELIBERATELY DOES NOT REQUIRE. A definition that names none of those types -- a lattice
index, a thread-team size, an element count, a pack's node count -- is correctly untemplated, and
demanding a parameter it does not use would make every call site spell one.

THE SHARED CONSTANT TABLES ARE THE ONE KNOWN EXCEPTION, and it is listed rather than hidden.
Accessors like cvfem_hex8_snet_tbl() return a reference to a `const double[...]` table, so an f32
chain that reads one promotes to double. Templating them means a copy of each table per
precision, which is a decision about storage rather than a missing parameter, and it is not one
to make silently inside a refactor -- the measured consequence today is that the f32 comparisons
agree to f32's own round-off, i.e. the promotion costs accuracy nowhere and lane count nowhere.
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
KERNELS = ROOT / "src" / "kernels"

TYPES = ("scalar_t", "geom_t", "idx_t", "count_t", "pack_idx_t", "jacobian_t")

# Accessors returning a reference to a shared constant table; see the note above.
TABLE = re.compile(r"const\s+(?:int|double|Hex8Face)\s*\(\s*&\s*\w+\s*\)")


def close_paren(text, i):
    depth = 0
    while i < len(text):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


DEF = re.compile(r"^(?:static|inline|template)\b", re.M)


def offenders(path):
    text = path.read_text()
    lines = text.split("\n")
    out = []
    for i, line in enumerate(lines):
        if not (line.startswith("static ") or line.startswith("inline ")):
            continue
        if "(" not in line or "constexpr" in line or " using " in line:
            continue
        # The template header may be several lines (`template <bool RC = false, ...,\n
        # typename scalar_t = ::scalar_t>`) and may be separated from the definition by comment
        # or blank lines. Walk back over the comments first, then over the header's own lines.
        j = i - 1
        while j >= 0 and (not lines[j].strip() or lines[j].strip().startswith("//")):
            j -= 1
        header = []
        while j >= 0:
            stripped = lines[j].strip()
            header.insert(0, lines[j])
            if stripped.startswith("template <"):
                break
            # Anything that is not a continuation of a template header ends the search.
            if (stripped.endswith(";") or stripped.endswith("}") or stripped.endswith("{")
                    or stripped.startswith("static ") or stripped.startswith("inline ")
                    or stripped.startswith("#")):
                header = []
                break
            j -= 1
        else:
            header = []

        start = text.index(line, sum(len(l) + 1 for l in lines[:i]))
        op = text.index("(", start)
        cp = close_paren(text, op)
        sig = line + "\n" + text[op:cp + 1]
        if TABLE.search(sig):
            continue
        declared = "\n".join(header)
        for t in TYPES:
            if re.search(r"(?<![\w])" + t + r"(?![\w])", sig) and not re.search(
                    r"typename\s+" + t + r"(?![\w])", declared):
                out.append((i + 1, t, line.strip()[:70]))
                break
    return out


def main():
    bad = 0
    checked = 0
    for path in sorted(KERNELS.rglob("*.hpp")) + sorted(KERNELS.rglob("*.cuh")):
        checked += 1
        for lineno, t, what in offenders(path):
            print(f"{path.relative_to(ROOT)}:{lineno}: names `{t}` but does not declare it")
            print(f"    {what}")
            bad += 1
    if bad:
        print(f"\ncvfem_kernels_are_templated: {bad} definition(s) bound to the build's types")
        return 1
    print(f"cvfem_kernels_are_templated: PASSED, {checked} headers; every definition that names a "
          f"computation type declares it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
