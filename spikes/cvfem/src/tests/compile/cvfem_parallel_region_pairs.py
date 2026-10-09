#!/usr/bin/env python3
"""A sweep call must not be left outside the parallel region that was meant to cover it.

`#pragma omp parallel` applies to the ONE statement that follows it. Splitting a sweep into an
affine and an isoparametric half turns one statement into two, and if the braces are not added
the second call runs after the region has closed -- once, on the encountering thread, with every
macro element of its half to itself.

That failure is quiet in every way this spike normally checks. The answer is right: all the
elements are still processed exactly once, the summation order inside the serial half is fixed,
and the fingerprints are taken at one thread where there is no difference to see. It is only
slower, and only on a mesh that has elements of that kind at all -- a box has none, so the whole
suite can pass while half of a sweep pair runs serially.

It happened: the naive residual pair went in unbraced and was found by a brace pass run for a
different sweep, not by a test. Six more were in the benchmark.

So this gate reads the sources and fails if a bare `#pragma omp parallel` governs a statement
that is followed by another sweep call. What it looks for is deliberately narrow -- a following
statement naming an `_affine(` or `_isoparam(` sweep, or one taking a cvfem_range -- because a
frontend helper that owns its own parallel region is a legitimate next statement:
sscvfem_apply_transient_action is called that way eight times and must stay allowed.
"""

import re
import sys
from pathlib import Path

# parents[3], the spike root: parents[2] is src/ itself, and the globs below are
# written relative to the root. Getting this wrong makes the gate scan nothing and
# pass unconditionally, which is exactly what it did on its first run.
ROOT = Path(__file__).resolve().parents[3]

# A call that must be INSIDE a parallel region: a split sweep half, or anything handed a range.
SWEEP = re.compile(r"\b\w+_(?:affine|isoparam)\s*\(|cvfem_range_split\s*\(|sscvfem_(?:affine|isoparam)_range\s*\(")


def offenders(path):
    lines = path.read_text().split("\n")
    out = []
    for i, line in enumerate(lines):
        if line.strip() != "#pragma omp parallel":
            continue
        if i + 1 >= len(lines) or lines[i + 1].strip().startswith("{"):
            continue  # a block: every statement in it is inside the region
        # the single statement the pragma governs
        j = i + 1
        while j < len(lines) and not lines[j].rstrip().endswith(");"):
            j += 1
        if j + 1 >= len(lines):
            continue
        nxt = lines[j + 1].strip()
        if SWEEP.search(nxt):
            out.append((i + 1, nxt[:72]))
    return out


def main():
    bad = 0
    for pattern in ("src/**/*.hpp", "src/**/*.cpp", "src/**/*.cu", "src/**/*.cuh"):
        for path in sorted(ROOT.glob(pattern)):
            for lineno, nxt in offenders(path):
                rel = path.relative_to(ROOT)
                print(f"{rel}:{lineno}: `#pragma omp parallel` governs one statement, but the next")
                print(f"{' ' * (len(str(rel)) + len(str(lineno)) + 2)}  one is a sweep call that will run OUTSIDE the region:")
                print(f"{' ' * (len(str(rel)) + len(str(lineno)) + 2)}  {nxt}")
                print("    Brace the pair.")
                bad += 1
    if bad:
        print(f"\ncvfem_parallel_region_pairs: {bad} sweep call(s) left outside a parallel region")
        return 1
    print("cvfem_parallel_region_pairs: PASSED, every sweep call sits inside its parallel region")
    return 0


if __name__ == "__main__":
    sys.exit(main())
