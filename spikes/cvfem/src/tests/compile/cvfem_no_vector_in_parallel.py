"""No std::vector declared inside a `#pragma omp parallel` region.

Scratch in this spike comes from the per-thread arena -- `thread_scratch<T>(slot, n)`, whose
slot register is at the head of kernels/packed/cvfem_pack_scratch.hpp -- and never from a
std::vector local. A vector declared inside a parallel region allocates and frees once per
thread per call, and these calls sit in the V-cycle's inner loop: the Vanka smoother had one
that was `nnodes` wide AND value-initialised to -1 on every application, per thread.

The rule is not about the cost in any one place. It is that the arena makes the per-thread
working set explicit and sized once, which is what lets the register above say what each thread
holds; a vector hides it, and the hiding is what lets an nnodes-wide allocation end up in a
smoother without anyone noticing. A local comment arguing the allocation is cheap at the rate
its routine runs is a reason the cost is small, not an exemption -- the block diagonal's
`pack_diag` carried exactly that comment and is now slot 13.

Two things this deliberately does not flag, because neither is per-thread scratch:

* a vector declared OUTSIDE any parallel region, which is owned data allocated once;
* `resize`/`assign` on a vector that lives in a staging object, which is also allocated once.

The brace tracking ignores text after `//` so that a `#pragma omp parallel` quoted in a comment
-- there is one in frontend/staging/cvfem_hex8_packed_launch.hpp, above a serial pattern build
-- does not open a region that never existed. That false positive is the reason the filter is
here rather than in a one-off grep.
"""
import os, re, glob, sys

# ctest runs with the build directory as the working directory.
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

PARALLEL = re.compile(r'#\s*pragma\s+omp\s+parallel(?!\s+for)')
DECL     = re.compile(r'std::vector<[^;]*>\s*[a-zA-Z_]\w*\s*[({]')

FILES = sorted(set(glob.glob("src/**/*.hpp", recursive=True)) |
               set(glob.glob("src/**/*.cpp", recursive=True)) |
               set(glob.glob("src/**/*.cu", recursive=True)))

bad, scanned = [], 0
for f in FILES:
    if "/subpar/" in f:
        continue   # quarantined variants are not held to this; see subpar/README.md
    scanned += 1
    depth, cur, pending = None, 0, False
    for no, line in enumerate(open(f).read().split("\n"), 1):
        code = line.split("//")[0]
        if PARALLEL.search(code):
            pending = True
        opens, closes = code.count("{"), code.count("}")
        if pending and opens:
            depth = cur + 1
            pending = False
        if depth is not None and DECL.search(code):
            bad.append((f, no, line.strip()[:100]))
        cur += opens - closes
        if depth is not None and cur < depth:
            depth = None

if bad:
    print(f"\n{len(bad)} std::vector local(s) inside a parallel region:\n")
    for f, no, l in bad:
        print(f"  {f}:{no}\n      {l}")
    print("\nTake the buffer from the per-thread arena instead: thread_scratch<T>(slot, n), with a")
    print("slot of its own added to the register in kernels/packed/cvfem_pack_scratch.hpp.")
    sys.exit(1)

print(f"{scanned} sources: every per-thread scratch buffer comes from the arena")
