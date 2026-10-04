#!/usr/bin/env python3
"""Fail if src/kernels/ reaches outside itself, or if src/ grows a directory nobody named.

DESIGN.md's first requirement for src/kernels/ is that it be "header only self-contained code
with templated types and localized macros (no library dependencies allowed, exceptions for CUDA,
OpenMP or other wrappers)". Reaching that took a long cascade -- the sweeps split into
range-taking kernels and launchers, fourteen staging helpers converted, three staging types out
of 38 signatures, and the launchers moved to the front end -- and every bit of it is undone by
one convenient #include.

So this is a test rather than a note. The property is cheap to state and cheap to check: no
header under src/kernels/ may include a quoted path that does not start with kernels/, and no
translation unit may live there at all, because every .cpp under src/ is compiled into the
operator library and a kernel .cpp would drag a whole kernel family into it.

The directory shape is checked for the same reason. src/ had eight top-level directories named
after elements, benchmarks and whatever arrived next; it now has the ones DESIGN.md asks for,
plus support/ for what fits nowhere else and ss/ for the semi-structured layer that is not yet
converted. A new one should be a deliberate act, not a side effect.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]          # src/
ALLOWED_DIRS = {
    "cases", "drivers", "frontend", "kernels", "support", "tests",
    # Not in DESIGN.md's list, and deliberately named here rather than silently tolerated: the
    # semi-structured sweeps still take staging objects, so they cannot move into kernels/ yet.
    # When they are converted this entry goes away with the directory.
    "ss",
}

def main():
    bad = []

    found = {p.name for p in (ROOT).iterdir() if p.is_dir()}
    for extra in sorted(found - ALLOWED_DIRS):
        bad.append(f"src/{extra}/ is not a directory this layout names. Add it to ALLOWED_DIRS "
                   f"in {pathlib.Path(__file__).name} with a reason, or put its contents where "
                   f"they belong.")

    kernels = ROOT / "kernels"
    for src in sorted(kernels.rglob("*")):
        if src.suffix in (".cpp", ".cu", ".cc"):
            bad.append(f"{src.relative_to(ROOT.parent)} is a translation unit under src/kernels/, "
                       f"which is header-only.")
        if src.suffix not in (".hpp", ".h", ".cuh"):
            continue
        for n, line in enumerate(src.read_text().splitlines(), 1):
            t = line.strip()
            if not t.startswith("#include") or '"' not in t:
                continue
            path = t.split('"')[1]
            if not path.startswith("kernels/"):
                bad.append(f'{src.relative_to(ROOT.parent)}:{n}: #include "{path}" reaches '
                           f"outside src/kernels/. The kernels carry no dependency on the rest "
                           f"of the tree; if a launcher needs this, the launcher belongs in "
                           f"src/frontend/staging/.")

    if bad:
        print("FAILED: src/kernels/ is not self-contained\n", file=sys.stderr)
        for b in bad:
            print("  " + b, file=sys.stderr)
        return 1
    n_hdr = sum(1 for _ in kernels.rglob("*.hpp"))
    print(f"PASSED: {n_hdr} headers under src/kernels/, none reaching outside it; "
          f"src/ has only the directories this layout names")
    return 0


if __name__ == "__main__":
    sys.exit(main())
