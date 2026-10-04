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
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]          # src/
ALLOWED_DIRS = {"cases", "drivers", "frontend", "kernels", "support", "tests"}

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
            # NO std::vector UNDER src/kernels/, IN A SIGNATURE OR A LOCAL.
            #
            # Patrick: "No std vector locals!" A kernel's scratch comes from the per-thread
            # arena, thread_scratch<T>(slot, n), which is also the only such mechanism the tree
            # has -- a std::vector local beside it would be a second one. And a std::vector in a
            # signature is what keeps this directory from being free of library types at all.
            # NO LIBRARY NAMESPACE IN THIS DIRECTORY'S CODE.
            #
            # The kernels name scalar_t, idx_t, count_t and geom_t unqualified and the including
            # translation unit supplies them; that is what lets a CUDA .cu, which cannot include a
            # family header, instantiate them. Twenty `smesh::idx_t` and `smesh::count_t`
            # spellings had survived in three headers, and no host build could see it because a
            # host translation unit always has smesh in scope. The first CUDA build in a while
            # failed on "namespace smesh has no member idx_t".
            for q in re.finditer(r"\b(smesh|sfem)::(\w+)", t.split("//")[0]):
                bad.append(f"{src.relative_to(ROOT.parent)}:{n}: {q.group(0)} under src/kernels/. "
                           f"Name the type unqualified; the includer supplies the alias.")
            if "std::vector" in t.split("//")[0]:
                bad.append(f"{src.relative_to(ROOT.parent)}:{n}: std::vector under src/kernels/. "
                           f"Scratch comes from thread_scratch<T>(slot, n); a buffer a caller "
                           f"owns arrives as a pointer.")
            if not t.startswith("#include") or '"' not in t:
                continue
            path = t.split('"')[1]
            if not path.startswith("kernels/"):
                bad.append(f'{src.relative_to(ROOT.parent)}:{n}: #include "{path}" reaches '
                           f"outside src/kernels/. The kernels carry no dependency on the rest "
                           f"of the tree; if a launcher needs this, the launcher belongs in "
                           f"src/frontend/staging/.")

    # EVERY SHELL TEST CTEST RUNS DIRECTLY MUST BE EXECUTABLE IN THE INDEX, NOT JUST ON DISK.
    #
    # CMake registers these as `COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/<script>`, so a script
    # without the bit fails as BAD_COMMAND -- "Process not started ... [permission denied]" --
    # and ctest reports it as a failure rather than a skip. Three of the nine were committed
    # 100644 and had only survived because the bit happened to be set in one working tree;
    # cvfem_warped_geometry.sh, the only oracle on a non-affine mesh, was among them. The test
    # reads the INDEX rather than the filesystem, because that is what a fresh clone gets.
    cml = (ROOT.parent / "CMakeLists.txt").read_text()
    want = sorted(set(re.findall(r"COMMAND \$\{CMAKE_CURRENT_SOURCE_DIR\}/(\S+\.sh)", cml)))
    if want:
        try:
            idx = subprocess.run(["git", "ls-files", "-s", "--", *want], cwd=ROOT.parent,
                                 capture_output=True, text=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError) as e:
            bad.append(f"could not read the git index to check script modes: {e}")
        else:
            modes = {}
            for line in idx.splitlines():
                head, _, path = line.partition("\t")
                modes[path] = head.split()[0]
            for f in want:
                m = modes.get(f)
                if m is None:
                    bad.append(f"{f} is registered as a ctest command but is not tracked by git.")
                elif m != "100755":
                    bad.append(f"{f} is mode {m} in the git index but ctest runs it directly, "
                               f"so a fresh clone gets BAD_COMMAND. "
                               f"Fix with: git update-index --chmod=+x {f}")

    if bad:
        print("FAILED: the layout this test holds has been broken\n", file=sys.stderr)
        for b in bad:
            print("  " + b, file=sys.stderr)
        return 1
    n_hdr = sum(1 for _ in kernels.rglob("*.hpp"))
    print(f"PASSED: {n_hdr} headers under src/kernels/, none reaching outside it; "
          f"src/ has only the directories this layout names; "
          f"all {len(want)} shell tests ctest runs are executable in the index")
    return 0


if __name__ == "__main__":
    sys.exit(main())
