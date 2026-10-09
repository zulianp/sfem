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

    # A RANGE-DRIVEN SWEEP MUST NOT CARRY A TRACE SCOPE.
    #
    # CVFEM_TRACE_SCOPE expands to a smesh::ScopedEvent whose destructor records on
    # smesh::Tracer::instance() -- a process-wide singleton that appends with no lock. A sweep
    # that takes a cvfem_range is entered once per thread by its launcher, so the scope is
    # constructed and destroyed concurrently and the tracer's container races. It surfaced as one
    # segfault in eight runs of cvfem_sshex8_agree, did not reproduce under a direct run of the
    # same binary, and took a bisect to locate. The scope belongs in the launcher, which is also
    # where one event per operator call is what a reader of the trace wants.
    for src in sorted(kernels.rglob("*.hpp")):
        text = src.read_text()
        for m in re.finditer(r"^(?:template <[^>]*>\n)?(?:static |inline )[^;{\n]*?\b(\w+)\(",
                             text, re.M):
            k = text.find("(", m.end() - 1)
            depth, j = 0, k
            while j < len(text):
                if text[j] == "(": depth += 1
                elif text[j] == ")":
                    depth -= 1
                    if depth == 0: break
                j += 1
            if text[j + 1:].lstrip(" \n")[:1] != "{": continue
            stop = text.find("\n}\n", m.start())
            if stop < 0: continue
            head, body = text[m.start():j + 1], text[j:stop]
            if "cvfem_range" in re.sub(r"//[^\n]*", "", head) and "CVFEM_TRACE_SCOPE" in body:
                ln = text[:m.start()].count("\n") + 1
                bad.append(f"{src.relative_to(ROOT.parent)}:{ln}: {m.group(1)} takes a "
                           f"cvfem_range and carries a CVFEM_TRACE_SCOPE. The launcher enters it "
                           f"once per thread and the tracer is an unlocked singleton; put the "
                           f"scope in the launcher.")

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
    #
    # This is the one check here that needs a git checkout, and the spike is routinely run from a
    # tree that is not one: every Alps run works from an rsync'd copy with no .git, where
    # `git ls-files` exits 128. Treating that as a layout failure made this gate fail on the
    # cluster while passing locally on the same commit -- a red gate saying nothing about the
    # code. So the check is LOST rather than failed, and lost loudly: the summary line says what
    # it could not do instead of claiming a check it never made. It is not weakened where it can
    # run, which is every local ctest and therefore every commit.
    skipped = ""
    if want:
        try:
            idx = subprocess.run(["git", "ls-files", "-s", "--", *want], cwd=ROOT.parent,
                                 capture_output=True, text=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            skipped = (f"script modes NOT CHECKED: no readable git index at {ROOT.parent} "
                       f"-- expected in an rsync'd or exported tree, a real failure in a clone")
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

    # AND NO KERNEL MAY CALL A FUNCTION DEFINED OUTSIDE src/kernels/.
    #
    # The include check above is necessary and was not sufficient. Ten pack staging functions --
    # the gathers, the fills and the scatter that every pack sweep runs around its element
    # kernel -- lived in src/frontend/staging/ and the sweeps called them anyway, reached by
    # INCLUDE ORDER: a launcher pulls the staging header in before the kernel header, so the
    # names are in scope by the time the kernel is parsed. There was no include from kernels/ to
    # object to, so this test passed for as long as that arrangement lasted. Four of those ten
    # named smesh::idx_t, which this very test forbids in kernels/ code, from inside a call tree
    # that kernels/ owned.
    #
    # The check is by NAME, against the set of functions defined elsewhere under src/. That
    # cannot see a name defined in no SFEM source -- a libc or OpenMP call, or one of the
    # contract aliases -- which is the right blind spot to have: those are what DESIGN.md's
    # parenthetical allows ("exceptions for CUDA, OpenMP or other wrappers").
    # NAMES THE INCLUDER SUPPLIES BY CONTRACT ARE NOT VIOLATIONS. cvfem_phases.hpp says so at
    # its head: the enabled expansion "can name PhaseAcc, g_breakdown, wall_time, g_phase and
    # the PH_ enumerators without this header depending on whatever defines them ... the same
    # contract the kernels already use for scalar_t". That is deliberate and is what lets the
    # instrumentation live in src/kernels/ while the breakdown's state lives with the driver
    # that owns the flag. The check below found them on its first run, which is how this list
    # came to exist -- and moving one of them here was the wrong fix, briefly made.
    #
    # Anything added here needs that kind of reason: a documented contract, not a convenience.
    CONTRACT = {"wall_time", "phase_now", "phase_reset"}
    DEF = re.compile(r"^(?:template <[^>\n]*>\s*\n)?(?:static |inline )[^;{\n]*?\b(\w+)\s*\(",
                     re.M)
    outside = {}
    for src in sorted(ROOT.rglob("*.hpp")):
        if "kernels" in src.relative_to(ROOT).parts[:1]:
            continue
        for m in DEF.finditer(src.read_text()):
            outside.setdefault(m.group(1), src.relative_to(ROOT.parent))
    inside = set()
    for src in sorted(kernels.rglob("*")):
        if src.suffix in (".hpp", ".h", ".cuh"):
            inside |= {m.group(1) for m in DEF.finditer(src.read_text())}
    reachable_only_by_order = {}
    for src in sorted(kernels.rglob("*")):
        # HOST HEADERS ONLY. A .cuh has the CUDA runtime's names in scope -- min, max, the
        # intrinsics -- and this check cannot tell CUDA's min() from one of ours that happens
        # to share the name, so it would report a violation that is not one.
        if src.suffix not in (".hpp", ".h"):
            continue
        for n, line in enumerate(src.read_text().splitlines(), 1):
            code = line.split("//")[0]
            for m in re.finditer(r"(?<![\w.>:])(\w+)\s*(?:<[^<>()]*>)?\s*\(", code):
                name = m.group(1)
                if name in inside or name not in outside or name in CONTRACT:
                    continue
                if name in ("if", "for", "while", "switch", "return", "sizeof", "static_cast",
                            "const_cast", "reinterpret_cast", "alignas", "assert"):
                    continue
                reachable_only_by_order.setdefault(name, (src.relative_to(ROOT.parent), n))
    for name, (where, n) in sorted(reachable_only_by_order.items()):
        bad.append(f"{where}:{n}: calls {name}(), which is defined in "
                   f"{outside[name]} -- outside src/kernels/. It is reachable only because a "
                   f"launcher includes that header first, which makes this directory's "
                   f"self-containment an accident of include order. Move the definition here, "
                   f"or make the caller a launcher.")

    if bad:
        print("FAILED: the layout this test holds has been broken\n", file=sys.stderr)
        for b in bad:
            print("  " + b, file=sys.stderr)
        return 1
    n_hdr = sum(1 for _ in kernels.rglob("*.hpp"))
    modes = (skipped if skipped
             else f"all {len(want)} shell tests ctest runs are executable in the index")
    print(f"PASSED: {n_hdr} headers under src/kernels/, none reaching outside it by "
          f"include or by call; "
          f"src/ has only the directories this layout names; "
          f"{modes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
