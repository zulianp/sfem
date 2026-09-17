"""Capture and compare a byte-exact snapshot of every generated kernel source.

This is the refactoring gate for the layering work recorded in
``ARCHITECTURE.html``.
Steps S0-S6 relocate decisions between layers without changing what is emitted,
so the acceptance criterion for each of them is that the generated C/C++ is
unchanged down to the byte.  That is a stronger statement than a passing test
suite, and it settles the performance question by construction: identical source
compiles to identical kernels.

Usage::

    python -m codegen.framework.tools.codegen_snapshot capture reference/
    python -m codegen.framework.tools.codegen_snapshot check reference/
    python -m codegen.framework.tools.codegen_snapshot check-tree

``capture`` regenerates every maintained material into a reference tree.
``check`` regenerates into a temporary tree and compares against that reference
tree, exiting non-zero with a unified diff of what moved -- this is the mode to
use while working, because it shows what changed.

There was a fourth mode, ``verify``, which compared hashes against a manifest
committed beside this module.  It is retired.  ``check-tree`` asks the strictly
better question -- whether ``frontend/ops/generated``, the tree CMake compiles
into libsfem, is what the generator produces today -- where ``verify`` asked
only whether the generator still agreed with a record of itself.  A record can
go stale without anything failing, and this one did: its manifest went unwritten
for sixty-three commits while ``check-tree`` passed on every one of them, so the
gate was dark and nothing said so.  A gate that can be silently wrong is worse
than no gate, because it is counted as cover.

The reference tree is roughly 44 MB and is deliberately not committed.  The
manifest is, so the gate itself is reviewable in the branch history.

The generators are run as subprocesses, the same way ``regenerate_all.sh`` runs
them, so the snapshot exercises the real entry points rather than an in-process
shortcut.

``check-tree`` asks a different question from the other three.  They compare the
generator against a record of itself; it compares the generator against
``frontend/ops/generated``, which is what CMake globs into libsfem and what
therefore actually ships.  Nothing checked that before, and the two drifted for
six weeks -- long enough for the shipped Laplacian to be a scalar operator whose
block scatter still indexed itself as a vector one.  This is the mode CI runs.
"""

import argparse
import difflib
import filecmp
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile


# Kept in step with generators/regenerate_all.sh.  Materials are captured in a
# fixed order so that a failure report reads the same way on every run.
MATERIALS = (
    "linear_elasticity",
    "laplace",
    "neohookean_ogden",
    "mooney_rivlin_kelvin_voigt_newmark",
    "neumann",
    "neumann_general",
    "two_phase_flow",
    "navier_stokes",
)

# Byte-for-byte comparison would otherwise trip over editor and interpreter
# droppings that no generator emits.
IGNORED_NAMES = ("__pycache__", ".DS_Store")

MAX_REPORTED_DIFFS = 5
MAX_DIFF_LINES = 60


#: Files the shipped tree carries that `regenerate_all.sh` does not write.
#:
#: The registration unit comes from `generators.op_registration`, which is
#: manifest-driven and is edited by hand to disable operators (Stokes is
#: commented out in it today), so it is not reproduced by a plain regeneration
#: and `check-tree` expects it rather than reporting it as drift.  Anything else
#: present in the tree and absent from a fresh generation is drift, which is the
#: whole point of the check.
#:
#: The four shared `.cuh` headers used to be here too, on the grounds that they
#: came from `generators.cuda` and that only runs under `SFEM_GENERATE_CUDA=1`.
#: What the exemption actually bought was four tracked files nothing produced,
#: nothing included and nothing checked, drifting several refactors behind their
#: `.hpp` counterparts.  They do not need a material -- they are what the target
#: spells -- so `generators.shared_headers` writes them on every regeneration
#: and they are checked like everything else.
UNGENERATED_TREE_PATHS = (
    "sfem_generated_ops_registration.cpp",
    "sfem_generated_ops_registration.hpp",
)


def shipped_tree():
    """The generated tree the build actually compiles.

    `CMakeLists.txt` globs `frontend/ops/generated/**.cpp` into libsfem and never
    regenerates it, so this directory -- not the generator -- is what ships.  The
    manifest checks the generator against itself; this is the path that checks
    the generator against the binary.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    repository = os.path.abspath(os.path.join(here, "..", "..", "..", ".."))
    return os.path.join(repository, "frontend", "ops", "generated")


def snapshot_root(base_dir):
    """The directory the generators actually write into, under ``base_dir``.

    This must be named ``generated``.  ``sfem.gen`` decides whether to hoist the
    shared primitive headers (``kernel_math.hpp`` and friends) into one copy at
    the tree root or to write a copy per material by inspecting the *name* of the
    output directory's parent -- see ``_uses_generated_shared_primitive_headers``
    in ``python/sfem/gen.py``.  Generating into a differently named directory
    silently produces a different tree, so the snapshot pins the production
    layout: ``<base>/generated/<material>``.
    """
    return os.path.join(base_dir, "generated")


def _python_root():
    """Absolute path of the repository's ``python/`` directory."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", ".."))


def _generator_env():
    env = dict(os.environ)
    python_root = _python_root()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = python_root + (os.pathsep + existing if existing else "")
    return env


#: Both targets, in the order `regenerate_all.sh` runs them.  The device pass
#: comes second because the two share a material's directory and both write the
#: target-independent matrix-format files.
GENERATION_TARGETS = ("openmp", "cuda")


def generate_all(out_dir, materials=MATERIALS, verbose=True):
    """Run every generator into ``out_dir``/<material>, for every target.

    Returns the list of materials that failed, empty when all succeeded.
    """
    env = _generator_env()
    failed = []
    for generation_target in GENERATION_TARGETS:
        for material in materials:
            target = os.path.join(out_dir, material)
            command = [
                sys.executable,
                "-m",
                "codegen.framework.generators.%s" % material,
                "--out-dir",
                target,
                "--target",
                generation_target,
            ]
            label = "%s (%s)" % (material, generation_target)
            if verbose:
                print("==> %s" % label, flush=True)
            completed = subprocess.run(
                command,
                env=env,
                cwd=_python_root(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            if completed.returncode != 0:
                failed.append(label)
                sys.stderr.write(
                    "generator '%s' failed with exit code %d:\n%s\n"
                    % (label, completed.returncode, completed.stdout.decode("utf-8", "replace"))
                )
    # The headers that belong to a target rather than to a material.  A material
    # run writes its own target's set beside its kernels, which is where the
    # `.hpp` ones come from; the `.cuh` ones have no material to ride along with
    # and are written here, exactly as `regenerate_all.sh` writes them.
    if verbose:
        print("==> shared headers", flush=True)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "codegen.framework.generators.shared_headers",
            "--out-dir",
            out_dir,
        ],
        env=env,
        cwd=_python_root(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        failed.append("shared_headers")
        sys.stderr.write(
            "generator 'shared_headers' failed with exit code %d:\n%s\n"
            % (completed.returncode, completed.stdout.decode("utf-8", "replace"))
        )
    return failed


def _relative_files(root):
    """Every file under ``root``, as sorted paths relative to it."""
    found = []
    for directory, subdirectories, names in os.walk(root):
        subdirectories[:] = sorted(d for d in subdirectories if d not in IGNORED_NAMES)
        for name in sorted(names):
            if name in IGNORED_NAMES:
                continue
            path = os.path.join(directory, name)
            found.append(os.path.relpath(path, root))
    return sorted(found)


def _file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def digests(root):
    """Map every file under ``root`` to its SHA-256, keyed by relative path."""
    return {relative: _file_digest(os.path.join(root, relative)) for relative in _relative_files(root)}


def _unified_diff(reference_path, candidate_path, relative):
    try:
        with open(reference_path, encoding="utf-8") as handle:
            reference_lines = handle.readlines()
        with open(candidate_path, encoding="utf-8") as handle:
            candidate_lines = handle.readlines()
    except UnicodeDecodeError:
        return ["  (binary file differs)"]
    diff = difflib.unified_diff(
        reference_lines,
        candidate_lines,
        fromfile="reference/%s" % relative,
        tofile="regenerated/%s" % relative,
        n=2,
    )
    lines = ["  " + line.rstrip("\n") for line in diff]
    if len(lines) > MAX_DIFF_LINES:
        omitted = len(lines) - MAX_DIFF_LINES
        lines = lines[:MAX_DIFF_LINES] + ["  ... %d further diff lines omitted" % omitted]
    return lines


def compare(reference_dir, candidate_dir):
    """Compare two generated trees.

    Returns ``(added, removed, changed)`` as sorted lists of relative paths.
    """
    reference_files = set(_relative_files(reference_dir))
    candidate_files = set(_relative_files(candidate_dir))
    added = sorted(candidate_files - reference_files)
    removed = sorted(reference_files - candidate_files)
    changed = []
    for relative in sorted(reference_files & candidate_files):
        reference_path = os.path.join(reference_dir, relative)
        candidate_path = os.path.join(candidate_dir, relative)
        if not filecmp.cmp(reference_path, candidate_path, shallow=False):
            changed.append(relative)
    return added, removed, changed


def report(reference_dir, candidate_dir, added, removed, changed):
    """Print a human-readable account of a failed comparison."""
    print("")
    print("generated output differs from the reference snapshot")
    print("  reference:   %s" % reference_dir)
    print("  regenerated: %s" % candidate_dir)
    print("")
    for label, paths in (("added", added), ("removed", removed), ("changed", changed)):
        if not paths:
            continue
        print("%s (%d):" % (label, len(paths)))
        for path in paths:
            print("  %s" % path)
        print("")
    for relative in changed[:MAX_REPORTED_DIFFS]:
        print("--- %s" % relative)
        for line in _unified_diff(
            os.path.join(reference_dir, relative),
            os.path.join(candidate_dir, relative),
            relative,
        ):
            print(line)
        print("")
    if len(changed) > MAX_REPORTED_DIFFS:
        print("(%d further changed files not shown)" % (len(changed) - MAX_REPORTED_DIFFS))


def command_capture(args):
    out_dir = os.path.abspath(args.directory)
    if os.path.exists(out_dir):
        if not args.force:
            sys.stderr.write(
                "refusing to overwrite existing snapshot '%s'; pass --force to replace it\n" % out_dir
            )
            return 2
        shutil.rmtree(out_dir)
    root = snapshot_root(out_dir)
    os.makedirs(root)
    failed = generate_all(root, verbose=not args.quiet)
    if failed:
        sys.stderr.write("capture incomplete; generators failed: %s\n" % ", ".join(failed))
        return 1
    files = _relative_files(root)
    print("")
    print("captured %d generated files from %d materials into %s" % (len(files), len(MATERIALS), root))
    return 0


def command_check(args):
    reference_dir = snapshot_root(os.path.abspath(args.directory))
    if not os.path.isdir(reference_dir):
        sys.stderr.write(
            "no reference snapshot at '%s'; run 'capture' first\n" % reference_dir
        )
        return 2
    candidate_parent = tempfile.mkdtemp(prefix="sfem_codegen_check_")
    candidate_dir = snapshot_root(candidate_parent)
    os.makedirs(candidate_dir)
    try:
        failed = generate_all(candidate_dir, verbose=not args.quiet)
        if failed:
            sys.stderr.write("check incomplete; generators failed: %s\n" % ", ".join(failed))
            return 1
        added, removed, changed = compare(reference_dir, candidate_dir)
        if added or removed or changed:
            report(reference_dir, candidate_dir, added, removed, changed)
            if args.keep:
                print("regenerated tree kept at %s" % candidate_dir)
                candidate_parent = None
            return 1
        print("")
        print(
            "generated output is byte-identical to the reference snapshot (%d files)"
            % len(_relative_files(reference_dir))
        )
        return 0
    finally:
        if candidate_parent is not None:
            shutil.rmtree(candidate_parent, ignore_errors=True)


def command_check_tree(args):
    """Compare a fresh generation against the tree the build compiles.

    `verify` and `check` both ask whether the generator still agrees with a
    record of itself.  Neither looks at `frontend/ops/generated`, which is what
    CMake actually builds, so the two drifted apart for six weeks without any
    check noticing -- and the drift hid a scalar Laplacian that read past the end
    of its own element matrix.

    `verify` has since been retired for a second reason of the same kind: its
    manifest went unwritten for sixty-three commits while this check passed on
    every one of them, so it was dark and nothing said so.  `check` remains,
    because a reference tree is captured deliberately and compared immediately;
    it cannot rot in the repository the way a committed record can.
    """
    tree_dir = os.path.abspath(args.tree or shipped_tree())
    if not os.path.isdir(tree_dir):
        sys.stderr.write("no generated tree at '%s'\n" % tree_dir)
        return 2
    candidate_parent = tempfile.mkdtemp(prefix="sfem_codegen_tree_")
    candidate_dir = snapshot_root(candidate_parent)
    os.makedirs(candidate_dir)
    try:
        failed = generate_all(candidate_dir, verbose=not args.quiet)
        if failed:
            sys.stderr.write("check-tree incomplete; generators failed: %s\n" % ", ".join(failed))
            return 1
        added, removed, changed = compare(tree_dir, candidate_dir)
        expected = set(UNGENERATED_TREE_PATHS)
        unexpected_removed = [path for path in removed if path not in expected]
        missing_expected = [path for path in expected if path not in set(removed)]
        if added or unexpected_removed or changed:
            report(tree_dir, candidate_dir, added, unexpected_removed, changed)
            print("the committed tree is what the build compiles; regenerate it with")
            print("  python/codegen/framework/generators/regenerate_all.sh")
            if args.keep:
                print("regenerated tree kept at %s" % candidate_dir)
                candidate_parent = None
            return 1
        print("")
        print(
            "the committed tree matches a fresh generation (%d files, %d not generated by regenerate_all.sh)"
            % (len(_relative_files(tree_dir)) - len(expected) + len(missing_expected), len(expected) - len(missing_expected))
        )
        for path in sorted(missing_expected):
            print("  note: '%s' is listed as ungenerated but is absent from the tree" % path)
        return 0
    finally:
        if candidate_parent is not None:
            shutil.rmtree(candidate_parent, ignore_errors=True)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Capture or compare a byte-exact snapshot of generated kernel sources.",
    )
    # Shared so that -q is accepted either before or after the subcommand.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-q", "--quiet", action="store_true", help="Do not print per-material progress.")
    parser.add_argument("-q", "--quiet", action="store_true", help=argparse.SUPPRESS)
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser(
        "capture", parents=[common], help="Regenerate every material into a reference tree."
    )
    capture.add_argument("directory", help="Directory to write the reference snapshot into.")
    capture.add_argument("--force", action="store_true", help="Replace an existing snapshot directory.")
    capture.set_defaults(handler=command_capture)

    check = subparsers.add_parser(
        "check", parents=[common], help="Regenerate and compare against a reference tree."
    )
    check.add_argument("directory", help="Reference snapshot to compare against.")
    check.add_argument(
        "--keep",
        action="store_true",
        help="Keep the regenerated tree when it differs, for manual inspection.",
    )
    check.set_defaults(handler=command_check)


    check_tree = subparsers.add_parser(
        "check-tree",
        parents=[common],
        help="Regenerate and compare against frontend/ops/generated, the tree the build compiles.",
    )
    check_tree.add_argument(
        "--tree",
        help="Generated tree to compare against (default: frontend/ops/generated).",
    )
    check_tree.add_argument(
        "--keep",
        action="store_true",
        help="Keep the regenerated tree when it differs, for manual inspection.",
    )
    check_tree.set_defaults(handler=command_check_tree)

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
