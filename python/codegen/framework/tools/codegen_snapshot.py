"""Capture and verify a byte-exact snapshot of every generated kernel source.

This is the refactoring gate for the layering work described in ``LAYERING.md``.
Steps S0-S6 relocate decisions between layers without changing what is emitted,
so the acceptance criterion for each of them is that the generated C/C++ is
unchanged down to the byte.  That is a stronger statement than a passing test
suite, and it settles the performance question by construction: identical source
compiles to identical kernels.

Usage::

    python -m codegen.framework.tools.codegen_snapshot capture reference/
    python -m codegen.framework.tools.codegen_snapshot check reference/
    python -m codegen.framework.tools.codegen_snapshot verify

``capture`` regenerates every maintained material into a reference tree and
writes a hash manifest beside this module.  ``check`` regenerates into a
temporary tree and compares against that reference tree, exiting non-zero with a
unified diff of what moved -- this is the mode to use while working, because it
shows what changed.  ``verify`` compares only against the committed manifest, so
it needs no local reference tree and is the mode to use in CI or on a fresh
clone; it can say which files changed but not how.

The reference tree is roughly 44 MB and is deliberately not committed.  The
manifest is, so the gate itself is reviewable in the branch history.

The generators are run as subprocesses, the same way ``regenerate_all.sh`` runs
them, so the snapshot exercises the real entry points rather than an in-process
shortcut.
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
    "saint_venant_kirchhoff",
    "modified_mooney_rivlin",
    "mooney_rivlin_kelvin_voigt_newmark",
    "neumann",
    "neumann_general",
    "poro_elasticity",
    "stokes",
    "two_phase_flow",
)

# Byte-for-byte comparison would otherwise trip over editor and interpreter
# droppings that no generator emits.
IGNORED_NAMES = ("__pycache__", ".DS_Store")

MAX_REPORTED_DIFFS = 5
MAX_DIFF_LINES = 60

DEFAULT_MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "codegen_snapshot.sha256")


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


def generate_all(out_dir, materials=MATERIALS, verbose=True):
    """Run every generator into ``out_dir``/<material>.

    Returns the list of materials that failed, empty when all succeeded.
    """
    env = _generator_env()
    failed = []
    for material in materials:
        target = os.path.join(out_dir, material)
        command = [
            sys.executable,
            "-m",
            "codegen.framework.generators.%s" % material,
            "--out-dir",
            target,
        ]
        if verbose:
            print("==> %s" % material, flush=True)
        completed = subprocess.run(
            command,
            env=env,
            cwd=_python_root(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if completed.returncode != 0:
            failed.append(material)
            sys.stderr.write(
                "generator '%s' failed with exit code %d:\n%s\n"
                % (material, completed.returncode, completed.stdout.decode("utf-8", "replace"))
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


def write_manifest(root, manifest_path):
    """Write a ``sha256␠␠path`` manifest for ``root``, sorted by path."""
    entries = digests(root)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        handle.write("# SHA-256 of every generated kernel source, one line per file.\n")
        handle.write("# Regenerate with: python -m codegen.framework.tools.codegen_snapshot capture <dir>\n")
        for relative in sorted(entries):
            handle.write("%s  %s\n" % (entries[relative], relative))
    return len(entries)


def read_manifest(manifest_path):
    entries = {}
    with open(manifest_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            digest, _, relative = line.partition("  ")
            if not relative:
                raise ValueError("malformed manifest line: %r" % line)
            entries[relative] = digest
    return entries


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
    if not args.no_manifest:
        manifest_path = os.path.abspath(args.manifest)
        count = write_manifest(root, manifest_path)
        print("wrote manifest of %d hashes to %s" % (count, manifest_path))
    return 0


def command_verify(args):
    manifest_path = os.path.abspath(args.manifest)
    if not os.path.isfile(manifest_path):
        sys.stderr.write("no manifest at '%s'; run 'capture' first\n" % manifest_path)
        return 2
    expected = read_manifest(manifest_path)
    candidate_parent = tempfile.mkdtemp(prefix="sfem_codegen_verify_")
    candidate_dir = snapshot_root(candidate_parent)
    os.makedirs(candidate_dir)
    try:
        failed = generate_all(candidate_dir, verbose=not args.quiet)
        if failed:
            sys.stderr.write("verify incomplete; generators failed: %s\n" % ", ".join(failed))
            return 1
        actual = digests(candidate_dir)
        added = sorted(set(actual) - set(expected))
        removed = sorted(set(expected) - set(actual))
        changed = sorted(p for p in set(expected) & set(actual) if expected[p] != actual[p])
        if added or removed or changed:
            print("")
            print("generated output does not match the manifest at %s" % manifest_path)
            for label, paths in (("added", added), ("removed", removed), ("changed", changed)):
                if not paths:
                    continue
                print("")
                print("%s (%d):" % (label, len(paths)))
                for path in paths:
                    print("  %s" % path)
            print("")
            print("run 'check <reference-dir>' against a captured tree to see the diffs")
            return 1
        print("")
        print("generated output matches the manifest (%d files)" % len(expected))
        return 0
    finally:
        shutil.rmtree(candidate_parent, ignore_errors=True)


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


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Capture or verify a byte-exact snapshot of generated kernel sources.",
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
    capture.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help="Path of the hash manifest to write (default: beside this module).",
    )
    capture.add_argument(
        "--no-manifest",
        action="store_true",
        help="Capture the reference tree without rewriting the manifest.",
    )
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

    verify = subparsers.add_parser(
        "verify",
        parents=[common],
        help="Regenerate and compare hashes against the committed manifest (no reference tree needed).",
    )
    verify.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help="Manifest to verify against (default: the one beside this module).",
    )
    verify.set_defaults(handler=command_verify)

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
