#!/usr/bin/env python3
"""Every device launch in the library names its stream.

The library's device work is ordered on the stream of the object it belongs to (an
ExecutionContext's, base/sfem_execution_context.hpp). This scan fails on what would order work
some other way:
  - a kernel launch <<<grid, block>>> or <<<grid, block, shmem>>>, without a stream;
  - a stream read through a pointer (cudaStream_t *): the stream is the cudaStream_t value;
  - the retired spellings: launch_stream(), LaunchStreamScope, current_launch_stream(),
    SFEM_DEFAULT_STREAM, and SFEM_DEBUG_SYNCHRONIZE() without its stream.

Usage: check_streams.py <source root>. Exit status 1 lists each finding as path:line.
"""
import os
import re
import sys

# Trees the scan leaves out, each with its reason.
EXCLUDED = {
    "external": "submodules, gated in their own repositories",
    "spikes": "separate projects, built and gated on their own",
    "resampling/cuda": "host-in, host-out entry points (allocate, copy, launch, copy back): synchronous by "
    "construction; to be replaced by sresample (MIGRATION_SRESAMPLE.md)",
    "benchmark": "not in the build",
}
SUFFIXES = (".cu", ".cuh", ".cpp", ".hpp", ".h", ".c")
RETIRED = re.compile(r"\b(launch_stream\s*\(|LaunchStreamScope|current_launch_stream|SFEM_DEFAULT_STREAM)\b|SFEM_DEBUG_SYNCHRONIZE\(\s*\)")
POINTER = re.compile(r"cudaStream_t\s*\*")
LAUNCH = re.compile(r"<<<(.*?)>>>", re.S)


def strip_comments(text):
    """Comments blanked to spaces, lengths kept, so offsets still give line numbers."""
    def blank(m):
        return re.sub(r"[^\n]", " ", m.group(0))

    text = re.sub(r"//[^\n]*", blank, text)
    return re.sub(r"/\*.*?\*/", blank, text, flags=re.S)


def top_level_commas(config):
    depth, commas = 0, 0
    for ch in config:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            commas += 1
    return commas


def scan(path, text):
    code = strip_comments(text)
    line = lambda offset: code.count("\n", 0, offset) + 1
    for m in LAUNCH.finditer(code):
        if top_level_commas(m.group(1)) < 3:
            yield "%s:%d: launch without a stream" % (path, line(m.start()))
    for m in POINTER.finditer(code):
        yield "%s:%d: stream through a pointer (cudaStream_t *)" % (path, line(m.start()))
    for m in RETIRED.finditer(code):
        yield "%s:%d: retired: %s" % (path, line(m.start()), m.group(0).strip())


def main(root):
    if os.path.abspath(__file__).startswith(os.path.abspath(root)):
        own = os.path.relpath(os.path.abspath(__file__), root)
    else:
        own = None
    findings, scanned = [], 0
    for directory, subdirectories, files in os.walk(root):
        rel = os.path.relpath(directory, root)
        rel = "" if rel == "." else rel
        subdirectories[:] = [
            d for d in subdirectories
            if not d.startswith(".") and not d.startswith("build") and os.path.join(rel, d) not in EXCLUDED
        ]
        for name in files:
            path = os.path.join(rel, name)
            if not name.endswith(SUFFIXES) or path == own:
                continue
            with open(os.path.join(root, path), errors="replace") as f:
                text = f.read()
            if "<<<" not in text and "cudaStream_t" not in text and "STREAM" not in text and "stream" not in text:
                continue
            scanned += 1
            findings.extend(scan(path, text))
    for finding in findings:
        print(finding)
    print("%d files with device code scanned, %d findings; left out: %s" % (scanned, len(findings), ", ".join(sorted(EXCLUDED))))
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
