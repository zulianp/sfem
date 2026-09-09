"""What a generated kernel contains that is not the computation.

A generated kernel is a hot loop over elements, and the standing rule for this
tree is that it stays lean: no validation in the element loop, no temporaries
nothing reads, no aliases that merely rename a value the kernel already has.
This measures the three departures from that rule that were found by surveying
the shipped tree, so each can be gated rather than re-found.

    python -m codegen.framework.tools.codegen_snapshot capture /tmp/gen
    python -m codegen.framework.tools.lean_audit /tmp/gen

**Dead assignments.**  A transitive dead-store scan over every emitted function
body.  The case that motivated it: a block form of a coupled system is handed
every field the system has, and the emitter staged each field's value *and*
gradient because the system as a whole used both.  The (p_w, p_w) block of
`two_phase_flow` reads p_c's value and never its gradient, so three gradient
accumulators were zero-filled, run over every trial function, mapped to the
physical element with a divide each, and dropped -- 28 of the 64 accumulate
statements in that kernel's trial loop.

**Single-statement lane loops.**  The lane loop is the vectorised inner loop.
Opening a fresh one per component emits a `#pragma omp simd` region per
component for stores that belong in one loop; a three-dimensional Jacobian
produced nine in a row.

**Per-kernel wrappers around shared helpers.**  `KernelDiagnostics` carries
free functions that take the record; wrapping each of them once per kernel per
precision published a name for a call the caller can already spell.
"""

import argparse
import collections
import glob
import os
import re
import sys


_IDENTIFIER = re.compile(r"\b[A-Za-z_]\w*\b")
_KEYWORDS = frozenset(
    "const static int double float void for while if else return break case"
    " default struct template typename inline sizeof true false".split()
)

_DECLARATION = re.compile(
    r"(?:const )?(?:s_t|double|float|int|idx_t|ptrdiff_t|count_t|real_t|geom_t)"
    r"\s+(\w+)(?:\[[^\]]*\])?\s*(?:=\s*(.*))?;$"
)
_WRITE = re.compile(r"(\w+)(\[[^;]*\])?\s*(?:\+=|=)\s*(.*);$")

_LANE_LOOP = re.compile(
    r"( *)#pragma omp simd\n *for \(int lane = 0; lane < ne; \+\+lane\) \{\n"
    r" *([^\n]*)\n *\}\n"
)


def source_files(generated):
    return sorted(
        glob.glob(os.path.join(generated, "**", "*.cpp"), recursive=True)
        + glob.glob(os.path.join(generated, "**", "*.hpp"), recursive=True)
    )


def function_bodies(source):
    """Every brace-matched function body in one source, with its name."""
    pattern = re.compile(
        r"\n(?:template <[^>]*>\n)?(?:static )?(?:SFEM_INLINE )?"
        r"[\w:<>,\*& ]+\s+(\w+)\([^;]*?\)\s*\{"
    )
    for match in pattern.finditer(source):
        open_brace = source.index("{", match.start())
        depth = 0
        index = open_brace
        while index < len(source):
            if source[index] == "{":
                depth += 1
            elif source[index] == "}":
                depth -= 1
                if depth == 0:
                    break
            index += 1
        yield match.group(1), source[open_brace : index + 1]


def dead_assignments(body):
    """Assignments in this body whose result nothing reads, transitively.

    A name becomes live when it appears anywhere that is not a definition of
    it: an index, a call argument, a scatter, a return.  Liveness then
    propagates backwards through the definitions until it stops moving, and
    what is left unlive is dead.
    """
    lines = body.split("\n")
    definitions = {}
    order = []
    live = set()
    for index, line in enumerate(lines):
        statement = line.strip()
        declaration = _DECLARATION.match(statement)
        if declaration and declaration.group(2) is not None:
            definitions[declaration.group(1)] = index
            order.append(
                (
                    declaration.group(1),
                    index,
                    set(_IDENTIFIER.findall(declaration.group(2))) - _KEYWORDS,
                )
            )
            continue
        write = _WRITE.match(statement)
        if write and write.group(1) in definitions:
            order.append(
                (
                    write.group(1),
                    index,
                    set(_IDENTIFIER.findall(write.group(3))) - _KEYWORDS,
                )
            )
            continue
        live |= set(_IDENTIFIER.findall(statement)) - _KEYWORDS
    changed = True
    while changed:
        changed = False
        for name, _index, uses in order:
            if name in live and not uses <= live:
                live |= uses
                changed = True
    return [
        (name, lines[index].strip())
        for name, index, _uses in order
        if name not in live
    ]


def lane_loop_runs(source):
    """Lengths of back-to-back single-statement lane loops in one source."""
    matches = list(_LANE_LOOP.finditer(source))
    runs = []
    index = 0
    while index < len(matches):
        end = index
        while (
            end + 1 < len(matches)
            and matches[end + 1].start() - matches[end].end() <= 1
        ):
            end += 1
        runs.append(end - index + 1)
        index = end + 1
    return runs


#: Wrappers that took a `KernelDiagnostics` record and called a free function
#: on it, once per kernel per precision.  Nothing referenced them.
WRAPPED_HELPER_SUFFIXES = ("_print_rate", "_arithmetic_intensity")


def wrapped_helper_entry_points(source):
    """`extern "C"` entry points that only wrap a shared diagnostics helper."""
    names = []
    for match in re.finditer(r'^extern "C" [\w:\* ]+?\**(\w+)\(', source, re.M):
        name = match.group(1)
        if any(
            name.endswith(suffix) or name.endswith(suffix + "_float")
            for suffix in WRAPPED_HELPER_SUFFIXES
        ):
            names.append(name)
    return names


def survey(generated):
    """Every measurement, over one generated tree."""
    dead = []
    runs = collections.Counter()
    wrappers = []
    for path in source_files(generated):
        with open(path, encoding="utf-8") as stream:
            source = stream.read()
        relative = os.path.relpath(path, generated)
        for name, body in function_bodies(source):
            for symbol, statement in dead_assignments(body):
                dead.append((relative, name, symbol, statement))
        for length in lane_loop_runs(source):
            runs[length] += 1
        wrappers.extend((relative, name) for name in wrapped_helper_entry_points(source))
    return {"dead": dead, "lane_loop_runs": runs, "wrapped_helpers": wrappers}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generated", help="a generated tree")
    parser.add_argument("--limit", type=int, default=15)
    args = parser.parse_args(argv)

    result = survey(args.generated)
    print("dead assignments:            %d" % len(result["dead"]))
    for row in result["dead"][: args.limit]:
        print("    %s: %s: %s" % (row[0], row[1], row[3][:90]))
    runs = result["lane_loop_runs"]
    print(
        "single-statement lane loops: %d in %d runs"
        % (sum(length * count for length, count in runs.items()), sum(runs.values()))
    )
    for length in sorted(runs, reverse=True)[:6]:
        print("    %2d in a row: %d" % (length, runs[length]))
    print("wrapped diagnostics helpers: %d" % len(result["wrapped_helpers"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
