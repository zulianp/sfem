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


_CONSTANT = re.compile(r"^\s*static constexpr int (\w+)\s*=")
_SIGNATURE = re.compile(
    r"\n(?:template <[^>]*>\n)?(?:static )?(?:SFEM_INLINE )?"
    r"[\w:<>,\*& ]+\s+\w+\((?P<params>[^;]*?)\)\s*\{"
)
_PARAMETER = re.compile(r"(\w+)\s*(?:\[[^\]]*\])?\s*$")
#: The trailing identifier of an *unnamed* parameter is its type, and a local
#: declaration picked up by the signature pattern carries an initializer, so
#: neither is a named parameter that nothing reads.
_TYPE_NAMES = frozenset(
    """void bool char short int long float double unsigned signed size_t
    ptrdiff_t idx_t geom_t real_t count_t element_idx_t int16_t uint16_t
    s_t g_t compressed_t metric_tensor_t""".split()
)


def _block_depths(lines):
    """The brace depth entering and leaving each line."""
    entering, leaving, depth = [], [], 0
    for line in lines:
        entering.append(depth)
        depth += line.count("{") - line.count("}")
        leaving.append(depth)
    return entering, leaving


def unused_constants(source):
    """`static constexpr int` declarations nothing in their own block reads.

    The block ends at the line whose own brace takes the depth back below the
    declaration's, so a later function's use of the same name does not keep a
    dead one alive.  A name read through a qualified reference -- the boundary
    reference-data structs publish `NS` and `NQ` and are read as
    `<struct><s_t>::NS` from outside -- is a class member, not a dead local.
    """
    lines = source.split("\n")
    entering, leaving = _block_depths(lines)
    for index, line in enumerate(lines):
        match = _CONSTANT.match(line)
        if match is None:
            continue
        name = match.group(1)
        if re.search(r"::%s\b" % re.escape(name), source):
            continue
        end = len(lines)
        for following in range(index + 1, len(lines)):
            if leaving[following] < entering[index]:
                end = following
                break
        if not re.search(r"\b%s\b" % re.escape(name), "\n".join(lines[index + 1 : end])):
            yield name, index + 1


def unused_parameters(source):
    """Named parameters of a definition that its body never reads.

    This is what `-Wextra -Werror` rejects under `SFEM_ENABLE_DEV_MODE`, and
    what the `(void)name;` discards used to hide.  Naming nothing instead says
    the same thing to the compiler without a statement to carry it.
    """
    for match in _SIGNATURE.finditer(source):
        open_brace = source.index("{", match.start())
        depth, index = 0, open_brace
        while index < len(source):
            if source[index] == "{":
                depth += 1
            elif source[index] == "}":
                depth -= 1
                if depth == 0:
                    break
            index += 1
        body = source[open_brace : index + 1]
        for parameter in match.group("params").split(","):
            parameter = parameter.strip()
            if "=" in parameter or len(parameter.split()) < 2:
                continue
            declared = _PARAMETER.search(parameter)
            if declared is None:
                continue
            name = declared.group(1)
            if name in _KEYWORDS or name in _TYPE_NAMES:
                continue
            if not re.search(r"\b%s\b" % re.escape(name), body):
                yield name


_DISCARD = re.compile(r"^\s*\(void\)(\w+);\s*$", re.M)


def void_discards(source):
    """`(void)name;` statements: a kernel apologising for what it declares.

    The statement exists only to stop `-Wextra -Werror` complaining about a
    name nothing reads, which means the declaration was the mistake.  A
    parameter that is part of the ABI and genuinely unread should carry no
    name; a constant nothing reads should not be declared.
    """
    return [match.group(1) for match in _DISCARD.finditer(source)]


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
    constants = []
    parameters = []
    discards = []
    for path in source_files(generated):
        with open(path, encoding="utf-8") as stream:
            source = stream.read()
        relative = os.path.relpath(path, generated)
        for name, body in function_bodies(source):
            for symbol, statement in dead_assignments(body):
                dead.append((relative, name, symbol, statement))
        constants.extend(
            (relative, name, line) for name, line in unused_constants(source)
        )
        parameters.extend((relative, name) for name in unused_parameters(source))
        discards.extend((relative, name) for name in void_discards(source))
        for length in lane_loop_runs(source):
            runs[length] += 1
        wrappers.extend((relative, name) for name in wrapped_helper_entry_points(source))
    return {
        "dead": dead,
        "lane_loop_runs": runs,
        "wrapped_helpers": wrappers,
        "unused_constants": constants,
        "unused_parameters": parameters,
        "void_discards": discards,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generated", help="a generated tree")
    parser.add_argument("--limit", type=int, default=15)
    args = parser.parse_args(argv)

    result = survey(args.generated)
    print("(void) discards:              %d" % len(result["void_discards"]))
    for row in result["void_discards"][: args.limit]:
        print("    %s: %s" % row)
    print("unused constants:            %d" % len(result["unused_constants"]))
    for row in result["unused_constants"][: args.limit]:
        print("    %s:%d: %s" % (row[0], row[2], row[1]))
    print("unused parameters:           %d" % len(result["unused_parameters"]))
    for row in result["unused_parameters"][: args.limit]:
        print("    %s: %s" % row)
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
