"""A kernel declares a constant only when its body reads it.

Every kernel emitter prints a prologue of `static constexpr int` declarations
before it knows what the body will look like, because the prologue has to come
first in the emitted text.  Printing the same block everywhere left 648 of them
unread: `ND` in kernels handed an adjugate they never differentiate, `NQ1` and
`NS1` wherever the point count is spelled numerically instead of being derived
by `integer_root`, `N_FIELD_STREAMS` in every local body that indexes its
streams by group.

`kernel_constant` defers the decision instead of guessing it.  It yields the
declaration the emitter would have written, so every existing line operation
keeps working, but tagged so that `resolve_kernel_constants` can tell a deferred
declaration from ordinary text and drop the ones nothing reads.  The emitter
still decides what a kernel *may* declare; only the question of what it *does*
declare waits until the body exists.

This replaces `_prune_unused_spatial_dim`, which asked the same question of
already-emitted source with a substring test that `NDOFS = NC * NS` defeated in
every one of its 132 misses, and which only ever saw the files one emitter
produced.
"""

import re


class KernelConstant(str):
    """A `static constexpr int` declaration that survives only if it is read.

    A `str` subclass so that the line behaves exactly like the literal it
    replaces everywhere between here and `resolve_kernel_constants`.
    """

    def __new__(cls, name, value, indent="  "):
        line = "%sstatic constexpr int %s = %s;" % (indent, name, value)
        constant = super(KernelConstant, cls).__new__(cls, line)
        constant.constant_name = name
        constant.constant_value = str(value)
        return constant


def kernel_constant(name, value, indent="  "):
    return KernelConstant(name, value, indent)


class KernelDiscard(str):
    """A `(void)name;` an emitter writes because the kernel ignores a parameter.

    The parameter is part of the ABI and the caller must still pass it, so it
    cannot simply be deleted; what it must not do is carry a name the body never
    mentions, because `-Wextra -Werror` under `SFEM_ENABLE_DEV_MODE` rejects
    that and the discard is what silences it.  Naming nothing says the same to
    the compiler in the signature itself, and is what the boundary emitter has
    always done.
    """

    def __new__(cls, name, indent="  "):
        discard = super(KernelDiscard, cls).__new__(cls, "%s(void)%s;" % (indent, name))
        discard.discard_name = name
        return discard


def discard_unused(name, indent="  "):
    return KernelDiscard(name, indent)


_WORD_PATTERNS = {}


def _reads(text, name):
    pattern = _WORD_PATTERNS.get(name)
    if pattern is None:
        pattern = _WORD_PATTERNS[name] = re.compile(r"\b%s\b" % re.escape(name))
    return pattern.search(text) is not None


def _scopes(lines):
    """Pair each deferred declaration with the lines its scope still covers.

    A constant declared inside a function is read inside that function, so the
    search stops where its enclosing block closes rather than running to the end
    of the file -- the mistake that made the substring test in the pass this
    replaces look like it was working.
    """
    depths = []
    leaving = []
    depth = 0
    for line in lines:
        depths.append(depth)
        depth += line.count("{") - line.count("}")
        leaving.append(depth)

    scopes = {}
    for index, line in enumerate(lines):
        if not isinstance(line, KernelConstant):
            continue
        # The block closes on the line whose own `}` takes the depth back below
        # the declaration's, and that line is the last one still inside it.
        # Testing the depth *entering* a line instead would let the scope run
        # past the closing brace to the end of the enclosing namespace, where a
        # later function's use of the same name keeps a dead constant alive.
        end = len(lines)
        for following in range(index + 1, len(lines)):
            if leaving[following] < depths[index]:
                end = following
                break
        scopes[index] = (index + 1, end)
    return scopes


def resolve_kernel_constants(lines):
    """Drop the deferred declarations nothing in their own scope reads.

    Liveness is transitive: `NDOFS = NC * NS` keeps `NC` and `NS`, and
    `NQ1 = integer_root(NQ, 3)` keeps `NQ`, so a constant read only by another
    live declaration stays.
    """
    lines = list(lines)
    scopes = _scopes(lines)
    if not scopes:
        return lines

    live = set()
    for index, (start, end) in scopes.items():
        declaration = lines[index]
        body = "\n".join(
            line
            for position, line in enumerate(lines[start:end], start)
            if position not in scopes and not isinstance(line, KernelDiscard)
        )
        if _reads(body, declaration.constant_name):
            live.add(index)

    changed = True
    while changed:
        changed = False
        for index, (start, end) in scopes.items():
            if index in live:
                continue
            name = lines[index].constant_name
            for other in range(start, end):
                if other in live and _reads(lines[other].constant_value, name):
                    live.add(index)
                    changed = True
                    break

    return [
        str(line) if position in scopes else line
        for position, line in enumerate(lines)
        if position not in scopes or position in live
    ]


# The brace that opens a body sits either on a line of its own or at the end of
# the last parameter, and both shapes occur in these emitters.
_SIGNATURE_END = re.compile(r"^.*\)\s*(->.*)?\{\s*$")
# `for (...) {` closes a parenthesis and opens a brace too, and a discard can be
# emitted inside one -- the two-pass packed kernel discards a parameter from
# inside its pack loop.  A control-flow head is stepped over rather than
# mistaken for the signature, because the parameter belongs to the function
# around the loop and unnaming a loop variable would not compile.
_CONTROL_FLOW = re.compile(r"\b(for|if|while|switch|catch)\s*\(")


def _unname(line, name):
    """Drop a parameter's name, keeping its type: `ptrdiff_t nnodes,` -> `ptrdiff_t,`."""
    return re.sub(r"[ \t]+\b%s\b" % re.escape(name), "", line, count=1)


def resolve_dead_parameters(lines):
    """Turn `(void)name;` into a parameter that never had a name to begin with.

    The emitter asserts, by writing the discard, that its body ignores the
    parameter.  That assertion is checked here rather than trusted: a parameter
    the body does read keeps its name and merely loses a discard it never
    needed, so a stale marker cannot produce source that fails to compile.

    A discard whose signature cannot be located is left exactly as it was.
    Dropping it regardless would remove the statement that silences
    `-Wextra -Werror` while leaving the name it was silencing, which turns a
    tidy-up into a build failure under `SFEM_ENABLE_DEV_MODE`.
    """
    lines = list(lines)
    discards = [
        index for index, line in enumerate(lines) if isinstance(line, KernelDiscard)
    ]
    if not discards:
        return lines

    pending = set(discards)
    dropped = set()
    for index in discards:
        name = lines[index].discard_name

        opening = None
        for position in range(index - 1, -1, -1):
            if _SIGNATURE_END.match(lines[position]):
                if _CONTROL_FLOW.search(lines[position]):
                    continue
                opening = position
                break
        if opening is None:
            continue

        depth = 0
        closing = len(lines)
        for position in range(opening, len(lines)):
            depth += lines[position].count("{") - lines[position].count("}")
            if depth == 0 and position > opening:
                closing = position
                break

        body = "\n".join(
            line
            for position, line in enumerate(lines[opening + 1:closing], opening + 1)
            if position not in pending
        )
        if _reads(body, name):
            dropped.add(index)
            continue

        signature = None
        for position in range(opening, -1, -1):
            if lines[position].rstrip().endswith("("):
                signature = position
                break
        if signature is None:
            continue

        for position in range(opening, signature, -1):
            if _reads(lines[position], name):
                lines[position] = _unname(lines[position], name)
                dropped.add(index)
                break

    return [line for position, line in enumerate(lines) if position not in dropped]



_RENDERED = re.compile(
    r"^(?P<indent>[ \t]*)static constexpr int (?P<name>[A-Za-z_]\w*) = (?P<value>.+);$"
)


def retag_constants(lines):
    """Re-tag constants that reached text through the IR printer.

    The IR path spells a constant as a `BufferDeclNode` and renders it with
    everything else, so it arrives here as a finished line rather than as a
    `KernelConstant`.  Recognising it at the point the emitter turns its own IR
    into text puts both paths under one liveness rule, instead of leaving the
    kernels that happen to be built as IR with the dead declarations the rest no
    longer have.
    """
    retagged = []
    for line in lines:
        match = None if isinstance(line, KernelConstant) else _RENDERED.match(line)
        retagged.append(
            KernelConstant(match.group("name"), match.group("value"), match.group("indent"))
            if match
            else line
        )
    return retagged
