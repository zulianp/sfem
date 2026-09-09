"""Does the emitted FLOPs model describe the kernel it is attached to?

Every generated operator carries a ``KernelDiagnostics`` record, and every
consumer of a timing divides by

    total_flops = nelements * (n_qp * flops_per_qp_lane + mesh_flops_per_element)

so a wrong model does not merely mis-report: it silently rescales every
GFLOP/s figure measured from these kernels.  Nothing checked it against the
arithmetic the generator actually printed.

This is that check.  It reads a generated tree, and for every kernel whose
element body is straight-line -- no loop, so the trip counts cannot be got
wrong -- it counts the floating-point operations the body contains and compares
them with what the model claims.  Straight-line bodies are what the
``EXPANDED`` evaluation strategy produces on lowest-order simplices, which is
the largest single family in the tree, and they are the case where a count from
the text is exact rather than an estimate.

    python -m codegen.framework.tools.codegen_snapshot capture /tmp/gen
    python -m codegen.framework.tools.flops_audit /tmp/gen

Integer arithmetic inside a subscript is address computation, not a FLOP, and
is excluded -- as the model excludes it.  Unary minus is excluded for the same
reason ``_op_counts`` in ``plans/scheduling.py`` does not count it: sympy folds
it into a ``Mul`` by ``-1``, and the C compiler folds it into the instruction
that consumes it.
"""

import argparse
import glob
import json
import os
import re
import sys


#: The per-operation weights the model uses, so a count from the text and a
#: count from the IR are weighed on one scale.  ``plans/scheduling.py`` owns
#: them; they are restated here as a mapping over the names the C text uses.
CALL_WEIGHTS = {
    "sqrt": 12,
    "exp": 20,
    "log": 20,
    "pow": 1,
    "sin": 24,
    "cos": 24,
    "tan": 24,
    "asin": 24,
    "acos": 24,
    "atan": 24,
    "atan2": 24,
    "sinh": 24,
    "cosh": 24,
    "tanh": 24,
}

DIVIDE_WEIGHT = 8

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


def _strip_comments(text):
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def function_body(source, signature_prefix):
    """The brace-matched body of the first function whose head starts with this."""
    start = source.find(signature_prefix)
    if start < 0:
        return None
    open_brace = source.find("{", start)
    if open_brace < 0:
        return None
    depth = 0
    for index in range(open_brace, len(source)):
        character = source[index]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return source[open_brace + 1 : index]
    return None


def has_loop(body):
    """Whether the body contains any repetition at all."""
    return bool(re.search(r"\b(for|while)\s*\(", body))


def count_flops(body):
    """Weighted floating-point operations in a straight-line C body.

    Counts binary ``+ - * /`` and the calls in ``CALL_WEIGHTS``, skipping
    anything inside ``[...]``, which is integer address arithmetic.  A ``-`` or
    ``+`` in prefix position -- at the start of an expression or straight after
    another operator, a ``(`` or a ``,`` -- is a sign, not an addition.
    """
    body = _strip_comments(body)
    counts = {"add": 0, "mul": 0, "div": 0, "call": 0}
    weighted = 0
    bracket_depth = 0
    previous = ""
    index = 0
    length = len(body)
    while index < length:
        character = body[index]
        if character == "[":
            bracket_depth += 1
            previous = character
            index += 1
            continue
        if character == "]":
            bracket_depth = max(0, bracket_depth - 1)
            previous = character
            index += 1
            continue
        if character.isspace():
            index += 1
            continue
        if bracket_depth:
            previous = character
            index += 1
            continue
        match = _IDENTIFIER.match(body, index)
        if match:
            name = match.group(0)
            after = body[match.end() :].lstrip()
            if after.startswith("(") and name in CALL_WEIGHTS:
                counts["call"] += 1
                weighted += CALL_WEIGHTS[name]
            index = match.end()
            previous = name
            continue
        if character in "+-":
            # `+=` and `-=` accumulate: one addition, and the `=` is consumed
            # with it so the operator is not seen twice.
            if body[index : index + 2] in ("+=", "-="):
                counts["add"] += 1
                weighted += 1
                index += 2
                previous = "="
                continue
            unary = previous == "" or previous in "+-*/(,=<>?:&|;{}"
            if not unary:
                counts["add"] += 1
                weighted += 1
            previous = character
            index += 1
            continue
        if character == "*":
            # `*const`, `*p` and `s_t *` are pointers, not products.  A product
            # always has an operand -- an identifier, a literal or a `)` --
            # immediately to its left.
            if previous and (previous[-1].isalnum() or previous[-1] in "_)"):
                if previous not in ("const", "RSTR", "s_t", "g_t", "real_t", "float", "double"):
                    counts["mul"] += 1
                    weighted += 1
            previous = character
            index += 1
            continue
        if character == "/":
            counts["div"] += 1
            weighted += DIVIDE_WEIGHT
            previous = character
            index += 1
            continue
        previous = character
        index += 1
    counts["weighted"] = weighted
    return counts


#: The order of the fields in the emitted ``KernelDiagnostics`` initializer,
#: which ``emitters/energy_codegen.py`` prints positionally.
DIAGNOSTICS_FIELDS = (
    "kernel_name",
    "element_type",
    "dim",
    "n_qp",
    "n_shape",
    "vector_size",
    "quadrature_order",
    "add",
    "mul",
    "div",
    "sqrt",
    "pow",
    "exp",
    "log",
    "trig",
    "loads",
    "stores",
    "flops_per_qp_lane",
    "affine_mesh_flops_per_element",
    "isoparametric_mesh_flops_per_element",
    "temporaries",
    "estimated_registers",
    "geometry_streams",
    "reference_scalars",
    "quadrature_weight_scalars",
    "material_scalars",
    "u_streams",
    "h_streams",
    "output_streams",
    "output_reads_per_element",
    "output_writes_per_element",
)

_DIAGNOSTICS = re.compile(
    r"static const KernelDiagnostics (\w+)_diagnostics_data = \{(.*?)\n\};", re.S
)


def diagnostics_records(source):
    """Every diagnostics record in one source, as name -> field mapping."""
    records = {}
    for name, block in _DIAGNOSTICS.findall(source):
        values = [value.strip() for value in block.strip().rstrip(",").split(",")]
        record = {}
        for field, value in zip(DIAGNOSTICS_FIELDS, values):
            value = value.strip()
            if value.startswith('"'):
                record[field] = value.strip('"')
            else:
                try:
                    record[field] = int(value)
                except ValueError:
                    record[field] = value
        records[name] = record
    return records


def model_flops_per_element(record, geometry):
    """What the model says one element costs, for one geometry mode."""
    return record["n_qp"] * record["flops_per_qp_lane"] + record[
        "%s_mesh_flops_per_element" % geometry
    ]


#: A kernel's mesh-geometry suffix, and the model term that describes it.
GEOMETRY_BY_SUFFIX = (
    ("_a_msoa", "affine"),
    ("_i_msoa", "isoparametric"),
)


def audit(generated):
    """One row per straight-line kernel body found in the tree."""
    rows = []
    pattern = os.path.join(generated, "**", "*_operator.cpp")
    for path in sorted(glob.glob(pattern, recursive=True)):
        with open(path, encoding="utf-8") as stream:
            source = stream.read()
        records = diagnostics_records(source)
        for name, record in sorted(records.items()):
            for suffix, geometry in GEOMETRY_BY_SUFFIX:
                # The diagnostics record is named after the SoA kernel; the
                # geometry variants share it.
                variant = name[: -len("_soa")] + suffix if name.endswith("_soa") else None
                if variant is None:
                    continue
                body = function_body(
                    source, "static SFEM_INLINE int %s_impl(" % variant
                )
                if body is None:
                    continue
                loop = function_body(body, "for (ptrdiff_t element")
                if loop is None or has_loop(loop):
                    continue
                counted = count_flops(loop)
                rows.append(
                    {
                        "file": os.path.relpath(path, generated),
                        "kernel": variant,
                        "element_type": record["element_type"],
                        "geometry": geometry,
                        "model": model_flops_per_element(record, geometry),
                        "counted": counted["weighted"],
                        "adds": counted["add"],
                        "muls": counted["mul"],
                        "divs": counted["div"],
                    }
                )
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generated", help="a generated tree")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    rows = audit(args.generated)
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    disagreeing = [row for row in rows if row["model"] != row["counted"]]
    print("straight-line kernels audited: %d" % len(rows))
    print("model disagrees with the emitted arithmetic: %d" % len(disagreeing))
    print()
    header = "%-58s %-14s %8s %8s" % ("kernel", "geometry", "model", "counted")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            "%-58s %-14s %8d %8d"
            % (row["kernel"], row["geometry"], row["model"], row["counted"])
        )
    return 1 if disagreeing else 0


if __name__ == "__main__":
    sys.exit(main())
