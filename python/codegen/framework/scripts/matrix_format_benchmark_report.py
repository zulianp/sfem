#!/usr/bin/env python3
import argparse
import csv
import json
import sys


FIELDNAMES = (
    "plan",
    "kernel",
    "variant",
    "format",
    "mesh_layout",
    "packed_pass",
    "assembly_kind",
    "index_policy",
    "value_layout",
    "accumulation_policy",
    "structural_compatibility",
    "reduction_policy",
    "node_index_filter",
    "row_dofs_per_element",
    "column_dofs_per_element",
    "entries_per_element",
    "nelements",
    "total_flops",
    "total_bytes",
    "arithmetic_intensity",
    "elapsed_seconds",
    "repeat",
    "seconds_per_call",
    "bandwidth_gb_s",
    "achieved_gflop_s",
)


def iter_matrix_format_rows(plan_name, plan, nelements, elapsed_seconds, repeat):
    for kernel in plan.get("kernels", ()):
        yield from _iter_kernel_rows(
            plan_name,
            kernel,
            nelements,
            elapsed_seconds,
            repeat,
        )


def _iter_kernel_rows(plan_name, kernel, nelements, elapsed_seconds, repeat):
    matrix_plan = kernel.get("matrix_format_plan")
    if matrix_plan:
        kernel_name = kernel.get("name", "")
        for variant in matrix_plan.get("variants", ()):
            yield _row(
                plan_name,
                kernel_name,
                variant,
                nelements,
                elapsed_seconds,
                repeat,
            )

    for block_kernel in kernel.get("block_kernels", ()):
        yield from _iter_kernel_rows(
            plan_name,
            block_kernel,
            nelements,
            elapsed_seconds,
            repeat,
        )


def _row(plan_name, kernel_name, variant, nelements, elapsed_seconds, repeat):
    assembly = _assembly_fields(variant.get("assembly_plan") or {})
    flops_per_element = float(variant.get("expected_flops_per_element", 0.0))
    bytes_per_element = int(variant.get("expected_bytes_per_element", 0))
    total_flops = flops_per_element * nelements
    total_bytes = bytes_per_element * nelements
    ai = flops_per_element / bytes_per_element if bytes_per_element else 0.0
    seconds_per_call = elapsed_seconds / repeat if repeat > 0 else 0.0
    bandwidth = total_bytes / seconds_per_call / 1.0e9 if seconds_per_call > 0.0 else 0.0
    gflops = total_flops / seconds_per_call / 1.0e9 if seconds_per_call > 0.0 else 0.0

    return {
        "plan": plan_name,
        "kernel": kernel_name,
        "variant": variant.get("name", ""),
        "format": variant.get("format", ""),
        "mesh_layout": variant.get("mesh_layout", ""),
        "packed_pass": variant.get("packed_pass", ""),
        "assembly_kind": assembly["assembly_kind"],
        "index_policy": assembly["index_policy"],
        "value_layout": assembly["value_layout"],
        "accumulation_policy": assembly["accumulation_policy"],
        "structural_compatibility": assembly["structural_compatibility"],
        "reduction_policy": assembly["reduction_policy"],
        "node_index_filter": int(bool(variant.get("node_index_filter", False))),
        "row_dofs_per_element": int(variant.get("row_dofs_per_element", 0)),
        "column_dofs_per_element": int(variant.get("column_dofs_per_element", 0)),
        "entries_per_element": int(variant.get("entries_per_element", 0)),
        "nelements": int(nelements),
        "total_flops": _number(total_flops),
        "total_bytes": int(total_bytes),
        "arithmetic_intensity": _number(ai),
        "elapsed_seconds": _number(elapsed_seconds),
        "repeat": int(repeat),
        "seconds_per_call": _number(seconds_per_call),
        "bandwidth_gb_s": _number(bandwidth),
        "achieved_gflop_s": _number(gflops),
    }


def _assembly_fields(plan):
    kind = plan.get("kind", "")
    fields = {
        "assembly_kind": kind,
        "index_policy": "",
        "value_layout": "",
        "accumulation_policy": plan.get("accumulation_policy", ""),
        "structural_compatibility": plan.get("structural_compatibility", ""),
        "reduction_policy": plan.get("reduction_policy", plan.get("duplicate_policy", "")),
    }
    if kind in ("crs", "bsr"):
        fields["index_policy"] = "_".join(
            part for part in (plan.get("row_pointer", ""), plan.get("column_index", "")) if part
        )
        fields["value_layout"] = plan.get("block_value_layout", "scalar_element_matrix")
    elif kind == "dia":
        fields["index_policy"] = plan.get("diagonal_offsets", "")
        fields["value_layout"] = plan.get("value_layout", "")
    elif kind == "coo":
        fields["index_policy"] = "_".join(
            part for part in (plan.get("row_index_stream", ""), plan.get("column_index_stream", "")) if part
        )
        fields["value_layout"] = "triplet_element_matrix"
    elif kind == "patch":
        fields["index_policy"] = plan.get("patch_graph", "")
        fields["value_layout"] = plan.get("patch_value_layout", "")
    return fields


def _number(value):
    return "%.17g" % float(value)


def write_csv(rows, output):
    writer = csv.DictWriter(output, fieldnames=FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Report matrix-format generation diagnostics and optional measured rates from plan dumps."
    )
    parser.add_argument("plans", nargs="+", help="Generation plan JSON files")
    parser.add_argument("--nelements", type=int, default=1, help="Element count used for total FLOP/byte estimates")
    parser.add_argument("--elapsed-seconds", type=float, default=0.0, help="Measured total elapsed time for repeat calls")
    parser.add_argument("--repeat", type=int, default=1, help="Number of calls included in elapsed time")
    args = parser.parse_args(argv)

    if args.nelements < 0:
        parser.error("--nelements must be non-negative")
    if args.elapsed_seconds < 0.0:
        parser.error("--elapsed-seconds must be non-negative")
    if args.repeat < 0:
        parser.error("--repeat must be non-negative")

    rows = []
    for path in args.plans:
        with open(path, encoding="utf-8") as input_file:
            plan = json.load(input_file)
        rows.extend(
            iter_matrix_format_rows(
                path,
                plan,
                args.nelements,
                args.elapsed_seconds,
                args.repeat,
            )
        )

    write_csv(rows, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
