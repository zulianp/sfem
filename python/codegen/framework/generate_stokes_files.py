#!/usr/bin/env python3
import argparse
import os
import re
import sys


PYTHON_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PYTHON_DIR)

from codegen.framework.materials.stokes import material  # noqa: E402
from sfem import gen  # noqa: E402


def _default_out_dir():
    return os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../../../frontend/ops/generated/stokes",
        )
    )


def _operator_dim(path, contents):
    match = re.search(r"static constexpr int DIM = ([0-9]+);", contents)
    if match is None:
        raise RuntimeError(
            "Stokes operator '%s' does not declare generated DIM"
            % os.path.basename(path)
        )
    return int(match.group(1))


def _unit_output_fields(unit):
    return tuple(
        ("%s%d" % (field.name, component)) if int(field.components) > 1 else field.name
        for field in unit.form_collection.fields
        for component in range(int(field.components))
    )


def validate_m6_3(result):
    plan = result.plan
    if plan is None:
        raise RuntimeError("Stokes generation did not return a generation plan")
    if tuple(plan.complete_system_kernels) != tuple(plan.units):
        raise RuntimeError("Stokes kernels are not all complete-system kernels")

    for unit in plan.units:
        if not unit.is_monolithic:
            raise RuntimeError("Stokes unit '%s' is not monolithic" % unit.name)
        if not unit.is_complete_system:
            raise RuntimeError("Stokes unit '%s' is not a complete coupled system" % unit.name)
        if not unit.blocks:
            raise RuntimeError("Stokes unit '%s' has no coupled form blocks" % unit.name)
        for block_kernel in unit.block_kernels:
            if not block_kernel.is_block:
                raise RuntimeError("Stokes block kernel '%s' does not use BLOCK scope" % block_kernel.name)
            if block_kernel.emission is not gen.KernelEmission.COVERED_BY_PARENT:
                raise RuntimeError("Stokes block kernel '%s' emits files separately" % block_kernel.name)

    operator_sources = tuple(path for path in result.sources if path.endswith("_operator.cpp"))
    local_headers = tuple(path for path in result.sources if path.endswith("_local.hpp"))
    if not operator_sources:
        raise RuntimeError("Stokes generation did not produce mesh-level operator sources")
    if not local_headers:
        raise RuntimeError("Stokes generation did not produce family-level local kernels")

    local_header_names = tuple(os.path.basename(path) for path in local_headers)
    for path in operator_sources:
        with open(path) as input_file:
            contents = input_file.read()
        operator_dim = _operator_dim(path, contents)
        field_names = tuple(
            field_name
            for unit in plan.units
            if int(unit.dim) == operator_dim
            for field_name in _unit_output_fields(unit)
        )
        if not field_names:
            raise RuntimeError(
                "Stokes operator '%s' has no matching plan unit for dimension %d"
                % (os.path.basename(path), operator_dim)
            )
        if not any('#include "%s"' % header in contents for header in local_header_names):
            raise RuntimeError(
                "Stokes operator '%s' does not include a generated local kernel"
                % os.path.basename(path)
            )
        for token in (
            "_residual_isoparametric_mesh_soa",
            "_jacobian_action_isoparametric_mesh_soa",
        ):
            if token not in contents:
                raise RuntimeError("Stokes operator '%s' is missing '%s'" % (os.path.basename(path), token))
        for field_name in field_names:
            if "%s_out" % field_name not in contents:
                raise RuntimeError(
                    "Stokes operator '%s' is missing output field '%s'"
                    % (os.path.basename(path), field_name)
                )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate and validate Taylor-Hood Stokes kernels."
    )
    parser.add_argument("--out-dir", default=_default_out_dir())
    parser.add_argument(
        "--element",
        "--element-type",
        action="append",
        dest="elements",
        help="Taylor-Hood element pair; may be repeated or comma-separated.",
    )
    parser.add_argument("--quadrature-order", type=int)
    parser.add_argument("--vector-size", type=int, default=gen.DEFAULT_VECTOR_SIZE)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Keep stale outputs from previous generator runs.",
    )
    parser.add_argument(
        "--no-plan-dump",
        action="store_true",
        help="Do not write generated_stokes_plan.json.",
    )
    args = parser.parse_args(argv)

    result = gen.generate(
        material,
        args.out_dir,
        elements=args.elements,
        vector_size=args.vector_size,
        quadrature_order=args.quadrature_order,
        compile=args.compile,
        clean=not args.keep_existing,
        dump_plan=not args.no_plan_dump,
    )
    validate_m6_3(result)

    print("Generated Stokes kernels:")
    for path in result.sources:
        print("  %s" % path)
    if result.objects:
        print("Compiled:")
        for path in result.objects:
            print("  %s" % path)
    if result.plan_dump:
        print("Plan:")
        print("  %s" % result.plan_dump)
    print("M6.3 validation: monolithic complete-system Stokes kernels")
    return result


if __name__ == "__main__":
    main()
