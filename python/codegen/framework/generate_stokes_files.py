#!/usr/bin/env python3
import argparse
import os
import re
try:
    from ._script_common import (
        bootstrap_python_path,
        generated_output_dir,
        print_generation_result,
    )
except ImportError:
    from _script_common import (
        bootstrap_python_path,
        generated_output_dir,
        print_generation_result,
    )


bootstrap_python_path(__file__, 2)

from codegen.framework.materials.stokes import material  # noqa: E402
from sfem import gen  # noqa: E402


def _default_out_dir():
    return generated_output_dir(__file__, "stokes", 3)


def _operator_dim(path, contents):
    match = re.search(r"static constexpr int DIM = ([0-9]+);", contents)
    if match is None:
        raise RuntimeError(
            "Stokes operator '%s' does not declare generated DIM"
            % os.path.basename(path)
        )
    return int(match.group(1))


def _unit_output_fields(unit):
    return tuple(field.name for field in unit.form_collection.fields)


def _field_output_fields(unit, field_name):
    return tuple(field.name for field in unit.form_collection.fields if field.name == field_name)


def _operator_output_fields(plan, operator_dim, basename):
    for unit in plan.units:
        if int(unit.dim) != operator_dim:
            continue
        for block_kernel in unit.block_kernels:
            prefix = "%s_" % block_kernel.name
            if not basename.startswith(prefix):
                continue
            if (
                block_kernel.block.form_order is gen.FormOrder.TWO
                and block_kernel.block.row_field == block_kernel.block.column_field
            ):
                return _field_output_fields(unit, block_kernel.block.row_field)
            return _unit_output_fields(unit)
        if basename.startswith("%s_" % unit.name):
            return _unit_output_fields(unit)
    return ()


def validate_m6_4(result):
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
            if block_kernel.emission is not gen.KernelEmission.FILES:
                raise RuntimeError("Stokes block kernel '%s' does not emit files" % block_kernel.name)
            if not block_kernel.block:
                raise RuntimeError("Stokes block kernel '%s' does not select a block" % block_kernel.name)
            if block_kernel.block.name not in block_kernel.name:
                raise RuntimeError(
                    "Stokes block kernel '%s' does not follow the selected-block naming convention"
                    % block_kernel.name
                )

    for block_kernel in plan.block_kernels:
        if block_kernel.block.name.endswith("_p_p"):
            raise RuntimeError("Stokes generated a zero pressure-pressure block kernel")

    operator_sources = tuple(path for path in result.sources if path.endswith("_operator.cpp"))
    local_headers = tuple(path for path in result.sources if path.endswith("_local.hpp"))
    if not operator_sources:
        raise RuntimeError("Stokes generation did not produce mesh-level operator sources")
    if not local_headers:
        raise RuntimeError("Stokes generation did not produce family-level local kernels")

    for path in operator_sources:
        basename = os.path.basename(path)
        with open(path) as input_file:
            contents = input_file.read()
        operator_dim = _operator_dim(path, contents)
        field_names = _operator_output_fields(plan, operator_dim, basename)
        if not field_names:
            raise RuntimeError(
                "Stokes operator '%s' has no matching plan unit for dimension %d"
                % (basename, operator_dim)
            )
        local_header_includes = tuple(
            os.path.relpath(header, start=os.path.dirname(path)).replace(os.sep, "/")
            for header in local_headers
        )
        if not any('#include "%s"' % header in contents for header in local_header_includes):
            raise RuntimeError(
                "Stokes operator '%s' does not include a generated local kernel"
                % basename
            )
        for token in (
            "_residual_affine_mesh_soa",
            "_residual_isoparametric_mesh_soa",
            "_jacobian_action_affine_mesh_soa",
            "_jacobian_action_isoparametric_mesh_soa",
        ):
            if token not in contents:
                raise RuntimeError("Stokes operator '%s' is missing '%s'" % (basename, token))
        for field_name in field_names:
            if "%s_out" % field_name not in contents:
                raise RuntimeError(
                    "Stokes operator '%s' is missing output field '%s'"
                    % (basename, field_name)
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
        help="Do not write stokes_plan.json.",
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
    validate_m6_4(result)

    print_generation_result(result, "Generated Stokes kernels:")
    print("M6.4 validation: monolithic and nonzero block Stokes kernels")
    return result


if __name__ == "__main__":
    main()
