#!/usr/bin/env python3
import argparse
import importlib
import os

try:
    from ._script_common import bootstrap_python_path, generated_output_dir, print_generation_result
except ImportError:
    from _script_common import bootstrap_python_path, generated_output_dir, print_generation_result


bootstrap_python_path(__file__, 2)

from sfem import gen  # noqa: E402


_MATERIAL_MODULES = {
    "mooney_rivlin": "codegen.framework.materials.mooney_rivlin",
    "neohookean_ogden": "codegen.framework.materials.neohookean_ogden",
    "neumann": "codegen.framework.materials.neumann",
    "neumann_general": "codegen.framework.materials.neumann_general",
    "poro_hyperelasticity": "codegen.framework.materials.poro_hyperelasticity",
    "stokes": "codegen.framework.materials.stokes",
    "two_phase_flow": "codegen.framework.materials.two_phase_flow",
}


def _material(name):
    module = importlib.import_module(_MATERIAL_MODULES[name])
    return module.material


def _default_out_dir(name):
    return generated_output_dir(__file__, "%s_cuda" % name, 3)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate CUDA kernels for existing framework materials."
    )
    parser.add_argument(
        "--material",
        choices=tuple(sorted(_MATERIAL_MODULES)),
        default="neohookean_ogden",
    )
    parser.add_argument("--out-dir")
    parser.add_argument(
        "--element",
        "--element-type",
        action="append",
        dest="elements",
        help="Element type; may be repeated or comma-separated.",
    )
    parser.add_argument("--quadrature-order", type=int)
    parser.add_argument("--vector-size", type=int, default=gen.DEFAULT_VECTOR_SIZE)
    parser.add_argument("--dump-plan", action="store_true")
    parser.add_argument("--plan-out")
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Keep stale outputs from previous generator runs.",
    )
    args = parser.parse_args(argv)

    material = _material(args.material)
    out_dir = os.path.abspath(args.out_dir or _default_out_dir(args.material))
    result = gen.generate(
        material,
        out_dir,
        elements=args.elements,
        vector_size=args.vector_size,
        quadrature_order=args.quadrature_order,
        clean=not args.keep_existing,
        dump_plan=args.dump_plan,
        plan_out=args.plan_out,
        target="cuda",
    )
    print_generation_result(
        result,
        "Generated CUDA kernels for %s:" % args.material,
    )


if __name__ == "__main__":
    main()
