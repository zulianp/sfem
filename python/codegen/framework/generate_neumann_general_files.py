#!/usr/bin/env python3
import argparse

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

from codegen.framework.materials.neumann_general import (  # noqa: E402
    DEFAULT_POLYNOMIAL_ORDER,
    create_material,
)
from sfem import gen  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate general polynomial Neumann boundary kernels."
    )
    parser.add_argument("--out-dir", default=generated_output_dir(__file__, "neumann_general", 3))
    parser.add_argument(
        "--element",
        "--element-type",
        action="append",
        dest="elements",
        help="Element type; may be repeated or comma-separated.",
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
        "--polynomial-order",
        type=int,
        default=DEFAULT_POLYNOMIAL_ORDER,
        help="Total polynomial order for the coordinate-dependent traction.",
    )
    args = parser.parse_args(argv)

    result = gen.generate(
        create_material(args.polynomial_order),
        args.out_dir,
        elements=args.elements,
        vector_size=args.vector_size,
        quadrature_order=args.quadrature_order,
        compile=args.compile,
        clean=not args.keep_existing,
    )
    print_generation_result(result)
    return result


if __name__ == "__main__":
    main()
