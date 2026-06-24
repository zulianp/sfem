#!/usr/bin/env python3
import argparse
import glob
import os
import shutil
import subprocess
import sys

import sympy as sp


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
sys.path.insert(0, ROOT_DIR)

from python.codegen.framework import (  # noqa: E402
    CoupledResidualSystem,
    TwoPhaseFlowConstitutiveModel,
    generate_coupled_residual_sfem_files,
)


DEFAULT_ELEMENTS = ("TRI3", "TET4", "QUAD4", "HEX8")
ELEMENT_DIMENSIONS = {
    "TRI3": 2,
    "TET4": 3,
    "QUAD4": 2,
    "HEX8": 3,
}


def build_two_phase_flow_system(dim):
    system = CoupledResidualSystem(dim)
    water = system.add_field("p_w")
    co2 = system.add_field("p_c")
    constitutive = TwoPhaseFlowConstitutiveModel.symbolic()
    dt = sp.Symbol("dt")
    permeability = sp.Matrix(dim, dim, sp.symbols("K_0:%d" % (dim * dim)))
    system.add_parameters(
        *constitutive.parameters.as_tuple(),
        dt,
        *tuple(permeability),
    )

    current = constitutive.state(water.value, co2.value)
    previous = constitutive.state(water.previous_value, co2.previous_value)
    porosity = constitutive.parameters.porosity
    water_accumulation = porosity * (
        current.water_saturation * current.water_density
        - previous.water_saturation * previous.water_density
    ) / dt
    co2_accumulation = porosity * (
        current.co2_saturation * current.co2_density
        - previous.co2_saturation * previous.co2_density
    ) / dt
    water_flux = -(
        current.water_density
        * current.water_mobility
        * permeability
        * sp.Matrix(water.gradient)
    )
    co2_flux = -(
        current.co2_density
        * current.co2_mobility
        * permeability
        * sp.Matrix(co2.gradient)
    )

    system.set_residual(
        water,
        water_accumulation * water.test_value
        - water_flux.dot(sp.Matrix(water.test_gradient)),
    )
    system.set_residual(
        co2,
        co2_accumulation * co2.test_value
        - co2_flux.dot(sp.Matrix(co2.test_gradient)),
    )
    return system


def parse_elements(values):
    if not values:
        return DEFAULT_ELEMENTS
    elements = []
    for value in values:
        for element in value.split(","):
            element = element.strip().upper()
            if element not in ELEMENT_DIMENSIONS:
                raise ValueError(
                    "element must be one of %s"
                    % ", ".join(ELEMENT_DIMENSIONS)
                )
            if element not in elements:
                elements.append(element)
    return tuple(elements)


def write_generated_files(out_dir, elements, vector_size):
    systems = {}
    written = {}
    for element in elements:
        dim = ELEMENT_DIMENSIONS[element]
        if dim not in systems:
            systems[dim] = build_two_phase_flow_system(dim)
        system = systems[dim]
        files = generate_coupled_residual_sfem_files(
            system,
            prefix="generated_two_phase_flow",
            element_type=element,
            vector_size=vector_size,
        )
        for generated in files:
            path = os.path.join(out_dir, generated.path)
            if generated.path == "kernel_math.hpp" and path in written:
                continue
            with open(path, "w", encoding="utf-8") as output:
                output.write(generated.source)
            written[path] = generated.source
    return tuple(written)


def clean_generated_files(out_dir):
    patterns = (
        "generated_two_phase_flow_*_local.hpp",
        "generated_two_phase_flow_*_operator.cpp",
        "generated_two_phase_flow_*_operator.o",
        "kernel_math.hpp",
        "kernel_diagnostics.hpp",
    )
    for pattern in patterns:
        for path in glob.glob(os.path.join(out_dir, pattern)):
            os.remove(path)


def compile_operators(paths, compiler):
    operators = tuple(path for path in paths if path.endswith("_operator.cpp"))
    for source in operators:
        output = "%s.o" % os.path.splitext(source)[0]
        subprocess.run(
            [
                compiler,
                "-std=c++14",
                "-O3",
                "-fopenmp-simd",
                "-Werror",
                "-c",
                source,
                "-I",
                os.path.dirname(source),
                "-o",
                output,
            ],
            check=True,
        )
    return tuple("%s.o" % os.path.splitext(path)[0] for path in operators)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate inspectable two-phase-flow residual and "
            "Jacobian-action element kernels."
        )
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(SCRIPT_DIR, "generated"),
    )
    parser.add_argument(
        "--element",
        action="append",
        help=(
            "Element type. May be repeated or comma-separated. "
            "Default: TRI3,TET4,QUAD4,HEX8."
        ),
    )
    parser.add_argument("--vector-size", type=int, default=16)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not remove files from previous generator runs.",
    )
    args = parser.parse_args()

    if args.vector_size <= 0:
        parser.error("--vector-size must be positive")
    try:
        elements = parse_elements(args.element)
    except ValueError as error:
        parser.error(str(error))

    os.makedirs(args.out_dir, exist_ok=True)
    if not args.keep_existing:
        clean_generated_files(args.out_dir)
    paths = write_generated_files(
        os.path.abspath(args.out_dir),
        elements,
        args.vector_size,
    )
    print("Generated:")
    for path in paths:
        print("  %s" % path)

    if args.compile:
        compiler = shutil.which("c++")
        if compiler is None:
            raise RuntimeError("c++ compiler is not available")
        objects = compile_operators(paths, compiler)
        print("Compiled:")
        for path in objects:
            print("  %s" % path)


if __name__ == "__main__":
    main()
