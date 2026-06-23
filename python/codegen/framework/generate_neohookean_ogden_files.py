#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys

import sympy as sp

sys.path.insert(0, os.path.dirname(__file__))

from symbolic import (
    KernelExpressions,
    generate_sfem_soa_cpp_files_for_element,
    matrix_inner,
    sfem_soa_element_specialization,
    sfem_soa_kernel_form,
    sfem_soa_weak_form,
)


def neohookean_ogden_energy(F, mu, lmbda):
    dim = F.shape[0]
    J = F.det()
    I1 = matrix_inner(F, F)
    logJ = sp.log(J)
    return mu * sp.Rational(1, 2) * (I1 - dim) - mu * logJ + (
        lmbda * sp.Rational(1, 2) * logJ * logJ
    )


def build_neohookean_forms(specialization):
    dim = specialization.dim
    mu, lmbda = sp.symbols("mu lmbda")
    F_symbols = sp.Matrix(
        dim,
        dim,
        tuple(sp.symbols("F[%d]" % i) for i in range(dim * dim)),
    )
    weak_form = sfem_soa_weak_form(
        neohookean_ogden_energy(F_symbols, mu, lmbda),
        F_symbols,
    )
    return {
        "weak_form": weak_form,
        "all_expressions": weak_form.diagnostic_expressions(has_direction=True),
    }


def build_combined_graph(forms):
    return (
        KernelExpressions()
        .add("operator_evaluation", forms["all_expressions"])
        .build_graph(
            data_symbols=forms["weak_form"].deformation_gradient,
            temporary_prefix="nh_inspect_tmp",
        )
    )


def build_sfem_soa_files(forms, specialization):
    weak_form = forms["weak_form"]

    return generate_sfem_soa_cpp_files_for_element(
        (
            sfem_soa_kernel_form(
                "objective",
                weak_form=weak_form,
                output_mode="accumulate",
            ),
            sfem_soa_kernel_form(
                "gradient",
                weak_form=weak_form,
                output_mode="accumulate",
            ),
            sfem_soa_kernel_form(
                "apply",
                weak_form=weak_form,
                has_direction=True,
                output_mode="accumulate",
            ),
        ),
        prefix="generated_neohookean_ogden",
        specialization=specialization,
    )


def write_text(path, content):
    with open(path, "w", encoding="utf-8") as output:
        output.write(content)


def summary_markdown(graph, specialization):
    quadrature_rule = specialization.quadrature_rule
    lines = [
        "# Neo-Hookean Ogden Generated Kernel Summary",
        "",
        "## Configuration",
        "",
        "- element_type: `%s`" % quadrature_rule.element_type,
        "- quadrature_order: `%d`" % quadrature_rule.order,
        "- dim: `%d`" % quadrature_rule.dim,
        "- n_nodes: `%d`" % quadrature_rule.n_shape,
        "- n_qp: `%d`" % quadrature_rule.n_qp,
        "- vector_size: `%d`" % specialization.vector_size,
        "- outputs: `%d`" % len(graph.outputs),
        "- statements: `%d`" % len(graph.evaluation_plan.statements),
        "- temporaries: `%d`" % len(graph.evaluation_plan.intermediates),
        "- flops: `%d`" % graph.cost.flops,
        "- estimated_registers: `%d`" % graph.cost.estimated_registers,
        "",
        "## Template Parameters",
        "",
    ]
    for parameter in graph.template_parameters:
        lines.append(
            "- `%s = %d` from `%s`"
            % (parameter.name, parameter.value, parameter.source)
        )
    lines.append("")
    return "\n".join(lines)


def compile_source(source_path, compiler):
    object_path = source_path + ".o"
    subprocess.run(
        [
            compiler,
            "-std=c++11",
            "-O2",
            "-c",
            source_path,
            "-o",
            object_path,
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return object_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--element-type", default="TRI3")
    parser.add_argument("--quadrature-order", type=int, default=None)
    parser.add_argument("--vector-size", type=int, default=8)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    specialization = sfem_soa_element_specialization(
        args.element_type,
        args.vector_size,
        args.quadrature_order,
    )
    forms = build_neohookean_forms(specialization)
    graph = build_combined_graph(forms)

    sfem_files = build_sfem_soa_files(
        forms,
        specialization,
    )

    outputs = [
        (
            "neohookean_ogden_summary.md",
            summary_markdown(graph, specialization),
        ),
        (
            "neohookean_ogden_reduced_outputs.txt",
            "\n\n".join(str(output) for output in graph.reduced_outputs) + "\n",
        ),
    ]
    outputs.extend((file.path, file.source) for file in sfem_files)

    written_paths = []
    for filename, source in outputs:
        path = os.path.join(args.out_dir, filename)
        write_text(path, source)
        written_paths.append(path)

    print("Generated:")
    for path in written_paths:
        print("  %s" % path)

    if args.compile:
        compiler = shutil.which("c++")
        if compiler is None:
            raise RuntimeError("c++ compiler is not available")
        compile_paths = [
            os.path.join(args.out_dir, "generated_neohookean_ogden_operator.cpp"),
        ]
        print("Compiled:")
        for path in compile_paths:
            print("  %s" % compile_source(path, compiler))


if __name__ == "__main__":
    main()
