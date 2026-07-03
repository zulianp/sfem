#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess

from codegen.framework.materials.laplace import material
from codegen.framework.mlir import (
    TensorProductLaplaceFormBatchedGPULowering,
    TensorProductLaplaceFormEBEGPULowering,
    TensorProductLaplaceFormEBEMetalLowering,
    TensorProductLaplaceFormGPULowering,
    TensorProductLaplaceFormLinalgLowering,
    TensorProductLaplaceFormMetalLowering,
    TensorProductLaplaceReferenceEvaluator,
    TensorProductSumFactorReferenceEvaluator,
    TensorProductSumFactorMLIRLowering,
    tensor_product_laplace_form_ir_from_user_input_stage,
    tensor_product_sum_factor_ir_from_user_input_stage,
)
from sfem import gen


def main():
    parser = argparse.ArgumentParser(
        description="Emit tensor-product Laplace SFEM IR, MLIR, generic GPU, and Metal inspection artifacts."
    )
    parser.add_argument("--output-dir", required=True, help="artifact output directory")
    parser.add_argument("--element", default="HEX27", help="tensor-product element, e.g. QUAD4, HEX8, HEX27")
    parser.add_argument("--vector-size", type=int, default=8)
    parser.add_argument("--quadrature-order", type=int, default=None)
    parser.add_argument("--max-elements", type=int, default=1024)
    parser.add_argument("--max-nodes", type=int, default=4096)
    parser.add_argument("--max-node-degree", type=int, default=32)
    parser.add_argument("--verify-reference", action="store_true", help="run deterministic CPU reference checks from SFEM IR")
    parser.add_argument(
        "--verify-performance-shape",
        action="store_true",
        help="check branch-free/no-atomic generated GPU and Metal kernel artifacts",
    )
    parser.add_argument("--validate-mlir", action="store_true", help="run mlir-opt --verify-diagnostics on .mlir files")
    parser.add_argument("--probe-iree-metal", action="store_true", help="try iree-compile --iree-hal-target-backends=metal-spirv")
    parser.add_argument(
        "--probe-iree-metal-matrix-unit",
        action="store_true",
        help="try iree-compile on the sum-factor matrix-unit pipeline boundary",
    )
    parser.add_argument(
        "--probe-iree-metal-gpu",
        action="store_true",
        help="try iree-compile on generated generic GPU dialect artifacts",
    )
    parser.add_argument("--run-iree-metal-runtime", action="store_true", help="compile and dispatch the full-form IREE Metal VMFB")
    parser.add_argument("--probe-metal-toolchain", action="store_true", help="try offline xcrun metal compilation for generated .metal files")
    parser.add_argument("--run-metal-smoke", action="store_true", help="compile and run local and EBE Metal smoke tests")
    parser.add_argument(
        "--require-iree-metal",
        action="store_true",
        help="return nonzero when the IREE Metal probe is skipped or fails",
    )
    parser.add_argument(
        "--require-iree-metal-matrix-unit",
        action="store_true",
        help="return nonzero when the matrix-unit IREE Metal probe is skipped or fails",
    )
    parser.add_argument(
        "--require-iree-metal-gpu",
        action="store_true",
        help="return nonzero when the generic GPU IREE Metal probe is skipped or fails",
    )
    parser.add_argument(
        "--require-iree-metal-runtime",
        action="store_true",
        help="return nonzero when the IREE Metal runtime smoke test is skipped or fails",
    )
    parser.add_argument(
        "--require-metal-device",
        action="store_true",
        help="return nonzero when Metal smoke tests compile but no default MTLDevice is available",
    )
    parser.add_argument(
        "--require-metal-toolchain",
        action="store_true",
        help="return nonzero when offline xcrun metal compilation is unavailable or fails",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    user_input = gen.UserInputStage.create(
        material,
        (args.element,),
        args.vector_size,
        args.quadrature_order,
    )
    sum_factor_ir = tensor_product_sum_factor_ir_from_user_input_stage(user_input)
    form_ir = tensor_product_laplace_form_ir_from_user_input_stage(user_input)

    artifact_groups = {}
    artifact_groups["sfem_ir"] = _write_ir_artifacts(output_dir / "sfem_ir", sum_factor_ir, form_ir)

    sum_factor = TensorProductSumFactorMLIRLowering(sum_factor_ir)
    artifact_groups["sum_factor"] = _write_artifacts(
        sum_factor.write_inspection_artifacts(output_dir / "sum_factor")
    )

    form_gpu = TensorProductLaplaceFormGPULowering(form_ir)
    artifact_groups["form"] = _write_artifacts(
        form_gpu.write_inspection_artifacts(output_dir / "form")
    )

    form_linalg = TensorProductLaplaceFormLinalgLowering(form_ir)
    artifact_groups["form_linalg"] = _write_artifacts(
        form_linalg.write_inspection_artifacts(output_dir / "form_linalg")
    )

    ebe_map = TensorProductLaplaceFormBatchedGPULowering(
        form_ir,
        max_elements=args.max_elements,
        max_nodes=args.max_nodes,
    )
    artifact_groups["ebe_map"] = _write_artifacts(
        ebe_map.write_inspection_artifacts(output_dir / "ebe_map")
    )

    ebe_full = TensorProductLaplaceFormEBEGPULowering(
        form_ir,
        max_elements=args.max_elements,
        max_nodes=args.max_nodes,
        max_node_degree=args.max_node_degree,
    )
    artifact_groups["ebe_full"] = _write_artifacts(
        ebe_full.write_inspection_artifacts(output_dir / "ebe_full")
    )

    ebe_metal = TensorProductLaplaceFormEBEMetalLowering(
        form_ir,
        max_elements=args.max_elements,
        max_nodes=args.max_nodes,
        max_node_degree=args.max_node_degree,
    )
    artifact_groups["ebe_metal"] = _write_artifacts(
        ebe_metal.write_inspection_artifacts(output_dir / "ebe_metal")
    )

    manifest = {
        "material": material.name,
        "element": sum_factor_ir.element_type,
        "vector_size": sum_factor_ir.vector_size,
        "quadrature_order": sum_factor_ir.quadrature_order,
        "dim": sum_factor_ir.dim,
        "n_shape": sum_factor_ir.n_shape,
        "n_qp": sum_factor_ir.n_qp,
        "n_shape_1d": sum_factor_ir.n_shape_1d,
        "n_qp_1d": sum_factor_ir.n_qp_1d,
        "max_elements": args.max_elements,
        "max_nodes": args.max_nodes,
        "max_node_degree": args.max_node_degree,
        "artifacts": artifact_groups,
        "iree_metal": [],
        "iree_metal_gpu": [],
        "iree_metal_matrix_unit": [],
        "iree_metal_runtime": [],
        "mlir_validation": [],
        "metal_smoke": [],
        "metal_toolchain": [],
        "performance_shape": [],
        "reference_verification": [],
        "spirv_binary_validation": [],
    }

    if args.verify_reference:
        manifest["reference_verification"] = _verify_reference(
            form_ir,
            args.max_elements,
            args.max_nodes,
            args.max_node_degree,
        )

    if args.verify_performance_shape:
        manifest["performance_shape"] = _verify_performance_shape(output_dir)

    if args.validate_mlir:
        manifest["mlir_validation"] = _validate_mlir_files(output_dir)
        manifest["spirv_binary_validation"] = _validate_spirv_binaries(output_dir)

    if args.probe_iree_metal:
        manifest["iree_metal"] = _probe_iree_metal(
            output_dir / "iree_metal",
            [
                ("sum_factor_linalg_pipeline_to_metal_vmfb", sum_factor),
                ("laplace_form_linalg_pipeline_to_metal_vmfb", form_linalg),
            ],
        )

    if args.probe_iree_metal_matrix_unit:
        manifest["iree_metal_matrix_unit"] = _probe_iree_metal(
            output_dir / "iree_metal_matrix_unit",
            [
                (
                    "sum_factor_matrix_unit_to_metal_vmfb",
                    sum_factor,
                    "matrix_unit",
                ),
                (
                    "sum_factor_matrix_unit_memref_to_metal_vmfb",
                    sum_factor,
                    "matrix_unit_memref",
                ),
                (
                    "sum_factor_matrix_unit_pipeline_to_metal_vmfb",
                    sum_factor,
                    "matrix_unit_pipeline",
                ),
            ],
        )

    if args.probe_iree_metal_gpu:
        manifest["iree_metal_gpu"] = _probe_iree_metal_gpu_artifacts(
            output_dir / "iree_metal_gpu",
            _gpu_artifact_paths(artifact_groups),
        )

    if args.run_iree_metal_runtime:
        manifest["iree_metal_runtime"] = [
            _run_sum_factor_iree_metal_runtime_smoke(
                output_dir / "iree_metal_runtime" / "sum_factor",
                sum_factor_ir,
                sum_factor,
            ),
            _run_iree_metal_runtime_smoke(output_dir / "iree_metal_runtime" / "form", form_ir, form_linalg),
        ]

    if args.probe_metal_toolchain:
        manifest["metal_toolchain"] = _probe_metal_toolchain(output_dir)

    if args.run_metal_smoke:
        manifest["metal_smoke"] = _run_metal_smoke_tests(
            output_dir,
            sum_factor,
            TensorProductLaplaceFormMetalLowering(form_ir),
            ebe_metal,
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(str(manifest_path))

    if args.require_metal_device:
        for result in manifest["metal_smoke"]:
            if result.get("no_default_device"):
                return 3
    if args.require_iree_metal:
        if not manifest["iree_metal"] or any(not item.get("ok", False) for item in manifest["iree_metal"]):
            return 5
    if args.require_iree_metal_matrix_unit:
        if not manifest["iree_metal_matrix_unit"] or any(not item.get("ok", False) for item in manifest["iree_metal_matrix_unit"]):
            return 11
    if args.require_iree_metal_gpu:
        if not manifest["iree_metal_gpu"] or any(not item.get("ok", False) for item in manifest["iree_metal_gpu"]):
            return 12
    if args.require_iree_metal_runtime:
        if not manifest["iree_metal_runtime"] or any(not item.get("ok", False) for item in manifest["iree_metal_runtime"]):
            return 9
    if args.require_metal_toolchain:
        if not manifest["metal_toolchain"] or any(not item.get("ok", False) for item in manifest["metal_toolchain"]):
            return 8
    if any(not item.get("ok", False) for item in manifest["mlir_validation"]):
        return 2
    if any(not item.get("ok", False) and not item.get("skipped", False) for item in manifest["spirv_binary_validation"]):
        return 10
    if any(not item.get("ok", False) and not item.get("no_default_device", False) for item in manifest["metal_smoke"]):
        return 4
    if any(not item.get("ok", False) for item in manifest["reference_verification"]):
        return 6
    if any(not item.get("ok", False) for item in manifest["performance_shape"]):
        return 7
    return 0


def _write_artifacts(artifacts):
    return [str(path) for path in artifacts.paths]


def _gpu_artifact_paths(artifact_groups):
    paths = []
    for group, files in artifact_groups.items():
        for path in files:
            if path.endswith(".gpu.mlir"):
                paths.append((group, Path(path)))
    return tuple(paths)


def _write_ir_artifacts(output_dir, sum_factor_ir, form_ir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "%s_%s" % (sum_factor_ir.material_name, sum_factor_ir.element_label)
    sum_factor_path = output_dir / ("%s.sum_factor.ir.json" % prefix)
    form_path = output_dir / ("%s.laplace_form.ir.json" % prefix)
    sum_factor_path.write_text(json.dumps(sum_factor_ir.to_dict(), indent=2, sort_keys=True) + "\n")
    form_path.write_text(json.dumps(form_ir.to_dict(), indent=2, sort_keys=True) + "\n")
    return [str(sum_factor_path), str(form_path)]


def _verify_reference(form_ir, max_elements, max_nodes, max_node_degree):
    return [
        _verify_sum_factor_reference(form_ir),
        _verify_laplace_reference(form_ir, max_elements, max_nodes, max_node_degree),
    ]


def _verify_sum_factor_reference(form_ir):
    sf = form_ir.sum_factor
    evaluator = TensorProductSumFactorReferenceEvaluator(sf)
    u = [0.5 + 0.03125 * (i + 1) for i in range(sf.n_shape)]
    max_field_diff = 0.0
    max_test_diff = 0.0
    field_checksum = 0.0
    test_checksum = 0.0
    for derivative in range(sf.dim):
        field = evaluator.field_gradient(u, derivative)
        direct_field = evaluator.direct_field_gradient(u, derivative)
        max_field_diff = max(
            max_field_diff,
            max(abs(a - b) for a, b in zip(field, direct_field)) if field else 0.0,
        )
        field_checksum += sum((i + 1) * value for i, value in enumerate(field))

        q_values = [0.25 + 0.015625 * (derivative + 1) * (i + 1) for i in range(sf.n_qp)]
        test = evaluator.test_contraction(q_values, derivative)
        direct_test = evaluator.direct_test_contraction(q_values, derivative)
        max_test_diff = max(
            max_test_diff,
            max(abs(a - b) for a, b in zip(test, direct_test)) if test else 0.0,
        )
        test_checksum += sum((i + 1) * value for i, value in enumerate(test))

    sum_factor_laplace = evaluator.apply_laplace_local(u, kappa=form_ir.parameter_default)
    direct_laplace = evaluator.direct_laplace_local(u, kappa=form_ir.parameter_default)
    max_laplace_diff = max(
        abs(a - b) for a, b in zip(sum_factor_laplace, direct_laplace)
    ) if sum_factor_laplace else 0.0
    return {
        "name": "tensor_product_sum_factor_pipeline_reference",
        "ok": max_field_diff <= 1.0e-6 and max_test_diff <= 1.0e-6 and max_laplace_diff <= 1.0e-6,
        "skipped": False,
        "max_field_gradient_diff": max_field_diff,
        "max_test_contraction_diff": max_test_diff,
        "max_laplace_residual_diff": max_laplace_diff,
        "field_gradient_checksum": field_checksum,
        "test_contraction_checksum": test_checksum,
    }


def _verify_laplace_reference(form_ir, max_elements, max_nodes, max_node_degree):
    sf = form_ir.sum_factor
    if max_elements < 1 or max_nodes < sf.n_shape:
        return {
            "name": "tensor_product_laplace_cpu_reference",
            "ok": False,
            "skipped": True,
            "reason": "bounds are too small for one tensor-product element",
        }
    num_elements = 1
    if max_elements >= 2 and max_nodes >= 2 * sf.n_shape - 1:
        num_elements = 2

    connectivity = [tuple(range(sf.n_shape))]
    if num_elements == 2:
        connectivity.append(tuple(range(sf.n_shape - 1, 2 * sf.n_shape - 1)))

    node_degree = [0 for _ in range(max_nodes)]
    node_to_element_map = [[0 for _ in range(max_node_degree)] for _ in range(max_nodes)]
    node_to_local_idx = [[0 for _ in range(max_node_degree)] for _ in range(max_nodes)]
    for elem, nodes in enumerate(connectivity):
        for local, node in enumerate(nodes):
            degree = node_degree[node]
            if degree >= max_node_degree:
                return {
                    "name": "tensor_product_laplace_cpu_reference",
                    "ok": False,
                    "skipped": True,
                    "reason": "max_node_degree is too small for generated fixture",
                }
            node_to_element_map[node][degree] = elem
            node_to_local_idx[node][degree] = local
            node_degree[node] = degree + 1

    evaluator = TensorProductLaplaceReferenceEvaluator(form_ir)
    constant_u = [1.0 for _ in range(max_nodes)]
    _, constant_out = evaluator.apply_ebe(
        connectivity,
        constant_u,
        node_degree,
        node_to_element_map,
        node_to_local_idx,
    )
    max_abs_constant = max(abs(value) for value in constant_out) if constant_out else 0.0

    deterministic_u = [0.5 + 0.03125 * (i + 1) for i in range(max_nodes)]
    element_out, out = evaluator.apply_ebe(
        connectivity,
        deterministic_u,
        node_degree,
        node_to_element_map,
        node_to_local_idx,
    )
    checksum = sum((i + 1) * value for i, value in enumerate(out))
    element_checksum = sum((elem + 1) * (local + 1) * value for elem, row in enumerate(element_out) for local, value in enumerate(row))
    return {
        "name": "tensor_product_laplace_cpu_reference",
        "ok": max_abs_constant <= 1.0e-5,
        "skipped": False,
        "num_elements": num_elements,
        "num_nodes": max_nodes,
        "max_abs_constant_residual": max_abs_constant,
        "deterministic_output_checksum": checksum,
        "deterministic_element_checksum": element_checksum,
    }


def _verify_performance_shape(output_dir):
    output_dir = Path(output_dir)
    results = []
    linalg_files = sorted(output_dir.rglob("*.linalg.mlir"))
    linalg_pipeline_files = sorted(output_dir.rglob("*.linalg_pipeline.mlir"))
    vector_files = sorted(output_dir.rglob("*.vector.mlir"))
    matrix_unit_files = sorted(output_dir.rglob("*.matrix_unit.mlir"))
    matrix_unit_memref_files = sorted(output_dir.rglob("*.matrix_unit_memref.mlir"))
    matrix_unit_pipeline_files = sorted(output_dir.rglob("*.matrix_unit_pipeline.mlir"))
    gpu_files = sorted(output_dir.rglob("*.gpu.mlir"))
    spirv_opencl_files = sorted(output_dir.rglob("*.spirv.opencl.mlir"))
    spirv_opencl_op_files = sorted(output_dir.rglob("*.spirv.opencl.op.mlir"))
    spirv_opencl_binary_files = sorted(output_dir.rglob("*.spirv.opencl.spv"))
    spirv_opencl_dispatch_files = sorted(output_dir.rglob("*.spirv.opencl.dispatch.json"))
    metal_files = sorted(output_dir.rglob("*.metal"))

    results.append(_require_token("linalg_matmul", linalg_files, "linalg.matmul"))
    results.append(_require_token("linalg_fill_accumulators", linalg_files + linalg_pipeline_files, "linalg.fill"))
    results.append(_require_token("linalg_pipeline_calls", linalg_pipeline_files, "func.call"))
    results.append(_require_token("linalg_pipeline_bridges", linalg_pipeline_files, "linalg.generic"))
    results.extend(_verify_laplace_form_derivative_coverage(path) for path in linalg_pipeline_files)
    results.append(_require_token("vector_matrix_multiply", vector_files, "vector.matrix_multiply"))
    results.append(_require_token("matrix_unit_vector_matrix_multiply", matrix_unit_files, "vector.matrix_multiply"))
    results.append(_require_token("matrix_unit_memref_transfer_read", matrix_unit_memref_files, "vector.transfer_read"))
    results.append(_require_token("matrix_unit_memref_transfer_write", matrix_unit_memref_files, "vector.transfer_write"))
    results.append(_require_token("matrix_unit_pipeline_calls", matrix_unit_pipeline_files, "func.call"))
    results.append(_require_token("matrix_unit_pipeline_scratch", matrix_unit_pipeline_files, "memref.alloc"))
    results.extend(_verify_matrix_unit_alignment(path) for path in matrix_unit_files)
    results.extend(_verify_matrix_unit_alignment(path) for path in matrix_unit_memref_files)
    results.extend(_verify_matrix_unit_alignment(path) for path in matrix_unit_pipeline_files)
    results.append(_require_token("gpu_spirv_entry_point_abi", gpu_files, "spirv.entry_point_abi"))
    results.append(_require_token("gpu_spirv_interface_var_abi", gpu_files, "spirv.interface_var_abi"))
    results.extend(_verify_sum_factor_gpu_dispatch_coverage(path) for path in gpu_files)
    results.extend(_verify_laplace_gpu_dispatch_coverage(path) for path in gpu_files)
    results.append(_require_token("spirv_opencl_module", spirv_opencl_files, "spirv.module Logical OpenCL"))
    results.append(_require_token("spirv_opencl_entry_points", spirv_opencl_files, 'spirv.EntryPoint "Kernel"'))
    results.append(_require_token("spirv_opencl_cross_workgroup_access", spirv_opencl_files, "spirv.AccessChain"))
    results.append(_require_token("spirv_opencl_unrolled_contractions", spirv_opencl_files, "spirv.FMul"))
    results.append(_require_token("spirv_opencl_accumulators", spirv_opencl_files, "spirv.FAdd"))
    results.append(_require_token("spirv_opencl_module_op", spirv_opencl_op_files, "spirv.module Logical OpenCL"))
    results.append(_require_nonempty_files("spirv_opencl_serialized_binary", spirv_opencl_binary_files))
    results.extend(_verify_spirv_opencl_dispatch(path) for path in spirv_opencl_dispatch_files)
    for path in gpu_files:
        results.append(_forbid_tokens("gpu_hot_loop_shape", path, ("scf.if", "arith.cmpi", "atomic")))
        if path.name.endswith(".ebe.full.gpu.mlir"):
            results.append(_verify_ebe_gpu_topology(path))
    for path in spirv_opencl_files:
        results.append(_forbid_tokens("spirv_opencl_hot_loop_shape", path, ("scf.if", "arith.cmpi", "atomic")))
    for path in spirv_opencl_op_files:
        results.append(_forbid_tokens("spirv_opencl_op_hot_loop_shape", path, ("scf.if", "arith.cmpi", "atomic")))
    for path in metal_files:
        results.append(_forbid_tokens("metal_hot_loop_shape", path, ("if (", "atomic")))
    return results


def _require_token(name, files, token):
    matching = []
    for path in files:
        if token in path.read_text():
            matching.append(str(path))
    return {
        "name": name,
        "ok": bool(matching),
        "token": token,
        "matching_paths": matching,
        "checked_paths": [str(path) for path in files],
    }


def _require_nonempty_files(name, files):
    matching = [str(path) for path in files if path.is_file() and path.stat().st_size > 0]
    return {
        "name": name,
        "ok": bool(matching),
        "matching_paths": matching,
        "checked_paths": [str(path) for path in files],
    }


def _verify_spirv_opencl_dispatch(path):
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return {
            "name": "spirv_opencl_dispatch_metadata",
            "path": str(path),
            "ok": False,
            "reason": str(exc),
        }
    stages = payload.get("stages", [])
    missing = []
    if payload.get("lowering") != "tensor_product_sum_factor_spirv_opencl":
        missing.append("lowering")
    if not stages:
        missing.append("stages")
    for index, stage in enumerate(stages):
        if stage.get("index") != index:
            missing.append("stage_index")
        for key in ("kernel", "stage", "operation", "global_work_items", "result_elements"):
            if key not in stage:
                missing.append(key)
        if int(stage.get("global_work_items", -1)) != int(stage.get("result_elements", -2)):
            missing.append("global_work_items")
    return {
        "name": "spirv_opencl_dispatch_metadata",
        "path": str(path),
        "ok": not missing,
        "n_stages": len(stages),
        "missing_tokens": missing,
    }


def _verify_laplace_form_derivative_coverage(path):
    text = path.read_text()
    if 'sfem.lowering = "tensor_product_laplace_form_linalg_pipeline"' not in text:
        return {
            "name": "laplace_form_derivative_coverage",
            "path": str(path),
            "ok": True,
            "skipped": True,
            "reason": "not a full-form linalg pipeline artifact",
        }
    if not re.search(r"func\.func @[A-Za-z0-9_]+_linalg_pipeline\([^\n]*sfem\.form = \"laplace\"", text):
        return {
            "name": "laplace_form_derivative_coverage",
            "path": str(path),
            "ok": False,
            "reason": "missing full-form linalg pipeline entry point",
        }
    field_matches = re.findall(r"@([A-Za-z0-9_]+)_field_gradient_d([0-9]+)_linalg_pipeline", text)
    test_matches = re.findall(r"@([A-Za-z0-9_]+)_test_gradient_d([0-9]+)_linalg_pipeline", text)
    prefixes = sorted({prefix for prefix, _ in field_matches + test_matches})
    if len(prefixes) != 1:
        return {
            "name": "laplace_form_derivative_coverage",
            "path": str(path),
            "ok": False,
            "reason": "expected one sum-factor derivative pipeline prefix",
            "prefixes": prefixes,
        }
    prefix = prefixes[0]
    field_ids = sorted(int(value) for _prefix, value in set(field_matches))
    test_ids = sorted(int(value) for _prefix, value in set(test_matches))
    expected = list(range(max(field_ids + test_ids) + 1)) if field_ids or test_ids else []
    missing = []
    if field_ids != expected:
        missing.append("field_gradient_derivatives")
    if test_ids != expected:
        missing.append("test_gradient_derivatives")
    for derivative in expected:
        for token in (
            "func.call @%s_field_gradient_d%d_linalg_pipeline" % (prefix, derivative),
            "func.call @%s_test_gradient_d%d_linalg_pipeline" % (prefix, derivative),
            "%%weighted_d%d" % derivative,
            "%%test_d%d" % derivative,
        ):
            if token not in text:
                missing.append(token)
    return {
        "name": "laplace_form_derivative_coverage",
        "path": str(path),
        "ok": bool(expected) and not missing,
        "derivatives": expected,
        "field_derivatives": field_ids,
        "test_derivatives": test_ids,
        "missing_tokens": missing,
    }


def _verify_sum_factor_gpu_dispatch_coverage(path):
    text = path.read_text()
    if 'sfem.lowering = "tensor_product_sum_factor_gpu"' not in text:
        return {
            "name": "sum_factor_gpu_dispatch_coverage",
            "path": str(path),
            "ok": True,
            "skipped": True,
            "reason": "not a sum-factor GPU artifact",
        }
    kernel_names = sorted(set(re.findall(r"gpu\.func @([A-Za-z0-9_]+_kernel)\(", text)))
    launch_targets = sorted(set(re.findall(r"gpu\.launch_func @[A-Za-z0-9_]+::@([A-Za-z0-9_]+_kernel)", text)))
    launch_functions = sorted(set(re.findall(r"func\.func @([A-Za-z0-9_]+_gpu)\(", text)))
    thread_constants = re.findall(r"%threads_x = arith\.constant ([0-9]+) : index\n    %threads_y = arith\.constant ([0-9]+) : index", text)
    derivative_ids = sorted(int(value) for value in set(re.findall(r"sfem\.sum_factor\.derivative = ([0-9]+) : i64", text)))
    missing = []
    if not kernel_names:
        missing.append("gpu.func")
    if kernel_names != launch_targets:
        missing.append("launch_targets")
    if len(launch_functions) != len(kernel_names):
        missing.append("launch_functions")
    if len(thread_constants) != len(kernel_names):
        missing.append("thread_constants")
    if text.count("spirv.entry_point_abi =") != len(kernel_names):
        missing.append("spirv.entry_point_abi")
    if text.count("spirv.interface_var_abi =") != 3 * len(kernel_names):
        missing.append("spirv.interface_var_abi")
    if any(int(x) <= 0 or int(y) <= 0 for x, y in thread_constants):
        missing.append("thread_geometry")
    if not derivative_ids or derivative_ids != list(range(max(derivative_ids) + 1)):
        missing.append("derivatives")
    return {
        "name": "sum_factor_gpu_dispatch_coverage",
        "path": str(path),
        "ok": not missing,
        "kernel_count": len(kernel_names),
        "launch_count": len(launch_targets),
        "launch_function_count": len(launch_functions),
        "derivatives": derivative_ids,
        "thread_geometries": [[int(x), int(y)] for x, y in thread_constants],
        "missing_tokens": missing,
    }


def _verify_laplace_gpu_dispatch_coverage(path):
    text = path.read_text()
    lowering_match = re.search(r'sfem\.lowering = "([^"]+)"', text)
    lowering = lowering_match.group(1) if lowering_match else ""
    expected_by_lowering = {
        "tensor_product_laplace_form_gpu": {
            "kernel_count": 1,
            "launch_count": 1,
            "interface_var_abi_count": 6,
            "thread_constant_count": 1,
            "required_tokens": (
                'sfem.form = "laplace"',
                'sfem.parameter = "kappa"',
            ),
        },
        "tensor_product_laplace_ebe_gpu_map": {
            "kernel_count": 1,
            "launch_count": 1,
            "interface_var_abi_count": 7,
            "thread_constant_count": 1,
            "required_tokens": (
                'sfem.mesh_phase = "ebe_map"',
                "%elem = gpu.block_id x",
                "%node = memref.load %connectivity[%elem, %trial]",
            ),
        },
        "tensor_product_laplace_ebe_gpu": {
            "kernel_count": 2,
            "launch_count": 2,
            "interface_var_abi_count": 12,
            "thread_constant_count": 2,
            "required_tokens": (
                'sfem.mesh_phases = "ebe_map,ebe_reduce"',
                "_ebe_map_kernel",
                "_ebe_reduce_kernel",
                "scf.for %i = %c0 to %degree",
            ),
        },
    }
    expected = expected_by_lowering.get(lowering)
    if expected is None:
        return {
            "name": "laplace_gpu_dispatch_coverage",
            "path": str(path),
            "ok": True,
            "skipped": True,
            "reason": "not a Laplace form or EBE GPU artifact",
        }

    kernel_names = sorted(set(re.findall(r"gpu\.func @([A-Za-z0-9_]+_kernel)\(", text)))
    launch_targets = sorted(set(re.findall(r"gpu\.launch_func @[A-Za-z0-9_]+::@([A-Za-z0-9_]+_kernel)", text)))
    thread_constants = [
        int(value)
        for value in re.findall(r"%[A-Za-z0-9_]*threads? = arith\.constant ([0-9]+) : index", text)
    ]
    missing = []
    if len(kernel_names) != expected["kernel_count"]:
        missing.append("gpu.func")
    if len(launch_targets) != expected["launch_count"] or kernel_names != launch_targets:
        missing.append("launch_targets")
    if text.count("spirv.entry_point_abi =") != expected["kernel_count"]:
        missing.append("spirv.entry_point_abi")
    if text.count("spirv.interface_var_abi =") != expected["interface_var_abi_count"]:
        missing.append("spirv.interface_var_abi")
    if len(thread_constants) != expected["thread_constant_count"] or any(value <= 0 for value in thread_constants):
        missing.append("thread_constants")
    missing.extend(token for token in expected["required_tokens"] if token not in text)
    return {
        "name": "laplace_gpu_dispatch_coverage",
        "path": str(path),
        "ok": not missing,
        "lowering": lowering,
        "kernel_count": len(kernel_names),
        "launch_count": len(launch_targets),
        "entry_point_abi_count": text.count("spirv.entry_point_abi ="),
        "interface_var_abi_count": text.count("spirv.interface_var_abi ="),
        "thread_constants": thread_constants,
        "missing_tokens": missing,
    }


def _forbid_tokens(name, path, tokens):
    text = path.read_text()
    found = [token for token in tokens if token in text]
    return {
        "name": name,
        "path": str(path),
        "ok": not found,
        "forbidden_tokens": list(tokens),
        "found_tokens": found,
    }


def _verify_matrix_unit_alignment(path):
    text = path.read_text()
    tile_match = re.search(r"sfem\.matrix_unit\.tile_size = ([0-9]+) : i64", text)
    if not tile_match:
        return {
            "name": "matrix_unit_alignment",
            "path": str(path),
            "ok": False,
            "reason": "missing sfem.matrix_unit.tile_size attribute",
        }
    tile_size = int(tile_match.group(1))
    matches = re.findall(
        r"lhs_rows = ([0-9]+) : i32, lhs_columns = ([0-9]+) : i32, rhs_columns = ([0-9]+) : i32",
        text,
    )
    misaligned = []
    for lhs_rows, lhs_cols, rhs_cols in matches:
        dims = (int(lhs_rows), int(lhs_cols), int(rhs_cols))
        if any(value % tile_size != 0 for value in dims):
            misaligned.append(dims)
    return {
        "name": "matrix_unit_alignment",
        "path": str(path),
        "ok": bool(matches) and not misaligned,
        "tile_size": tile_size,
        "checked_stages": len(matches),
        "misaligned_dimensions": misaligned,
    }


def _verify_ebe_gpu_topology(path):
    text = path.read_text()
    required_tokens = (
        'sfem.mesh_phases = "ebe_map,ebe_reduce"',
        "gpu.launch_func",
        "_ebe_map_kernel",
        "_ebe_reduce_kernel",
        "%node = gpu.thread_id x",
        "%elem = memref.load %node_to_element_map[%node, %i]",
        "%local = memref.load %node_to_local_idx[%node, %i]",
        "%value = memref.load %element_out[%elem, %local]",
        "memref.store %sum, %out[%node]",
    )
    missing = [token for token in required_tokens if token not in text]
    map_store_count = text.count("memref.store %sum_")
    reduce_store_count = text.count("memref.store %sum, %out[%node]")
    map_launch_count = text.count("_ebe_map_kernel")
    reduce_launch_count = text.count("_ebe_reduce_kernel")
    ok = (
        not missing
        and map_store_count > 0
        and reduce_store_count == 1
        and map_launch_count >= 2
        and reduce_launch_count >= 2
        and "atomic" not in text
    )
    return {
        "name": "ebe_gpu_topology",
        "path": str(path),
        "ok": ok,
        "missing_tokens": missing,
        "map_store_count": map_store_count,
        "reduce_store_count": reduce_store_count,
        "map_symbol_mentions": map_launch_count,
        "reduce_symbol_mentions": reduce_launch_count,
        "forbidden_tokens": ["atomic"],
        "found_tokens": ["atomic"] if "atomic" in text else [],
    }


def _validate_mlir_files(output_dir):
    mlir_opt = shutil.which("mlir-opt") or "/opt/homebrew/opt/llvm/bin/mlir-opt"
    files = sorted(Path(output_dir).rglob("*.mlir"))
    if not mlir_opt or not Path(mlir_opt).exists():
        return [
            {
                "path": str(path),
                "ok": False,
                "skipped": True,
                "reason": "mlir-opt is not available",
            }
            for path in files
        ]
    results = []
    for path in files:
        result = subprocess.run(
            [mlir_opt, str(path), "--verify-diagnostics"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        results.append(
            {
                "path": str(path),
                "ok": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
    return results


def _validate_spirv_binaries(output_dir):
    mlir_translate = shutil.which("mlir-translate") or "/opt/homebrew/opt/llvm/bin/mlir-translate"
    files = sorted(Path(output_dir).rglob("*.spirv.opencl.spv"))
    if not mlir_translate or not Path(mlir_translate).exists():
        return [
            {
                "path": str(path),
                "ok": False,
                "skipped": True,
                "reason": "mlir-translate is not available",
            }
            for path in files
        ]
    results = []
    for path in files:
        result = subprocess.run(
            [mlir_translate, str(path), "--deserialize-spirv"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        text = result.stdout
        expected_tokens = (
            "spirv.module Logical OpenCL",
            'spirv.EntryPoint "Kernel"',
            "spirv.AccessChain",
            "spirv.FMul",
            "spirv.FAdd",
        )
        missing = [token for token in expected_tokens if token not in text]
        results.append(
            {
                "path": str(path),
                "ok": result.returncode == 0 and not missing,
                "returncode": result.returncode,
                "stdout_preview": text[:1000],
                "stderr": result.stderr,
                "missing_tokens": missing,
                "size": path.stat().st_size if path.exists() else 0,
            }
        )
    return results


def _probe_iree_metal(output_dir, lowerings):
    iree_compile = shutil.which("iree-compile")
    if iree_compile is None:
        return [
            {
                "name": _iree_probe_name(config),
                "ok": False,
                "skipped": True,
                "input_kind": _iree_probe_input_kind(config),
                "reason": "iree-compile is not available",
            }
            for config in lowerings
        ]
    output_dir = Path(output_dir)
    results = []
    for lowering_config in lowerings:
        if len(lowering_config) == 2:
            name, lowering = lowering_config
            input_kind = None
        elif len(lowering_config) == 3:
            name, lowering, input_kind = lowering_config
        else:
            raise ValueError("IREE Metal probe entries must have name, lowering, and optional input kind")
        try:
            compile_kwargs = {"iree_compile": iree_compile}
            if input_kind is not None:
                compile_kwargs["input_kind"] = input_kind
            output_vmfb, result = lowering.compile_with_iree_metal(output_dir / name, **compile_kwargs)
        except subprocess.CalledProcessError as exc:
            stdout_info = _write_probe_stream(output_dir / name, "stdout", exc.stdout or "")
            stderr_info = _write_probe_stream(output_dir / name, "stderr", exc.stderr or "")
            diagnostic = _classify_iree_probe_diagnostic(exc.stderr or "")
            results.append(
                {
                    "name": name,
                    "ok": False,
                    "skipped": False,
                    "input_kind": input_kind,
                    "returncode": exc.returncode,
                    **diagnostic,
                    **stdout_info,
                    **stderr_info,
                }
            )
            continue
        stdout_info = _write_probe_stream(output_dir / name, "stdout", result.stdout or "")
        stderr_info = _write_probe_stream(output_dir / name, "stderr", result.stderr or "")
        diagnostic = _classify_iree_probe_diagnostic(result.stderr or "")
        results.append(
            {
                "name": name,
                "ok": result.returncode == 0 and output_vmfb.exists(),
                "skipped": False,
                "input_kind": input_kind,
                "returncode": result.returncode,
                "output_vmfb": str(output_vmfb),
                **diagnostic,
                **stdout_info,
                **stderr_info,
            }
        )
    return results


def _probe_iree_metal_gpu_artifacts(output_dir, gpu_artifacts):
    iree_compile = shutil.which("iree-compile")
    if iree_compile is None:
        return [
            {
                "name": _iree_gpu_probe_name(group, path),
                "ok": False,
                "skipped": True,
                "artifact_group": group,
                "input_path": str(path),
                "reason": "iree-compile is not available",
            }
            for group, path in gpu_artifacts
        ]
    output_dir = Path(output_dir)
    results = []
    for group, path in gpu_artifacts:
        name = _iree_gpu_probe_name(group, path)
        probe_dir = output_dir / name
        probe_dir.mkdir(parents=True, exist_ok=True)
        output_vmfb = probe_dir / (Path(path).stem + ".metal.vmfb")
        result = subprocess.run(
            [
                iree_compile,
                str(path),
                "--iree-hal-target-backends=metal-spirv",
                "--iree-metal-compile-to-metallib=false",
                "-o",
                str(output_vmfb),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout_info = _write_probe_stream(probe_dir, "stdout", result.stdout or "")
        stderr_info = _write_probe_stream(probe_dir, "stderr", result.stderr or "")
        diagnostic = _classify_iree_probe_diagnostic(result.stderr or "")
        results.append(
            {
                "name": name,
                "ok": result.returncode == 0 and output_vmfb.exists(),
                "skipped": False,
                "artifact_group": group,
                "input_path": str(path),
                "returncode": result.returncode,
                "output_vmfb": str(output_vmfb) if output_vmfb.exists() else "",
                **diagnostic,
                **stdout_info,
                **stderr_info,
            }
        )
    return results


def _iree_gpu_probe_name(group, path):
    stem = Path(path).name
    for suffix in (
        ".ebe.full.gpu.mlir",
        ".ebe.gpu.mlir",
        ".gpu.mlir",
    ):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    safe_group = re.sub(r"[^A-Za-z0-9_]+", "_", group).strip("_")
    safe_stem = re.sub(r"[^A-Za-z0-9_]+", "_", stem).strip("_")
    return "%s_%s_gpu_to_metal_vmfb" % (safe_group, safe_stem)


def _iree_probe_name(config):
    return config[0]


def _iree_probe_input_kind(config):
    return config[2] if len(config) > 2 else None


def _classify_iree_probe_diagnostic(stderr):
    stderr = stderr or ""
    if not stderr:
        return {
            "diagnostic_kind": "",
            "diagnostic_summary": "",
        }
    if "failed to legalize unresolved materialization" in stderr:
        if "vector.matrix_multiply" in stderr or "vector.transfer_read" in stderr:
            return {
                "diagnostic_kind": "iree_vm_matrix_unit_abi_conversion",
                "diagnostic_summary": (
                    "IREE VM conversion left unresolved materialization around matrix-unit "
                    "vector/memref operands"
                ),
            }
        if "gpu.launch_func" in stderr or "scf.for" in stderr:
            return {
                "diagnostic_kind": "iree_vm_generic_gpu_index_conversion",
                "diagnostic_summary": (
                    "IREE VM conversion left unresolved index materialization in generic GPU artifact"
                ),
            }
        return {
            "diagnostic_kind": "iree_vm_unresolved_materialization",
            "diagnostic_summary": "IREE VM conversion left unresolved materialization",
        }
    if "Resolution of CPU to CPU-features is not implemented" in stderr:
        return {
            "diagnostic_kind": "iree_host_cpu_feature_resolution",
            "diagnostic_summary": "IREE emitted a host CPU feature resolution diagnostic",
        }
    return {
        "diagnostic_kind": "iree_compile_failure",
        "diagnostic_summary": stderr.splitlines()[0] if stderr.splitlines() else "iree-compile failed",
    }


def _write_probe_stream(output_dir, stream_name, text, preview_bytes=4096):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    text = text or ""
    key_prefix = stream_name
    path = output_dir / ("iree_compile.%s.txt" % stream_name)
    if text:
        path.write_text(text)
        preview = text[:preview_bytes]
        return {
            "%s_path" % key_prefix: str(path),
            "%s_bytes" % key_prefix: len(text.encode("utf-8")),
            "%s_preview" % key_prefix: preview,
            "%s_truncated" % key_prefix: len(text) > len(preview),
        }
    return {
        "%s_path" % key_prefix: "",
        "%s_bytes" % key_prefix: 0,
        "%s_preview" % key_prefix: "",
        "%s_truncated" % key_prefix: False,
    }


def _run_sum_factor_iree_metal_runtime_smoke(output_dir, sum_factor_ir, sum_factor_lowering):
    try:
        import numpy as np
        import iree.runtime as iree_runtime
    except ImportError as exc:
        return {
            "name": "sum_factor_linalg_pipeline_metal_runtime",
            "ok": False,
            "skipped": True,
            "reason": "iree-runtime is not available: %s" % exc,
        }
    iree_compile = shutil.which("iree-compile")
    if iree_compile is None:
        return {
            "name": "sum_factor_linalg_pipeline_metal_runtime",
            "ok": False,
            "skipped": True,
            "reason": "iree-compile is not available",
        }
    if "metal" not in iree_runtime.query_available_drivers():
        return {
            "name": "sum_factor_linalg_pipeline_metal_runtime",
            "ok": False,
            "skipped": True,
            "reason": "IREE runtime metal driver is not available",
        }
    try:
        iree_runtime.get_driver("metal").create_default_device()
    except Exception as exc:
        return {
            "name": "sum_factor_linalg_pipeline_metal_runtime",
            "ok": False,
            "skipped": True,
            "no_default_device": True,
            "reason": str(exc),
        }
    try:
        output_vmfb, compile_result = sum_factor_lowering.compile_with_iree_metal(
            output_dir,
            iree_compile=iree_compile,
            input_kind="linalg_pipeline",
        )
        module = iree_runtime.load_vm_flatbuffer_file(str(output_vmfb), driver="metal")
        evaluator = TensorProductSumFactorReferenceEvaluator(sum_factor_ir)
        u_values = [0.5 + 0.03125 * (index + 1) for index in range(sum_factor_ir.n_shape)]
        max_abs_diff = 0.0
        function_shapes = {}
        function_count = 0
        for derivative in range(sum_factor_ir.dim):
            field_stages = tuple(
                stage for stage in sum_factor_ir.field_gradient_stages if stage.derivative == derivative
            )
            field_result = _run_sum_factor_iree_pipeline_function(
                module,
                evaluator,
                sum_factor_ir.function_prefix + "_field_gradient_d%d_linalg_pipeline" % derivative,
                field_stages,
                u_values,
                np,
            )
            max_abs_diff = max(max_abs_diff, field_result["max_abs_diff"])
            function_shapes[field_result["function"]] = field_result["result_shape"]
            function_count += 1

            q_values = [0.25 + 0.015625 * (derivative + 1) * (index + 1) for index in range(sum_factor_ir.n_qp)]
            test_stages = tuple(
                stage for stage in sum_factor_ir.test_gradient_stages if stage.derivative == derivative
            )
            test_input = evaluator._reorder_canonical_to_stage_input(q_values, test_stages[0])
            test_result = _run_sum_factor_iree_pipeline_function(
                module,
                evaluator,
                sum_factor_ir.function_prefix + "_test_gradient_d%d_linalg_pipeline" % derivative,
                test_stages,
                test_input,
                np,
            )
            max_abs_diff = max(max_abs_diff, test_result["max_abs_diff"])
            function_shapes[test_result["function"]] = test_result["result_shape"]
            function_count += 1
    except subprocess.CalledProcessError as exc:
        return {
            "name": "sum_factor_linalg_pipeline_metal_runtime",
            "ok": False,
            "skipped": False,
            "phase": "compile",
            "returncode": exc.returncode,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
        }
    except Exception as exc:
        return {
            "name": "sum_factor_linalg_pipeline_metal_runtime",
            "ok": False,
            "skipped": False,
            "phase": "runtime",
            "reason": str(exc),
        }
    return {
        "name": "sum_factor_linalg_pipeline_metal_runtime",
        "ok": compile_result.returncode == 0 and output_vmfb.exists() and max_abs_diff <= 1.0e-5,
        "skipped": False,
        "returncode": compile_result.returncode,
        "output_vmfb": str(output_vmfb),
        "stdout": compile_result.stdout,
        "stderr": compile_result.stderr,
        "driver": "metal",
        "function_count": function_count,
        "function_shapes": function_shapes,
        "max_abs_diff": max_abs_diff,
    }


def _run_sum_factor_iree_pipeline_function(module, evaluator, function_name, stages, input_values, np):
    function = getattr(module, function_name)
    arguments = [
        np.array(evaluator._stage_basis_matrix(stage), dtype=np.float32).reshape(stage.lhs_rows, stage.lhs_cols)
        for stage in stages
    ]
    arguments.append(
        np.array(input_values, dtype=np.float32).reshape(stages[0].rhs_rows, stages[0].rhs_cols)
    )
    result = np.asarray(function(*arguments))
    reference = np.array(
        evaluator.apply_pipeline(stages, tuple(float(value) for value in input_values)),
        dtype=np.float32,
    ).reshape(result.shape)
    return {
        "function": function_name,
        "max_abs_diff": float(np.max(np.abs(result - reference))) if result.size else 0.0,
        "result_shape": list(result.shape),
    }


def _run_iree_metal_runtime_smoke(output_dir, form_ir, form_linalg_lowering):
    try:
        import numpy as np
        import iree.runtime as iree_runtime
    except ImportError as exc:
        return {
            "name": "laplace_form_linalg_pipeline_metal_runtime",
            "ok": False,
            "skipped": True,
            "reason": "iree-runtime is not available: %s" % exc,
        }
    iree_compile = shutil.which("iree-compile")
    if iree_compile is None:
        return {
            "name": "laplace_form_linalg_pipeline_metal_runtime",
            "ok": False,
            "skipped": True,
            "reason": "iree-compile is not available",
        }
    if "metal" not in iree_runtime.query_available_drivers():
        return {
            "name": "laplace_form_linalg_pipeline_metal_runtime",
            "ok": False,
            "skipped": True,
            "reason": "IREE runtime metal driver is not available",
        }
    try:
        iree_runtime.get_driver("metal").create_default_device()
    except Exception as exc:
        return {
            "name": "laplace_form_linalg_pipeline_metal_runtime",
            "ok": False,
            "skipped": True,
            "no_default_device": True,
            "reason": str(exc),
        }
    try:
        output_vmfb, compile_result = form_linalg_lowering.compile_with_iree_metal(
            output_dir,
            iree_compile=iree_compile,
        )
        module = iree_runtime.load_vm_flatbuffer_file(str(output_vmfb), driver="metal")
        function = getattr(module, form_ir.function_prefix + "_linalg_pipeline")
        sf = form_ir.sum_factor
        shape = np.array(sf.shape_values_1d, dtype=np.float32).reshape(sf.n_qp_1d, sf.n_shape_1d)
        grad = np.array(sf.shape_gradients_1d, dtype=np.float32).reshape(sf.n_qp_1d, sf.n_shape_1d)
        weights = np.array(sf.weights_1d, dtype=np.float32)
        kappa = np.array([form_ir.parameter_default], dtype=np.float32)
        u_values = np.array(
            [0.5 + 0.03125 * (index + 1) for index in range(sf.n_shape)],
            dtype=np.float32,
        )
        u = u_values.reshape(sf.n_shape_1d, sf.n_shape // sf.n_shape_1d)
        result = np.asarray(
            function(
                shape,
                grad,
                shape.T.copy(),
                grad.T.copy(),
                weights,
                kappa,
                u,
            )
        )
        reference = np.array(
            TensorProductLaplaceReferenceEvaluator(form_ir).apply_local(tuple(float(value) for value in u_values)),
            dtype=np.float32,
        ).reshape(result.shape)
    except subprocess.CalledProcessError as exc:
        return {
            "name": "laplace_form_linalg_pipeline_metal_runtime",
            "ok": False,
            "skipped": False,
            "phase": "compile",
            "returncode": exc.returncode,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
        }
    except Exception as exc:
        return {
            "name": "laplace_form_linalg_pipeline_metal_runtime",
            "ok": False,
            "skipped": False,
            "phase": "runtime",
            "reason": str(exc),
        }
    max_abs_diff = float(np.max(np.abs(result - reference))) if result.size else 0.0
    return {
        "name": "laplace_form_linalg_pipeline_metal_runtime",
        "ok": compile_result.returncode == 0 and output_vmfb.exists() and max_abs_diff <= 1.0e-5,
        "skipped": False,
        "returncode": compile_result.returncode,
        "output_vmfb": str(output_vmfb),
        "stdout": compile_result.stdout,
        "stderr": compile_result.stderr,
        "driver": "metal",
        "max_abs_diff": max_abs_diff,
        "result_shape": list(result.shape),
    }


def _probe_metal_toolchain(output_dir):
    xcrun = shutil.which("xcrun")
    metal_files = sorted(Path(output_dir).rglob("*.metal"))
    if xcrun is None:
        return [
            {
                "name": "metal_toolchain_compile",
                "path": str(path),
                "ok": False,
                "skipped": True,
                "reason": "xcrun is not available",
            }
            for path in metal_files
        ]
    results = []
    air_dir = Path(output_dir) / "metal_toolchain"
    air_dir.mkdir(parents=True, exist_ok=True)
    for path in metal_files:
        air_path = air_dir / (path.stem + ".air")
        result = subprocess.run(
            [xcrun, "metal", "-c", str(path), "-o", str(air_path)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        results.append(
            {
                "name": "metal_toolchain_compile",
                "path": str(path),
                "ok": result.returncode == 0 and air_path.exists(),
                "skipped": False,
                "returncode": result.returncode,
                "output_air": str(air_path),
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
    return results


def _run_metal_smoke_tests(output_dir, sum_factor_metal, local_metal, ebe_metal):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        return [
            {
                "name": "metal",
                "ok": False,
                "skipped": True,
                "reason": "xcrun is not available",
            }
        ]
    results = []
    for name, lowering in (
        ("sum_factor_stages", sum_factor_metal),
        ("local_apply", local_metal),
        ("ebe_map_reduce", ebe_metal),
    ):
        try:
            result = lowering.run_metal_smoke_test(output_dir / "metal_smoke" / name, xcrun=xcrun)
        except ValueError as exc:
            results.append(
                {
                    "name": name,
                    "ok": False,
                    "skipped": True,
                    "reason": str(exc),
                }
            )
            continue
        results.append(
            {
                "name": name,
                "ok": result.success,
                "compiled": result.compiled,
                "no_default_device": result.no_default_device,
                "compile_returncode": result.compile_returncode,
                "run_returncode": result.run_returncode,
                "harness_path": str(result.harness_path),
                "executable_path": str(result.executable_path),
                "compile_stdout": result.compile_stdout,
                "compile_stderr": result.compile_stderr,
                "run_stdout": result.run_stdout,
                "run_stderr": result.run_stderr,
            }
        )
    return results


if __name__ == "__main__":
    raise SystemExit(main())
