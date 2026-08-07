#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess

from codegen.framework.materials import laplace as laplace_material_module
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
from codegen.framework.mlir.sum_factorization import (
    _c_float_literal,
    _float_initializer,
    _objc_string_literal,
)
from sfem import gen

material = laplace_material_module.material


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
    parser.add_argument(
        "--probe-mlir-gpu-to-spirv",
        action="store_true",
        help="try the standard MLIR GPU-to-SPIR-V pass pipeline on generated generic GPU dialect artifacts",
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
    parser.add_argument(
        "--require-pipeline-evidence",
        action="store_true",
        help="return nonzero when the summarized SFEM IR to IREE/Metal pipeline evidence is incomplete",
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
        "form_module": laplace_material_module.__name__,
        "form_source": _module_source_path(laplace_material_module),
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
        "iree_metal_executable_files": [],
        "iree_metal_executable_sources": [],
        "iree_metal_gpu": [],
        "iree_metal_matrix_unit": [],
        "iree_metal_runtime": [],
        "generated_sfem_comparison": [],
        "mlir_gpu_to_spirv": [],
        "mlir_validation": [],
        "metal_smoke": [],
        "metal_toolchain": [],
        "performance_shape": [],
        "pipeline_evidence": {},
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
        linalg_iree_probes = [
            ("sum_factor_linalg_pipeline_to_metal_vmfb", sum_factor),
            ("laplace_form_linalg_pipeline_to_metal_vmfb", form_linalg),
        ]
        manifest["iree_metal"] = _probe_iree_metal(output_dir / "iree_metal", linalg_iree_probes)
        manifest["iree_metal_executable_sources"] = _probe_iree_metal_executable_sources(
            output_dir / "iree_metal_executable_sources",
            linalg_iree_probes,
        )
        manifest["iree_metal_executable_files"] = _probe_iree_metal_executable_files(
            output_dir / "iree_metal_executable_files",
            linalg_iree_probes,
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

    if args.probe_mlir_gpu_to_spirv:
        manifest["mlir_gpu_to_spirv"] = _probe_mlir_gpu_to_spirv(
            output_dir / "mlir_gpu_to_spirv",
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
        manifest["metal_toolchain"] = _probe_metal_toolchain(
            output_dir,
            manifest["iree_metal_executable_files"],
        )

    if args.run_metal_smoke:
        local_metal = TensorProductLaplaceFormMetalLowering(form_ir)
        manifest["metal_smoke"] = _run_metal_smoke_tests(
            output_dir,
            sum_factor,
            local_metal,
            ebe_metal,
        )
        manifest["generated_sfem_comparison"] = [
            _run_generated_sfem_laplace_comparison(
                output_dir / "generated_sfem_comparison",
                args,
                form_ir,
                local_metal,
            )
        ]

    manifest["pipeline_evidence"] = _write_pipeline_evidence(output_dir, manifest)

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
    if args.require_pipeline_evidence:
        if not manifest["pipeline_evidence"].get("metal_pipeline_ok", False):
            return 13
    if any(not item.get("ok", False) for item in manifest["mlir_validation"]):
        return 2
    if any(not item.get("ok", False) and not item.get("skipped", False) for item in manifest["spirv_binary_validation"]):
        return 10
    if any(not item.get("ok", False) and not item.get("no_default_device", False) for item in manifest["metal_smoke"]):
        return 4
    if any(
        not item.get("ok", False) and not item.get("skipped", False)
        for item in manifest["generated_sfem_comparison"]
    ):
        return 14
    if any(not item.get("ok", False) for item in manifest["reference_verification"]):
        return 6
    if any(not item.get("ok", False) for item in manifest["performance_shape"]):
        return 7
    return 0


def _write_artifacts(artifacts):
    return [str(path) for path in artifacts.paths]


def _module_source_path(module):
    path = Path(getattr(module, "__file__", "")).resolve()
    try:
        return str(path.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


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


def _write_pipeline_evidence(output_dir, manifest):
    evidence = _build_pipeline_evidence(manifest)
    path = Path(output_dir) / "pipeline_evidence.json"
    evidence["path"] = str(path)
    path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    return evidence


def _build_pipeline_evidence(manifest):
    artifacts = manifest.get("artifacts", {})
    metal_toolchain = manifest.get("metal_toolchain", [])
    iree_backend_files = manifest.get("iree_metal_executable_files", [])
    form_source = manifest.get("form_source", "")
    metal_smoke_timings = [
        timing
        for result in manifest.get("metal_smoke", [])
        for timing in result.get("timings", [])
    ]
    generated_sfem_comparison = manifest.get("generated_sfem_comparison", [])
    iree_backend_source_counts = {}
    for item in metal_toolchain:
        source_kind = item.get("source_kind", "")
        if source_kind:
            iree_backend_source_counts[source_kind] = iree_backend_source_counts.get(source_kind, 0) + 1

    raw_gpu_diagnostics = sorted(
        {
            item.get("diagnostic_kind", "")
            for item in manifest.get("iree_metal_gpu", [])
            if item.get("diagnostic_kind")
        }
    )
    raw_gpu_spirv_diagnostics = sorted(
        {
            item.get("diagnostic_kind", "")
            for item in manifest.get("mlir_gpu_to_spirv", [])
            if item.get("diagnostic_kind")
        }
    )
    matrix_unit_diagnostics = sorted(
        {
            item.get("diagnostic_kind", "")
            for item in manifest.get("iree_metal_matrix_unit", [])
            if item.get("diagnostic_kind")
        }
    )

    evidence = {
        "name": "tensor_product_laplace_pipeline_evidence",
        "material": manifest.get("material"),
        "form_module": manifest.get("form_module"),
        "form_source": form_source,
        "form_source_ok": _form_source_matches_material(form_source, manifest.get("material")),
        "element": manifest.get("element"),
        "quadrature_order": manifest.get("quadrature_order"),
        "n_shape": manifest.get("n_shape"),
        "n_qp": manifest.get("n_qp"),
        "sfem_ir_files": len(artifacts.get("sfem_ir", [])),
        "sum_factor_artifacts": len(artifacts.get("sum_factor", [])),
        "form_linalg_artifacts": len(artifacts.get("form_linalg", [])),
        "gpu_artifact_groups": sorted(
            group
            for group in ("sum_factor", "form", "ebe_map", "ebe_full")
            if any(str(path).endswith(".gpu.mlir") for path in artifacts.get(group, []))
        ),
        "reference_ok": _all_required_ok(manifest.get("reference_verification", [])),
        "performance_shape_ok": _all_required_ok(manifest.get("performance_shape", [])),
        "mlir_validation_ok": _all_required_ok(manifest.get("mlir_validation", [])),
        "direct_spirv_validation_ok": _all_required_ok(manifest.get("spirv_binary_validation", [])),
        "iree_metal_vmfb_ok": _all_required_ok(manifest.get("iree_metal", [])),
        "iree_hal_executable_sources_ok": _all_required_ok(manifest.get("iree_metal_executable_sources", [])),
        "iree_backend_files_ok": _all_required_ok(iree_backend_files),
        "iree_backend_metal_source_count": sum(item.get("metal_source_count", 0) for item in iree_backend_files),
        "iree_backend_spirv_binary_count": sum(item.get("spirv_binary_count", 0) for item in iree_backend_files),
        "iree_backend_spirv_bytes": sum(item.get("spirv_binary_total_bytes", 0) for item in iree_backend_files),
        "iree_backend_spirv_deserialization_ok": all(
            all(spv.get("ok", False) or spv.get("skipped", False) for spv in item.get("spirv_deserialization", []))
            for item in iree_backend_files
        ),
        "metal_toolchain_ok": _all_required_ok(metal_toolchain),
        "metal_toolchain_source_counts": iree_backend_source_counts,
        "metal_smoke_ok": _all_required_ok(manifest.get("metal_smoke", []), allowed_skip_key="no_default_device"),
        "metal_smoke_timing_count": len(metal_smoke_timings),
        "generated_sfem_comparison_ok": _all_required_ok(generated_sfem_comparison)
        if generated_sfem_comparison
        else None,
        "iree_metal_runtime_ok": _all_required_ok(manifest.get("iree_metal_runtime", [])),
        "raw_gpu_iree_probe_ok": _all_required_ok(manifest.get("iree_metal_gpu", [])),
        "raw_gpu_iree_diagnostics": raw_gpu_diagnostics,
        "raw_gpu_spirv_pass_ok": _all_required_ok(manifest.get("mlir_gpu_to_spirv", [])),
        "raw_gpu_spirv_kernel_ok": _all_optional_key_ok(manifest.get("mlir_gpu_to_spirv", []), "kernel_ok"),
        "raw_gpu_spirv_kernel_module_ok": _all_optional_key_ok(
            manifest.get("mlir_gpu_to_spirv", []),
            "kernel_module_ok",
        ),
        "raw_gpu_spirv_kernel_binary_ok": _all_optional_key_ok(
            manifest.get("mlir_gpu_to_spirv", []),
            "kernel_binary_ok",
        ),
        "raw_gpu_spirv_host_launch_ok": _all_optional_key_ok(
            manifest.get("mlir_gpu_to_spirv", []),
            "host_launch_ok",
        ),
        "raw_gpu_spirv_host_wrapper_ok": _all_optional_key_ok(
            manifest.get("mlir_gpu_to_spirv", []),
            "host_wrapper_ok",
        ),
        "raw_gpu_spirv_diagnostics": raw_gpu_spirv_diagnostics,
        "matrix_unit_iree_probe_ok": _all_required_ok(manifest.get("iree_metal_matrix_unit", [])),
        "matrix_unit_iree_diagnostics": matrix_unit_diagnostics,
    }
    required_keys = (
        "reference_ok",
        "form_source_ok",
        "performance_shape_ok",
        "mlir_validation_ok",
        "direct_spirv_validation_ok",
        "iree_metal_vmfb_ok",
        "iree_hal_executable_sources_ok",
        "iree_backend_files_ok",
        "metal_toolchain_ok",
        "metal_smoke_ok",
        "iree_metal_runtime_ok",
    )
    evidence["metal_pipeline_ok"] = all(evidence[key] for key in required_keys)
    return evidence


def _form_source_matches_material(form_source, material_name):
    if not form_source or not material_name:
        return False
    path = Path(form_source)
    return path.name == "%s.py" % material_name and "materials" in path.parts


def _all_required_ok(items, allowed_skip_key=None):
    if not items:
        return False
    for item in items:
        if item.get("ok", False):
            continue
        if item.get("skipped", False):
            return False
        if allowed_skip_key is not None and item.get(allowed_skip_key, False):
            return False
        return False
    return True


def _all_optional_key_ok(items, key):
    return bool(items) and all(item.get(key, False) for item in items)


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
    lowering = payload.get("lowering")
    if lowering == "tensor_product_laplace_form_spirv_opencl":
        missing = []
        for key in ("kernel", "form", "global_work_items", "result_elements", "n_shape", "n_qp"):
            if key not in payload:
                missing.append(key)
        if payload.get("form") != "laplace":
            missing.append("form")
        if int(payload.get("global_work_items", -1)) != int(payload.get("result_elements", -2)):
            missing.append("global_work_items")
        if int(payload.get("global_work_items", -1)) != int(payload.get("n_shape", -2)):
            missing.append("n_shape")
        return {
            "name": "spirv_opencl_dispatch_metadata",
            "path": str(path),
            "ok": not missing,
            "lowering": lowering,
            "n_stages": 0,
            "kernel": payload.get("kernel", ""),
            "global_work_items": payload.get("global_work_items", 0),
            "missing_tokens": missing,
        }
    stages = payload.get("stages", [])
    missing = []
    if lowering != "tensor_product_sum_factor_spirv_opencl":
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
        "lowering": lowering,
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
    output_dir = Path(output_dir)
    files = sorted(
        path
        for path in output_dir.rglob("*.mlir")
        if _is_sfem_generated_mlir_artifact(output_dir, path)
    )
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


def _is_sfem_generated_mlir_artifact(output_dir, path):
    probe_dirs = {
        "iree_metal",
        "iree_metal_executable_files",
        "iree_metal_executable_sources",
        "iree_metal_gpu",
        "iree_metal_matrix_unit",
        "iree_metal_runtime",
    }
    try:
        parts = Path(path).relative_to(output_dir).parts
    except ValueError:
        return True
    return not any(part in probe_dirs for part in parts)


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


def _probe_iree_metal_executable_sources(output_dir, lowerings):
    iree_compile = shutil.which("iree-compile")
    if iree_compile is None:
        return [
            {
                "name": _iree_executable_source_probe_name(config),
                "ok": False,
                "skipped": True,
                "reason": "iree-compile is not available",
            }
            for config in lowerings
        ]
    output_dir = Path(output_dir)
    results = []
    for lowering_config in lowerings:
        name = _iree_executable_source_probe_name(lowering_config)
        lowering = lowering_config[1]
        probe_dir = output_dir / name
        probe_dir.mkdir(parents=True, exist_ok=True)
        input_mlir = probe_dir / ("%s.linalg_pipeline.mlir" % lowering.ir.function_prefix)
        output_mlir = probe_dir / ("%s.executable_sources.mlir" % lowering.ir.function_prefix)
        input_mlir.write_text(lowering.render_linalg_pipeline_module())
        command = [
            iree_compile,
            str(input_mlir),
            "--iree-hal-target-backends=metal-spirv",
            "--iree-metal-compile-to-metallib=false",
            "--compile-to=executable-sources",
            "-o",
            str(output_mlir),
        ]
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout_info = _write_probe_stream(probe_dir, "stdout", result.stdout or "")
        stderr_info = _write_probe_stream(probe_dir, "stderr", result.stderr or "")
        diagnostic = _classify_iree_probe_diagnostic(result.stderr or "")
        source_info = _verify_iree_metal_executable_source(output_mlir)
        ok = result.returncode == 0 and output_mlir.exists() and source_info["executable_source_ok"]
        results.append(
            {
                "name": name,
                "ok": ok,
                "skipped": False,
                "input_path": str(input_mlir),
                "output_mlir": str(output_mlir) if output_mlir.exists() else "",
                "command": command,
                "returncode": result.returncode,
                **diagnostic,
                **source_info,
                **stdout_info,
                **stderr_info,
            }
        )
    return results


def _probe_iree_metal_executable_files(output_dir, lowerings):
    iree_compile = shutil.which("iree-compile")
    if iree_compile is None:
        return [
            {
                "name": _iree_executable_file_probe_name(config),
                "ok": False,
                "skipped": True,
                "reason": "iree-compile is not available",
            }
            for config in lowerings
        ]
    output_dir = Path(output_dir)
    results = []
    for lowering_config in lowerings:
        name = _iree_executable_file_probe_name(lowering_config)
        lowering = lowering_config[1]
        probe_dir = output_dir / name
        dump_dir = probe_dir / "executable_files"
        probe_dir.mkdir(parents=True, exist_ok=True)
        dump_dir.mkdir(parents=True, exist_ok=True)
        input_mlir = probe_dir / ("%s.linalg_pipeline.mlir" % lowering.ir.function_prefix)
        output_vmfb = probe_dir / ("%s.executable_files.metal.vmfb" % lowering.ir.function_prefix)
        input_mlir.write_text(lowering.render_linalg_pipeline_module())
        command = [
            iree_compile,
            str(input_mlir),
            "--iree-hal-target-backends=metal-spirv",
            "--iree-metal-compile-to-metallib=false",
            "--iree-hal-dump-executable-files-to=%s" % dump_dir,
            "-o",
            str(output_vmfb),
        ]
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout_info = _write_probe_stream(probe_dir, "stdout", result.stdout or "")
        stderr_info = _write_probe_stream(probe_dir, "stderr", result.stderr or "")
        diagnostic = _classify_iree_probe_diagnostic(result.stderr or "")
        file_info = _verify_iree_metal_executable_files(dump_dir)
        ok = result.returncode == 0 and output_vmfb.exists() and file_info["executable_files_ok"]
        results.append(
            {
                "name": name,
                "ok": ok,
                "skipped": False,
                "input_path": str(input_mlir),
                "output_vmfb": str(output_vmfb) if output_vmfb.exists() else "",
                "dump_dir": str(dump_dir),
                "command": command,
                "returncode": result.returncode,
                **diagnostic,
                **file_info,
                **stdout_info,
                **stderr_info,
            }
        )
    return results


def _iree_executable_source_probe_name(config):
    return _iree_probe_name(config).replace("_to_metal_vmfb", "_to_metal_executable_sources")


def _iree_executable_file_probe_name(config):
    return _iree_probe_name(config).replace("_to_metal_vmfb", "_to_metal_executable_files")


def _verify_iree_metal_executable_source(path):
    path = Path(path)
    required_tokens = (
        "hal.executable",
        "hal.executable.variant",
        "hal.executable.export",
        "stream.cmd.dispatch",
        "flow.dispatch.tensor.load",
        "flow.dispatch.tensor.store",
    )
    if not path.exists():
        return {
            "executable_source_ok": False,
            "missing_tokens": list(required_tokens),
            "hal_executable_count": 0,
            "stream_dispatch_count": 0,
            "flow_tensor_load_count": 0,
            "flow_tensor_store_count": 0,
        }
    text = path.read_text()
    missing = [token for token in required_tokens if token not in text]
    return {
        "executable_source_ok": not missing,
        "missing_tokens": missing,
        "hal_executable_count": text.count("hal.executable private @"),
        "stream_dispatch_count": text.count("stream.cmd.dispatch"),
        "flow_tensor_load_count": text.count("flow.dispatch.tensor.load"),
        "flow_tensor_store_count": text.count("flow.dispatch.tensor.store"),
    }


def _verify_iree_metal_executable_files(dump_dir):
    dump_dir = Path(dump_dir)
    files = sorted(path for path in dump_dir.rglob("*") if path.is_file())
    metal_files = [path for path in files if path.suffix == ".metal"]
    spv_files = [path for path in files if path.suffix == ".spv"]
    mlir_files = [path for path in files if path.suffix == ".mlir"]
    configured_files = [path for path in mlir_files if path.name.startswith("configured_module_")]
    benchmark_files = [path for path in mlir_files if path.stem.endswith("_benchmark")]
    target_mlir_files = [
        path for path in mlir_files if "_metal_msl_fb" in path.stem and path not in benchmark_files
    ]
    missing = []
    if not configured_files:
        missing.append("configured_module_mlir")
    if not target_mlir_files:
        missing.append("metal_target_mlir")
    if not metal_files:
        missing.append("metal_source")
    if not spv_files:
        missing.append("spirv_binary")
    empty_spv_files = [path for path in spv_files if path.stat().st_size == 0]
    if empty_spv_files:
        missing.append("empty_spirv_binary")
    spirv_deserialization = _validate_iree_metal_spirv_binaries(spv_files)
    if any(not item.get("ok", False) and not item.get("skipped", False) for item in spirv_deserialization):
        missing.append("spirv_deserialization")
    metal_source_checks = _verify_iree_metal_source_files(metal_files)
    if not metal_source_checks["ok"]:
        missing.append("metal_source_tokens")
    spv_total_bytes = sum(path.stat().st_size for path in spv_files)
    return {
        "executable_files_ok": not missing,
        "missing_outputs": missing,
        "file_count": len(files),
        "configured_mlir_count": len(configured_files),
        "target_mlir_count": len(target_mlir_files),
        "benchmark_mlir_count": len(benchmark_files),
        "metal_source_count": len(metal_files),
        "spirv_binary_count": len(spv_files),
        "spirv_binary_total_bytes": spv_total_bytes,
        "empty_spirv_binary_paths": [str(path) for path in empty_spv_files],
        "spirv_deserialization": spirv_deserialization,
        "metal_source_paths": [str(path) for path in metal_files],
        "spirv_binary_paths": [str(path) for path in spv_files],
        "configured_mlir_paths": [str(path) for path in configured_files],
        "target_mlir_paths": [str(path) for path in target_mlir_files],
        "metal_source_checks": metal_source_checks,
    }


def _validate_iree_metal_spirv_binaries(spv_files):
    mlir_translate = shutil.which("mlir-translate") or "/opt/homebrew/opt/llvm/bin/mlir-translate"
    if not mlir_translate or not Path(mlir_translate).exists():
        return [
            {
                "path": str(path),
                "ok": False,
                "skipped": True,
                "reason": "mlir-translate is not available",
            }
            for path in spv_files
        ]
    expected_tokens = (
        "spirv.module Logical GLSL450",
        "spirv.GlobalVariable",
        "StorageBuffer",
        "spirv.func",
    )
    results = []
    for path in spv_files:
        result = subprocess.run(
            [mlir_translate, str(path), "--deserialize-spirv"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        text = result.stdout
        missing = [token for token in expected_tokens if token not in text]
        results.append(
            {
                "path": str(path),
                "ok": result.returncode == 0 and not missing,
                "skipped": False,
                "returncode": result.returncode,
                "missing_tokens": missing,
                "size": path.stat().st_size if path.exists() else 0,
                "stdout_preview": text[:1000],
                "stderr": result.stderr,
            }
        )
    return results


def _verify_iree_metal_source_files(metal_files):
    required_tokens = (
        "#include <metal_stdlib>",
        "using namespace metal",
        "kernel void",
        "device",
        "[[buffer(",
        "thread_position",
    )
    if not metal_files:
        return {
            "ok": False,
            "checked_files": 0,
            "missing_tokens": list(required_tokens),
            "missing_by_file": [],
            "kernel_count": 0,
            "fma_count": 0,
        }
    missing_by_file = []
    kernel_count = 0
    fma_count = 0
    for path in metal_files:
        text = path.read_text()
        kernel_count += text.count("kernel void")
        fma_count += text.count("fma(")
        missing_tokens = [token for token in required_tokens if token not in text]
        if missing_tokens:
            missing_by_file.append(
                {
                    "path": str(path),
                    "missing_tokens": missing_tokens,
                }
            )
    return {
        "ok": not missing_by_file,
        "checked_files": len(metal_files),
        "missing_tokens": sorted({token for item in missing_by_file for token in item["missing_tokens"]}),
        "missing_by_file": missing_by_file,
        "kernel_count": kernel_count,
        "fma_count": fma_count,
    }


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
        attempts = []
        for attempt_name, extra_flags in _iree_metal_gpu_probe_attempts():
            attempt_dir = probe_dir / attempt_name
            attempt_dir.mkdir(parents=True, exist_ok=True)
            output_vmfb = attempt_dir / (Path(path).stem + ".metal.vmfb")
            command = [
                iree_compile,
                str(path),
                "--iree-hal-target-backends=metal-spirv",
                "--iree-metal-compile-to-metallib=false",
                *extra_flags,
                "-o",
                str(output_vmfb),
            ]
            result = subprocess.run(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout_info = _write_probe_stream(attempt_dir, "stdout", result.stdout or "")
            stderr_info = _write_probe_stream(attempt_dir, "stderr", result.stderr or "")
            diagnostic = _classify_iree_probe_diagnostic(result.stderr or "")
            ok = result.returncode == 0 and output_vmfb.exists()
            attempts.append(
                {
                    "name": attempt_name,
                    "ok": ok,
                    "returncode": result.returncode,
                    "command": command,
                    "extra_flags": list(extra_flags),
                    "output_vmfb": str(output_vmfb) if output_vmfb.exists() else "",
                    **diagnostic,
                    **stdout_info,
                    **stderr_info,
                }
            )
            if ok:
                break
        selected = next((attempt for attempt in attempts if attempt["ok"]), attempts[0])
        results.append(
            {
                "name": name,
                "ok": selected["ok"],
                "skipped": False,
                "artifact_group": group,
                "input_path": str(path),
                "attempt_count": len(attempts),
                "attempts": attempts,
                "command": selected["command"],
                "extra_flags": selected["extra_flags"],
                "returncode": selected["returncode"],
                "output_vmfb": selected["output_vmfb"],
                "diagnostic_kind": selected["diagnostic_kind"],
                "diagnostic_summary": selected["diagnostic_summary"],
                "stdout_path": selected["stdout_path"],
                "stdout_bytes": selected["stdout_bytes"],
                "stdout_preview": selected["stdout_preview"],
                "stdout_truncated": selected["stdout_truncated"],
                "stderr_path": selected["stderr_path"],
                "stderr_bytes": selected["stderr_bytes"],
                "stderr_preview": selected["stderr_preview"],
                "stderr_truncated": selected["stderr_truncated"],
            }
        )
    return results


def _probe_mlir_gpu_to_spirv(output_dir, gpu_artifacts):
    mlir_opt = shutil.which("mlir-opt") or "/opt/homebrew/opt/llvm/bin/mlir-opt"
    mlir_translate = shutil.which("mlir-translate") or "/opt/homebrew/opt/llvm/bin/mlir-translate"
    if not mlir_opt or not Path(mlir_opt).exists():
        return [
            {
                "name": _mlir_gpu_to_spirv_probe_name(group, path),
                "ok": False,
                "skipped": True,
                "artifact_group": group,
                "input_path": str(path),
                "reason": "mlir-opt is not available",
            }
            for group, path in gpu_artifacts
        ]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    pass_pipeline = (
        "builtin.module("
        "spirv-attach-target{client_api=OpenCL},"
        "map-memref-spirv-storage-class{client-api=opencl},"
        "convert-to-spirv{convert-gpu-modules}"
        ")"
    )
    for group, path in gpu_artifacts:
        name = _mlir_gpu_to_spirv_probe_name(group, path)
        probe_dir = output_dir / name
        probe_dir.mkdir(parents=True, exist_ok=True)
        output_mlir = probe_dir / ("%s.gpu_to_spirv.mlir" % Path(path).stem)
        command = [mlir_opt, str(path), "--pass-pipeline=%s" % pass_pipeline, "--verify-diagnostics"]
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.stdout:
            output_mlir.write_text(result.stdout)
        stdout_info = _write_probe_stream(probe_dir, "stdout", result.stdout or "")
        stderr_info = _write_probe_stream(probe_dir, "stderr", result.stderr or "")
        output_text = result.stdout or ""
        diagnostic = _classify_mlir_gpu_to_spirv_diagnostic(result.stderr or "", output_text)
        remaining_gpu_launch_count = output_text.count("gpu.launch_func")
        remaining_gpu_module_count = output_text.count("gpu.module @")
        host_func_count = output_text.count("func.func @")
        host_wrapper_ok = result.returncode == 0 and host_func_count > 0 and remaining_gpu_launch_count > 0
        memref_load_count = output_text.count("memref.load")
        memref_store_count = output_text.count("memref.store")
        kernel_module_info = _probe_extracted_spirv_kernel_modules(
            probe_dir,
            output_text,
            mlir_opt=mlir_opt,
            mlir_translate=mlir_translate,
        )
        kernel_ok = (
            result.returncode == 0
            and "spirv.module" in output_text
            and output_text.count("spirv.func") > 0
            and kernel_module_info.get("kernel_module_ok", False)
        )
        host_launch_ok = remaining_gpu_launch_count == 0 and remaining_gpu_module_count == 0
        results.append(
            {
                "name": name,
                "ok": kernel_ok and host_launch_ok,
                "kernel_ok": kernel_ok,
                "host_launch_ok": host_launch_ok,
                "host_wrapper_ok": host_wrapper_ok,
                "skipped": False,
                "artifact_group": group,
                "input_path": str(path),
                "output_mlir": str(output_mlir) if output_mlir.exists() else "",
                "command": command,
                "returncode": result.returncode,
                "spirv_module_count": output_text.count("spirv.module"),
                "spirv_func_count": output_text.count("spirv.func"),
                "entry_point_count": output_text.count('spirv.EntryPoint "GLCompute"')
                + output_text.count('spirv.EntryPoint "Kernel"'),
                **kernel_module_info,
                "remaining_gpu_launch_count": remaining_gpu_launch_count,
                "remaining_gpu_module_count": remaining_gpu_module_count,
                "remaining_host_func_count": host_func_count,
                "remaining_memref_load_count": memref_load_count,
                "remaining_memref_store_count": memref_store_count,
                **diagnostic,
                **stdout_info,
                **stderr_info,
            }
        )
    return results


def _probe_extracted_spirv_kernel_modules(output_dir, output_text, *, mlir_opt, mlir_translate):
    modules = _extract_nested_spirv_modules(output_text)
    if not modules:
        return {
            "kernel_module_ok": False,
            "kernel_module_count": 0,
            "kernel_module_paths": [],
            "kernel_module_verification": [],
            "kernel_binary_ok": False,
            "kernel_binary_count": 0,
            "kernel_binary_total_bytes": 0,
            "kernel_binary_paths": [],
            "kernel_binary_serialization": [],
        }
    output_dir = Path(output_dir)
    verification = []
    serialization = []
    binary_paths = []
    module_paths = []
    for index, module_text in enumerate(modules):
        module_path = output_dir / ("kernel_%d.spirv.mlir" % index)
        binary_path = output_dir / ("kernel_%d.spv" % index)
        module_path.write_text(module_text)
        module_paths.append(str(module_path))
        verify_result = subprocess.run(
            [mlir_opt, str(module_path), "--verify-diagnostics"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        verification.append(
            {
                "path": str(module_path),
                "ok": verify_result.returncode == 0,
                "returncode": verify_result.returncode,
                "stdout_preview": verify_result.stdout[:1000],
                "stderr_preview": verify_result.stderr[:1000],
            }
        )
        if not mlir_translate or not Path(mlir_translate).exists():
            serialization.append(
                {
                    "path": str(module_path),
                    "output_path": str(binary_path),
                    "ok": False,
                    "skipped": True,
                    "reason": "mlir-translate is not available",
                }
            )
            continue
        serialize_result = subprocess.run(
            [
                mlir_translate,
                str(module_path),
                "--no-implicit-module",
                "--serialize-spirv",
                "-o",
                str(binary_path),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        binary_size = binary_path.stat().st_size if binary_path.exists() else 0
        binary_ok = serialize_result.returncode == 0 and binary_size > 0
        if binary_ok:
            binary_paths.append(str(binary_path))
        serialization.append(
            {
                "path": str(module_path),
                "output_path": str(binary_path),
                "ok": binary_ok,
                "skipped": False,
                "returncode": serialize_result.returncode,
                "size": binary_size,
                **_classify_spirv_serialization_diagnostic(serialize_result.stderr or ""),
                "stdout_preview": serialize_result.stdout[:1000],
                "stderr_preview": serialize_result.stderr[:1000],
            }
        )
    binary_total_bytes = sum(Path(path).stat().st_size for path in binary_paths)
    return {
        "kernel_module_ok": bool(verification) and all(item["ok"] for item in verification),
        "kernel_module_count": len(modules),
        "kernel_module_paths": module_paths,
        "kernel_module_verification": verification,
        "kernel_binary_ok": bool(serialization) and all(item["ok"] for item in serialization),
        "kernel_binary_count": len(binary_paths),
        "kernel_binary_total_bytes": binary_total_bytes,
        "kernel_binary_paths": binary_paths,
        "kernel_binary_serialization": serialization,
    }


def _extract_nested_spirv_modules(module_text):
    modules = []
    search_from = 0
    needle = "spirv.module"
    while True:
        start = module_text.find(needle, search_from)
        if start < 0:
            break
        if start > 0 and (module_text[start - 1].isalnum() or module_text[start - 1] in "._"):
            search_from = start + len(needle)
            continue
        brace = module_text.find("{", start)
        if brace < 0:
            break
        end = _find_balanced_brace_end(module_text, brace)
        if end < 0:
            break
        module_op = module_text[start:end].rstrip() + "\n"
        modules.append(_normalize_extracted_spirv_module(module_op))
        search_from = end
    return modules


def _find_balanced_brace_end(text, open_brace):
    depth = 0
    in_string = False
    escaped = False
    for index in range(open_brace, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index + 1
    return -1


def _normalize_extracted_spirv_module(module_op):
    lines = module_op.splitlines()
    if not lines:
        return module_op
    header = lines[0].lstrip()
    header = re.sub(r"^spirv\.module\s+@[^ ]+\s+", "spirv.module ", header, count=1)
    if " requires #spirv.vce<" not in header:
        header = _attach_spirv_vce_triple(header)
    return "\n".join([header, *lines[1:]]) + "\n"


def _attach_spirv_vce_triple(header):
    if header.startswith("spirv.module Logical GLSL450"):
        return header.replace(
            "spirv.module Logical GLSL450",
            "spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>",
            1,
        )
    if header.startswith("spirv.module Logical OpenCL"):
        return header.replace(
            "spirv.module Logical OpenCL",
            "spirv.module Logical OpenCL requires #spirv.vce<v1.0, [Kernel, Addresses], []>",
            1,
        )
    return header


def _classify_spirv_serialization_diagnostic(stderr):
    if "module must have 'vce_triple' attribute" in stderr:
        return {
            "diagnostic_kind": "spirv_serialization_missing_vce_triple",
            "diagnostic_summary": "SPIR-V module lacks the vce_triple metadata required for binary serialization",
        }
    if "missing 'spirv.target_env' attribute" in stderr or "missing SPIR-V target env attribute" in stderr:
        return {
            "diagnostic_kind": "spirv_serialization_missing_target_env",
            "diagnostic_summary": "SPIR-V module lacks target environment metadata required for ABI/VCE lowering",
        }
    if "expected a 'spirv.module' op" in stderr:
        return {
            "diagnostic_kind": "spirv_serialization_module_boundary",
            "diagnostic_summary": "mlir-translate did not receive a standalone top-level spirv.module operation",
        }
    return {
        "diagnostic_kind": "",
        "diagnostic_summary": stderr.splitlines()[0] if stderr.splitlines() else "",
    }


def _mlir_gpu_to_spirv_probe_name(group, path):
    return "%s_%s_gpu_to_spirv" % (group, Path(path).stem.replace(".", "_"))


def _classify_mlir_gpu_to_spirv_diagnostic(stderr, output_text=""):
    if "failed to legalize operation 'memref.load'" in stderr:
        return {
            "diagnostic_kind": "mlir_gpu_to_spirv_memref_load_legalization",
            "diagnostic_summary": "standard GPU-to-SPIR-V pass pipeline did not legalize memref.load",
        }
    if "failed to legalize operation 'memref.store'" in stderr:
        return {
            "diagnostic_kind": "mlir_gpu_to_spirv_memref_store_legalization",
            "diagnostic_summary": "standard GPU-to-SPIR-V pass pipeline did not legalize memref.store",
        }
    if "failed to legalize operation" in stderr:
        match = re.search(r"failed to legalize operation '([^']+)'", stderr)
        operation = match.group(1) if match else "unknown"
        return {
            "diagnostic_kind": "mlir_gpu_to_spirv_operation_legalization",
            "diagnostic_summary": "standard GPU-to-SPIR-V pass pipeline did not legalize %s" % operation,
        }
    if output_text and "spirv.module" in output_text and (
        "gpu.launch_func" in output_text or "gpu.module @" in output_text
    ):
        return {
            "diagnostic_kind": "mlir_gpu_to_spirv_host_launch_boundary",
            "diagnostic_summary": "GPU kernels lowered to SPIR-V but host gpu.launch_func/gpu.module remains",
        }
    return {
        "diagnostic_kind": "",
        "diagnostic_summary": stderr.splitlines()[0] if stderr.splitlines() else "",
    }


def _iree_metal_gpu_probe_attempts():
    return (
        ("baseline", ()),
        ("vm_index_32", ("--iree-vm-target-index-bits=32",)),
        ("spirv_index_32", ("--iree-spirv-index-bits=32",)),
        ("demote_i64_to_i32", ("--iree-input-demote-i64-to-i32",)),
        ("input_type_none", ("--iree-input-type=none",)),
    )


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


def _probe_metal_toolchain(output_dir, iree_metal_executable_files=()):
    xcrun = shutil.which("xcrun")
    output_dir = Path(output_dir)
    metal_files = _metal_toolchain_source_paths(output_dir, iree_metal_executable_files)
    if xcrun is None:
        return [
            {
                "name": "metal_toolchain_compile",
                "path": str(path),
                **_metal_toolchain_source_info(output_dir, path),
                "ok": False,
                "skipped": True,
                "reason": "xcrun is not available",
            }
            for path in metal_files
        ]
    results = []
    air_dir = output_dir / "metal_toolchain"
    air_dir.mkdir(parents=True, exist_ok=True)
    for path in metal_files:
        air_path = _metal_toolchain_air_path(output_dir, air_dir, path)
        air_path.parent.mkdir(parents=True, exist_ok=True)
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
                **_metal_toolchain_source_info(output_dir, path),
                "ok": result.returncode == 0 and air_path.exists(),
                "skipped": False,
                "returncode": result.returncode,
                "output_air": str(air_path),
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
    return results


def _metal_toolchain_source_paths(output_dir, iree_metal_executable_files=()):
    output_dir = Path(output_dir)
    paths = {Path(path) for path in output_dir.rglob("*.metal")}
    for item in iree_metal_executable_files or ():
        for path in item.get("metal_source_paths", ()):
            path = Path(path)
            if path.suffix == ".metal":
                paths.add(path)
    return sorted(paths, key=lambda path: str(path))


def _metal_toolchain_air_path(output_dir, air_dir, path):
    try:
        relative = Path(path).relative_to(output_dir)
    except ValueError:
        relative = Path(path.name)
    return Path(air_dir) / relative.with_suffix(".air")


def _metal_toolchain_source_info(output_dir, path):
    try:
        relative = Path(path).relative_to(output_dir)
    except ValueError:
        return {
            "source_kind": "external_metal_source",
            "source_group": "",
        }
    parts = relative.parts
    if parts and parts[0] == "iree_metal_executable_files":
        return {
            "source_kind": "iree_metal_executable_file",
            "source_group": parts[1] if len(parts) > 1 else "",
        }
    return {
        "source_kind": "sfem_generated_metal",
        "source_group": parts[0] if parts else "",
    }


_METAL_TIMING_RE = re.compile(
    r"^metal_timing name=(?P<name>\S+) iterations=(?P<iterations>\d+) "
    r"total_gpu_us=(?P<total_gpu_us>[0-9.+\-eE]+) avg_gpu_us=(?P<avg_gpu_us>[0-9.+\-eE]+)$"
)
_GENERATED_SFEM_TIMING_RE = re.compile(
    r"^generated_sfem_timing name=(?P<name>\S+) iterations=(?P<iterations>\d+) "
    r"total_cpu_us=(?P<total_cpu_us>[0-9.+\-eE]+) avg_cpu_us=(?P<avg_cpu_us>[0-9.+\-eE]+)$"
)
_GENERATED_SFEM_COMPARE_RE = re.compile(
    r"^generated_sfem_compare target=(?P<target>\S+) output_max_abs_diff=(?P<max_abs_diff>[0-9.+\-eE]+) "
    r"tolerance=(?P<tolerance>[0-9.+\-eE]+) ok=(?P<ok>[01])$"
)


def _parse_metal_timing_stdout(stdout):
    timings = []
    for line in stdout.splitlines():
        match = _METAL_TIMING_RE.match(line.strip())
        if match is None:
            continue
        timings.append(
            {
                "name": match.group("name"),
                "iterations": int(match.group("iterations")),
                "total_gpu_us": float(match.group("total_gpu_us")),
                "avg_gpu_us": float(match.group("avg_gpu_us")),
            }
        )
    return timings


def _parse_generated_sfem_timing_stdout(stdout):
    timings = []
    for line in stdout.splitlines():
        match = _GENERATED_SFEM_TIMING_RE.match(line.strip())
        if match is None:
            continue
        timings.append(
            {
                "name": match.group("name"),
                "iterations": int(match.group("iterations")),
                "total_cpu_us": float(match.group("total_cpu_us")),
                "avg_cpu_us": float(match.group("avg_cpu_us")),
            }
        )
    return timings


def _parse_generated_sfem_compare_stdout(stdout):
    for line in stdout.splitlines():
        match = _GENERATED_SFEM_COMPARE_RE.match(line.strip())
        if match is None:
            continue
        return {
            "target": match.group("target"),
            "max_abs_diff": float(match.group("max_abs_diff")),
            "tolerance": float(match.group("tolerance")),
            "ok": match.group("ok") == "1",
        }
    return {}


def _run_generated_sfem_laplace_comparison(output_dir, args, form_ir, local_metal):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        return {
            "name": "generated_sfem_laplace_local_apply_comparison",
            "ok": False,
            "skipped": True,
            "reason": "xcrun is not available",
        }
    sf = form_ir.sum_factor
    if sf.dim != 3:
        return {
            "name": "generated_sfem_laplace_local_apply_comparison",
            "ok": False,
            "skipped": True,
            "reason": "generated SFEM comparison harness currently covers 3D tensor-product elements",
        }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_dir = output_dir / "generated_laplace"
    try:
        gen.generate(
            material,
            generated_dir,
            elements=(sf.element_type,),
            vector_size=sf.vector_size,
            quadrature_order=sf.quadrature_order,
            clean=True,
            dump_plan=True,
            target="openmp",
        )
    except Exception as exc:
        return {
            "name": "generated_sfem_laplace_local_apply_comparison",
            "ok": False,
            "phase": "generate",
            "reason": str(exc),
            "generated_dir": str(generated_dir),
        }

    element_slug = sf.element_type.lower()
    generated_operator = generated_dir / "d3" / element_slug / ("laplace_%s_operator.cpp" % element_slug)
    if not generated_operator.is_file():
        return {
            "name": "generated_sfem_laplace_local_apply_comparison",
            "ok": False,
            "phase": "generate",
            "reason": "generated SFEM operator source was not emitted",
            "generated_dir": str(generated_dir),
            "generated_operator": str(generated_operator),
        }

    harness = output_dir / ("%s_generated_sfem_compare.mm" % form_ir.function_prefix)
    executable = output_dir / ("%s_generated_sfem_compare" % form_ir.function_prefix)
    harness.write_text(_render_generated_sfem_laplace_comparison_harness(form_ir, local_metal, generated_operator))
    compile_result = subprocess.run(
        [
            xcrun,
            "clang++",
            "-std=c++17",
            "-O3",
            str(harness),
            "-fobjc-arc",
            "-framework",
            "Foundation",
            "-framework",
            "Metal",
            "-o",
            str(executable),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if compile_result.returncode != 0:
        return {
            "name": "generated_sfem_laplace_local_apply_comparison",
            "ok": False,
            "phase": "compile",
            "generated_dir": str(generated_dir),
            "generated_operator": str(generated_operator),
            "harness_path": str(harness),
            "executable_path": str(executable),
            "compile_returncode": compile_result.returncode,
            "compile_stdout": compile_result.stdout,
            "compile_stderr": compile_result.stderr,
        }

    run_result = subprocess.run(
        [str(executable)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    comparison = _parse_generated_sfem_compare_stdout(run_result.stdout)
    return {
        "name": "generated_sfem_laplace_local_apply_comparison",
        "ok": run_result.returncode == 0 and comparison.get("ok", False),
        "phase": "runtime",
        "generated_dir": str(generated_dir),
        "generated_operator": str(generated_operator),
        "harness_path": str(harness),
        "executable_path": str(executable),
        "compile_returncode": compile_result.returncode,
        "run_returncode": run_result.returncode,
        "compile_stdout": compile_result.stdout,
        "compile_stderr": compile_result.stderr,
        "run_stdout": run_result.stdout,
        "run_stderr": run_result.stderr,
        "comparison": comparison,
        "metal_timings": _parse_metal_timing_stdout(run_result.stdout),
        "generated_sfem_timings": _parse_generated_sfem_timing_stdout(run_result.stdout),
    }


def _render_generated_sfem_laplace_comparison_harness(form_ir, local_metal, generated_operator):
    sf = form_ir.sum_factor
    element_slug = sf.element_type.lower()
    generated_kernel = "laplace_%s_residual_element_soa_float" % element_slug
    u = tuple(float(0.5 + 0.03125 * (i + 1)) for i in range(sf.n_shape))
    return _GENERATED_SFEM_LAPLACE_COMPARISON_TEMPLATE % {
        "generated_operator": str(generated_operator),
        "source": _objc_string_literal(local_metal.render_metal_source()),
        "metal_kernel_name": local_metal._metal_kernel_name(),
        "generated_kernel": generated_kernel,
        "n_shape": sf.n_shape,
        "n_qp": sf.n_qp,
        "u": _float_initializer(u),
        "kappa": _c_float_literal(form_ir.parameter_default),
        "metal_iterations": 64,
        "generated_iterations": 4096,
        "tolerance": _c_float_literal(5.0e-4),
    }


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
                "timings": _parse_metal_timing_stdout(result.run_stdout),
            }
        )
    return results


_GENERATED_SFEM_LAPLACE_COMPARISON_TEMPLATE = r'''#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "%(generated_operator)s"

static constexpr unsigned N_SHAPE = %(n_shape)d;
static constexpr unsigned N_QP = %(n_qp)d;
static constexpr unsigned METAL_TIMING_ITERATIONS = %(metal_iterations)d;
static constexpr unsigned GENERATED_SFEM_TIMING_ITERATIONS = %(generated_iterations)d;
static constexpr float COMPARISON_TOLERANCE = %(tolerance)sf;

struct GeneratedSfemFixture {
    float determinant[N_QP];
    float adjugate_storage[9][N_QP];
    const float *adjugate[9];
    float current_storage[N_SHAPE];
    float output_storage[N_SHAPE];
    const float *current[N_SHAPE];
    float *output[N_SHAPE];
};

static void initialize_generated_sfem_fixture(GeneratedSfemFixture &fixture) {
    static const float input[N_SHAPE] = {%(u)s};
    for (unsigned q = 0; q < N_QP; ++q) {
        fixture.determinant[q] = 1.0f;
        for (unsigned j = 0; j < 9; ++j) {
            fixture.adjugate_storage[j][q] = 0.0f;
        }
        fixture.adjugate_storage[0][q] = 1.0f;
        fixture.adjugate_storage[4][q] = 1.0f;
        fixture.adjugate_storage[8][q] = 1.0f;
    }
    for (unsigned j = 0; j < 9; ++j) {
        fixture.adjugate[j] = fixture.adjugate_storage[j];
    }
    for (unsigned i = 0; i < N_SHAPE; ++i) {
        fixture.current_storage[i] = input[i];
        fixture.output_storage[i] = 0.0f;
        fixture.current[i] = &fixture.current_storage[i];
        fixture.output[i] = &fixture.output_storage[i];
    }
}

static int run_generated_sfem_kernel(GeneratedSfemFixture &fixture, float *out) {
    for (unsigned i = 0; i < N_SHAPE; ++i) {
        fixture.output_storage[i] = 0.0f;
    }
    const int status = %(generated_kernel)s(
            1,
            1,
            fixture.determinant,
            fixture.adjugate,
            fixture.current,
            %(kappa)sf,
            fixture.output);
    if (status != 0) {
        return status;
    }
    for (unsigned i = 0; i < N_SHAPE; ++i) {
        out[i] = fixture.output_storage[i];
    }
    return 0;
}

static int run_metal_kernel(float *out, double *total_gpu_us, double *avg_gpu_us) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            return 77;
        }

        NSString *source = %(source)s;
        NSError *error = nil;
        MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
        id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
        if (!library) {
            std::fprintf(stderr, "Metal library compilation failed: %%s\n", [[error localizedDescription] UTF8String]);
            return 78;
        }

        id<MTLFunction> function = [library newFunctionWithName:@"%(metal_kernel_name)s"];
        if (!function) {
            std::fprintf(stderr, "Metal function lookup failed\n");
            return 79;
        }

        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) {
            std::fprintf(stderr, "Metal pipeline creation failed: %%s\n", [[error localizedDescription] UTF8String]);
            return 80;
        }

        static const float input[N_SHAPE] = {%(u)s};
        const float kappa = %(kappa)sf;
        id<MTLBuffer> u_buffer = [device newBufferWithBytes:input length:sizeof(input) options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_buffer = [device newBufferWithLength:N_SHAPE * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> kappa_buffer = [device newBufferWithBytes:&kappa length:sizeof(kappa) options:MTLResourceStorageModeShared];
        id<MTLCommandQueue> queue = [device newCommandQueue];

        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:u_buffer offset:0 atIndex:0];
        [encoder setBuffer:out_buffer offset:0 atIndex:1];
        [encoder setBuffer:kappa_buffer offset:0 atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(N_SHAPE, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(N_SHAPE, 1, 1)];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if ([command_buffer status] != MTLCommandBufferStatusCompleted) {
            std::fprintf(stderr, "Metal command failed\n");
            return 81;
        }
        const float *result = static_cast<const float *>([out_buffer contents]);
        for (unsigned i = 0; i < N_SHAPE; ++i) {
            out[i] = result[i];
        }

        double total_gpu_seconds = 0.0;
        for (unsigned iter = 0; iter < METAL_TIMING_ITERATIONS; ++iter) {
            id<MTLCommandBuffer> timing_command_buffer = [queue commandBuffer];
            id<MTLComputeCommandEncoder> timing_encoder = [timing_command_buffer computeCommandEncoder];
            [timing_encoder setComputePipelineState:pipeline];
            [timing_encoder setBuffer:u_buffer offset:0 atIndex:0];
            [timing_encoder setBuffer:out_buffer offset:0 atIndex:1];
            [timing_encoder setBuffer:kappa_buffer offset:0 atIndex:2];
            [timing_encoder dispatchThreads:MTLSizeMake(N_SHAPE, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(N_SHAPE, 1, 1)];
            [timing_encoder endEncoding];
            [timing_command_buffer commit];
            [timing_command_buffer waitUntilCompleted];
            if ([timing_command_buffer status] != MTLCommandBufferStatusCompleted) {
                std::fprintf(stderr, "Metal timing command failed\n");
                return 82;
            }
            const double gpu_seconds = [timing_command_buffer GPUEndTime] - [timing_command_buffer GPUStartTime];
            if (gpu_seconds > 0.0) {
                total_gpu_seconds += gpu_seconds;
            }
        }
        *total_gpu_us = total_gpu_seconds * 1.0e6;
        *avg_gpu_us = *total_gpu_us / static_cast<double>(METAL_TIMING_ITERATIONS);
        return 0;
    }
}

int main() {
    GeneratedSfemFixture fixture;
    initialize_generated_sfem_fixture(fixture);

    float generated_out[N_SHAPE];
    float metal_out[N_SHAPE];
    double total_gpu_us = 0.0;
    double avg_gpu_us = 0.0;

    const int generated_status = run_generated_sfem_kernel(fixture, generated_out);
    if (generated_status != 0) {
        std::fprintf(stderr, "generated SFEM kernel failed: %%d\n", generated_status);
        return 90;
    }

    const int metal_status = run_metal_kernel(metal_out, &total_gpu_us, &avg_gpu_us);
    if (metal_status != 0) {
        return metal_status;
    }

    float max_abs_diff = 0.0f;
    for (unsigned i = 0; i < N_SHAPE; ++i) {
        const float diff = std::fabs(metal_out[i] - generated_out[i]);
        if (diff > max_abs_diff) {
            max_abs_diff = diff;
        }
    }
    const bool outputs_match = max_abs_diff <= COMPARISON_TOLERANCE;
    std::printf("generated_sfem_compare target=%(generated_kernel)s output_max_abs_diff=%%.9g tolerance=%%.9g ok=%%d\n",
                static_cast<double>(max_abs_diff),
                static_cast<double>(COMPARISON_TOLERANCE),
                outputs_match ? 1 : 0);
    std::printf("metal_timing name=%(metal_kernel_name)s iterations=%%u total_gpu_us=%%.6f avg_gpu_us=%%.6f\n",
                METAL_TIMING_ITERATIONS, total_gpu_us, avg_gpu_us);

    volatile float sink = 0.0f;
    const auto cpu_start = std::chrono::steady_clock::now();
    for (unsigned iter = 0; iter < GENERATED_SFEM_TIMING_ITERATIONS; ++iter) {
        const int status = run_generated_sfem_kernel(fixture, generated_out);
        if (status != 0) {
            return 91;
        }
        sink += generated_out[iter %% N_SHAPE];
    }
    const auto cpu_end = std::chrono::steady_clock::now();
    const double total_cpu_us =
            std::chrono::duration<double, std::micro>(cpu_end - cpu_start).count();
    const double avg_cpu_us = total_cpu_us / static_cast<double>(GENERATED_SFEM_TIMING_ITERATIONS);
    std::printf("generated_sfem_timing name=%(generated_kernel)s iterations=%%u total_cpu_us=%%.6f avg_cpu_us=%%.6f\n",
                GENERATED_SFEM_TIMING_ITERATIONS, total_cpu_us, avg_cpu_us);
    return outputs_match ? 0 : 92;
}
'''


if __name__ == "__main__":
    raise SystemExit(main())
