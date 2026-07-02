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
    TensorProductLaplaceFormMetalLowering,
    TensorProductLaplaceReferenceEvaluator,
    TensorProductSumFactorReferenceEvaluator,
    TensorProductSumFactorMLIRLowering,
    tensor_product_laplace_form_ir_from_material,
    tensor_product_sum_factor_ir_from_material,
)


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
    parser.add_argument("--probe-iree-metal", action="store_true", help="try iree-compile --iree-hal-target-backends=metal")
    parser.add_argument("--probe-metal-toolchain", action="store_true", help="try offline xcrun metal compilation for generated .metal files")
    parser.add_argument("--run-metal-smoke", action="store_true", help="compile and run local and EBE Metal smoke tests")
    parser.add_argument(
        "--require-iree-metal",
        action="store_true",
        help="return nonzero when the IREE Metal probe is skipped or fails",
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

    sum_factor_ir = tensor_product_sum_factor_ir_from_material(
        material,
        element=args.element,
        vector_size=args.vector_size,
        quadrature_order=args.quadrature_order,
    )
    form_ir = tensor_product_laplace_form_ir_from_material(
        material,
        element=args.element,
        vector_size=args.vector_size,
        quadrature_order=args.quadrature_order,
    )

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
        "mlir_validation": [],
        "metal_smoke": [],
        "metal_toolchain": [],
        "performance_shape": [],
        "reference_verification": [],
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

    if args.probe_iree_metal:
        manifest["iree_metal"] = [
            _probe_iree_metal(output_dir / "iree_metal", sum_factor)
        ]

    if args.probe_metal_toolchain:
        manifest["metal_toolchain"] = _probe_metal_toolchain(output_dir)

    if args.run_metal_smoke:
        manifest["metal_smoke"] = _run_metal_smoke_tests(
            output_dir,
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
    if args.require_metal_toolchain:
        if not manifest["metal_toolchain"] or any(not item.get("ok", False) for item in manifest["metal_toolchain"]):
            return 8
    if any(not item.get("ok", False) for item in manifest["mlir_validation"]):
        return 2
    if any(not item.get("ok", False) and not item.get("no_default_device", False) for item in manifest["metal_smoke"]):
        return 4
    if any(not item.get("ok", False) for item in manifest["reference_verification"]):
        return 6
    if any(not item.get("ok", False) for item in manifest["performance_shape"]):
        return 7
    return 0


def _write_artifacts(artifacts):
    return [str(path) for path in artifacts.paths]


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
    vector_files = sorted(output_dir.rglob("*.vector.mlir"))
    matrix_unit_files = sorted(output_dir.rglob("*.matrix_unit.mlir"))
    matrix_unit_memref_files = sorted(output_dir.rglob("*.matrix_unit_memref.mlir"))
    matrix_unit_pipeline_files = sorted(output_dir.rglob("*.matrix_unit_pipeline.mlir"))
    gpu_files = sorted(output_dir.rglob("*.gpu.mlir"))
    metal_files = sorted(output_dir.rglob("*.metal"))

    results.append(_require_token("linalg_matmul", linalg_files, "linalg.matmul"))
    results.append(_require_token("vector_matrix_multiply", vector_files, "vector.matrix_multiply"))
    results.append(_require_token("matrix_unit_vector_matrix_multiply", matrix_unit_files, "vector.matrix_multiply"))
    results.append(_require_token("matrix_unit_memref_transfer_read", matrix_unit_memref_files, "vector.transfer_read"))
    results.append(_require_token("matrix_unit_memref_transfer_write", matrix_unit_memref_files, "vector.transfer_write"))
    results.append(_require_token("matrix_unit_pipeline_calls", matrix_unit_pipeline_files, "func.call"))
    results.append(_require_token("matrix_unit_pipeline_scratch", matrix_unit_pipeline_files, "memref.alloc"))
    results.extend(_verify_matrix_unit_alignment(path) for path in matrix_unit_files)
    results.extend(_verify_matrix_unit_alignment(path) for path in matrix_unit_memref_files)
    results.extend(_verify_matrix_unit_alignment(path) for path in matrix_unit_pipeline_files)
    for path in gpu_files:
        results.append(_forbid_tokens("gpu_hot_loop_shape", path, ("scf.if", "arith.cmpi", "atomic")))
        if path.name.endswith(".ebe.full.gpu.mlir"):
            results.append(_verify_ebe_gpu_topology(path))
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


def _probe_iree_metal(output_dir, sum_factor_lowering):
    iree_compile = shutil.which("iree-compile")
    if iree_compile is None:
        return {
            "name": "sum_factor_matrix_unit_pipeline_to_metal_vmfb",
            "ok": False,
            "skipped": True,
            "reason": "iree-compile is not available",
        }
    try:
        output_vmfb, result = sum_factor_lowering.compile_with_iree_metal(
            output_dir,
            iree_compile=iree_compile,
        )
    except subprocess.CalledProcessError as exc:
        return {
            "name": "sum_factor_matrix_unit_pipeline_to_metal_vmfb",
            "ok": False,
            "skipped": False,
            "returncode": exc.returncode,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
        }
    return {
        "name": "sum_factor_matrix_unit_pipeline_to_metal_vmfb",
        "ok": result.returncode == 0 and output_vmfb.exists(),
        "skipped": False,
        "returncode": result.returncode,
        "output_vmfb": str(output_vmfb),
        "stdout": result.stdout,
        "stderr": result.stderr,
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


def _run_metal_smoke_tests(output_dir, local_metal, ebe_metal):
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
    for name, lowering in (("local_apply", local_metal), ("ebe_map_reduce", ebe_metal)):
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
