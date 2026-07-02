import importlib.util
import os
from pathlib import Path
import subprocess
import sys

from .common import PyMLIRAvailability


def _pymlir_namespace_conflict_hint():
    parser_spec = importlib.util.find_spec("mlir.parser")
    if parser_spec is None or parser_spec.origin is None:
        return ""
    if "astnodes" not in parser_spec.origin:
        return ""
    return (
        "pymlir conflicts with mlir-python-bindings in the mlir namespace; "
        "run: pip uninstall pymlir"
    )


def llvm_mlir_availability():
    try:
        from mlir import ir  # noqa: F401
        import mlir as module
        return PyMLIRAvailability(
            True,
            "mlir",
            str(getattr(module, "__file__", "")),
            "",
        )
    except Exception as exc:
        hint = _pymlir_namespace_conflict_hint()
        reason = f"{type(exc).__name__}: {exc}"
        if hint:
            reason = f"{reason} ({hint})"
        return PyMLIRAvailability(
            False,
            "",
            "",
            f"mlir-python-bindings are not importable by {sys.executable}; {reason}",
        )

def _find_mlir_opt():
    candidates = (
        os.environ.get("MLIR_OPT", ""),
        "/opt/homebrew/opt/llvm/bin/mlir-opt",
        "/Users/patrickzulian/.triton/llvm/llvm-064f02da-macos-arm64/bin/mlir-opt",
        "mlir-opt",
    )
    for candidate in candidates:
        if not candidate:
            continue
        try:
            subprocess.run(
                [candidate, "--version"],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            return candidate
        except (OSError, subprocess.CalledProcessError):
            continue
    raise FileNotFoundError("mlir-opt was not found; set MLIR_OPT")


def _find_mlir_runner():
    candidates = (
        os.environ.get("MLIR_RUNNER", ""),
        "/opt/homebrew/opt/llvm/bin/mlir-runner",
        "mlir-runner",
    )
    for candidate in candidates:
        if not candidate:
            continue
        try:
            subprocess.run(
                [candidate, "--version"],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            return candidate
        except (OSError, subprocess.CalledProcessError):
            continue
    raise FileNotFoundError("mlir-runner was not found; set MLIR_RUNNER")


def _find_mlir_translate():
    candidates = (
        os.environ.get("MLIR_TRANSLATE", ""),
        "/opt/homebrew/opt/llvm/bin/mlir-translate",
        "mlir-translate",
    )
    for candidate in candidates:
        if not candidate:
            continue
        try:
            subprocess.run(
                [candidate, "--version"],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            return candidate
        except (OSError, subprocess.CalledProcessError):
            continue
    raise FileNotFoundError("mlir-translate was not found; set MLIR_TRANSLATE")


def _translate_mlir_to_llvm_ir(source_path, output_path, mlir_translate=None):
    mlir_translate = mlir_translate or _find_mlir_translate()
    subprocess.run(
        [
            mlir_translate,
            str(source_path),
            "--mlir-to-llvmir",
            "-o",
            str(output_path),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _translate_emitc_to_cpp(module_text, mlir_translate=None):
    mlir_translate = mlir_translate or _find_mlir_translate()
    result = subprocess.run(
        [
            mlir_translate,
            "-",
            "--mlir-to-cpp",
        ],
        input=module_text,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def _translate_emitc_file_to_cpp(source_path, output_path, mlir_translate=None):
    mlir_translate = mlir_translate or _find_mlir_translate()
    subprocess.run(
        [
            mlir_translate,
            str(source_path),
            "--mlir-to-cpp",
            "-o",
            str(output_path),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _serialize_spirv_module(source_path, output_path, mlir_translate=None):
    mlir_translate = mlir_translate or _find_mlir_translate()
    subprocess.run(
        [
            mlir_translate,
            str(source_path),
            "--no-implicit-module",
            "--serialize-spirv",
            "-o",
            str(output_path),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _extract_single_top_level_operation(module_text, op_name):
    from mlir import ir

    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(module_text)
        operations = list(module.body.operations)
        if len(operations) != 1 or operations[0].name != op_name:
            raise ValueError(f"expected exactly one top-level {op_name} operation")
        return str(operations[0])


def _find_runner_library(name):
    candidates = (
        Path(os.environ.get("MLIR_RUNNER_LIB_DIR", "")) / name
        if os.environ.get("MLIR_RUNNER_LIB_DIR")
        else None,
        Path("/opt/homebrew/opt/llvm/lib") / name,
        Path("/usr/local/opt/llvm/lib") / name,
    )
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"{name} was not found; set MLIR_RUNNER_LIB_DIR")


def _parse_mlir_runner_i32_result(stdout):
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            return int(stripped)
        except ValueError:
            continue
    raise ValueError("mlir-runner did not print an i32 result")
