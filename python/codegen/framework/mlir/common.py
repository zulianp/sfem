from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class CodeInspectionArtifacts:
    output_dir: str
    files: tuple

    @property
    def paths(self):
        from pathlib import Path

        return tuple(Path(path) for path in self.files)

    def file_for_suffix(self, suffix):
        for path in self.paths:
            if path.name.endswith(suffix):
                return path
        raise KeyError(f"no inspection artifact ends with {suffix}")


@dataclass(frozen=True)
class PyMLIRAvailability:
    available: bool
    module_name: str = ""
    module_path: str = ""
    reason: str = ""


class MLIROptimizationStrategy(Enum):
    PARITY = "parity"
    CANONICAL = "canonical"
    SCALAR = "scalar"
    AGGRESSIVE = "aggressive"


@dataclass(frozen=True)
class MLIROptimizationPlan:
    strategy: MLIROptimizationStrategy
    pre_lowering_passes: tuple
    expected_effect: str


MLIR_OPTIMIZATION_PLANS = {
    MLIROptimizationStrategy.PARITY: MLIROptimizationPlan(
        MLIROptimizationStrategy.PARITY,
        (),
        "no extra MLIR optimization; structural parity with the current lowering",
    ),
    MLIROptimizationStrategy.CANONICAL: MLIROptimizationPlan(
        MLIROptimizationStrategy.CANONICAL,
        (
            "--canonicalize",
            "--cse",
            "--symbol-dce",
        ),
        "canonical forms and CSE before conversion",
    ),
    MLIROptimizationStrategy.SCALAR: MLIROptimizationPlan(
        MLIROptimizationStrategy.SCALAR,
        (
            "--sccp",
            "--canonicalize",
            "--cse",
            "--loop-invariant-code-motion",
            "--canonicalize",
            "--cse",
            "--symbol-dce",
        ),
        "constant propagation, CSE, and loop-invariant hoisting before conversion",
    ),
    MLIROptimizationStrategy.AGGRESSIVE: MLIROptimizationPlan(
        MLIROptimizationStrategy.AGGRESSIVE,
        (
            "--inline",
            "--sccp",
            "--canonicalize",
            "--cse",
            "--loop-invariant-code-motion",
            "--control-flow-sink",
            "--canonicalize",
            "--cse",
            "--symbol-dce",
        ),
        "aggressive scalar cleanup before OpenMP/LLVM conversion",
    ),
}


def mlir_optimization_strategy(value=None):
    if value is None:
        return MLIROptimizationStrategy.PARITY
    if isinstance(value, MLIROptimizationStrategy):
        return value
    lowered = str(value).strip().lower().replace("-", "_")
    aliases = {
        "none": MLIROptimizationStrategy.PARITY,
        "baseline": MLIROptimizationStrategy.PARITY,
        "parity": MLIROptimizationStrategy.PARITY,
        "canonical": MLIROptimizationStrategy.CANONICAL,
        "cse": MLIROptimizationStrategy.CANONICAL,
        "scalar": MLIROptimizationStrategy.SCALAR,
        "scalar_cleanup": MLIROptimizationStrategy.SCALAR,
        "aggressive": MLIROptimizationStrategy.AGGRESSIVE,
        "aggressive_scalar": MLIROptimizationStrategy.AGGRESSIVE,
    }
    try:
        return aliases[lowered]
    except KeyError as exc:
        valid = ", ".join(strategy.value for strategy in MLIROptimizationStrategy)
        raise ValueError(f"unknown MLIR optimization strategy {value!r}; expected one of: {valid}") from exc
