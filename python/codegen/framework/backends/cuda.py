from dataclasses import dataclass
import re

from codegen.framework.symbolic.forms import FormOrder
from codegen.framework.plans.energy import energy_soa_kernel_emission_plan
from codegen.framework.emitters.energy import CUDAEnergySoAEmitter
from codegen.framework.plans.generation import KernelTarget, MeshPhase
from codegen.framework.backends.targets import CUDATarget, HIPTarget


@dataclass(frozen=True)
class CUDASoAEmission:
    files: tuple

    def __iter__(self):
        return iter(self.files)


@dataclass(frozen=True)
class CUDASoABackend:
    """CUDA/SoA backend boundary for planned material code-generation units."""

    supports_op_wrapper: bool = False
    target: object = CUDATarget()
    emitter: object = None

    def emit(self, unit, context):
        _require_gpu_target(unit)
        unit.validate_for_context(context)
        kind = _kind_value(unit.kind)
        if kind != "energy_soa":
            raise ValueError(
                "CUDA SoA backend currently supports energy_soa units; got '%s'"
                % kind
            )
        files = tuple(self._emit_energy(unit, context))
        _validate_cuda_source_contract(files)
        return CUDASoAEmission(files)

    def _emit_energy(self, unit, context):
        _validate_energy_plan(unit)
        emitter = self.emitter
        if emitter is None:
            operator_extension = "hip" if isinstance(self.target, HIPTarget) else "cu"
            emitter = CUDAEnergySoAEmitter(
                target=self.target,
                operator_extension=operator_extension,
            )
        return emitter.emit_plan(energy_soa_kernel_emission_plan(unit, context))


def _kind_value(kind):
    return getattr(kind, "value", str(kind))


def _validate_energy_plan(unit):
    _require_form_metadata(unit, tuple(form.order for form in unit.form_collection.forms))
    _require_geometry_modes(unit, ("affine", "isoparametric"))
    _require_mesh_phases(
        unit,
        (
            MeshPhase.GEOMETRY,
            MeshPhase.LOCAL_CALL,
            MeshPhase.SCATTER,
        ),
    )


def _require_gpu_target(unit):
    if unit.target not in (KernelTarget.OPENMP, KernelTarget.CUDA, KernelTarget.HIP):
        raise ValueError(
            "CUDA/HIP SoA backend cannot emit target '%s'"
            % getattr(unit.target, "value", unit.target)
        )


def _require_form_metadata(unit, orders):
    for order in orders:
        try:
            unit.form_collection.form_metadata(order)
        except ValueError as exc:
            raise ValueError(
                "kernel plan '%s' is missing FormMetadata for %s"
                % (unit.name, FormOrder(order).name)
            ) from exc


def _require_geometry_modes(unit, modes):
    geometries = ()
    for phase in unit.mesh_phase_plans:
        if phase.phase is MeshPhase.GEOMETRY:
            geometries = phase.geometries
            break
    available = {geometry.mode.value for geometry in geometries}
    missing = tuple(mode for mode in modes if mode not in available)
    if missing:
        raise ValueError(
            "kernel plan '%s' is missing geometry phase modes: %s"
            % (unit.name, ", ".join(missing))
        )


def _require_mesh_phases(unit, phases):
    actual = tuple(phase.phase for phase in unit.mesh_phase_plans)
    expected = tuple(MeshPhase(phase) for phase in phases)
    if actual != expected:
        raise ValueError(
            "kernel plan '%s' mesh phases %s do not match expected %s"
            % (
                unit.name,
                ", ".join(phase.value for phase in actual),
                ", ".join(phase.value for phase in expected),
            )
        )


def _validate_cuda_source_contract(files):
    operator_sources = tuple(
        file
        for file in files
        if file.path.endswith("_operator.cu") or file.path.endswith("_operator.hip")
    )
    if not operator_sources:
        raise RuntimeError("CUDA SoA backend did not emit a CUDA mesh operator")
    for operator in operator_sources:
        if "__global__ void" not in operator.source:
            raise RuntimeError("CUDA operator '%s' does not emit CUDA kernels" % operator.path)
        if "#pragma omp" in operator.source:
            raise RuntimeError("CUDA operator '%s' contains OpenMP pragmas" % operator.path)
    for file in files:
        if "#pragma omp" in file.source:
            raise RuntimeError("CUDA file '%s' contains OpenMP pragmas" % file.path)
        if "lane" in file.source:
            raise RuntimeError("CUDA file '%s' contains vector-lane lowering" % file.path)
        if "const ptrdiff_t thread = 0" in file.source:
            raise RuntimeError("CUDA file '%s' contains fake thread lowering" % file.path)
        if "SFEM_INLINE" in file.source:
            raise RuntimeError("CUDA file '%s' contains SFEM_INLINE macro usage" % file.path)
        if re.search(r"(?<![A-Za-z0-9_])pow\s*\(", file.source):
            raise RuntimeError("CUDA file '%s' contains generic pow usage" % file.path)
