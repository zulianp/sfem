"""The SoA backend bound to a GPU target.

Everything about *how* a planned unit becomes files is
`backends/soa.SoABackend`, which is the same whatever the target.  What is here
is only the four answers a device target gives.

Until the mesh-kernel lowering moved onto `TargetPlatform`, this backend
accepted `energy_soa` units and raised on everything else.  That was never a
statement about the residual family -- it was that the traversal which lowers
it lived behind a class this module could not name, and that the emitters wrote
their mesh loop, their kernel signature, their scatter and their launch by hand
in the CPU's spelling.
"""

from dataclasses import dataclass
import re

from codegen.framework.backends.soa import SoABackend, SoAEmission
from codegen.framework.forms.forms import FormOrder
from codegen.framework.emitters.energy import CUDAEnergySoAEmitter
from codegen.framework.plans.generation import KernelTarget, MeshPhase
from codegen.framework.targets import CUDATarget, HIPTarget, TargetLanguage, use_target

#: The name this module published before the traversal moved.
CUDASoAEmission = SoAEmission


@dataclass(frozen=True)
class CUDASoABackend(SoABackend):
    """CUDA/HIP SoA backend boundary for planned material code-generation units."""

    supports_op_wrapper: bool = True
    target: object = CUDATarget()
    emitter: object = None

    def mesh_source_extension(self):
        return self.target.mesh_source_extension()

    def local_header_extension(self):
        #: `.cuh` for HIP too: the header is included by device code either
        #: way, and `hipcc` reads `.cuh` without complaint.  The extension that
        #: matters is the translation unit's.
        return "cuh"

    def _shared_emitter(self):
        return CUDAEnergySoAEmitter(
            target=self.target, operator_extension=self.mesh_source_extension()
        )

    def _require_backend_target(self):
        if self.target.language not in (TargetLanguage.CUDA, TargetLanguage.HIP):
            raise ValueError("CUDA SoA backend requires a device target")

    def _require_unit_target(self, unit):
        _require_gpu_target(unit)

    def _validate_emitted(self, files, traversal):
        _validate_cuda_source_contract(files)

    def emit_inexact(self, material, unit, context):
        """No device lowering for the inexact-apply family yet.

        The backend is asked rather than the driver testing the target, so the
        answer lives where the target's capabilities do.  The residual family
        reached CUDA when the mesh-kernel lowering moved onto the target; this
        one has not been measured under a device target yet, and an untested
        `()` is a more honest answer than an untested file.
        """
        return ()


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
        #: A `pow` whose exponent is a literal should have been one of the
        #: target's `pow_m1` / `pow2` helpers, and that is what this catches.
        #: A `pow` whose exponent is a runtime value -- van Genuchten's `m`,
        #: two-phase flow's `C_kw1` -- has no specialisation to have missed,
        #: and CUDA provides `pow(double, double)` for device code.  Banning
        #: the name outright rejected eleven correct calls in
        #: `two_phase_flow_d2_simplex_local` and said nothing about any of them.
        literal_exponent = re.search(
            r"(?<![A-Za-z0-9_])pow\s*\([^()]*,\s*-?\d+(?:\.\d+)?\s*\)", file.source
        )
        if literal_exponent:
            raise RuntimeError(
                "CUDA file '%s' calls pow with a literal exponent: %s"
                % (file.path, literal_exponent.group(0))
            )
