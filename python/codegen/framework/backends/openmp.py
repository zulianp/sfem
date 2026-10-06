"""The SoA backend bound to a CPU target.

Everything about *how* a planned unit becomes files -- which plans to build, in
what order, which emitter consumes each -- is `backends/soa.SoABackend` and is
the same whatever the target.  What is here is only the four answers a CPU
target gives: which unit targets it accepts, what it requires of its own
target, which energy emitter it builds, and what it checks about the files.
"""

from dataclasses import dataclass

from codegen.framework.backends.soa import SoABackend, SoAEmission, SoATraversal
from codegen.framework.emitters.energy import OpenMPEnergySoAEmitter
from codegen.framework.plans.generation import KernelTarget
from codegen.framework.targets import OpenMPTarget, TargetLanguage

#: The name this module published before the traversal moved.  Kept because
#: `pipeline/driver.py` and the tests name the emission type, and renaming it
#: would be churn in exchange for nothing.
OpenMPSoAEmission = SoAEmission

#: Likewise for the traversal record, which the tests construct directly.
_OpenMPTraversal = SoATraversal


@dataclass(frozen=True)
class OpenMPSoABackend(SoABackend):
    """Single OpenMP/SoA backend boundary for planned code-generation units."""

    supports_op_wrapper: bool = True
    target: object = OpenMPTarget()
    emitter: object = None

    def _shared_emitter(self):
        return OpenMPEnergySoAEmitter(target=self.target)

    def mesh_source_extension(self):
        return self.target.mesh_source_extension()

    def local_header_extension(self):
        return "hpp"

    def _require_backend_target(self):
        if self.target.language is not TargetLanguage.CPP:
            raise ValueError("OpenMP SoA backend requires a C++ CPU target")

    def _require_unit_target(self, unit):
        _require_openmp(unit)

    def _validate_emitted(self, files, traversal):
        # A unit that emitted nothing published nothing for this element -- a
        # merit-only unit on an element with no patch kernel is the case.  There
        # is no translation unit to hold to a contract, and an empty set is
        # unambiguous: a unit that should have emitted still emits its headers.
        if not files:
            return
        if traversal.local_name:
            self._validate_common_source_contract(files, traversal.local_prefix)
        else:
            self._validate_mesh_source_contract(files)

    @staticmethod
    def _validate_mesh_source_contract(files):
        operator_sources = tuple(
            file for file in files if file.path.endswith("_operator.cpp")
        )
        if not operator_sources:
            raise RuntimeError("OpenMP SoA backend did not emit a mesh operator")

    @staticmethod
    def _validate_common_source_contract(files, local_prefix):
        source_by_path = {file.path: file.source for file in files}
        local_name = "%s_local.hpp" % local_prefix
        OpenMPSoABackend._validate_mesh_source_contract(files)
        # A unit need not have an element-local kernel header.  One that
        # publishes no form has no element-local kernel to put in a header, and
        # emitting an empty one rather than none is what this contract exists to
        # catch -- so its absence is the correct state, not a missing file.  The
        # checks below are about the header a unit *does* emit.
        for local_source in filter(None, (source_by_path.get(local_name),)):
            OpenMPSoABackend._validate_local_kernel_contract(
                files, local_prefix, local_name, local_source
            )

    @staticmethod
    def _validate_local_kernel_contract(files, local_prefix, local_name, local_source):
        if "template <typename s_t, int NQ" not in local_source:
            raise RuntimeError(
                "OpenMP SoA local kernel '%s' is not templated on NQ" % local_name
            )
        if "int VS" not in local_source:
            raise RuntimeError(
                "OpenMP SoA local kernel '%s' is not templated on VS"
                % local_name
            )
        block_name = "%s_" % local_prefix
        if block_name not in local_source:
            raise RuntimeError(
                "OpenMP SoA local kernel '%s' does not use local prefix '%s'"
                % (local_name, local_prefix)
            )
        include = '#include "%s"' % local_name
        for operator in (
            file
            for file in files
            if file.path.endswith("_operator.cpp")
            and not file.path.endswith("_matrix_format_operator.cpp")
        ):
            if include not in operator.source:
                raise RuntimeError(
                    "OpenMP SoA operator '%s' does not include '%s'"
                    % (operator.path, local_name)
                )


def _require_openmp(unit):
    if unit.target not in (
        KernelTarget.OPENMP,
        KernelTarget.AVX512,
        KernelTarget.ARM_SVE,
        KernelTarget.ARM_SME,
    ):
        raise ValueError(
            "OpenMP SoA backend cannot emit target '%s'"
            % getattr(unit.target, "value", unit.target)
        )


