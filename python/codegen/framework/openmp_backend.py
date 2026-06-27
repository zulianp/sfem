from dataclasses import dataclass

from .symbolic import (
    GeneratedKernelFile,
    generate_sfem_soa_cpp_files_for_element,
)
from .residual_codegen import (
    generate_coupled_residual_sfem_files,
    generate_mixed_residual_sfem_files,
)


@dataclass(frozen=True)
class OpenMPSoAEmission:
    files: tuple

    def __iter__(self):
        return iter(self.files)


@dataclass(frozen=True)
class OpenMPSoABackend:
    """Single OpenMP/SoA backend boundary for planned code-generation units."""

    def emit_energy(
        self,
        kernel_forms,
        *,
        prefix,
        local_prefix,
        specialization,
    ):
        files = tuple(
            generate_sfem_soa_cpp_files_for_element(
                kernel_forms,
                prefix=prefix,
                local_prefix=local_prefix,
                specialization=specialization,
            )
        )
        self._validate_common_source_contract(files, local_prefix)
        return OpenMPSoAEmission(files)

    def emit_residual(
        self,
        system,
        *,
        prefix,
        element_type,
        vector_size,
        quadrature_order,
        specialization,
        residual_coeffs,
        action_coeffs,
        local_prefix,
        local_name,
        operator_prefix,
        operator_name,
    ):
        files = tuple(
            generate_coupled_residual_sfem_files(
                system,
                prefix=prefix,
                element_type=element_type,
                vector_size=vector_size,
                quadrature_order=quadrature_order,
                specialization=specialization,
                residual_coeffs=residual_coeffs,
                action_coeffs=action_coeffs,
                local_prefix=local_prefix,
                local_name=local_name,
                operator_prefix=operator_prefix,
                operator_name=operator_name,
            )
        )
        self._validate_common_source_contract(files, local_prefix)
        return OpenMPSoAEmission(files)

    def emit_mixed_residual(
        self,
        system,
        *,
        prefix,
        compatible_element,
        vector_size,
        quadrature_order,
        residual_coeffs,
        action_coeffs,
        field_element_types,
        local_prefix,
        local_name,
        operator_prefix,
        operator_name,
    ):
        files = tuple(
            generate_mixed_residual_sfem_files(
                system,
                prefix=prefix,
                compatible_element=compatible_element,
                vector_size=vector_size,
                quadrature_order=quadrature_order,
                residual_coeffs=residual_coeffs,
                action_coeffs=action_coeffs,
                field_element_types=field_element_types,
                local_prefix=local_prefix,
                local_name=local_name,
                operator_prefix=operator_prefix,
                operator_name=operator_name,
            )
        )
        self._validate_common_source_contract(files, local_prefix)
        return OpenMPSoAEmission(files)

    @staticmethod
    def _validate_common_source_contract(files, local_prefix):
        source_by_path = {file.path: file.source for file in files}
        local_name = "%s_local.hpp" % local_prefix
        local_source = source_by_path.get(local_name)
        if local_source is None:
            raise RuntimeError("OpenMP SoA backend did not emit '%s'" % local_name)
        operator_sources = tuple(
            file for file in files if file.path.endswith("_operator.cpp")
        )
        if not operator_sources:
            raise RuntimeError("OpenMP SoA backend did not emit a mesh operator")
        if "template <typename scalar_t, int N_QP" not in local_source:
            raise RuntimeError(
                "OpenMP SoA local kernel '%s' is not templated on N_QP" % local_name
            )
        if "int VECTOR_SIZE" not in local_source:
            raise RuntimeError(
                "OpenMP SoA local kernel '%s' is not templated on VECTOR_SIZE"
                % local_name
            )
        block_name = "%s_" % local_prefix
        if block_name not in local_source:
            raise RuntimeError(
                "OpenMP SoA local kernel '%s' does not use local prefix '%s'"
                % (local_name, local_prefix)
            )
        include = '#include "%s"' % local_name
        for operator in operator_sources:
            if include not in operator.source:
                raise RuntimeError(
                    "OpenMP SoA operator '%s' does not include '%s'"
                    % (operator.path, local_name)
                )
