from codegen.framework.plans.conventions import restrict_prelude
from dataclasses import dataclass

from codegen.framework.emitters.ast_printer import (
    element_loop_lines,
    mesh_loop_lines,
)
from codegen.framework.emitters.tensor_product_geometry import geometry_kernels_header_source_for
from codegen.framework.plans.layout import is_tensor_product_family
from codegen.framework.targets import CUDATarget, OpenMPTarget


def _join_lines(lines):
    return "\n".join(line for line in lines if line != "")


@dataclass(frozen=True)
class OpenMPEnergySoASourceBuilder:
    operator_extension: str = "cpp"
    emit_objective_steps: bool = True
    target: object = OpenMPTarget()

    def header_name(self, stem):
        return self.target.header_name(stem)

    def header_guard_suffix(self):
        return self.target.header_guard_suffix()

    def inline_qualifier(self):
        return self.target.inline_qualifier()

    def local_header_preamble_lines(self, math_name, tensor_product_name, basis_family):
        return (
            '#include "%s"' % math_name,
            '#include "%s"' % tensor_product_name,
            "",
            *self.target.inline_definition_lines(),
            "",
            *restrict_prelude(""),
        )

    def operator_preamble_lines(self, local_name, geometry_name, diagnostics_name, extra_headers=()):
        return (
            "#include <cstdio>",
            "#include <type_traits>",
            '#include "%s"' % local_name,
            *('#include "%s"' % header for header in extra_headers),
            '#include "%s"' % geometry_name,
            '#include "%s"' % diagnostics_name,
            *self.target.includes(),
        )

    def geometry_header_source(self):
        return geometry_kernels_header_source_for(self.target)

    def emits_tensor_product_header(self, basis_family):
        return True

    def tensor_product_header_source(self):
        from codegen.framework.emitters.tensor_product_kernels import (
            tensor_product_kernels_header_source_for,
        )

        return tensor_product_kernels_header_source_for(self.target)

    def simd_lines(self):
        pragma = self.target.vectorize_pragma()
        return () if pragma is None else (pragma,)

    def work_item_index(self):
        return self.target.work_item_index()

    def work_item_name(self, name, component):
        return self.target.work_item_name(name, component)

    def diagnostic_work_item(self):
        return self.target.diagnostic_work_item()

    def work_item_loop_lines(self, indent):
        return self.target.work_item_loop_lines(indent)

    def parallel_for_lines(self, reduction=None):
        return self.target.parallel_element_loop_lines("static", reduction)

    def atomic_update_lines(self):
        """The atomic-update pragma, or nothing where the target has none.

        A tuple rather than a string so a target without the concept -- CUDA,
        which uses ``atomicAdd`` instead -- contributes no line, matching
        ``simd_lines``.
        """
        pragma = self.target.atomic_update_pragma()
        return () if pragma is None else (pragma,)

    def effective_vector_size(self, vector_size):
        return int(vector_size)

    def element_loop_lines(self, pragma_indent="", reduction=None):
        return element_loop_lines(self.target, pragma_indent, reduction=reduction)

    def mesh_loop_lines(self):
        return mesh_loop_lines(self.target)

    def mesh_template_line(self, geometry_mode, extra_template_params=()):
        """The mesh kernel's template head.

        `extra_template_params` carries the extents the body would otherwise
        declare as `static constexpr` -- today the vector width.  It is a
        parameter rather than a constant in the body because the width is a
        tuning choice, and it carries no default because a caller must choose:
        the generated entry point spells the number it wants at the point where
        it names the concrete kernel.
        """
        return "template <typename s_t, typename g_t%s>" % "".join(
            ", %s" % parameter for parameter in extra_template_params
        )

    def mesh_function_line(self, implementation_name):
        return self.target.mesh_function_line(implementation_name)

    def success_return_lines(self):
        return self.target.success_return_lines()

    def wrapper_call_lines(self, implementation_name, scalar_type, extra_template_args, wrapper_args):
        return self.target.mesh_launch_lines(
            implementation_name, "%s%s" % (scalar_type, extra_template_args), wrapper_args
        )

    def scatter_add_lines(self, lhs, rhs, indent):
        return self.target.scatter_add_lines(lhs, rhs, indent)


@dataclass(frozen=True)
class CUDAEnergySoASourceBuilder:
    operator_extension: str = "cu"
    #: The same answer the host builder gives, and for the reason
    #: `plans.form_emission.objective_kernel_variants` states: the plain
    #: objective is the stepped one with a single alpha of zero, and `x + 0*h`
    #: is `x` exactly, so emitting the plain kernel instead of the stepped one
    #: is a second path to one computation rather than a cheaper one.  While
    #: this said False the device published `objective` where the host
    #: published `objective_steps`, and the `Op`'s `value` -- which calls the
    #: stepped dispatch on both targets -- had nothing to call on the device:
    #: "isoparametric objective_steps 3d dispatch was not generated", at run
    #: time, on a GPU, for the merit.
    emit_objective_steps: bool = True
    target: object = CUDATarget()

    def header_name(self, stem):
        return self.target.header_name(stem)

    def header_guard_suffix(self):
        return self.target.header_guard_suffix()

    def inline_qualifier(self):
        return self.target.inline_qualifier()

    def local_header_preamble_lines(self, math_name, tensor_product_name, basis_family):
        tensor_include = (
            ('#include "%s"' % tensor_product_name,)
            if is_tensor_product_family(basis_family)
            else ()
        )
        return (
            '#include "%s"' % math_name,
            *tensor_include,
            "",
            *restrict_prelude(),
        )

    def operator_preamble_lines(self, local_name, geometry_name, diagnostics_name, extra_headers=()):
        return (
            "#include <type_traits>",
            *self.target.includes(),
            '#include "%s"' % local_name,
            *('#include "%s"' % header for header in extra_headers),
            '#include "%s"' % geometry_name,
            '#include "%s"' % diagnostics_name,
        )

    def geometry_header_source(self):
        return geometry_kernels_header_source_for(self.target)

    def emits_tensor_product_header(self, basis_family):
        return is_tensor_product_family(basis_family)

    def tensor_product_header_source(self):
        from codegen.framework.emitters.tensor_product_kernels import (
            tensor_product_kernels_header_source_for,
        )

        return tensor_product_kernels_header_source_for(self.target)

    def simd_lines(self):
        return ()

    def work_item_index(self):
        return self.target.work_item_index()

    def work_item_name(self, name, component):
        return self.target.work_item_name(name, component)

    def diagnostic_work_item(self):
        return self.target.diagnostic_work_item()

    def work_item_loop_lines(self, indent):
        return self.target.work_item_loop_lines(indent)

    def parallel_for_lines(self, reduction=None):
        return ()

    def atomic_update_lines(self):
        """Nothing: CUDA scatters with ``atomicAdd``, not a pragma."""
        return ()

    def effective_vector_size(self, vector_size):
        return 1

    def element_loop_lines(self, pragma_indent="", reduction=None):
        return element_loop_lines(self.target, pragma_indent, reduction=reduction)

    def mesh_loop_lines(self):
        return mesh_loop_lines(self.target)

    def mesh_template_line(self, geometry_mode, extra_template_params=()):
        """The mesh kernel's template head.

        `extra_template_params` carries the extents the body would otherwise
        declare as `static constexpr` -- today the vector width.  It is a
        parameter rather than a constant in the body because the width is a
        tuning choice, and it carries no default because a caller must choose:
        the generated entry point spells the number it wants at the point where
        it names the concrete kernel.
        """
        return "template <typename s_t, typename g_t%s>" % "".join(
            ", %s" % parameter for parameter in extra_template_params
        )

    def mesh_function_line(self, implementation_name):
        return self.target.mesh_function_line(implementation_name)

    def success_return_lines(self):
        return self.target.success_return_lines()

    def wrapper_call_lines(self, implementation_name, scalar_type, extra_template_args, wrapper_args):
        return self.target.mesh_launch_lines(
            implementation_name, "%s%s" % (scalar_type, extra_template_args), wrapper_args
        )

    def scatter_add_lines(self, lhs, rhs, indent):
        return self.target.scatter_add_lines(lhs, rhs, indent)


@dataclass(frozen=True)
class OpenMPEnergySoAEmitter:
    """Opaque OpenMP emitter: consume an energy-SoA emission plan, emit files."""

    supports_op_wrapper: bool = True
    target: object = OpenMPTarget()

    def _source_builder(self):
        return OpenMPEnergySoASourceBuilder(target=self.target)

    def shared_primitive_files(self):
        """The headers this target spells, with no material in the question.

        The tensor-product family is asked for by name because these headers
        belong to the target, not to any one element: whether a *given* element
        is tensor-product decides whether that element's kernels include the
        micro-kernels, not whether the tree has them.  Some element always is.
        """
        from codegen.framework.emitters.energy_codegen import shared_primitive_files

        return shared_primitive_files(self._source_builder(), "tensor_product")

    def emit_plan(self, plan):
        from codegen.framework.emitters.energy_codegen import generate_sfem_soa_cpp_files_for_element

        return generate_sfem_soa_cpp_files_for_element(
            plan, source_builder=self._source_builder()
        )


@dataclass(frozen=True)
class CUDAEnergySoAEmitter:
    """Opaque CUDA emitter: consume an energy-SoA emission plan, emit files."""

    supports_op_wrapper: bool = False
    target: object = CUDATarget()
    operator_extension: str = "cu"

    def _source_builder(self):
        return CUDAEnergySoASourceBuilder(
            operator_extension=self.operator_extension,
            target=self.target,
        )

    def shared_primitive_files(self):
        """The headers this target spells, with no material in the question.

        The tensor-product family is asked for by name because these headers
        belong to the target, not to any one element: whether a *given* element
        is tensor-product decides whether that element's kernels include the
        micro-kernels, not whether the tree has them.  Some element always is.
        """
        from codegen.framework.emitters.energy_codegen import shared_primitive_files

        return shared_primitive_files(self._source_builder(), "tensor_product")

    def emit_plan(self, plan):
        import dataclasses

        from codegen.framework.emitters.energy_codegen import generate_sfem_soa_cpp_files_for_element

        # No matrix assembly on this target, said out loud.  The façade used to
        # express it by leaving `matrix_format_plan` out of the seven arguments
        # it unpacked, which is the kind of decision that survives only as long
        # as nobody tidies the argument list -- passing the plan whole brought
        # the formats back and two backend tests caught it immediately.
        return generate_sfem_soa_cpp_files_for_element(
            dataclasses.replace(plan, matrix_format_plan=None),
            source_builder=self._source_builder(),
        )
