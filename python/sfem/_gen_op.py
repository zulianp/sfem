import json
import os
import re


def generate_op_registration_files(manifests, function_name="register_generated_ops"):
    entries = _registration_entries_from_manifests(manifests)
    header_name = "sfem_generated_ops_registration.hpp"
    source_name = "sfem_generated_ops_registration.cpp"
    return {
        header_name: _registration_aggregate_header(function_name),
        source_name: _registration_aggregate_source(header_name, function_name, entries),
    }


def generate_op_files(material, elements, kernel_sources=None):
    c_abi_header = "sfem_%s_c_abi.hpp" % material.op_name if kernel_sources else None
    kernel_sources = dict(kernel_sources or {})
    dispatch_sources = (
        _dispatch_sources(material, elements, c_abi_header, kernel_sources)
        if c_abi_header
        else {}
    )
    abi_sources = dict(kernel_sources)
    abi_sources.update(dispatch_sources)
    systems_by_dim = _systems_by_dim(material, elements)
    equations = _representative_equations(systems_by_dim)
    if len(equations) > 1:
        header, source = _coupled_energy_residual_op(
            material, elements, c_abi_header, systems_by_dim
        )
    elif not equations:
        raise ValueError("generated Op wrappers require at least one equation")
    elif equations[0].name:
        raise ValueError("single-equation generated Op wrappers require an unnamed equation")
    elif equations[0].is_energy:
        form_collections = _single_equation_form_collections(systems_by_dim, equations[0])
        header, source = _hyperelastic_op(
            material, elements, c_abi_header, form_collections, abi_sources
        )
    elif equations[0].is_residual:
        form_collections = _single_equation_form_collections(systems_by_dim, equations[0])
        measures = {collection.measure for collection in form_collections.values()}
        if measures == {"ds"}:
            header, source = _boundary_residual_op(
                material, elements, c_abi_header, form_collections
            )
        elif "ds" in measures:
            raise ValueError("generated residual Op wrappers cannot mix dx and ds forms yet")
        else:
            header, source = _residual_op(
                material, elements, c_abi_header, form_collections, abi_sources
            )
    else:
        raise ValueError("unsupported generated Op equation form")
    wrapper_header = "op/sfem_%s.hpp" % material.op_name
    wrapper_source = "op/sfem_%s.cpp" % material.op_name
    registration_source = "op/sfem_%s_registration.cpp" % material.op_name
    files = {
        wrapper_header: header,
        wrapper_source: source,
        registration_source: _registration_source(material, wrapper_header),
    }
    files.update(dispatch_sources)
    if c_abi_header:
        c_abi_path = "op/%s" % c_abi_header
        files[c_abi_path] = _c_abi_header(material, abi_sources)
        files["op/sfem_%s_manifest.json" % material.op_name] = _op_manifest(
            material,
            abi_sources,
            wrapper_header,
            wrapper_source,
            registration_source,
            c_abi_path,
        )
    return files


def _registration_entries_from_manifests(manifests):
    entries = []
    seen_operators = set()
    for manifest in manifests:
        if isinstance(manifest, str):
            manifest = json.loads(manifest)
        _validate_op_manifest(manifest)
        registration = manifest["registration"]
        operator_name = registration["operator_name"]
        if operator_name in seen_operators:
            raise ValueError(
                "generated Op manifest registration operator '%s' is duplicated"
                % operator_name
            )
        seen_operators.add(operator_name)
        entries.append(
            (
                operator_name,
                registration["function"].replace("sfem::", ""),
            )
        )
    return tuple(sorted(entries))


def _validate_op_manifest(manifest):
    if not isinstance(manifest, dict):
        raise TypeError("generated Op manifest must be a JSON object")
    if manifest.get("schema") != "sfem.generated_op_manifest.v1":
        raise ValueError("generated Op manifest has unsupported schema")

    material = manifest.get("material")
    op_name = manifest.get("op_name")
    if not _nonempty_string(material):
        raise ValueError("generated Op manifest requires a material name")
    if not _nonempty_string(op_name):
        raise ValueError("generated Op manifest requires an op_name")

    wrapper = _required_mapping(manifest, "wrapper")
    _required_string(wrapper, "header", "generated Op manifest wrapper")
    _required_string(wrapper, "source", "generated Op manifest wrapper")
    _required_string(wrapper, "c_abi_header", "generated Op manifest wrapper")

    registration = _required_mapping(manifest, "registration")
    _required_string(registration, "source", "generated Op manifest registration")
    function = _required_string(registration, "function", "generated Op manifest registration")
    operator_name = _required_string(
        registration,
        "operator_name",
        "generated Op manifest registration",
    )
    if operator_name != op_name:
        raise ValueError("generated Op manifest registration operator_name must match op_name")
    if not function.startswith("sfem::"):
        raise ValueError("generated Op manifest registration function must be namespace-qualified")

    factory = _required_mapping(manifest, "factory")
    _required_string(factory, "class", "generated Op manifest factory")
    _required_string(factory, "create", "generated Op manifest factory")
    _required_string(factory, "create_from_yaml", "generated Op manifest factory")

    include_paths = manifest.get("generated_include_paths")
    if not isinstance(include_paths, (list, tuple)) or not include_paths:
        raise ValueError("generated Op manifest requires generated_include_paths")
    if not all(_nonempty_string(path) for path in include_paths):
        raise ValueError("generated Op manifest include paths must be strings")

    c_abi = manifest.get("c_abi")
    if not isinstance(c_abi, (list, tuple)) or not c_abi:
        raise ValueError("generated Op manifest requires c_abi declarations")
    c_abi_names = _validate_manifest_c_abi(c_abi)
    _validate_manifest_runtime_operations(manifest.get("runtime_operations"), c_abi_names)


def _validate_manifest_c_abi(c_abi):
    names = set()
    for index, entry in enumerate(c_abi):
        if not isinstance(entry, dict):
            raise ValueError("generated Op manifest c_abi entry %d must be an object" % index)
        name = _required_string(entry, "name", "generated Op manifest c_abi entry")
        declaration = _required_string(
            entry,
            "declaration",
            "generated Op manifest c_abi entry",
        )
        if name in names:
            raise ValueError("generated Op manifest c_abi function '%s' is duplicated" % name)
        if 'extern "C"' not in declaration or not declaration.rstrip().endswith(";"):
            raise ValueError(
                "generated Op manifest c_abi function '%s' must be an extern C declaration"
                % name
            )
        if _c_abi_function_name(declaration) != name:
            raise ValueError(
                "generated Op manifest c_abi function '%s' does not match its declaration"
                % name
            )
        names.add(name)
    return names


def _validate_manifest_runtime_operations(runtime_operations, c_abi_names):
    if not isinstance(runtime_operations, (list, tuple)) or not runtime_operations:
        raise ValueError("generated Op manifest requires runtime_operations")
    for operation in runtime_operations:
        if not isinstance(operation, dict):
            raise ValueError("generated Op manifest runtime operation must be an object")
        _required_string(operation, "name", "generated Op manifest runtime operation")
        variants = operation.get("variants")
        if not isinstance(variants, (list, tuple)) or not variants:
            raise ValueError("generated Op manifest runtime operation requires variants")
        for variant in variants:
            if not isinstance(variant, dict):
                raise ValueError("generated Op manifest runtime variant must be an object")
            _required_string(variant, "variant", "generated Op manifest runtime variant")
            _required_string(variant, "scalar_type", "generated Op manifest runtime variant")
            function = _required_string(
                variant,
                "function",
                "generated Op manifest runtime variant",
            )
            _required_string(variant, "target", "generated Op manifest runtime variant")
            if function not in c_abi_names:
                raise ValueError(
                    "generated Op manifest runtime function '%s' is not declared in c_abi"
                    % function
                )


def _required_mapping(mapping, key):
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise ValueError("generated Op manifest requires %s metadata" % key)
    return value


def _required_string(mapping, key, context):
    value = mapping.get(key)
    if not _nonempty_string(value):
        raise ValueError("%s requires %s" % (context, key))
    return value


def _nonempty_string(value):
    return isinstance(value, str) and bool(value)


def _registration_aggregate_header(function_name):
    return """#pragma once

namespace sfem {
    void %(function)s();
}  // namespace sfem
""" % {
        "function": function_name,
    }


def _registration_aggregate_source(header_name, function_name, entries):
    declarations = "\n".join("    void %s();" % function for _, function in entries)
    calls = "\n".join("        %s();" % function for _, function in entries)
    if declarations:
        declarations += "\n"
    if calls:
        calls += "\n"
    return """#include "%(header)s"

namespace sfem {
%(declarations)s
    void %(function)s() {
%(calls)s    }
}  // namespace sfem
""" % {
        "header": header_name,
        "declarations": declarations,
        "function": function_name,
        "calls": calls,
    }


def _systems_by_dim(material, elements):
    systems = getattr(material, "systems", None)
    if systems is None:
        raise TypeError("generated Op wrappers require a CodeGenerator with equation systems")
    return {
        dim: systems.for_dim(dim)
        for dim in sorted({_element_dim(element) for element in elements})
    }


def _representative_equations(systems_by_dim):
    first_dim = next(iter(sorted(systems_by_dim)))
    return tuple(systems_by_dim[first_dim].equations)


def _single_equation_form_collections(systems_by_dim, representative_equation):
    orders = _equation_form_orders(representative_equation)
    collections = {}
    for dim, system in systems_by_dim.items():
        equations = tuple(system.equations)
        if len(equations) != 1:
            raise ValueError("single-equation generated Op wrappers require one equation per dimension")
        collections[dim] = system.form_collection(equations[0], orders=orders)
    return collections


def _equation_form_orders(equation):
    if equation.is_energy:
        orders = [_form_order_zero()]
        for kernel in equation.kernels:
            if kernel == "objective":
                orders.append(_form_order_zero())
            elif kernel == "gradient":
                orders.append(_form_order_one())
            elif kernel == "apply":
                orders.append(_form_order_two())
        return tuple(dict.fromkeys(orders))
    if equation.is_residual:
        return (_form_order_zero(), _form_order_one(), _form_order_two())
    raise ValueError("unsupported equation form")


def _header(material, residual):
    extra = """
        int update(const real_t *const x) override;
        int update(const real_t *const previous, const real_t *const current) override;
        void set_field(const char *name,
                       const std::shared_ptr<Buffer<real_t>> &values,
                       int component) override;""" if residual else ""
    value_steps = "" if residual else """
        int value_steps(const real_t *x,
                        const real_t *h,
                        const int nsteps,
                        const real_t *const steps,
                        real_t *const out) override;"""
    matrix_methods = """
        int hessian_bsr(const real_t *const x,
                        const count_t *const rowptr,
                        const idx_t *const colidx,
                        real_t *const values) override;
        int hessian_dia(const real_t *const x,
                        const int *const diag_offsets,
                        const ptrdiff_t ndiag,
                        real_t *const values) override;""" if residual else """
        int hessian_bsr(const real_t *const x,
                        const count_t *const rowptr,
                        const idx_t *const colidx,
                        real_t *const values) override;
        int hessian_dia(const real_t *const x,
                        const int *const diag_offsets,
                        const ptrdiff_t ndiag,
                        real_t *const values) override;
        int hessian_coo(const real_t *const x,
                        const ptrdiff_t nnz,
                        const idx_t *const rows,
                        const idx_t *const cols,
                        real_t *const values);
        int hessian_patch(const real_t *const x,
                          const count_t *const rowptr,
                          const idx_t *const colidx,
                          real_t *const values);"""
    return """#pragma once

#include "sfem_Op.hpp"
#include "sfem_NeumannConditions.hpp"

namespace sfem {
    class %(op)s final : public Op {
    public:
        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space);

        explicit %(op)s(const std::shared_ptr<FunctionSpace> &space);
        ~%(op)s() override;

        const char *name() const override { return "%(op)s"; }
        bool is_linear() const override { return false; }
        ptrdiff_t n_dofs_domain() const override;
        ptrdiff_t n_dofs_image() const override;
        double flops_value() const override;
        double flops_gradient() const override;
        double flops_apply() const override;
        size_t memory_traffic_bytes_value() const override;
        size_t memory_traffic_bytes_gradient() const override;
        size_t memory_traffic_bytes_apply() const override;

        int initialize(const std::vector<std::string> &block_names = {}) override;%(extra)s
        int gradient(const real_t *const x, real_t *const out) override;
        int apply(const real_t *const x,
                  const real_t *const h,
                  real_t *const out) override;
        int value(const real_t *x, real_t *const out) override;%(value_steps)s
        int hessian_crs(const real_t *const x,
                        const count_t *const rowptr,
                        const idx_t *const colidx,
                        real_t *const values) override;%(matrix_methods)s
        void set_option(const std::string &name, bool val) override;
        void set_value_in_block(const std::string &block_name,
                                const std::string &var_name,
                                real_t value) override;
#ifdef SFEM_ENABLE_RYAML
        std::shared_ptr<Op> create_from_yaml(const std::shared_ptr<FunctionSpace> &space,
                                             const ryml::ConstNodeRef             &node) override;
#endif  // SFEM_ENABLE_RYAML

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
}  // namespace sfem
""" % {
        "op": material.op_name,
        "extra": extra,
        "value_steps": value_steps,
        "matrix_methods": matrix_methods,
    }


def _hyperelastic_op(
    material, elements, c_abi_header=None, form_collections=None, kernel_sources=None
):
    if form_collections is None:
        raise ValueError("energy generated Op requires form collections")
    if kernel_sources is None:
        kernel_sources = {}
    parameters = tuple(str(name) for name, _ in material.parameter_defaults)
    defaults = _seed_lines(material.parameter_defaults)
    declarations = []
    gradient_cases = []
    apply_cases = []
    objective_cases = []
    objective_steps_cases = []
    hessian_crs_cases = []
    hessian_bsr_cases = []
    hessian_dia_cases = []
    hessian_coo_cases = []
    hessian_patch_cases = []
    generated_packed_apply = any("_packed_" in source for source in kernel_sources.values())
    packed_scratch_include = '#include "packed_thread_scratch.hpp"\n#include "smesh_env.hpp"' if generated_packed_apply else ""
    packed_scratch_prealloc = (
        """        impl_->use_packed_two_pass = smesh::Env::read("SFEM_PACKED_TWO_PASS", false);
        if (impl_->space->has_packed_mesh()) {
            auto packed = impl_->space->packed_mesh();
            const ptrdiff_t max_nodes_per_pack = packed->max_nodes_per_pack();
            const int dim = impl_->space->mesh_ptr()->spatial_dimension();
            const size_t scratch_size = (size_t)dim * (size_t)max_nodes_per_pack;
            sfem::codegen::prealloc_thread_scratch<real_t>(0, scratch_size);
            sfem::codegen::prealloc_thread_scratch<real_t>(1, scratch_size);
            sfem::codegen::prealloc_thread_scratch<real_t>(2, scratch_size);
            sfem::codegen::prealloc_thread_scratch<real_t>(3, scratch_size);
            impl_->packed_ghost_buf.resize((size_t)packed->n_blocks());
            for (int b = 0; b < packed->n_blocks(); ++b) {
                const ptrdiff_t n_ghost = packed->n_ghost_entries(b);
                const ptrdiff_t n_slots = (n_ghost > 0 ? n_ghost : 1) * (ptrdiff_t)dim;
                impl_->packed_ghost_buf[b] = create_host_buffer<real_t>(n_slots);
            }
        }"""
        if generated_packed_apply
        else ""
    )
    performance_cases = {"value": [], "gradient": [], "apply": []}
    dependencies_by_dim = {}
    gradient_affine_aos_flags = []
    apply_affine_aos_flags = []
    for element in elements:
        dim = _element_dim(element)
        dependencies = dependencies_by_dim.get(dim)
        if dependencies is None:
            collection = form_collections[dim]
            dependencies = (
                collection.form_metadata(_form_order_zero()).dependencies,
                collection.form_metadata(_form_order_one()).dependencies,
                collection.form_metadata(_form_order_two()).dependencies,
            )
            dependencies_by_dim[dim] = dependencies
        objective_dependencies, gradient_dependencies, apply_dependencies = dependencies
        stem = "%s_%s_%s" % (
            material.name,
            _element_name(element).lower(),
            _element_name(element).lower(),
        )
        performance_cases["value"].append(
            _performance_case(
                element,
                ("%s_objective_soa_diagnostics" % stem,),
                affine_flags=("objective_uses_affine",),
            )
        )
        performance_cases["gradient"].append(
            _performance_case(
                element,
                ("%s_gradient_soa_diagnostics" % stem,),
                affine_flags=("gradient_uses_affine",),
            )
        )
        performance_cases["apply"].append(
            _performance_case(
                element,
                ("%s_apply_soa_diagnostics" % stem,),
                affine_flags=("apply_uses_affine",),
            )
        )
        components = _components(dim)
        declarations.extend(
            _hyperelastic_declarations(stem, dim, parameters, dependencies)
        )
        objective_args = "".join(
            ", %s" % arg for arg in _dependency_domain_parameter_args(objective_dependencies)
        )
        gradient_args = "".join(
            ", %s" % arg for arg in _dependency_domain_parameter_args(gradient_dependencies)
        )
        apply_args = "".join(
            ", %s" % arg for arg in _dependency_domain_parameter_args(apply_dependencies)
        )
        gradient_common_isoparametric_args = (
            "domain.block->n_elements(), mesh->n_nodes(), "
            "domain.block->elements()->data(), points%s" % gradient_args
        )
        gradient_common_affine_args = (
            "domain.block->n_elements(), mesh->n_nodes(), "
            "domain.block->elements()->data(), %s, determinant%s"
            % (_affine_geometry_offsets(dim), gradient_args)
        )
        gradient_common_affine_aos_args = (
            "domain.block->n_elements(), mesh->n_nodes(), "
            "domain.block->elements()->data(), adjugate_aos, determinant%s"
            % gradient_args
        )
        apply_common_isoparametric_args = (
            "domain.block->n_elements(), mesh->n_nodes(), "
            "domain.block->elements()->data(), points%s" % apply_args
        )
        apply_common_affine_args = (
            "domain.block->n_elements(), mesh->n_nodes(), "
            "domain.block->elements()->data(), %s, determinant%s"
            % (_affine_geometry_offsets(dim), apply_args)
        )
        apply_common_affine_aos_args = (
            "domain.block->n_elements(), mesh->n_nodes(), "
            "domain.block->elements()->data(), adjugate_aos, determinant%s"
            % apply_args
        )
        gradient_affine_uses_aos = _c_abi_function_exists(
            kernel_sources, "%s_gradient_affine_mesh_soa_aos_unit" % stem
        )
        apply_affine_uses_aos = _c_abi_function_exists(
            kernel_sources, "%s_apply_affine_mesh_soa_aos_unit" % stem
        )
        gradient_affine_aos_flags.append(gradient_affine_uses_aos)
        apply_affine_aos_flags.append(apply_affine_uses_aos)
        gradient_affine_args = ", ".join(
            _nonempty(
                gradient_common_affine_args,
                *_energy_field_args(
                    gradient_dependencies, dim, components, current="x"
                ),
                *_energy_output_args(dim, components),
            )
        )
        gradient_isoparametric_args = ", ".join(
            _nonempty(
                gradient_common_isoparametric_args,
                *_energy_field_args(
                    gradient_dependencies, dim, components, current="x"
                ),
                *_energy_output_args(dim, components),
            )
        )
        if gradient_affine_uses_aos:
            gradient_cases.append(
                _dual_aos_unit_case(
                    element,
                    "gradient_uses_affine",
                    "%s_gradient_affine_mesh_soa_aos_unit" % stem,
                    ", ".join(
                        _nonempty(
                            gradient_common_affine_aos_args,
                            *_energy_field_args(
                                gradient_dependencies, dim, components, current="x"
                            ),
                            *_energy_output_args(dim, components),
                        )
                    ),
                    "%s_gradient_affine_mesh_soa" % stem,
                    gradient_affine_args,
                    "%s_gradient_isoparametric_mesh_soa" % stem,
                    gradient_isoparametric_args,
                )
            )
        else:
            gradient_cases.append(
                _dual_case(
                    element,
                    "gradient_uses_affine",
                    "%s_gradient_affine_mesh_soa" % stem,
                    gradient_affine_args,
                    "%s_gradient_isoparametric_mesh_soa" % stem,
                    gradient_isoparametric_args,
                )
            )
        apply_affine_args = ", ".join(
            _nonempty(
                apply_common_affine_args,
                *_energy_field_args(
                    apply_dependencies,
                    dim,
                    components,
                    current="x",
                    direction="h",
                ),
                *_energy_output_args(dim, components),
            )
        )
        apply_isoparametric_args = ", ".join(
            _nonempty(
                apply_common_isoparametric_args,
                *_energy_field_args(
                    apply_dependencies,
                    dim,
                    components,
                    current="x",
                    direction="h",
                ),
                *_energy_output_args(dim, components),
            )
        )
        if apply_affine_uses_aos:
            apply_cases.append(
                _dual_aos_unit_case(
                    element,
                    "apply_uses_affine",
                    "%s_apply_affine_mesh_soa_aos_unit" % stem,
                    ", ".join(
                        _nonempty(
                            apply_common_affine_aos_args,
                            *_energy_field_args(
                                apply_dependencies,
                                dim,
                                components,
                                current="x",
                                direction="h",
                            ),
                            *_energy_output_args(dim, components),
                        )
                    ),
                    "%s_apply_affine_mesh_soa" % stem,
                    apply_affine_args,
                    "%s_apply_isoparametric_mesh_soa" % stem,
                    apply_isoparametric_args,
                )
            )
        else:
            apply_cases.append(
                _dual_case(
                    element,
                    "apply_uses_affine",
                    "%s_apply_affine_mesh_soa" % stem,
                    apply_affine_args,
                    "%s_apply_isoparametric_mesh_soa" % stem,
                    apply_isoparametric_args,
                )
            )
        objective_cases.append(
            _dual_status_case(
                element,
                "%s_objective_affine_mesh_soa" % stem,
                ", ".join(_nonempty(
                    "nelements, mesh->n_nodes(), domain.block->elements()->data(), %s, determinant%s"
                    % (_affine_geometry_offsets(dim), objective_args),
                    *_energy_field_args(objective_dependencies, dim, components, current="x"),
                    "impl_->element_values.get()",
                )),
                "%s_objective_isoparametric_mesh_soa" % stem,
                ", ".join(_nonempty(
                    "nelements, mesh->n_nodes(), domain.block->elements()->data(), points%s"
                    % objective_args,
                    *_energy_field_args(objective_dependencies, dim, components, current="x"),
                    "impl_->element_values.get()",
                )),
            )
        )
        objective_steps_cases.append(
            _dual_status_case(
                element,
                "%s_objective_steps_affine_mesh_soa" % stem,
                ", ".join(_nonempty(
                    "nelements, mesh->n_nodes(), domain.block->elements()->data(), %s, determinant%s"
                    % (_affine_geometry_offsets(dim), objective_args),
                    *_energy_field_args(objective_dependencies, dim, components, current="x"),
                    dim,
                    _offsets("h", components),
                    "nsteps",
                    "steps",
                    "impl_->element_values.get()",
                )),
                "%s_objective_steps_isoparametric_mesh_soa" % stem,
                ", ".join(_nonempty(
                    "nelements, mesh->n_nodes(), domain.block->elements()->data(), points%s"
                    % objective_args,
                    *_energy_field_args(objective_dependencies, dim, components, current="x"),
                    dim,
                    _offsets("h", components),
                    "nsteps",
                    "steps",
                    "impl_->element_values.get()",
                )),
            )
        )
        hessian_state_args = ", ".join(
            _nonempty(
                apply_common_isoparametric_args,
                *_energy_field_args(
                    apply_dependencies,
                    dim,
                    components,
                    current="current",
                ),
            )
        )
        hessian_crs_function = "%s_hessian_crs_isoparametric_mesh_soa" % stem
        if _c_abi_function_exists(kernel_sources, hessian_crs_function):
            hessian_crs_cases.append(
                _case(
                    element,
                    hessian_crs_function,
                    ", ".join(
                        _nonempty(
                            hessian_state_args,
                            "rowptr",
                            "colidx",
                            "values",
                        )
                    ),
                )
            )
        hessian_bsr_function = "%s_hessian_bsr_isoparametric_mesh_soa" % stem
        if _c_abi_function_exists(kernel_sources, hessian_bsr_function):
            hessian_bsr_cases.append(
                _case(
                    element,
                    hessian_bsr_function,
                    ", ".join(
                        _nonempty(
                            hessian_state_args,
                            "rowptr",
                            "colidx",
                            "values",
                        )
                    ),
                )
            )
        hessian_dia_function = "%s_hessian_dia_isoparametric_mesh_soa" % stem
        if _c_abi_function_exists(kernel_sources, hessian_dia_function):
            hessian_dia_cases.append(
                _case(
                    element,
                    hessian_dia_function,
                    ", ".join(
                        _nonempty(
                            hessian_state_args,
                            "diag_offsets",
                            "ndiag",
                            "values",
                        )
                    ),
                )
            )
        hessian_coo_function = "%s_hessian_coo_isoparametric_mesh_soa" % stem
        if _c_abi_function_exists(kernel_sources, hessian_coo_function):
            hessian_coo_cases.append(
                _case(
                    element,
                    hessian_coo_function,
                    ", ".join(
                        _nonempty(
                            hessian_state_args,
                            "nnz",
                            "rows",
                            "cols",
                            "values",
                        )
                    ),
                )
            )
        hessian_patch_function = "%s_hessian_patch_isoparametric_mesh_soa" % stem
        if _c_abi_function_exists(kernel_sources, hessian_patch_function):
            hessian_patch_cases.append(
                _case(
                    element,
                    hessian_patch_function,
                    ", ".join(
                        _nonempty(
                            hessian_state_args,
                            "rowptr",
                            "colidx",
                            "values",
                        )
                    ),
                )
            )
    source = """#include "sfem_%(op)s.hpp"
%(c_abi_include)s
%(packed_scratch_include)s

#include "sfem_FunctionSpace.hpp"
#include "sfem_MultiDomainOp.hpp"
#include "sfem_OpTracer.hpp"
#include "sfem_Parameters.hpp"
#include "smesh_kernel_data.hpp"
#include "smesh_mesh.hpp"

#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

%(declaration_block)s

namespace sfem {
    namespace {
        void seed_parameters(Parameters &parameters) {
%(defaults)s
        }

        void seed_material(MultiDomainOp &domains) {
            for (auto &entry : domains.domains()) {
                seed_parameters(*entry.second.parameters);
            }
        }

%(yaml_helpers)s

        smesh::block_idx_t block_id_for_domain(const smesh::Mesh &mesh,
                                               const smesh::Mesh::Block &block) {
            for (size_t i = 0; i < mesh.n_blocks(); ++i) {
                if (mesh.block(i).get() == &block) {
                    return static_cast<smesh::block_idx_t>(i);
                }
            }
            SFEM_ERROR("%(op)s: mesh block pointer not found in mesh.blocks()\\n");
            return 0;
        }

        int packed_block_id_for_domain(const FunctionSpace::PackedMesh &packed,
                                       const smesh::Mesh::Block &block) {
            for (ptrdiff_t i = 0; i < packed.n_blocks(); ++i) {
                if (packed.block_name(i) == block.name()) {
                    return static_cast<int>(i);
                }
            }
            return -1;
        }

        struct AffineGeometryCache {
            std::shared_ptr<smesh::JacobianAdjugateAndDeterminant> jacobian_soa;
            std::shared_ptr<smesh::JacobianAdjugateAndDeterminant> jacobian_aos;
        };

        int cache_affine_geometry(const std::shared_ptr<FunctionSpace> &space,
                                  MultiDomainOp &domains) {
            auto mesh = space->mesh_ptr();
            const bool needs_jacobian_aos =
                    %(gradient_affine_uses_jacobian_aos)s ||
                    %(apply_affine_uses_jacobian_aos)s;
            for (auto &entry : domains.domains()) {
                const smesh::block_idx_t block_id =
                        block_id_for_domain(*mesh, *entry.second.block);
                auto cache = std::make_shared<AffineGeometryCache>();
                cache->jacobian_soa = smesh::JacobianAdjugateAndDeterminant::create_SoA(
                        mesh, smesh::MEMORY_SPACE_HOST, block_id);
                if (!cache->jacobian_soa) {
                    return SFEM_FAILURE;
                }
                if (needs_jacobian_aos) {
                    cache->jacobian_aos = smesh::JacobianAdjugateAndDeterminant::create_AoS(
                            mesh, smesh::MEMORY_SPACE_HOST, block_id);
                    if (!cache->jacobian_aos) {
                        return SFEM_FAILURE;
                    }
                }
                entry.second.user_data = std::static_pointer_cast<void>(cache);
            }
            return SFEM_SUCCESS;
        }
    }  // namespace

    class %(op)s::Impl {
    public:
        explicit Impl(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        std::shared_ptr<FunctionSpace> space;
        std::shared_ptr<MultiDomainOp> domains;
        std::unique_ptr<real_t[]> element_values;
        ptrdiff_t element_capacity{0};
        bool objective_uses_affine{false};
        bool gradient_uses_affine{false};
        bool apply_uses_affine{false};
        bool use_packed_two_pass{false};
        std::vector<SharedBuffer<real_t>> packed_ghost_buf;
    };

    std::unique_ptr<Op> %(op)s::create(const std::shared_ptr<FunctionSpace> &space) {
        if (space->block_size() != space->mesh_ptr()->spatial_dimension()) {
            SFEM_ERROR("%(op)s requires block_size=spatial_dimension\\n");
            return nullptr;
        }
        auto op = std::make_unique<%(op)s>(space);
        op->initialize();
        return op;
    }

    %(op)s::%(op)s(const std::shared_ptr<FunctionSpace> &space)
        : impl_(std::make_unique<Impl>(space)) {}
    %(op)s::~%(op)s() = default;

    ptrdiff_t %(op)s::n_dofs_domain() const { return impl_->space->n_dofs(); }
    ptrdiff_t %(op)s::n_dofs_image() const { return impl_->space->n_dofs(); }

%(performance_methods)s

    int %(op)s::initialize(const std::vector<std::string> &block_names) {
        SFEM_TRACE_SCOPE("%(op)s::initialize");
        impl_->domains = std::make_shared<MultiDomainOp>(impl_->space, block_names);
        auto mesh = impl_->space->mesh_ptr();
        const bool needs_affine_geometry =
                impl_->objective_uses_affine ||
                impl_->gradient_uses_affine ||
                impl_->apply_uses_affine;
        for (auto &entry : impl_->domains->domains()) {
            seed_parameters(*entry.second.parameters);
            impl_->element_capacity =
                    std::max(impl_->element_capacity, entry.second.block->n_elements());
            if (needs_affine_geometry) {
                const smesh::block_idx_t block_id =
                        block_id_for_domain(*mesh, *entry.second.block);
                auto cache = std::make_shared<AffineGeometryCache>();
                cache->jacobian_soa = smesh::JacobianAdjugateAndDeterminant::create_SoA(
                        mesh, smesh::MEMORY_SPACE_HOST, block_id);
                if (!cache->jacobian_soa) {
                    return SFEM_FAILURE;
                }
                if ((impl_->gradient_uses_affine && %(gradient_affine_uses_jacobian_aos)s) ||
                    (impl_->apply_uses_affine && %(apply_affine_uses_jacobian_aos)s)) {
                    cache->jacobian_aos = smesh::JacobianAdjugateAndDeterminant::create_AoS(
                            mesh, smesh::MEMORY_SPACE_HOST, block_id);
                    if (!cache->jacobian_aos) {
                        return SFEM_FAILURE;
                    }
                }
                entry.second.user_data = std::static_pointer_cast<void>(cache);
            }
        }
        impl_->element_values.reset(new real_t[impl_->element_capacity]);
%(packed_scratch_prealloc)s
        return SFEM_SUCCESS;
    }

    int %(op)s::gradient(const real_t *const x, real_t *const out) {
        SFEM_TRACE_SCOPE("%(op)s::gradient");
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const geom_t *const *adjugate = nullptr;
            const geom_t *adjugate_aos = nullptr;
            const geom_t *determinant = nullptr;
            if (impl_->gradient_uses_affine) {
                auto cache = std::static_pointer_cast<AffineGeometryCache>(
                        domain.user_data);
                if (!cache || !cache->jacobian_soa) {
                    SFEM_ERROR("%(op)s affine gradient requires cached geometry\\n");
                    return SFEM_FAILURE;
                }
                adjugate = reinterpret_cast<const geom_t *const *>(
                        cache->jacobian_soa->jacobian_adjugate_SoA()->data());
                determinant = reinterpret_cast<const geom_t *>(
                        cache->jacobian_soa->jacobian_determinant()->data());
                if (%(gradient_affine_uses_jacobian_aos)s) {
                    if (!cache->jacobian_aos) {
                        SFEM_ERROR("%(op)s affine gradient requires cached AoS geometry\\n");
                        return SFEM_FAILURE;
                    }
                    adjugate_aos = reinterpret_cast<const geom_t *>(
                            cache->jacobian_aos->jacobian_adjugate_AoS()->data());
                    determinant = reinterpret_cast<const geom_t *>(
                            cache->jacobian_aos->jacobian_determinant()->data());
                }
            }
%(gradient_packed_dispatch_body)s
            switch (domain.element_type) {
%(gradient_cases)s
                default:
                    SFEM_ERROR("%(op)s does not support element type %%d\\n",
                               domain.element_type);
                    return SFEM_FAILURE;
            }
        });
    }

    int %(op)s::apply(const real_t *const x,
                      const real_t *const h,
                      real_t *const out) {
        SFEM_TRACE_SCOPE("%(op)s::apply");
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const geom_t *const *adjugate = nullptr;
            const geom_t *adjugate_aos = nullptr;
            const geom_t *determinant = nullptr;
            if (impl_->apply_uses_affine) {
                auto cache = std::static_pointer_cast<AffineGeometryCache>(
                        domain.user_data);
                if (!cache || !cache->jacobian_soa) {
                    SFEM_ERROR("%(op)s affine hessian action requires cached geometry\\n");
                    return SFEM_FAILURE;
                }
                adjugate = reinterpret_cast<const geom_t *const *>(
                        cache->jacobian_soa->jacobian_adjugate_SoA()->data());
                determinant = reinterpret_cast<const geom_t *>(
                        cache->jacobian_soa->jacobian_determinant()->data());
                if (%(apply_affine_uses_jacobian_aos)s) {
                    if (!cache->jacobian_aos) {
                        SFEM_ERROR("%(op)s affine hessian action requires cached AoS geometry\\n");
                        return SFEM_FAILURE;
                    }
                    adjugate_aos = reinterpret_cast<const geom_t *>(
                            cache->jacobian_aos->jacobian_adjugate_AoS()->data());
                    determinant = reinterpret_cast<const geom_t *>(
                            cache->jacobian_aos->jacobian_determinant()->data());
                }
            }
%(apply_dispatch_body)s
        });
    }

    int %(op)s::value(const real_t *x, real_t *const out) {
        SFEM_TRACE_SCOPE("%(op)s::value");
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        *out = 0;
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const ptrdiff_t nelements = domain.block->n_elements();
            const geom_t *const *adjugate = nullptr;
            const geom_t *determinant = nullptr;
            if (impl_->objective_uses_affine) {
                auto cache = std::static_pointer_cast<AffineGeometryCache>(
                        domain.user_data);
                if (!cache || !cache->jacobian_soa) {
                    SFEM_ERROR("%(op)s affine objective requires cached geometry\\n");
                    return SFEM_FAILURE;
                }
                adjugate = reinterpret_cast<const geom_t *const *>(
                        cache->jacobian_soa->jacobian_adjugate_SoA()->data());
                determinant = reinterpret_cast<const geom_t *>(
                        cache->jacobian_soa->jacobian_determinant()->data());
            }
            std::fill(impl_->element_values.get(),
                      impl_->element_values.get() + nelements,
                      0);
            int status = SFEM_FAILURE;
            switch (domain.element_type) {
%(objective_cases)s
                default:
                    SFEM_ERROR("%(op)s does not support element type %%d\\n",
                               domain.element_type);
                    return SFEM_FAILURE;
            }
            if (status != SFEM_SUCCESS) return status;
            real_t sum = 0;
#pragma omp simd reduction(+ : sum)
            for (ptrdiff_t element = 0; element < nelements; ++element) {
                sum += impl_->element_values[element];
            }
            *out += sum;
            return SFEM_SUCCESS;
        });
    }

    int %(op)s::value_steps(const real_t *x,
                            const real_t *h,
                            const int nsteps,
                            const real_t *const steps,
                            real_t *const out) {
        SFEM_TRACE_SCOPE("%(op)s::value_steps");
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        if (nsteps <= 0) {
            return SFEM_SUCCESS;
        }
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const ptrdiff_t nelements = domain.block->n_elements();
            const ptrdiff_t nvalues = (ptrdiff_t)nsteps * nelements;
            const geom_t *const *adjugate = nullptr;
            const geom_t *determinant = nullptr;
            if (impl_->objective_uses_affine) {
                auto cache = std::static_pointer_cast<AffineGeometryCache>(
                        domain.user_data);
                if (!cache || !cache->jacobian_soa) {
                    SFEM_ERROR("%(op)s affine objective_steps requires cached geometry\\n");
                    return SFEM_FAILURE;
                }
                adjugate = reinterpret_cast<const geom_t *const *>(
                        cache->jacobian_soa->jacobian_adjugate_SoA()->data());
                determinant = reinterpret_cast<const geom_t *>(
                        cache->jacobian_soa->jacobian_determinant()->data());
            }
            if (nvalues > impl_->element_capacity) {
                impl_->element_values.reset(new real_t[nvalues]);
                impl_->element_capacity = nvalues;
            }
            std::fill(impl_->element_values.get(),
                      impl_->element_values.get() + nvalues,
                      real_t(0));
            int status = SFEM_FAILURE;
%(objective_steps_packed_dispatch_body)s
            if (status == SFEM_FAILURE) {
            switch (domain.element_type) {
%(objective_steps_cases)s
                default:
                    SFEM_ERROR("%(op)s does not support element type %%d\\n",
                               domain.element_type);
                    return SFEM_FAILURE;
            }
            }
            if (status != SFEM_SUCCESS) return status;
            for (int step = 0; step < nsteps; ++step) {
                real_t sum = 0;
#pragma omp simd reduction(+ : sum)
                for (ptrdiff_t element = 0; element < nelements; ++element) {
                    sum += impl_->element_values[(ptrdiff_t)step * nelements + element];
                }
                out[step] += sum;
            }
            return SFEM_SUCCESS;
        });
    }

    int %(op)s::hessian_crs(const real_t *const x,
                            const count_t *const rowptr,
                            const idx_t *const colidx,
                            real_t *const values) {
        SFEM_TRACE_SCOPE("%(op)s::hessian_crs");
        const real_t *const current = x;
        if (!current) {
            SFEM_ERROR("%(op)s::hessian_crs requires a current state\\n");
            return SFEM_FAILURE;
        }
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
%(hessian_crs_dispatch_body)s
        });
    }

    int %(op)s::hessian_bsr(const real_t *const x,
                            const count_t *const rowptr,
                            const idx_t *const colidx,
                            real_t *const values) {
        SFEM_TRACE_SCOPE("%(op)s::hessian_bsr");
        const real_t *const current = x;
        if (!current) {
            SFEM_ERROR("%(op)s::hessian_bsr requires a current state\\n");
            return SFEM_FAILURE;
        }
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
%(hessian_bsr_dispatch_body)s
        });
    }

    int %(op)s::hessian_dia(const real_t *const x,
                            const int *const diag_offsets,
                            const ptrdiff_t ndiag,
                            real_t *const values) {
        SFEM_TRACE_SCOPE("%(op)s::hessian_dia");
        const real_t *const current = x;
        if (!current) {
            SFEM_ERROR("%(op)s::hessian_dia requires a current state\\n");
            return SFEM_FAILURE;
        }
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
%(hessian_dia_dispatch_body)s
        });
    }

    int %(op)s::hessian_coo(const real_t *const x,
                            const ptrdiff_t nnz,
                            const idx_t *const rows,
                            const idx_t *const cols,
                            real_t *const values) {
        SFEM_TRACE_SCOPE("%(op)s::hessian_coo");
        const real_t *const current = x;
        if (!current) {
            SFEM_ERROR("%(op)s::hessian_coo requires a current state\\n");
            return SFEM_FAILURE;
        }
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
%(hessian_coo_dispatch_body)s
        });
    }

    int %(op)s::hessian_patch(const real_t *const x,
                              const count_t *const rowptr,
                              const idx_t *const colidx,
                              real_t *const values) {
        SFEM_TRACE_SCOPE("%(op)s::hessian_patch");
        const real_t *const current = x;
        if (!current) {
            SFEM_ERROR("%(op)s::hessian_patch requires a current state\\n");
            return SFEM_FAILURE;
        }
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
%(hessian_patch_dispatch_body)s
        });
    }

    void %(op)s::set_option(const std::string &name, const bool val) {
        SFEM_TRACE_SCOPE("%(op)s::set_option");
        AffineOption options[] = {
%(affine_options)s
        };
        const bool matched = set_affine_option(name, val, options, sizeof(options) / sizeof(options[0]));
        if (matched && val && impl_->domains) {
            if (cache_affine_geometry(impl_->space, *impl_->domains) != SFEM_SUCCESS) {
                SFEM_ERROR("%(op)s failed to cache affine geometry\\n");
            }
        }
    }

    void %(op)s::set_value_in_block(const std::string &block_name,
                                    const std::string &var_name,
                                    const real_t value) {
        SFEM_TRACE_SCOPE("%(op)s::set_value_in_block");
        impl_->domains->set_value_in_block(block_name, var_name, value);
    }

#ifdef SFEM_ENABLE_RYAML
    std::shared_ptr<Op> %(op)s::create_from_yaml(const std::shared_ptr<FunctionSpace> &space,
                                                 const ryml::ConstNodeRef             &node) {
        SFEM_TRACE_SCOPE("%(op)s::create_from_yaml");
        auto ret = std::make_shared<%(op)s>(space);

        std::vector<std::string> block_names;
        if (node.has_child("blocks")) {
            for (auto block : node["blocks"].children()) {
                if (block.has_child("name")) {
                    block_names.push_back(yaml_read_string(block["name"]));
                }
            }
        }

        AffineOption options[] = {
%(yaml_affine_options)s
        };
        read_affine_options(node, options, sizeof(options) / sizeof(options[0]));

        if (ret->initialize(block_names) != SFEM_SUCCESS) {
            return nullptr;
        }

        real_t defaults[N_MATERIAL_PARAMETERS];
        material_defaults(defaults);
        real_t top_values[N_MATERIAL_PARAMETERS];
        copy_material_parameters(defaults, top_values);
        if (material_from_yaml(node, defaults, top_values)) {
            set_material(*ret->impl_->domains, top_values);
        }

        if (node.has_child("blocks")) {
            for (auto block : node["blocks"].children()) {
                if (!block.has_child("name")) {
                    continue;
                }

                real_t block_values[N_MATERIAL_PARAMETERS];
                copy_material_parameters(top_values, block_values);
                if (!material_from_yaml(block, top_values, block_values)) {
                    continue;
                }

                const std::string block_name = yaml_read_string(block["name"]);
                set_material_in_block(*ret->impl_->domains, block_name, block_values);
            }
        }

        return ret;
    }
#endif  // SFEM_ENABLE_RYAML
}  // namespace sfem
""" % {
        "op": material.op_name,
        "c_abi_include": '#include "%s"' % c_abi_header if c_abi_header else "",
        "packed_scratch_include": packed_scratch_include,
        "packed_scratch_prealloc": packed_scratch_prealloc,
        "declaration_block": (
            'extern "C" {\n%s\n}' % "\n".join(declarations)
            if declarations
            else ""
        ),
        "declarations": "\n".join(declarations),
        "defaults": defaults,
        "yaml_helpers": _yaml_helpers(material.parameter_defaults),
        "gradient_affine_uses_jacobian_aos": _cpp_bool(any(gradient_affine_aos_flags)),
        "apply_affine_uses_jacobian_aos": _cpp_bool(any(apply_affine_aos_flags)),
        "gradient_cases": "\n".join(gradient_cases),
        "apply_cases": "\n".join(apply_cases),
        "objective_cases": "\n".join(objective_cases),
        "objective_steps_cases": "\n".join(objective_steps_cases),
        "hessian_crs_cases": "\n".join(hessian_crs_cases),
        "hessian_bsr_cases": "\n".join(hessian_bsr_cases),
        "hessian_dia_cases": "\n".join(hessian_dia_cases),
        "hessian_coo_cases": "\n".join(hessian_coo_cases),
        "hessian_patch_cases": "\n".join(hessian_patch_cases),
        "apply_dispatch_body": _hyperelastic_apply_dispatch_body(
            material.name,
            kernel_sources,
            {dim: deps[2] for dim, deps in dependencies_by_dim.items()},
            indent="            ",
        ),
        "gradient_packed_dispatch_body": _hyperelastic_gradient_packed_dispatch_body(
            material.name,
            kernel_sources,
            {dim: deps[1] for dim, deps in dependencies_by_dim.items()},
            indent="            ",
        ),
        "objective_steps_packed_dispatch_body": _hyperelastic_objective_steps_packed_dispatch_body(
            material.name,
            kernel_sources,
            {dim: deps[0] for dim, deps in dependencies_by_dim.items()},
            indent="            ",
        ),
        "hessian_crs_dispatch_body": _hyperelastic_hessian_dispatch_body(
            material.name,
            "hessian_crs",
            kernel_sources,
            {dim: deps[2] for dim, deps in dependencies_by_dim.items()},
            ("rowptr", "colidx", "values"),
            indent="            ",
        ),
        "hessian_bsr_dispatch_body": _hyperelastic_hessian_dispatch_body(
            material.name,
            "hessian_bsr",
            kernel_sources,
            {dim: deps[2] for dim, deps in dependencies_by_dim.items()},
            ("rowptr", "colidx", "values"),
            indent="            ",
        ),
        "hessian_dia_dispatch_body": _hyperelastic_hessian_dispatch_body(
            material.name,
            "hessian_dia",
            kernel_sources,
            {dim: deps[2] for dim, deps in dependencies_by_dim.items()},
            ("diag_offsets", "ndiag", "values"),
            indent="            ",
        ),
        "hessian_coo_dispatch_body": _hyperelastic_hessian_dispatch_body(
            material.name,
            "hessian_coo",
            kernel_sources,
            {dim: deps[2] for dim, deps in dependencies_by_dim.items()},
            ("nnz", "rows", "cols", "values"),
            indent="            ",
        ),
        "hessian_patch_dispatch_body": _hyperelastic_hessian_dispatch_body(
            material.name,
            "hessian_patch",
            kernel_sources,
            {dim: deps[2] for dim, deps in dependencies_by_dim.items()},
            ("rowptr", "colidx", "values"),
            indent="            ",
        ),
        "performance_methods": _performance_methods(material.op_name, performance_cases),
        "affine_options": _affine_option_entries(
            "objective_uses_affine",
            "gradient_uses_affine",
            "apply_uses_affine",
        ),
        "yaml_affine_options": _affine_option_entries(
            "objective_uses_affine",
            "gradient_uses_affine",
            "apply_uses_affine",
            owner="ret->impl_",
        ),
    }
    return _header(material, False), source


def _residual_op(material, elements, c_abi_header=None, form_collections=None, kernel_sources=None):
    if form_collections is None:
        raise ValueError("residual generated Op requires form collections")
    measures = {collection.measure for collection in form_collections.values()}
    if measures == {"ds"}:
        return _boundary_residual_op(material, elements, c_abi_header, form_collections)
    if "ds" in measures:
        raise ValueError("generated residual Op wrappers cannot mix dx and ds forms yet")

    defaults = _seed_lines(material.parameter_defaults)
    declarations = []
    residual_cases = []
    action_cases = []
    hessian_crs_cases = []
    hessian_bsr_cases = []
    hessian_dia_cases = []
    performance_cases = {"value": [], "gradient": [], "apply": []}
    dependencies_by_dim = {}
    parameter_names_by_dim = {}
    fields_by_dim = {}
    block_size_by_dim = {}
    residual_affine_metric_flags = []
    action_affine_metric_flags = []
    residual_affine_metric_soa_flags = []
    action_affine_metric_soa_flags = []
    residual_affine_metric_aos_flags = []
    action_affine_metric_aos_flags = []
    residual_affine_metric_aos_elements_by_dim = {}
    action_affine_metric_aos_elements_by_dim = {}
    residual_affine_metric_aos_unit_elements_by_dim = {}
    action_affine_metric_aos_unit_elements_by_dim = {}
    for element in elements:
        dim = _element_dim(element)
        dependencies = dependencies_by_dim.get(dim)
        if dependencies is None:
            collection = form_collections[dim]
            system = collection.source
            dependencies = (
                collection.form_metadata(_form_order_one()).dependencies,
                collection.form_metadata(_form_order_two()).dependencies,
            )
            dependencies_by_dim[dim] = dependencies
            parameter_names_by_dim[dim] = tuple(str(symbol) for symbol in system.parameters)
            fields_by_dim[dim] = tuple(collection.fields)
            block_size_by_dim[dim] = sum(int(field.components) for field in collection.fields)
        residual_dependencies, action_dependencies = dependencies
        parameter_index = {
            name: index for index, name in enumerate(parameter_names_by_dim[dim])
        }
        stem = "%s_%s" % (material.name, _element_name(element).lower())
        performance_cases["gradient"].append(
            _performance_case(
                element,
                ("%s_residual_element_soa_diagnostics" % stem,),
                affine_flags=("residual_uses_affine",),
            )
        )
        performance_cases["apply"].append(
            _performance_case(
                element,
                ("%s_jacobian_action_element_soa_diagnostics" % stem,),
                affine_flags=("jacobian_action_uses_affine",),
            )
        )
        residual_pointer_params = []
        if residual_dependencies.current:
            residual_pointer_params.append("const real_t *")
        if residual_dependencies.previous:
            residual_pointer_params.append("const real_t *")
        declarations.append(
            "int %s_residual_isoparametric_mesh_aos("
            "ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, "
            "const real_t *, %sreal_t *);"
            % (stem, "".join("%s, " % param for param in residual_pointer_params))
        )
        action_pointer_params = []
        if action_dependencies.current:
            action_pointer_params.append("const real_t *")
        if action_dependencies.previous:
            action_pointer_params.append("const real_t *")
        if action_dependencies.direction:
            action_pointer_params.append("const real_t *")
        declarations.append(
            "int %s_jacobian_action_isoparametric_mesh_aos("
            "ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, "
            "const real_t *, %sreal_t *);"
            % (stem, "".join("%s, " % param for param in action_pointer_params))
        )
        common_isoparametric = (
            "domain.block->n_elements(), mesh->n_nodes(), "
            "domain.block->elements()->data(), points"
        )
        residual_affine_uses_metric = _c_abi_function_uses_cached_metric(
            kernel_sources, "%s_residual_affine_mesh_soa" % stem
        )
        action_affine_uses_metric = _c_abi_function_uses_cached_metric(
            kernel_sources, "%s_jacobian_action_affine_mesh_soa" % stem
        )
        residual_affine_uses_metric_aos = _c_abi_function_exists(
            kernel_sources, "%s_residual_affine_mesh_soa_aos" % stem
        )
        action_affine_uses_metric_aos = _c_abi_function_exists(
            kernel_sources, "%s_jacobian_action_affine_mesh_soa_aos" % stem
        )
        residual_affine_uses_metric_aos_unit = _c_abi_function_exists(
            kernel_sources, "%s_residual_affine_mesh_soa_aos_unit" % stem
        )
        action_affine_uses_metric_aos_unit = _c_abi_function_exists(
            kernel_sources, "%s_jacobian_action_affine_mesh_soa_aos_unit" % stem
        )
        residual_affine_metric_flags.append(residual_affine_uses_metric)
        action_affine_metric_flags.append(action_affine_uses_metric)
        residual_affine_metric_soa_flags.append(
            residual_affine_uses_metric and not residual_affine_uses_metric_aos
        )
        action_affine_metric_soa_flags.append(
            action_affine_uses_metric and not action_affine_uses_metric_aos
        )
        residual_affine_metric_aos_flags.append(residual_affine_uses_metric_aos)
        action_affine_metric_aos_flags.append(action_affine_uses_metric_aos)
        mesh_element = _mesh_element_name(element)
        if residual_affine_uses_metric_aos:
            residual_affine_metric_aos_elements_by_dim.setdefault(dim, []).append(mesh_element)
        if action_affine_uses_metric_aos:
            action_affine_metric_aos_elements_by_dim.setdefault(dim, []).append(mesh_element)
        if residual_affine_uses_metric_aos_unit:
            residual_affine_metric_aos_unit_elements_by_dim.setdefault(dim, []).append(mesh_element)
        if action_affine_uses_metric_aos_unit:
            action_affine_metric_aos_unit_elements_by_dim.setdefault(dim, []).append(mesh_element)
        common_affine_residual = (
            "domain.block->n_elements(), mesh->n_nodes(), "
            "domain.block->elements()->data(), %s"
            % (
                _affine_metric_offsets(dim)
                if residual_affine_uses_metric
                else "%s, determinant" % _affine_geometry_offsets(dim)
            )
        )
        common_affine_residual_aos = (
            "domain.block->n_elements(), mesh->n_nodes(), "
            "domain.block->elements()->data(), geom_metric_aos"
        )
        common_affine_action = (
            "domain.block->n_elements(), mesh->n_nodes(), "
            "domain.block->elements()->data(), %s"
            % (
                _affine_metric_offsets(dim)
                if action_affine_uses_metric
                else "%s, determinant" % _affine_geometry_offsets(dim)
            )
        )
        common_affine_action_aos = (
            "domain.block->n_elements(), mesh->n_nodes(), "
            "domain.block->elements()->data(), geom_metric_aos"
        )
        residual_common_args = []
        residual_common_args.extend(
            _dependency_storage_args(residual_dependencies.parameters, parameter_index)
        )
        residual_setup = []
        if residual_dependencies.current:
            residual_setup.extend(
                _residual_soa_view_declarations(
                    fields_by_dim[dim],
                    "state",
                    "data",
                    "const real_t",
                )
            )
            residual_common_args.append("FIELD_STRIDE")
            residual_common_args.extend(
                _residual_soa_field_argument_names(fields_by_dim[dim], "data")
            )
        if residual_dependencies.previous:
            residual_setup.extend(
                _residual_soa_view_declarations(
                    fields_by_dim[dim],
                    "previous",
                    "old_data",
                    "const real_t",
                )
            )
            residual_common_args.append("FIELD_STRIDE")
            residual_common_args.extend(
                _residual_soa_field_argument_names(fields_by_dim[dim], "old_data")
            )
        residual_setup.extend(
            _residual_soa_view_declarations(
                fields_by_dim[dim],
                "out",
                "out",
                "real_t",
            )
        )
        residual_common_args.append("FIELD_STRIDE")
        residual_common_args.extend(_residual_soa_field_argument_names(fields_by_dim[dim], "out"))
        residual_unit_args = []
        if residual_dependencies.current:
            residual_unit_args.extend(
                _residual_soa_field_argument_names(fields_by_dim[dim], "data")
            )
        residual_unit_args.extend(_residual_soa_field_argument_names(fields_by_dim[dim], "out"))
        residual_cases.append(
            _residual_dual_soa_case(
                element,
                "residual_uses_affine",
                (
                    "%s_residual_affine_mesh_soa_aos" % stem
                    if residual_affine_uses_metric_aos
                    else "%s_residual_affine_mesh_soa" % stem
                ),
                ", ".join(
                    (
                        common_affine_residual_aos
                        if residual_affine_uses_metric_aos
                        else common_affine_residual,
                        *residual_common_args,
                    )
                ),
                "%s_residual_isoparametric_mesh_soa" % stem,
                ", ".join((common_isoparametric, *residual_common_args)),
                block_size_by_dim[dim],
                residual_setup,
                affine_unit_function=(
                    "%s_residual_affine_mesh_soa_aos_unit" % stem
                    if residual_affine_uses_metric_aos_unit
                    else None
                ),
                affine_unit_arguments=(
                    ", ".join((common_affine_residual_aos, *residual_unit_args))
                    if residual_affine_uses_metric_aos_unit
                    else None
                ),
                affine_unit_condition=(
                    "storage[0] == real_t(1)"
                    if residual_affine_uses_metric_aos_unit
                    else None
                ),
            )
        )
        action_common_args = []
        action_common_args.extend(
            _dependency_storage_args(action_dependencies.parameters, parameter_index)
        )
        action_setup = []
        if action_dependencies.current:
            action_setup.extend(
                _residual_soa_view_declarations(
                    fields_by_dim[dim],
                    "current",
                    "data",
                    "const real_t",
                )
            )
            action_common_args.append("FIELD_STRIDE")
            action_common_args.extend(
                _residual_soa_field_argument_names(fields_by_dim[dim], "data")
            )
        if action_dependencies.previous:
            action_setup.extend(
                _residual_soa_view_declarations(
                    fields_by_dim[dim],
                    "previous",
                    "old_data",
                    "const real_t",
                )
            )
            action_common_args.append("FIELD_STRIDE")
            action_common_args.extend(
                _residual_soa_field_argument_names(fields_by_dim[dim], "old_data")
            )
        if action_dependencies.direction:
            action_setup.extend(
                _residual_soa_view_declarations(
                    fields_by_dim[dim],
                    "direction",
                    "direction_data",
                    "const real_t",
                )
            )
            action_common_args.append("FIELD_STRIDE")
            action_common_args.extend(
                _residual_soa_field_argument_names(fields_by_dim[dim], "direction_data")
            )
        action_setup.extend(
            _residual_soa_view_declarations(
                fields_by_dim[dim],
                "out",
                "out",
                "real_t",
            )
        )
        action_common_args.append("FIELD_STRIDE")
        action_common_args.extend(_residual_soa_field_argument_names(fields_by_dim[dim], "out"))
        action_unit_args = []
        if action_dependencies.direction:
            action_unit_args.extend(
                _residual_soa_field_argument_names(fields_by_dim[dim], "direction_data")
            )
        action_unit_args.extend(_residual_soa_field_argument_names(fields_by_dim[dim], "out"))
        action_cases.append(
            _residual_dual_soa_case(
                element,
                "jacobian_action_uses_affine",
                (
                    "%s_jacobian_action_affine_mesh_soa_aos" % stem
                    if action_affine_uses_metric_aos
                    else "%s_jacobian_action_affine_mesh_soa" % stem
                ),
                ", ".join(
                    (
                        common_affine_action_aos
                        if action_affine_uses_metric_aos
                        else common_affine_action,
                        *action_common_args,
                    )
                ),
                "%s_jacobian_action_isoparametric_mesh_soa" % stem,
                ", ".join((common_isoparametric, *action_common_args)),
                block_size_by_dim[dim],
                action_setup,
                affine_unit_function=(
                    "%s_jacobian_action_affine_mesh_soa_aos_unit" % stem
                    if action_affine_uses_metric_aos_unit
                    else None
                ),
                affine_unit_arguments=(
                    ", ".join((common_affine_action_aos, *action_unit_args))
                    if action_affine_uses_metric_aos_unit
                    else None
                ),
                affine_unit_condition=(
                    "storage[0] == real_t(1)"
                    if action_affine_uses_metric_aos_unit
                    else None
                ),
            )
        )
        hessian_common_args = []
        hessian_common_args.extend(
            _dependency_storage_args(action_dependencies.parameters, parameter_index)
        )
        hessian_setup = []
        if action_dependencies.current:
            hessian_setup.extend(
                _residual_soa_view_declarations(
                    fields_by_dim[dim],
                    "current",
                    "data",
                    "const real_t",
                )
            )
            hessian_common_args.append("FIELD_STRIDE")
            hessian_common_args.extend(
                _residual_soa_field_argument_names(fields_by_dim[dim], "data")
            )
        if action_dependencies.previous:
            hessian_setup.extend(
                _residual_soa_view_declarations(
                    fields_by_dim[dim],
                    "previous",
                    "old_data",
                    "const real_t",
                )
            )
            hessian_common_args.append("FIELD_STRIDE")
            hessian_common_args.extend(
                _residual_soa_field_argument_names(fields_by_dim[dim], "old_data")
            )
        hessian_crs_function = "%s_hessian_crs_isoparametric_mesh_soa" % stem
        if _c_abi_function_exists(kernel_sources, hessian_crs_function):
            hessian_crs_cases.append(
                _residual_soa_case(
                    element,
                    hessian_crs_function,
                    ", ".join(
                        (
                            common_isoparametric,
                            *hessian_common_args,
                            "rowptr",
                            "colidx",
                            "values",
                        )
                    ),
                    block_size_by_dim[dim],
                    hessian_setup,
                )
            )
        hessian_bsr_function = "%s_hessian_bsr_isoparametric_mesh_soa" % stem
        if _c_abi_function_exists(kernel_sources, hessian_bsr_function):
            hessian_bsr_cases.append(
                _residual_soa_case(
                    element,
                    hessian_bsr_function,
                    ", ".join(
                        (
                            common_isoparametric,
                            *hessian_common_args,
                            "rowptr",
                            "colidx",
                            "values",
                        )
                    ),
                    block_size_by_dim[dim],
                    hessian_setup,
                )
            )
        hessian_dia_function = "%s_hessian_dia_isoparametric_mesh_soa" % stem
        if _c_abi_function_exists(kernel_sources, hessian_dia_function):
            hessian_dia_cases.append(
                _case(
                    element,
                    hessian_dia_function,
                    ", ".join(
                        (
                            common_isoparametric,
                            *_dependency_storage_args(
                                action_dependencies.parameters,
                                parameter_index,
                            ),
                            "diag_offsets",
                            "ndiag",
                            "values",
                        )
                    ),
                )
            )

    residual_uses_previous = any(
        dependencies[0].previous for dependencies in dependencies_by_dim.values()
    )
    action_uses_current = any(
        dependencies[1].current for dependencies in dependencies_by_dim.values()
    )
    action_uses_previous = any(
        dependencies[1].previous for dependencies in dependencies_by_dim.values()
    )
    hessian_state_alias = (
        "        const real_t *const current = state ? state : impl_->current;"
        if action_uses_current
        else ""
    )
    hessian_state_check = (
        "        if (%s) {\n"
        '            SFEM_ERROR("%s requires %s\\n");\n'
        "            return SFEM_FAILURE;\n"
        "        }"
        % (
            " || ".join(
                condition
                for condition in (
                    "!current" if action_uses_current else "",
                    "!impl_->previous" if action_uses_previous else "",
                )
                if condition
            ),
            material.op_name,
            (
                "current and previous states"
                if action_uses_current and action_uses_previous
                else (
                    "a current state"
                    if action_uses_current
                    else "a previous state"
                )
            ),
        )
        if action_uses_current or action_uses_previous
        else ""
    )
    hessian_previous_alias = (
        "            const real_t *const previous = impl_->previous;"
        if action_uses_previous
        else ""
    )
    residual_affine_uses_metric = any(residual_affine_metric_flags)
    action_affine_uses_metric = any(action_affine_metric_flags)
    residual_affine_uses_metric_soa = any(residual_affine_metric_soa_flags)
    action_affine_uses_metric_soa = any(action_affine_metric_soa_flags)
    action_packed_affine_uses_metric_soa = any(
        _c_abi_function_uses_cached_metric(
            kernel_sources,
            "%s_jacobian_action_packed_%dd_affine_mesh_soa" % (material.name, dim),
        )
        for dim in (2, 3)
    )
    laplace_tet4_packed_affine_uses_metric_soa = (
        material.name == "laplace"
        and _c_abi_function_exists(
            kernel_sources,
            "laplace_tet4_jacobian_action_packed_affine_mesh_soa",
        )
    )
    laplace_proteus_hex8_packed_metric = (
        material.name == "laplace"
        and _c_abi_function_exists(
            kernel_sources,
            "laplace_proteus_hex8_private_metric_jacobian_action_packed_mesh_soa",
        )
    )
    laplace_tet10_packed_metric = (
        material.name == "laplace"
        and _c_abi_function_exists(
            kernel_sources,
            "laplace_tet10_private_metric_jacobian_action_packed_mesh_soa",
        )
    )
    action_affine_uses_metric_soa = (
        action_affine_uses_metric_soa
        or action_packed_affine_uses_metric_soa
        or laplace_tet4_packed_affine_uses_metric_soa
    )
    residual_affine_uses_metric_aos = any(residual_affine_metric_aos_flags)
    action_affine_uses_metric_aos = (
        any(action_affine_metric_aos_flags)
        or laplace_proteus_hex8_packed_metric
        or laplace_tet10_packed_metric
    )
    residual_affine_uses_jacobian = not all(residual_affine_metric_flags)
    action_affine_uses_jacobian = not all(action_affine_metric_flags)

    max_parameters = max(
        1,
        len(material.parameter_defaults),
        *(len(names) for names in parameter_names_by_dim.values()),
    )
    parameter_lines = _residual_parameter_array_lines(parameter_names_by_dim)
    laplace_packed_apply = material.name == "laplace"
    generated_packed_apply = any("_jacobian_action_packed_" in source for source in kernel_sources.values())
    use_laplace_packed_fast_path = laplace_packed_apply and not generated_packed_apply
    laplace_packed_include = '#include "sfem_PackedLaplacian.hpp"' if use_laplace_packed_fast_path else ""
    packed_scratch_include = '#include "packed_thread_scratch.hpp"' if generated_packed_apply else ""
    packed_scratch_prealloc = (
        """        if (impl_->space->has_packed_mesh()) {
            auto packed = impl_->space->packed_mesh();
            const ptrdiff_t max_nodes_per_pack = packed->max_nodes_per_pack();
            const int dim = impl_->space->mesh_ptr()->spatial_dimension();
            sfem::codegen::prealloc_thread_scratch<real_t>(
                    0, (size_t)dim * (size_t)max_nodes_per_pack);
            sfem::codegen::prealloc_thread_scratch<real_t>(
                    1, (size_t)max_nodes_per_pack);
            sfem::codegen::prealloc_thread_scratch<real_t>(
                    2, (size_t)max_nodes_per_pack);
            sfem::codegen::prealloc_thread_scratch<real_t>(
                    3, (size_t)max_nodes_per_pack);
        }"""
        if generated_packed_apply
        else ""
    )
    laplace_packed_helpers = (
        """
        bool packed_laplacian_apply_supported(const smesh::ElemType element_type) {
            switch (element_type) {
                case smesh::TET4:
                case smesh::TET10:
                case smesh::HEX8:
                    return true;
                default:
                    return false;
            }
        }

        bool can_use_packed_laplacian_apply(const FunctionSpace &space,
                                            MultiDomainOp &domains) {
            if (!space.has_packed_mesh()) {
                return false;
            }

            for (auto &entry : domains.domains()) {
                const OpDomain &domain = entry.second;
                if (!packed_laplacian_apply_supported(domain.element_type)) {
                    return false;
                }
                if (domain.parameters->require_real_value("kappa") != real_t(1)) {
                    return false;
                }
            }

            return true;
        }
"""
        if laplace_packed_apply
        and use_laplace_packed_fast_path
        else ""
    )
    laplace_packed_member = (
        "        std::shared_ptr<Op> packed_affine_apply;"
        if use_laplace_packed_fast_path
        else ""
    )
    laplace_packed_apply_fast_path = (
        """
        if (impl_->jacobian_action_uses_affine &&
            can_use_packed_laplacian_apply(*impl_->space, *impl_->domains)) {
            if (!impl_->packed_affine_apply) {
                impl_->packed_affine_apply = std::make_shared<PackedLaplacian>(impl_->space);
                if (impl_->packed_affine_apply->initialize() != SFEM_SUCCESS) {
                    SFEM_ERROR("%s failed to initialize packed affine apply backend\\n");
                    return SFEM_FAILURE;
                }
            }
            return impl_->packed_affine_apply->apply(current, direction, out);
        }
"""
        % material.op_name
        if use_laplace_packed_fast_path
        else ""
    )
    private_declarations = []
    if laplace_tet4_packed_affine_uses_metric_soa:
        private_declarations.append(
            """int laplace_tet4_jacobian_action_packed_affine_mesh_soa(
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const SFEM_RESTRICT elements,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx,
        const geom_t *const SFEM_RESTRICT g_geom_metric0,
        const geom_t *const SFEM_RESTRICT g_geom_metric1,
        const geom_t *const SFEM_RESTRICT g_geom_metric2,
        const geom_t *const SFEM_RESTRICT g_geom_metric3,
        const geom_t *const SFEM_RESTRICT g_geom_metric4,
        const geom_t *const SFEM_RESTRICT g_geom_metric5,
        const double kappa,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);"""
        )
    for private_name in (
        "laplace_proteus_hex8_private_metric_jacobian_action_packed_mesh_soa",
        "laplace_tet10_private_metric_jacobian_action_packed_mesh_soa",
    ):
        if material.name == "laplace" and _c_abi_function_exists(kernel_sources, private_name):
            private_declarations.append(
                """int %s(
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const SFEM_RESTRICT elements,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx,
        const geom_t *const SFEM_RESTRICT g_geom_metric,
        const double kappa,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);"""
                % private_name
            )
    if c_abi_header:
        declaration_block = (
            'extern "C" {\n%s\n}' % "\n".join(private_declarations)
            if private_declarations
            else ""
        )
    else:
        declaration_block = 'extern "C" {\n%s\n}' % "\n".join(
            [*declarations, *private_declarations]
        )
    source = """#include "sfem_%(op)s.hpp"
%(c_abi_include)s
%(laplace_packed_include)s
%(packed_scratch_include)s

#include "sfem_FunctionSpace.hpp"
#include "sfem_MultiDomainOp.hpp"
#include "sfem_OpTracer.hpp"
#include "sfem_Parameters.hpp"
#include "smesh_kernel_data.hpp"
#include "smesh_mesh.hpp"

#include <cstring>

%(declaration_block)s

namespace sfem {
    namespace {
        constexpr int MAX_PARAMETERS = %(max_parameters)d;

        void seed_parameters(Parameters &parameters) {
%(defaults)s
        }

        void seed_material(MultiDomainOp &domains) {
            for (auto &entry : domains.domains()) {
                seed_parameters(*entry.second.parameters);
            }
        }

%(yaml_helpers)s

        smesh::block_idx_t block_id_for_domain(const smesh::Mesh &mesh,
                                               const smesh::Mesh::Block &block) {
            for (size_t i = 0; i < mesh.n_blocks(); ++i) {
                if (mesh.block(i).get() == &block) {
                    return static_cast<smesh::block_idx_t>(i);
                }
            }
            SFEM_ERROR("%(op)s: mesh block pointer not found in mesh.blocks()\\n");
            return 0;
        }

        int packed_block_id_for_domain(const FunctionSpace::PackedMesh &packed,
                                       const smesh::Mesh::Block &block) {
            for (ptrdiff_t i = 0; i < packed.n_blocks(); ++i) {
                if (packed.block_name(i) == block.name()) {
                    return static_cast<int>(i);
                }
            }
            return -1;
        }

        struct AffineGeometryCache {
            std::shared_ptr<smesh::JacobianAdjugateAndDeterminant> jacobian;
            std::shared_ptr<smesh::FFF> metric_soa;
            std::shared_ptr<smesh::FFF> metric_aos;
        };

        int cache_affine_geometry(const std::shared_ptr<FunctionSpace> &space,
                                  MultiDomainOp &domains,
                                  const bool needs_jacobian,
                                  const bool needs_metric_soa,
                                  const bool needs_metric_aos) {
            auto mesh = space->mesh_ptr();
            for (auto &entry : domains.domains()) {
                auto cache = std::static_pointer_cast<AffineGeometryCache>(
                        entry.second.user_data);
                if (!cache) {
                    cache = std::make_shared<AffineGeometryCache>();
                }
                const smesh::block_idx_t block_id =
                        block_id_for_domain(*mesh, *entry.second.block);
                if (needs_jacobian && !cache->jacobian) {
                    cache->jacobian = smesh::JacobianAdjugateAndDeterminant::create_SoA(
                            mesh, smesh::MEMORY_SPACE_HOST, block_id);
                    if (!cache->jacobian) {
                        return SFEM_FAILURE;
                    }
                }
                if (needs_metric_soa && !cache->metric_soa) {
                    cache->metric_soa = smesh::FFF::create_SoA(
                            mesh, smesh::MEMORY_SPACE_HOST, block_id);
                    if (!cache->metric_soa) {
                        return SFEM_FAILURE;
                    }
                }
                if (needs_metric_aos && !cache->metric_aos) {
                    cache->metric_aos = smesh::FFF::create_AoS(
                            mesh, smesh::MEMORY_SPACE_HOST, block_id);
                    if (!cache->metric_aos) {
                        return SFEM_FAILURE;
                    }
                }
                entry.second.user_data = std::static_pointer_cast<void>(cache);
            }
            return SFEM_SUCCESS;
        }

        void parameter_array(const Parameters &parameters,
                             const int dim,
                             real_t *const values) {
            int index = 0;
%(parameter_lines)s
        }

        ptrdiff_t block_size_for_dim(const int dim) {
%(block_size_lines)s
        }
%(laplace_packed_helpers)s
    }  // namespace

    class %(op)s::Impl {
    public:
        explicit Impl(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        std::shared_ptr<FunctionSpace> space;
        std::shared_ptr<MultiDomainOp> domains;
%(laplace_packed_member)s
        std::shared_ptr<Buffer<real_t>> previous_buffer;
        const real_t *previous{nullptr};
        const real_t *current{nullptr};
        bool residual_uses_affine{false};
        bool jacobian_action_uses_affine{false};
    };

    std::unique_ptr<Op> %(op)s::create(const std::shared_ptr<FunctionSpace> &space) {
        const ptrdiff_t expected_block_size =
                block_size_for_dim(space->mesh_ptr()->spatial_dimension());
        if (space->block_size() != expected_block_size) {
            SFEM_ERROR("%(op)s requires block_size=%%ld\\n",
                       static_cast<long>(expected_block_size));
            return nullptr;
        }
        auto op = std::make_unique<%(op)s>(space);
        op->initialize();
        return op;
    }

    %(op)s::%(op)s(const std::shared_ptr<FunctionSpace> &space)
        : impl_(std::make_unique<Impl>(space)) {}
    %(op)s::~%(op)s() = default;

    ptrdiff_t %(op)s::n_dofs_domain() const { return impl_->space->n_dofs(); }
    ptrdiff_t %(op)s::n_dofs_image() const { return impl_->space->n_dofs(); }

%(performance_methods)s

    int %(op)s::initialize(const std::vector<std::string> &block_names) {
        SFEM_TRACE_SCOPE("%(op)s::initialize");
        impl_->domains = std::make_shared<MultiDomainOp>(impl_->space, block_names);
        seed_material(*impl_->domains);
        const bool needs_affine_jacobian =
                (impl_->residual_uses_affine && %(residual_affine_uses_jacobian)s) ||
                (impl_->jacobian_action_uses_affine && %(action_affine_uses_jacobian)s);
        const bool needs_affine_metric =
                (impl_->residual_uses_affine && (%(residual_affine_uses_metric_soa)s || %(residual_affine_uses_metric_aos)s)) ||
                (impl_->jacobian_action_uses_affine && (%(action_affine_uses_metric_soa)s || %(action_affine_uses_metric_aos)s));
        const bool needs_affine_metric_soa =
                (impl_->residual_uses_affine && %(residual_affine_uses_metric_soa)s) ||
                (impl_->jacobian_action_uses_affine && %(action_affine_uses_metric_soa)s);
        const bool needs_affine_metric_aos =
                (impl_->residual_uses_affine && %(residual_affine_uses_metric_aos)s) ||
                (impl_->jacobian_action_uses_affine && %(action_affine_uses_metric_aos)s);
        if (needs_affine_jacobian || needs_affine_metric) {
            const int status = cache_affine_geometry(impl_->space,
                                                     *impl_->domains,
                                                     needs_affine_jacobian,
                                                     needs_affine_metric_soa,
                                                     needs_affine_metric_aos);
            if (status != SFEM_SUCCESS) return status;
        }
%(packed_scratch_prealloc)s
        return SFEM_SUCCESS;
    }

    int %(op)s::update(const real_t *const x) {
        SFEM_TRACE_SCOPE("%(op)s::update");
        impl_->current = x;
        return SFEM_SUCCESS;
    }

    int %(op)s::update(const real_t *const previous,
                       const real_t *const current) {
        SFEM_TRACE_SCOPE("%(op)s::update");
        impl_->previous_buffer.reset();
        impl_->previous = previous;
        impl_->current = current;
        return SFEM_SUCCESS;
    }

    int %(op)s::gradient(const real_t *const state, real_t *const out) {
        SFEM_TRACE_SCOPE("%(op)s::gradient");
%(gradient_previous_check)s
        impl_->current = state;
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const geom_t *const *adjugate = nullptr;
            const geom_t *determinant = nullptr;
            const geom_t *const *geom_metric = nullptr;
            const geom_t *geom_metric_aos = nullptr;
            if (impl_->residual_uses_affine) {
                auto cache = std::static_pointer_cast<AffineGeometryCache>(
                        domain.user_data);
                if (!cache) {
                    SFEM_ERROR("%(op)s affine residual requires cached geometry\\n");
                    return SFEM_FAILURE;
                }
                if (%(residual_affine_uses_jacobian)s) {
                    if (!cache->jacobian) {
                        SFEM_ERROR("%(op)s affine residual requires cached jacobian geometry\\n");
                        return SFEM_FAILURE;
                    }
                    adjugate = reinterpret_cast<const geom_t *const *>(
                            cache->jacobian->jacobian_adjugate_SoA()->data());
                    determinant = reinterpret_cast<const geom_t *>(
                            cache->jacobian->jacobian_determinant()->data());
                }
                if (%(residual_affine_uses_metric_soa)s) {
                    if (!cache->metric_soa) {
                        SFEM_ERROR("%(op)s affine residual requires cached SoA metric geometry\\n");
                        return SFEM_FAILURE;
                    }
                    geom_metric = reinterpret_cast<const geom_t *const *>(
                            cache->metric_soa->fff_SoA()->data());
                }
                if (%(residual_affine_uses_metric_aos)s) {
                    if (!cache->metric_aos) {
                        SFEM_ERROR("%(op)s affine residual requires cached AoS metric geometry\\n");
                        return SFEM_FAILURE;
                    }
                    geom_metric_aos = reinterpret_cast<const geom_t *>(
                            cache->metric_aos->fff_AoS()->data());
                }
            }
            real_t storage[MAX_PARAMETERS];
            parameter_array(*domain.parameters,
                            mesh->spatial_dimension(),
                            storage);
%(gradient_previous_alias)s
%(residual_dispatch_body)s
        });
    }

    int %(op)s::apply(const real_t *const state,
                      const real_t *const direction,
                      real_t *const out) {
        SFEM_TRACE_SCOPE("%(op)s::apply");
        const real_t *const current = state ? state : impl_->current;
%(apply_state_check)s
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
%(laplace_packed_apply_fast_path)s
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const geom_t *const *adjugate = nullptr;
            const geom_t *determinant = nullptr;
            const geom_t *const *geom_metric = nullptr;
            const geom_t *geom_metric_aos = nullptr;
            if (impl_->jacobian_action_uses_affine) {
                auto cache = std::static_pointer_cast<AffineGeometryCache>(
                        domain.user_data);
                if (!cache) {
                    SFEM_ERROR("%(op)s affine jacobian action requires cached geometry\\n");
                    return SFEM_FAILURE;
                }
                if (%(action_affine_uses_jacobian)s) {
                    if (!cache->jacobian) {
                        SFEM_ERROR("%(op)s affine jacobian action requires cached jacobian geometry\\n");
                        return SFEM_FAILURE;
                    }
                    adjugate = reinterpret_cast<const geom_t *const *>(
                            cache->jacobian->jacobian_adjugate_SoA()->data());
                    determinant = reinterpret_cast<const geom_t *>(
                            cache->jacobian->jacobian_determinant()->data());
                }
                if (%(action_affine_uses_metric_soa)s) {
                    if (!cache->metric_soa) {
                        SFEM_ERROR("%(op)s affine jacobian action requires cached SoA metric geometry\\n");
                        return SFEM_FAILURE;
                    }
                    geom_metric = reinterpret_cast<const geom_t *const *>(
                            cache->metric_soa->fff_SoA()->data());
                }
                if (%(action_affine_uses_metric_aos)s) {
                    if (!cache->metric_aos) {
                        SFEM_ERROR("%(op)s affine jacobian action requires cached AoS metric geometry\\n");
                        return SFEM_FAILURE;
                    }
                    geom_metric_aos = reinterpret_cast<const geom_t *>(
                            cache->metric_aos->fff_AoS()->data());
                }
            }
            real_t storage[MAX_PARAMETERS];
            parameter_array(*domain.parameters,
                            mesh->spatial_dimension(),
                            storage);
%(apply_previous_alias)s
%(action_dispatch_body)s
        });
    }

    void %(op)s::set_field(const char *name,
                           const std::shared_ptr<Buffer<real_t>> &values,
                           const int component) {
        SFEM_TRACE_SCOPE("%(op)s::set_field");
        if (component != 0 || std::strcmp(name, "previous") != 0) {
            SFEM_ERROR("%(op)s supports set_field(\\"previous\\", buffer, 0)\\n");
            return;
        }
        impl_->previous_buffer = values;
        impl_->previous = values->data();
    }

    void %(op)s::set_value_in_block(const std::string &block_name,
                                    const std::string &var_name,
                                    const real_t value) {
        SFEM_TRACE_SCOPE("%(op)s::set_value_in_block");
        impl_->domains->set_value_in_block(block_name, var_name, value);
    }

    void %(op)s::set_option(const std::string &name, const bool val) {
        SFEM_TRACE_SCOPE("%(op)s::set_option");
        AffineOption options[] = {
%(affine_options)s
        };
        const bool matched = set_affine_option(name, val, options, sizeof(options) / sizeof(options[0]));
        if (matched && val && impl_->domains) {
            const bool needs_affine_jacobian =
                    (impl_->residual_uses_affine && %(residual_affine_uses_jacobian)s) ||
                    (impl_->jacobian_action_uses_affine && %(action_affine_uses_jacobian)s);
            const bool needs_affine_metric =
                    (impl_->residual_uses_affine && (%(residual_affine_uses_metric_soa)s || %(residual_affine_uses_metric_aos)s)) ||
                    (impl_->jacobian_action_uses_affine && (%(action_affine_uses_metric_soa)s || %(action_affine_uses_metric_aos)s));
            const bool needs_affine_metric_soa =
                    (impl_->residual_uses_affine && %(residual_affine_uses_metric_soa)s) ||
                    (impl_->jacobian_action_uses_affine && %(action_affine_uses_metric_soa)s);
            const bool needs_affine_metric_aos =
                    (impl_->residual_uses_affine && %(residual_affine_uses_metric_aos)s) ||
                    (impl_->jacobian_action_uses_affine && %(action_affine_uses_metric_aos)s);
            if (cache_affine_geometry(impl_->space,
                                      *impl_->domains,
                                      needs_affine_jacobian,
                                      needs_affine_metric_soa,
                                      needs_affine_metric_aos) != SFEM_SUCCESS) {
                SFEM_ERROR("%(op)s failed to cache affine geometry\\n");
            }
        }
    }

#ifdef SFEM_ENABLE_RYAML
    std::shared_ptr<Op> %(op)s::create_from_yaml(const std::shared_ptr<FunctionSpace> &space,
                                                 const ryml::ConstNodeRef             &node) {
        SFEM_TRACE_SCOPE("%(op)s::create_from_yaml");
        auto ret = std::make_shared<%(op)s>(space);

        std::vector<std::string> block_names;
        if (node.has_child("blocks")) {
            for (auto block : node["blocks"].children()) {
                if (block.has_child("name")) {
                    block_names.push_back(yaml_read_string(block["name"]));
                }
            }
        }

        AffineOption options[] = {
%(yaml_affine_options)s
        };
        read_affine_options(node, options, sizeof(options) / sizeof(options[0]));

        if (ret->initialize(block_names) != SFEM_SUCCESS) {
            return nullptr;
        }

        real_t defaults[N_MATERIAL_PARAMETERS];
        material_defaults(defaults);
        real_t top_values[N_MATERIAL_PARAMETERS];
        copy_material_parameters(defaults, top_values);
        if (material_from_yaml(node, defaults, top_values)) {
            set_material(*ret->impl_->domains, top_values);
        }

        if (node.has_child("blocks")) {
            for (auto block : node["blocks"].children()) {
                if (!block.has_child("name")) {
                    continue;
                }

                real_t block_values[N_MATERIAL_PARAMETERS];
                copy_material_parameters(top_values, block_values);
                if (!material_from_yaml(block, top_values, block_values)) {
                    continue;
                }

                const std::string block_name = yaml_read_string(block["name"]);
                set_material_in_block(*ret->impl_->domains, block_name, block_values);
            }
        }

        return ret;
    }
#endif  // SFEM_ENABLE_RYAML

    int %(op)s::hessian_crs(const real_t *const state,
                            const count_t *const rowptr,
                            const idx_t *const colidx,
                            real_t *const values) {
        SFEM_TRACE_SCOPE("%(op)s::hessian_crs");
%(hessian_crs_body)s
    }

    int %(op)s::hessian_bsr(const real_t *const state,
                            const count_t *const rowptr,
                            const idx_t *const colidx,
                            real_t *const values) {
        SFEM_TRACE_SCOPE("%(op)s::hessian_bsr");
%(hessian_bsr_body)s
    }

    int %(op)s::hessian_dia(const real_t *const state,
                            const int *const diag_offsets,
                            const ptrdiff_t ndiag,
                            real_t *const values) {
        SFEM_TRACE_SCOPE("%(op)s::hessian_dia");
%(hessian_dia_body)s
    }

    int %(op)s::value(const real_t *, real_t *const) {
        SFEM_TRACE_SCOPE("%(op)s::value");
        return SFEM_FAILURE;
    }
}  // namespace sfem
""" % {
        "op": material.op_name,
        "c_abi_include": '#include "%s"' % c_abi_header if c_abi_header else "",
        "laplace_packed_include": laplace_packed_include,
        "packed_scratch_include": packed_scratch_include,
        "packed_scratch_prealloc": packed_scratch_prealloc,
        "declaration_block": (
            declaration_block
        ),
        "declarations": "\n".join(declarations),
        "max_parameters": max_parameters,
        "defaults": defaults,
        "yaml_helpers": _yaml_helpers(material.parameter_defaults),
        "parameter_lines": parameter_lines,
        "block_size_lines": _residual_block_size_lines(block_size_by_dim),
        "laplace_packed_helpers": laplace_packed_helpers,
        "laplace_packed_member": laplace_packed_member,
        "laplace_packed_apply_fast_path": laplace_packed_apply_fast_path,
        "performance_methods": _performance_methods(material.op_name, performance_cases),
        "residual_cases": "\n".join(residual_cases),
        "action_cases": "\n".join(action_cases),
        "residual_dispatch_body": _residual_apply_dispatch_body(
            material.name,
            "residual",
            "residual_uses_affine",
            "state",
            kernel_sources,
            {dim: deps[0] for dim, deps in dependencies_by_dim.items()},
            parameter_names_by_dim,
            fields_by_dim,
            block_size_by_dim,
            residual_affine_metric_aos_elements_by_dim,
            residual_affine_metric_aos_unit_elements_by_dim,
            "            ",
        ),
        "action_dispatch_body": _residual_apply_dispatch_body(
            material.name,
            "jacobian_action",
            "jacobian_action_uses_affine",
            "current",
            kernel_sources,
            {dim: deps[1] for dim, deps in dependencies_by_dim.items()},
            parameter_names_by_dim,
            fields_by_dim,
            block_size_by_dim,
            action_affine_metric_aos_elements_by_dim,
            action_affine_metric_aos_unit_elements_by_dim,
            "            ",
        ),
        "hessian_crs_body": (
            "%s\n"
            "%s\n"
            "        auto mesh = impl_->space->mesh_ptr();\n"
            "        auto points = const_cast<const geom_t *const *>(mesh->points()->data());\n"
            "        return impl_->domains->iterate([&](const OpDomain &domain) {\n"
            "            real_t storage[MAX_PARAMETERS];\n"
            "            parameter_array(*domain.parameters,\n"
            "                            mesh->spatial_dimension(),\n"
            "                            storage);\n"
            "%s\n"
            "%s\n"
            "        });"
            % (
                hessian_state_alias,
                hessian_state_check,
                hessian_previous_alias,
                _residual_hessian_dispatch_body(
                    material.name,
                    "hessian_crs",
                    kernel_sources,
                    {dim: deps[1] for dim, deps in dependencies_by_dim.items()},
                    parameter_names_by_dim,
                    fields_by_dim,
                    block_size_by_dim,
                    ("rowptr", "colidx", "values"),
                    "            ",
                ),
            )
            if hessian_crs_cases
            else "        return SFEM_FAILURE;"
        ),
        "hessian_bsr_body": (
            "%s\n"
            "%s\n"
            "        auto mesh = impl_->space->mesh_ptr();\n"
            "        auto points = const_cast<const geom_t *const *>(mesh->points()->data());\n"
            "        return impl_->domains->iterate([&](const OpDomain &domain) {\n"
            "            real_t storage[MAX_PARAMETERS];\n"
            "            parameter_array(*domain.parameters,\n"
            "                            mesh->spatial_dimension(),\n"
            "                            storage);\n"
            "%s\n"
            "%s\n"
            "        });"
            % (
                hessian_state_alias,
                hessian_state_check,
                hessian_previous_alias,
                _residual_hessian_dispatch_body(
                    material.name,
                    "hessian_bsr",
                    kernel_sources,
                    {dim: deps[1] for dim, deps in dependencies_by_dim.items()},
                    parameter_names_by_dim,
                    fields_by_dim,
                    block_size_by_dim,
                    ("rowptr", "colidx", "values"),
                    "            ",
                ),
            )
            if hessian_bsr_cases
            else "        return SFEM_FAILURE;"
        ),
        "hessian_dia_body": (
            "%s\n"
            "%s\n"
            "        auto mesh = impl_->space->mesh_ptr();\n"
            "        auto points = const_cast<const geom_t *const *>(mesh->points()->data());\n"
            "        return impl_->domains->iterate([&](const OpDomain &domain) {\n"
            "            real_t storage[MAX_PARAMETERS];\n"
            "            parameter_array(*domain.parameters,\n"
            "                            mesh->spatial_dimension(),\n"
            "                            storage);\n"
            "%s\n"
            "%s\n"
            "        });"
            % (
                hessian_state_alias,
                hessian_state_check,
                hessian_previous_alias,
                _residual_hessian_dispatch_body(
                    material.name,
                    "hessian_dia",
                    kernel_sources,
                    {dim: deps[1] for dim, deps in dependencies_by_dim.items()},
                    parameter_names_by_dim,
                    fields_by_dim,
                    block_size_by_dim,
                    ("diag_offsets", "ndiag", "values"),
                    "            ",
                ),
            )
            if hessian_dia_cases
            else "        return SFEM_FAILURE;"
        ),
        "affine_options": _affine_option_entries(
            "residual_uses_affine",
            "jacobian_action_uses_affine",
        ),
        "residual_affine_uses_jacobian": _cpp_bool(residual_affine_uses_jacobian),
        "action_affine_uses_jacobian": _cpp_bool(action_affine_uses_jacobian),
        "residual_affine_uses_metric": _cpp_bool(residual_affine_uses_metric),
        "action_affine_uses_metric": _cpp_bool(action_affine_uses_metric),
        "residual_affine_uses_metric_soa": _cpp_bool(residual_affine_uses_metric_soa),
        "action_affine_uses_metric_soa": _cpp_bool(action_affine_uses_metric_soa),
        "residual_affine_uses_metric_aos": _cpp_bool(residual_affine_uses_metric_aos),
        "action_affine_uses_metric_aos": _cpp_bool(action_affine_uses_metric_aos),
        "yaml_affine_options": _affine_option_entries(
            "residual_uses_affine",
            "jacobian_action_uses_affine",
            owner="ret->impl_",
        ),
        "gradient_previous_check": (
            "        if (!impl_->previous) {\n"
            '            SFEM_ERROR("%s requires a previous state\\n");\n'
            "            return SFEM_FAILURE;\n"
            "        }" % material.op_name
            if residual_uses_previous
            else ""
        ),
        "gradient_previous_alias": (
            "            const real_t *const previous = impl_->previous;"
            if residual_uses_previous
            else ""
        ),
        "apply_state_check": (
            "        if (%s) {\n"
            '            SFEM_ERROR("%s requires %s\\n");\n'
            "            return SFEM_FAILURE;\n"
            "        }"
            % (
                " || ".join(
                    condition
                    for condition in (
                        "!current" if action_uses_current else "",
                        "!impl_->previous" if action_uses_previous else "",
                    )
                    if condition
                ),
                material.op_name,
                (
                    "current and previous states"
                    if action_uses_current and action_uses_previous
                    else (
                        "a current state"
                        if action_uses_current
                        else "a previous state"
                    )
                ),
            )
            if action_uses_current or action_uses_previous
            else ""
        ),
        "apply_previous_alias": (
            "            const real_t *const previous = impl_->previous;"
            if action_uses_previous
            else ""
        ),
    }
    return _header(material, True), source


def _boundary_residual_op(material, elements, c_abi_header=None, form_collections=None):
    defaults = _seed_lines(material.parameter_defaults)
    if form_collections is None:
        raise ValueError("boundary residual generated Op requires form collections")

    material_parameter_index = {
        str(name): index
        for index, (name, _) in enumerate(material.parameter_defaults)
    }
    material_parameter_names = set(material_parameter_index)
    parameter_names_by_dim = {}
    fields_by_dim = {}
    block_size_by_dim = {}
    for dim, collection in form_collections.items():
        if collection.measure != "ds":
            raise ValueError("boundary residual generated Op requires ds measure")
        fields = tuple(collection.fields)
        if len(fields) != 1:
            raise ValueError("boundary residual generated Op currently supports one field")
        parameter_names_by_dim[dim] = _boundary_residual_parameter_names(
            collection, material_parameter_names
        )
        fields_by_dim[dim] = fields
        block_size_by_dim[dim] = sum(int(field.components) for field in fields)

    max_parameters = max(1, len(material.parameter_defaults))
    parameter_lines = _residual_parameter_array_lines(parameter_names_by_dim)
    gradient_cases = []
    for element in elements:
        dim = _element_dim(element)
        fields = fields_by_dim[dim]
        block_size = block_size_by_dim[dim]
        stem = "%s_%s_%s_boundary_residual_sideset_soa" % (
            material.name,
            _element_name(element).lower(),
            _boundary_surface_name(element),
        )
        setup = _residual_soa_view_declarations(fields, "out", "out", "real_t")
        parameter_args = ", ".join(
            "condition.values->data()[%d]" % material_parameter_index[name]
            for name in parameter_names_by_dim[dim]
        )
        output_args = _boundary_soa_component_argument_names(fields, "out")
        call_args = _nonempty(
            "sideset->size()",
            "mesh->n_nodes()",
            "domain.block->elements()->data()",
            "sideset->parent()->data()",
            "sideset->lfi()->data()",
            "points",
            parameter_args,
            "FIELD_STRIDE",
            *output_args,
        )
        gradient_cases.append(
            _boundary_residual_soa_case(
                element,
                stem,
                ", ".join(call_args),
                block_size,
                setup,
            )
        )

    source = """#include "sfem_%(op)s.hpp"
%(c_abi_include)s

#include "sfem_aliases.hpp"
#include "sfem_FunctionSpace.hpp"
#include "sfem_MultiDomainOp.hpp"
#include "sfem_NeumannConditions.hpp"
#include "sfem_OpTracer.hpp"
#include "sfem_Parameters.hpp"
#include "smesh_mesh.hpp"
#include "smesh_sideset.hpp"

#include <cstring>
#include <memory>
#include <vector>

%(declaration_block)s

namespace sfem {
    namespace {
        constexpr int MAX_PARAMETERS = %(max_parameters)d;

        void seed_parameters(Parameters &parameters) {
%(defaults)s
        }

        void seed_material(MultiDomainOp &domains) {
            for (auto &entry : domains.domains()) {
                seed_parameters(*entry.second.parameters);
            }
        }

%(yaml_helpers)s

        void parameter_array(const Parameters &parameters,
                             const int dim,
                             real_t *const values) {
            int index = 0;
%(parameter_lines)s
        }

        ptrdiff_t block_size_for_dim(const int dim) {
%(block_size_lines)s
        }

        smesh::block_idx_t block_id_for_domain(const smesh::Mesh &mesh,
                                               const smesh::Mesh::Block &block) {
            for (size_t i = 0; i < mesh.n_blocks(); ++i) {
                if (mesh.block(i).get() == &block) {
                    return static_cast<smesh::block_idx_t>(i);
                }
            }
            SFEM_ERROR("%(op)s: mesh block pointer not found in mesh.blocks()\\n");
            return 0;
        }

#ifdef SFEM_ENABLE_RYAML
        std::shared_ptr<smesh::Sideset> sideset_from_yaml(
                const std::shared_ptr<FunctionSpace> &space,
                const ryml::ConstNodeRef             &node) {
            const bool is_sideset = node["type"].readable() && node["type"].val() == "sideset";
            const bool is_file    = node["format"].readable() && node["format"].val() == "file";
            const bool is_expr    = node["format"].readable() && node["format"].val() == "expr";

            if (!is_sideset && node.has_child("type")) {
                SFEM_ERROR("%(op)s neumann condition requires type=sideset\\n");
                return nullptr;
            }

            if (is_file || node.has_child("path")) {
                if (!node.has_child("path")) {
                    SFEM_ERROR("%(op)s file sideset condition requires path\\n");
                    return nullptr;
                }
                const std::string path = yaml_read_string(node["path"]);
                return smesh::Sideset::create_from_file(
                        space->mesh_ptr()->comm(), smesh::Path(path));
            }

            if (is_expr || (node.has_child("parent") && node.has_child("lfi"))) {
                if (!node["parent"].is_seq() || !node["lfi"].is_seq()) {
                    SFEM_ERROR("%(op)s expr sideset condition requires parent/lfi sequences\\n");
                    return nullptr;
                }

                const ptrdiff_t size = node["parent"].num_children();
                if (node["lfi"].num_children() != size) {
                    SFEM_ERROR("%(op)s expr sideset parent/lfi length mismatch\\n");
                    return nullptr;
                }

                auto parent = create_host_buffer<element_idx_t>(size);
                auto lfi    = create_host_buffer<int16_t>(size);

                ptrdiff_t parent_count = 0;
                for (auto p : node["parent"].children()) {
                    p >> parent->data()[parent_count++];
                }

                ptrdiff_t lfi_count = 0;
                for (auto p : node["lfi"].children()) {
                    p >> lfi->data()[lfi_count++];
                }

                return std::make_shared<smesh::Sideset>(
                        space->mesh_ptr()->comm(), parent, lfi);
            }

            SFEM_ERROR("%(op)s neumann condition requires format=file or format=expr\\n");
            return nullptr;
        }
#endif  // SFEM_ENABLE_RYAML
    }  // namespace

    class %(op)s::Impl {
    public:
        explicit Impl(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        std::shared_ptr<FunctionSpace> space;
        std::shared_ptr<MultiDomainOp> domains;
        std::vector<NeumannConditions::Condition> conditions;
    };

    std::unique_ptr<Op> %(op)s::create(const std::shared_ptr<FunctionSpace> &space) {
        const ptrdiff_t expected_block_size =
                block_size_for_dim(space->mesh_ptr()->spatial_dimension());
        if (space->block_size() != expected_block_size) {
            SFEM_ERROR("%(op)s requires block_size=%%ld\\n",
                       static_cast<long>(expected_block_size));
            return nullptr;
        }
        auto op = std::make_unique<%(op)s>(space);
        op->initialize();
        return op;
    }

    %(op)s::%(op)s(const std::shared_ptr<FunctionSpace> &space)
        : impl_(std::make_unique<Impl>(space)) {}
    %(op)s::~%(op)s() = default;

    ptrdiff_t %(op)s::n_dofs_domain() const { return impl_->space->n_dofs(); }
    ptrdiff_t %(op)s::n_dofs_image() const { return impl_->space->n_dofs(); }

%(performance_methods)s

    int %(op)s::initialize(const std::vector<std::string> &block_names) {
        SFEM_TRACE_SCOPE("%(op)s::initialize");
        impl_->domains = std::make_shared<MultiDomainOp>(impl_->space, block_names);
        seed_material(*impl_->domains);
        return SFEM_SUCCESS;
    }

    void %(op)s::add_sideset(const std::shared_ptr<smesh::Sideset> &sideset) {
        real_t values[MAX_PARAMETERS];
        material_defaults(values);
        add_sideset(sideset, values);
    }

    void %(op)s::add_sideset(const std::shared_ptr<smesh::Sideset> &sideset,
                             const real_t *const parameters) {
        SFEM_TRACE_SCOPE("%(op)s::add_sideset");
        NeumannConditions::Condition condition;
        condition.sidesets = {sideset};
        condition.values = create_host_buffer<real_t>(MAX_PARAMETERS);
        for (int i = 0; i < MAX_PARAMETERS; ++i) {
            condition.values->data()[i] = parameters[i];
        }
        condition.value = parameters[0];
        condition.component = 0;
        add_condition(condition);
    }

    void %(op)s::add_condition(const NeumannConditions::Condition &condition) {
        SFEM_TRACE_SCOPE("%(op)s::add_condition");
        impl_->conditions.push_back(condition);
    }

    int %(op)s::gradient(const real_t *const, real_t *const out) {
        SFEM_TRACE_SCOPE("%(op)s::gradient");
        if (impl_->conditions.empty()) {
            return SFEM_SUCCESS;
        }
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const smesh::block_idx_t block_id = block_id_for_domain(*mesh, *domain.block);
            int status = SFEM_SUCCESS;
            for (const auto &condition : impl_->conditions) {
                const auto sideset = condition.sidesets.empty() ? nullptr : condition.sidesets[0];
                if (!sideset || !condition.values || sideset->block_id() != block_id) {
                    continue;
                }
                switch (domain.element_type) {
%(gradient_cases)s
                    default:
                        SFEM_ERROR("%(op)s does not support element type %%d\\n",
                                   domain.element_type);
                        return SFEM_FAILURE;
                }
            }
            return status;
        });
    }

    int %(op)s::apply(const real_t *const,
                      const real_t *const,
                      real_t *const) {
        SFEM_TRACE_SCOPE("%(op)s::apply");
        return SFEM_SUCCESS;
    }

    int %(op)s::value(const real_t *, real_t *const) {
        SFEM_TRACE_SCOPE("%(op)s::value");
        return SFEM_SUCCESS;
    }

    int %(op)s::hessian_crs(const real_t *const,
                            const count_t *const,
                            const idx_t *const,
                            real_t *const) {
        SFEM_TRACE_SCOPE("%(op)s::hessian_crs");
        return SFEM_SUCCESS;
    }

    void %(op)s::set_field(const char *,
                           const std::shared_ptr<Buffer<real_t>> &,
                           const int) {
        SFEM_TRACE_SCOPE("%(op)s::set_field");
    }

    void %(op)s::set_option(const std::string &, const bool) {
        SFEM_TRACE_SCOPE("%(op)s::set_option");
    }

    void %(op)s::set_value_in_block(const std::string &block_name,
                                    const std::string &var_name,
                                    const real_t value) {
        SFEM_TRACE_SCOPE("%(op)s::set_value_in_block");
        impl_->domains->set_value_in_block(block_name, var_name, value);
    }

#ifdef SFEM_ENABLE_RYAML
    std::shared_ptr<Op> %(op)s::create_from_yaml(const std::shared_ptr<FunctionSpace> &space,
                                                 const ryml::ConstNodeRef             &node) {
        SFEM_TRACE_SCOPE("%(op)s::create_from_yaml");
        auto ret = std::make_shared<%(op)s>(space);

        std::vector<std::string> block_names;
        if (node.has_child("blocks")) {
            for (auto block : node["blocks"].children()) {
                if (block.has_child("name")) {
                    block_names.push_back(yaml_read_string(block["name"]));
                }
            }
        }

        if (ret->initialize(block_names) != SFEM_SUCCESS) {
            return nullptr;
        }

        real_t defaults[MAX_PARAMETERS];
        material_defaults(defaults);
        real_t top_values[MAX_PARAMETERS];
        copy_material_parameters(defaults, top_values);
        material_from_yaml(node, defaults, top_values);

        const auto neumann_node =
                node.has_child("neumann_conditions") ? node["neumann_conditions"] :
                 ryml::ConstNodeRef();
        if (neumann_node.readable() && neumann_node.is_seq()) {
            for (auto condition_node : neumann_node.children()) {
                auto sideset = sideset_from_yaml(space, condition_node);
                if (!sideset) {
                    return nullptr;
                }
                real_t condition_values[MAX_PARAMETERS];
                material_from_yaml(condition_node, top_values, condition_values);
                ret->add_sideset(sideset, condition_values);
            }
        }

        return ret;
    }
#endif  // SFEM_ENABLE_RYAML
}  // namespace sfem
""" % {
        "op": material.op_name,
        "c_abi_include": '#include "%s"' % c_abi_header if c_abi_header else "",
        "declaration_block": "",
        "max_parameters": max_parameters,
        "defaults": defaults,
        "yaml_helpers": _yaml_helpers(material.parameter_defaults),
        "parameter_lines": parameter_lines,
        "block_size_lines": _residual_block_size_lines(block_size_by_dim),
        "performance_methods": _performance_methods(material.op_name, {}),
        "gradient_cases": "\n".join(gradient_cases),
    }
    return _boundary_header(material), source


def _boundary_header(material):
    return """#pragma once

#include "sfem_NeumannConditions.hpp"
#include "sfem_Op.hpp"

namespace smesh {
    class Sideset;
}

namespace sfem {
    class %(op)s final : public Op {
    public:
        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space);

        explicit %(op)s(const std::shared_ptr<FunctionSpace> &space);
        ~%(op)s() override;

        const char *name() const override { return "%(op)s"; }
        bool is_linear() const override { return true; }
        ptrdiff_t n_dofs_domain() const override;
        ptrdiff_t n_dofs_image() const override;
        double flops_value() const override;
        double flops_gradient() const override;
        double flops_apply() const override;
        size_t memory_traffic_bytes_value() const override;
        size_t memory_traffic_bytes_gradient() const override;
        size_t memory_traffic_bytes_apply() const override;

        int initialize(const std::vector<std::string> &block_names = {}) override;
        void add_condition(const NeumannConditions::Condition &condition);
        void add_sideset(const std::shared_ptr<smesh::Sideset> &sideset);
        void add_sideset(const std::shared_ptr<smesh::Sideset> &sideset,
                         const real_t *parameters);
        int gradient(const real_t *const x, real_t *const out) override;
        int apply(const real_t *const x,
                  const real_t *const h,
                  real_t *const out) override;
        int value(const real_t *x, real_t *const out) override;
        int hessian_crs(const real_t *const x,
                        const count_t *const rowptr,
                        const idx_t *const colidx,
                        real_t *const values) override;
        void set_field(const char *name,
                       const std::shared_ptr<Buffer<real_t>> &values,
                       int component) override;
        void set_option(const std::string &name, bool val) override;
        void set_value_in_block(const std::string &block_name,
                                const std::string &var_name,
                                real_t value) override;
#ifdef SFEM_ENABLE_RYAML
        std::shared_ptr<Op> create_from_yaml(const std::shared_ptr<FunctionSpace> &space,
                                             const ryml::ConstNodeRef             &node) override;
#endif  // SFEM_ENABLE_RYAML

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
}  // namespace sfem
""" % {"op": material.op_name}


def _coupled_energy_residual_op(material, elements, c_abi_header=None, systems_by_dim=None):
    if systems_by_dim is None:
        systems_by_dim = _systems_by_dim(material, elements)
    equations_by_dim = {
        dim: tuple(system.equations)
        for dim, system in systems_by_dim.items()
    }
    representative = next(iter(equations_by_dim.values()))
    energy_equations = tuple(equation for equation in representative if equation.is_energy)
    residual_equations = tuple(equation for equation in representative if equation.is_residual)
    if len(energy_equations) != 1 or len(residual_equations) != 1:
        raise ValueError(
            "generated coupled Op wrappers currently require one energy and one residual equation"
        )

    energy_name = energy_equations[0].name
    residual_name = residual_equations[0].name
    if not energy_name or not residual_name:
        raise ValueError("coupled generated Op equations must be named")

    defaults = _seed_lines(material.parameter_defaults)
    parameter_index = {
        str(name): index
        for index, (name, _) in enumerate(material.parameter_defaults)
    }
    cases = _coupled_cases(
        material,
        elements,
        systems_by_dim,
        energy_name,
        residual_name,
        parameter_index,
    )
    dependency_flags = _coupled_dependency_flags(
        systems_by_dim,
        energy_name,
        residual_name,
    )
    max_parameters = max(1, len(material.parameter_defaults))
    source = """#include "sfem_%(op)s.hpp"
%(c_abi_include)s

#include "sfem_FunctionSpace.hpp"
#include "sfem_MultiDomainOp.hpp"
#include "sfem_OpTracer.hpp"
#include "sfem_Parameters.hpp"
#include "smesh_kernel_data.hpp"
#include "smesh_mesh.hpp"

#include <algorithm>
#include <cstring>
#include <memory>

%(declaration_block)s

namespace sfem {
    namespace {
        constexpr int MAX_PARAMETERS = %(max_parameters)d;

        void seed_parameters(Parameters &parameters) {
%(defaults)s
        }

        void seed_material(MultiDomainOp &domains) {
            for (auto &entry : domains.domains()) {
                seed_parameters(*entry.second.parameters);
            }
        }

%(yaml_helpers)s

        smesh::block_idx_t block_id_for_domain(const smesh::Mesh &mesh,
                                               const smesh::Mesh::Block &block) {
            for (size_t i = 0; i < mesh.n_blocks(); ++i) {
                if (mesh.block(i).get() == &block) {
                    return static_cast<smesh::block_idx_t>(i);
                }
            }
            SFEM_ERROR("%(op)s: mesh block pointer not found in mesh.blocks()\\n");
            return 0;
        }

        int cache_affine_geometry(const std::shared_ptr<FunctionSpace> &space,
                                  MultiDomainOp &domains) {
            auto mesh = space->mesh_ptr();
            for (auto &entry : domains.domains()) {
                if (entry.second.user_data) {
                    continue;
                }
                const smesh::block_idx_t block_id =
                        block_id_for_domain(*mesh, *entry.second.block);
                auto jacobian = smesh::JacobianAdjugateAndDeterminant::create_SoA(
                        mesh, smesh::MEMORY_SPACE_HOST, block_id);
                if (!jacobian) {
                    return SFEM_FAILURE;
                }
                entry.second.user_data = std::static_pointer_cast<void>(jacobian);
            }
            return SFEM_SUCCESS;
        }

        void parameter_array(const Parameters &parameters,
                             real_t *const values) {
%(parameter_lines)s
        }

        ptrdiff_t block_size_for_dim(const int dim) {
            switch (dim) {
%(block_size_lines)s
                default:
                    SFEM_ERROR("unsupported spatial dimension %%d for generated coupled block size\\n", dim);
                    return 0;
            }
        }
    }  // namespace

    class %(op)s::Impl {
    public:
        explicit Impl(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        std::shared_ptr<FunctionSpace> space;
        std::shared_ptr<MultiDomainOp> domains;
        std::shared_ptr<Buffer<real_t>> previous_buffer;
        std::unique_ptr<real_t[]> element_values;
        ptrdiff_t element_capacity{0};
        const real_t *previous{nullptr};
        const real_t *current{nullptr};
        bool objective_uses_affine{false};
        bool gradient_uses_affine{false};
        bool apply_uses_affine{false};
        bool residual_uses_affine{false};
        bool jacobian_action_uses_affine{false};
    };

    std::unique_ptr<Op> %(op)s::create(const std::shared_ptr<FunctionSpace> &space) {
        const ptrdiff_t expected_block_size =
                block_size_for_dim(space->mesh_ptr()->spatial_dimension());
        if (space->block_size() != expected_block_size) {
            SFEM_ERROR("%(op)s requires block_size=%%ld\\n",
                       static_cast<long>(expected_block_size));
            return nullptr;
        }
        auto op = std::make_unique<%(op)s>(space);
        op->initialize();
        return op;
    }

    %(op)s::%(op)s(const std::shared_ptr<FunctionSpace> &space)
        : impl_(std::make_unique<Impl>(space)) {}
    %(op)s::~%(op)s() = default;

    ptrdiff_t %(op)s::n_dofs_domain() const { return impl_->space->n_dofs(); }
    ptrdiff_t %(op)s::n_dofs_image() const { return impl_->space->n_dofs(); }

%(performance_methods)s

    int %(op)s::initialize(const std::vector<std::string> &block_names) {
        SFEM_TRACE_SCOPE("%(op)s::initialize");
        impl_->domains = std::make_shared<MultiDomainOp>(impl_->space, block_names);
        seed_material(*impl_->domains);
        auto mesh = impl_->space->mesh_ptr();
        const bool needs_affine_geometry =
                impl_->objective_uses_affine ||
                impl_->gradient_uses_affine ||
                impl_->apply_uses_affine ||
                impl_->residual_uses_affine ||
                impl_->jacobian_action_uses_affine;
        for (auto &entry : impl_->domains->domains()) {
            impl_->element_capacity =
                    std::max(impl_->element_capacity, entry.second.block->n_elements());
            if (needs_affine_geometry) {
                const smesh::block_idx_t block_id =
                        block_id_for_domain(*mesh, *entry.second.block);
                auto jacobian = smesh::JacobianAdjugateAndDeterminant::create_SoA(
                        mesh, smesh::MEMORY_SPACE_HOST, block_id);
                if (!jacobian) {
                    return SFEM_FAILURE;
                }
                entry.second.user_data = std::static_pointer_cast<void>(jacobian);
            }
        }
        impl_->element_values.reset(new real_t[impl_->element_capacity]);
        return SFEM_SUCCESS;
    }

    int %(op)s::update(const real_t *const x) {
        SFEM_TRACE_SCOPE("%(op)s::update");
        impl_->current = x;
        return SFEM_SUCCESS;
    }

    int %(op)s::update(const real_t *const previous,
                       const real_t *const current) {
        SFEM_TRACE_SCOPE("%(op)s::update");
        impl_->previous_buffer.reset();
        impl_->previous = previous;
        impl_->current = current;
        return SFEM_SUCCESS;
    }

    int %(op)s::gradient(const real_t *const state, real_t *const out) {
        SFEM_TRACE_SCOPE("%(op)s::gradient");
%(gradient_previous_check)s
        impl_->current = state;
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const geom_t *const *adjugate = nullptr;
            const geom_t *determinant = nullptr;
            if (impl_->gradient_uses_affine || impl_->residual_uses_affine) {
                auto jacobian = std::static_pointer_cast<smesh::JacobianAdjugateAndDeterminant>(
                        domain.user_data);
                if (!jacobian) {
                    SFEM_ERROR("%(op)s affine gradient/residual requires cached geometry\\n");
                    return SFEM_FAILURE;
                }
                adjugate = reinterpret_cast<const geom_t *const *>(
                        jacobian->jacobian_adjugate_SoA()->data());
                determinant = reinterpret_cast<const geom_t *>(
                        jacobian->jacobian_determinant()->data());
            }
            real_t storage[MAX_PARAMETERS];
            parameter_array(*domain.parameters, storage);
%(gradient_previous_alias)s
            switch (domain.element_type) {
%(gradient_cases)s
                default:
                    SFEM_ERROR("%(op)s does not support element type %%d\\n",
                               domain.element_type);
                    return SFEM_FAILURE;
            }
        });
    }

    int %(op)s::apply(const real_t *const state,
                      const real_t *const direction,
                      real_t *const out) {
        SFEM_TRACE_SCOPE("%(op)s::apply");
        const real_t *const current = state ? state : impl_->current;
%(apply_state_check)s
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const geom_t *const *adjugate = nullptr;
            const geom_t *determinant = nullptr;
            if (impl_->apply_uses_affine || impl_->jacobian_action_uses_affine) {
                auto jacobian = std::static_pointer_cast<smesh::JacobianAdjugateAndDeterminant>(
                        domain.user_data);
                if (!jacobian) {
                    SFEM_ERROR("%(op)s affine hessian/jacobian action requires cached geometry\\n");
                    return SFEM_FAILURE;
                }
                adjugate = reinterpret_cast<const geom_t *const *>(
                        jacobian->jacobian_adjugate_SoA()->data());
                determinant = reinterpret_cast<const geom_t *>(
                        jacobian->jacobian_determinant()->data());
            }
            real_t storage[MAX_PARAMETERS];
            parameter_array(*domain.parameters, storage);
%(apply_previous_alias)s
            switch (domain.element_type) {
%(apply_cases)s
                default:
                    SFEM_ERROR("%(op)s does not support element type %%d\\n",
                               domain.element_type);
                    return SFEM_FAILURE;
            }
        });
    }

    int %(op)s::value(const real_t *state, real_t *const out) {
        SFEM_TRACE_SCOPE("%(op)s::value");
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        *out = 0;
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const ptrdiff_t nelements = domain.block->n_elements();
            const geom_t *const *adjugate = nullptr;
            const geom_t *determinant = nullptr;
            if (impl_->objective_uses_affine) {
                auto jacobian = std::static_pointer_cast<smesh::JacobianAdjugateAndDeterminant>(
                        domain.user_data);
                if (!jacobian) {
                    SFEM_ERROR("%(op)s affine objective requires cached geometry\\n");
                    return SFEM_FAILURE;
                }
                adjugate = reinterpret_cast<const geom_t *const *>(
                        jacobian->jacobian_adjugate_SoA()->data());
                determinant = reinterpret_cast<const geom_t *>(
                        jacobian->jacobian_determinant()->data());
            }
            std::fill(impl_->element_values.get(),
                      impl_->element_values.get() + nelements,
                      0);
            real_t storage[MAX_PARAMETERS];
            parameter_array(*domain.parameters, storage);
            int status = SFEM_FAILURE;
            switch (domain.element_type) {
%(objective_cases)s
                default:
                    SFEM_ERROR("%(op)s does not support element type %%d\\n",
                               domain.element_type);
                    return SFEM_FAILURE;
            }
            if (status != SFEM_SUCCESS) return status;
            real_t sum = 0;
#pragma omp simd reduction(+ : sum)
            for (ptrdiff_t element = 0; element < nelements; ++element) {
                sum += impl_->element_values[element];
            }
            *out += sum;
            return SFEM_SUCCESS;
        });
    }
    void %(op)s::set_field(const char *name,
                           const std::shared_ptr<Buffer<real_t>> &values,
                           const int component) {
        SFEM_TRACE_SCOPE("%(op)s::set_field");
        if (component != 0 || std::strcmp(name, "previous") != 0) {
            SFEM_ERROR("%(op)s supports set_field(\\"previous\\", buffer, 0)\\n");
            return;
        }
        impl_->previous_buffer = values;
        impl_->previous = values->data();
    }

    void %(op)s::set_option(const std::string &name, const bool val) {
        SFEM_TRACE_SCOPE("%(op)s::set_option");
        AffineOption options[] = {
%(affine_options)s
        };
        const bool matched = set_affine_option(name, val, options, sizeof(options) / sizeof(options[0]));
        if (matched && val && impl_->domains &&
            cache_affine_geometry(impl_->space, *impl_->domains) != SFEM_SUCCESS) {
            SFEM_ERROR("%(op)s failed to cache affine geometry\\n");
        }
    }

    void %(op)s::set_value_in_block(const std::string &block_name,
                                    const std::string &var_name,
                                    const real_t value) {
        SFEM_TRACE_SCOPE("%(op)s::set_value_in_block");
        impl_->domains->set_value_in_block(block_name, var_name, value);
    }

#ifdef SFEM_ENABLE_RYAML
    std::shared_ptr<Op> %(op)s::create_from_yaml(const std::shared_ptr<FunctionSpace> &space,
                                                 const ryml::ConstNodeRef             &node) {
        SFEM_TRACE_SCOPE("%(op)s::create_from_yaml");
        auto ret = std::make_shared<%(op)s>(space);

        std::vector<std::string> block_names;
        if (node.has_child("blocks")) {
            for (auto block : node["blocks"].children()) {
                if (block.has_child("name")) {
                    block_names.push_back(yaml_read_string(block["name"]));
                }
            }
        }

        AffineOption options[] = {
%(yaml_affine_options)s
        };
        read_affine_options(node, options, sizeof(options) / sizeof(options[0]));

        if (ret->initialize(block_names) != SFEM_SUCCESS) {
            return nullptr;
        }

        real_t defaults[N_MATERIAL_PARAMETERS];
        material_defaults(defaults);
        real_t top_values[N_MATERIAL_PARAMETERS];
        copy_material_parameters(defaults, top_values);
        if (material_from_yaml(node, defaults, top_values)) {
            set_material(*ret->impl_->domains, top_values);
        }

        if (node.has_child("blocks")) {
            for (auto block : node["blocks"].children()) {
                if (!block.has_child("name")) {
                    continue;
                }

                real_t block_values[N_MATERIAL_PARAMETERS];
                copy_material_parameters(top_values, block_values);
                if (!material_from_yaml(block, top_values, block_values)) {
                    continue;
                }

                const std::string block_name = yaml_read_string(block["name"]);
                set_material_in_block(*ret->impl_->domains, block_name, block_values);
            }
        }

        return ret;
    }
#endif  // SFEM_ENABLE_RYAML

    int %(op)s::hessian_crs(const real_t *const,
                            const count_t *const,
                            const idx_t *const,
                            real_t *const) {
        SFEM_TRACE_SCOPE("%(op)s::hessian_crs");
        return SFEM_FAILURE;
    }

    int %(op)s::hessian_bsr(const real_t *const,
                            const count_t *const,
                            const idx_t *const,
                            real_t *const) {
        SFEM_TRACE_SCOPE("%(op)s::hessian_bsr");
        return SFEM_FAILURE;
    }

    int %(op)s::hessian_dia(const real_t *const,
                            const int *const,
                            const ptrdiff_t,
                            real_t *const) {
        SFEM_TRACE_SCOPE("%(op)s::hessian_dia");
        return SFEM_FAILURE;
    }
}  // namespace sfem
""" % {
        "op": material.op_name,
        "c_abi_include": '#include "%s"' % c_abi_header if c_abi_header else "",
        "declaration_block": "",
        "max_parameters": max_parameters,
        "defaults": defaults,
        "yaml_helpers": _yaml_helpers(material.parameter_defaults),
        "parameter_lines": _coupled_parameter_array_lines(material.parameter_defaults),
        "block_size_lines": _coupled_block_size_lines(systems_by_dim),
        "performance_methods": _performance_methods(material.op_name, cases["performance"]),
        "gradient_previous_check": (
            "        if (!impl_->previous) {\n"
            '            SFEM_ERROR("%s requires a previous state\\n");\n'
            "            return SFEM_FAILURE;\n"
            "        }" % material.op_name
            if dependency_flags["gradient_previous"]
            else ""
        ),
        "gradient_previous_alias": (
            "            const real_t *const previous = impl_->previous;"
            if dependency_flags["gradient_previous"]
            else ""
        ),
        "apply_state_check": _coupled_apply_state_check(
            material.op_name,
            dependency_flags["apply_current"],
            dependency_flags["apply_previous"],
        ),
        "apply_previous_alias": (
            "            const real_t *const previous = impl_->previous;"
            if dependency_flags["apply_previous"]
            else ""
        ),
        "gradient_cases": "\n".join(cases["gradient"]),
        "apply_cases": "\n".join(cases["apply"]),
        "objective_cases": "\n".join(cases["objective"]),
        "affine_options": _affine_option_entries(
            "objective_uses_affine",
            "gradient_uses_affine",
            "apply_uses_affine",
            "residual_uses_affine",
            "jacobian_action_uses_affine",
        ),
        "yaml_affine_options": _affine_option_entries(
            "objective_uses_affine",
            "gradient_uses_affine",
            "apply_uses_affine",
            "residual_uses_affine",
            "jacobian_action_uses_affine",
            owner="ret->impl_",
        ),
    }
    return _header(material, True), source


def _coupled_dependency_flags(systems_by_dim, energy_name, residual_name):
    from codegen.framework.symbolic.forms import FormOrder

    flags = {
        "gradient_previous": False,
        "apply_current": False,
        "apply_previous": False,
    }
    for system in systems_by_dim.values():
        energy_equation = next(equation for equation in system.equations if equation.name == energy_name)
        residual_equation = next(equation for equation in system.equations if equation.name == residual_name)
        energy_collection = system.form_collection(energy_equation)
        residual_collection = system.form_collection(residual_equation)
        energy_apply = energy_collection.form_metadata(FormOrder.TWO).dependencies
        residual_gradient = residual_collection.form_metadata(FormOrder.ONE).dependencies
        residual_apply = residual_collection.form_metadata(FormOrder.TWO).dependencies
        flags["gradient_previous"] |= bool(getattr(residual_gradient, "previous", False))
        flags["apply_current"] |= bool(getattr(energy_apply, "current", False)) or bool(
            getattr(residual_apply, "current", False)
        )
        flags["apply_previous"] |= bool(getattr(residual_apply, "previous", False))
    return flags


def _coupled_apply_state_check(op_name, uses_current, uses_previous):
    conditions = []
    if uses_current:
        conditions.append("!current")
    if uses_previous:
        conditions.append("!impl_->previous")
    if not conditions:
        return ""
    requirement = (
        "current and previous states"
        if uses_current and uses_previous
        else ("a current state" if uses_current else "a previous state")
    )
    return (
        "        if (%s) {\n"
        '            SFEM_ERROR("%s requires %s\\n");\n'
        "            return SFEM_FAILURE;\n"
        "        }"
        % (" || ".join(conditions), op_name, requirement)
    )


def _coupled_cases(material, elements, systems_by_dim, energy_name, residual_name, parameter_index):
    from codegen.framework.symbolic.forms import FormOrder

    cases = {
        "gradient": [],
        "apply": [],
        "objective": [],
        "performance": {"value": [], "gradient": [], "apply": []},
    }
    for element in elements:
        dim = _element_dim(element)
        system = systems_by_dim[dim]
        energy_equation = next(equation for equation in system.equations if equation.name == energy_name)
        residual_equation = next(equation for equation in system.equations if equation.name == residual_name)
        energy_collection = system.form_collection(energy_equation)
        residual_collection = system.form_collection(residual_equation)
        energy_field = energy_collection.fields[0]
        residual_fields = tuple(residual_collection.fields)
        block_size = sum(int(field.components) for field in residual_fields)
        energy_element = _compatible_element_for_field(element, energy_field)
        energy_label = energy_element.lower()
        mixed_label = _element_name(element).lower()
        energy_stem = "%s_%s_%s_%s" % (material.name, energy_name, energy_label, energy_label)
        residual_stem = "%s_%s_%s" % (material.name, residual_name, mixed_label)
        cases["performance"]["value"].append(
            _performance_case(
                element,
                ("%s_objective_soa_diagnostics" % energy_stem,),
                affine_flags=("objective_uses_affine",),
            )
        )
        cases["performance"]["gradient"].append(
            _performance_case(
                element,
                (
                    "%s_gradient_soa_diagnostics" % energy_stem,
                    "%s_residual_element_soa_diagnostics" % residual_stem,
                ),
                affine_flags=("gradient_uses_affine", "residual_uses_affine"),
            )
        )
        cases["performance"]["apply"].append(
            _performance_case(
                element,
                (
                    "%s_apply_soa_diagnostics" % energy_stem,
                    "%s_jacobian_action_element_soa_diagnostics" % residual_stem,
                ),
                affine_flags=("apply_uses_affine", "jacobian_action_uses_affine"),
            )
        )
        energy_objective_dependencies = energy_collection.form_metadata(FormOrder.ZERO).dependencies
        energy_gradient_dependencies = energy_collection.form_metadata(FormOrder.ONE).dependencies
        energy_apply_dependencies = energy_collection.form_metadata(FormOrder.TWO).dependencies
        residual_dependencies = residual_collection.form_metadata(FormOrder.ONE).dependencies
        residual_apply_dependencies = residual_collection.form_metadata(FormOrder.TWO).dependencies
        energy_params = _metadata_parameter_args(energy_gradient_dependencies, parameter_index)
        energy_apply_params = _metadata_parameter_args(energy_apply_dependencies, parameter_index)
        energy_objective_params = _metadata_parameter_args(energy_objective_dependencies, parameter_index)
        residual_params = _dependency_parameter_args(
            residual_dependencies.parameters,
            parameter_index,
        )
        residual_apply_params = _dependency_parameter_args(
            residual_apply_dependencies.parameters,
            parameter_index,
        )

        field_offsets = _field_offsets(residual_fields)
        energy_components = tuple(range(int(energy_field.components)))
        energy_data = _component_offsets("state", field_offsets[energy_field.name], energy_components)
        energy_direction = _component_offsets("direction", field_offsets[energy_field.name], energy_components)
        energy_out = _component_offsets("out", field_offsets[energy_field.name], energy_components)
        residual_state_setup = _residual_soa_view_declarations(residual_fields, "state", "data", "const real_t")
        residual_current_setup = _residual_soa_view_declarations(residual_fields, "current", "data", "const real_t")
        residual_previous_setup = _residual_soa_view_declarations(residual_fields, "previous", "old_data", "const real_t")
        residual_direction_setup = _residual_soa_view_declarations(residual_fields, "direction", "direction_data", "const real_t")
        residual_out_setup = _residual_soa_view_declarations(residual_fields, "out", "out", "real_t")
        residual_state_args = _residual_soa_field_argument_names(residual_fields, "data")
        residual_previous_args = _residual_soa_field_argument_names(residual_fields, "old_data")
        residual_direction_args = _residual_soa_field_argument_names(residual_fields, "direction_data")
        residual_out_args = _residual_soa_field_argument_names(residual_fields, "out")

        geometry_affine = _affine_geometry_offsets(dim) + ", determinant"
        common_iso = "domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points"
        common_affine = "domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), %s" % geometry_affine
        energy_grad_args = ", ".join(
            _nonempty(
                *_coupled_energy_field_args(
                    energy_gradient_dependencies,
                    block_size,
                    current=energy_data,
                ),
                block_size,
                energy_out,
            )
        )
        energy_grad_affine = "%s_gradient_affine_mesh_soa(%s%s, %s)" % (
            energy_stem, common_affine, energy_params, energy_grad_args
        )
        energy_grad_iso = "%s_gradient_isoparametric_mesh_soa(%s%s, %s)" % (
            energy_stem, common_iso, energy_params, energy_grad_args
        )
        residual_gradient_args = []
        residual_gradient_setup = []
        if residual_dependencies.current:
            residual_gradient_setup.extend(residual_state_setup)
            residual_gradient_args.extend((str(block_size), *residual_state_args))
        if residual_dependencies.previous:
            residual_gradient_setup.extend(residual_previous_setup)
            residual_gradient_args.extend((str(block_size), *residual_previous_args))
        residual_gradient_setup.extend(residual_out_setup)
        residual_gradient_args.extend((str(block_size), *residual_out_args))
        residual_args_common = ", ".join(
            _nonempty(
                residual_params[2:] if residual_params.startswith(", ") else residual_params,
                *residual_gradient_args,
            )
        )
        residual_grad_affine = "%s_residual_affine_mesh_soa(%s, %s)" % (
            residual_stem, common_affine, residual_args_common
        )
        residual_grad_iso = "%s_residual_isoparametric_mesh_soa(%s, %s)" % (
            residual_stem, common_iso, residual_args_common
        )
        cases["gradient"].append(
            _coupled_case(
                element,
                block_size,
                residual_gradient_setup,
                (
                    "                    int status = impl_->gradient_uses_affine ? %s : %s;\n"
                    "                    if (status != SFEM_SUCCESS) return status;\n"
                    "                    return impl_->residual_uses_affine ? %s : %s;"
                ) % (energy_grad_affine, energy_grad_iso, residual_grad_affine, residual_grad_iso),
            )
        )

        energy_apply_args = ", ".join(
            _nonempty(
                *_coupled_energy_field_args(
                    energy_apply_dependencies,
                    block_size,
                    current=energy_data,
                    direction=energy_direction,
                ),
                block_size,
                energy_out,
            )
        )
        energy_apply_affine = "%s_apply_affine_mesh_soa(%s%s, %s)" % (
            energy_stem, common_affine, energy_apply_params, energy_apply_args
        )
        energy_apply_iso = "%s_apply_isoparametric_mesh_soa(%s%s, %s)" % (
            energy_stem, common_iso, energy_apply_params, energy_apply_args
        )
        residual_apply_args = []
        residual_apply_setup = []
        if residual_apply_dependencies.current:
            residual_apply_setup.extend(residual_current_setup)
            residual_apply_args.extend((str(block_size), *residual_state_args))
        if residual_apply_dependencies.previous:
            residual_apply_setup.extend(residual_previous_setup)
            residual_apply_args.extend((str(block_size), *residual_previous_args))
        if residual_apply_dependencies.direction:
            residual_apply_setup.extend(residual_direction_setup)
            residual_apply_args.extend((str(block_size), *residual_direction_args))
        residual_apply_setup.extend(residual_out_setup)
        residual_apply_args.extend((str(block_size), *residual_out_args))
        residual_apply_args_common = ", ".join(
            _nonempty(
                residual_apply_params[2:] if residual_apply_params.startswith(", ") else residual_apply_params,
                *residual_apply_args,
            )
        )
        residual_apply_affine = "%s_jacobian_action_affine_mesh_soa(%s, %s)" % (
            residual_stem, common_affine, residual_apply_args_common
        )
        residual_apply_iso = "%s_jacobian_action_isoparametric_mesh_soa(%s, %s)" % (
            residual_stem, common_iso, residual_apply_args_common
        )
        cases["apply"].append(
            _coupled_case(
                element,
                block_size,
                residual_apply_setup,
                (
                    "                    int status = impl_->apply_uses_affine ? %s : %s;\n"
                    "                    if (status != SFEM_SUCCESS) return status;\n"
                    "                    return impl_->jacobian_action_uses_affine ? %s : %s;"
                ) % (energy_apply_affine, energy_apply_iso, residual_apply_affine, residual_apply_iso),
            )
        )

        energy_objective_args = ", ".join(
            _nonempty(
                *_coupled_energy_field_args(
                    energy_objective_dependencies,
                    block_size,
                    current=energy_data,
                ),
                "impl_->element_values.get()",
            )
        )
        energy_objective_affine = "%s_objective_affine_mesh_soa(%s%s, %s)" % (
            energy_stem, common_affine, energy_objective_params, energy_objective_args
        )
        energy_objective_iso = "%s_objective_isoparametric_mesh_soa(%s%s, %s)" % (
            energy_stem, common_iso, energy_objective_params, energy_objective_args
        )
        cases["objective"].append(
            """                case smesh::%(element)s:
                    status = impl_->objective_uses_affine ? %(affine)s : %(isoparametric)s;
                    break;""" % {
                "element": _mesh_element_name(element),
                "affine": energy_objective_affine,
                "isoparametric": energy_objective_iso,
            }
        )
    return cases


def _coupled_case(element, block_size, setup_lines, body):
    return """                case smesh::%(element)s: {
                    static constexpr ptrdiff_t FIELD_STRIDE = %(block_size)d;
%(setup)s
%(body)s
                }""" % {
        "element": _mesh_element_name(element),
        "block_size": block_size,
        "setup": "\n".join(setup_lines),
        "body": body,
    }


def _coupled_parameter_array_lines(defaults):
    lines = []
    for index, (name, _) in enumerate(defaults):
        lines.append(
            '            values[%d] = parameters.require_real_value("%s");'
            % (index, name)
        )
    if not lines:
        lines.append("            values[0] = 0;")
    return "\n".join(lines)


def _coupled_block_size_lines(systems_by_dim):
    lines = []
    for dim in sorted(systems_by_dim):
        fields = systems_by_dim[dim].fields
        block_size = sum(int(field.components) for field in fields)
        lines.append("                case %d: return %d;" % (dim, block_size))
    return "\n".join(lines)


def _field_offsets(fields):
    offsets = {}
    offset = 0
    for field in fields:
        offsets[field.name] = offset
        offset += int(field.components)
    return offsets


def _component_offsets(base, offset, components):
    return ", ".join("%s + %d" % (base, offset + component) for component in components)


def _coupled_energy_field_args(dependencies, block_size, current=None, direction=None):
    args = []
    if current is not None and getattr(dependencies, "current", False):
        args.extend((str(block_size), current))
    if direction is not None and getattr(dependencies, "direction", False):
        args.extend((str(block_size), direction))
    return tuple(args)


def _metadata_parameter_args(parameters, parameter_index):
    parameters = _dependency_parameters(parameters)
    names = set()
    for parameter in parameters or ():
        name = str(parameter)
        if name in parameter_index:
            names.add(name)
    return "".join(
        ", storage[%d]" % parameter_index[name]
        for name in sorted(names, key=lambda value: parameter_index[value])
    )


def _dependency_parameter_args(parameters, parameter_index):
    parameters = _dependency_parameters(parameters)
    names = []
    for parameter in parameters or ():
        name = str(parameter)
        if name in parameter_index and name not in names:
            names.append(name)
    return "".join(", storage[%d]" % parameter_index[name] for name in names)


def _dependency_storage_args(parameters, parameter_index):
    parameters = _dependency_parameters(parameters)
    names = []
    for parameter in parameters or ():
        name = str(parameter)
        if name in parameter_index and name not in names:
            names.append(name)
    return tuple("storage[%d]" % parameter_index[name] for name in names)


def _dependency_parameters(dependencies):
    return tuple(getattr(dependencies, "parameters", dependencies or ()))


def _dependency_domain_parameter_args(dependencies):
    return tuple(
        'domain.parameters->require_real_value("%s")' % str(parameter)
        for parameter in _dependency_parameters(getattr(dependencies, "parameters", ()))
    )


def _compatible_element_for_field(element, field):
    if hasattr(element, "element_for_field"):
        return element.element_for_field(
            getattr(field, "family", "") or getattr(field, "name", "")
        )
    return _element_name(element)


def _nonempty(*values):
    return tuple(str(value) for value in values if str(value))


def _boundary_residual_parameter_names(collection, available_parameters):
    dependencies = collection.form_metadata(_form_order_one()).dependencies
    used = {
        str(symbol)
        for symbol in dependencies.parameters
        if str(symbol) in available_parameters
    }
    return tuple(
        str(symbol)
        for symbol in collection.source.parameters
        if str(symbol) in used
    )


def _form_order_zero():
    from codegen.framework.symbolic.forms import FormOrder

    return FormOrder.ZERO


def _form_order_one():
    from codegen.framework.symbolic.forms import FormOrder

    return FormOrder.ONE


def _form_order_two():
    from codegen.framework.symbolic.forms import FormOrder

    return FormOrder.TWO


def _residual_parameter_array_lines(parameter_names_by_dim):
    lines = ["            switch (dim) {"]
    for dim in sorted(parameter_names_by_dim):
        lines.append("                case %d:" % dim)
        for name in parameter_names_by_dim[dim]:
            lines.append(
                '                    values[index++] = parameters.require_real_value("%s");'
                % name
            )
        lines.append("                    break;")
    lines.extend(
        [
            "                default:",
            '                    SFEM_ERROR("unsupported spatial dimension %d for generated residual parameters\\n", dim);',
            "                    break;",
            "            }",
        ]
    )
    return "\n".join(lines)


def _residual_block_size_lines(block_size_by_dim):
    lines = ["            switch (dim) {"]
    for dim in sorted(block_size_by_dim):
        lines.append("                case %d: return %d;" % (dim, block_size_by_dim[dim]))
    lines.extend(
        [
            "                default:",
            '                    SFEM_ERROR("unsupported spatial dimension %d for generated residual block size\\n", dim);',
            "                    return 0;",
            "            }",
        ]
    )
    return "\n".join(lines)


def _residual_soa_view_declarations(fields, base, suffix, scalar_type):
    lines = []
    offset = 0
    for field in fields:
        components = int(field.components)
        name = _safe_identifier("%s_%s" % (field.name, suffix))
        if components == 1:
            lines.append(
                "                    %s *const SFEM_RESTRICT %s = %s + %d;"
                % (scalar_type, name, base, offset)
            )
        else:
            entries = ", ".join("%s + %d" % (base, offset + component) for component in range(components))
            lines.append(
                "                    %s *const SFEM_RESTRICT %s[%d] = {%s};"
                % (scalar_type, name, components, entries)
            )
        offset += components
    return lines


def _residual_soa_field_argument_names(fields, suffix):
    return tuple(
        _safe_identifier("%s_%s" % (field.name, suffix))
        for field in fields
    )


def _boundary_soa_component_argument_names(fields, suffix):
    names = []
    for field in fields:
        components = int(field.components)
        name = _safe_identifier("%s_%s" % (field.name, suffix))
        if components == 1:
            names.append(name)
        else:
            names.extend("%s[%d]" % (name, component) for component in range(components))
    return tuple(names)


def _residual_soa_case(element, function, arguments, field_stride, setup_lines):
    return """                case smesh::%(element)s: {
                    static constexpr ptrdiff_t FIELD_STRIDE = %(field_stride)d;
%(setup)s
                    return %(function)s(%(arguments)s);
                }""" % {
        "element": _mesh_element_name(element),
        "function": function,
        "arguments": arguments,
        "field_stride": field_stride,
        "setup": "\n".join(setup_lines),
    }


def _boundary_residual_soa_case(element, function, arguments, field_stride, setup_lines):
    return """                    case smesh::%(element)s: {
                        static constexpr ptrdiff_t FIELD_STRIDE = %(field_stride)d;
%(setup)s
                        status |= %(function)s(%(arguments)s);
                        break;
                    }""" % {
        "element": _mesh_element_name(element),
        "function": function,
        "arguments": arguments,
        "field_stride": field_stride,
        "setup": "\n".join(setup_lines),
    }


def _residual_dual_soa_case(
        element,
        flag,
        affine_function,
        affine_arguments,
        isoparametric_function,
        isoparametric_arguments,
        field_stride,
        setup_lines,
        affine_unit_function=None,
        affine_unit_arguments=None,
        affine_unit_condition=None):
    if affine_unit_function:
        body = """                    if (impl_->%(flag)s) {
                        if (%(affine_unit_condition)s) {
                            return %(affine_unit_function)s(%(affine_unit_arguments)s);
                        }
                        return %(affine_function)s(%(affine_arguments)s);
                    }
                    return %(isoparametric_function)s(%(isoparametric_arguments)s);""" % {
            "flag": flag,
            "affine_unit_condition": affine_unit_condition,
            "affine_unit_function": affine_unit_function,
            "affine_unit_arguments": affine_unit_arguments,
            "affine_function": affine_function,
            "affine_arguments": affine_arguments,
            "isoparametric_function": isoparametric_function,
            "isoparametric_arguments": isoparametric_arguments,
        }
    else:
        body = """                    return impl_->%(flag)s ? %(affine_function)s(%(affine_arguments)s) : %(isoparametric_function)s(%(isoparametric_arguments)s);""" % {
            "flag": flag,
            "affine_function": affine_function,
            "affine_arguments": affine_arguments,
            "isoparametric_function": isoparametric_function,
            "isoparametric_arguments": isoparametric_arguments,
        }
    return """                case smesh::%(element)s: {
                    static constexpr ptrdiff_t FIELD_STRIDE = %(field_stride)d;
%(setup)s
%(body)s
                }""" % {
        "element": _mesh_element_name(element),
        "field_stride": field_stride,
        "setup": "\n".join(setup_lines),
        "body": body,
    }


def _safe_identifier(name):
    return re.sub(r"[^0-9A-Za-z_]", "_", str(name))


def _dispatch_sources(material, elements, c_abi_header, kernel_sources):
    declarations = _extract_c_abi_declarations(kernel_sources, public_only=False)
    groups = _dispatch_groups(material, elements, declarations)
    if not groups:
        return {}

    lines = [
        '#include "%s"' % c_abi_header,
        "#include <cstdio>",
        "",
        "#ifndef SFEM_SUCCESS",
        "#define SFEM_SUCCESS 0",
        "#endif",
        "#ifndef SFEM_FAILURE",
        "#define SFEM_FAILURE 1",
        "#endif",
        "#ifndef SFEM_CODEGEN_PUBLIC_C_ABI",
        "#define SFEM_CODEGEN_PUBLIC_C_ABI",
        "#endif",
        "",
    ]
    private_declarations = []
    for group in groups:
        for variant in group["variants"]:
            private_declarations.append(variant["declaration"])
    lines.extend(_unique(private_declarations))
    if private_declarations:
        lines.append("")

    for group in groups:
        lines.extend(_dispatch_function_lines(group))

    return {"op/sfem_%s_dispatch.cpp" % material.op_name: "\n".join(lines) + "\n"}


def _dispatch_groups(material, elements, declarations):
    element_names = {
        _element_name(element).lower(): (_mesh_element_name(element), _element_dim(element))
        for element in elements
    }
    groups = {}
    for declaration in declarations:
        name = _c_abi_function_name(declaration)
        if not name or not declaration.startswith('extern "C" int '):
            continue
        mapped = _dispatch_mapping(material.name, name, element_names)
        if mapped is None:
            continue
        dispatch_name, mesh_element, dim = mapped
        params = _c_abi_parameters(declaration)
        if not params:
            continue
        key = (dispatch_name, tuple(params))
        groups.setdefault(
            key,
            {
                "name": dispatch_name,
                "params": tuple(params),
                "dim": dim,
                "variants": [],
            },
        )["variants"].append(
            {
                "mesh_element": mesh_element,
                "function": name,
                "declaration": declaration,
            }
        )

    ordered = []
    emitted_names = set()
    for _, group in sorted(groups.items(), key=lambda item: item[0][0]):
        if group["name"] in emitted_names:
            continue
        emitted_names.add(group["name"])
        group["variants"] = tuple(
            sorted(group["variants"], key=lambda item: item["mesh_element"])
        )
        ordered.append(group)
    return tuple(ordered)


def _dispatch_mapping(material_name, function_name, element_names):
    prefix = "%s_" % material_name
    if not function_name.startswith(prefix):
        return None
    suffix = function_name[len(prefix) :]
    for element_name, (mesh_element, dim) in sorted(
        element_names.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        element_prefix = "%s_" % element_name
        dispatch_prefix = ""
        if suffix.startswith(element_prefix):
            op_suffix = suffix[len(element_prefix) :]
        else:
            element_marker = "_%s_" % element_name
            marker_index = suffix.find(element_marker)
            if marker_index < 0:
                continue
            dispatch_prefix = suffix[:marker_index]
            op_suffix = suffix[marker_index + len(element_marker) :]
        repeated_prefix = "%s_" % element_name
        if op_suffix.startswith(repeated_prefix):
            op_suffix = op_suffix[len(repeated_prefix) :]
        dispatch_suffix = _insert_dispatch_dimension(op_suffix, dim)
        if dispatch_suffix is None:
            return None
        if dispatch_prefix:
            dispatch_suffix = "%s_%s" % (dispatch_prefix, dispatch_suffix)
        return "%s_%s" % (material_name, dispatch_suffix), mesh_element, dim
    return None


def _insert_dispatch_dimension(op_suffix, dim):
    for marker in ("_affine_", "_isoparametric_", "_sideset_"):
        index = op_suffix.find(marker)
        if index >= 0:
            return "%s_%dd%s" % (op_suffix[:index], dim, op_suffix[index:])
    return None


def _c_abi_parameters(declaration):
    begin = declaration.find("(")
    end = declaration.rfind(")")
    if begin < 0 or end < begin:
        return ()
    body = declaration[begin + 1 : end].strip()
    if not body or body == "void":
        return ()
    return tuple(_split_c_parameters(body))


def _split_c_parameters(body):
    params = []
    current = []
    depth = 0
    for char in body:
        if char == "," and depth == 0:
            params.append("".join(current).strip())
            current = []
            continue
        current.append(char)
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
    if current:
        params.append("".join(current).strip())
    return params


def _dispatch_function_lines(group):
    params = ("const smesh::ElemType element_type",) + tuple(group["params"])
    arg_names = tuple(_c_parameter_name(param) for param in group["params"])
    lines = [
        'SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int %s(' % group["name"],
    ]
    for index, param in enumerate(params):
        lines.append("        %s%s" % (param, "," if index + 1 < len(params) else ""))
    lines.extend(
        [
            ") {",
            "    switch (element_type) {",
        ]
    )
    for variant in group["variants"]:
        lines.extend(
            [
                "        case smesh::%s:" % variant["mesh_element"],
                "            return %s(%s);" % (variant["function"], ", ".join(arg_names)),
            ]
        )
    lines.extend(
        [
            "        default:",
            '            std::fprintf(stderr, "%s does not support element type %%d\\n", (int)element_type);'
            % group["name"],
            "            return SFEM_FAILURE;",
            "    }",
            "}",
            "",
        ]
    )
    return lines


def _c_parameter_name(param):
    cleaned = param.strip()
    cleaned = cleaned.replace(" SFEM_RESTRICT", "")
    cleaned = cleaned.replace("SFEM_RESTRICT ", "")
    cleaned = cleaned.rstrip()
    match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[[^\]]*\])?$", cleaned)
    if not match:
        raise ValueError("could not extract C parameter name from '%s'" % param)
    return match.group(1)


def _unique(values):
    seen = set()
    ret = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ret.append(value)
    return ret


def _c_abi_header(material, kernel_sources):
    declarations = _extract_c_abi_declarations(kernel_sources, public_only=True)
    body = "\n\n".join(declarations)
    if body:
        body += "\n"
    matrix_formats_include = (
        '#include "../matrix_formats.hpp"\n'
        if "sfem_MatrixAssemblyDiagnostics" in body
        else ""
    )
    return """#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__has_include)
#if __has_include("sfem_base.hpp")
#include "sfem_base.hpp"
#define SFEM_CODEGEN_OP_HAS_SFEM_BASE
#endif
#endif

#ifndef SFEM_CODEGEN_OP_HAS_SFEM_BASE
typedef ptrdiff_t idx_t;
typedef ptrdiff_t element_idx_t;
typedef ptrdiff_t count_t;
typedef double real_t;
typedef double geom_t;
#endif

#ifndef SFEM_RESTRICT
#define SFEM_RESTRICT __restrict__
#endif

#include "../kernel_diagnostics.hpp"
%(matrix_formats_include)s
%(smesh_include)s

#ifndef SFEM_CODEGEN_PUBLIC_C_ABI
#define SFEM_CODEGEN_PUBLIC_C_ABI
#endif

%(body)s""" % {
        "body": body,
        "matrix_formats_include": matrix_formats_include,
        "smesh_include": '#include "smesh_mesh.hpp"' if "smesh::ElemType" in body else "",
    }


def _registration_source(material, wrapper_header):
    function = _registration_function(material)
    return """#include "%(header)s"
#include "sfem_OpFactory.hpp"

namespace sfem {
    void %(function)s() {
        Factory::register_op("%(op)s", %(op)s::create);
        Factory::register_op("ss:%(op)s", %(op)s::create);
    }
}  // namespace sfem
""" % {
        "header": os.path.basename(wrapper_header),
        "function": function,
        "op": material.op_name,
    }


def _registration_function(material):
    return "register_%s_generated_op" % _safe_identifier(material.op_name)


def _op_manifest(material, kernel_sources, wrapper_header, wrapper_source, registration_source, c_abi_header):
    declarations = _extract_c_abi_declarations(kernel_sources, public_only=True)
    c_abi = [
        {
            "name": _c_abi_function_name(declaration),
            "declaration": declaration,
        }
        for declaration in declarations
    ]
    manifest = {
        "schema": "sfem.generated_op_manifest.v1",
        "material": material.name,
        "op_name": material.op_name,
        "wrapper": {
            "header": wrapper_header,
            "source": wrapper_source,
            "c_abi_header": c_abi_header,
        },
        "registration": {
            "source": registration_source,
            "function": "sfem::%s" % _registration_function(material),
            "operator_name": material.op_name,
        },
        "factory": {
            "class": "sfem::%s" % material.op_name,
            "create": "sfem::%s::create" % material.op_name,
            "create_from_yaml": "sfem::%s::create_from_yaml" % material.op_name,
        },
        "generated_include_paths": _generated_include_paths(kernel_sources),
        "matrix_formats": _matrix_format_sources(kernel_sources),
        "runtime_operations": _runtime_operations(c_abi),
        "c_abi": c_abi,
    }
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def _generated_include_paths(kernel_sources):
    paths = set([".", "op"])
    for path in kernel_sources:
        if path.startswith("op/"):
            continue
        if not path.endswith((".hpp", ".cuh", ".cpp", ".cu")):
            continue
        dirname = os.path.dirname(path)
        paths.add(dirname if dirname else ".")
    return tuple(sorted(paths))


def _matrix_format_sources(kernel_sources):
    entries = []
    for path in sorted(kernel_sources):
        if path.endswith("_matrix_format_operator.cpp"):
            entries.append(
                {
                    "source": path,
                    "header": "matrix_formats.hpp",
                }
            )
    return tuple(entries)


def _extract_c_abi_declarations(kernel_sources, public_only=False):
    declarations = {}
    for path, source in sorted(kernel_sources.items()):
        if not path.endswith((".cpp", ".hpp")) or path.startswith("op/"):
            if not (public_only and path.endswith("_dispatch.cpp")):
                continue
        offset = 0
        while True:
            start = source.find('extern "C"', offset)
            if start < 0:
                break
            brace = source.find("{", start)
            semicolon = source.find(";", start)
            if semicolon >= 0 and (brace < 0 or semicolon < brace):
                declaration = source[start:semicolon + 1]
                offset = semicolon + 1
            elif brace >= 0:
                declaration = source[start:brace].rstrip() + ";"
                offset = brace + 1
            else:
                break
            name = _c_abi_function_name(declaration)
            declaration_prefix = source[max(0, start - 32) : start]
            if public_only and "SFEM_CODEGEN_PUBLIC_C_ABI" not in declaration_prefix:
                is_int_kernel = declaration.startswith('extern "C" int ')
                is_metadata = name and "_matrix_assembly_" in name
                if is_int_kernel and not is_metadata:
                    continue
            if name and name not in declarations:
                declarations[name] = declaration
    return tuple(declarations[name] for name in sorted(declarations))


def _c_abi_function_name(declaration):
    match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", declaration)
    return match.group(1) if match else None


def _c_abi_function_exists(kernel_sources, function_name, public_only=False):
    if not kernel_sources:
        return False
    for declaration in _extract_c_abi_declarations(kernel_sources, public_only=public_only):
        if _c_abi_function_name(declaration) == function_name:
            return True
    return False


_RUNTIME_OPERATION_MARKERS = (
    ("jacobian_action", "_jacobian_action_"),
    ("hessian_bsr", "_hessian_bsr_"),
    ("hessian_coo_triplet", "_hessian_coo_triplet_"),
    ("hessian_coo", "_hessian_coo_"),
    ("hessian_crs", "_hessian_crs_"),
    ("hessian_dia", "_hessian_dia_"),
    ("hessian_patch", "_hessian_patch_"),
    ("bsr_apply", "_bsr_apply_"),
    ("dia_apply", "_dia_apply_"),
    ("patch_apply", "_patch_apply_"),
    ("boundary_residual", "_boundary_residual_"),
    ("objective_steps", "_objective_steps_"),
    ("objective", "_objective_"),
    ("gradient", "_gradient_"),
    ("apply", "_apply_"),
    ("residual", "_residual_"),
)

_RUNTIME_VARIANT_SUFFIXES = (
    ("affine", "_affine_mesh_soa"),
    ("isoparametric", "_isoparametric_mesh_soa"),
    ("sideset", "_sideset_soa"),
)

_AFFINE_OPTION_ALIASES = {
    "objective_uses_affine": (
        "ASSUME_AFFINE_OBJECTIVE",
        "objective_assume_affine",
    ),
    "gradient_uses_affine": (
        "ASSUME_AFFINE_GRADIENT",
        "gradient_assume_affine",
    ),
    "apply_uses_affine": (
        "ASSUME_AFFINE_HESSIAN_ACTION",
        "hessian_action_assume_affine",
        "ASSUME_AFFINE_APPLY",
        "apply_assume_affine",
    ),
    "residual_uses_affine": (
        "ASSUME_AFFINE_RESIDUAL",
        "residual_assume_affine",
        "ASSUME_AFFINE_GRADIENT",
        "gradient_assume_affine",
    ),
    "jacobian_action_uses_affine": (
        "ASSUME_AFFINE_JACOBIAN_ACTION",
        "jacobian_action_assume_affine",
        "ASSUME_AFFINE_APPLY",
        "apply_assume_affine",
    ),
}


def _runtime_operations(c_abi):
    variants_by_operation = {}
    seen = set()
    for entry in c_abi:
        name = entry["name"]
        operation, target = _runtime_operation_and_target(name)
        variant, scalar_type = _runtime_variant_and_scalar_type(name)
        if operation is None or variant is None:
            continue
        key = (operation, variant, scalar_type, name)
        if key in seen:
            continue
        seen.add(key)
        variants_by_operation.setdefault(operation, []).append(
            {
                "variant": variant,
                "scalar_type": scalar_type,
                "target": target,
                "function": name,
            }
        )
    return tuple(
        {
            "name": operation,
            "variants": tuple(
                sorted(
                    variants,
                    key=lambda item: (
                        item["variant"],
                        item["scalar_type"],
                        item["target"],
                        item["function"],
                    ),
                )
            ),
        }
        for operation, variants in sorted(variants_by_operation.items())
    )


def _runtime_operation_and_target(name):
    for operation, marker in _RUNTIME_OPERATION_MARKERS:
        marker_index = name.find(marker)
        if marker_index >= 0:
            return operation, name[:marker_index]
    return None, None


def _runtime_variant_and_scalar_type(name):
    for variant, suffix in _RUNTIME_VARIANT_SUFFIXES:
        if name.endswith(suffix):
            return variant, "real_t"
        if name.endswith("%s_float" % suffix):
            return variant, "float"
    return None, None


def _performance_case(
    element,
    diagnostics,
    count_expression="domain.block->n_elements()",
    affine_flags=None,
):
    diagnostics = tuple(diagnostics)
    if affine_flags is None:
        affine_flags = (None,) * len(diagnostics)
    else:
        affine_flags = tuple(affine_flags)
    if len(affine_flags) != len(diagnostics):
        raise ValueError("performance diagnostics and affine flags length mismatch")
    diagnostic_entries = tuple(
        {"name": diagnostic, "affine_flag": affine_flag}
        for diagnostic, affine_flag in dict.fromkeys(zip(diagnostics, affine_flags))
    )
    return {
        "element": element,
        "diagnostics": diagnostic_entries,
        "count": count_expression,
    }


def _performance_methods(op_name, cases_by_method):
    methods = []
    for method in ("value", "gradient", "apply"):
        cases = tuple(cases_by_method.get(method, ()))
        methods.append(_performance_flops_method(op_name, method, cases))
        methods.append(_performance_bytes_method(op_name, method, cases))
    return "\n\n".join(methods)


def _performance_flops_method(op_name, method, cases):
    return """    double %(op)s::flops_%(method)s() const {
        double total = 0;
        if (!impl_->domains) {
            return total;
        }

        impl_->domains->iterate([&](const OpDomain &domain) {
            switch (domain.element_type) {
%(cases)s
                default:
                    break;
            }
            return SFEM_SUCCESS;
        });

        return total;
    }""" % {
        "op": op_name,
        "method": method,
        "cases": _performance_flops_cases(cases),
    }


def _performance_bytes_method(op_name, method, cases):
    return """    size_t %(op)s::memory_traffic_bytes_%(method)s() const {
        size_t total = 0;
        if (!impl_->domains) {
            return total;
        }

        impl_->domains->iterate([&](const OpDomain &domain) {
            switch (domain.element_type) {
%(cases)s
                default:
                    break;
            }
            return SFEM_SUCCESS;
        });

        return total;
    }""" % {
        "op": op_name,
        "method": method,
        "cases": _performance_bytes_cases(cases),
    }


def _performance_flops_cases(cases):
    lines = []
    for case in cases:
        lines.append("                case smesh::%s: {" % _mesh_element_name(case["element"]))
        lines.append("                    const ptrdiff_t nelements = %s;" % case["count"])
        for diagnostic in case["diagnostics"]:
            name = diagnostic["name"]
            affine_flag = diagnostic["affine_flag"]
            if affine_flag is None:
                lines.append(
                    "                    total += sfem::codegen::KernelDiagnostics_total_flops(%s(), nelements);"
                    % name
                )
            else:
                lines.append(
                    "                    total += impl_->%s ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(%s(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(%s(), nelements);"
                    % (affine_flag, name, name)
                )
        lines.append("                    break;")
        lines.append("                }")
    return "\n".join(lines)


def _performance_bytes_cases(cases):
    lines = []
    for case in cases:
        lines.append("                case smesh::%s: {" % _mesh_element_name(case["element"]))
        lines.append("                    const ptrdiff_t nelements = %s;" % case["count"])
        for diagnostic in case["diagnostics"]:
            name = diagnostic["name"]
            affine_flag = diagnostic["affine_flag"]
            if affine_flag is None:
                lines.append(
                    "                    total += sfem::codegen::KernelDiagnostics_total_bytes(%s(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));"
                    % name
                )
            else:
                lines.append(
                    "                    total += impl_->%s ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(%s(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(%s(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));"
                    % (affine_flag, name, name)
                )
        lines.append("                    break;")
        lines.append("                }")
    return "\n".join(lines)


def _affine_option_entries(*flags, owner="impl_"):
    lines = []
    for flag in flags:
        for alias in _AFFINE_OPTION_ALIASES[flag]:
            lines.append('            {"%s", &%s->%s},' % (alias, owner, flag))
    return "\n".join(lines)


def _energy_field_args(dependencies, dim, components, current=None, direction=None):
    args = []
    if current is not None and (
        getattr(dependencies, "current", False) if dependencies is not None else True
    ):
        args.extend((dim, _offsets(current, components)))
    if direction is not None and (
        getattr(dependencies, "direction", False) if dependencies is not None else True
    ):
        args.extend((dim, _offsets(direction, components)))
    return tuple(args)


def _energy_output_args(dim, components):
    return (dim, _offsets("out", components))


def _energy_declaration_field_args(dependencies, dim, components, current=False, direction=False):
    args = []
    vectors = "".join(", const real_t *" for _ in components)
    if current and (
        getattr(dependencies, "current", False) if dependencies is not None else True
    ):
        args.append("ptrdiff_t%s" % vectors)
    if direction and (
        getattr(dependencies, "direction", False) if dependencies is not None else True
    ):
        args.append("ptrdiff_t%s" % vectors)
    return "".join(", %s" % arg for arg in args)


def _hyperelastic_declarations(stem, dim, parameters, dependencies=None):
    components = _components(dim)
    if dependencies is None:
        dependencies = (None, None, None)
    objective_dependencies, gradient_dependencies, apply_dependencies = dependencies
    parameter_decl = "".join(", const real_t %s" % name for name in parameters)
    vectors = "".join(", const real_t *" for _ in components)
    outputs = "".join(", real_t *" for _ in components)
    isoparametric_common = (
        "ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *"
        + parameter_decl
    )
    affine_common = (
        "ptrdiff_t, ptrdiff_t, idx_t **"
        + "".join(", const geom_t *" for _ in range(dim * dim))
        + ", const geom_t *"
        + parameter_decl
    )
    affine_aos_unit_common = (
        "ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *"
        + parameter_decl
    )
    objective_steps_field_decl = (
        _energy_declaration_field_args(
            objective_dependencies,
            dim,
            components,
            current=True,
        )
        + _energy_declaration_field_args(
            apply_dependencies,
            dim,
            components,
            direction=True,
        )
    )
    return (
        "int %s_objective_isoparametric_mesh_soa(%s%s, real_t *);"
        % (
            stem,
            isoparametric_common,
            _energy_declaration_field_args(
                objective_dependencies,
                dim,
                components,
                current=True,
            ),
        ),
        "int %s_gradient_isoparametric_mesh_soa(%s%s, ptrdiff_t%s);"
        % (
            stem,
            isoparametric_common,
            _energy_declaration_field_args(
                gradient_dependencies,
                dim,
                components,
                current=True,
            ),
            outputs,
        ),
        "int %s_apply_isoparametric_mesh_soa(%s%s, ptrdiff_t%s);"
        % (
            stem,
            isoparametric_common,
            _energy_declaration_field_args(
                apply_dependencies,
                dim,
                components,
                current=True,
                direction=True,
            ),
            outputs,
        ),
        "int %s_objective_steps_isoparametric_mesh_soa(%s%s, ptrdiff_t, const real_t *, real_t *);"
        % (
            stem,
            isoparametric_common,
            objective_steps_field_decl,
        ),
        "int %s_objective_affine_mesh_soa(%s%s, real_t *);"
        % (
            stem,
            affine_common,
            _energy_declaration_field_args(
                objective_dependencies,
                dim,
                components,
                current=True,
            ),
        ),
        "int %s_gradient_affine_mesh_soa(%s%s, ptrdiff_t%s);"
        % (
            stem,
            affine_common,
            _energy_declaration_field_args(
                gradient_dependencies,
                dim,
                components,
                current=True,
            ),
            outputs,
        ),
        "int %s_gradient_affine_mesh_soa_aos_unit(%s%s, ptrdiff_t%s);"
        % (
            stem,
            affine_aos_unit_common,
            _energy_declaration_field_args(
                gradient_dependencies,
                dim,
                components,
                current=True,
            ),
            outputs,
        ),
        "int %s_apply_affine_mesh_soa(%s%s, ptrdiff_t%s);"
        % (
            stem,
            affine_common,
            _energy_declaration_field_args(
                apply_dependencies,
                dim,
                components,
                current=True,
                direction=True,
            ),
            outputs,
        ),
        "int %s_apply_affine_mesh_soa_aos_unit(%s%s, ptrdiff_t%s);"
        % (
            stem,
            affine_aos_unit_common,
            _energy_declaration_field_args(
                apply_dependencies,
                dim,
                components,
                current=True,
                direction=True,
            ),
            outputs,
        ),
        "int %s_objective_steps_affine_mesh_soa(%s%s, ptrdiff_t, const real_t *, real_t *);"
        % (
            stem,
            affine_common,
            objective_steps_field_decl,
        ),
    )


def _case(element, function, arguments):
    return """                case smesh::%(element)s:
                    return %(function)s(%(arguments)s);""" % {
        "element": _mesh_element_name(element),
        "function": function,
        "arguments": arguments,
    }


def _residual_hessian_dispatch_body(
    material_name,
    operation,
    kernel_sources,
    action_dependencies_by_dim,
    parameter_names_by_dim,
    fields_by_dim,
    block_size_by_dim,
    tail_args,
    indent,
):
    lines = ["%sconst int dim = mesh->spatial_dimension();" % indent]
    for dim in (2, 3):
        dependencies = action_dependencies_by_dim.get(dim)
        if dependencies is None:
            continue
        function = "%s_%s_%dd_isoparametric_mesh_soa" % (
            material_name,
            operation,
            dim,
        )
        prefix = "if" if not any(line.endswith("{") for line in lines) else "else if"
        lines.append("%s%s (dim == %d) {" % (indent, prefix, dim))
        lines.append(
            "%s    static constexpr ptrdiff_t FIELD_STRIDE = %d;"
            % (indent, block_size_by_dim[dim])
        )
        setup = []
        args = [
            "domain.element_type",
            "domain.block->n_elements()",
            "mesh->n_nodes()",
            "domain.block->elements()->data()",
            "points",
        ]
        parameter_index = {
            name: index for index, name in enumerate(parameter_names_by_dim[dim])
        }
        args.extend(_dependency_storage_args(dependencies.parameters, parameter_index))
        fields = fields_by_dim[dim]
        if dependencies.current:
            setup.extend(
                _residual_soa_view_declarations(
                    fields,
                    "current",
                    "data",
                    "const real_t",
                )
            )
            args.append("FIELD_STRIDE")
            args.extend(_residual_soa_field_argument_names(fields, "data"))
        if dependencies.previous:
            setup.extend(
                _residual_soa_view_declarations(
                    fields,
                    "previous",
                    "old_data",
                    "const real_t",
                )
            )
            args.append("FIELD_STRIDE")
            args.extend(_residual_soa_field_argument_names(fields, "old_data"))
        args.extend(tail_args)
        for line in setup:
            lines.append(line)
        if _c_abi_function_exists(kernel_sources, function, public_only=True):
            lines.append("%s    return %s(%s);" % (indent, function, ", ".join(args)))
        else:
            lines.append(
                '%s    SFEM_ERROR("%s %s %dd dispatch was not generated\\n");'
                % (indent, material_name, operation, dim)
            )
            lines.append("%s    return SFEM_FAILURE;" % indent)
        lines.append("%s}" % indent)
    lines.extend(
        [
            '%sSFEM_ERROR("%s %s does not support spatial dimension %%d\\n", dim);'
            % (indent, material_name, operation),
            "%sreturn SFEM_FAILURE;" % indent,
        ]
    )
    return "\n".join(lines)


def _element_condition(variable_name, mesh_elements):
    names = tuple(sorted(set(mesh_elements)))
    if not names:
        return "false"
    return " || ".join("%s == smesh::%s" % (variable_name, name) for name in names)


def _residual_apply_dispatch_body(
    material_name,
    operation,
    affine_flag,
    current_base,
    kernel_sources,
    dependencies_by_dim,
    parameter_names_by_dim,
    fields_by_dim,
    block_size_by_dim,
    affine_aos_elements_by_dim,
    affine_aos_unit_elements_by_dim,
    indent,
):
    lines = ["%sconst int dim = mesh->spatial_dimension();" % indent]
    for dim in (2, 3):
        dependencies = dependencies_by_dim.get(dim)
        if dependencies is None:
            continue
        prefix = "if" if not any(line.endswith("{") for line in lines) else "else if"
        lines.append("%s%s (dim == %d) {" % (indent, prefix, dim))
        lines.append(
            "%s    static constexpr ptrdiff_t FIELD_STRIDE = %d;"
            % (indent, block_size_by_dim[dim])
        )
        setup = []
        parameter_index = {
            name: index for index, name in enumerate(parameter_names_by_dim[dim])
        }
        common_args = [
            "domain.element_type",
            "domain.block->n_elements()",
            "mesh->n_nodes()",
            "domain.block->elements()->data()",
        ]
        field_args = []
        unit_field_args = []
        if dependencies.current:
            setup.extend(
                _residual_soa_view_declarations(
                    fields_by_dim[dim],
                    current_base,
                    "data",
                    "const real_t",
                )
            )
            field_args.append("FIELD_STRIDE")
            field_args.extend(_residual_soa_field_argument_names(fields_by_dim[dim], "data"))
            unit_field_args.extend(_residual_soa_field_argument_names(fields_by_dim[dim], "data"))
        if dependencies.previous:
            setup.extend(
                _residual_soa_view_declarations(
                    fields_by_dim[dim],
                    "previous",
                    "old_data",
                    "const real_t",
                )
            )
            field_args.append("FIELD_STRIDE")
            field_args.extend(_residual_soa_field_argument_names(fields_by_dim[dim], "old_data"))
        if dependencies.direction:
            setup.extend(
                _residual_soa_view_declarations(
                    fields_by_dim[dim],
                    "direction",
                    "direction_data",
                    "const real_t",
                )
            )
            field_args.append("FIELD_STRIDE")
            field_args.extend(
                _residual_soa_field_argument_names(fields_by_dim[dim], "direction_data")
            )
            unit_field_args.extend(
                _residual_soa_field_argument_names(fields_by_dim[dim], "direction_data")
            )
        setup.extend(
            _residual_soa_view_declarations(
                fields_by_dim[dim],
                "out",
                "out",
                "real_t",
            )
        )
        field_args.append("FIELD_STRIDE")
        field_args.extend(_residual_soa_field_argument_names(fields_by_dim[dim], "out"))
        unit_field_args.extend(_residual_soa_field_argument_names(fields_by_dim[dim], "out"))
        for line in setup:
            lines.append(line)

        storage_args = list(
            _dependency_storage_args(dependencies.parameters, parameter_index)
        )
        affine_soa = "%s_%s_%dd_affine_mesh_soa" % (material_name, operation, dim)
        affine_aos = "%s_%s_%dd_affine_mesh_soa_aos" % (material_name, operation, dim)
        affine_aos_unit = "%s_%s_%dd_affine_mesh_soa_aos_unit" % (
            material_name,
            operation,
            dim,
        )
        isop = "%s_%s_%dd_isoparametric_mesh_soa" % (material_name, operation, dim)
        packed_affine = "%s_%s_packed_%dd_affine_mesh_soa" % (
            material_name,
            operation,
            dim,
        )
        packed = "%s_%s_packed_%dd_isoparametric_mesh_soa" % (
            material_name,
            operation,
            dim,
        )
        aos_condition = _element_condition(
            "domain.element_type", affine_aos_elements_by_dim.get(dim, ())
        )
        unit_condition = _element_condition(
            "domain.element_type", affine_aos_unit_elements_by_dim.get(dim, ())
        )

        lines.append("%s    if (impl_->%s) {" % (indent, affine_flag))
        if operation == "jacobian_action" and _c_abi_function_exists(
            kernel_sources, packed_affine, public_only=True
        ):
            packed_affine_geometry_args = (
                _affine_metric_offsets(dim).split(", ")
                if _c_abi_function_uses_cached_metric(kernel_sources, packed_affine)
                else [*_affine_geometry_offsets(dim).split(", "), "determinant"]
            )
            lines.extend(
                [
                    "%s        if (impl_->space->has_packed_mesh()) {" % indent,
                    "%s            auto packed = impl_->space->packed_mesh();" % indent,
                    "%s            const int packed_block = packed_block_id_for_domain(*packed, *domain.block);" % indent,
                    "%s            if (packed_block >= 0) {" % indent,
                    "%s                auto packed_elements = packed->elements(packed_block);" % indent,
                    "%s                auto owned_nodes_ptr = packed->owned_nodes_ptr(packed_block);" % indent,
                    "%s                auto n_shared_nodes = packed->n_shared(packed_block);" % indent,
                    "%s                auto ghost_ptr = packed->ghost_ptr(packed_block);" % indent,
                    "%s                auto ghost_idx = packed->ghost_idx(packed_block);" % indent,
                    *(
                        [
                            "%s                if (domain.element_type == smesh::TET4) {" % indent,
                            "%s                    return laplace_tet4_jacobian_action_packed_affine_mesh_soa(%s);"
                            % (
                                indent,
                                ", ".join(
                                    [
                                        "packed->n_packs(packed_block)",
                                        "packed->n_elements_per_pack(packed_block)",
                                        "domain.block->n_elements()",
                                        "mesh->n_nodes()",
                                        "packed->max_nodes_per_pack()",
                                        "packed_elements->data()",
                                        "owned_nodes_ptr->data()",
                                        "n_shared_nodes->data()",
                                        "ghost_ptr->data()",
                                        "ghost_idx->data()",
                                        "geom_metric[0]",
                                        "geom_metric[1]",
                                        "geom_metric[2]",
                                        "geom_metric[3]",
                                        "geom_metric[4]",
                                        "geom_metric[5]",
                                        *storage_args,
                                        *field_args,
                                    ]
                                ),
                            ),
                            "%s                }" % indent,
                        ]
                        if material_name == "laplace"
                        and operation == "jacobian_action"
                        and dim == 3
                        and _c_abi_function_exists(
                            kernel_sources,
                            "laplace_tet4_jacobian_action_packed_affine_mesh_soa",
                        )
                        else []
                    ),
                    *(
                        [
                            "%s                if (domain.element_type == smesh::HEX8) {" % indent,
                            "%s                    uint16_t *proteus_elements[8] = {packed_elements->data()[0], packed_elements->data()[1], packed_elements->data()[3], packed_elements->data()[2], packed_elements->data()[4], packed_elements->data()[5], packed_elements->data()[7], packed_elements->data()[6]};" % indent,
                            "%s                    return laplace_proteus_hex8_private_metric_jacobian_action_packed_mesh_soa(%s);"
                            % (
                                indent,
                                ", ".join(
                                    [
                                        "packed->n_packs(packed_block)",
                                        "packed->n_elements_per_pack(packed_block)",
                                        "domain.block->n_elements()",
                                        "mesh->n_nodes()",
                                        "packed->max_nodes_per_pack()",
                                        "proteus_elements",
                                        "owned_nodes_ptr->data()",
                                        "n_shared_nodes->data()",
                                        "ghost_ptr->data()",
                                        "ghost_idx->data()",
                                        "geom_metric_aos",
                                        *storage_args,
                                        *field_args,
                                    ]
                                ),
                            ),
                            "%s                }" % indent,
                            "%s                if (domain.element_type == smesh::PROTEUS_HEX8) {" % indent,
                            "%s                    return laplace_proteus_hex8_private_metric_jacobian_action_packed_mesh_soa(%s);"
                            % (
                                indent,
                                ", ".join(
                                    [
                                        "packed->n_packs(packed_block)",
                                        "packed->n_elements_per_pack(packed_block)",
                                        "domain.block->n_elements()",
                                        "mesh->n_nodes()",
                                        "packed->max_nodes_per_pack()",
                                        "packed_elements->data()",
                                        "owned_nodes_ptr->data()",
                                        "n_shared_nodes->data()",
                                        "ghost_ptr->data()",
                                        "ghost_idx->data()",
                                        "geom_metric_aos",
                                        *storage_args,
                                        *field_args,
                                    ]
                                ),
                            ),
                            "%s                }" % indent,
                        ]
                        if material_name == "laplace"
                        and operation == "jacobian_action"
                        and dim == 3
                        and _c_abi_function_exists(
                            kernel_sources,
                            "laplace_proteus_hex8_private_metric_jacobian_action_packed_mesh_soa",
                        )
                        else []
                    ),
                    *(
                        [
                            "%s                if (domain.element_type == smesh::TET10) {" % indent,
                            "%s                    return laplace_tet10_private_metric_jacobian_action_packed_mesh_soa(%s);"
                            % (
                                indent,
                                ", ".join(
                                    [
                                        "packed->n_packs(packed_block)",
                                        "packed->n_elements_per_pack(packed_block)",
                                        "domain.block->n_elements()",
                                        "mesh->n_nodes()",
                                        "packed->max_nodes_per_pack()",
                                        "packed_elements->data()",
                                        "owned_nodes_ptr->data()",
                                        "n_shared_nodes->data()",
                                        "ghost_ptr->data()",
                                        "ghost_idx->data()",
                                        "geom_metric_aos",
                                        *storage_args,
                                        *field_args,
                                    ]
                                ),
                            ),
                            "%s                }" % indent,
                        ]
                        if material_name == "laplace"
                        and operation == "jacobian_action"
                        and dim == 3
                        and _c_abi_function_exists(
                            kernel_sources,
                            "laplace_tet10_private_metric_jacobian_action_packed_mesh_soa",
                        )
                        else []
                    ),
                    "%s                return %s(%s);"
                    % (
                        indent,
                        packed_affine,
                        ", ".join(
                            [
                                "domain.element_type",
                                "packed->n_packs(packed_block)",
                                "packed->n_elements_per_pack(packed_block)",
                                "domain.block->n_elements()",
                                "mesh->n_nodes()",
                                "packed->max_nodes_per_pack()",
                                "packed_elements->data()",
                                "owned_nodes_ptr->data()",
                                "n_shared_nodes->data()",
                                "ghost_ptr->data()",
                                "ghost_idx->data()",
                                *packed_affine_geometry_args,
                                *storage_args,
                                *field_args,
                            ]
                        ),
                    ),
                    "%s            }" % indent,
                    "%s        }" % indent,
                ]
            )
        if _c_abi_function_exists(kernel_sources, affine_aos_unit, public_only=True):
            lines.append(
                "%s        if ((%s) && storage[0] == real_t(1)) {"
                % (indent, unit_condition)
            )
            lines.append(
                "%s            return %s(%s);"
                % (
                    indent,
                    affine_aos_unit,
                    ", ".join(
                        [
                            *common_args,
                            "geom_metric_aos",
                            *unit_field_args,
                        ]
                    ),
                )
            )
            lines.append("%s        }" % indent)
        if _c_abi_function_exists(kernel_sources, affine_aos, public_only=True):
            lines.append("%s        if (%s) {" % (indent, aos_condition))
            lines.append(
                "%s            return %s(%s);"
                % (
                    indent,
                    affine_aos,
                    ", ".join(
                        [
                            *common_args,
                            "geom_metric_aos",
                            *storage_args,
                            *field_args,
                        ]
                    ),
                )
            )
            lines.append("%s        }" % indent)
        if _c_abi_function_exists(kernel_sources, affine_soa, public_only=True):
            lines.append(
                "%s        return %s(%s);"
                % (
                    indent,
                    affine_soa,
                    ", ".join(
                        [
                            *common_args,
                            *_affine_geometry_offsets(dim).split(", "),
                            "determinant",
                            *storage_args,
                            *field_args,
                        ]
                    ),
                )
            )
        else:
            lines.append(
                '%s        SFEM_ERROR("%s %s affine %dd dispatch was not generated\\n");'
                % (indent, material_name, operation, dim)
            )
            lines.append("%s        return SFEM_FAILURE;" % indent)
        lines.append("%s    }" % indent)
        if operation == "jacobian_action" and _c_abi_function_exists(
            kernel_sources, packed, public_only=True
        ):
            lines.extend(
                [
                    "%s    if (impl_->space->has_packed_mesh()) {" % indent,
                    "%s        auto packed = impl_->space->packed_mesh();" % indent,
                    "%s        const int packed_block = packed_block_id_for_domain(*packed, *domain.block);" % indent,
                    "%s        if (packed_block >= 0) {" % indent,
                    "%s            auto packed_elements = packed->elements(packed_block);" % indent,
                    "%s            auto owned_nodes_ptr = packed->owned_nodes_ptr(packed_block);" % indent,
                    "%s            auto n_shared_nodes = packed->n_shared(packed_block);" % indent,
                    "%s            auto ghost_ptr = packed->ghost_ptr(packed_block);" % indent,
                    "%s            auto ghost_idx = packed->ghost_idx(packed_block);" % indent,
                    "%s            return %s(%s);"
                    % (
                        indent,
                        packed,
                        ", ".join(
                            [
                                "domain.element_type",
                                "packed->n_packs(packed_block)",
                                "packed->n_elements_per_pack(packed_block)",
                                "domain.block->n_elements()",
                                "mesh->n_nodes()",
                                "packed->max_nodes_per_pack()",
                                "packed_elements->data()",
                                "owned_nodes_ptr->data()",
                                "n_shared_nodes->data()",
                                "ghost_ptr->data()",
                                "ghost_idx->data()",
                                "points",
                                *storage_args,
                                *field_args,
                            ]
                        ),
                    ),
                    "%s        }" % indent,
                    "%s    }" % indent,
                ]
            )
        if _c_abi_function_exists(kernel_sources, isop, public_only=True):
            lines.append(
                "%s    return %s(%s);"
                % (
                    indent,
                    isop,
                    ", ".join(
                        [
                            *common_args,
                            "points",
                            *storage_args,
                            *field_args,
                        ]
                    ),
                )
            )
        else:
            lines.append(
                '%s    SFEM_ERROR("%s %s isoparametric %dd dispatch was not generated\\n");'
                % (indent, material_name, operation, dim)
            )
            lines.append("%s    return SFEM_FAILURE;" % indent)
        lines.append("%s}" % indent)
    lines.extend(
        [
            '%sSFEM_ERROR("%s %s does not support spatial dimension %%d\\n", dim);'
            % (indent, material_name, operation),
            "%sreturn SFEM_FAILURE;" % indent,
        ]
    )
    return "\n".join(lines)


def _packed_two_pass_function(function):
    if "_packed_two_pass_" in function:
        return function
    return function.replace("_packed_", "_packed_two_pass_", 1)


def _packed_call_args_common():
    return [
        "packed->n_packs(packed_block)",
        "packed->n_elements_per_pack(packed_block)",
        "domain.block->n_elements()",
        "mesh->n_nodes()",
        "packed->max_nodes_per_pack()",
        "packed_elements->data()",
        "owned_nodes_ptr->data()",
        "n_shared_nodes->data()",
        "ghost_ptr->data()",
        "ghost_idx->data()",
    ]


def _packed_two_pass_extra_args():
    return [
        "packed->n_ghost_entries(packed_block)",
        "packed->n_ghost_reduce_rows(packed_block)",
        "ghost_reduce_ptr->data()",
        "ghost_reduce_idx->data()",
        "ghost_reduce_dest->data()",
        "impl_->packed_ghost_buf[packed_block]->data()",
    ]


def _hyperelastic_packed_return(indent, function, leading_args, trailing_args, kernel_sources):
    """leading_args usually ['domain.element_type']; trailing follows ghost_idx."""
    two_pass = _packed_two_pass_function(function)
    base = list(leading_args) + _packed_call_args_common()
    one_call = ", ".join(base + list(trailing_args))
    if not _c_abi_function_exists(kernel_sources, two_pass, public_only=True):
        return ["%sreturn %s(%s);" % (indent, function, one_call)]
    two_call = ", ".join(
        list(leading_args)
        + _packed_call_args_common()
        + _packed_two_pass_extra_args()
        + list(trailing_args)
    )
    return [
        "%sif (impl_->use_packed_two_pass) {" % indent,
        "%s    return %s(%s);" % (indent, two_pass, two_call),
        "%s}" % indent,
        "%sreturn %s(%s);" % (indent, function, one_call),
    ]


def _hyperelastic_apply_dispatch_body(material_name, kernel_sources, apply_dependencies_by_dim, indent):
    lines = [
        "%sconst int dim = mesh->spatial_dimension();" % indent,
    ]
    for dim in (2, 3):
        prefix = "if" if dim == 2 else "else if"
        dependencies = apply_dependencies_by_dim.get(dim)
        uses_current = getattr(dependencies, "current", True)
        uses_direction = getattr(dependencies, "direction", True)
        parameter_args = list(_dependency_domain_parameter_args(dependencies))
        current_args = ([str(dim)] + ["x + %d" % d for d in range(dim)]) if uses_current else []
        direction_args = ([str(dim)] + ["h + %d" % d for d in range(dim)]) if uses_direction else []
        output_args = [str(dim)] + ["out + %d" % d for d in range(dim)]
        affine = "%s_apply_%dd_affine_mesh_soa" % (material_name, dim)
        isop = "%s_apply_%dd_isoparametric_mesh_soa" % (material_name, dim)
        packed = "%s_apply_packed_%dd_isoparametric_mesh_soa" % (material_name, dim)
        packed_affine = "%s_apply_packed_%dd_affine_mesh_soa" % (material_name, dim)
        lines.append("%s%s (dim == %d) {" % (indent, prefix, dim))
        lines.append("%s    if (impl_->apply_uses_affine) {" % indent)
        if _c_abi_function_exists(kernel_sources, affine, public_only=True):
            if _c_abi_function_exists(kernel_sources, packed_affine, public_only=True):
                lines.extend(
                    [
                        "%s        if (impl_->space->has_packed_mesh()) {" % indent,
                        "%s            auto packed = impl_->space->packed_mesh();" % indent,
                        "%s            const int packed_block = packed_block_id_for_domain(*packed, *domain.block);" % indent,
                        "%s            if (packed_block >= 0) {" % indent,
                        "%s                auto packed_elements = packed->elements(packed_block);" % indent,
                        "%s                auto owned_nodes_ptr = packed->owned_nodes_ptr(packed_block);" % indent,
                        "%s                auto n_shared_nodes = packed->n_shared(packed_block);" % indent,
                        "%s                auto ghost_ptr = packed->ghost_ptr(packed_block);" % indent,
                        "%s                auto ghost_idx = packed->ghost_idx(packed_block);" % indent,
                        "%s                auto ghost_reduce_ptr = packed->ghost_reduce_ptr(packed_block);" % indent,
                        "%s                auto ghost_reduce_idx = packed->ghost_reduce_idx(packed_block);" % indent,
                        "%s                auto ghost_reduce_dest = packed->ghost_reduce_dest(packed_block);" % indent,
                    ]
                )
                lines.extend(
                    _hyperelastic_packed_return(
                        indent + "                ",
                        packed_affine,
                        ["domain.element_type"],
                        [
                            *("adjugate[%d]" % i for i in range(dim * dim)),
                            "determinant",
                            *parameter_args,
                            *current_args,
                            *direction_args,
                            *output_args,
                        ],
                        kernel_sources,
                    )
                )
                lines.extend(
                    [
                        "%s            }" % indent,
                        "%s        }" % indent,
                    ]
                )
            lines.append(
                "%s        return %s(%s);"
                % (
                    indent,
                    affine,
                    ", ".join(
                        [
                            "domain.element_type",
                            "domain.block->n_elements()",
                            "mesh->n_nodes()",
                            "domain.block->elements()->data()",
                            *("adjugate[%d]" % i for i in range(dim * dim)),
                            "determinant",
                            *parameter_args,
                            *current_args,
                            *direction_args,
                            *output_args,
                        ]
                    ),
                )
            )
        else:
            lines.append('%s        SFEM_ERROR("%s affine apply %dd dispatch was not generated\\n");' % (indent, material_name, dim))
            lines.append("%s        return SFEM_FAILURE;" % indent)
        lines.append("%s    }" % indent)
        if _c_abi_function_exists(kernel_sources, packed, public_only=True):
            lines.extend(
                [
                    "%s    if (impl_->space->has_packed_mesh()) {" % indent,
                    "%s        auto packed = impl_->space->packed_mesh();" % indent,
                    "%s        const int packed_block = packed_block_id_for_domain(*packed, *domain.block);" % indent,
                    "%s        if (packed_block >= 0) {" % indent,
                    "%s            auto packed_elements = packed->elements(packed_block);" % indent,
                    "%s            auto owned_nodes_ptr = packed->owned_nodes_ptr(packed_block);" % indent,
                    "%s            auto n_shared_nodes = packed->n_shared(packed_block);" % indent,
                    "%s            auto ghost_ptr = packed->ghost_ptr(packed_block);" % indent,
                    "%s            auto ghost_idx = packed->ghost_idx(packed_block);" % indent,
                    "%s            auto ghost_reduce_ptr = packed->ghost_reduce_ptr(packed_block);" % indent,
                    "%s            auto ghost_reduce_idx = packed->ghost_reduce_idx(packed_block);" % indent,
                    "%s            auto ghost_reduce_dest = packed->ghost_reduce_dest(packed_block);" % indent,
                ]
            )
            lines.extend(
                _hyperelastic_packed_return(
                    indent + "            ",
                    packed,
                    ["domain.element_type"],
                    [
                        "points",
                        *parameter_args,
                        *current_args,
                        *direction_args,
                        *output_args,
                    ],
                    kernel_sources,
                )
            )
            lines.extend(
                [
                    "%s        }" % indent,
                    "%s    }" % indent,
                ]
            )
        if _c_abi_function_exists(kernel_sources, isop, public_only=True):
            lines.append(
                "%s    return %s(%s);"
                % (
                    indent,
                    isop,
                    ", ".join(
                        [
                            "domain.element_type",
                            "domain.block->n_elements()",
                            "mesh->n_nodes()",
                            "domain.block->elements()->data()",
                            "points",
                            *parameter_args,
                            *current_args,
                            *direction_args,
                            *output_args,
                        ]
                    ),
                )
            )
        else:
            lines.append('%s    SFEM_ERROR("%s isoparametric apply %dd dispatch was not generated\\n");' % (indent, material_name, dim))
            lines.append("%s    return SFEM_FAILURE;" % indent)
        lines.append("%s}" % indent)
    lines.extend(
        [
            '%sSFEM_ERROR("%s apply does not support spatial dimension %%d\\n", dim);' % (indent, material_name),
            "%sreturn SFEM_FAILURE;" % indent,
        ]
    )
    return "\n".join(lines)


def _hyperelastic_gradient_packed_dispatch_body(material_name, kernel_sources, gradient_dependencies_by_dim, indent):
    affine_lines = [
        "%sif (impl_->gradient_uses_affine && impl_->space->has_packed_mesh()) {" % indent,
        "%s    auto packed = impl_->space->packed_mesh();" % indent,
        "%s    const int packed_block = packed_block_id_for_domain(*packed, *domain.block);" % indent,
        "%s    if (packed_block >= 0) {" % indent,
        "%s        auto packed_elements = packed->elements(packed_block);" % indent,
        "%s        auto owned_nodes_ptr = packed->owned_nodes_ptr(packed_block);" % indent,
        "%s        auto n_shared_nodes = packed->n_shared(packed_block);" % indent,
        "%s        auto ghost_ptr = packed->ghost_ptr(packed_block);" % indent,
        "%s        auto ghost_idx = packed->ghost_idx(packed_block);" % indent,
        "%s        auto ghost_reduce_ptr = packed->ghost_reduce_ptr(packed_block);" % indent,
        "%s        auto ghost_reduce_idx = packed->ghost_reduce_idx(packed_block);" % indent,
        "%s        auto ghost_reduce_dest = packed->ghost_reduce_dest(packed_block);" % indent,
        "%s        const int dim = mesh->spatial_dimension();" % indent,
    ]
    emitted_affine = False
    for dim in (2, 3):
        dependencies = gradient_dependencies_by_dim.get(dim)
        if dependencies is None:
            continue
        function = "%s_gradient_packed_%dd_affine_mesh_soa" % (material_name, dim)
        if not _c_abi_function_exists(kernel_sources, function, public_only=True):
            continue
        prefix = "if" if not emitted_affine else "else if"
        emitted_affine = True
        parameter_args = list(_dependency_domain_parameter_args(dependencies))
        current_args = (
            [str(dim)] + ["x + %d" % d for d in range(dim)]
            if getattr(dependencies, "current", True)
            else []
        )
        output_args = [str(dim)] + ["out + %d" % d for d in range(dim)]
        affine_lines.append("%s        %s (dim == %d) {" % (indent, prefix, dim))
        affine_lines.extend(
            _hyperelastic_packed_return(
                indent + "            ",
                function,
                ["domain.element_type"],
                [
                    *("adjugate[%d]" % i for i in range(dim * dim)),
                    "determinant",
                    *parameter_args,
                    *current_args,
                    *output_args,
                ],
                kernel_sources,
            )
        )
        affine_lines.append("%s        }" % indent)
    affine_lines.extend(
        [
            "%s    }" % indent,
            "%s}" % indent,
        ]
    )
    lines = [
        "%sif (!impl_->gradient_uses_affine && impl_->space->has_packed_mesh()) {" % indent,
        "%s    auto packed = impl_->space->packed_mesh();" % indent,
        "%s    const int packed_block = packed_block_id_for_domain(*packed, *domain.block);" % indent,
        "%s    if (packed_block >= 0) {" % indent,
        "%s        auto packed_elements = packed->elements(packed_block);" % indent,
        "%s        auto owned_nodes_ptr = packed->owned_nodes_ptr(packed_block);" % indent,
        "%s        auto n_shared_nodes = packed->n_shared(packed_block);" % indent,
        "%s        auto ghost_ptr = packed->ghost_ptr(packed_block);" % indent,
        "%s        auto ghost_idx = packed->ghost_idx(packed_block);" % indent,
        "%s        auto ghost_reduce_ptr = packed->ghost_reduce_ptr(packed_block);" % indent,
        "%s        auto ghost_reduce_idx = packed->ghost_reduce_idx(packed_block);" % indent,
        "%s        auto ghost_reduce_dest = packed->ghost_reduce_dest(packed_block);" % indent,
        "%s        const int dim = mesh->spatial_dimension();" % indent,
    ]
    emitted = False
    for dim in (2, 3):
        dependencies = gradient_dependencies_by_dim.get(dim)
        if dependencies is None:
            continue
        function = "%s_gradient_packed_%dd_isoparametric_mesh_soa" % (material_name, dim)
        if not _c_abi_function_exists(kernel_sources, function, public_only=True):
            continue
        emitted = True
        prefix = "if" if emitted and not any("if (dim ==" in line for line in lines) else "else if"
        parameter_args = list(_dependency_domain_parameter_args(dependencies))
        current_args = (
            [str(dim)] + ["x + %d" % d for d in range(dim)]
            if getattr(dependencies, "current", True)
            else []
        )
        output_args = [str(dim)] + ["out + %d" % d for d in range(dim)]
        lines.append("%s        %s (dim == %d) {" % (indent, prefix, dim))
        lines.extend(
            _hyperelastic_packed_return(
                indent + "            ",
                function,
                ["domain.element_type"],
                [
                    "points",
                    *parameter_args,
                    *current_args,
                    *output_args,
                ],
                kernel_sources,
            )
        )
        lines.append("%s        }" % indent)
    lines.extend(
        [
            "%s    }" % indent,
            "%s}" % indent,
        ]
    )
    packed_blocks = []
    if emitted_affine:
        packed_blocks.append("\n".join(affine_lines))
    if emitted:
        packed_blocks.append("\n".join(lines))
    return "\n".join(packed_blocks)


def _hyperelastic_objective_steps_packed_dispatch_body(material_name, kernel_sources, apply_dependencies_by_dim, indent):
    affine_lines = [
        "%sif (impl_->objective_uses_affine && impl_->space->has_packed_mesh()) {" % indent,
        "%s    auto packed = impl_->space->packed_mesh();" % indent,
        "%s    const int packed_block = packed_block_id_for_domain(*packed, *domain.block);" % indent,
        "%s    if (packed_block >= 0) {" % indent,
        "%s        auto packed_elements = packed->elements(packed_block);" % indent,
        "%s        auto owned_nodes_ptr = packed->owned_nodes_ptr(packed_block);" % indent,
        "%s        auto n_shared_nodes = packed->n_shared(packed_block);" % indent,
        "%s        auto ghost_ptr = packed->ghost_ptr(packed_block);" % indent,
        "%s        auto ghost_idx = packed->ghost_idx(packed_block);" % indent,
        "%s        const int dim = mesh->spatial_dimension();" % indent,
    ]
    emitted_affine = False
    for dim in (2, 3):
        dependencies = apply_dependencies_by_dim.get(dim)
        if dependencies is None:
            continue
        function = "%s_objective_steps_packed_%dd_affine_mesh_soa" % (material_name, dim)
        if not _c_abi_function_exists(kernel_sources, function, public_only=True):
            continue
        prefix = "if" if not emitted_affine else "else if"
        emitted_affine = True
        parameter_args = list(_dependency_domain_parameter_args(dependencies))
        current_args = (
            [str(dim)] + ["x + %d" % d for d in range(dim)]
            if getattr(dependencies, "current", True)
            else []
        )
        direction_args = [str(dim)] + ["h + %d" % d for d in range(dim)]
        affine_lines.extend(
            [
                "%s        %s (dim == %d) {" % (indent, prefix, dim),
                "%s            status = %s(%s);" % (
                    indent,
                    function,
                    ", ".join(
                        [
                            "domain.element_type",
                            "packed->n_packs(packed_block)",
                            "packed->n_elements_per_pack(packed_block)",
                            "domain.block->n_elements()",
                            "mesh->n_nodes()",
                            "packed->max_nodes_per_pack()",
                            "packed_elements->data()",
                            "owned_nodes_ptr->data()",
                            "n_shared_nodes->data()",
                            "ghost_ptr->data()",
                            "ghost_idx->data()",
                            *("adjugate[%d]" % i for i in range(dim * dim)),
                            "determinant",
                            *parameter_args,
                            *current_args,
                            *direction_args,
                            "nsteps",
                            "steps",
                            "impl_->element_values.get()",
                        ]
                    ),
                ),
                "%s        }" % indent,
            ]
        )
    affine_lines.extend(
        [
            "%s    }" % indent,
            "%s}" % indent,
        ]
    )
    lines = [
        "%sif (!impl_->objective_uses_affine && impl_->space->has_packed_mesh()) {" % indent,
        "%s    auto packed = impl_->space->packed_mesh();" % indent,
        "%s    const int packed_block = packed_block_id_for_domain(*packed, *domain.block);" % indent,
        "%s    if (packed_block >= 0) {" % indent,
        "%s        auto packed_elements = packed->elements(packed_block);" % indent,
        "%s        auto owned_nodes_ptr = packed->owned_nodes_ptr(packed_block);" % indent,
        "%s        auto n_shared_nodes = packed->n_shared(packed_block);" % indent,
        "%s        auto ghost_ptr = packed->ghost_ptr(packed_block);" % indent,
        "%s        auto ghost_idx = packed->ghost_idx(packed_block);" % indent,
        "%s        const int dim = mesh->spatial_dimension();" % indent,
    ]
    emitted = False
    for dim in (2, 3):
        dependencies = apply_dependencies_by_dim.get(dim)
        if dependencies is None:
            continue
        function = "%s_objective_steps_packed_%dd_isoparametric_mesh_soa" % (material_name, dim)
        if not _c_abi_function_exists(kernel_sources, function, public_only=True):
            continue
        emitted = True
        prefix = "if" if emitted and not any("if (dim ==" in line for line in lines) else "else if"
        parameter_args = list(_dependency_domain_parameter_args(dependencies))
        current_args = (
            [str(dim)] + ["x + %d" % d for d in range(dim)]
            if getattr(dependencies, "current", True)
            else []
        )
        direction_args = (
            [str(dim)] + ["h + %d" % d for d in range(dim)]
        )
        lines.extend(
            [
                "%s        %s (dim == %d) {" % (indent, prefix, dim),
                "%s            status = %s(%s);" % (
                    indent,
                    function,
                    ", ".join(
                        [
                            "domain.element_type",
                            "packed->n_packs(packed_block)",
                            "packed->n_elements_per_pack(packed_block)",
                            "domain.block->n_elements()",
                            "mesh->n_nodes()",
                            "packed->max_nodes_per_pack()",
                            "packed_elements->data()",
                            "owned_nodes_ptr->data()",
                            "n_shared_nodes->data()",
                            "ghost_ptr->data()",
                            "ghost_idx->data()",
                            "points",
                            *parameter_args,
                            *current_args,
                            *direction_args,
                            "nsteps",
                            "steps",
                            "impl_->element_values.get()",
                        ]
                    ),
                ),
                "%s        }" % indent,
            ]
        )
    lines.extend(
        [
            "%s    }" % indent,
            "%s}" % indent,
        ]
    )
    packed_blocks = []
    if emitted_affine:
        packed_blocks.append("\n".join(affine_lines))
    if emitted:
        packed_blocks.append("\n".join(lines))
    return "\n".join(packed_blocks)


def _hyperelastic_hessian_dispatch_body(material_name, operation, kernel_sources, apply_dependencies_by_dim, tail_args, indent):
    lines = ["%sconst int dim = mesh->spatial_dimension();" % indent]
    for dim in (2, 3):
        prefix = "if" if dim == 2 else "else if"
        function = "%s_%s_%dd_isoparametric_mesh_soa" % (
            material_name,
            operation,
            dim,
        )
        dependencies = apply_dependencies_by_dim.get(dim)
        parameter_args = list(_dependency_domain_parameter_args(dependencies))
        current_args = (
            [str(dim)] + ["current + %d" % d for d in range(dim)]
            if getattr(dependencies, "current", True)
            else []
        )
        lines.append("%s%s (dim == %d) {" % (indent, prefix, dim))
        if _c_abi_function_exists(kernel_sources, function, public_only=True):
            lines.append(
                "%s    return %s(%s);"
                % (
                    indent,
                    function,
                    ", ".join(
                        [
                            "domain.element_type",
                            "domain.block->n_elements()",
                            "mesh->n_nodes()",
                            "domain.block->elements()->data()",
                            "points",
                            *parameter_args,
                            *current_args,
                            *tail_args,
                        ]
                    ),
                )
            )
        else:
            lines.append('%s    SFEM_ERROR("%s %s %dd dispatch was not generated\\n");' % (indent, material_name, operation, dim))
            lines.append("%s    return SFEM_FAILURE;" % indent)
        lines.append("%s}" % indent)
    lines.extend(
        [
            '%sSFEM_ERROR("%s %s does not support spatial dimension %%d\\n", dim);' % (indent, material_name, operation),
            "%sreturn SFEM_FAILURE;" % indent,
        ]
    )
    return "\n".join(lines)


def _dual_case(element, flag, affine_function, affine_arguments, isoparametric_function, isoparametric_arguments):
    return """                case smesh::%(element)s:
                    return impl_->%(flag)s ? %(affine_function)s(%(affine_arguments)s) : %(isoparametric_function)s(%(isoparametric_arguments)s);""" % {
        "element": _mesh_element_name(element),
        "flag": flag,
        "affine_function": affine_function,
        "affine_arguments": affine_arguments,
        "isoparametric_function": isoparametric_function,
        "isoparametric_arguments": isoparametric_arguments,
    }


def _dual_aos_unit_case(
    element,
    flag,
    affine_aos_function,
    affine_aos_arguments,
    affine_function,
    affine_arguments,
    isoparametric_function,
    isoparametric_arguments,
):
    return """                case smesh::%(element)s:
                    if (impl_->%(flag)s) {
                        return adjugate_aos ? %(affine_aos_function)s(%(affine_aos_arguments)s) : %(affine_function)s(%(affine_arguments)s);
                    }
                    return %(isoparametric_function)s(%(isoparametric_arguments)s);""" % {
        "element": _mesh_element_name(element),
        "flag": flag,
        "affine_aos_function": affine_aos_function,
        "affine_aos_arguments": affine_aos_arguments,
        "affine_function": affine_function,
        "affine_arguments": affine_arguments,
        "isoparametric_function": isoparametric_function,
        "isoparametric_arguments": isoparametric_arguments,
    }


def _dual_status_case(element, affine_function, affine_arguments, isoparametric_function, isoparametric_arguments):
    return """                case smesh::%(element)s:
                    status = impl_->objective_uses_affine ? %(affine_function)s(%(affine_arguments)s) : %(isoparametric_function)s(%(isoparametric_arguments)s);
                    break;""" % {
        "element": _mesh_element_name(element),
        "affine_function": affine_function,
        "affine_arguments": affine_arguments,
        "isoparametric_function": isoparametric_function,
        "isoparametric_arguments": isoparametric_arguments,
    }


def _seed_lines(defaults):
    return "\n".join(
        '            parameters.set_value("%s", %.17g);' % (name, value)
        for name, value in defaults
    )


def _yaml_helpers(defaults):
    nparameters = len(defaults)
    storage_size = max(1, nparameters)
    names = ", ".join('"%s"' % name for name, _ in defaults) or "nullptr"
    default_lines = []
    for i, (_, value) in enumerate(defaults):
        default_lines.append("            values[%d] = %.17g;" % (i, value))
    if not default_lines:
        default_lines.append("            values[0] = 0;")
    return """        struct AffineOption {
            const char *name;
            bool       *flag;
        };

        inline bool set_affine_option(const std::string &name,
                                      const bool val,
                                      const AffineOption *const options,
                                      const int n_options) {
            if (name == "ASSUME_AFFINE" || name == "assume_affine") {
                for (int i = 0; i < n_options; ++i) {
                    *options[i].flag = val;
                }
                return true;
            }
            bool matched = false;
            for (int i = 0; i < n_options; ++i) {
                if (name == options[i].name) {
                    *options[i].flag = val;
                    matched = true;
                }
            }
            return matched;
        }

        void material_defaults(real_t *const values) {
%(default_lines)s
        }

#ifdef SFEM_ENABLE_RYAML
        constexpr int N_DEFINED_MATERIAL_PARAMETERS = %(nparameters)d;
        constexpr int N_MATERIAL_PARAMETERS = %(storage_size)d;
        static const char *const MATERIAL_PARAMETER_NAMES[N_MATERIAL_PARAMETERS] = {%(names)s};

        bool yaml_read_real(const ryml::ConstNodeRef &node,
                            const char *const key,
                            real_t &value) {
            if (!node.has_child(key)) {
                return false;
            }
            node[key] >> value;
            return true;
        }

        bool yaml_read_parameter(const ryml::ConstNodeRef &node,
                                 const char *const key,
                                 real_t &value) {
            if (yaml_read_real(node, key, value)) {
                return true;
            }
            if (node.has_child("parameters") &&
                yaml_read_real(node["parameters"], key, value)) {
                return true;
            }
            if (node.has_child("material") &&
                yaml_read_real(node["material"], key, value)) {
                return true;
            }
            return false;
        }

        std::string yaml_read_string(const ryml::ConstNodeRef &node) {
            const auto value = node.val();
            return std::string(value.str, value.len);
        }

        void copy_material_parameters(const real_t *const src,
                                      real_t *const dst) {
            for (int i = 0; i < N_MATERIAL_PARAMETERS; ++i) {
                dst[i] = src[i];
            }
        }

        bool material_from_yaml(const ryml::ConstNodeRef &node,
                                const real_t *const base,
                                real_t *const values) {
            copy_material_parameters(base, values);
            bool changed = false;
            for (int i = 0; i < N_DEFINED_MATERIAL_PARAMETERS; ++i) {
                changed |= yaml_read_parameter(node,
                                               MATERIAL_PARAMETER_NAMES[i],
                                               values[i]);
            }
            return changed;
        }

        void set_material(MultiDomainOp &domains,
                          const real_t *const values) {
            for (auto &entry : domains.domains()) {
                for (int i = 0; i < N_DEFINED_MATERIAL_PARAMETERS; ++i) {
                    entry.second.parameters->set_value(MATERIAL_PARAMETER_NAMES[i],
                                                       values[i]);
                }
            }
        }

        void set_material_in_block(MultiDomainOp &domains,
                                   const std::string &block_name,
                                   const real_t *const values) {
            for (int i = 0; i < N_DEFINED_MATERIAL_PARAMETERS; ++i) {
                domains.set_value_in_block(block_name,
                                           MATERIAL_PARAMETER_NAMES[i],
                                           values[i]);
            }
        }

        bool yaml_read_bool(const ryml::ConstNodeRef &node,
                            const char *const key,
                            bool &value) {
            if (!node.has_child(key)) {
                return false;
            }
            int raw = value ? 1 : 0;
            node[key] >> raw;
            value = raw != 0;
            return true;
        }

        inline void read_affine_options(const ryml::ConstNodeRef &node,
                                        const AffineOption *const options,
                                        const int n_options) {
            bool all = true;
            for (int i = 0; i < n_options; ++i) {
                all = all && *options[i].flag;
            }
            if (yaml_read_bool(node, "ASSUME_AFFINE", all) ||
                yaml_read_bool(node, "assume_affine", all)) {
                for (int i = 0; i < n_options; ++i) {
                    *options[i].flag = all;
                }
            }
            for (int i = 0; i < n_options; ++i) {
                yaml_read_bool(node, options[i].name, *options[i].flag);
            }
        }
#endif  // SFEM_ENABLE_RYAML""" % {
        "nparameters": nparameters,
        "storage_size": storage_size,
        "names": names,
        "default_lines": "\n".join(default_lines),
    }


def _parameter_args(parameters):
    return "".join(
        ', domain.parameters->require_real_value("%s")' % name
        for name in parameters
    )


def _components(dim):
    return ("x", "y", "z")[:dim]


def _offsets(name, components):
    return ", ".join("%s + %d" % (name, i) for i, _ in enumerate(components))


def _cpp_bool(value):
    return "true" if value else "false"


def _affine_geometry_offsets(dim):
    return ", ".join("adjugate[%d]" % i for i in range(dim * dim))


def _affine_metric_offsets(dim):
    return ", ".join("geom_metric[%d]" % i for i in range(dim * (dim + 1) // 2))


def _c_abi_function_uses_cached_metric(kernel_sources, function_name):
    if not kernel_sources:
        return False
    for declaration in _extract_c_abi_declarations(kernel_sources):
        if _c_abi_function_name(declaration) == function_name:
            return "g_geom_metric0" in declaration
    return False


def _boundary_surface_name(element):
    name = _element_name(element)
    surface_by_cell = {
        "TRI3": "edgeshell2",
        "QUAD4": "edgeshell2",
        "PROTEUS_QUAD4": "edgeshell2",
        "TET4": "trishell3",
        "TET10": "trishell6",
        "HEX8": "quadshell4",
        "HEX27": "quadshell9",
        "PROTEUS_HEX8": "proteus_quadshell4",
        "PROTEUS_HEX27": "proteus_quadshell9",
        "PROTEUS_HEX64": "proteus_quadshell16",
        "PROTEUS_HEX125": "proteus_quadshell25",
        "PROTEUS_HEX216": "proteus_quadshell36",
        "PROTEUS_HEX343": "proteus_quadshell49",
        "PROTEUS_HEX512": "proteus_quadshell64",
        "PROTEUS_HEX729": "proteus_quadshell81",
    }
    try:
        return surface_by_cell[name]
    except KeyError as exc:
        raise ValueError("unsupported generated boundary Op element %s" % element) from exc


def _element_dim(element):
    name = _element_name(element)
    if name in ("TRI3", "TRI6", "QUAD4", "PROTEUS_QUAD4") or name.startswith(("TRI6_", "QUAD4_", "PROTEUS_QUAD4_")):
        return 2
    if (
        name in ("TET4", "TET10", "HEX8", "HEX27")
        or name.startswith(("TET10_", "HEX27_", "PROTEUS_HEX"))
    ):
        return 3
    raise ValueError("unsupported generated Op element %s" % element)


def _mesh_element_name(element):
    return getattr(element, "cell_element_type", _element_name(element))


def _element_name(element):
    return getattr(element, "name", str(element).upper())
