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
        header, source = _hyperelastic_op(material, elements, c_abi_header, form_collections)
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
                material, elements, c_abi_header, form_collections
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
    if c_abi_header:
        c_abi_path = "op/%s" % c_abi_header
        files[c_abi_path] = _c_abi_header(material, kernel_sources)
        files["op/sfem_%s_manifest.json" % material.op_name] = _op_manifest(
            material,
            kernel_sources,
            wrapper_header,
            wrapper_source,
            registration_source,
            c_abi_path,
        )
    return files


def _registration_entries_from_manifests(manifests):
    entries = []
    for manifest in manifests:
        if isinstance(manifest, str):
            manifest = json.loads(manifest)
        registration = manifest["registration"]
        entries.append(
            (
                registration["operator_name"],
                registration["function"].replace("sfem::", ""),
            )
        )
    return tuple(sorted(dict(entries).items()))


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
    return """#pragma once

#include "sfem_Op.hpp"

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

        int initialize(const std::vector<std::string> &block_names = {}) override;%(extra)s
        int gradient(const real_t *const x, real_t *const out) override;
        int apply(const real_t *const x,
                  const real_t *const h,
                  real_t *const out) override;
        int value(const real_t *x, real_t *const out) override;%(value_steps)s
        int hessian_crs(const real_t *const x,
                        const count_t *const rowptr,
                        const idx_t *const colidx,
                        real_t *const values) override;
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
""" % {"op": material.op_name, "extra": extra, "value_steps": value_steps}


def _hyperelastic_op(material, elements, c_abi_header=None, form_collections=None):
    if form_collections is None:
        raise ValueError("energy generated Op requires form collections")
    parameters = tuple(str(name) for name, _ in material.parameter_defaults)
    defaults = _seed_lines(material.parameter_defaults)
    declarations = []
    gradient_cases = []
    apply_cases = []
    objective_cases = []
    objective_steps_cases = []
    dependencies_by_dim = {}
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
        components = _components(dim)
        declarations.extend(
            _hyperelastic_declarations(stem, dim, parameters, dependencies)
        )
        args = _parameter_args(parameters)
        common_isoparametric_args = (
            "domain.block->n_elements(), mesh->n_nodes(), "
            "domain.block->elements()->data(), points%s" % args
        )
        common_affine_args = (
            "domain.block->n_elements(), mesh->n_nodes(), "
            "domain.block->elements()->data(), %s, determinant%s"
            % (_affine_geometry_offsets(dim), args)
        )
        gradient_cases.append(
            _dual_case(
                element,
                "gradient_uses_affine",
                "%s_gradient_affine_mesh_soa" % stem,
                ", ".join(_nonempty(
                    common_affine_args,
                    *_energy_field_args(gradient_dependencies, dim, components, current="x"),
                    *_energy_output_args(dim, components),
                )),
                "%s_gradient_isoparametric_mesh_soa" % stem,
                ", ".join(_nonempty(
                    common_isoparametric_args,
                    *_energy_field_args(gradient_dependencies, dim, components, current="x"),
                    *_energy_output_args(dim, components),
                )),
            )
        )
        apply_cases.append(
            _dual_case(
                element,
                "apply_uses_affine",
                "%s_apply_affine_mesh_soa" % stem,
                ", ".join(_nonempty(
                    common_affine_args,
                    *_energy_field_args(apply_dependencies, dim, components, current="x", direction="h"),
                    *_energy_output_args(dim, components),
                )),
                "%s_apply_isoparametric_mesh_soa" % stem,
                ", ".join(_nonempty(
                    common_isoparametric_args,
                    *_energy_field_args(apply_dependencies, dim, components, current="x", direction="h"),
                    *_energy_output_args(dim, components),
                )),
            )
        )
        objective_cases.append(
            _dual_status_case(
                element,
                "%s_objective_affine_mesh_soa" % stem,
                ", ".join(_nonempty(
                    "nelements, mesh->n_nodes(), domain.block->elements()->data(), %s, determinant%s"
                    % (_affine_geometry_offsets(dim), args),
                    *_energy_field_args(objective_dependencies, dim, components, current="x"),
                    "impl_->element_values.get()",
                )),
                "%s_objective_isoparametric_mesh_soa" % stem,
                ", ".join(_nonempty(
                    "nelements, mesh->n_nodes(), domain.block->elements()->data(), points%s"
                    % args,
                    *_energy_field_args(objective_dependencies, dim, components, current="x"),
                    "impl_->element_values.get()",
                )),
            )
        )
        objective_steps_cases.append(
            _dual_status_case(
                element,
                "%s_objective_steps_affine_mesh_soa" % stem,
                "%s, %d, %s, %d, %s, nsteps, steps, impl_->element_values.get()"
                % (
                    "nelements, mesh->n_nodes(), domain.block->elements()->data(), %s, determinant%s"
                    % (_affine_geometry_offsets(dim), args),
                    dim,
                    _offsets("x", components),
                    dim,
                    _offsets("h", components),
                ),
                "%s_objective_steps_isoparametric_mesh_soa" % stem,
                "%s, %d, %s, %d, %s, nsteps, steps, impl_->element_values.get()"
                % (
                    "nelements, mesh->n_nodes(), domain.block->elements()->data(), points%s"
                    % args,
                    dim,
                    _offsets("x", components),
                    dim,
                    _offsets("h", components),
                ),
            )
        )

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

    int %(op)s::gradient(const real_t *const x, real_t *const out) {
        SFEM_TRACE_SCOPE("%(op)s::gradient");
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const real_t *const *adjugate = nullptr;
            const real_t *determinant = nullptr;
            if (impl_->gradient_uses_affine) {
                auto jacobian = std::static_pointer_cast<smesh::JacobianAdjugateAndDeterminant>(
                        domain.user_data);
                if (!jacobian) {
                    SFEM_ERROR("%(op)s affine gradient requires cached geometry\\n");
                    return SFEM_FAILURE;
                }
                adjugate = reinterpret_cast<const real_t *const *>(
                        jacobian->jacobian_adjugate_SoA()->data());
                determinant = reinterpret_cast<const real_t *>(
                        jacobian->jacobian_determinant()->data());
            }
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
            const real_t *const *adjugate = nullptr;
            const real_t *determinant = nullptr;
            if (impl_->apply_uses_affine) {
                auto jacobian = std::static_pointer_cast<smesh::JacobianAdjugateAndDeterminant>(
                        domain.user_data);
                if (!jacobian) {
                    SFEM_ERROR("%(op)s affine hessian action requires cached geometry\\n");
                    return SFEM_FAILURE;
                }
                adjugate = reinterpret_cast<const real_t *const *>(
                        jacobian->jacobian_adjugate_SoA()->data());
                determinant = reinterpret_cast<const real_t *>(
                        jacobian->jacobian_determinant()->data());
            }
            switch (domain.element_type) {
%(apply_cases)s
                default:
                    SFEM_ERROR("%(op)s does not support element type %%d\\n",
                               domain.element_type);
                    return SFEM_FAILURE;
            }
        });
    }

    int %(op)s::value(const real_t *x, real_t *const out) {
        SFEM_TRACE_SCOPE("%(op)s::value");
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        *out = 0;
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const ptrdiff_t nelements = domain.block->n_elements();
            const real_t *const *adjugate = nullptr;
            const real_t *determinant = nullptr;
            if (impl_->objective_uses_affine) {
                auto jacobian = std::static_pointer_cast<smesh::JacobianAdjugateAndDeterminant>(
                        domain.user_data);
                if (!jacobian) {
                    SFEM_ERROR("%(op)s affine objective requires cached geometry\\n");
                    return SFEM_FAILURE;
                }
                adjugate = reinterpret_cast<const real_t *const *>(
                        jacobian->jacobian_adjugate_SoA()->data());
                determinant = reinterpret_cast<const real_t *>(
                        jacobian->jacobian_determinant()->data());
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
            const real_t *const *adjugate = nullptr;
            const real_t *determinant = nullptr;
            if (impl_->objective_uses_affine) {
                auto jacobian = std::static_pointer_cast<smesh::JacobianAdjugateAndDeterminant>(
                        domain.user_data);
                if (!jacobian) {
                    SFEM_ERROR("%(op)s affine objective_steps requires cached geometry\\n");
                    return SFEM_FAILURE;
                }
                adjugate = reinterpret_cast<const real_t *const *>(
                        jacobian->jacobian_adjugate_SoA()->data());
                determinant = reinterpret_cast<const real_t *>(
                        jacobian->jacobian_determinant()->data());
            }
            if (nvalues > impl_->element_capacity) {
                impl_->element_values.reset(new real_t[nvalues]);
                impl_->element_capacity = nvalues;
            }
            std::fill(impl_->element_values.get(),
                      impl_->element_values.get() + nvalues,
                      real_t(0));
            int status = SFEM_FAILURE;
            switch (domain.element_type) {
%(objective_steps_cases)s
                default:
                    SFEM_ERROR("%(op)s does not support element type %%d\\n",
                               domain.element_type);
                    return SFEM_FAILURE;
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

    int %(op)s::hessian_crs(const real_t *const,
                            const count_t *const,
                            const idx_t *const,
                            real_t *const) {
        SFEM_TRACE_SCOPE("%(op)s::hessian_crs");
        return SFEM_FAILURE;
    }

    void %(op)s::set_option(const std::string &name, const bool val) {
        SFEM_TRACE_SCOPE("%(op)s::set_option");
        AffineOption options[] = {
%(affine_options)s
        };
        set_affine_option(name, val, options, sizeof(options) / sizeof(options[0]));
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

        real_t defaults[N_MATERIAL_PARAMETERS];
        material_defaults(defaults);
        real_t top_values[N_MATERIAL_PARAMETERS];
        copy_material_parameters(defaults, top_values);
        if (material_from_yaml(node, defaults, top_values)) {
            set_material(*ret->impl_->domains, top_values);
        }

        AffineOption options[] = {
%(affine_options)s
        };
        read_affine_options(node, options, sizeof(options) / sizeof(options[0]));

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
        "declaration_block": (
            ""
            if c_abi_header
            else 'extern "C" {\n%s\n}' % "\n".join(declarations)
        ),
        "declarations": "\n".join(declarations),
        "defaults": defaults,
        "yaml_helpers": _yaml_helpers(material.parameter_defaults),
        "gradient_cases": "\n".join(gradient_cases),
        "apply_cases": "\n".join(apply_cases),
        "objective_cases": "\n".join(objective_cases),
        "objective_steps_cases": "\n".join(objective_steps_cases),
        "affine_options": _affine_option_entries(
            "objective_uses_affine",
            "gradient_uses_affine",
            "apply_uses_affine",
        ),
    }
    return _header(material, False), source


def _residual_op(material, elements, c_abi_header=None, form_collections=None):
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
    dependencies_by_dim = {}
    parameter_names_by_dim = {}
    fields_by_dim = {}
    block_size_by_dim = {}
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
        stem = "%s_%s" % (material.name, _element_name(element).lower())
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
        common_affine = (
            "domain.block->n_elements(), mesh->n_nodes(), "
            "domain.block->elements()->data(), %s, determinant"
            % _affine_geometry_offsets(dim)
        )
        residual_common_args = []
        residual_common_args.extend(
            "storage[%d]" % index
            for index, _ in enumerate(parameter_names_by_dim[dim])
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
        residual_cases.append(
            _residual_dual_soa_case(
                element,
                "residual_uses_affine",
                "%s_residual_affine_mesh_soa" % stem,
                ", ".join((common_affine, *residual_common_args)),
                "%s_residual_isoparametric_mesh_soa" % stem,
                ", ".join((common_isoparametric, *residual_common_args)),
                block_size_by_dim[dim],
                residual_setup,
            )
        )
        action_common_args = []
        action_common_args.extend(
            "storage[%d]" % index
            for index, _ in enumerate(parameter_names_by_dim[dim])
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
        action_cases.append(
            _residual_dual_soa_case(
                element,
                "jacobian_action_uses_affine",
                "%s_jacobian_action_affine_mesh_soa" % stem,
                ", ".join((common_affine, *action_common_args)),
                "%s_jacobian_action_isoparametric_mesh_soa" % stem,
                ", ".join((common_isoparametric, *action_common_args)),
                block_size_by_dim[dim],
                action_setup,
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

    max_parameters = max(len(names) for names in parameter_names_by_dim.values())
    parameter_lines = _residual_parameter_array_lines(parameter_names_by_dim)
    source = """#include "sfem_%(op)s.hpp"
%(c_abi_include)s

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

        void parameter_array(const Parameters &parameters,
                             const int dim,
                             real_t *const values) {
            int index = 0;
%(parameter_lines)s
        }

        ptrdiff_t block_size_for_dim(const int dim) {
%(block_size_lines)s
        }
    }  // namespace

    class %(op)s::Impl {
    public:
        explicit Impl(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        std::shared_ptr<FunctionSpace> space;
        std::shared_ptr<MultiDomainOp> domains;
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

    int %(op)s::initialize(const std::vector<std::string> &block_names) {
        SFEM_TRACE_SCOPE("%(op)s::initialize");
        impl_->domains = std::make_shared<MultiDomainOp>(impl_->space, block_names);
        seed_material(*impl_->domains);
        auto mesh = impl_->space->mesh_ptr();
        const bool needs_affine_geometry =
                impl_->residual_uses_affine ||
                impl_->jacobian_action_uses_affine;
        if (needs_affine_geometry) {
            for (auto &entry : impl_->domains->domains()) {
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
            const real_t *const *adjugate = nullptr;
            const real_t *determinant = nullptr;
            if (impl_->residual_uses_affine) {
                auto jacobian = std::static_pointer_cast<smesh::JacobianAdjugateAndDeterminant>(
                        domain.user_data);
                if (!jacobian) {
                    SFEM_ERROR("%(op)s affine residual requires cached geometry\\n");
                    return SFEM_FAILURE;
                }
                adjugate = reinterpret_cast<const real_t *const *>(
                        jacobian->jacobian_adjugate_SoA()->data());
                determinant = reinterpret_cast<const real_t *>(
                        jacobian->jacobian_determinant()->data());
            }
            real_t storage[MAX_PARAMETERS];
            parameter_array(*domain.parameters,
                            mesh->spatial_dimension(),
                            storage);
%(gradient_previous_alias)s
            switch (domain.element_type) {
%(residual_cases)s
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
            const real_t *const *adjugate = nullptr;
            const real_t *determinant = nullptr;
            if (impl_->jacobian_action_uses_affine) {
                auto jacobian = std::static_pointer_cast<smesh::JacobianAdjugateAndDeterminant>(
                        domain.user_data);
                if (!jacobian) {
                    SFEM_ERROR("%(op)s affine jacobian action requires cached geometry\\n");
                    return SFEM_FAILURE;
                }
                adjugate = reinterpret_cast<const real_t *const *>(
                        jacobian->jacobian_adjugate_SoA()->data());
                determinant = reinterpret_cast<const real_t *>(
                        jacobian->jacobian_determinant()->data());
            }
            real_t storage[MAX_PARAMETERS];
            parameter_array(*domain.parameters,
                            mesh->spatial_dimension(),
                            storage);
%(apply_previous_alias)s
            switch (domain.element_type) {
%(action_cases)s
                default:
                    SFEM_ERROR("%(op)s does not support element type %%d\\n",
                               domain.element_type);
                    return SFEM_FAILURE;
            }
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
        set_affine_option(name, val, options, sizeof(options) / sizeof(options[0]));
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

        real_t defaults[N_MATERIAL_PARAMETERS];
        material_defaults(defaults);
        real_t top_values[N_MATERIAL_PARAMETERS];
        copy_material_parameters(defaults, top_values);
        if (material_from_yaml(node, defaults, top_values)) {
            set_material(*ret->impl_->domains, top_values);
        }

        AffineOption options[] = {
%(affine_options)s
        };
        read_affine_options(node, options, sizeof(options) / sizeof(options[0]));

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

    int %(op)s::value(const real_t *, real_t *const) {
        SFEM_TRACE_SCOPE("%(op)s::value");
        return SFEM_FAILURE;
    }
}  // namespace sfem
""" % {
        "op": material.op_name,
        "c_abi_include": '#include "%s"' % c_abi_header if c_abi_header else "",
        "declaration_block": (
            ""
            if c_abi_header
            else 'extern "C" {\n%s\n}' % "\n".join(declarations)
        ),
        "declarations": "\n".join(declarations),
        "max_parameters": max_parameters,
        "defaults": defaults,
        "yaml_helpers": _yaml_helpers(material.parameter_defaults),
        "parameter_lines": parameter_lines,
        "block_size_lines": _residual_block_size_lines(block_size_by_dim),
        "residual_cases": "\n".join(residual_cases),
        "action_cases": "\n".join(action_cases),
        "affine_options": _affine_option_entries(
            "residual_uses_affine",
            "jacobian_action_uses_affine",
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
            "condition.parameters[%d]" % material_parameter_index[name]
            for name in parameter_names_by_dim[dim]
        )
        output_args = _boundary_soa_component_argument_names(fields, "out")
        call_args = _nonempty(
            "condition.sideset->size()",
            "mesh->n_nodes()",
            "domain.block->elements()->data()",
            "condition.sideset->parent()->data()",
            "condition.sideset->lfi()->data()",
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
#include "sfem_OpTracer.hpp"
#include "sfem_Parameters.hpp"
#include "smesh_mesh.hpp"
#include "smesh_sideset.hpp"

#include <array>
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
        struct BoundaryCondition {
            std::shared_ptr<smesh::Sideset> sideset;
            std::array<real_t, MAX_PARAMETERS> parameters;
        };

        explicit Impl(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        std::shared_ptr<FunctionSpace> space;
        std::shared_ptr<MultiDomainOp> domains;
        std::vector<BoundaryCondition> conditions;
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
        Impl::BoundaryCondition condition;
        condition.sideset = sideset;
        for (int i = 0; i < MAX_PARAMETERS; ++i) {
            condition.parameters[i] = parameters[i];
        }
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
                if (!condition.sideset || condition.sideset->block_id() != block_id) {
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
        "gradient_cases": "\n".join(gradient_cases),
    }
    return _boundary_header(material), source


def _boundary_header(material):
    return """#pragma once

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

        int initialize(const std::vector<std::string> &block_names = {}) override;
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
            const real_t *const *adjugate = nullptr;
            const real_t *determinant = nullptr;
            if (impl_->gradient_uses_affine || impl_->residual_uses_affine) {
                auto jacobian = std::static_pointer_cast<smesh::JacobianAdjugateAndDeterminant>(
                        domain.user_data);
                if (!jacobian) {
                    SFEM_ERROR("%(op)s affine gradient/residual requires cached geometry\\n");
                    return SFEM_FAILURE;
                }
                adjugate = reinterpret_cast<const real_t *const *>(
                        jacobian->jacobian_adjugate_SoA()->data());
                determinant = reinterpret_cast<const real_t *>(
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
            const real_t *const *adjugate = nullptr;
            const real_t *determinant = nullptr;
            if (impl_->apply_uses_affine || impl_->jacobian_action_uses_affine) {
                auto jacobian = std::static_pointer_cast<smesh::JacobianAdjugateAndDeterminant>(
                        domain.user_data);
                if (!jacobian) {
                    SFEM_ERROR("%(op)s affine hessian/jacobian action requires cached geometry\\n");
                    return SFEM_FAILURE;
                }
                adjugate = reinterpret_cast<const real_t *const *>(
                        jacobian->jacobian_adjugate_SoA()->data());
                determinant = reinterpret_cast<const real_t *>(
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
            const real_t *const *adjugate = nullptr;
            const real_t *determinant = nullptr;
            if (impl_->objective_uses_affine) {
                auto jacobian = std::static_pointer_cast<smesh::JacobianAdjugateAndDeterminant>(
                        domain.user_data);
                if (!jacobian) {
                    SFEM_ERROR("%(op)s affine objective requires cached geometry\\n");
                    return SFEM_FAILURE;
                }
                adjugate = reinterpret_cast<const real_t *const *>(
                        jacobian->jacobian_adjugate_SoA()->data());
                determinant = reinterpret_cast<const real_t *>(
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
        set_affine_option(name, val, options, sizeof(options) / sizeof(options[0]));
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

        real_t defaults[N_MATERIAL_PARAMETERS];
        material_defaults(defaults);
        real_t top_values[N_MATERIAL_PARAMETERS];
        copy_material_parameters(defaults, top_values);
        if (material_from_yaml(node, defaults, top_values)) {
            set_material(*ret->impl_->domains, top_values);
        }

        AffineOption options[] = {
%(affine_options)s
        };
        read_affine_options(node, options, sizeof(options) / sizeof(options[0]));

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
    }
    return _header(material, True), source


def _coupled_dependency_flags(systems_by_dim, energy_name, residual_name):
    from codegen.framework.forms import FormOrder

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
    from codegen.framework.forms import FormOrder

    cases = {"gradient": [], "apply": [], "objective": []}
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


def _dependency_parameters(dependencies):
    return tuple(getattr(dependencies, "parameters", dependencies or ()))


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
    from codegen.framework.forms import FormOrder

    return FormOrder.ZERO


def _form_order_one():
    from codegen.framework.forms import FormOrder

    return FormOrder.ONE


def _form_order_two():
    from codegen.framework.forms import FormOrder

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
        setup_lines):
    return """                case smesh::%(element)s: {
                    static constexpr ptrdiff_t FIELD_STRIDE = %(field_stride)d;
%(setup)s
                    return impl_->%(flag)s ? %(affine_function)s(%(affine_arguments)s) : %(isoparametric_function)s(%(isoparametric_arguments)s);
                }""" % {
        "element": _mesh_element_name(element),
        "flag": flag,
        "affine_function": affine_function,
        "affine_arguments": affine_arguments,
        "isoparametric_function": isoparametric_function,
        "isoparametric_arguments": isoparametric_arguments,
        "field_stride": field_stride,
        "setup": "\n".join(setup_lines),
    }


def _safe_identifier(name):
    return re.sub(r"[^0-9A-Za-z_]", "_", str(name))


def _c_abi_header(material, kernel_sources):
    declarations = _extract_c_abi_declarations(kernel_sources)
    body = "\n\n".join(declarations)
    if body:
        body += "\n"
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
typedef double real_t;
typedef double geom_t;
#endif

#ifndef SFEM_RESTRICT
#define SFEM_RESTRICT __restrict__
#endif

#include "../kernel_diagnostics.hpp"

%(body)s""" % {
        "body": body,
    }


def _registration_source(material, wrapper_header):
    function = _registration_function(material)
    return """#include "%(header)s"
#include "sfem_OpFactory.hpp"

namespace sfem {
    void %(function)s() {
        Factory::register_op("%(op)s", %(op)s::create);
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
    declarations = _extract_c_abi_declarations(kernel_sources)
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


def _extract_c_abi_declarations(kernel_sources):
    declarations = {}
    for path, source in sorted(kernel_sources.items()):
        if not path.endswith((".cpp", ".hpp")) or path.startswith("op/"):
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
            if name and name not in declarations:
                declarations[name] = declaration
    return tuple(declarations[name] for name in sorted(declarations))


def _c_abi_function_name(declaration):
    match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", declaration)
    return match.group(1) if match else None


_RUNTIME_OPERATION_MARKERS = (
    ("jacobian_action", "_jacobian_action_"),
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


def _affine_option_entries(*flags):
    lines = []
    for flag in flags:
        for alias in _AFFINE_OPTION_ALIASES[flag]:
            lines.append('            {"%s", &impl_->%s},' % (alias, flag))
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
        + "".join(", const real_t *" for _ in range(dim * dim))
        + ", const real_t *"
        + parameter_decl
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
    )


def _case(element, function, arguments):
    return """                case smesh::%(element)s:
                    return %(function)s(%(arguments)s);""" % {
        "element": _mesh_element_name(element),
        "function": function,
        "arguments": arguments,
    }


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

        void material_defaults(real_t *const values) {
%(default_lines)s
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


def _affine_geometry_offsets(dim):
    return ", ".join("adjugate[%d]" % i for i in range(dim * dim))


def _boundary_surface_name(element):
    name = _element_name(element)
    surface_by_cell = {
        "TRI3": "edgeshell2",
        "QUAD4": "edgeshell2",
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
    if name in ("TRI3", "TRI6", "QUAD4") or name.startswith(("TRI6_", "QUAD4_")):
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
