import re


def generate_op_files(material, elements, kernel_sources=None):
    c_abi_header = "sfem_%s_c_abi.hpp" % material.op_name if kernel_sources else None
    if hasattr(material, "energy"):
        header, source = _hyperelastic_op(material, elements, c_abi_header)
    else:
        header, source = _residual_op(material, elements, c_abi_header)
    files = {
        "op/sfem_%s.hpp" % material.op_name: header,
        "op/sfem_%s.cpp" % material.op_name: source,
    }
    if c_abi_header:
        files["op/%s" % c_abi_header] = _c_abi_header(material, kernel_sources)
    return files


def _header(material, residual):
    extra = """
        int update(const real_t *const x) override;
        int update(const real_t *const previous, const real_t *const current) override;
        void set_field(const char *name,
                       const std::shared_ptr<Buffer<real_t>> &values,
                       int component) override;""" if residual else ""
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
        int value(const real_t *x, real_t *const out) override;
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
""" % {"op": material.op_name, "extra": extra}


def _hyperelastic_op(material, elements, c_abi_header=None):
    parameters = tuple(str(name) for name, _ in material.parameter_defaults)
    defaults = _seed_lines(material.parameter_defaults)
    declarations = []
    gradient_cases = []
    apply_cases = []
    objective_cases = []
    for element in elements:
        dim = _element_dim(element)
        stem = "%s_%s_%s" % (
            material.name,
            _element_name(element).lower(),
            _element_name(element).lower(),
        )
        components = _components(dim)
        declarations.extend(
            _hyperelastic_declarations(stem, dim, parameters)
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
                "%s, %d, %s, %d, %s"
                % (
                    common_affine_args,
                    dim,
                    _offsets("x", components),
                    dim,
                    _offsets("out", components),
                ),
                "%s_gradient_isoparametric_mesh_soa" % stem,
                "%s, %d, %s, %d, %s"
                % (
                    common_isoparametric_args,
                    dim,
                    _offsets("x", components),
                    dim,
                    _offsets("out", components),
                ),
            )
        )
        apply_cases.append(
            _dual_case(
                element,
                "apply_uses_affine",
                "%s_apply_affine_mesh_soa" % stem,
                "%s, %d, %s, %d, %s, %d, %s"
                % (
                    common_affine_args,
                    dim,
                    _offsets("x", components),
                    dim,
                    _offsets("h", components),
                    dim,
                    _offsets("out", components),
                ),
                "%s_apply_isoparametric_mesh_soa" % stem,
                "%s, %d, %s, %d, %s, %d, %s"
                % (
                    common_isoparametric_args,
                    dim,
                    _offsets("x", components),
                    dim,
                    _offsets("h", components),
                    dim,
                    _offsets("out", components),
                ),
            )
        )
        objective_cases.append(
            _dual_status_case(
                element,
                "%s_objective_affine_mesh_soa" % stem,
                "%s, %d, %s, impl_->element_values.get()"
                % (
                    "nelements, mesh->n_nodes(), domain.block->elements()->data(), %s, determinant%s"
                    % (_affine_geometry_offsets(dim), args),
                    dim,
                    _offsets("x", components),
                ),
                "%s_objective_isoparametric_mesh_soa" % stem,
                "%s, %d, %s, impl_->element_values.get()"
                % (
                    "nelements, mesh->n_nodes(), domain.block->elements()->data(), points%s"
                    % args,
                    dim,
                    _offsets("x", components),
                ),
            )
        )

    source = """#include "sfem_%(op)s.hpp"
%(c_abi_include)s

#include "sfem_FunctionSpace.hpp"
#include "sfem_MultiDomainOp.hpp"
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

    int %(op)s::hessian_crs(const real_t *const,
                            const count_t *const,
                            const idx_t *const,
                            real_t *const) {
        return SFEM_FAILURE;
    }

    void %(op)s::set_option(const std::string &name, const bool val) {
        if (name == "assume_affine") {
            impl_->objective_uses_affine = val;
            impl_->gradient_uses_affine = val;
            impl_->apply_uses_affine = val;
        } else if (name == "objective_assume_affine") {
            impl_->objective_uses_affine = val;
        } else if (name == "gradient_assume_affine") {
            impl_->gradient_uses_affine = val;
        } else if (name == "hessian_action_assume_affine" ||
                   name == "apply_assume_affine") {
            impl_->apply_uses_affine = val;
        }
    }

    void %(op)s::set_value_in_block(const std::string &block_name,
                                    const std::string &var_name,
                                    const real_t value) {
        impl_->domains->set_value_in_block(block_name, var_name, value);
    }

#ifdef SFEM_ENABLE_RYAML
    std::shared_ptr<Op> %(op)s::create_from_yaml(const std::shared_ptr<FunctionSpace> &space,
                                                 const ryml::ConstNodeRef             &node) {
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

        read_affine_options(node,
                            ret->impl_->objective_uses_affine,
                            ret->impl_->gradient_uses_affine,
                            ret->impl_->apply_uses_affine);

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
    }
    return _header(material, False), source


def _residual_op(material, elements, c_abi_header=None):
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
            collection = _residual_form_collection(material, dim)
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
        common = (
            "domain.block->n_elements(), mesh->n_nodes(), "
            "domain.block->elements()->data(), points"
        )
        residual_args = [common]
        residual_args.extend(
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
            residual_args.append("FIELD_STRIDE")
            residual_args.extend(
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
            residual_args.append("FIELD_STRIDE")
            residual_args.extend(
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
        residual_args.append("FIELD_STRIDE")
        residual_args.extend(_residual_soa_field_argument_names(fields_by_dim[dim], "out"))
        residual_cases.append(
            _residual_soa_case(
                element,
                "%s_residual_isoparametric_mesh_soa" % stem,
                ", ".join(residual_args),
                block_size_by_dim[dim],
                residual_setup,
            )
        )
        action_args = [common]
        action_args.extend(
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
            action_args.append("FIELD_STRIDE")
            action_args.extend(
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
            action_args.append("FIELD_STRIDE")
            action_args.extend(
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
            action_args.append("FIELD_STRIDE")
            action_args.extend(
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
        action_args.append("FIELD_STRIDE")
        action_args.extend(_residual_soa_field_argument_names(fields_by_dim[dim], "out"))
        action_cases.append(
            _residual_soa_case(
                element,
                "%s_jacobian_action_isoparametric_mesh_soa" % stem,
                ", ".join(action_args),
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
#include "sfem_Parameters.hpp"
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
        impl_->domains = std::make_shared<MultiDomainOp>(impl_->space, block_names);
        seed_material(*impl_->domains);
        return SFEM_SUCCESS;
    }

    int %(op)s::update(const real_t *const x) {
        impl_->current = x;
        return SFEM_SUCCESS;
    }

    int %(op)s::update(const real_t *const previous,
                       const real_t *const current) {
        impl_->previous_buffer.reset();
        impl_->previous = previous;
        impl_->current = current;
        return SFEM_SUCCESS;
    }

    int %(op)s::gradient(const real_t *const state, real_t *const out) {
%(gradient_previous_check)s
        impl_->current = state;
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
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
        const real_t *const current = state ? state : impl_->current;
%(apply_state_check)s
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
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
        impl_->domains->set_value_in_block(block_name, var_name, value);
    }

    void %(op)s::set_option(const std::string &, const bool) {}

#ifdef SFEM_ENABLE_RYAML
    std::shared_ptr<Op> %(op)s::create_from_yaml(const std::shared_ptr<FunctionSpace> &space,
                                                 const ryml::ConstNodeRef             &node) {
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
        return SFEM_FAILURE;
    }

    int %(op)s::value(const real_t *, real_t *const) {
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


def _residual_form_collection(material, dim):
    collections = getattr(material, "form_collections", None)
    if collections is not None:
        return collections[dim]

    from codegen.framework.equations import EquationSystem

    system = EquationSystem(dim)
    equation = system.add_residual("", material.define, fields=())
    return system.form_collection(equation)


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


def _safe_identifier(name):
    return re.sub(r"[^0-9A-Za-z_]", "_", str(name))


def _c_abi_header(material, kernel_sources):
    declarations = _extract_c_abi_declarations(kernel_sources)
    body = "\n\n".join(declarations)
    if body:
        body += "\n"
    return """#pragma once

#include <cstddef>

#if defined(__has_include)
#if __has_include("sfem_base.hpp")
#include "sfem_base.hpp"
#define SFEM_CODEGEN_OP_HAS_SFEM_BASE
#endif
#endif

#ifndef SFEM_CODEGEN_OP_HAS_SFEM_BASE
typedef ptrdiff_t idx_t;
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


def _hyperelastic_declarations(stem, dim, parameters):
    components = _components(dim)
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
        "int %s_objective_isoparametric_mesh_soa(%s, ptrdiff_t%s, real_t *);"
        % (stem, isoparametric_common, vectors),
        "int %s_gradient_isoparametric_mesh_soa(%s, ptrdiff_t%s, ptrdiff_t%s);"
        % (stem, isoparametric_common, vectors, outputs),
        "int %s_apply_isoparametric_mesh_soa(%s, ptrdiff_t%s, ptrdiff_t%s, ptrdiff_t%s);"
        % (stem, isoparametric_common, vectors, vectors, outputs),
        "int %s_objective_affine_mesh_soa(%s, ptrdiff_t%s, real_t *);"
        % (stem, affine_common, vectors),
        "int %s_gradient_affine_mesh_soa(%s, ptrdiff_t%s, ptrdiff_t%s);"
        % (stem, affine_common, vectors, outputs),
        "int %s_apply_affine_mesh_soa(%s, ptrdiff_t%s, ptrdiff_t%s, ptrdiff_t%s);"
        % (stem, affine_common, vectors, vectors, outputs),
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
    return """#ifdef SFEM_ENABLE_RYAML
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

        void read_affine_options(const ryml::ConstNodeRef &node,
                                 bool &objective,
                                 bool &gradient,
                                 bool &hessian_action) {
            bool all = objective && gradient && hessian_action;
            if (yaml_read_bool(node, "ASSUME_AFFINE", all) ||
                yaml_read_bool(node, "assume_affine", all)) {
                objective = all;
                gradient = all;
                hessian_action = all;
            }
            yaml_read_bool(node, "ASSUME_AFFINE_OBJECTIVE", objective);
            yaml_read_bool(node, "objective_assume_affine", objective);
            yaml_read_bool(node, "ASSUME_AFFINE_GRADIENT", gradient);
            yaml_read_bool(node, "gradient_assume_affine", gradient);
            yaml_read_bool(node, "ASSUME_AFFINE_HESSIAN_ACTION", hessian_action);
            yaml_read_bool(node, "hessian_action_assume_affine", hessian_action);
            yaml_read_bool(node, "ASSUME_AFFINE_APPLY", hessian_action);
            yaml_read_bool(node, "apply_assume_affine", hessian_action);
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
