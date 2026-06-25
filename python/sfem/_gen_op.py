def generate_op_files(material, elements):
    if hasattr(material, "energy"):
        header, source = _hyperelastic_op(material, elements)
    else:
        header, source = _residual_op(material, elements)
    return {
        "op/sfem_%s.hpp" % material.op_name: header,
        "op/sfem_%s.cpp" % material.op_name: source,
    }


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
        void set_value_in_block(const std::string &block_name,
                                const std::string &var_name,
                                real_t value) override;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
}  // namespace sfem
""" % {"op": material.op_name, "extra": extra}


def _hyperelastic_op(material, elements):
    parameters = tuple(str(name) for name, _ in material.parameter_defaults)
    defaults = _seed_lines(material.parameter_defaults)
    declarations = []
    gradient_cases = []
    apply_cases = []
    objective_cases = []
    for element in elements:
        dim = _element_dim(element)
        stem = "generated_%s_%s_%s" % (
            material.name,
            element.lower(),
            element.lower(),
        )
        components = _components(dim)
        declarations.extend(
            _hyperelastic_declarations(stem, dim, parameters)
        )
        args = _parameter_args(parameters)
        gradient_cases.append(
            _case(
                element,
                "%s_gradient_isoparametric_mesh_soa" % stem,
                "domain.block->n_elements(), mesh->n_nodes(), "
                "domain.block->elements()->data(), points%s, %d, %s, %d, %s"
                % (
                    args,
                    dim,
                    _offsets("x", components),
                    dim,
                    _offsets("out", components),
                ),
            )
        )
        apply_cases.append(
            _case(
                element,
                "%s_apply_isoparametric_mesh_soa" % stem,
                "domain.block->n_elements(), mesh->n_nodes(), "
                "domain.block->elements()->data(), points%s, %d, %s, %d, %s, %d, %s"
                % (
                    args,
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
            """                case smesh::%(element)s:
                    status = %(function)s(%(arguments)s);
                    break;"""
            % {
                "element": _mesh_element_name(element),
                "function": "%s_objective_isoparametric_mesh_soa" % stem,
                "arguments": (
                    "nelements, mesh->n_nodes(), "
                    "domain.block->elements()->data(), "
                    "points%s, %d, %s, impl_->element_values.get()"
                    % (args, dim, _offsets("x", components))
                ),
            }
        )

    source = """#include "sfem_%(op)s.hpp"

#include "sfem_FunctionSpace.hpp"
#include "sfem_MultiDomainOp.hpp"
#include "sfem_Parameters.hpp"
#include "smesh_mesh.hpp"

#include <algorithm>
#include <cstring>
#include <memory>

extern "C" {
%(declarations)s
}

namespace sfem {
    namespace {
        void seed_parameters(Parameters &parameters) {
%(defaults)s
        }
    }  // namespace

    class %(op)s::Impl {
    public:
        explicit Impl(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        std::shared_ptr<FunctionSpace> space;
        std::shared_ptr<MultiDomainOp> domains;
        std::unique_ptr<real_t[]> element_values;
        ptrdiff_t element_capacity{0};
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
        for (auto &entry : impl_->domains->domains()) {
            seed_parameters(*entry.second.parameters);
            impl_->element_capacity =
                    std::max(impl_->element_capacity, entry.second.block->n_elements());
        }
        impl_->element_values.reset(new real_t[impl_->element_capacity]);
        return SFEM_SUCCESS;
    }

    int %(op)s::gradient(const real_t *const x, real_t *const out) {
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
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

    void %(op)s::set_value_in_block(const std::string &block_name,
                                    const std::string &var_name,
                                    const real_t value) {
        impl_->domains->set_value_in_block(block_name, var_name, value);
    }
}  // namespace sfem
""" % {
        "op": material.op_name,
        "declarations": "\n".join(declarations),
        "defaults": defaults,
        "gradient_cases": "\n".join(gradient_cases),
        "apply_cases": "\n".join(apply_cases),
        "objective_cases": "\n".join(objective_cases),
    }
    return _header(material, False), source


def _residual_op(material, elements):
    defaults = _seed_lines(material.parameter_defaults)
    declarations = []
    residual_cases = []
    action_cases = []
    for element in elements:
        stem = "generated_%s_%s" % (material.name, element.lower())
        declarations.append(
            "int %s_residual_isoparametric_mesh_aos("
            "ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, "
            "const real_t *, const real_t *, const real_t *, real_t *);"
            % stem
        )
        declarations.append(
            "int %s_jacobian_action_isoparametric_mesh_aos("
            "ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, "
            "const real_t *, const real_t *, const real_t *, "
            "const real_t *, real_t *);" % stem
        )
        common = (
            "domain.block->n_elements(), mesh->n_nodes(), "
            "domain.block->elements()->data(), points, parameters"
        )
        residual_cases.append(
            _case(
                element,
                "%s_residual_isoparametric_mesh_aos" % stem,
                common + ", state, previous, out",
            )
        )
        action_cases.append(
            _case(
                element,
                "%s_jacobian_action_isoparametric_mesh_aos" % stem,
                common + ", state, previous, direction, out",
            )
        )

    parameter_names = tuple(str(name) for name, _ in material.parameter_defaults)
    parameter_lines = "\n".join(
        '            values[index++] = parameters.require_real_value("%s");' % name
        for name in parameter_names
        if not name.startswith("K_")
    )
    max_parameters = len(parameter_names)
    source = """#include "sfem_%(op)s.hpp"

#include "sfem_FunctionSpace.hpp"
#include "sfem_MultiDomainOp.hpp"
#include "sfem_Parameters.hpp"
#include "smesh_mesh.hpp"

#include <cstring>

extern "C" {
%(declarations)s
}

namespace sfem {
    namespace {
        constexpr int MAX_PARAMETERS = %(max_parameters)d;

        void seed_parameters(Parameters &parameters) {
%(defaults)s
        }

        void parameter_array(const Parameters &parameters,
                             const int dim,
                             real_t *const values) {
            int index = 0;
%(parameter_lines)s
            for (int i = 0; i < dim * dim; ++i) {
                values[index++] =
                        parameters.require_real_value("K_" + std::to_string(i));
            }
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
        if (space->block_size() != %(nfields)d) {
            SFEM_ERROR("%(op)s requires block_size=%(nfields)d\\n");
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
        for (auto &entry : impl_->domains->domains()) {
            seed_parameters(*entry.second.parameters);
        }
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
        if (!impl_->previous) {
            SFEM_ERROR("%(op)s requires a previous state\\n");
            return SFEM_FAILURE;
        }
        impl_->current = state;
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
            real_t storage[MAX_PARAMETERS];
            parameter_array(*domain.parameters,
                            mesh->spatial_dimension(),
                            storage);
            const real_t *const parameters = storage;
            const real_t *const previous = impl_->previous;
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
        if (!current || !impl_->previous) {
            SFEM_ERROR("%(op)s requires current and previous states\\n");
            return SFEM_FAILURE;
        }
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
            real_t storage[MAX_PARAMETERS];
            parameter_array(*domain.parameters,
                            mesh->spatial_dimension(),
                            storage);
            const real_t *const parameters = storage;
            const real_t *const previous = impl_->previous;
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
        "nfields": 2,
        "declarations": "\n".join(declarations),
        "max_parameters": max_parameters,
        "defaults": defaults,
        "parameter_lines": parameter_lines,
        "residual_cases": "\n".join(residual_cases),
        "action_cases": "\n".join(
            case.replace(", state, previous, direction, out", ", current, previous, direction, out")
            for case in action_cases
        ),
    }
    return _header(material, True), source


def _hyperelastic_declarations(stem, dim, parameters):
    components = _components(dim)
    parameter_decl = "".join(", const real_t %s" % name for name in parameters)
    vectors = "".join(", const real_t *" for _ in components)
    outputs = "".join(", real_t *" for _ in components)
    common = (
        "ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *"
        + parameter_decl
    )
    return (
        "int %s_objective_isoparametric_mesh_soa(%s, ptrdiff_t%s, real_t *);"
        % (stem, common, vectors),
        "int %s_gradient_isoparametric_mesh_soa(%s, ptrdiff_t%s, ptrdiff_t%s);"
        % (stem, common, vectors, outputs),
        "int %s_apply_isoparametric_mesh_soa(%s, ptrdiff_t%s, ptrdiff_t%s, ptrdiff_t%s);"
        % (stem, common, vectors, vectors, outputs),
    )


def _case(element, function, arguments):
    return """                case smesh::%(element)s:
                    return %(function)s(%(arguments)s);""" % {
        "element": _mesh_element_name(element),
        "function": function,
        "arguments": arguments,
    }


def _seed_lines(defaults):
    return "\n".join(
        '            parameters.set_value("%s", %.17g);' % (name, value)
        for name, value in defaults
    )


def _parameter_args(parameters):
    return "".join(
        ', domain.parameters->require_real_value("%s")' % name
        for name in parameters
    )


def _components(dim):
    return ("x", "y", "z")[:dim]


def _offsets(name, components):
    return ", ".join("%s + %d" % (name, i) for i, _ in enumerate(components))


def _element_dim(element):
    if element in ("TRI3", "TRI6", "QUAD4"):
        return 2
    if element in ("TET4", "TET10", "HEX8", "HEX27"):
        return 3
    raise ValueError("unsupported generated Op element %s" % element)


def _mesh_element_name(element):
    if element == "HEX27":
        return "PROTEUS_HEX27"
    return element
