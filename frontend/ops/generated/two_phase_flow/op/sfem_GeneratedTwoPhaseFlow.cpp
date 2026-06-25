#include "sfem_GeneratedTwoPhaseFlow.hpp"

#include "sfem_FunctionSpace.hpp"
#include "sfem_MultiDomainOp.hpp"
#include "sfem_Parameters.hpp"
#include "smesh_mesh.hpp"

#include <cstring>

extern "C" {
int generated_two_phase_flow_tri3_residual_isoparametric_mesh_aos(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t *, const real_t *, const real_t *, real_t *);
int generated_two_phase_flow_tri3_jacobian_action_isoparametric_mesh_aos(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t *, const real_t *, const real_t *, const real_t *, real_t *);
int generated_two_phase_flow_tet4_residual_isoparametric_mesh_aos(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t *, const real_t *, const real_t *, real_t *);
int generated_two_phase_flow_tet4_jacobian_action_isoparametric_mesh_aos(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t *, const real_t *, const real_t *, const real_t *, real_t *);
int generated_two_phase_flow_quad4_residual_isoparametric_mesh_aos(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t *, const real_t *, const real_t *, real_t *);
int generated_two_phase_flow_quad4_jacobian_action_isoparametric_mesh_aos(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t *, const real_t *, const real_t *, const real_t *, real_t *);
int generated_two_phase_flow_hex8_residual_isoparametric_mesh_aos(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t *, const real_t *, const real_t *, real_t *);
int generated_two_phase_flow_hex8_jacobian_action_isoparametric_mesh_aos(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t *, const real_t *, const real_t *, const real_t *, real_t *);
}

namespace sfem {
    namespace {
        constexpr int MAX_PARAMETERS = 26;

        void seed_parameters(Parameters &parameters) {
            parameters.set_value("porosity", 0.20000000000000001);
            parameters.set_value("S_res", 0.10000000000000001);
            parameters.set_value("P_r", 100000);
            parameters.set_value("m", 2);
            parameters.set_value("rho_w0", 1000);
            parameters.set_value("kappa_T", 1.0000000000000001e-09);
            parameters.set_value("p_wr", 100000);
            parameters.set_value("M_c", 0.043999999999999997);
            parameters.set_value("Z", 1);
            parameters.set_value("R", 8.3144626180000003);
            parameters.set_value("T", 300);
            parameters.set_value("mu_w", 0.001);
            parameters.set_value("mu_c", 1.5e-05);
            parameters.set_value("C_kw1", 2);
            parameters.set_value("C_ka1", 2);
            parameters.set_value("C_ka2", 2);
            parameters.set_value("dt", 1);
            parameters.set_value("K_0", 9.9999999999999998e-13);
            parameters.set_value("K_1", 0);
            parameters.set_value("K_2", 0);
            parameters.set_value("K_3", 0);
            parameters.set_value("K_4", 9.9999999999999998e-13);
            parameters.set_value("K_5", 0);
            parameters.set_value("K_6", 0);
            parameters.set_value("K_7", 0);
            parameters.set_value("K_8", 9.9999999999999998e-13);
        }

        void parameter_array(const Parameters &parameters,
                             const int dim,
                             real_t *const values) {
            int index = 0;
            values[index++] = parameters.require_real_value("porosity");
            values[index++] = parameters.require_real_value("S_res");
            values[index++] = parameters.require_real_value("P_r");
            values[index++] = parameters.require_real_value("m");
            values[index++] = parameters.require_real_value("rho_w0");
            values[index++] = parameters.require_real_value("kappa_T");
            values[index++] = parameters.require_real_value("p_wr");
            values[index++] = parameters.require_real_value("M_c");
            values[index++] = parameters.require_real_value("Z");
            values[index++] = parameters.require_real_value("R");
            values[index++] = parameters.require_real_value("T");
            values[index++] = parameters.require_real_value("mu_w");
            values[index++] = parameters.require_real_value("mu_c");
            values[index++] = parameters.require_real_value("C_kw1");
            values[index++] = parameters.require_real_value("C_ka1");
            values[index++] = parameters.require_real_value("C_ka2");
            values[index++] = parameters.require_real_value("dt");
            for (int i = 0; i < dim * dim; ++i) {
                values[index++] =
                        parameters.require_real_value("K_" + std::to_string(i));
            }
        }
    }  // namespace

    class GeneratedTwoPhaseFlow::Impl {
    public:
        explicit Impl(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        std::shared_ptr<FunctionSpace> space;
        std::shared_ptr<MultiDomainOp> domains;
        std::shared_ptr<Buffer<real_t>> previous_buffer;
        const real_t *previous{nullptr};
        const real_t *current{nullptr};
    };

    std::unique_ptr<Op> GeneratedTwoPhaseFlow::create(const std::shared_ptr<FunctionSpace> &space) {
        if (space->block_size() != 2) {
            SFEM_ERROR("GeneratedTwoPhaseFlow requires block_size=2\n");
            return nullptr;
        }
        auto op = std::make_unique<GeneratedTwoPhaseFlow>(space);
        op->initialize();
        return op;
    }

    GeneratedTwoPhaseFlow::GeneratedTwoPhaseFlow(const std::shared_ptr<FunctionSpace> &space)
        : impl_(std::make_unique<Impl>(space)) {}
    GeneratedTwoPhaseFlow::~GeneratedTwoPhaseFlow() = default;

    ptrdiff_t GeneratedTwoPhaseFlow::n_dofs_domain() const { return impl_->space->n_dofs(); }
    ptrdiff_t GeneratedTwoPhaseFlow::n_dofs_image() const { return impl_->space->n_dofs(); }

    int GeneratedTwoPhaseFlow::initialize(const std::vector<std::string> &block_names) {
        impl_->domains = std::make_shared<MultiDomainOp>(impl_->space, block_names);
        for (auto &entry : impl_->domains->domains()) {
            seed_parameters(*entry.second.parameters);
        }
        return SFEM_SUCCESS;
    }

    int GeneratedTwoPhaseFlow::update(const real_t *const x) {
        impl_->current = x;
        return SFEM_SUCCESS;
    }

    int GeneratedTwoPhaseFlow::update(const real_t *const previous,
                       const real_t *const current) {
        impl_->previous_buffer.reset();
        impl_->previous = previous;
        impl_->current = current;
        return SFEM_SUCCESS;
    }

    int GeneratedTwoPhaseFlow::gradient(const real_t *const state, real_t *const out) {
        if (!impl_->previous) {
            SFEM_ERROR("GeneratedTwoPhaseFlow requires a previous state\n");
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
                case smesh::TRI3:
                    return generated_two_phase_flow_tri3_residual_isoparametric_mesh_aos(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, parameters, state, previous, out);
                case smesh::TET4:
                    return generated_two_phase_flow_tet4_residual_isoparametric_mesh_aos(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, parameters, state, previous, out);
                case smesh::QUAD4:
                    return generated_two_phase_flow_quad4_residual_isoparametric_mesh_aos(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, parameters, state, previous, out);
                case smesh::HEX8:
                    return generated_two_phase_flow_hex8_residual_isoparametric_mesh_aos(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, parameters, state, previous, out);
                default:
                    SFEM_ERROR("GeneratedTwoPhaseFlow does not support element type %d\n",
                               domain.element_type);
                    return SFEM_FAILURE;
            }
        });
    }

    int GeneratedTwoPhaseFlow::apply(const real_t *const state,
                      const real_t *const direction,
                      real_t *const out) {
        const real_t *const current = state ? state : impl_->current;
        if (!current || !impl_->previous) {
            SFEM_ERROR("GeneratedTwoPhaseFlow requires current and previous states\n");
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
                case smesh::TRI3:
                    return generated_two_phase_flow_tri3_jacobian_action_isoparametric_mesh_aos(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, parameters, current, previous, direction, out);
                case smesh::TET4:
                    return generated_two_phase_flow_tet4_jacobian_action_isoparametric_mesh_aos(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, parameters, current, previous, direction, out);
                case smesh::QUAD4:
                    return generated_two_phase_flow_quad4_jacobian_action_isoparametric_mesh_aos(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, parameters, current, previous, direction, out);
                case smesh::HEX8:
                    return generated_two_phase_flow_hex8_jacobian_action_isoparametric_mesh_aos(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, parameters, current, previous, direction, out);
                default:
                    SFEM_ERROR("GeneratedTwoPhaseFlow does not support element type %d\n",
                               domain.element_type);
                    return SFEM_FAILURE;
            }
        });
    }

    void GeneratedTwoPhaseFlow::set_field(const char *name,
                           const std::shared_ptr<Buffer<real_t>> &values,
                           const int component) {
        if (component != 0 || std::strcmp(name, "previous") != 0) {
            SFEM_ERROR("GeneratedTwoPhaseFlow supports set_field(\"previous\", buffer, 0)\n");
            return;
        }
        impl_->previous_buffer = values;
        impl_->previous = values->data();
    }

    void GeneratedTwoPhaseFlow::set_value_in_block(const std::string &block_name,
                                    const std::string &var_name,
                                    const real_t value) {
        impl_->domains->set_value_in_block(block_name, var_name, value);
    }

    int GeneratedTwoPhaseFlow::hessian_crs(const real_t *const,
                            const count_t *const,
                            const idx_t *const,
                            real_t *const) {
        return SFEM_FAILURE;
    }

    int GeneratedTwoPhaseFlow::value(const real_t *, real_t *const) {
        return SFEM_FAILURE;
    }
}  // namespace sfem
