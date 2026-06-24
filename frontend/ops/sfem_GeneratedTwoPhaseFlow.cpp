#include "sfem_GeneratedTwoPhaseFlow.hpp"

#include "sfem_FunctionSpace.hpp"
#include "sfem_MultiDomainOp.hpp"
#include "sfem_Parameters.hpp"
#include "smesh_env.hpp"
#include "smesh_mesh.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>

extern "C" {
#define DECLARE_TWO_PHASE_KERNELS(element)                                                        \
    int generated_two_phase_flow_##element##_residual_isoparametric_mesh_aos(                    \
            ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t *,                \
            const real_t *, const real_t *, real_t *);                                            \
    int generated_two_phase_flow_##element##_jacobian_action_isoparametric_mesh_aos(              \
            ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t *,                \
            const real_t *, const real_t *, const real_t *, real_t *)

DECLARE_TWO_PHASE_KERNELS(tri3);
DECLARE_TWO_PHASE_KERNELS(tet4);
DECLARE_TWO_PHASE_KERNELS(quad4);
DECLARE_TWO_PHASE_KERNELS(hex8);
#undef DECLARE_TWO_PHASE_KERNELS
}

namespace sfem {
    namespace {
        constexpr int MAX_PARAMETERS = 26;

        using ResidualKernel = int (*)(ptrdiff_t,
                                       ptrdiff_t,
                                       idx_t **,
                                       const geom_t *const *,
                                       const real_t *,
                                       const real_t *,
                                       const real_t *,
                                       real_t *);
        using ActionKernel = int (*)(ptrdiff_t,
                                     ptrdiff_t,
                                     idx_t **,
                                     const geom_t *const *,
                                     const real_t *,
                                     const real_t *,
                                     const real_t *,
                                     const real_t *,
                                     real_t *);

        ResidualKernel residual_kernel(const smesh::ElemType type) {
            switch (type) {
                case smesh::TRI3:
                    return generated_two_phase_flow_tri3_residual_isoparametric_mesh_aos;
                case smesh::TET4:
                    return generated_two_phase_flow_tet4_residual_isoparametric_mesh_aos;
                case smesh::QUAD4:
                    return generated_two_phase_flow_quad4_residual_isoparametric_mesh_aos;
                case smesh::HEX8:
                    return generated_two_phase_flow_hex8_residual_isoparametric_mesh_aos;
                default:
                    return nullptr;
            }
        }

        ActionKernel action_kernel(const smesh::ElemType type) {
            switch (type) {
                case smesh::TRI3:
                    return generated_two_phase_flow_tri3_jacobian_action_isoparametric_mesh_aos;
                case smesh::TET4:
                    return generated_two_phase_flow_tet4_jacobian_action_isoparametric_mesh_aos;
                case smesh::QUAD4:
                    return generated_two_phase_flow_quad4_jacobian_action_isoparametric_mesh_aos;
                case smesh::HEX8:
                    return generated_two_phase_flow_hex8_jacobian_action_isoparametric_mesh_aos;
                default:
                    return nullptr;
            }
        }

        void seed_parameters(Parameters &p) {
            p.set_value("porosity", 0.2);
            p.set_value("S_res", 0.1);
            p.set_value("P_r", 1e5);
            p.set_value("m", 2.0);
            p.set_value("rho_w0", 1000.0);
            p.set_value("kappa_T", 1e-9);
            p.set_value("p_wr", 1e5);
            p.set_value("M_c", 0.044);
            p.set_value("Z", 1.0);
            p.set_value("R", 8.314462618);
            p.set_value("T", 300.0);
            p.set_value("mu_w", 1e-3);
            p.set_value("mu_c", 1.5e-5);
            p.set_value("C_kw1", 2.0);
            p.set_value("C_ka1", 2.0);
            p.set_value("C_ka2", 2.0);
            p.set_value("dt", 1.0);
            for (int i = 0; i < 9; ++i) {
                p.set_value("K_" + std::to_string(i), i % 4 == 0 ? 1e-12 : 0.0);
            }
        }

        void parameter_array(const Parameters &p, const int dim, real_t *const values) {
            static const char *names[] = {"porosity", "S_res", "P_r", "m", "rho_w0", "kappa_T",
                                          "p_wr", "M_c", "Z", "R", "T", "mu_w", "mu_c", "C_kw1",
                                          "C_ka1", "C_ka2", "dt"};
            for (int i = 0; i < 17; ++i) {
                values[i] = p.require_real_value(names[i]);
            }
            for (int i = 0; i < dim * dim; ++i) {
                values[17 + i] = p.require_real_value("K_" + std::to_string(i));
            }
        }
    }  // namespace

    class GeneratedTwoPhaseFlow::Impl {
    public:
        explicit Impl(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        std::shared_ptr<FunctionSpace> space;
        std::shared_ptr<MultiDomainOp> domains;
        std::shared_ptr<Buffer<real_t>> previous;
        const real_t *previous_ptr{nullptr};
        const real_t *current{nullptr};
        GeneratedTwoPhaseFlowStats stats;

        int residual(const real_t *const state, real_t *const out) const {
            if (!previous_ptr) {
                SFEM_ERROR("GeneratedTwoPhaseFlow requires a previous state\n");
                return SFEM_FAILURE;
            }

            auto mesh = space->mesh_ptr();
            auto points = const_cast<const geom_t *const *>(mesh->points()->data());
            return domains->iterate([&](const OpDomain &domain) {
                auto kernel = residual_kernel(domain.element_type);
                if (!kernel) {
                    SFEM_ERROR("GeneratedTwoPhaseFlow does not support element type %d\n", domain.element_type);
                    return SFEM_FAILURE;
                }
                std::array<real_t, MAX_PARAMETERS> parameters{};
                parameter_array(*domain.parameters, mesh->spatial_dimension(), parameters.data());
                return kernel(domain.block->n_elements(),
                              mesh->n_nodes(),
                              domain.block->elements()->data(),
                              points,
                              parameters.data(),
                              state,
                              previous_ptr,
                              out);
            });
        }

        int action(const real_t *const state, const real_t *const direction, real_t *const out) const {
            if (!state || !previous_ptr) {
                SFEM_ERROR("GeneratedTwoPhaseFlow requires current and previous states\n");
                return SFEM_FAILURE;
            }

            auto mesh = space->mesh_ptr();
            auto points = const_cast<const geom_t *const *>(mesh->points()->data());
            return domains->iterate([&](const OpDomain &domain) {
                auto kernel = action_kernel(domain.element_type);
                if (!kernel) {
                    SFEM_ERROR("GeneratedTwoPhaseFlow does not support element type %d\n", domain.element_type);
                    return SFEM_FAILURE;
                }
                std::array<real_t, MAX_PARAMETERS> parameters{};
                parameter_array(*domain.parameters, mesh->spatial_dimension(), parameters.data());
                return kernel(domain.block->n_elements(),
                              mesh->n_nodes(),
                              domain.block->elements()->data(),
                              points,
                              parameters.data(),
                              state,
                              previous_ptr,
                              direction,
                              out);
            });
        }
    };

    std::unique_ptr<Op> GeneratedTwoPhaseFlow::create(const std::shared_ptr<FunctionSpace> &space) {
        if (space->block_size() != 2) {
            SFEM_ERROR("GeneratedTwoPhaseFlow requires block_size=2\n");
            return nullptr;
        }
        auto ret = std::make_unique<GeneratedTwoPhaseFlow>(space);
        ret->initialize();
        return ret;
    }

    GeneratedTwoPhaseFlow::GeneratedTwoPhaseFlow(const std::shared_ptr<FunctionSpace> &space)
        : impl_(std::make_unique<Impl>(space)) {}
    GeneratedTwoPhaseFlow::~GeneratedTwoPhaseFlow() = default;

    ptrdiff_t GeneratedTwoPhaseFlow::n_dofs_domain() const { return impl_->space->n_dofs(); }
    ptrdiff_t GeneratedTwoPhaseFlow::n_dofs_image() const { return impl_->space->n_dofs(); }

    int GeneratedTwoPhaseFlow::initialize(const std::vector<std::string> &block_names) {
        impl_->domains = std::make_shared<MultiDomainOp>(impl_->space, block_names);
        for (auto &domain : impl_->domains->domains()) {
            seed_parameters(*domain.second.parameters);
        }
        return SFEM_SUCCESS;
    }

    int GeneratedTwoPhaseFlow::update(const real_t *const x) {
        impl_->current = x;
        return SFEM_SUCCESS;
    }

    int GeneratedTwoPhaseFlow::update(const real_t *const x_prev, const real_t *const x_curr) {
        impl_->previous.reset();
        impl_->previous_ptr = x_prev;
        impl_->current = x_curr;
        return SFEM_SUCCESS;
    }

    int GeneratedTwoPhaseFlow::gradient(const real_t *const x, real_t *const out) {
        impl_->current = x;
        const auto begin = std::chrono::steady_clock::now();
        const int status = impl_->residual(x, out);
        const auto end = std::chrono::steady_clock::now();
        impl_->stats.residual_seconds +=
                std::chrono::duration<double>(end - begin).count();
        ++impl_->stats.residual_calls;
        return status;
    }

    int GeneratedTwoPhaseFlow::apply(const real_t *const x, const real_t *const h, real_t *const out) {
        const real_t *state = x ? x : impl_->current;
        const auto begin = std::chrono::steady_clock::now();
        const int status = impl_->action(state, h, out);
        const auto end = std::chrono::steady_clock::now();
        impl_->stats.jacobian_seconds +=
                std::chrono::duration<double>(end - begin).count();
        ++impl_->stats.jacobian_calls;
        return status;
    }

    void GeneratedTwoPhaseFlow::set_field(const char *name,
                                          const std::shared_ptr<Buffer<real_t>> &values,
                                          const int component) {
        if (component != 0 || std::strcmp(name, "previous") != 0) {
            SFEM_ERROR("GeneratedTwoPhaseFlow supports set_field(\"previous\", buffer, 0)\n");
            return;
        }
        impl_->previous = values;
        impl_->previous_ptr = values->data();
    }

    void GeneratedTwoPhaseFlow::set_value_in_block(const std::string &block_name,
                                                    const std::string &var_name,
                                                    const real_t value) {
        impl_->domains->set_value_in_block(block_name, var_name, value);
    }

    std::shared_ptr<Operator<real_t>> GeneratedTwoPhaseFlow::make_linear_operator(
            const real_t *current,
            std::function<void(const real_t *const, real_t *const)> apply_constraints) {
        update(current);
        const ptrdiff_t ndofs = n_dofs_domain();
        return make_op<real_t>(
                ndofs,
                ndofs,
                [this, apply_constraints](const real_t *const direction, real_t *const out) {
                    std::fill(out, out + n_dofs_image(), 0);
                    const int err = apply(nullptr, direction, out);
                    if (err != SFEM_SUCCESS) {
                        SFEM_ERROR("GeneratedTwoPhaseFlow Jacobian action failed\n");
                    }
                    if (apply_constraints) {
                        apply_constraints(direction, out);
                    }
                },
                EXECUTION_SPACE_HOST);
    }

    void GeneratedTwoPhaseFlow::reset_stats() {
        impl_->stats = {};
    }

    GeneratedTwoPhaseFlowStats GeneratedTwoPhaseFlow::stats() const {
        return impl_->stats;
    }

    int GeneratedTwoPhaseFlow::hessian_crs(const real_t *const,
                                            const count_t *const,
                                            const idx_t *const,
                                            real_t *const) {
        return SFEM_FAILURE;
    }

    int GeneratedTwoPhaseFlow::value(const real_t *, real_t *const) { return SFEM_FAILURE; }
}  // namespace sfem
