#include "sfem_TwoPhaseFlowTimeIntegration.hpp"

#include "sfem_base.hpp"
#include "smesh_glob.hpp"
#include "smesh_mesh.hpp"
#include "smesh_path.hpp"
#include "smesh_types.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>

namespace sfem {
    class TwoPhaseFlowTimeIntegration::Impl {
    public:
        Impl(const std::shared_ptr<smesh::Mesh> &mesh,
             const TwoPhaseFlowTimeConfig &config)
            : mesh(mesh), config(config) {}

        std::shared_ptr<smesh::Mesh> mesh;
        TwoPhaseFlowTimeConfig config;
        std::shared_ptr<Buffer<real_t>> accepted;
        std::shared_ptr<Buffer<real_t>> trial;
        std::shared_ptr<Buffer<idx_t>> left_nodes;
        std::shared_ptr<Buffer<idx_t>> right_nodes;
        real_t time{0};
        ptrdiff_t step{0};

        int find_boundary_nodes() {
            const ptrdiff_t nnodes = mesh->n_nodes();
            const smesh::geom_t *const x = mesh->points()->data()[0];
            smesh::geom_t xmin = std::numeric_limits<smesh::geom_t>::max();
            smesh::geom_t xmax = std::numeric_limits<smesh::geom_t>::lowest();
            for (ptrdiff_t node = 0; node < nnodes; ++node) {
                xmin = std::min(xmin, x[node]);
                xmax = std::max(xmax, x[node]);
            }
            const smesh::geom_t tolerance =
                    std::max<smesh::geom_t>(1, std::abs(xmax - xmin)) *
                    64 * std::numeric_limits<smesh::geom_t>::epsilon();
            ptrdiff_t nleft = 0;
            ptrdiff_t nright = 0;
            for (ptrdiff_t node = 0; node < nnodes; ++node) {
                nleft += std::abs(x[node] - xmin) <= tolerance;
                nright += std::abs(x[node] - xmax) <= tolerance;
            }
            left_nodes = create_host_buffer<idx_t>(nleft);
            right_nodes = create_host_buffer<idx_t>(nright);
            nleft = 0;
            nright = 0;
            for (ptrdiff_t node = 0; node < nnodes; ++node) {
                if (std::abs(x[node] - xmin) <= tolerance) {
                    left_nodes->data()[nleft++] = node;
                }
                if (std::abs(x[node] - xmax) <= tolerance) {
                    right_nodes->data()[nright++] = node;
                }
            }
            return SFEM_SUCCESS;
        }
    };

    TwoPhaseFlowTimeIntegration::TwoPhaseFlowTimeIntegration(
            const std::shared_ptr<smesh::Mesh> &mesh,
            const TwoPhaseFlowTimeConfig &config)
        : impl_(std::make_unique<Impl>(mesh, config)) {}

    TwoPhaseFlowTimeIntegration::~TwoPhaseFlowTimeIntegration() = default;

    int TwoPhaseFlowTimeIntegration::initialize() {
        const ptrdiff_t ndofs = 2 * impl_->mesh->n_nodes();
        impl_->accepted = create_host_buffer<real_t>(ndofs);
        impl_->trial = create_host_buffer<real_t>(ndofs);
        for (ptrdiff_t node = 0; node < impl_->mesh->n_nodes(); ++node) {
            impl_->accepted->data()[2 * node + 0] =
                    impl_->config.initial_water_pressure;
            impl_->accepted->data()[2 * node + 1] =
                    impl_->config.initial_co2_pressure;
        }
        impl_->time = 0;
        impl_->step = 0;
        impl_->find_boundary_nodes();
        apply_boundary(0, impl_->accepted->data());
        std::copy(impl_->accepted->data(),
                  impl_->accepted->data() + ndofs,
                  impl_->trial->data());
        return SFEM_SUCCESS;
    }

    void TwoPhaseFlowTimeIntegration::apply_boundary(
            const real_t time,
            real_t *const state) const {
        const real_t ramp =
                impl_->config.ramp_duration > 0
                        ? std::min<real_t>(1, std::max<real_t>(0, time / impl_->config.ramp_duration))
                        : 1;
        const real_t left_co2 =
                impl_->config.initial_co2_pressure +
                ramp * (impl_->config.injection_co2_pressure -
                        impl_->config.initial_co2_pressure);
        for (size_t i = 0; i < impl_->left_nodes->size(); ++i) {
            const idx_t node = impl_->left_nodes->data()[i];
            state[2 * node + 0] = impl_->config.initial_water_pressure;
            state[2 * node + 1] = left_co2;
        }
        for (size_t i = 0; i < impl_->right_nodes->size(); ++i) {
            const idx_t node = impl_->right_nodes->data()[i];
            state[2 * node + 0] = impl_->config.initial_water_pressure;
            state[2 * node + 1] = impl_->config.initial_co2_pressure;
        }
    }

    void TwoPhaseFlowTimeIntegration::constrain_residual(
            real_t *const residual) const {
        for (size_t i = 0; i < impl_->left_nodes->size(); ++i) {
            const idx_t node = impl_->left_nodes->data()[i];
            residual[2 * node + 0] = 0;
            residual[2 * node + 1] = 0;
        }
        for (size_t i = 0; i < impl_->right_nodes->size(); ++i) {
            const idx_t node = impl_->right_nodes->data()[i];
            residual[2 * node + 0] = 0;
            residual[2 * node + 1] = 0;
        }
    }

    void TwoPhaseFlowTimeIntegration::constrain_direction(
            real_t *const direction) const {
        constrain_residual(direction);
    }

    void TwoPhaseFlowTimeIntegration::constrain_linear(
            const real_t *const direction,
            real_t *const output) const {
        for (size_t i = 0; i < impl_->left_nodes->size(); ++i) {
            const idx_t node = impl_->left_nodes->data()[i];
            output[2 * node + 0] = direction[2 * node + 0];
            output[2 * node + 1] = direction[2 * node + 1];
        }
        for (size_t i = 0; i < impl_->right_nodes->size(); ++i) {
            const idx_t node = impl_->right_nodes->data()[i];
            output[2 * node + 0] = direction[2 * node + 0];
            output[2 * node + 1] = direction[2 * node + 1];
        }
    }

    TwoPhaseFlowBalance TwoPhaseFlowTimeIntegration::balance(
            const real_t *const residual) const {
        TwoPhaseFlowBalance result;
        const ptrdiff_t nnodes = impl_->mesh->n_nodes();
        for (ptrdiff_t node = 0; node < nnodes; ++node) {
            result.total[0] += residual[2 * node + 0];
            result.total[1] += residual[2 * node + 1];
        }
        for (size_t i = 0; i < impl_->left_nodes->size(); ++i) {
            const idx_t node = impl_->left_nodes->data()[i];
            result.left[0] += residual[2 * node + 0];
            result.left[1] += residual[2 * node + 1];
        }
        for (size_t i = 0; i < impl_->right_nodes->size(); ++i) {
            const idx_t node = impl_->right_nodes->data()[i];
            result.right[0] += residual[2 * node + 0];
            result.right[1] += residual[2 * node + 1];
        }
        result.interior[0] =
                result.total[0] - result.left[0] - result.right[0];
        result.interior[1] =
                result.total[1] - result.left[1] - result.right[1];
        return result;
    }

    int TwoPhaseFlowTimeIntegration::advance(
            const real_t dt,
            const StepSolver &solver) {
        if (!(dt > 0) || !solver) {
            return SFEM_FAILURE;
        }
        const ptrdiff_t ndofs = 2 * impl_->mesh->n_nodes();
        std::copy(impl_->accepted->data(),
                  impl_->accepted->data() + ndofs,
                  impl_->trial->data());
        const real_t next_time = impl_->time + dt;
        apply_boundary(next_time, impl_->trial->data());
        if (solver(impl_->accepted->data(), impl_->trial->data(), next_time, dt) !=
            SFEM_SUCCESS) {
            return SFEM_FAILURE;
        }
        apply_boundary(next_time, impl_->trial->data());
        std::swap(impl_->accepted, impl_->trial);
        impl_->time = next_time;
        ++impl_->step;
        return SFEM_SUCCESS;
    }

    int TwoPhaseFlowTimeIntegration::save_restart(
            const smesh::Path &folder) const {
        if (smesh::create_directory(folder) != SFEM_SUCCESS) {
            return SFEM_FAILURE;
        }
        const smesh::Path state_path =
                folder / (std::string("state.") +
                          std::string(smesh::TypeToString<real_t>::value()));
        if (impl_->accepted->to_file(state_path) != SFEM_SUCCESS) {
            return SFEM_FAILURE;
        }
        std::ofstream metadata((folder / "restart.txt").c_str());
        if (!metadata) {
            return SFEM_FAILURE;
        }
        metadata.precision(17);
        metadata << impl_->step << "\n" << impl_->time << "\n";
        return metadata ? SFEM_SUCCESS : SFEM_FAILURE;
    }

    int TwoPhaseFlowTimeIntegration::load_restart(
            const smesh::Path &folder) {
        const smesh::Path state_path =
                folder / (std::string("state.") +
                          std::string(smesh::TypeToString<real_t>::value()));
        auto state = Buffer<real_t>::from_file(state_path);
        if (!state || state->size() != 2 * impl_->mesh->n_nodes()) {
            return SFEM_FAILURE;
        }
        std::ifstream metadata((folder / "restart.txt").c_str());
        if (!metadata || !(metadata >> impl_->step >> impl_->time)) {
            return SFEM_FAILURE;
        }
        impl_->accepted = state;
        impl_->trial = create_host_buffer<real_t>(state->size());
        std::copy(state->data(), state->data() + state->size(), impl_->trial->data());
        impl_->find_boundary_nodes();
        apply_boundary(impl_->time, impl_->accepted->data());
        return SFEM_SUCCESS;
    }

    const std::shared_ptr<Buffer<real_t>> &
    TwoPhaseFlowTimeIntegration::accepted() const {
        return impl_->accepted;
    }

    const std::shared_ptr<Buffer<real_t>> &
    TwoPhaseFlowTimeIntegration::trial() const {
        return impl_->trial;
    }

    real_t TwoPhaseFlowTimeIntegration::time() const { return impl_->time; }
    ptrdiff_t TwoPhaseFlowTimeIntegration::step() const { return impl_->step; }
}  // namespace sfem
