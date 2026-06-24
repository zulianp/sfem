#include "sfem_TwoPhaseFlowTimeIntegration.hpp"

#include "sfem_DirichletConditions.hpp"
#include "sfem_FunctionSpace.hpp"
#include "sfem_base.hpp"
#include "smesh_glob.hpp"
#include "smesh_mesh.hpp"
#include "smesh_path.hpp"
#include "smesh_types.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>

namespace sfem {
    class TwoPhaseFlowTimeIntegration::Impl {
    public:
        Impl(const std::shared_ptr<smesh::Mesh> &mesh,
             const std::shared_ptr<Buffer<real_t>> &initial_state,
             const std::shared_ptr<DirichletConditions> &dirichlet,
             BoundaryUpdate boundary_update)
            : mesh(mesh),
              initial_state(initial_state),
              dirichlet(dirichlet),
              boundary_update(std::move(boundary_update)) {}

        std::shared_ptr<smesh::Mesh> mesh;
        std::shared_ptr<Buffer<real_t>> initial_state;
        std::shared_ptr<Buffer<real_t>> accepted;
        std::shared_ptr<Buffer<real_t>> trial;
        std::shared_ptr<DirichletConditions> dirichlet;
        BoundaryUpdate boundary_update;
        real_t time{0};
        ptrdiff_t step{0};
    };

    TwoPhaseFlowTimeIntegration::TwoPhaseFlowTimeIntegration(
            const std::shared_ptr<smesh::Mesh> &mesh,
            const std::shared_ptr<Buffer<real_t>> &initial_state,
            const std::shared_ptr<DirichletConditions> &dirichlet,
            BoundaryUpdate boundary_update)
        : impl_(std::make_unique<Impl>(
                  mesh, initial_state, dirichlet, std::move(boundary_update))) {}

    TwoPhaseFlowTimeIntegration::~TwoPhaseFlowTimeIntegration() = default;

    int TwoPhaseFlowTimeIntegration::initialize() {
        const ptrdiff_t ndofs = 2 * impl_->mesh->n_nodes();
        if (!impl_->initial_state || impl_->initial_state->size() != ndofs ||
            !impl_->dirichlet) {
            return SFEM_FAILURE;
        }
        impl_->accepted = create_host_buffer<real_t>(ndofs);
        impl_->trial = create_host_buffer<real_t>(ndofs);
        std::copy(impl_->initial_state->data(),
                  impl_->initial_state->data() + ndofs,
                  impl_->accepted->data());
        impl_->time = 0;
        impl_->step = 0;
        apply_boundary(0, impl_->accepted->data());
        std::copy(impl_->accepted->data(),
                  impl_->accepted->data() + ndofs,
                  impl_->trial->data());
        return SFEM_SUCCESS;
    }

    void TwoPhaseFlowTimeIntegration::apply_boundary(
            const real_t time,
            real_t *const state) const {
        if (impl_->boundary_update) {
            impl_->boundary_update(time, *impl_->dirichlet);
        }
        impl_->dirichlet->apply(state);
    }

    void TwoPhaseFlowTimeIntegration::constrain_residual(
            const real_t *const state,
            real_t *const residual) const {
        impl_->dirichlet->gradient(state, residual);
    }

    void TwoPhaseFlowTimeIntegration::constrain_direction(
            real_t *const direction) const {
        impl_->dirichlet->apply_value(0, direction);
    }

    void TwoPhaseFlowTimeIntegration::constrain_linear(
            const real_t *const direction,
            real_t *const output) const {
        impl_->dirichlet->copy_constrained_dofs(direction, output);
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
