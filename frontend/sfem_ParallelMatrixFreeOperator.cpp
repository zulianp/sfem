#include "sfem_ParallelMatrixFreeOperator.hpp"

#include "sfem_FunctionSpace.hpp"
#include "sfem_logger.hpp"
#include "smesh_exchange.hpp"
#include "smesh_mesh.hpp"

#include <algorithm>
#include <cstring>

namespace sfem {

    class ParallelMatrixFreeOperator::Impl {
    public:
        std::shared_ptr<Function>        function;
        std::shared_ptr<Buffer<real_t>>  state;
        std::shared_ptr<smesh::Exchange> exchange;
        ptrdiff_t                        owned_dofs{0};
        ptrdiff_t                        local_dofs{0};
        int                              block_size{1};
        ExecutionSpace                   execution_space{EXECUTION_SPACE_INVALID};
    };

    ParallelMatrixFreeOperator::ParallelMatrixFreeOperator(const std::shared_ptr<Function>       &function,
                                                           const std::shared_ptr<Buffer<real_t>> &state,
                                                           const ExecutionSpace                   execution_space)
        : impl_(std::make_unique<Impl>()) {
        impl_->function        = function;
        impl_->state           = state;
        impl_->execution_space = execution_space;

        auto space = function->space();
        auto mesh  = space->mesh_ptr();
        auto dist  = mesh->distributed();

        impl_->block_size = space->block_size();
        impl_->owned_dofs = dist->n_nodes_owned() * impl_->block_size;
        impl_->local_dofs = dist->n_nodes_local() * impl_->block_size;
        impl_->exchange   = smesh::Exchange::create_nodal(mesh, smesh::Exchange::ExchangeScope::GhostsAndAura);

        if (state) {
            if (state->size() != static_cast<size_t>(impl_->local_dofs)) {
                SFEM_ERROR(
                        "ParallelMatrixFreeOperator state size %ld != col_allocation_size %ld "
                        "(owned+ghosts+aura); allocate with col_allocation_size()\n",
                        (long)state->size(),
                        (long)impl_->local_dofs);
            }
            update_state();
        }
    }

    ParallelMatrixFreeOperator::~ParallelMatrixFreeOperator() = default;

    int ParallelMatrixFreeOperator::update_state() {
        SFEM_TRACE_SCOPE("ParallelMatrixFreeOperator::update_state");

        if (!impl_->state) {
            return SFEM_SUCCESS;
        }

        if (impl_->state->size() != static_cast<size_t>(impl_->local_dofs)) {
            SFEM_ERROR("ParallelMatrixFreeOperator::update_state: state size %ld != col_allocation_size %ld\n",
                       (long)impl_->state->size(),
                       (long)impl_->local_dofs);
            return SFEM_FAILURE;
        }

        return impl_->exchange->gather(impl_->state->data(), impl_->block_size);
    }

    int ParallelMatrixFreeOperator::apply(const real_t *const x, real_t *const y) {
        SFEM_TRACE_SCOPE("ParallelMatrixFreeOperator::apply");

        if (impl_->execution_space != EXECUTION_SPACE_HOST) {
            SFEM_ERROR("ParallelMatrixFreeOperator supports host execution only\n");
            return SFEM_FAILURE;
        }

        // Ghost/aura slots are written by gather; owned values are preserved.
        real_t *const x_mut = const_cast<real_t *>(x);
        std::fill(x_mut + impl_->owned_dofs, x_mut + impl_->local_dofs, real_t(0));
        if (impl_->exchange->gather(x_mut, impl_->block_size) != SFEM_SUCCESS) {
            return SFEM_FAILURE;
        }

        const real_t *const state_local = impl_->state ? impl_->state->data() : nullptr;

        std::fill(y, y + impl_->local_dofs, real_t(0));  // TODO: should not be necessary, since apply semantics is based on Add
        return impl_->function->apply(state_local, x_mut, y);
    }

    std::ptrdiff_t ParallelMatrixFreeOperator::rows() const { return impl_->owned_dofs; }

    std::ptrdiff_t ParallelMatrixFreeOperator::cols() const { return impl_->owned_dofs; }

    ExecutionSpace ParallelMatrixFreeOperator::execution_space() const { return impl_->execution_space; }

    std::shared_ptr<Communicator> ParallelMatrixFreeOperator::comm() const {
        return impl_->function->space()->mesh_ptr()->comm();
    }

    std::ptrdiff_t ParallelMatrixFreeOperator::row_allocation_size() const { return impl_->local_dofs; }

    std::ptrdiff_t ParallelMatrixFreeOperator::col_allocation_size() const { return impl_->local_dofs; }

    std::shared_ptr<ParallelOperator<real_t>> create_parallel_matrix_free_operator(const std::shared_ptr<Function> &function,
                                                                                   const std::shared_ptr<Buffer<real_t>> &state,
                                                                                   const ExecutionSpace execution_space) {
        auto space = function->space();
        auto mesh  = space->mesh_ptr();

        if (mesh->comm()->size() == 1) {
            return make_parallel_op<real_t>(
                    mesh->comm(),
                    space->n_dofs(),
                    space->n_dofs(),
                    [=](const real_t *const x, real_t *const y) { function->apply(state ? state->data() : nullptr, x, y); },
                    execution_space);
        }

        if (execution_space != EXECUTION_SPACE_HOST) {
            SFEM_ERROR("create_parallel_matrix_free_operator supports distributed host execution only\n");
            return nullptr;
        }

        return std::make_shared<ParallelMatrixFreeOperator>(function, state, execution_space);
    }

}  // namespace sfem
