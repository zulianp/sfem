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
        SharedBuffer<real_t>             input_scratch;
        SharedBuffer<real_t>             output_scratch;
        SharedBuffer<real_t>             state_scratch;
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

        impl_->input_scratch  = create_host_buffer<real_t>(impl_->local_dofs);
        impl_->output_scratch = create_host_buffer<real_t>(impl_->local_dofs);
        if (state) {
            impl_->state_scratch = create_host_buffer<real_t>(impl_->local_dofs);
            update_state();
        }
    }

    ParallelMatrixFreeOperator::~ParallelMatrixFreeOperator() = default;

    int ParallelMatrixFreeOperator::update_state() {
        SFEM_TRACE_SCOPE("ParallelMatrixFreeOperator::update_state");

        if (!impl_->state) {
            return SFEM_SUCCESS;
        }

        real_t *const state_scratch = impl_->state_scratch->data();
        if (impl_->state->size() == static_cast<size_t>(impl_->local_dofs)) {
            std::memcpy(state_scratch, impl_->state->data(), sizeof(real_t) * impl_->local_dofs);
        } else if (impl_->state->size() == static_cast<size_t>(impl_->owned_dofs)) {
            std::memcpy(state_scratch, impl_->state->data(), sizeof(real_t) * impl_->owned_dofs);
            std::fill(state_scratch + impl_->owned_dofs, state_scratch + impl_->local_dofs, real_t(0));
        } else {
            SFEM_ERROR("ParallelMatrixFreeOperator state has invalid size %ld, expected %ld or %ld\n",
                       (long)impl_->state->size(),
                       (long)impl_->owned_dofs,
                       (long)impl_->local_dofs);
            return SFEM_FAILURE;
        }

        return impl_->exchange->gather(state_scratch, impl_->block_size);
    }

    int ParallelMatrixFreeOperator::apply(const real_t *const x, real_t *const y) {
        SFEM_TRACE_SCOPE("ParallelMatrixFreeOperator::apply");

        if (impl_->execution_space != EXECUTION_SPACE_HOST) {
            SFEM_ERROR("ParallelMatrixFreeOperator supports host execution only\n");
            return SFEM_FAILURE;
        }

        real_t *const h_local = impl_->input_scratch->data();
        real_t *const y_local = impl_->output_scratch->data();

        std::memcpy(h_local, x, sizeof(real_t) * impl_->owned_dofs);
        std::fill(h_local + impl_->owned_dofs, h_local + impl_->local_dofs, real_t(0));

        if (impl_->exchange->gather(h_local, impl_->block_size) != SFEM_SUCCESS) {
            return SFEM_FAILURE;
        }

        const real_t *const state_local = impl_->state_scratch ? impl_->state_scratch->data() : nullptr;

        std::fill(y_local, y_local + impl_->local_dofs, real_t(0));
        if (impl_->function->apply(state_local, h_local, y_local) != SFEM_SUCCESS) {
            return SFEM_FAILURE;
        }

        std::memcpy(y, y_local, sizeof(real_t) * impl_->owned_dofs);
        return SFEM_SUCCESS;
    }

    std::ptrdiff_t ParallelMatrixFreeOperator::rows() const { return impl_->owned_dofs; }

    std::ptrdiff_t ParallelMatrixFreeOperator::cols() const { return impl_->owned_dofs; }

    ExecutionSpace ParallelMatrixFreeOperator::execution_space() const { return impl_->execution_space; }

    std::shared_ptr<Communicator> ParallelMatrixFreeOperator::comm() const {
        return impl_->function->space()->mesh_ptr()->comm();
    }

    std::ptrdiff_t ParallelMatrixFreeOperator::row_allocation_size() const { return impl_->local_dofs; }

    std::ptrdiff_t ParallelMatrixFreeOperator::col_allocation_size() const { return impl_->local_dofs; }

    std::shared_ptr<ParallelOperator<real_t>> create_parallel_matrix_free_operator(
            const std::shared_ptr<Function>       &function,
            const std::shared_ptr<Buffer<real_t>> &state,
            const ExecutionSpace                   execution_space) {
        auto space = function->space();
        auto mesh  = space->mesh_ptr();

        if (mesh->comm()->size() == 1) {
            return make_parallel_op<real_t>(
                    mesh->comm(),
                    space->n_dofs(),
                    space->n_dofs(),
                    [=](const real_t *const x, real_t *const y) {
                        function->apply(state ? state->data() : nullptr, x, y);
                    },
                    execution_space);
        }

        if (execution_space != EXECUTION_SPACE_HOST) {
            SFEM_ERROR("create_parallel_matrix_free_operator supports distributed host execution only\n");
            return nullptr;
        }

        return std::make_shared<ParallelMatrixFreeOperator>(function, state, execution_space);
    }

}  // namespace sfem
