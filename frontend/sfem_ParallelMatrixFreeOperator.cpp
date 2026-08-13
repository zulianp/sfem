#include "sfem_ParallelMatrixFreeOperator.hpp"

#include "sfem_ElementScope.hpp"
#include "sfem_Function.hpp"
#include "sfem_FunctionSpace.hpp"
#include "sfem_logger.hpp"
#include "smesh_env.hpp"
#include "smesh_exchange.hpp"
#include "smesh_mesh.hpp"

#include <algorithm>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

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
        if (execution_space != EXECUTION_SPACE_HOST) {
            SFEM_ERROR(
                    "ParallelMatrixFreeOperator: EXECUTION_SPACE_DEVICE not implemented yet.\n"
                    "Need CUDA-aware Exchange::gather on device pointers (device pack/unpack +\n"
                    "MPI on device buffers), then mirror the host ElementScope overlap path.\n"
                    "Do not stage via host D2H/H2D; x/y are already device allocation-sized.\n");
        }

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

        assert_mesh_supports_distributed_element_scopes(*mesh);

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

        real_t *const       x_mut       = const_cast<real_t *>(x);
        const real_t *const state_local = impl_->state ? impl_->state->data() : nullptr;
        auto                mesh        = impl_->function->space()->mesh_ptr();

        // SFEM_PARALLEL_MF_LEGACY=1: original blocking gather + full ALL apply (no ElementScope overlap).
        const bool legacy = smesh::Env::read<int>("SFEM_PARALLEL_MF_LEGACY", 0) != 0;

        // -------------------------------------------------------------------------
        // DEVICE (not implemented): x/y are already on device (allocation-sized).
        // Intended path once Exchange supports CUDA-aware MPI on device pointers:
        //   1. d_memset(y, 0, local_dofs); optionally zero ghost/aura slots of x on device
        //   2. Launch apply(OWNED_NOT_SHARED) async on the same device stream as the Op
        //   3. exchange->gather(x) / gather_begin+wait with device pack + MPI on device bufs
        //      (no host staging, no per-call alloc, no full-vector D2H/H2D)
        //   4. Device sync as needed so ghosts are ready for SHARED_AND_AURA
        //   5. apply(SHARED_AND_AURA); copy_constrained_dofs on device constraints
        // Prerequisites: device pack/unpack in Exchange; CUDA-aware MPI build.
        // -------------------------------------------------------------------------

        std::fill(x_mut + impl_->owned_dofs, x_mut + impl_->local_dofs, real_t(0));
        std::fill(y, y + impl_->local_dofs, real_t(0));

        if (legacy) {
            if (impl_->exchange->gather(x_mut, impl_->block_size) != SFEM_SUCCESS) {
                return SFEM_FAILURE;
            }
            return impl_->function->apply(state_local, x_mut, y);
        }

        if (impl_->exchange->gather_begin(x_mut, impl_->block_size) != SFEM_SUCCESS) {
            return SFEM_FAILURE;
        }

#ifdef _OPENMP
        if (omp_get_max_threads() > 1) {
            int err = SFEM_SUCCESS;
#pragma omp parallel reduction(| : err)
            {
                const int tid = omp_get_thread_num();
                const int nt  = omp_get_num_threads();
                if (tid == 0) {
                    if (impl_->exchange->gather_wait() != SFEM_SUCCESS) {
                        err = SFEM_FAILURE;
                    }
                } else {
                    const int  n_workers = nt - 1;
                    const auto flat_chunk =
                            static_chunk(count_block_elements(*mesh, ElementScope::OWNED_NOT_SHARED), tid - 1, n_workers);
                    if (impl_->function->apply_scope_flat_range(
                                state_local,
                                x_mut,
                                y,
                                ElementScope::OWNED_NOT_SHARED,
                                flat_chunk.begin,
                                flat_chunk.end) != SFEM_SUCCESS) {
                        err = SFEM_FAILURE;
                    }
                }
            }
            if (err != SFEM_SUCCESS) {
                return SFEM_FAILURE;
            }
        } else
#endif
        {
            if (impl_->function->apply(state_local, x_mut, y, ElementScope::OWNED_NOT_SHARED) != SFEM_SUCCESS) {
                return SFEM_FAILURE;
            }
            if (impl_->exchange->gather_wait() != SFEM_SUCCESS) {
                return SFEM_FAILURE;
            }
        }

        if (impl_->function->apply(state_local, x_mut, y, ElementScope::SHARED_AND_AURA) != SFEM_SUCCESS) {
            return SFEM_FAILURE;
        }
        // Match Function::apply(ALL): Dirichlet identity on constrained DOFs.
        return impl_->function->copy_constrained_dofs(x_mut, y);
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
            SFEM_ERROR(
                    "create_parallel_matrix_free_operator: distributed DEVICE apply needs CUDA-aware\n"
                    "Exchange::gather on device pointers (see ParallelMatrixFreeOperator::apply comments).\n");
            return nullptr;
        }

        return std::make_shared<ParallelMatrixFreeOperator>(function, state, execution_space);
    }

}  // namespace sfem
