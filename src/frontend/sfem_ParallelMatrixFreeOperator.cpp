#include "sfem_ParallelMatrixFreeOperator.hpp"

#include "sfem_ElementScope.hpp"
#include "sfem_Function.hpp"
#include "sfem_FunctionSpace.hpp"
#include "sfem_logger.hpp"
#include "smesh_exchange.hpp"
#include "smesh_mesh.hpp"

#include <algorithm>


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
        auto mesh  = space->mesh();
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

        // A function without a team form (create_parallel_matrix_free_operator takes the team
        // one): the elements away from the interface while the ghost and aura values travel,
        // then the others.
        std::fill(x_mut + impl_->owned_dofs, x_mut + impl_->local_dofs, real_t(0));
        std::fill(y, y + impl_->local_dofs, real_t(0));

        if (impl_->exchange->gather_begin(x_mut, impl_->block_size) != SFEM_SUCCESS) {
            return SFEM_FAILURE;
        }
        if (impl_->function->apply(state_local, x_mut, y, ElementScope::OWNED_NOT_SHARED) != SFEM_SUCCESS) {
            return SFEM_FAILURE;
        }
        if (impl_->exchange->gather_wait() != SFEM_SUCCESS) {
            return SFEM_FAILURE;
        }

        if (impl_->function->apply(state_local, x_mut, y, ElementScope::SHARED_AND_AURA) != SFEM_SUCCESS) {
            return SFEM_FAILURE;
        }
        // Match Function::apply(ALL): Dirichlet identity on constrained DOFs.
        return impl_->function->copy_constrained_dofs(x_mut, y);
    }

    std::ptrdiff_t ParallelMatrixFreeOperator::rows() const { return impl_->owned_dofs; }

    std::ptrdiff_t ParallelMatrixFreeOperator::cols() const { return impl_->owned_dofs; }

    ExecutionContext ParallelMatrixFreeOperator::execution_context() const { return impl_->execution_space; }

    std::shared_ptr<Communicator> ParallelMatrixFreeOperator::comm() const {
        return impl_->function->space()->mesh()->comm();
    }

    std::ptrdiff_t ParallelMatrixFreeOperator::row_allocation_size() const { return impl_->local_dofs; }

    std::ptrdiff_t ParallelMatrixFreeOperator::col_allocation_size() const { return impl_->local_dofs; }

    std::shared_ptr<ParallelOperator<real_t>> create_distributed_team_operator(const std::shared_ptr<Function> &function,
                                                                               ScopedTeamApply                  apply,
                                                                               std::function<int()>             prepare) {
        auto space = function->space();
        auto mesh  = space->mesh();
        auto dist  = mesh->distributed();
        assert_mesh_supports_distributed_element_scopes(*mesh);

        const int       bs    = space->block_size();
        const ptrdiff_t owned = dist->n_nodes_owned() * bs;
        const ptrdiff_t local = dist->n_nodes_local() * bs;
        return make_team_op<real_t>(
                mesh->comm(),
                owned,
                owned,
                local,
                local,
                false,
                [=](const Team &team, const real_t *const x, real_t *const y) {
                    real_t *const x_mut = const_cast<real_t *>(x);
                    if (team.tid == 0) {
                        // The rows beyond the owned ones are scratch: the scatter adds into them.
                        std::fill(y + owned, y + local, real_t(0));
                        if (space->synchronize_begin(x_mut, EXECUTION_SPACE_HOST) != SFEM_SUCCESS) {
                            SFEM_ERROR("distributed team operator: the gather of x did not start\n");
                        }
                    }
                    team.barrier();
                    apply(team, x, y, ElementScope::OWNED_NOT_SHARED);
                    if (team.tid == 0 && space->synchronize_end(EXECUTION_SPACE_HOST) != SFEM_SUCCESS) {
                        SFEM_ERROR("distributed team operator: the gather of x did not complete\n");
                    }
                    team.barrier();
                    apply(team, x, y, ElementScope::SHARED_AND_AURA);
                    team.barrier();
                    function->team_copy_constrained_dofs(team, x, y);
                },
                std::move(prepare));
    }


    std::shared_ptr<ParallelOperator<real_t>> create_parallel_matrix_free_operator(const std::shared_ptr<Function> &function,
                                                                                   const std::shared_ptr<Buffer<real_t>> &state,
                                                                                   const ExecutionSpace execution_space) {
        auto space = function->space();
        auto mesh  = space->mesh();

        if (mesh->comm()->size() > 1 && execution_space == EXECUTION_SPACE_HOST && function->team_supported()) {
            if (state && state->size() != static_cast<size_t>(mesh->distributed()->n_nodes_local() * space->block_size())) {
                SFEM_ERROR("create_parallel_matrix_free_operator: state size %ld != col_allocation_size (owned+ghosts+aura)\n",
                           (long)state->size());
                return nullptr;
            }
            // The state's ghost and aura values, gathered before each solve's region.
            return create_distributed_team_operator(
                    function,
                    [function, state](const Team &team, const real_t *const x, real_t *const y, const ElementScope scope) {
                        function->team_apply(team, state ? state->data() : nullptr, x, y, scope);
                    },
                    [function, state, space] {
                        const int err = function->team_prepare();
                        if (err != SFEM_SUCCESS || !state) return err;
                        return space->synchronize(state->data(), EXECUTION_SPACE_HOST);
                    });
        }

        if (mesh->comm()->size() == 1 && execution_space == EXECUTION_SPACE_HOST && function->team_supported()) {
            const ptrdiff_t n = space->n_dofs();
            return make_team_op<real_t>(
                    mesh->comm(),
                    n,
                    n,
                    n,
                    n,
                    false,
                    [=](const Team &team, const real_t *const x, real_t *const y) {
                        function->team_apply(team, state ? state->data() : nullptr, x, y);
                    },
                    [=] { return function->team_prepare(); });
        }

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
