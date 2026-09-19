#include "sfem_ParallelPreconditioner.hpp"

#include "sfem_ElementScope.hpp"
#include "sfem_FunctionSpace.hpp"
#include "sfem_logger.hpp"
#include "smesh_exchange.hpp"
#include "smesh_mesh.hpp"

namespace sfem {

    std::shared_ptr<ParallelOperator<real_t>> create_parallel_preconditioner(
            const std::shared_ptr<Operator<real_t>> &preconditioner,
            const std::shared_ptr<FunctionSpace>    &space,
            const ExecutionSpace                     execution_space) {
        if (!preconditioner || !space) {
            return nullptr;
        }

        auto mesh = space->mesh_ptr();

        // One rank: no ghosts, no aura, nothing to gather. Hand back the inner operator's own
        // apply so the serial path is untouched rather than merely equivalent.
        if (mesh->comm()->size() == 1) {
            return make_parallel_op<real_t>(
                    mesh->comm(),
                    space->n_dofs(),
                    space->n_dofs(),
                    [preconditioner](const real_t *const x, real_t *const y) { preconditioner->apply(x, y); },
                    execution_space);
        }

        if (execution_space != EXECUTION_SPACE_HOST) {
            SFEM_ERROR(
                    "create_parallel_preconditioner: distributed DEVICE apply needs CUDA-aware\n"
                    "Exchange::gather on device pointers, as ParallelMatrixFreeOperator does.\n");
            return nullptr;
        }

        const int            block_size = space->block_size();
        const std::ptrdiff_t owned_dofs = space->n_owned_dofs();
        const std::ptrdiff_t local_dofs = space->n_dofs();

        // GhostsAndAura: a patch anchored on an aura element reads nodes beyond this rank's
        // ghosts, so a ghosts-only gather would leave exactly the slots this class exists to
        // fill.
        auto exchange = smesh::Exchange::create_nodal(mesh, smesh::Exchange::ExchangeScope::GhostsAndAura);
        assert_mesh_supports_distributed_element_scopes(*mesh);

        return make_parallel_op<real_t>(
                mesh->comm(),
                owned_dofs,
                owned_dofs,
                local_dofs,
                local_dofs,
                [preconditioner, exchange, block_size](const real_t *const r, real_t *const y) {
                    // In place, into the caller's buffer, as ParallelMatrixFreeOperator::apply
                    // does: the gather has nowhere else to put the ghost and aura entries, and
                    // the owned prefix it reads from is already correct.
                    exchange->gather(const_cast<real_t *>(r), block_size);
                    preconditioner->apply(r, y);
                },
                execution_space);
    }

}  // namespace sfem
