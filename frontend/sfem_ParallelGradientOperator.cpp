#include "sfem_ParallelGradientOperator.hpp"

#include "sfem_ElementScope.hpp"
#include "sfem_Function.hpp"
#include "sfem_FunctionSpace.hpp"
#include "sfem_logger.hpp"
#include "smesh_exchange.hpp"
#include "smesh_mesh.hpp"

#include <algorithm>

namespace sfem {

    class ParallelGradientOperator::Impl {
    public:
        std::shared_ptr<Function>        function;
        std::shared_ptr<smesh::Exchange> exchange;
        std::ptrdiff_t                   owned_dofs{0};
        std::ptrdiff_t                   local_dofs{0};
        int                              block_size{1};
        ExecutionSpace                   execution_space{EXECUTION_SPACE_INVALID};
        bool                             distributed{false};
    };

    ParallelGradientOperator::ParallelGradientOperator(const std::shared_ptr<Function> &function,
                                                       const ExecutionSpace             execution_space)
        : impl_(std::make_unique<Impl>()) {
        if (execution_space != EXECUTION_SPACE_HOST) {
            SFEM_ERROR(
                    "ParallelGradientOperator: EXECUTION_SPACE_DEVICE not implemented yet.\n"
                    "Needs CUDA-aware Exchange::gather on device pointers, as\n"
                    "ParallelMatrixFreeOperator does.\n");
        }

        impl_->function        = function;
        impl_->execution_space = execution_space;

        auto space = function->space();
        auto mesh  = space->mesh_ptr();

        impl_->block_size = space->block_size();
        impl_->owned_dofs = space->n_owned_dofs();
        impl_->local_dofs = space->n_dofs();

        // The space already carries the owned/local split (FunctionSpace::initialize_dof_counts
        // derives both from the mesh's Distributed metadata, and collapses them on a serial
        // mesh), so the counts are taken from there rather than recomputed from the mesh.
        impl_->distributed = mesh_is_distributed(*mesh);

        if (impl_->distributed) {
            // GhostsAndAura, not GhostsOnly. Aura elements are evaluated below, and they read
            // nodes beyond this rank's ghosts; a ghosts-only gather would leave those unfilled
            // and reintroduce exactly the error this class exists to remove.
            impl_->exchange = smesh::Exchange::create_nodal(mesh, smesh::Exchange::ExchangeScope::GhostsAndAura);
            assert_mesh_supports_distributed_element_scopes(*mesh);
        }
    }

    ParallelGradientOperator::~ParallelGradientOperator() = default;

    int ParallelGradientOperator::gradient(real_t *const x, real_t *const out) {
        SFEM_TRACE_SCOPE("ParallelGradientOperator::gradient");

        if (!impl_->distributed) {
            // One rank: no ghosts, no aura, nothing to gather. Zeroing and evaluating exactly
            // as the caller would have done alone, so serial results are unchanged bit for bit.
            std::fill(out, out + impl_->local_dofs, real_t(0));
            return impl_->function->gradient(x, out, ElementScope::ALL);
        }

        // Clear the ghost and aura slots of the INPUT before gathering them.
        //
        // The gather fills the slots it owns an entry for; clearing first means anything it
        // does not cover reads as zero rather than as whatever the previous evaluation left
        // there. ParallelMatrixFreeOperator::apply does the same for the same reason.
        std::fill(x + impl_->owned_dofs, x + impl_->local_dofs, real_t(0));

        // The OUTPUT is zeroed across the LOCAL range, not the owned one: the aura elements
        // evaluated below deposit partial sums into ghost rows, and those rows have to start
        // from zero even though nobody reads them.
        std::fill(out, out + impl_->local_dofs, real_t(0));

        if (impl_->exchange->gather(x, impl_->block_size) != SFEM_SUCCESS) {
            return SFEM_FAILURE;
        }

        // ElementScope::ALL: every local element, aura included. That is what makes the owned
        // rows complete without a scatter, and it also means Function::gradient applies
        // constraints_gradient itself -- the scoped paths skip that, this one does not.
        return impl_->function->gradient(x, out, ElementScope::ALL);
    }

    std::shared_ptr<Communicator> ParallelGradientOperator::comm() const {
        return impl_->function->space()->mesh_ptr()->comm();
    }

    std::ptrdiff_t ParallelGradientOperator::owned_dofs() const { return impl_->owned_dofs; }
    std::ptrdiff_t ParallelGradientOperator::row_allocation_size() const { return impl_->local_dofs; }
    std::ptrdiff_t ParallelGradientOperator::col_allocation_size() const { return impl_->local_dofs; }

    std::shared_ptr<ParallelGradientOperator> create_parallel_gradient_operator(
            const std::shared_ptr<Function> &function,
            const ExecutionSpace             execution_space) {
        return std::make_shared<ParallelGradientOperator>(function, execution_space);
    }

}  // namespace sfem
