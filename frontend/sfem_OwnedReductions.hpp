#ifndef SFEM_OWNED_REDUCTIONS_HPP
#define SFEM_OWNED_REDUCTIONS_HPP

#include <memory>

#include "sfem_API.hpp"
#include "sfem_FunctionSpace.hpp"

namespace sfem {

    /// Reductions of vectors laid out as a space's dofs: the owned prefix, then ghosts and aura.
    /// Only the owned entries count, each once over the ranks; ghost and aura entries are copies or
    /// partial sums of other ranks' rows. On one rank the plain BLAS reductions.
    class OwnedReductions {
    public:
        OwnedReductions(const std::shared_ptr<FunctionSpace> &space, const ExecutionSpace es)
            : comm_(space->mesh_ptr()->comm()), blas_(blas<real_t>(es)), n_owned_(space->n_owned_dofs()) {}

        real_t dot(const real_t *const x, const real_t *const y) const { return sum(blas_->dot(n_owned_, x, y)); }
        real_t norm2(const real_t *const x) const { return std::sqrt(dot(x, x)); }

        /// Of one value per rank.
        real_t sum(const real_t local) const { return comm_->size() > 1 ? comm_->sum(local) : local; }
        real_t max(const real_t local) const { return comm_->size() > 1 ? comm_->max(local) : local; }
        real_t min(const real_t local) const { return comm_->size() > 1 ? -comm_->max(-local) : local; }

        ptrdiff_t                            n_owned() const { return n_owned_; }
        const std::shared_ptr<Communicator> &comm() const { return comm_; }

    private:
        std::shared_ptr<Communicator> comm_;
        std::shared_ptr<BLAS<real_t>> blas_;
        ptrdiff_t                     n_owned_;
    };

}  // namespace sfem

#endif  // SFEM_OWNED_REDUCTIONS_HPP
