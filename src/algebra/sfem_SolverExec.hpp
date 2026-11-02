#ifndef SFEM_SOLVER_EXEC_HPP
#define SFEM_SOLVER_EXEC_HPP

#include <cstddef>
#include <vector>

#include "sfem_Operator.hpp"
#include "sfem_Team.hpp"
#include "sfem_TeamBLAS.hpp"
#include "sfem_TeamOperator.hpp"
#include "sfem_TeamReduction.hpp"
#include "sfem_tpl_blas.hpp"

// Where a Krylov iteration's vector work runs, so that each solver is written once.
//
//   TeamExec  the threads of one parallel region, the solver's whole run: each thread updates the
//             entries of its Team share, the sums are the team's (TeamReduction), the operator
//             is applied by them all (TeamOperator::team_apply). Every thread holds the same
//             scalars and so takes the same decisions; what is said or kept is the leader's.
//   BlasExec  whole vectors through the execution space's BLAS, one call each: the device, and a
//             host operator that has no team form.
//
// An iteration calls apply(op, in, out) for every product: the output is zeroed first unless the
// operator overwrites it, and the input is complete when the product reads it.

namespace sfem {

    template <typename T>
    class BlasExec {
    public:
        /// n entries; an output zeroed before a product spans `allocation` (under MPI the ghosts too).
        BlasExec(BLAS<T> &blas, const ptrdiff_t n, const ptrdiff_t allocation = -1)
            : blas_(blas), n_(n), allocation_(allocation > n ? allocation : n) {}

        T    dot(const T *const a, const T *const b) { return blas_.dot(n_, a, b); }
        void axpby(const T alpha, const T *const x, const T beta, T *const y) { blas_.axpby(n_, alpha, x, beta, y); }
        void copy(const T *const src, T *const dest) { blas_.copy(n_, src, dest); }
        void zeros(T *const y) { blas_.zeros(allocation_, y); }
        /// y <- alpha x
        void scale_into(const T alpha, const T *const x, T *const y) { blas_.zaxpby(n_, alpha, x, T(0), x, y); }
        BLAS<T> &blas() { return blas_; }
        ptrdiff_t size() const { return n_; }

        int apply(Operator<T> &op, const T *const in, T *const out, const bool overwrites) {
            if (!overwrites) zeros(out);
            return op.apply(in, out);
        }

        void barrier() const {}
        bool leader() const { return true; }

    private:
        BLAS<T>        &blas_;
        const ptrdiff_t n_;
        const ptrdiff_t allocation_;
    };

    template <typename T>
    class TeamExec {
    public:
        TeamExec(const Team &team, TeamReduction<T, 1> &sums) : team_(team), sums_(sums) {}

        T    dot(const T *const a, const T *const b) { return team_blas::dot(team_, sums_, a, b); }
        void axpby(const T alpha, const T *const x, const T beta, T *const y) {
            team_blas::axpby(team_.begin, team_.end, alpha, x, beta, y);
        }
        void copy(const T *const src, T *const dest) { team_blas::copy(team_.begin, team_.end, src, dest); }
        void zeros(T *const y) { team_blas::zeros(team_.begin, team_.end, y); }
        /// y <- alpha x
        void scale_into(const T alpha, const T *const x, T *const y) {
            for (ptrdiff_t i = team_.begin; i < team_.end; i++) y[i] = alpha * x[i];
        }

        /// The input's entries are the threads' own until the product reads them: one barrier, after
        /// this thread's share of the output is zeroed, covers both -- none where the operator reads
        /// beyond the thread's share only after a barrier of its own and nothing was zeroed.
        int apply(TeamOperator<T> &op, const T *const in, T *const out, const bool overwrites) {
            if (!overwrites) zeros(out);
            if (!overwrites || !op.input_in_share()) team_.barrier();
            op.team_apply(team_, in, out);
            return SFEM_SUCCESS;
        }

        /// One pass over the thread's entries, f(i) each, and the team's sum of what it returns.
        template <typename F>
        T sweep_sum(F f) {
            T out[1];
            sums_.sum(team_, {T(0)}, false, [&](const ptrdiff_t b, const ptrdiff_t e, T (&s)[1]) {
                T acc = s[0];
                for (ptrdiff_t i = b; i < e; i++) acc += f(i);
                s[0] = acc;
            }, out);
            return out[0];
        }

        void barrier() const { team_.barrier(); }
        bool leader() const { return team_.leader(); }
        const Team &team() const { return team_; }

    private:
        const Team          &team_;
        TeamReduction<T, 1> &sums_;
    };

    /// The reduction buffer of a solver's region over n entries, grown as needed.
    template <typename T>
    T *team_reduction_buffer(std::vector<T> &buffer, const ptrdiff_t n) {
        const ptrdiff_t size = TeamReduction<T, 1>::buffer_size(team_chunk_count(n));
        if ((ptrdiff_t)buffer.size() < size) buffer.resize((size_t)size);
        return buffer.data();
    }

}  // namespace sfem

#endif  // SFEM_SOLVER_EXEC_HPP
