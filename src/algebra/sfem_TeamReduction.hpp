#ifndef SFEM_TEAM_REDUCTION_HPP
#define SFEM_TEAM_REDUCTION_HPP

#include <cassert>
#include <cstddef>

#include "sfem_Team.hpp"
#include "sfem_fixed_chunks.hpp"
#include "smesh_communicator.hpp"
#include "smesh_types.hpp"

namespace sfem {

    /// K sums over the entries of a parallel region's Team, deterministic: each thread sums its
    /// chunks, every thread then combines all the chunks in index order, so every thread holds the
    /// same K values, whatever the number of threads. The first K - 1 are added; the last is added
    /// too, or with `min_last` the minimum. Under MPI the leader completes them over the ranks and
    /// the others read them back.
    ///
    /// Each thread of the region takes its own TeamReduction over one buffer shared by them all
    /// (buffer_size(capacity) entries for teams of at most `capacity` chunks, outliving the
    /// region). Consecutive sums alternate between the buffer's
    /// halves: a thread still combining one sum is never overwritten by a thread already filling
    /// the next, and the barrier of that next sum keeps the one after it off the first half until
    /// every thread has read it.
    template <typename T, int K>
    class TeamReduction {
    public:
        static_assert(K > 0);
        /// A half: the chunks' partial sums, K per chunk, then the K values over the ranks.
        static constexpr ptrdiff_t half_size(const ptrdiff_t capacity) { return K * capacity + K; }
        static constexpr ptrdiff_t buffer_size(const ptrdiff_t capacity) { return 2 * half_size(capacity); }

        TeamReduction(T* const buffer, const ptrdiff_t capacity, const smesh::Communicator* const comm = nullptr)
            : buffer_(buffer), capacity_(capacity), comm_(comm && comm->size() > 1 ? comm : nullptr) {
            assert(buffer_);
        }

        /// pass(b, e, v) adds the entries [b, e) into v, from init; out the K values.
        template <typename F>
        void sum(const Team& team, const T (&init)[K], const bool min_last, F pass, T (&out)[K]) {
            assert(team.chunks <= capacity_);
            T* const part = buffer_ + flip_ * half_size(capacity_);
            for (ptrdiff_t c = team.first_chunk; c < team.end_chunk; c++) {
                ptrdiff_t b, e;
                team.chunk_entries(c, &b, &e);
                T v[K];
                for (int k = 0; k < K; k++) v[k] = init[k];
                pass(b, e, v);
                for (int k = 0; k < K; k++) part[K * c + k] = v[k];
            }
            team.barrier();
            combine(team.chunks, part, init, min_last, out);
            if (comm_) {
                T* const ranks = part + K * capacity_;
                if (team.leader()) {
                    if (min_last) {
                        comm_->sum_and_min_last(out, K, smesh::TypeToEnum<T>::value());
                    } else {
                        comm_->sum(out, K, smesh::TypeToEnum<T>::value());
                    }
                    for (int k = 0; k < K; k++) ranks[k] = out[k];
                }
                team.barrier();
                for (int k = 0; k < K; k++) out[k] = ranks[k];
            }
            flip_ ^= 1;
        }

        /// The chunks' sums in `part` combined in index order from `init`.
        static void combine(const ptrdiff_t chunks, const T* const part, const T (&init)[K], const bool min_last, T (&out)[K]) {
            for (int k = 0; k < K; k++) out[k] = init[k];
            for (ptrdiff_t c = 0; c < chunks; c++) {
                for (int k = 0; k < K - 1; k++) out[k] += part[K * c + k];
                const T last = part[K * c + K - 1];
                out[K - 1]   = min_last ? (last < out[K - 1] ? last : out[K - 1]) : out[K - 1] + last;
            }
        }

    private:
        T* const                         buffer_;
        const ptrdiff_t                  capacity_;
        const smesh::Communicator* const comm_;
        int                              flip_{0};
    };

}  // namespace sfem

#endif  // SFEM_TEAM_REDUCTION_HPP
