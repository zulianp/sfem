#ifndef SFEM_OPENMP_BLAS_HPP
#define SFEM_OPENMP_BLAS_HPP

#include "sfem_Team.hpp"
#include "sfem_TeamBLAS.hpp"
#include "sfem_TeamReduction.hpp"
#include "sfem_fixed_chunks.hpp"
#include "sfem_tpl_blas.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#else
#ifdef SFEM_ENABLE_OPENMP
#error "_OPENMP is undefined!"
#endif
#endif

namespace sfem {

    // The host BLAS: each operation one region, every thread the static share of the entries a
    // `parallel for` would give it, through the functions a solver's region calls on its Team
    // shares (sfem_TeamBLAS.hpp).
    //
    // Dot products are deterministic: partial sums over chunks fixed by the length
    // (team_chunk_count), combined in index order (TeamReduction), so the result is identical
    // for any number of threads, and the same a solver's region computes. OpenMP's reduction
    // clause combines in an unspecified order over chunks that follow the thread count; in the
    // CVFEM spike that alone left a bit-deterministic operator's Krylov solve varying by about
    // twenty percent in iterations, and the fixed chunks were also faster (8509 to 7672
    // microseconds an iteration). Below blas_serial_below entries the chunks are summed by the
    // calling thread: a region costs more than the arithmetic.
    template <typename T>
    class OpenMPBLAS final : public BLAS<T> {
    public:
        OpenMPBLAS() : BLAS<T>(ExecutionContext::host()) {}

        static std::shared_ptr<OpenMPBLAS<T>> create() { return std::make_shared<OpenMPBLAS<T>>(); }

        auto allocate(const std::size_t n) -> T* override { return (T*)calloc(n, sizeof(T)); }

        void destroy(void* a) override { free(a); }

        void copy(const ptrdiff_t n, const T* const src, T* const dest) override { team_blas::copy(0, n, src, dest); }

        auto dot(const ptrdiff_t n, const T* const l, const T* const r) -> T override {
            const ptrdiff_t chunks = team_chunk_count(n);
            if ((ptrdiff_t)scratch_.size() < TeamReduction<T, 1>::buffer_size(chunks))
                scratch_.resize((size_t)TeamReduction<T, 1>::buffer_size(chunks));
            T* const buffer = scratch_.data();
            if (n < blas_serial_below) {
                TeamReduction<T, 1> reduction(buffer, chunks);
                return team_blas::dot(Team::of(n, 0, 1), reduction, l, r);
            }
            T ret = 0;
#pragma omp parallel
            {
                TeamReduction<T, 1> reduction(buffer, chunks);
                const Team          team = Team::current(n);
                const T             v    = team_blas::dot(team, reduction, l, r);
                if (team.leader()) ret = v;
            }
            return ret;
        }

        void dot(const ptrdiff_t n, const T* const l, const T* const r, T* const result) override { *result = dot(n, l, r); }

        void axpy(const ptrdiff_t n, const T alpha, const T* const x, T* const y) override {
            over_shares(n, [=](const ptrdiff_t b, const ptrdiff_t e) { team_blas::axpy(b, e, alpha, x, y); });
        }

        void axpby(const ptrdiff_t n, const T alpha, const T* const x, const T beta, T* const y) override {
            over_shares(n, [=](const ptrdiff_t b, const ptrdiff_t e) { team_blas::axpby(b, e, alpha, x, beta, y); });
        }

        void zaxpby(const ptrdiff_t n, const T alpha, const T* const x, const T beta, const T* const y, T* const z) override {
            over_shares(n, [=](const ptrdiff_t b, const ptrdiff_t e) { team_blas::zaxpby(b, e, alpha, x, beta, y, z); });
        }

        void zeros(const std::size_t size, T* const x) override {
            over_shares((ptrdiff_t)size, [=](const ptrdiff_t b, const ptrdiff_t e) { team_blas::zeros(b, e, x); });
        }

        auto norm2(const ptrdiff_t n, const T* const x) -> T override { return sqrt(dot(n, x, x)); }

        void values(const std::size_t n, const T v, T* const x) override {
            over_shares((ptrdiff_t)n, [=](const ptrdiff_t b, const ptrdiff_t e) { team_blas::values(b, e, v, x); });
        }

        void scal(const std::ptrdiff_t n, const T alpha, T* const x) override {
            over_shares(n, [=](const ptrdiff_t b, const ptrdiff_t e) { team_blas::scal(b, e, alpha, x); });
        }

        void reciprocal(const std::ptrdiff_t n, const T alpha, T* const x) override {
            over_shares(n, [=](const ptrdiff_t b, const ptrdiff_t e) { team_blas::reciprocal(b, e, alpha, x); });
        }

        void xypaz(const std::ptrdiff_t n, const T* const x, const T* const y, const T alpha, T* const z) override {
            over_shares(n, [=](const ptrdiff_t b, const ptrdiff_t e) { team_blas::xypaz(b, e, x, y, alpha, z); });
        }

    private:
        // The dot products' partial sums, grown to the longest vector's chunks.
        std::vector<T> scratch_;

        // One region, f(b, e) on every thread's static share of the n entries.
        template <typename F>
        static void over_shares(const ptrdiff_t n, F f) {
#pragma omp parallel
            {
                const sfem_team_t team = sfem_team_current(0);
                ptrdiff_t         b, e;
                sfem_team_share(&team, n, 1, 0, &b, &e);
                f(b, e);
            }
        }
    };

}  // namespace sfem

#endif  // SFEM_OPENMP_BLAS_HPP
