#ifndef SFEM_TEAM_BLAS_HPP
#define SFEM_TEAM_BLAS_HPP

#include <cstddef>
#include <cstring>

#include "sfem_Team.hpp"
#include "sfem_TeamReduction.hpp"

// The host BLAS over a range [b, e) of the entries, for the thread that owns it: inside a solver's
// region the range is the thread's Team share, so a thread updates and reads back only its own
// entries between barriers; OpenMPBLAS runs the same functions over the static share of a region
// of its own. The dot product is the Team's deterministic sum (TeamReduction).

namespace sfem {
    namespace team_blas {

        template <typename T>
        void axpy(const ptrdiff_t b, const ptrdiff_t e, const T alpha, const T* const x, T* const y) {
            for (ptrdiff_t i = b; i < e; i++) {
                y[i] += alpha * x[i];
            }
        }

        template <typename T>
        void axpby(const ptrdiff_t b, const ptrdiff_t e, const T alpha, const T* const x, const T beta, T* const y) {
            for (ptrdiff_t i = b; i < e; i++) {
                y[i] = alpha * x[i] + beta * y[i];
            }
        }

        template <typename T>
        void zaxpby(const ptrdiff_t b, const ptrdiff_t e, const T alpha, const T* const x, const T beta, const T* const y, T* const z) {
            for (ptrdiff_t i = b; i < e; i++) {
                z[i] = alpha * x[i] + beta * y[i];
            }
        }

        template <typename T>
        void zeros(const ptrdiff_t b, const ptrdiff_t e, T* const x) {
            if (e > b) memset(&x[b], 0, (size_t)(e - b) * sizeof(T));
        }

        template <typename T>
        void copy(const ptrdiff_t b, const ptrdiff_t e, const T* const src, T* const dest) {
            if (e > b) memcpy(&dest[b], &src[b], (size_t)(e - b) * sizeof(T));
        }

        template <typename T>
        void values(const ptrdiff_t b, const ptrdiff_t e, const T v, T* const x) {
            for (ptrdiff_t i = b; i < e; i++) {
                x[i] = v;
            }
        }

        template <typename T>
        void scal(const ptrdiff_t b, const ptrdiff_t e, const T alpha, T* const x) {
            for (ptrdiff_t i = b; i < e; i++) {
                x[i] *= alpha;
            }
        }

        template <typename T>
        void reciprocal(const ptrdiff_t b, const ptrdiff_t e, const T alpha, T* const x) {
            for (ptrdiff_t i = b; i < e; i++) {
                if (x[i]) x[i] = alpha / x[i];
            }
        }

        template <typename T>
        void xypaz(const ptrdiff_t b, const ptrdiff_t e, const T* const x, const T* const y, const T alpha, T* const z) {
            for (ptrdiff_t i = b; i < e; i++) {
                z[i] = x[i] * y[i] + alpha * z[i];
            }
        }

        /// l . r over the team's entries, the same on every thread of the region.
        template <typename T>
        T dot(const Team& team, TeamReduction<T, 1>& reduction, const T* const l, const T* const r) {
            T out[1];
            reduction.sum(team, {T(0)}, false, [l, r](const ptrdiff_t b, const ptrdiff_t e, T (&s)[1]) {
                T acc = s[0];
                for (ptrdiff_t i = b; i < e; i++) acc += l[i] * r[i];
                s[0] = acc;
            }, out);
            return out[0];
        }

    }  // namespace team_blas
}  // namespace sfem

#endif  // SFEM_TEAM_BLAS_HPP
