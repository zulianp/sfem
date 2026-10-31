#pragma once

#include <type_traits>

#include "sfem_base.hpp"

namespace sfem {
    namespace inexact {

        /// What the stores read of a generated element (`reference/inexact_<element>_q<n>.hpp`)
        /// beyond its stages, in one place: the generated header carries only what is data, and
        /// what is a closed form in the index -- the identity pairing of a point store, the
        /// lexicographic node order, the fold's per-point masks on the tensor-product lattice -- is
        /// computed here rather than tabulated.  What reads the header's arrays is host code: a
        /// device function may not name them, so the device kernels read them directly, as
        /// constant expressions (cuda/sfem_cuda_InexactStore.cuh).
        /// The difference form's stored blocks of an N-node element as a table, row and column
        /// node of block k (`ElementTables::block_row`, `block_col`): the host's traversal of
        /// hundreds of blocks reads it rather than searching for each pair.
        template <int N>
        struct NodePairs {
            static constexpr int NP = N * (N - 1) / 2;
            int                  row[2 * NP]{}, col[2 * NP]{};
            constexpr NodePairs() {
                int k = 0;
                for (int p = 0; p < N; ++p)
                    for (int q = p + 1; q < N; ++q, ++k) {
                        row[k] = p, col[k] = q;
                        row[NP + k] = q, col[NP + k] = p;
                    }
            }
        };

        template <class El>
        struct ElementTables {
            /// The gradient point stored tensor j contracts the increment's gradient from, and the one
            /// its output goes to: the point itself for a point store, the moment's pair otherwise.
            static constexpr int pa(const int j) {
                if constexpr (El::moments) {
                    return El::PA[j];
                } else {
                    return j;
                }
            }
            static constexpr int pb(const int j) {
                if constexpr (El::moments) {
                    return El::PB[j];
                } else {
                    return j;
                }
            }
            /// How often tensor j occurs in the symmetric sum the element mean is taken over.
            static constexpr double mult(const int j) {
                if constexpr (El::moments) {
                    return El::MULT[j];
                } else {
                    return 1;
                }
            }
            /// Stored block k of the difference form: node pair p < q in p-major order, `K_pq`
            /// for k below the pair count, `K_qp` above it (an unsymmetric tangent stores both).
            static constexpr SFEM_HOST_DEVICE int block_row(const int k) {
                constexpr int NP = El::N * (El::N - 1) / 2;
                return k < NP ? pair_first(k) : pair_second(k - NP);
            }
            static constexpr SFEM_HOST_DEVICE int block_col(const int k) {
                constexpr int NP = El::N * (El::N - 1) / 2;
                return k < NP ? pair_second(k) : pair_first(k - NP);
            }
            static constexpr SFEM_HOST_DEVICE int pair_first(int pair) {
                int p = 0;
                while (pair >= El::N - 1 - p) pair -= El::N - 1 - p++;
                return p;
            }
            static constexpr SFEM_HOST_DEVICE int pair_second(int pair) {
                int p = 0;
                while (pair >= El::N - 1 - p) pair -= El::N - 1 - p++;
                return p + 1 + pair;
            }
            /// The mesh node staged node s reads.
            static constexpr int shape_node(const int s) {
                if constexpr (El::reordered) {
                    return El::SHAPE_ORDER[s];
                } else {
                    return s;
                }
            }
        };

        /// The folded store of a tensor-product element for a tangent of one packing: the
        /// generated tables (`El::fold_symmetric` or `El::fold_full`: the slot directions, the
        /// slots each mask keeps, the offsets and the folded reference weights) and the lattice
        /// the masks follow.  The point whose index in direction d is `El::DROP` keeps no m = n = d
        /// components (bit d of its mask); they move to the kept points of its line,
        /// `T_j += coefficient(j, d) * T_source(j, d)`.
        template <class El, class Tables>
        struct Fold : Tables {
            static constexpr int Q1 = El::Q1;

            static constexpr SFEM_HOST_DEVICE int index(const int j, const int d) {
                int r = j;
                for (int k = 0; k < d; ++k) r /= Q1;
                return r % Q1;
            }
            static constexpr SFEM_HOST_DEVICE int stride(const int d) {
                int s = 1;
                for (int k = 0; k < d; ++k) s *= Q1;
                return s;
            }
            static constexpr SFEM_HOST_DEVICE int mask(const int j) {
                int m = 0;
                for (int d = 0; d < El::DIM; ++d) m |= (index(j, d) == El::DROP) << d;
                return m;
            }
            static constexpr SFEM_HOST_DEVICE int source(const int j, const int d) {
                return j + (El::DROP - index(j, d)) * stride(d);
            }
            static constexpr double coefficient(const int j, const int d) { return El::COEF1[index(j, d)]; }
        };

        /// No fold: an element whose lines have no point to spare.
        template <class M>
        struct NoFold {
            static constexpr int NC = M::NC;
        };

        template <class M, bool FOLDABLE = M::element::foldable>
        struct FoldOf;

        template <class M>
        struct FoldOf<M, true> {
            using type = Fold<typename M::element,
                              std::conditional_t<M::symmetric,
                                                 typename M::element::fold_symmetric,
                                                 typename M::element::fold_full>>;
        };

        template <class M>
        struct FoldOf<M, false> {
            using type = NoFold<M>;
        };

    }  // namespace inexact
}  // namespace sfem
