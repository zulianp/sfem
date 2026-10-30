#pragma once

#include <type_traits>

#include "sfem_base.hpp"

namespace sfem {
    namespace inexact {

        /// What the stores read of a generated element (`reference/inexact_<element>_q<n>.hpp`)
        /// beyond its stages, in one place: the generated header carries only what is data, and
        /// what is a closed form in the index -- the identity pairing of a point store, the
        /// lexicographic node order, the fold's per-point masks on the tensor-product lattice -- is
        /// computed here rather than tabulated.
        template <class El>
        struct ElementTables {
            /// The gradient point stored tensor j contracts the increment's gradient from, and the one
            /// its output goes to: the point itself for a point store, the moment's pair otherwise.
            static constexpr SFEM_INLINE int pa(const int j) {
                if constexpr (El::moments) {
                    return El::PA[j];
                } else {
                    return j;
                }
            }
            static constexpr SFEM_INLINE int pb(const int j) {
                if constexpr (El::moments) {
                    return El::PB[j];
                } else {
                    return j;
                }
            }
            /// How often tensor j occurs in the symmetric sum the element mean is taken over.
            static constexpr SFEM_INLINE double mult(const int j) {
                if constexpr (El::moments) {
                    return El::MULT[j];
                } else {
                    return 1;
                }
            }
            /// The mesh node staged node s reads.
            static constexpr SFEM_INLINE int shape_node(const int s) {
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

            static constexpr SFEM_INLINE int index(const int j, const int d) {
                int r = j;
                for (int k = 0; k < d; ++k) r /= Q1;
                return r % Q1;
            }
            static constexpr SFEM_INLINE int stride(const int d) {
                int s = 1;
                for (int k = 0; k < d; ++k) s *= Q1;
                return s;
            }
            static constexpr SFEM_INLINE int mask(const int j) {
                int m = 0;
                for (int d = 0; d < El::DIM; ++d) m |= (index(j, d) == El::DROP) << d;
                return m;
            }
            static constexpr SFEM_INLINE int source(const int j, const int d) {
                return j + (El::DROP - index(j, d)) * stride(d);
            }
            static constexpr SFEM_INLINE double coefficient(const int j, const int d) { return El::COEF1[index(j, d)]; }
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
