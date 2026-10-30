#pragma once

#include <string>

#include "sfem_config.h"

#ifdef SFEM_ENABLE_RYAML
#include <ryml.hpp>
#endif

namespace sfem {

    /// What the stored-tangent (INEXACT) operator stores and how it applies it.
    ///
    /// The tangent is stored at the points of the rule the exact operator integrates with, so
    /// the stored operator is the exact Jacobian up to the encoding's rounding; nothing here
    /// trades accuracy except the encoding.
    ///
    ///   store     points          the weighted, pulled-back tangent at every point (moments on
    ///                             TET10), contracted with the increment's reference gradients
    ///             element_matrix  the element matrices in difference form
    ///             bsr             the operator's assembled tangent, quantized (sfem_QuantBSR.hpp)
    ///   encoding  points:         f64, f32, f16, i8md (int8 deviations from an f32 element mean)
    ///             element_matrix: f64, f32, f16, i8pair (int8 with an 8-bit factor per node pair)
    ///             bsr:            f16 (one scale), i8 (a scale per block)
    ///   fold      auto, on, off   the folded point store of the tensor-product elements (exact)
    ///   traversal auto, packed, atomic
    ///
    /// int8 with one scale per element is not offered: it is not a usable Newton operator
    /// (docs/compressed_partial_assembly, RESULTS_QP.md).
    struct InexactOptions {
        enum class Store { POINTS, ELEMENT_MATRIX, BSR };
        enum class Encoding { F64, F32, F16, I8_MEAN_DEVIATION, I8_PAIR_SCALE, I8_BLOCK };
        enum class Switch { AUTO, ON, OFF };
        enum class Traversal { AUTO, PACKED, ATOMIC };

        Store     store{Store::POINTS};
        Encoding  encoding{Encoding::F32};
        Switch    fold{Switch::AUTO};
        Traversal traversal{Traversal::AUTO};

        /// Refuses a combination the stores do not offer.
        void validate() const;

        static InexactOptions from_strings(const std::string &store,
                                           const std::string &encoding,
                                           const std::string &fold      = "auto",
                                           const std::string &traversal = "auto");
#ifdef SFEM_ENABLE_RYAML
        /// `solver.linear.inexact: {store, encoding, fold, traversal}`; absent keys keep the defaults.
        static InexactOptions from_yaml(const ryml::ConstNodeRef &node);
#endif
        std::string describe() const;
    };

}  // namespace sfem
