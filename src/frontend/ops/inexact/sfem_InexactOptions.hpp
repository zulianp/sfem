#pragma once

#include <string>

#include "sfem_config.h"
#include "smesh_elem_type.hpp"
#include "smesh_spaces.hpp"

// Not in device code: nvcc cannot read what ryml includes on some hosts (arm_neon.h on Grace),
// and nothing on the device reads the options from YAML.
#if defined(SFEM_ENABLE_RYAML) && !defined(__CUDACC__)
#define SFEM_INEXACT_OPTIONS_YAML
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
    /// store and encoding `auto` (both or neither) take the format measured fastest to a Newton
    /// solution for the element and the execution space (`resolve`).
    ///
    /// int8 with one scale per element is not offered: it is not a usable Newton operator
    /// (docs/compressed_partial_assembly, RESULTS_QP.md).
    struct InexactOptions {
        enum class Store { AUTO, POINTS, ELEMENT_MATRIX, BSR };
        enum class Encoding { AUTO, F64, F32, F16, I8_MEAN_DEVIATION, I8_PAIR_SCALE, I8_BLOCK };
        enum class Switch { AUTO, ON, OFF };
        enum class Traversal { AUTO, PACKED, ATOMIC };

        Store     store{Store::POINTS};
        Encoding  encoding{Encoding::F32};
        Switch    fold{Switch::AUTO};
        Traversal traversal{Traversal::AUTO};

        /// Refuses a combination the stores do not offer.
        void validate() const;

        bool is_auto() const { return store == Store::AUTO; }
        /// `auto` made concrete for an element and an execution space: the fastest format to a
        /// Newton solution at about 1e7 dof (docs/compressed_partial_assembly, tables/summary*.tex):
        ///
        ///                  host (Grace)          device (H100)
        ///   TET4           bsr f16               bsr f16
        ///   TET10          bsr f16               bsr i8
        ///   HEX8           bsr i8                bsr i8
        ///   HEX27          element_matrix f16    points f16
        ///
        /// Device TET10 was points i8md, the fastest to one Newton solution with its set-up; a
        /// contact QP applies the tangent thousands of times per set-up, and there bsr i8 is 15%
        /// faster per QP iteration with the same Newton counts and utopia metrics (LDP
        /// FF200_mesh_refined on TET10, 8.5M dof).  points f32 for any other element. Options that are not `auto` are returned as they are.
        InexactOptions resolve(const smesh::ElemType element, const smesh::ExecutionSpace space) const;

        static InexactOptions from_strings(const std::string &store,
                                           const std::string &encoding,
                                           const std::string &fold      = "auto",
                                           const std::string &traversal = "auto");
        std::string describe() const;
    };

#ifdef SFEM_INEXACT_OPTIONS_YAML
    /// `solver.linear.inexact: {store, encoding, fold, traversal}`; absent keys keep the defaults.
    InexactOptions inexact_options_from_yaml(const ryml::ConstNodeRef &node);
#endif

}  // namespace sfem
