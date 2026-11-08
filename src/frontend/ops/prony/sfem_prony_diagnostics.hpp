#pragma once

#include "sfem_config.h"

// What the storage format does to the history updates, counted by the host loops when SFEM is
// configured with SFEM_ENABLE_PRONY_DIAGNOSTICS and compiled out otherwise.
//
// An update is a stored value that `advance` meant to change; it is lost when the value read back
// after re-encoding is the one read before. A long relaxation time makes the change per step small
// against the value, and a quantized store then rounds it away -- the history stops moving while
// the stress driving it does not.

namespace sfem {
    namespace prony {

        /// Cumulative over the process, every block and group alike.
        struct UpdateCounts {
            long long updated{0};  ///< values advance meant to change
            long long lost{0};     ///< of those, read back unchanged
        };

        inline UpdateCounts &update_counts() {
            static UpdateCounts counts;
            return counts;
        }

    }  // namespace prony
}  // namespace sfem

#ifdef SFEM_ENABLE_PRONY_DIAGNOSTICS
/// Per-thread counters of a loop body; committed with SFEM_PRONY_COMMIT_COUNTS.
#define SFEM_PRONY_COUNTERS long long prony_updated_ = 0, prony_lost_ = 0
/// A value meant to go from `before` to `meant` was stored as `after`.
#define SFEM_PRONY_COUNT(before, meant, after)                  \
    do {                                                        \
        if ((meant) != (before)) {                              \
            ++prony_updated_;                                   \
            prony_lost_ += ((after) == (before));               \
        }                                                       \
    } while (0)
#define SFEM_PRONY_COMMIT_COUNTS                                              \
    do {                                                                      \
        _Pragma("omp atomic") sfem::prony::update_counts().updated += prony_updated_; \
        _Pragma("omp atomic") sfem::prony::update_counts().lost += prony_lost_;       \
    } while (0)
/// A statement that exists only for the counters.
#define SFEM_PRONY_DIAGNOSTIC(statement) statement
#else
#define SFEM_PRONY_COUNTERS
#define SFEM_PRONY_COUNT(before, meant, after)
#define SFEM_PRONY_COMMIT_COUNTS
#define SFEM_PRONY_DIAGNOSTIC(statement)
#endif
