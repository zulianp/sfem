#ifndef SFEM_MEMORY_LEDGER_HPP
#define SFEM_MEMORY_LEDGER_HPP

#include <cstddef>
#include <memory>
#include <set>

#include "sfem_base.hpp"
#include "smesh_buffer.hpp"

namespace sfem {

    /// The bytes a run holds, by what they are for, apart on the host and on the device. Each
    /// object that keeps buffers between steps accounts for its own (`account`): there is no
    /// allocator that sees them all. Transient scratch, held only inside a call, is not counted.
    class MemoryLedger {
    public:
        enum Category {
            MESH,               // coordinates and connectivity
            GEOMETRY,           // cached element geometry (adjugates, determinants)
            GRAPH,              // the node graph a space keeps
            TANGENT_VALUES,     // the tangent's numbers: BSR values, or a stored tangent's codes
            TANGENT_STRUCTURE,  // its index arrays: the BSR's rowptr and colidx
            TANGENT_WORK,       // vectors only its format needs (a composed product's, the power method's)
            STEP,               // the linearised step's vectors and its solver's workspace
            CONTACT,            // contact nodes' state
            OBSTACLE_GRID,      // the obstacle's distance field
            SCHEME,             // the time scheme's state
            NEWTON,             // the Newton solver's vectors
            N_CATEGORIES
        };

        static const char *name(const Category c) {
            static const char *const names[N_CATEGORIES] = {"mesh",
                                                            "geometry",
                                                            "graph",
                                                            "tangent_values",
                                                            "tangent_structure",
                                                            "tangent_work",
                                                            "step",
                                                            "contact",
                                                            "obstacle_grid",
                                                            "scheme",
                                                            "newton"};
            return names[c];
        }

        void add(const Category c, const std::size_t bytes, const smesh::MemorySpace space) {
            (space == smesh::MEMORY_SPACE_DEVICE ? device_ : host_)[c] += bytes;
        }

        /// A buffer's bytes, where it lives; nothing for a null one, nor for one already counted
        /// (objects share buffers: the first to account for it names its category).
        template <class B>
        void add(const Category c, const std::shared_ptr<B> &buffer) {
            if (!buffer || !seen_.insert(static_cast<const void *>(buffer->data())).second) return;
            add(c, buffer->nbytes(), buffer->mem_space());
        }

        std::size_t host(const Category c) const { return host_[c]; }
        std::size_t device(const Category c) const { return device_[c]; }

    private:
        std::size_t           host_[N_CATEGORIES]{};
        std::size_t           device_[N_CATEGORIES]{};
        std::set<const void *> seen_;
    };

}  // namespace sfem

#endif  // SFEM_MEMORY_LEDGER_HPP
