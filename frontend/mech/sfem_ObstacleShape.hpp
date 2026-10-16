#ifndef SFEM_OBSTACLE_SHAPE_HPP
#define SFEM_OBSTACLE_SHAPE_HPP

#include <memory>

#include "sfem_aliases.hpp"
#include "sfem_base.hpp"
#include "smesh_grid.hpp"

namespace sfem {

    /// A rigid obstacle under a body, as a signed distance function (positive inside) sampled on
    /// an SDF grid. The body approaches it along -dir. Unset lengths (negative) take the defaults
    /// derived from the body's bounding box.
    struct ObstacleShape {
        enum class Type {
            /// A plane at `plane` along dir.
            PLANE,
            /// A plane sloping by `slope` along `slope_axis`, highest under the body at `plane`.
            SLOPE,
            /// A staircase descending along `slope_axis` from `plane` under the body, treads
            /// `step_width` wide, each `step_drop` (< 0) below the last.
            STAIRCASE,
            /// A hemisphere on a floor, by default centred under the body, its top at `plane`.
            HEMISPHERE
        };

        Type   type{Type::HEMISPHERE};
        int    dir{2};
        real_t plane{0};
        real_t slope{real_t(0.2)};
        /// A tangential axis of dir; -1: the first one.
        int    slope_axis{-1};
        real_t step_width{real_t(0.25)};
        real_t step_drop{real_t(-0.05)};
        /// -1: a quarter of the body's smaller footprint side.
        real_t radius{-1};
        bool   has_center{false};
        real_t center[3]{0, 0, 0};
        /// The rounding of creases and edges; -1: the whole step depth for a staircase, half the
        /// radius otherwise.
        real_t fillet{-1};
        /// Cells along the grid's longest side, or finer where the fillet needs it.
        int    grid_cells{128};
    };

    /// The obstacle's SDF grid over a box around the body [lo, hi] (its bounding box) and a
    /// margin; this grid is the obstacle the contact sees. Prints what it built.
    std::shared_ptr<smesh::Grid<geom_t>> create_obstacle_sdf(const std::shared_ptr<Communicator> &comm,
                                                             const ObstacleShape                 &shape,
                                                             const real_t *const                  lo,
                                                             const real_t *const                  hi);

}  // namespace sfem

#endif  // SFEM_OBSTACLE_SHAPE_HPP
