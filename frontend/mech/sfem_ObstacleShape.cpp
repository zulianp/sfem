#include "sfem_ObstacleShape.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>

namespace sfem {

    namespace {
        // The obstacle as a signed distance function, positive inside: a shape cut off by a box wide
        // enough for the mesh to fall onto it and spread over it, the cut's edges rounded with `edge`.
        // It is sampled on an SDF grid over the box and a margin (see create_obstacle_sdf); the grid is
        // the obstacle the contact sees. The body approaches the obstacle along -dir.
        //
        // A contact node is held along one normal, so the normal must not jump and the curvature must
        // stay bounded: the obstacles' concave creases (the hemisphere and the floor, a riser and the
        // tread below) and convex edges (a step's nose) are rounded with radius `fillet`. Elsewhere
        // each is an exact signed distance outside the obstacle, where contact is decided.
        struct Obstacle {
            std::function<real_t(const real_t, const real_t, const real_t)> sdf;
            real_t                                                          lo[3], hi[3];  // the grid's box
            real_t                                                          fillet;
        };

        // The union of two obstacles, positive inside, rounded within k of a crease (polynomial smooth
        // maximum; the plain maximum for k = 0).
        real_t rounded_union(const real_t a, const real_t b, const real_t k) {
            const real_t blend = k > 0 ? std::max(k - std::abs(a - b), real_t(0)) / k : 0;
            return std::max(a, b) + blend * blend * k / 4;
        }

        bool create_obstacle(const ObstacleShape &shape, const real_t *const min_c, const real_t *const max_c, Obstacle &obstacle) {
            const int dir = shape.dir;
            const int h1  = dir == 0 ? 1 : 0;
            const int h2  = dir == 2 ? 1 : 2;
            const int s   = shape.slope_axis < 0 ? h1 : shape.slope_axis;

            using Type = ObstacleShape::Type;
            if ((shape.type == Type::SLOPE || shape.type == Type::STAIRCASE) && (s == dir || s > 2)) {
                fprintf(stderr, "ObstacleShape: slope_axis %d must be a tangential axis of the direction %d\n", s, dir);
                return false;
            }
            if (shape.type == Type::STAIRCASE && !(shape.step_drop < 0 && shape.step_width > 0)) {
                fprintf(stderr,
                        "ObstacleShape: step_drop %g must be negative and step_width %g positive\n",
                        (double)shape.step_drop,
                        (double)shape.step_width);
                return false;
            }

            const real_t radius = shape.radius > 0 ? shape.radius : real_t(0.25) * std::min(max_c[h1] - min_c[h1], max_c[h2] - min_c[h2]);
            real_t       center[3];
            center[h1]  = (min_c[h1] + max_c[h1]) / 2;
            center[h2]  = (min_c[h2] + max_c[h2]) / 2;
            center[dir] = shape.plane - radius;
            if (shape.has_center) std::copy(shape.center, shape.center + 3, center);

            // A step is rounded over its whole depth: the interpolating B-spline does not smooth, and a
            // tighter nose resolved by two grid cells left Newton unable to slide the body over it.
            const real_t fillet = shape.fillet >= 0 ? shape.fillet
                                  : shape.type == Type::STAIRCASE ? std::min(shape.step_width, -shape.step_drop)
                                                                   : radius / 2;
            const real_t plane = shape.plane;

            std::function<real_t(const real_t, const real_t, const real_t)> sdf;
            real_t                                                          lowest = plane;
            switch (shape.type) {
                case Type::PLANE: {
                    printf("Obstacle: plane at %g along axis %d\n", (double)plane, dir);
                    sdf = [=](const real_t x, const real_t y, const real_t z) -> real_t {
                        const real_t p[3] = {x, y, z};
                        return plane - p[dir];
                    };
                    break;
                }
                case Type::SLOPE: {
                    const real_t slope = shape.slope;
                    const real_t top   = slope > 0 ? max_c[s] : min_c[s];
                    printf("Obstacle: plane sloping by %g along axis %d, at %g along axis %d where %g\n",
                           (double)slope,
                           s,
                           (double)plane,
                           dir,
                           (double)top);
                    const real_t scale = 1 / std::sqrt(1 + slope * slope);
                    sdf                = [=](const real_t x, const real_t y, const real_t z) -> real_t {
                        const real_t p[3] = {x, y, z};
                        return (plane + slope * (p[s] - top) - p[dir]) * scale;
                    };
                    lowest = plane - std::abs(slope) * (max_c[s] - min_c[s]);
                    break;
                }
                case Type::STAIRCASE: {
                    const real_t width = shape.step_width;
                    const real_t drop  = shape.step_drop;
                    const real_t start = min_c[s];
                    const int    steps = (int)std::ceil((max_c[s] - min_c[s]) / width) + 1;
                    printf("Obstacle: %d steps descending along axis %d from %g, %g wide, %g apart, fillet %g\n",
                           steps,
                           s,
                           (double)plane,
                           (double)width,
                           (double)drop,
                           (double)fillet);
                    // Under the stairs is the union of the quarter-spaces {p_s <= start + (k + 1) w,
                    // p_dir <= plane + k drop}; the last one runs on, the floor beyond. Each is inset by
                    // the fillet and offset back out, which rounds its nose and keeps its faces.
                    sdf = [=](const real_t x, const real_t y, const real_t z) -> real_t {
                        const real_t p[3] = {x, y, z};
                        real_t       d    = plane + (steps - 1) * drop - p[dir];
                        for (int k = 0; k + 1 < steps; ++k) {
                            const real_t a = start + (k + 1) * width - p[s] - fillet;
                            const real_t b = plane + k * drop - p[dir] - fillet;
                            d = rounded_union(d, ((a < 0 && b < 0) ? -std::sqrt(a * a + b * b) : std::min(a, b)) + fillet, fillet);
                        }
                        return d;
                    };
                    lowest = plane + (steps - 1) * drop;
                    break;
                }
                case Type::HEMISPHERE: {
                    const real_t c[3] = {center[0], center[1], center[2]};
                    const real_t r    = radius;
                    printf("Obstacle: hemisphere centre (%g, %g, %g), radius %g, on a floor at %g along axis %d, fillet %g\n",
                           (double)c[0],
                           (double)c[1],
                           (double)c[2],
                           (double)r,
                           (double)c[dir],
                           dir,
                           (double)fillet);
                    sdf = [=](const real_t x, const real_t y, const real_t z) -> real_t {
                        const real_t p[3]   = {x, y, z};
                        const real_t sphere = r - std::sqrt((x - c[0]) * (x - c[0]) + (y - c[1]) * (y - c[1]) + (z - c[2]) * (z - c[2]));
                        return rounded_union(sphere, c[dir] - p[dir], fillet);
                    };
                    lowest = c[dir];
                    break;
                }
            }

            // Where the box cuts the obstacle a body slides over the edge with its face, and node-to-
            // obstacle contact cannot keep a sharp edge from entering a face between nodes: the edge is
            // rounded on the scale of the body, at least an eighth of its footprint.
            const real_t edge = std::max(fillet, std::min(max_c[h1] - min_c[h1], max_c[h2] - min_c[h2]) / 8);

            real_t extent = 0;
            for (int d = 0; d < 3; ++d) {
                extent = std::max(extent, max_c[d] - min_c[d]);
            }
            real_t lo[3], hi[3];
            for (int d = 0; d < 3; ++d) {
                lo[d] = min_c[d] - extent / 2;
                hi[d] = max_c[d] + extent / 2;
            }
            lo[dir] = std::min(lowest, min_c[dir]) - extent / 2;

            obstacle.fillet = fillet;
            obstacle.sdf    = [=](const real_t x, const real_t y, const real_t z) -> real_t {
                // The box's signed distance, positive inside; the intersection is the smaller field.
                const real_t p[3]    = {x, y, z};
                real_t       outside = 0, inside = std::numeric_limits<real_t>::max(), beyond = -std::numeric_limits<real_t>::max();
                for (int d = 0; d < 3; ++d) {
                    const real_t a = std::max(lo[d] - p[d], p[d] - hi[d]);
                    beyond         = std::max(beyond, a);
                    inside         = std::min(inside, -a);
                    if (a > 0) outside += a * a;
                }
                const real_t box = beyond <= 0 ? inside : -std::sqrt(outside);
                return -rounded_union(-sdf(x, y, z), -box, edge);
            };
            // The grid reaches past the box, so the cut and its rounding are on it.
            for (int d = 0; d < 3; ++d) {
                obstacle.lo[d] = lo[d] - 2 * edge;
                obstacle.hi[d] = hi[d] + 2 * edge;
            }
            return true;
        }
    }  // namespace

    // The obstacle on an SDF grid over its box, `grid_cells` cells along the longest side, or
    // finer: at most half the fillet apart, so the grid resolves the rounding whose curvature the
    // contact tangent uses. This grid is the obstacle the contact sees, and the one written for
    // visualisation.
    std::shared_ptr<smesh::Grid<geom_t>> create_obstacle_sdf(const std::shared_ptr<Communicator> &comm,
                                                             const ObstacleShape                 &shape,
                                                             const real_t *const                  min_c,
                                                             const real_t *const                  max_c) {
        Obstacle obstacle;
        if (!create_obstacle(shape, min_c, max_c, obstacle)) return nullptr;

        real_t extent = 0;
        for (int d = 0; d < 3; ++d) {
            extent = std::max(extent, obstacle.hi[d] - obstacle.lo[d]);
        }
        const real_t h =
                obstacle.fillet > 0 ? std::min(extent / shape.grid_cells, obstacle.fillet / 2) : extent / shape.grid_cells;

        // And 8 cells more on every side: the B-spline's prefilter extends each grid line beyond its
        // ends (sfem_sdf_sampling.cpp), which pulls the curvature towards zero within a few cells of
        // them (by 0.268^k at k cells), so the obstacle's box keeps clear of that layer.
        real_t    lo[3], hi[3];
        ptrdiff_t n[3];
        for (int d = 0; d < 3; ++d) {
            lo[d] = obstacle.lo[d] - 8 * h;
            hi[d] = obstacle.hi[d] + 8 * h;
            n[d]  = (ptrdiff_t)std::ceil((hi[d] - lo[d]) / h) + 1;
        }
        return smesh::create_sdf(
                comm,
                n[0],
                n[1],
                n[2],
                lo[0],
                lo[1],
                lo[2],
                hi[0],
                hi[1],
                hi[2],
                [&obstacle](const geom_t x, const geom_t y, const geom_t z) -> geom_t { return obstacle.sdf(x, y, z); });
    }

}  // namespace sfem
