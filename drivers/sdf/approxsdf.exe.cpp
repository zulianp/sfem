#include "sfem_API.hpp"

#include "adj_table.h"
#include "div.h"
#include "mass.h"
#include "point_triangle_distance.h"
#include "sfem_macros.h"
#include "sortreduce.h"

#include "sfem_API.hpp"

namespace sfem {
    class CellList {
    public:
        ptrdiff_t n[3];
        ptrdiff_t stride[3];
        geom_t    o[3];
        geom_t    delta[3];
        geom_t    radius;

        std::shared_ptr<Buffer<ptrdiff_t>> cell_ptr;
        std::shared_ptr<Buffer<ptrdiff_t>> cell_idx;

        void print(std::ostream &os) const {
            os << "----------------------------\n";
            os << n[0] << " x " << n[1] << " x " << n[2] << "\n";
            os << stride[0] << " x " << stride[1] << " x " << stride[2] << "\n";
            os << o[0] << " x " << o[1] << " x " << o[2] << "\n";
            os << delta[0] << " x " << delta[1] << " x " << delta[2] << "\n";

            ptrdiff_t n  = cell_ptr->size() - 1;
            auto      cp = cell_ptr->data();
            auto      ci = cell_idx->data();

            // os << "cells:\n";

            // for (ptrdiff_t i = 0; i < n; i++) {
            //     os << i << ") ";
            //     for (ptrdiff_t k = cp[i]; k < cp[i + 1]; k++) {
            //         os << ci[k] << " ";
            //     }

            //     os << "\n";
            // }

            os << "----------------------------\n";
        }
    };

    static SFEM_INLINE void cell_list_coords(const int                            dim,
                                             const ptrdiff_t *const SFEM_RESTRICT n,
                                             const geom_t *const SFEM_RESTRICT    o,
                                             const geom_t *const SFEM_RESTRICT    delta,
                                             const geom_t *const SFEM_RESTRICT    p,
                                             ptrdiff_t *const SFEM_RESTRICT       coords) {
        for (int d = 0; d < dim; d++) {
            geom_t val = p[d] - o[d];
            val /= delta[d];
            coords[d] = floor(val);
            assert(coords[d] < n[d]);
        }
    }

    static SFEM_INLINE ptrdiff_t cell_list_idx(const int              dim,
                                               const ptrdiff_t *const stride,
                                               const geom_t *const    o,
                                               const geom_t *const    delta,
                                               const geom_t *const    p) {
        ptrdiff_t idx = 0;
        for (int d = 0; d < dim; d++) {
            geom_t val = p[d] - o[d];
            val /= delta[d];
            idx += floor(val) * stride[d];
        }

        return idx;
    }

    std::shared_ptr<CellList> create_cell_list_from_elements(const std::shared_ptr<sfem::Mesh> &mesh) {
        const ptrdiff_t nnodes = mesh->n_nodes();
        auto            points = mesh->points();

        const ptrdiff_t nelements = mesh->n_elements();
        auto            elements  = mesh->elements();
        const int       dim       = mesh->spatial_dimension();
        const int       nxe       = elem_num_nodes(mesh->element_type());

        auto p     = points->data();
        auto elems = elements->data();

        geom_t min[3]    = {10000, 10000, 10000};
        geom_t max[3]    = {-10000, -10000, -10000};
        geom_t radius[3] = {0, 0, 0};

        for (int d = 0; d < dim; d++) {
            for (ptrdiff_t i = 0; i < nnodes; i++) {
                min[d] = MIN(min[d], p[d][i]);
                max[d] = MAX(max[d], p[d][i]);
            }
        }

        for (int d = 0; d < dim; d++) {
            for (ptrdiff_t i = 0; i < nelements; i++) {
                for (int v1 = 0; v1 < nxe; v1++) {
                    for (int v2 = 1; v2 < nxe; v2++) {
                        geom_t dist = fabs(p[d][elems[v1][i]] - p[d][elems[v2][i]]);
                        radius[d]   = MAX(dist, radius[d]);
                    }
                }
            }
        }

        auto ret = std::make_shared<CellList>();

        ptrdiff_t ncells = 1;
        for (int d = 0; d < dim; d++) {
            geom_t extent = (max[d] - min[d]);
            ret->n[d]     = extent / (2 * radius[d]);
            ret->delta[d] = extent / ret->n[d];
            ret->o[d]     = min[d];

            printf("%d) %ld %f\n", d, ret->n[d], ret->delta[d]);

            ncells *= ret->n[d];
        }

        ret->stride[0] = 1;
        ret->stride[1] = 1;
        ret->stride[2] = 1;
        for (int d = 1; d < dim; d++) {
            ret->stride[d] = ret->stride[d - 1] * ret->n[d - 1];
        }

        printf("nelements %ld\n", nelements);
        printf("ncells: %ld x %ld = %ld\n", ret->n[0], ret->n[1], ncells);

        auto cell_ptr = sfem::create_host_buffer<ptrdiff_t>(ncells + 1);
        auto cp       = cell_ptr->data();

        for (ptrdiff_t i = 0; i < nelements; i++) {
            geom_t emin[3];
            for (int d = 0; d < dim; d++) {
                emin[d] = p[d][elems[0][i]];
            }

            for (int v1 = 1; v1 < nxe; v1++) {
                for (int d = 0; d < dim; d++) {
                    emin[d] = MIN(p[d][elems[v1][i]], emin[d]);
                }
            }

            const ptrdiff_t idx = cell_list_idx(dim, ret->stride, ret->o, ret->delta, emin);
            cp[idx + 1]++;
        }

        for (ptrdiff_t i = 0; i < ncells; i++) {
            cp[i + 1] += cp[i];
        }

        auto cell_idx = sfem::create_host_buffer<ptrdiff_t>(cp[ncells]);
        auto ci       = cell_idx->data();

        auto bookeepping = sfem::create_host_buffer<ptrdiff_t>(ncells);
        auto bk          = bookeepping->data();

        for (ptrdiff_t i = 0; i < nelements; i++) {
            geom_t emin[3];
            for (int d = 0; d < dim; d++) {
                emin[d] = p[d][elems[0][i]];
            }

            for (int v1 = 1; v1 < nxe; v1++) {
                for (int d = 0; d < dim; d++) {
                    emin[d] = MIN(p[d][elems[v1][i]], emin[d]);
                }
            }

            const ptrdiff_t idx     = cell_list_idx(dim, ret->stride, ret->o, ret->delta, emin);
            ci[cp[idx] + bk[idx]++] = i;
        }

        // cell_idx->print(std::cout);
        ret->cell_idx = cell_idx;
        ret->cell_ptr = cell_ptr;
        return ret;
    }

    std::shared_ptr<CellList> create_cell_list_from_nodes(const std::shared_ptr<sfem::Mesh> &mesh, const geom_t radius_factor = 1) {
        const ptrdiff_t nnodes = mesh->n_nodes();
        auto            points = mesh->points();

        const ptrdiff_t nelements = mesh->n_elements();
        auto            elements  = mesh->elements();
        const int       dim       = mesh->spatial_dimension();
        const int       nxe       = elem_num_nodes(mesh->element_type());

        auto p     = points->data();
        auto elems = elements->data();

        geom_t min[3]    = {10000, 10000, 10000};
        geom_t max[3]    = {-10000, -10000, -10000};
        geom_t radius[3] = {0, 0, 0};

        for (int d = 0; d < dim; d++) {
            for (ptrdiff_t i = 0; i < nnodes; i++) {
                min[d] = MIN(min[d], p[d][i]);
                max[d] = MAX(max[d], p[d][i]);
            }

            min[d] -= 1e-6;
            max[d] += 1e-6;
        }

        for (int d = 0; d < dim; d++) {
            for (ptrdiff_t i = 0; i < nelements; i++) {
                for (int v1 = 0; v1 < nxe; v1++) {
                    for (int v2 = 1; v2 < nxe; v2++) {
                        geom_t dist = fabs(p[d][elems[v1][i]] - p[d][elems[v2][i]]);
                        radius[d]   = MAX(dist, radius[d]);
                    }
                }
            }
        }

        auto ret = std::make_shared<CellList>();

        geom_t max_radius = radius[0];
        for (int d = 1; d < dim; d++) {
            max_radius = MAX(radius[d], max_radius);
        }

        max_radius *= radius_factor;

        ret->radius = max_radius;

        ptrdiff_t ncells = 1;
        for (int d = 0; d < dim; d++) {
            geom_t extent = (max[d] - min[d]);
            ret->n[d]     = extent / (2 * max_radius);
            ret->delta[d] = extent / ret->n[d];
            ret->o[d]     = min[d];

            printf("%d) %ld %f\n", d, ret->n[d], ret->delta[d]);

            ncells *= ret->n[d];
        }

        ret->stride[0] = 1;
        ret->stride[1] = 1;
        ret->stride[2] = 1;
        for (int d = 1; d < dim; d++) {
            ret->stride[d] = ret->stride[d - 1] * ret->n[d - 1];
        }

        printf("nelements %ld\n", nnodes);
        printf("ncells: %ld x %ld = %ld\n", ret->n[0], ret->n[1], ncells);

        auto cell_ptr = sfem::create_host_buffer<ptrdiff_t>(ncells + 1);
        auto cp       = cell_ptr->data();

        for (ptrdiff_t i = 0; i < nnodes; i++) {
            geom_t point[3];
            for (int d = 0; d < dim; d++) {
                point[d] = p[d][i];
            }

            const ptrdiff_t idx = cell_list_idx(dim, ret->stride, ret->o, ret->delta, point);
            cp[idx + 1]++;
        }

        for (ptrdiff_t i = 0; i < ncells; i++) {
            cp[i + 1] += cp[i];
        }

        auto cell_idx = sfem::create_host_buffer<ptrdiff_t>(cp[ncells]);
        auto ci       = cell_idx->data();

        auto bookeepping = sfem::create_host_buffer<ptrdiff_t>(ncells);
        auto bk          = bookeepping->data();

        for (ptrdiff_t i = 0; i < nnodes; i++) {
            geom_t point[3];
            for (int d = 0; d < dim; d++) {
                point[d] = p[d][i];
            }

            const ptrdiff_t idx     = cell_list_idx(dim, ret->stride, ret->o, ret->delta, point);
            ci[cp[idx] + bk[idx]++] = i;
        }

        // cell_idx->print(std::cout);
        ret->cell_idx = cell_idx;
        ret->cell_ptr = cell_ptr;
        return ret;
    }

    static SFEM_INLINE geom_t closest_distance_and_normal(const geom_t *const SFEM_RESTRICT p,
                                                          const geom_t                      line[3][3],
                                                          const geom_t                      line_normal[3][3],
                                                          const geom_t                      orientation,
                                                          geom_t *const SFEM_RESTRICT       point_normal) {
        // Line segment endpoints
        const geom_t l0x = line[0][0];
        const geom_t l0y = line[1][0];
        const geom_t l1x = line[0][1];
        const geom_t l1y = line[1][1];

        // Line direction vector
        const geom_t vx = l1x - l0x;
        const geom_t vy = l1y - l0y;

        // Vector from line start to point
        const geom_t wx = p[0] - l0x;
        const geom_t wy = p[1] - l0y;

        // Line segment length squared
        const geom_t len_sq = vx * vx + vy * vy;

        // Project point onto line using dot product
        const geom_t t = (len_sq != 0) ? (wx * vx + wy * vy) / len_sq : 0;

        // Clamp projection to segment
        const geom_t t_clamped = t < 0 ? 0 : (t > 1 ? 1 : t);

        // Closest point on line

        const geom_t cx = l0x + t_clamped * vx;
        const geom_t cy = l0y + t_clamped * vy;

        // Distance vector
        const geom_t dx = p[0] - cx;
        const geom_t dy = p[1] - cy;

        const geom_t pnx = line_normal[0][0] * (1 - t_clamped) + line_normal[0][1] * t_clamped;
        const geom_t pny = line_normal[1][0] * (1 - t_clamped) + line_normal[1][1] * t_clamped;

        if (t_clamped == 1 || t_clamped == 0) {
            point_normal[0] = pnx;
            point_normal[1] = pny;
        } else {
            point_normal[0] = vy;
            point_normal[1] = -vx;
        }

        point_normal[0] *= orientation;
        point_normal[1] *= orientation;

        const geom_t len_normal = sqrt(point_normal[0] * point_normal[0] + point_normal[1] * point_normal[1]);
        assert(len_normal != 0);

        point_normal[0] /= len_normal;
        point_normal[1] /= len_normal;

        const geom_t sign = orientation * (pnx * (p[0] - cx) + pny * (p[1] - cy)) < 0 ? -1 : 1;

        // Return distance
        return sign * sqrt(dx * dx + dy * dy);
    }

    // static SFEM_INLINE geom_t point_triangle_distance_3d(const geom_t *const p,  // Point coordinates
    //                                                      const geom_t        x[3][3])   // Triangle vertex coordinates
    // {
    //     // Triangle vectors
    //     geom_t v0[3], v1[3], v2[3];
    //     for (int d = 0; d < 3; d++) {
    //         v0[d] = x[d][1] - x[d][0];  // edge 0
    //         v1[d] = x[d][2] - x[d][0];  // edge 1
    //         v2[d] = p[d] - x[d][0];     // point to vertex
    //     }

    //     // Compute dot products
    //     const geom_t dot00 = v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2];
    //     const geom_t dot01 = v0[0] * v1[0] + v0[1] * v1[1] + v0[2] * v1[2];
    //     const geom_t dot02 = v0[0] * v2[0] + v0[1] * v2[1] + v0[2] * v2[2];
    //     const geom_t dot11 = v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2];
    //     const geom_t dot12 = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];

    //     // Compute barycentric coordinates
    //     const geom_t denom = dot00 * dot11 - dot01 * dot01;
    //     const geom_t s     = (dot11 * dot02 - dot01 * dot12) / denom;
    //     const geom_t t     = (dot00 * dot12 - dot01 * dot02) / denom;

    //     if (s >= 0 && t >= 0 && s + t <= 1) {
    //         // Inside triangle - compute distance to plane
    //         geom_t closest[3];
    //         for (int d = 0; d < 3; d++) {
    //             closest[d] = x[d][0] + s * v0[d] + t * v1[d];
    //         }
    //         geom_t dist_sq = 0;
    //         for (int d = 0; d < 3; d++) {
    //             const geom_t diff = p[d] - closest[d];
    //             dist_sq += diff * diff;
    //         }
    //         return sqrt(dist_sq);
    //     }

    //     // Outside triangle - find closest point on edges/vertices
    //     geom_t min_dist = INFINITY;

    //     // Check vertices
    //     for (int i = 0; i < 3; i++) {
    //         geom_t dist_sq = 0;
    //         for (int d = 0; d < 3; d++) {
    //             const geom_t diff = p[d] - x[d][i];
    //             dist_sq += diff * diff;
    //         }
    //         min_dist = MIN(min_dist, sqrt(dist_sq));
    //     }

    //     // Check edges
    //     const int edges[3][2] = {{0, 1}, {1, 2}, {2, 0}};
    //     for (int e = 0; e < 3; e++) {
    //         const int i0 = edges[e][0];
    //         const int i1 = edges[e][1];

    //         geom_t edge[3], diff[3];
    //         for (int d = 0; d < 3; d++) {
    //             edge[d] = x[d][i1] - x[d][i0];
    //             diff[d] = p[d] - x[d][i0];
    //         }

    //         const geom_t len_sq = edge[0] * edge[0] + edge[1] * edge[1] + edge[2] * edge[2];
    //         geom_t       t      = (edge[0] * diff[0] + edge[1] * diff[1] + edge[2] * diff[2]) / len_sq;
    //         t                   = MAX(0, MIN(1, t));

    //         geom_t dist_sq = 0;
    //         for (int d = 0; d < 3; d++) {
    //             const geom_t proj = x[d][i0] + t * edge[d];
    //             const geom_t d    = p[d] - proj;
    //             dist_sq += d * d;
    //         }
    //         min_dist = MIN(min_dist, sqrt(dist_sq));
    //     }

    //     return min_dist;
    // }

    void init_sdf_brute_force(const std::shared_ptr<Mesh>             &surface,
                              const std::shared_ptr<Buffer<real_t *>> &surface_normals,
                              const geom_t                             orientation,
                              const std::shared_ptr<Mesh>             &mesh,
                              const std::shared_ptr<Buffer<real_t>>   &distance,
                              const std::shared_ptr<Buffer<real_t *>> &normals) {
        auto            e_surf   = surface->elements()->data();
        auto            p_surf   = surface->points()->data();
        auto            n_surf   = surface_normals->data();
        const ptrdiff_t ne_surf  = surface->n_elements();
        const int       nxe_surf = elem_num_nodes(surface->element_type());

        auto            p_mesh = mesh->points()->data();
        auto            n_mesh = normals->data();
        auto            dist   = distance->data();
        const ptrdiff_t nnodes = mesh->n_nodes();
        const int       dim    = mesh->spatial_dimension();

        for (ptrdiff_t node = 0; node < nnodes; node++) {
            geom_t p[3] = {0, 0, 0};

            for (int d = 0; d < dim; d++) {
                p[d] = p_mesh[d][node];
            }

            for (ptrdiff_t i_surf = 0; i_surf < ne_surf; i_surf++) {
                geom_t xx_surf[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
                geom_t nn_surf[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};

                for (int d = 0; d < dim; d++) {
                    for (int v_surf = 0; v_surf < nxe_surf; v_surf++) {
                        idx_t node_surf    = e_surf[v_surf][i_surf];
                        xx_surf[d][v_surf] = p_surf[d][node_surf];
                        nn_surf[d][v_surf] = n_surf[d][node_surf];
                    }
                }

                if (dim == 2) {
                    geom_t point_normal[2] = {0, 0};
                    geom_t dd              = closest_distance_and_normal(p, xx_surf, nn_surf, orientation, point_normal);

                    if (fabs(dd) < fabs(dist[node])) {
                        dist[node] = dd;
                        for (int d = 0; d < dim; d++) {
                            n_mesh[d][node] = point_normal[d];
                        }
                    }
                } else {
                    SFEM_ERROR("NOT IMPLEMENTED!\n");
                }
            }
        }
    }

    void init_sdf(const std::shared_ptr<Mesh>             &surface,
                  const std::shared_ptr<Buffer<real_t *>> &surface_normals,
                  const geom_t                             orientation,
                  const std::shared_ptr<CellList>         &cell_list,
                  const std::shared_ptr<Mesh>             &mesh,
                  const std::shared_ptr<Buffer<real_t>>   &distance,
                  const std::shared_ptr<Buffer<real_t *>> &normals) {
        const ptrdiff_t ne_surf  = surface->n_elements();
        const ptrdiff_t ne_mesh  = mesh->n_elements();
        const int       dim      = mesh->spatial_dimension();
        const int       nxe_surf = elem_num_nodes(surface->element_type());

        auto e_surf = surface->elements()->data();
        auto p_surf = surface->points()->data();
        auto n_surf = surface_normals->data();

        auto p_mesh = mesh->points()->data();

        auto dist = distance->data();

        auto stride   = cell_list->stride;
        auto o        = cell_list->o;
        auto delta    = cell_list->delta;
        auto n        = cell_list->n;
        auto cell_ptr = cell_list->cell_ptr->data();
        auto cell_idx = cell_list->cell_idx->data();

        auto radius = cell_list->radius;

        for (ptrdiff_t i_surf = 0; i_surf < ne_surf; i_surf++) {
            geom_t box_min[3] = {0, 0, 0}, box_max[3] = {0, 0, 0};
            geom_t xx_surf[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
            geom_t nn_surf[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};

            for (int d = 0; d < dim; d++) {
                for (int v_surf = 0; v_surf < nxe_surf; v_surf++) {
                    idx_t node_surf    = e_surf[v_surf][i_surf];
                    xx_surf[d][v_surf] = p_surf[d][node_surf];
                    nn_surf[d][v_surf] = n_surf[d][node_surf];
                }
            }

            for (int d = 0; d < dim; d++) {
                box_min[d] = xx_surf[d][0];
                box_max[d] = xx_surf[d][0];
            }

            for (int v_surf = 1; v_surf < nxe_surf; v_surf++) {
                for (int d = 0; d < dim; d++) {
                    box_min[d] = MIN(xx_surf[d][v_surf], box_min[d]);
                    box_max[d] = MAX(xx_surf[d][v_surf], box_max[d]);
                }
            }

            ptrdiff_t start_coord[3] = {1, 1, 0};
            ptrdiff_t end_coord[3]   = {1, 1, 1};

            cell_list_coords(dim, n, o, delta, box_min, start_coord);
            cell_list_coords(dim, n, o, delta, box_max, end_coord);

            for (int d = 0; d < dim; d++) {
                start_coord[d] = MAX(start_coord[d] - 1, 0);
                end_coord[d]   = MIN(end_coord[d] + 2, n[d]);

                assert(start_coord[d] <= end_coord[d]);
            }

            // printf("-----------------------\n");

            // printf("%g %g -- %g %g\n", box_min[0], box_min[1], box_max[0], box_max[1]);

            // printf("%ld %ld %ld -- %ld %ld %ld\n",
            //        start_coord[0],
            //        start_coord[1],
            //        start_coord[2],
            //        end_coord[0],
            //        end_coord[1],
            //        end_coord[2]);

            // printf("-----------------------\n");

            for (int zi = start_coord[2]; zi < end_coord[2]; zi++) {
                for (int yi = start_coord[1]; yi < end_coord[1]; yi++) {
                    for (int xi = start_coord[0]; xi < end_coord[0]; xi++) {
                        const ptrdiff_t idx = xi * stride[0] + yi * stride[1] + zi * stride[2];

                        for (ptrdiff_t k = cell_ptr[idx]; k < cell_ptr[idx + 1]; k++) {
                            ptrdiff_t node = cell_idx[k];

                            geom_t p[3];
                            for (int d = 0; d < dim; d++) {
                                p[d] = p_mesh[d][node];
                            }

                            if (dim == 2) {
                                geom_t point_normal[2] = {0, 0};
                                geom_t dd = closest_distance_and_normal(p, xx_surf, nn_surf, orientation, point_normal);

                                if (fabs(dd) < fabs(dist[node]) && fabs(dd) < radius) {
                                    dist[node] = dd;
                                    for (int d = 0; d < dim; d++) {
                                        normals->data()[d][node] = point_normal[d];
                                    }
                                }

                            } else if (dim == 3) {
                                assert(false);
                                // geom_t p[3];

                                // geom_t dd  = point_triangle_distance_3d(p, xx_surf);
                                // dist[node] = MIN(dist[node], dd);
                            }
                        }
                    }
                }
            }
        }
    }

}  // namespace sfem

void compute_pseudo_normals(enum ElemType                element_type,
                            const ptrdiff_t              n_elements,
                            const ptrdiff_t              n_nodes,
                            idx_t **const SFEM_RESTRICT  elements,
                            geom_t **const SFEM_RESTRICT points,
                            real_t **const SFEM_RESTRICT normals) {
    // TODO: pseudo normal computation using surface
    assert(element_type == EDGE2 && "IMPLEMENT OTHER CASES");

    for (ptrdiff_t i = 0; i < n_elements; i++) {
        const idx_t i0 = elements[0][i];
        const idx_t i1 = elements[1][i];

        const geom_t ux = points[0][i1] - points[0][i0];
        const geom_t uy = points[1][i1] - points[1][i0];

        const geom_t len = sqrt(ux * ux + uy * uy);
        assert(len != 0);

        const geom_t nx = uy;
        const geom_t ny = -ux;

        normals[0][i0] += nx;
        normals[1][i0] += ny;

        normals[0][i1] += nx;
        normals[1][i1] += ny;

        assert(nx * nx + ny * ny > 0);
    }

    for (ptrdiff_t i = 0; i < n_nodes; i++) {
        geom_t nx  = normals[0][i];
        geom_t ny  = normals[1][i];
        geom_t len = sqrt(nx * nx + ny * ny);

        if (len > 0) {
            nx /= len;
            ny /= len;
        }

        normals[0][i] = nx;
        normals[1][i] = ny;
    }
}

static SFEM_INLINE void tri3_div_points(const scalar_t                      px0,
                                        const scalar_t                      px1,
                                        const scalar_t                      px2,
                                        const scalar_t                      py0,
                                        const scalar_t                      py1,
                                        const scalar_t                      py2,
                                        const scalar_t *const SFEM_RESTRICT ux,
                                        const scalar_t *const SFEM_RESTRICT uy,
                                        scalar_t *const SFEM_RESTRICT       element_vector) {
    const scalar_t x0 = -px0 + px1;
    const scalar_t x1 = -py0 + py2;
    const scalar_t x2 = px0 - px2;
    const scalar_t x3 = py0 - py1;
    const scalar_t x4 = x0 * x1 - x2 * x3;
    const scalar_t x5 = 1.0 / x4;
    const scalar_t x6 = (1.0 / 6.0) * x5;
    const scalar_t x7 = ux[0] * x6;
    const scalar_t x8 = uy[0] * x6;
    const scalar_t x9 = x4 * ((1.0 / 6.0) * ux[1] * x1 * x5 + (1.0 / 6.0) * ux[2] * x3 * x5 + (1.0 / 6.0) * uy[1] * x2 * x5 +
                              (1.0 / 6.0) * uy[2] * x0 * x5 - x0 * x8 - x1 * x7 - x2 * x8 - x3 * x7);
    element_vector[0] = x9;
    element_vector[1] = x9;
    element_vector[2] = x9;
}

int tri3_div_apply(const ptrdiff_t     nelements,
                   const ptrdiff_t     nnodes,
                   idx_t **const       elems,
                   geom_t **const      xyz,
                   const real_t *const ux,
                   const real_t *const uy,
                   real_t *const       values) {
    SFEM_UNUSED(nnodes);

    idx_t  ev[3];
    real_t element_vector[3];
    real_t element_ux[3];
    real_t element_uy[3];

    for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(3)
        for (int v = 0; v < 3; ++v) {
            ev[v] = elems[v][i];
        }

        for (int v = 0; v < 3; ++v) {
            element_ux[v] = ux[ev[v]];
        }

        for (int v = 0; v < 3; ++v) {
            element_uy[v] = uy[ev[v]];
        }

        // Element indices
        const idx_t i0 = ev[0];
        const idx_t i1 = ev[1];
        const idx_t i2 = ev[2];

        tri3_div_points(
                // X-coordinates
                xyz[0][i0],
                xyz[0][i1],
                xyz[0][i2],
                // Y-coordinates
                xyz[1][i0],
                xyz[1][i1],
                xyz[1][i2],
                // Data
                element_ux,
                element_uy,
                // Output
                element_vector);

        for (int edof_i = 0; edof_i < 3; ++edof_i) {
            const idx_t dof_i = ev[edof_i];
            values[dof_i] += element_vector[edof_i];
        }
    }

    return SFEM_SUCCESS;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    SFEM_TRACE_SCOPE("approxsdf.main");

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // if (argc != 4) {
    //     if (!rank) {
    //         fprintf(stderr, "usage: %s <trisurf> <mesh> <output_folder>\n", argv[0]);
    //     }

    //     return EXIT_FAILURE;
    // }

    // auto surf = argv[1];

    // // auto        trisurf       = sfem::Mesh::create_from_file(comm, argv[1]);
    // auto        mesh          = sfem::Mesh::create_from_file(comm, argv[2]);
    // const char *output_folder = argv[3];

    // int SFEM_ENABLE_ORACLE = 0;
    // SFEM_READ_ENV(SFEM_ENABLE_ORACLE, atoi);

    auto es   = sfem::EXECUTION_SPACE_HOST;
    auto blas = sfem::blas<real_t>(es);

    int         demo          = argc == 1 || strcmp(argv[1], "demo") == 0;
    std::string output_folder = "test_approxsdf";
    sfem::create_directory(output_folder.c_str());

    ptrdiff_t nx = 1000;
    // auto mesh = sfem::Mesh::create_tri3_square(comm, 4, 4, 0, 0, 1, 1);
    auto mesh = sfem::Mesh::create_tri3_square(comm, nx, nx, 0, 0, 1, 1);
    // auto mesh = sfem::Mesh::create_hex8_cube(comm, 10, 10, 10, 0, 0, 0, 1, 1, 1);

    mesh->write((output_folder + "/mesh").c_str());

    auto fs = sfem::FunctionSpace::create(mesh, 1);

    const ptrdiff_t nnodes = mesh->n_nodes();
    auto            points = mesh->points();
    const int       dim    = mesh->spatial_dimension();

    auto p = points->data();

    auto                distance = sfem::create_host_buffer<real_t>(nnodes);
    auto                normals  = sfem::create_host_buffer<real_t>(dim, nnodes);
    static const real_t infty    = 1000000000;
    blas->values(distance->size(), infty, distance->data());

    auto oracle = sfem::create_host_buffer<real_t>(nnodes);

    auto d = distance->data();
    auto n = normals->data();

    // Include mesh surface

    ptrdiff_t      n_surf_elements = 0;
    element_idx_t *parent          = 0;
    int16_t       *side_idx        = 0;

    if (extract_skin_sideset(mesh->n_elements(),
                             mesh->n_nodes(),
                             mesh->element_type(),
                             mesh->elements()->data(),
                             &n_surf_elements,
                             &parent,
                             &side_idx) != SFEM_SUCCESS) {
        SFEM_ERROR("Failed to extract skin!\n");
    }

    auto sideset = std::make_shared<sfem::Sideset>(
            comm, sfem::manage_host_buffer(n_surf_elements, parent), sfem::manage_host_buffer(n_surf_elements, side_idx));

    const auto st    = side_type(mesh->element_type());
    const int  nnxs  = elem_num_nodes(st);
    auto       sides = sfem::create_host_buffer<idx_t>(nnxs, sideset->parent()->size());

    if (extract_surface_from_sideset(fs->element_type(),
                                     mesh->elements()->data(),
                                     sideset->parent()->size(),
                                     sideset->parent()->data(),
                                     sideset->lfi()->data(),
                                     sides->data()) != SFEM_SUCCESS) {
        SFEM_ERROR("Unable to extract surface from sideset!\n");
    }

    compute_pseudo_normals(st, sides->extent(1), nnodes, sides->data(), p, n);

    auto            boundary_nodes = create_nodeset_from_sideset(fs, sideset);
    auto            bn             = boundary_nodes->data();
    const ptrdiff_t nbnodes        = boundary_nodes->size();

    for (ptrdiff_t i = 0; i < nbnodes; i++) {
        d[bn[i]] = 0;
    }

    auto boundary_surface = std::make_shared<sfem::Mesh>(mesh->spatial_dimension(),
                                                         st,
                                                         sides->extent(1),
                                                         sides->data(),
                                                         mesh->n_nodes(),
                                                         mesh->points()->data(),
                                                         [mesh, sides](void *) {});

    boundary_surface->write((output_folder + "/surface").c_str());
    normals->to_files((output_folder + "/surface/pseudo_normals.%d.raw").c_str());

    //
    auto cell_list = sfem::create_cell_list_from_nodes(mesh, MAX(2, 0.03*nx));
    cell_list->print(std::cout);

    // sfem::init_sdf_brute_force(boundary_surface, normals, 1, mesh, distance, normals);

    {

        SFEM_TRACE_SCOPE("sfem::init_sdf(boundary)");
        sfem::init_sdf(boundary_surface, normals, 1, cell_list, mesh,  distance, normals);

    }

    const geom_t c[3]   = {0.5, 0.5, 0.5};
    const geom_t radius = 0.3333;

    if (demo) {
        printf("Demo!!\n");
        // Toy setup to be replaced with mesh
        const geom_t dist_tol = 0.05;
        for (ptrdiff_t i = 0; i < nnodes; i++) {
            geom_t pdist = 0;
            geom_t vec[3];

            for (int d = 0; d < dim; d++) {
                auto dx = p[d][i] - c[d];
                pdist += dx * dx;
                vec[d] = dx;
            }

            pdist = radius - sqrt(pdist);

            oracle->data()[i] = pdist;

            // Avoid square-root
            if (fabs(pdist) < dist_tol && fabs(d[i]) > fabs(pdist)) {
                d[i] = pdist;

                auto neg = signbit(pdist);

                geom_t norm_vec = 0;
                for (int d = 0; d < dim; d++) {
                    auto dx = vec[d];
                    norm_vec += dx * dx;
                }

                assert(norm_vec != 0);

                norm_vec = sqrt(norm_vec);
                // norm_vec = neg ? -norm_vec : norm_vec;
                for (int d = 0; d < dim; d++) {
                    n[d][i] = vec[d] / norm_vec;
                }
            }
        }
    } else {
        const std::string surface_path    = argv[1];
        auto              surface         = sfem::Mesh::create_from_file(comm, surface_path.c_str());
        auto              surface_normals = sfem::create_host_buffer<real_t>(dim, surface->n_nodes());
        compute_pseudo_normals(surface->element_type(),
                               surface->n_elements(),
                               surface->n_nodes(),
                               surface->elements()->data(),
                               surface->points()->data(),
                               surface_normals->data());

        surface_normals->to_files((surface_path + "/normals.%d.raw").c_str());

        // sfem::init_sdf_brute_force(surface, surface_normals, -1, mesh, distance, normals);
        sfem::init_sdf(surface, surface_normals, -1, cell_list, mesh, distance, normals);

        // auto cell_list_structure = sfem::create_cell_list_from_nodes(surface);
        // cell_list_structure->print(std::cout);

        // auto cell_list_boundary = sfem::create_cell_list_from_nodes(boundary_surface);
        // cell_list_boundary->print(std::cout);
        // sfem::init_sdf(
        //     surface, surface_normals, cell_list, mesh, 0, distance, normals);
    }

    ptrdiff_t nconstraints = 0;
    for (ptrdiff_t i = 0; i < nnodes; i++) {
        nconstraints += (d[i] != infty);
    }

    auto      nodeset = sfem::create_host_buffer<idx_t>(nconstraints);
    ptrdiff_t offset  = 0;
    for (ptrdiff_t i = 0; i < nnodes; i++) {
        if (d[i] != infty) {
            nodeset->data()[offset++] = i;
        } else {
            d[i] = 0;
        }
    }

    distance->to_file((output_folder + "/input_distance.raw").c_str());
    normals->to_files((output_folder + "/input_normals.%d.raw").c_str());

    auto op = sfem::create_op(fs, "Laplacian", es);
    op->initialize();

    auto f = sfem::Function::create(fs);
    f->add_operator(op);

    sfem::DirichletConditions::Condition prescribed_normal{.nodeset = nodeset, .value = 0, .component = 0};
    auto                                 conds = sfem::create_dirichlet_conditions(fs, {prescribed_normal}, es);
    f->add_constraint(conds);

    auto linear_op     = sfem::create_linear_operator("CRS", f, nullptr, es);
    auto linear_solver = sfem::create_cg<real_t>(linear_op, es);

    auto rhs        = sfem::create_host_buffer<real_t>(fs->n_dofs());
    auto correction = sfem::create_host_buffer<real_t>(fs->n_dofs());

    for (int d = 0; d < dim; d++) {
        auto normal_comp = sfem::sub(normals, d);
        blas->zeros(rhs->size(), rhs->data());

        linear_op->apply(normal_comp->data(), rhs->data());
        blas->zeros(correction->size(), correction->data());

        f->apply_constraints(rhs->data());
        linear_solver->apply(rhs->data(), correction->data());
        blas->axpy(correction->size(), -1, correction->data(), normal_comp->data());
    }

    for (ptrdiff_t i = 0; i < nnodes; i++) {
        geom_t norm_vec = 0;
        for (int d = 0; d < dim; d++) {
            auto dx = n[d][i];
            norm_vec += dx * dx;
        }

        norm_vec = sqrt(norm_vec);

        for (int d = 0; d < dim; d++) {
            assert(norm_vec != 0);
            n[d][i] /= norm_vec;
        }
    }

    auto div = sfem::create_host_buffer<real_t>(fs->n_dofs());

    // TODO: divergence of normal field for different element types
    tri3_div_apply(
            mesh->n_elements(), mesh->n_nodes(), mesh->elements()->data(), mesh->points()->data(), n[0], n[1], div->data());

    // solve potential for the distance
    blas->zeros(rhs->size(), rhs->data());
    linear_op->apply(distance->data(), rhs->data());
    blas->axpy(rhs->size(), 1, div->data(), rhs->data());

    f->apply_constraints(rhs->data());

    blas->zeros(correction->size(), correction->data());
    linear_solver->apply(rhs->data(), correction->data());
    blas->axpy(correction->size(), -1, correction->data(), distance->data());

    // TODO: visualize divergence
    // ...
    distance->to_file((output_folder + "/distance.raw").c_str());
    normals->to_files((output_folder + "/normals.%d.raw").c_str());

    blas->zeros(correction->size(), correction->data());

    apply_inv_lumped_mass(mesh->element_type(),
                          mesh->n_elements(),
                          mesh->n_nodes(),
                          mesh->elements()->data(),
                          mesh->points()->data(),
                          div->data(),
                          correction->data());

    correction->to_file((output_folder + "/divergence.raw").c_str());

    // if (SFEM_ENABLE_ORACLE) {
    //     oracle->to_file((output_folder + "/oracle.raw").c_str());

    //     blas->axpy(oracle->size(), -1, oracle->data(), distance->data());
    //     distance->to_file((output_folder + "/difference.raw").c_str());
    // }

    return MPI_Finalize();
}
