#include <memory>

#include "sfem_test.h"

#include "sfem_Function.hpp"

#include "sfem_Buffer.hpp"
#include "sfem_base.h"
#include "sfem_crs_SpMV.hpp"
#include "spmv.h"

#include "matrixio_array.h"

#include "sfem_API.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_Function_incore_cuda.hpp"
#include "sfem_cuda_blas.h"
#include "sfem_cuda_solver.hpp"
#endif

#include "sfem_ssmgc.hpp"

// Strategy:
// 1) Create mesh dual graph (adj table)
// 2) Create surface sideset (parent and side_idx)
// 3) Broad-phase detection using cell list (cell fully contains largest element, cell-by-cell we check all pairs surrounding
// lattice)
// 4) Build contact pairs (discard using normal orientation, discard connected elements using dual graph)
//    BASELINE: a) Node to surface
//    MORTAR:   b) Surface-to-surface
//              c) Edge-to-surface
//              d) Node-to-edge
//              e) Node-to-Node (?)

using namespace sfem;

static const geom_t pos_infty = 100000;
static const geom_t neg_infty = -100000;
static const geom_t eps       = 1e-4;

static SFEM_INLINE void normalize(real_t *const vec3) {
    const real_t len = sqrt(vec3[0] * vec3[0] + vec3[1] * vec3[1] + vec3[2] * vec3[2]);
    vec3[0] /= len;
    vec3[1] /= len;
    vec3[2] /= len;
}

static SFEM_INLINE void normal(const idx_t                  i0,
                               const idx_t                  i1,
                               const idx_t                  i2,
                               geom_t **const SFEM_RESTRICT points,
                               real_t *const SFEM_RESTRICT  n) {
    real_t u[3] = {points[0][i1] - points[0][i0], points[1][i1] - points[1][i0], points[2][i1] - points[2][i0]};
    real_t v[3] = {points[0][i2] - points[0][i0], points[1][i2] - points[1][i0], points[2][i2] - points[2][i0]};

    normalize(u);
    normalize(v);

    n[0] = u[1] * v[2] - u[2] * v[1];
    n[1] = u[2] * v[0] - u[0] * v[2];
    n[2] = u[0] * v[1] - u[1] * v[0];

    normalize(n);
}

class SelfContactSurface {
public:
    std::shared_ptr<sfem::Buffer<element_idx_t>> table;
    std::shared_ptr<sfem::Sideset>               sideset;

    static std::shared_ptr<SelfContactSurface> create(const std::shared_ptr<sfem::Mesh> &mesh) {
        SFEM_TRACE_SCOPE("SelfContactSurface::create");

        enum ElemType element_type_hack = mesh->element_type();
        switch (element_type_hack) {
            case TRI6: {
                element_type_hack = TRI3;
                break;
            }
            case TET10: {
                element_type_hack = TET4;
                break;
            }
            case EDGE3: {
                element_type_hack = EDGE2;
                break;
            }
            default:
                break;
        }

        const ptrdiff_t n_elements = mesh->n_elements();
        const ptrdiff_t n_nodes    = mesh->n_nodes();

        auto elements = mesh->elements()->data();
        auto points   = mesh->points()->data();

        enum ElemType st   = side_type(element_type_hack);
        const int     nnxs = elem_num_nodes(st);
        const int     dim  = mesh->spatial_dimension();
        const int     ns   = elem_num_sides(element_type_hack);
        const int     nsxe = elem_num_sides(mesh->element_type());

        ptrdiff_t      n_surf_elements{0};
        element_idx_t *parent_buff{nullptr};
        int16_t       *side_idx_buff{nullptr};
        element_idx_t *table_buff{nullptr};

        create_element_adj_table(n_elements, n_nodes, element_type_hack, elements, &table_buff);

        if (extract_sideset_from_adj_table(
                    element_type_hack, n_elements, table_buff, &n_surf_elements, &parent_buff, &side_idx_buff) != SFEM_SUCCESS) {
            SFEM_ERROR("Failed to extract extract_sideset_from_adj_table!\n");
        }

        auto ret     = std::make_shared<SelfContactSurface>();
        ret->sideset = sfem::Sideset::create(mesh->comm(),
                                             sfem::manage_host_buffer<element_idx_t>(n_surf_elements, parent_buff),
                                             sfem::manage_host_buffer<int16_t>(n_surf_elements, side_idx_buff));

        ret->table = sfem::manage_host_buffer<element_idx_t>(n_surf_elements * nsxe, table_buff);
        return ret;
    }
};

class CellList {
public:
    int       dim{0};
    ptrdiff_t n[3]{1, 1, 1};
    ptrdiff_t stride[3] = {1, 1, 1};
    geom_t    o[3]      = {0, 0, 0};
    geom_t    delta[3]  = {0, 0, 0};
    geom_t    radius{0};

    std::shared_ptr<Buffer<ptrdiff_t>> cell_ptr;
    std::shared_ptr<Buffer<ptrdiff_t>> cell_idx;

    static std::shared_ptr<CellList> create(const int                          dim,
                                            const ptrdiff_t *const             n,
                                            const ptrdiff_t *const             stride,
                                            const geom_t *const                o,
                                            const geom_t *const                delta,
                                            const geom_t                       radius,
                                            std::shared_ptr<Buffer<ptrdiff_t>> cell_ptr,
                                            std::shared_ptr<Buffer<ptrdiff_t>> cell_idx) {
        auto ret = std::make_shared<CellList>();
        ret->dim = dim;
        memcpy(ret->n, n, dim * sizeof(ptrdiff_t));
        memcpy(ret->stride, stride, dim * sizeof(ptrdiff_t));
        memcpy(ret->o, o, dim * sizeof(geom_t));
        memcpy(ret->delta, delta, dim * sizeof(geom_t));
        ret->radius   = radius;
        ret->cell_ptr = cell_ptr;
        ret->cell_idx = cell_idx;
        return ret;
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

static SFEM_INLINE ptrdiff_t cell_list_idx(const int                            dim,
                                           const ptrdiff_t *const SFEM_RESTRICT stride,
                                           const geom_t *const SFEM_RESTRICT    o,
                                           const geom_t *const SFEM_RESTRICT    delta,
                                           const geom_t *const SFEM_RESTRICT    p) {
    ptrdiff_t idx = 0;
    for (int d = 0; d < dim; d++) {
        geom_t val = p[d] - o[d];
        val /= delta[d];
        idx += floor(val) * stride[d];
    }

    return idx;
}

std::shared_ptr<CellList> create_cell_list_from_sideset(const std::shared_ptr<sfem::Mesh>    &mesh,
                                                        const std::shared_ptr<sfem::Sideset> &sideset,
                                                        const geom_t                          radius_factor = 1) {
    SFEM_TRACE_SCOPE("create_cell_list_from_sideset");

    const ptrdiff_t nnodes = mesh->n_nodes();

    const ptrdiff_t nelements = mesh->n_elements();
    const int       dim       = mesh->spatial_dimension();
    const int       nxe       = elem_num_nodes(mesh->element_type());

    auto points   = mesh->points()->data();
    auto elements = mesh->elements()->data();

    const ptrdiff_t nsides   = sideset->size();
    auto            aabb_min = sfem::create_host_buffer<geom_t>(dim, nsides);
    auto            aabb_max = sfem::create_host_buffer<geom_t>(dim, nsides);

    const int           nsxe             = elem_num_sides(mesh->element_type());
    const enum ElemType st               = side_type(mesh->element_type());
    const int           nnxs             = elem_num_nodes(st);
    auto                local_side_table = sfem::create_host_buffer<int>(nsxe * nnxs);
    fill_local_side_table(mesh->element_type(), local_side_table->data());

    auto bmin   = aabb_min->data();
    auto bmax   = aabb_max->data();
    auto parent = sideset->parent()->data();
    auto lfi    = sideset->lfi()->data();
    auto lst    = local_side_table->data();

    geom_t min[3]    = {pos_infty, pos_infty, pos_infty};
    geom_t max[3]    = {neg_infty, neg_infty, neg_infty};
    geom_t radius[3] = {0, 0, 0};

    for (int d = 0; d < dim; d++) {
#pragma omp parallel for
        for (ptrdiff_t s = 0; s < nsides; s++) {
            bmin[d][s] = pos_infty;
            bmax[d][s] = neg_infty;
        }

        const geom_t *const x = points[d];

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < nsides; i++) {
            const element_idx_t e    = parent[i];
            const int16_t       side = lfi[i];

            geom_t lmin = pos_infty;
            geom_t lmax = neg_infty;
            for (int v = 0; v < nnxs; v++) {
                const idx_t idx = elements[lst[side * nnxs + v]][e];
                lmin            = MIN(x[idx], lmin);
                lmax            = MAX(x[idx], lmax);
            }

            assert(lmax >= lmin);

            bmin[d][i] = lmin;
            bmax[d][i] = lmax;
        }
    }

    for (int d = 0; d < dim; d++) {
        for (ptrdiff_t i = 0; i < nsides; i++) {
            min[d]            = MIN(min[d], (bmin[d][i] - eps));
            max[d]            = MAX(max[d], (bmax[d][i] + eps));
            const geom_t diff = fabs(bmax[d][i] - bmin[d][i]);

            assert(bmin[d][i] > min[d]);
            assert(bmax[d][i] < max[d]);

            // Diagonal of bounding box
            radius[d] = MAX(radius[d], diff);
        }
    }

    geom_t max_radius = radius[0];
    for (int d = 1; d < dim; d++) {
        max_radius = MAX(radius[d], max_radius);
    }

    assert(max_radius != 0);
    max_radius *= radius_factor;

    ptrdiff_t n[3];
    ptrdiff_t stride[3];
    geom_t    o[3];
    geom_t    delta[3];

    ptrdiff_t ncells = 1;
    for (int d = 0; d < dim; d++) {
        geom_t extent = (max[d] - min[d]);
        n[d]          = extent / max_radius;
        delta[d]      = extent / n[d];
        o[d]          = min[d];

        printf("%d) %ld %f\n", d, n[d], delta[d]);
        ncells *= n[d];
    }

    stride[0] = 1;
    stride[1] = 1;
    stride[2] = 1;
    for (int d = 1; d < dim; d++) {
        stride[d] = stride[d - 1] * n[d - 1];
    }

    printf("nelements %ld\n", nsides);
    printf("ncells: %ld x %ld x %ld = %ld\n", n[0], n[1], n[2], ncells);

    auto cell_ptr = sfem::create_host_buffer<ptrdiff_t>(ncells + 1);
    auto cp       = cell_ptr->data();

    // #pragma omp parallel for
    for (ptrdiff_t i = 0; i < nsides; i++) {
        geom_t point[3];
        for (int d = 0; d < dim; d++) {
            point[d] = bmin[d][i];
            assert(bmin[d][i] > min[d]);
            assert(bmax[d][i] < max[d]);
        }

        const ptrdiff_t idx = cell_list_idx(dim, stride, o, delta, point);
        // #pragma atomic update
        cp[idx + 1]++;
    }

    for (ptrdiff_t i = 0; i < ncells; i++) {
        cp[i + 1] += cp[i];
    }

    auto cell_idx = sfem::create_host_buffer<ptrdiff_t>(cp[ncells]);
    auto ci       = cell_idx->data();

    auto bookeepping = sfem::create_host_buffer<ptrdiff_t>(ncells);
    auto bk          = bookeepping->data();

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nsides; i++) {
        geom_t point[3];
        for (int d = 0; d < dim; d++) {
            point[d] = bmin[d][i];
        }

        const ptrdiff_t idx = cell_list_idx(dim, stride, o, delta, point);

        {
            ptrdiff_t offset;
#pragma atomic update
            offset               = bk[idx]++;
            ci[cp[idx] + offset] = i;
        }
    }

    return CellList::create(dim, n, stride, o, delta, max_radius, cell_ptr, cell_idx);
}

int self_contact(sfem::Context &context, int argc, char *argv[]) {
    SFEM_TRACE_SCOPE("self_contact");

    auto comm = context.comm();

    if (argc != 3) {
        fprintf(stderr, "usage: %s <mesh> <output>\n", argv[0]);
        return SFEM_FAILURE;
    }

    const char *mesh_path   = argv[1];
    std::string output_path = argv[2];

    auto mesh = sfem::Mesh::create_from_file(comm, mesh_path);

    const int dim = mesh->spatial_dimension();
    const int nxe = elem_num_nodes(mesh->element_type());

    auto surface   = SelfContactSurface::create(mesh);
    auto cell_list = create_cell_list_from_sideset(mesh, surface->sideset);

    const ptrdiff_t ncells   = cell_list->cell_ptr->size() - 1;
    auto            cell_ptr = cell_list->cell_ptr->data();
    auto            cell_idx = cell_list->cell_idx->data();

    ptrdiff_t ncandidates = 0;
    ptrdiff_t skipped     = 0;

    {
        SFEM_TRACE_SCOPE("detect");

        const int           nsxe = elem_num_sides(mesh->element_type());
        const enum ElemType st   = side_type(mesh->element_type());
        const int           nnxs = elem_num_nodes(st);

        auto parent   = surface->sideset->parent()->data();
        auto lfi      = surface->sideset->lfi()->data();
        auto elements = mesh->elements()->data();
        auto points   = mesh->points()->data();

        auto local_side_table = sfem::create_host_buffer<int>(nsxe * nnxs);
        fill_local_side_table(mesh->element_type(), local_side_table->data());
        auto lst = local_side_table->data();
        auto adj = surface->table->data();

        const real_t cos_angle_threshold = 0;
        printf("cos_angle_threshold = %g\n", cos_angle_threshold);

        for (ptrdiff_t c = 0; c < ncells; c++) {
            // For every cell
            for (ptrdiff_t k = cell_ptr[c]; k < cell_ptr[c + 1]; k++) {
                const ptrdiff_t     idx = cell_idx[k];
                const element_idx_t sp  = parent[idx];
                const int16_t       s   = lfi[idx];

                geom_t lmin[3] = {pos_infty, pos_infty, pos_infty};
                geom_t lmax[3] = {neg_infty, neg_infty, neg_infty};

                for (int d = 0; d < dim; d++) {
                    for (int v = 0; v < nnxs; v++) {
                        const idx_t idx = elements[lst[s * nnxs + v]][sp];
                        lmin[d]         = MIN(points[d][idx], lmin[d]);
                        lmax[d]         = MAX(points[d][idx], lmax[d]);
                    }
                }

                real_t side_normal[3] = {0, 0, 0};

                normal(elements[lst[s * nnxs + 0]][sp],
                       elements[lst[s * nnxs + 1]][sp],
                       elements[lst[s * nnxs + 2]][sp],
                       points,
                       side_normal);

                ptrdiff_t start_coord[3] = {1, 1, 0};
                ptrdiff_t end_coord[3]   = {1, 1, 1};

                cell_list_coords(dim, cell_list->n, cell_list->o, cell_list->delta, lmin, start_coord);
                cell_list_coords(dim, cell_list->n, cell_list->o, cell_list->delta, lmax, end_coord);

                for (int d = 0; d < dim; d++) {
                    const ptrdiff_t center = start_coord[d];
                    start_coord[d]         = MAX(start_coord[d] - 1, 0);
                    end_coord[d]           = MIN(end_coord[d] + 1, cell_list->n[d]);
                }

                // TODO Construct:
                // - Local interpolation matrix
                // - Local gaps
                // - Local normals

                for (ptrdiff_t zi = start_coord[2]; zi < end_coord[2]; zi++) {
                    for (ptrdiff_t yi = start_coord[1]; yi < end_coord[1]; yi++) {
                        for (ptrdiff_t xi = start_coord[0]; xi < end_coord[0]; xi++) {
                            const ptrdiff_t other_c =
                                    xi * cell_list->stride[0] + yi * cell_list->stride[1] + zi * cell_list->stride[2];

                            for (ptrdiff_t other_k = cell_ptr[other_c]; other_k < cell_ptr[other_c + 1]; other_k++) {
                                const ptrdiff_t     other_idx = cell_idx[other_k];
                                const element_idx_t other_sp  = parent[other_idx];
                                const int16_t       other_s   = lfi[other_idx];

                                // Check every neighboring cell including this one
                                // (Discard using normal orientation, Discard connected elements using dual graph)

                                if (other_sp == sp) {
                                    skipped++;
                                    continue;
                                }

                                bool skip = false;
                                for (int i = 0; i < nnxs; i++) {
                                    if (adj[sp * nsxe + i] == other_sp) {
                                        skipped++;
                                        skip = true;
                                    }
                                }

                                if (skip) continue;

                                // Check normals
                                real_t other_side_normal[3] = {0, 0, 0};

                                normal(elements[lst[other_s * nnxs + 0]][other_sp],
                                       elements[lst[other_s * nnxs + 1]][other_sp],
                                       elements[lst[other_s * nnxs + 2]][other_sp],
                                       points,
                                       other_side_normal);

                                real_t cos_angle = 0;
                                for (int d = 0; d < dim; d++) {
                                    cos_angle += side_normal[d] * other_side_normal[d];
                                }

                                if (cos_angle >= cos_angle_threshold) {
                                    skipped++;
                                    continue;
                                }

                                // Check if element is in prescribed radius?

                                ncandidates++;

                                // Sample distance?
                            }
                        }
                    }
                }
            }
        }
    }

    printf("#nelements %ld #nnodes %ld #nsides %ld #candidates %ld #skipped %ld\n",
           mesh->n_elements(),
           mesh->n_nodes(),
           surface->sideset->parent()->size(),
           ncandidates,
           skipped);

    return SFEM_SUCCESS;
}

int main(int argc, char *argv[]) {
    sfem::Context context(argc, argv);
    return self_contact(context, argc, argv);
}
