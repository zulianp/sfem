#include "sfem_mesh.h"
#include "sfem_defs.h"

#include <math.h>
#include <stdio.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

void mesh_init(mesh_t *mesh) {
    mesh->comm      = MPI_COMM_NULL;
    mesh->mem_space = 0;

    mesh->spatial_dim  = 0;
    mesh->element_type = 0;

    mesh->nelements = 0;
    mesh->nnodes    = 0;

    mesh->elements = 0;
    mesh->points   = 0;

    mesh->n_owned_nodes             = 0;
    mesh->n_owned_nodes_with_ghosts = 0;

    mesh->n_owned_elements             = 0;
    mesh->n_owned_elements_with_ghosts = 0;
    mesh->n_shared_elements            = 0;

    mesh->node_mapping = 0;
    mesh->node_owner   = 0;

    mesh->element_mapping = 0;

    mesh->node_offsets = 0;
    mesh->ghosts       = 0;
}

void mesh_create_reference_hex8_cube(mesh_t *mesh) {
    mesh_init(mesh);
    mesh->comm         = MPI_COMM_SELF;
    mesh->spatial_dim  = 3;
    mesh->element_type = HEX8;
    mesh->nelements    = 1;
    mesh->elements     = malloc(8 * sizeof(idx_t *));
    mesh->nnodes       = 8;
    mesh->points       = malloc(3 * sizeof(geom_t *));

    for (int i = 0; i < 8; i++) {
        mesh->elements[i]    = malloc(sizeof(idx_t));
        mesh->elements[i][0] = i;
    }

    for (int d = 0; d < 3; d++) {
        mesh->points[d] = malloc(8 * sizeof(geom_t));
    }

    mesh->points[0][0] = 0;
    mesh->points[1][0] = 0;
    mesh->points[2][0] = 0;

    mesh->points[0][1] = 1;
    mesh->points[1][1] = 0;
    mesh->points[2][1] = 0;

    mesh->points[0][2] = 1;
    mesh->points[1][2] = 1;
    mesh->points[2][2] = 0;

    mesh->points[0][3] = 0;
    mesh->points[1][3] = 1;
    mesh->points[2][3] = 0;

    mesh->points[0][4] = 0;
    mesh->points[1][4] = 0;
    mesh->points[2][4] = 1;

    mesh->points[0][5] = 1;
    mesh->points[1][5] = 0;
    mesh->points[2][5] = 1;

    mesh->points[0][6] = 1;
    mesh->points[1][6] = 1;
    mesh->points[2][6] = 1;

    mesh->points[0][7] = 0;
    mesh->points[1][7] = 1;
    mesh->points[2][7] = 1;
}

void mesh_create_hex8_cube(mesh_t      *mesh,
                           const int    nx,
                           const int    ny,
                           const int    nz,
                           const geom_t xmin,
                           const geom_t ymin,
                           const geom_t zmin,
                           const geom_t xmax,
                           const geom_t ymax,
                           const geom_t zmax) {
    const ptrdiff_t nelements = nx * ny * nz;
    const ptrdiff_t nnodes    = (nx + 1) * (ny + 1) * (nz + 1);

    mesh_init(mesh);
    mesh->comm         = MPI_COMM_SELF;
    mesh->spatial_dim  = 3;
    mesh->element_type = HEX8;
    mesh->nelements    = nelements;
    mesh->elements     = malloc(8 * sizeof(idx_t *));
    mesh->nnodes       = nnodes;
    mesh->points       = malloc(3 * sizeof(geom_t *));

    for (int i = 0; i < 8; i++) {
        mesh->elements[i]    = malloc(nelements * sizeof(idx_t));
        mesh->elements[i][0] = i;
    }

    for (int d = 0; d < 3; d++) {
        mesh->points[d] = malloc(nnodes * sizeof(geom_t));
    }

    const ptrdiff_t ldz = (ny + 1) * (nx + 1);
    const ptrdiff_t ldy = nx + 1;
    const ptrdiff_t ldx = 1;

    const double hx = (xmax - xmin) * 1. / nx;
    const double hy = (ymax - ymin) * 1. / ny;
    const double hz = (zmax - zmin) * 1. / nz;

    assert(hx > 0);
    assert(hy > 0);
    assert(hz > 0);

    for (ptrdiff_t zi = 0; zi < nz; zi++) {
        for (ptrdiff_t yi = 0; yi < ny; yi++) {
            for (ptrdiff_t xi = 0; xi < nx; xi++) {
                const ptrdiff_t e = zi * ny * nx + yi * nx + xi;

                const idx_t i0 = (xi + 0) * ldx + (yi + 0) * ldy + (zi + 0) * ldz;
                const idx_t i1 = (xi + 1) * ldx + (yi + 0) * ldy + (zi + 0) * ldz;
                const idx_t i2 = (xi + 1) * ldx + (yi + 1) * ldy + (zi + 0) * ldz;
                const idx_t i3 = (xi + 0) * ldx + (yi + 1) * ldy + (zi + 0) * ldz;

                const idx_t i4 = (xi + 0) * ldx + (yi + 0) * ldy + (zi + 1) * ldz;
                const idx_t i5 = (xi + 1) * ldx + (yi + 0) * ldy + (zi + 1) * ldz;
                const idx_t i6 = (xi + 1) * ldx + (yi + 1) * ldy + (zi + 1) * ldz;
                const idx_t i7 = (xi + 0) * ldx + (yi + 1) * ldy + (zi + 1) * ldz;

                mesh->elements[0][e] = i0;
                mesh->elements[1][e] = i1;
                mesh->elements[2][e] = i2;
                mesh->elements[3][e] = i3;

                mesh->elements[4][e] = i4;
                mesh->elements[5][e] = i5;
                mesh->elements[6][e] = i6;
                mesh->elements[7][e] = i7;
            }
        }
    }

    for (ptrdiff_t zi = 0; zi <= nz; zi++) {
        for (ptrdiff_t yi = 0; yi <= ny; yi++) {
            for (ptrdiff_t xi = 0; xi <= nx; xi++) {
                ptrdiff_t node        = xi * ldx + yi * ldy + zi * ldz;
                mesh->points[0][node] = (double)xmin + xi * hx;
                mesh->points[1][node] = (double)ymin + yi * hy;
                mesh->points[2][node] = (double)zmin + zi * hz;
            }
        }
    }
}

void mesh_create_tri3_square(mesh_t      *mesh,
                             const int    nx,
                             const int    ny,
                             const geom_t xmin,
                             const geom_t ymin,
                             const geom_t xmax,
                             const geom_t ymax) {
    const ptrdiff_t nelements = 2 * nx * ny;
    const ptrdiff_t nnodes    = (nx + 1) * (ny + 1);

    mesh_init(mesh);
    mesh->comm         = MPI_COMM_SELF;
    mesh->spatial_dim  = 2;
    mesh->element_type = TRI3;
    mesh->nelements    = nelements;
    mesh->elements     = malloc(3 * sizeof(idx_t *));
    mesh->nnodes       = nnodes;
    mesh->points       = malloc(2 * sizeof(geom_t *));

    for (int i = 0; i < 3; i++) {
        mesh->elements[i]    = malloc(nelements * sizeof(idx_t));
        mesh->elements[i][0] = i;
    }

    for (int d = 0; d < 2; d++) {
        mesh->points[d] = malloc(nnodes * sizeof(geom_t));
    }

    const ptrdiff_t ldy = nx + 1;
    const ptrdiff_t ldx = 1;

    const double hx = (xmax - xmin) * 1. / nx;
    const double hy = (ymax - ymin) * 1. / ny;

    assert(hx > 0);
    assert(hy > 0);

    for (ptrdiff_t yi = 0; yi < ny; yi++) {
        for (ptrdiff_t xi = 0; xi < nx; xi++) {
            const ptrdiff_t e = 2 * (yi * nx + xi);

            const idx_t i0 = (xi + 0) * ldx + (yi + 0) * ldy;
            const idx_t i1 = (xi + 1) * ldx + (yi + 0) * ldy;
            const idx_t i2 = (xi + 1) * ldx + (yi + 1) * ldy;
            const idx_t i3 = (xi + 0) * ldx + (yi + 1) * ldy;

            mesh->elements[0][e] = i0;
            mesh->elements[1][e] = i1;
            mesh->elements[2][e] = i3;

            mesh->elements[0][e + 1] = i1;
            mesh->elements[1][e + 1] = i2;
            mesh->elements[2][e + 1] = i3;
        }
    }

    for (ptrdiff_t yi = 0; yi <= ny; yi++) {
        for (ptrdiff_t xi = 0; xi <= nx; xi++) {
            ptrdiff_t node        = xi * ldx + yi * ldy;
            mesh->points[0][node] = (double)xmin + xi * hx;
            mesh->points[1][node] = (double)ymin + yi * hy;
        }
    }
}

void mesh_create_quad4_square(mesh_t      *mesh,
                              const int    nx,
                              const int    ny,
                              const geom_t xmin,
                              const geom_t ymin,
                              const geom_t xmax,
                              const geom_t ymax) {
    const ptrdiff_t nelements = nx * ny;
    const ptrdiff_t nnodes    = (nx + 1) * (ny + 1);

    mesh_init(mesh);
    mesh->comm         = MPI_COMM_SELF;
    mesh->spatial_dim  = 2;
    mesh->element_type = QUAD4;
    mesh->nelements    = nelements;
    mesh->elements     = malloc(4 * sizeof(idx_t *));
    mesh->nnodes       = nnodes;
    mesh->points       = malloc(2 * sizeof(geom_t *));

    for (int i = 0; i < 4; i++) {
        mesh->elements[i]    = malloc(nelements * sizeof(idx_t));
        mesh->elements[i][0] = i;
    }

    for (int d = 0; d < 2; d++) {
        mesh->points[d] = malloc(nnodes * sizeof(geom_t));
    }

    const ptrdiff_t ldy = nx + 1;
    const ptrdiff_t ldx = 1;

    const double hx = (xmax - xmin) * 1. / nx;
    const double hy = (ymax - ymin) * 1. / ny;

    assert(hx > 0);
    assert(hy > 0);

    for (ptrdiff_t yi = 0; yi < ny; yi++) {
        for (ptrdiff_t xi = 0; xi < nx; xi++) {
            const ptrdiff_t e = yi * nx + xi;

            const idx_t i0 = (xi + 0) * ldx + (yi + 0) * ldy;
            const idx_t i1 = (xi + 1) * ldx + (yi + 0) * ldy;
            const idx_t i2 = (xi + 1) * ldx + (yi + 1) * ldy;
            const idx_t i3 = (xi + 0) * ldx + (yi + 1) * ldy;

            mesh->elements[0][e] = i0;
            mesh->elements[1][e] = i1;
            mesh->elements[2][e] = i2;
            mesh->elements[3][e] = i3;
        }
    }

    for (ptrdiff_t yi = 0; yi <= ny; yi++) {
        for (ptrdiff_t xi = 0; xi <= nx; xi++) {
            ptrdiff_t node        = xi * ldx + yi * ldy;
            mesh->points[0][node] = (double)xmin + xi * hx;
            mesh->points[1][node] = (double)ymin + yi * hy;
        }
    }
}

void mesh_create_serial(mesh_t       *mesh,
                        int           spatial_dim,
                        enum ElemType element_type,
                        ptrdiff_t     nelements,
                        idx_t       **elements,
                        ptrdiff_t     nnodes,
                        geom_t      **points) {
    mesh_init(mesh);
    mesh->comm         = MPI_COMM_SELF;
    mesh->spatial_dim  = spatial_dim;
    mesh->element_type = element_type;
    mesh->nelements    = nelements;
    mesh->elements     = elements;
    mesh->nnodes       = nnodes;
    mesh->points       = points;
}

void mesh_minmax_edge_length(const mesh_t *const mesh, real_t *emin, real_t *emax) {
    const int nnxe = elem_num_nodes(mesh->element_type);
    *emin          = 1e10;
    *emax          = 0;

    const int sdim = mesh->spatial_dim;

#pragma omp parallel
    {
#pragma omp for
        for (ptrdiff_t e = 0; e < mesh->nelements; e++) {
            real_t len_min = 1e10;
            real_t len_max = 0;
            for (int i = 0; i < nnxe; i++) {
                idx_t node_i = mesh->elements[i][e];

                for (int j = i + 1; j < nnxe; j++) {
                    idx_t node_j = mesh->elements[j][e];

                    real_t len = 0;
                    for (int d = 0; d < sdim; d++) {
                        real_t diff = mesh->points[d][node_i] - mesh->points[d][node_j];
                        len += diff * diff;
                    }

                    len = sqrt(len);

                    len_min = MIN(len_min, len);
                    len_max = MAX(len_max, len);
                }
            }

#pragma omp critical
            {
                *emin = MIN(*emin, len_min);
                *emax = MAX(*emax, len_max);
            }
        }
    }

    int size;
    MPI_Comm_size(mesh->comm, &size);

    if (size > 1) {
        SFEM_ERROR("IMPLEMENT ME");
    }
}

void mesh_destroy(mesh_t *mesh) {
    const int nxe = elem_num_nodes(mesh->element_type);
    for (int d = 0; d < nxe; ++d) {
        free(mesh->elements[d]);
        mesh->elements[d] = 0;
    }
    if (nxe) {
        free(mesh->elements);
    }

    if (mesh->points) {
        for (int d = 0; d < mesh->spatial_dim; ++d) {
            free(mesh->points[d]);
            mesh->points[d] = 0;
        }

        free(mesh->points);
    }

    free(mesh->node_mapping);
    free(mesh->node_owner);
    free(mesh->element_mapping);

    mesh->comm      = MPI_COMM_NULL;
    mesh->mem_space = SFEM_MEM_SPACE_NONE;

    mesh->spatial_dim  = 0;
    mesh->element_type = 0;

    mesh->nelements = 0;
    mesh->nnodes    = 0;

    if (mesh->ghosts) {
        free(mesh->ghosts);
    }

    if (mesh->node_offsets) {
        free(mesh->node_offsets);
    }
}

void mesh_create_shared_elements_block(mesh_t *mesh, element_block_t *block) {
    //
    block->nelements = mesh->n_shared_elements;
    const int nn     = elem_num_nodes(mesh->element_type);

    block->elements = (idx_t **)malloc(nn * sizeof(idx_t *));

    for (int i = 0; i < nn; i++) {
        block->elements[i] = &mesh->elements[i][mesh->n_owned_elements];
    }
}

void mesh_destroy_shared_elements_block(mesh_t *mesh, element_block_t *block) {
    free(block->elements);
    block->elements = 0;
    block->elements = 0;
}

#include <stdio.h>

void remap_elements_to_contiguous_index(const ptrdiff_t  n_elements,
                                        const int        nxe,
                                        idx_t **const    elements,
                                        ptrdiff_t *const out_n_contiguous,
                                        idx_t **const    out_node_mapping) {
    idx_t n = 0;
    for (int d = 0; d < nxe; d++) {
        for (ptrdiff_t i = 0; i < n_elements; i++) {
            n = MAX(elements[d][i], n);
        }
    }

    n += 1;

    idx_t *remap = (idx_t *)malloc(n * sizeof(idx_t));
    for (ptrdiff_t i = 0; i < n; ++i) {
        remap[i] = SFEM_IDX_INVALID;
    }

    ptrdiff_t n_contiguous = 0;
    for (ptrdiff_t i = 0; i < n_elements; ++i) {
        for (int d = 0; d < nxe; ++d) {
            idx_t idx = elements[d][i];
            if (remap[idx] < 0) {
                remap[idx] = n_contiguous++;
            }
        }
    }

    for (int d = 0; d < nxe; d++) {
        for (ptrdiff_t i = 0; i < n_elements; i++) {
            elements[d][i] = remap[elements[d][i]];
        }
    }

    idx_t *node_mapping = malloc(n_contiguous * sizeof(idx_t));
    for (ptrdiff_t i = 0; i < n; ++i) {
        if (remap[i] != SFEM_IDX_INVALID) {
            node_mapping[remap[i]] = i;
        }
    }

    free(remap);
    *out_n_contiguous = n_contiguous;
    *out_node_mapping = node_mapping;
}
