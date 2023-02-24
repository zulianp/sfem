#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matrix.io/array_dtof.h"
#include "matrix.io/matrixio_array.h"
#include "matrix.io/matrixio_crs.h"
#include "matrix.io/utils.h"

#include "crs_graph.h"
#include "read_mesh.h"
#include "sfem_base.h"
#include "sfem_mesh_write.h"

typedef struct {
    ptrdiff_t n_cells;
    ptrdiff_t n_entries;
    ptrdiff_t *cell_ptr;
    ptrdiff_t *idx;
    geom_t shift;
    geom_t scaling;
} cell_list_1D_t;

void cell_list_1D_print(cell_list_1D_t *cl) {
    printf("shift: %g\n", (double)cl->shift);
    printf("scaling: %g\n", (double)cl->scaling);
    printf("n_cells: %ld\n", (long)cl->n_cells);
    printf("n_entries: %ld\n", (long)cl->n_entries);

    printf("---------------------\n");
    for (ptrdiff_t i = 0; i < cl->n_cells; i++) {
        ptrdiff_t begin = cl->cell_ptr[i];
        ptrdiff_t end = cl->cell_ptr[i + 1];

        assert(end <= cl->n_entries);

        printf("%ld)\n", (long)i);
        for (ptrdiff_t k = begin; k < end; k++) {
            printf("%ld ", (long)cl->idx[k]);
        }
        printf("\n");
    }

    printf("---------------------\n");

    printf("cell_ptr:\n");
    for (ptrdiff_t i = 0; i < cl->n_cells + 1; i++) {
        printf("%ld ", (long)cl->cell_ptr[i]);
    }
    printf("\n");

    printf("---------------------\n");

    printf("idx:\n");
    for (ptrdiff_t i = 0; i < cl->n_entries; i++) {
        printf("%ld ", (long)cl->idx[i]);
    }
    printf("\n");

    printf("---------------------\n");

    fflush(stdout);
}

static void histogram(const ptrdiff_t nnodes,
                      const geom_t *SFEM_RESTRICT x,
                      const geom_t shift,
                      const geom_t scaling,
                      const ptrdiff_t n_cells,
                      ptrdiff_t *SFEM_RESTRICT histo) {
    memset(histo, 0, n_cells * sizeof(ptrdiff_t));
    for (ptrdiff_t i = 0; i < nnodes; ++i) {
        ptrdiff_t idx = scaling * (x[i] + shift);
        histo[idx] += 1;
    }
}

static void bounding_intervals(const ptrdiff_t n_elements,
                               const int n_nodes_per_elem,
                               idx_t **const SFEM_RESTRICT elems,
                               const geom_t *const SFEM_RESTRICT x,
                               geom_t *const SFEM_RESTRICT bi_min,
                               geom_t *const SFEM_RESTRICT bi_max) {
    for (ptrdiff_t e = 0; e < n_elements; e++) {
        const idx_t i = elems[0][e];
        bi_min[e] = x[i];
        bi_max[e] = x[i];
    }

    for (int d = 1; d < n_nodes_per_elem; ++d) {
        const idx_t *idx = elems[d];

        for (ptrdiff_t e = 0; e < n_elements; e++) {
            const idx_t i = idx[e];
            bi_min[e] = MIN(bi_min[e], x[i]);
            bi_max[e] = MAX(bi_max[e], x[i]);
        }
    }
}

static SFEM_INLINE geom_t array_min(const ptrdiff_t n, const geom_t *a) {
    geom_t ret = a[0];
    for (ptrdiff_t i = 1; i < n; ++i) {
        ret = MIN(ret, a[i]);
    }

    return ret;
}

static SFEM_INLINE geom_t array_max(const ptrdiff_t n, const geom_t *const SFEM_RESTRICT a) {
    geom_t ret = a[0];
    for (ptrdiff_t i = 1; i < n; ++i) {
        ret = MAX(ret, a[i]);
    }

    return ret;
}

static SFEM_INLINE geom_t array_max_range(const ptrdiff_t n,
                                          const geom_t *const SFEM_RESTRICT start,
                                          const geom_t *const SFEM_RESTRICT end) {
    geom_t ret = end[0] - start[0];
    for (ptrdiff_t i = 1; i < n; ++i) {
        const geom_t val = end[i] - start[i];
        ret = MAX(ret, val);
    }

    return ret;
}

typedef struct {
    ptrdiff_t begin;
    ptrdiff_t end;
} cell_list_1D_query_t;

static SFEM_INLINE cell_list_1D_query_t cell_list_1D_query(const cell_list_1D_t *const cl,
                                                           const geom_t bi_min,
                                                           const geom_t bi_max) {
    cell_list_1D_query_t ret;
    ret.begin = MAX(0, floor(cl->scaling * (bi_min + cl->shift)));
    ret.end = MIN(ceil(cl->scaling * (bi_max + cl->shift)) + 1, cl->n_cells);
    return ret;
}

void cell_list_1D_create(cell_list_1D_t *const cl,
                         const ptrdiff_t n,
                         const geom_t *const SFEM_RESTRICT bi_min,
                         const geom_t *const SFEM_RESTRICT bi_max) {
    const geom_t x_min = array_min(n, bi_min);
    const geom_t x_max = array_max(n, bi_max);
    const geom_t x_range = x_max - x_min;
    const geom_t max_cell_range = array_max_range(n, bi_min, bi_max);

    // Make sure any interval overlaps with max 2 cells
    ptrdiff_t n_cells = floor(x_range / max_cell_range);
    const geom_t scaling = n_cells / x_range;
    const geom_t shift = -x_min;

    ptrdiff_t *zhisto = malloc(n_cells * sizeof(ptrdiff_t));
    histogram(n, bi_min, shift, scaling, n_cells, zhisto);

    ptrdiff_t *cell_ptr = malloc((n_cells + 1) * sizeof(ptrdiff_t));

    cell_ptr[0] = 0;
    for (ptrdiff_t i = 0; i < n_cells; i++) {
        cell_ptr[i + 1] = cell_ptr[i] + zhisto[i];
    }

    const ptrdiff_t n_entries = cell_ptr[n_cells];
    ptrdiff_t *idx = malloc((n_entries) * sizeof(ptrdiff_t));
    memset(zhisto, 0, n_cells * sizeof(ptrdiff_t));

    // Fill cell-list
    for (ptrdiff_t i = 0; i < n; i++) {
        ptrdiff_t cell = (bi_min[i] + shift) * scaling;
        idx[cell_ptr[cell] + zhisto[cell]] = i;
        zhisto[cell]++;
    }

#ifndef NDEBUG
    for (ptrdiff_t i = 0; i < n_cells; i++) {
        assert(cell_ptr[i + 1] == cell_ptr[i] + zhisto[i]);
    }
#endif

    cl->cell_ptr = cell_ptr;
    cl->idx = idx;
    cl->shift = shift;
    cl->scaling = scaling;
    cl->n_cells = n_cells;
    cl->n_entries = n_entries;

    cell_list_1D_print(cl);
    free(zhisto);
}

void cell_list_1D_destroy(cell_list_1D_t *const cl) {
    free(cl->cell_ptr);
    free(cl->idx);

    cl->shift = 0;
    cl->scaling = 0;
    cl->n_cells = 0;
}

typedef struct {
    int size;
    real_t *x;
    real_t *y;
    real_t *z;
    real_t *w;
} quadrature_t;

void quadrature_create(quadrature_t *const q, const int size) {
    q->size = size;
    q->x = (real_t *)malloc(size * sizeof(real_t));
    q->y = (real_t *)malloc(size * sizeof(real_t));
    q->z = (real_t *)malloc(size * sizeof(real_t));
    q->w = (real_t *)malloc(size * sizeof(real_t));
}

void quadrature_destroy(quadrature_t *const q) {
    q->size = 0;
    free(q->x);
    free(q->y);
    free(q->z);
    free(q->w);
}

void quadrature_create_tet_4_order_1(quadrature_t *const q) {
    quadrature_create(q, 4);

    q->x[0] = 0.0;
    q->x[1] = 1.0;
    q->x[2] = 0.0;
    q->x[3] = 0.0;

    q->y[0] = 0.0;
    q->y[1] = 0.0;
    q->y[2] = 1.0;
    q->y[3] = 0.0;

    q->z[0] = 0.0;
    q->z[1] = 0.0;
    q->z[2] = 0.0;
    q->z[3] = 1.0;

    q->w[0] = 0.25;
    q->w[1] = 0.25;
    q->w[2] = 0.25;
    q->w[3] = 0.25;
}

void quadrature_create_tet_4_order_2(quadrature_t *const q) {
    quadrature_create(q, 8);

    q->x[0] = 0.0;
    q->x[1] = 1.0;
    q->x[2] = 0.0;
    q->x[3] = 0.0;
    q->x[4] = 0.333333333333;
    q->x[5] = 0.333333333333;
    q->x[6] = 0.0;
    q->x[7] = 0.333333333333;

    q->y[0] = 0.0;
    q->y[1] = 0.0;
    q->y[2] = 1.0;
    q->y[3] = 0.0;
    q->y[4] = 0.333333333333;
    q->y[5] = 0.0;
    q->y[6] = 0.333333333333;
    q->y[7] = 0.333333333333;

    q->z[0] = 0.0;
    q->z[1] = 0.0;
    q->z[2] = 0.0;
    q->z[3] = 1.0;
    q->z[4] = 0.0;
    q->z[5] = 0.333333333333;
    q->z[6] = 0.333333333333;
    q->z[7] = 0.333333333333;

    q->w[0] = 0.025;
    q->w[1] = 0.025;
    q->w[2] = 0.025;
    q->w[3] = 0.025;
    q->w[4] = 0.225;
    q->w[5] = 0.225;
    q->w[6] = 0.225;
    q->w[7] = 0.225;
}

void quadrature_create_tet_4_order_6(quadrature_t *const q) {
    quadrature_create(q, 45);

    int idx = 0;
    q->x[idx++] = 0.2500000000000000;
    q->x[idx++] = 0.6175871903000830;
    q->x[idx++] = 0.1274709365666390;
    q->x[idx++] = 0.1274709365666390;
    q->x[idx++] = 0.1274709365666390;
    q->x[idx++] = 0.9037635088221031;
    q->x[idx++] = 0.0320788303926323;
    q->x[idx++] = 0.0320788303926323;
    q->x[idx++] = 0.0320788303926323;
    q->x[idx++] = 0.4502229043567190;
    q->x[idx++] = 0.0497770956432810;
    q->x[idx++] = 0.0497770956432810;
    q->x[idx++] = 0.0497770956432810;
    q->x[idx++] = 0.4502229043567190;
    q->x[idx++] = 0.4502229043567190;
    q->x[idx++] = 0.3162695526014501;
    q->x[idx++] = 0.1837304473985499;
    q->x[idx++] = 0.1837304473985499;
    q->x[idx++] = 0.1837304473985499;
    q->x[idx++] = 0.3162695526014501;
    q->x[idx++] = 0.3162695526014501;
    q->x[idx++] = 0.0229177878448171;
    q->x[idx++] = 0.2319010893971509;
    q->x[idx++] = 0.2319010893971509;
    q->x[idx++] = 0.5132800333608811;
    q->x[idx++] = 0.2319010893971509;
    q->x[idx++] = 0.2319010893971509;
    q->x[idx++] = 0.2319010893971509;
    q->x[idx++] = 0.0229177878448171;
    q->x[idx++] = 0.5132800333608811;
    q->x[idx++] = 0.2319010893971509;
    q->x[idx++] = 0.0229177878448171;
    q->x[idx++] = 0.5132800333608811;
    q->x[idx++] = 0.7303134278075384;
    q->x[idx++] = 0.0379700484718286;
    q->x[idx++] = 0.0379700484718286;
    q->x[idx++] = 0.1937464752488044;
    q->x[idx++] = 0.0379700484718286;
    q->x[idx++] = 0.0379700484718286;
    q->x[idx++] = 0.0379700484718286;
    q->x[idx++] = 0.7303134278075384;
    q->x[idx++] = 0.1937464752488044;
    q->x[idx++] = 0.0379700484718286;
    q->x[idx++] = 0.7303134278075384;
    q->x[idx++] = 0.1937464752488044;

    idx = 0;
    q->y[idx++] = 0.2500000000000000;
    q->y[idx++] = 0.1274709365666390;
    q->y[idx++] = 0.1274709365666390;
    q->y[idx++] = 0.1274709365666390;
    q->y[idx++] = 0.6175871903000830;
    q->y[idx++] = 0.0320788303926323;
    q->y[idx++] = 0.0320788303926323;
    q->y[idx++] = 0.0320788303926323;
    q->y[idx++] = 0.9037635088221031;
    q->y[idx++] = 0.0497770956432810;
    q->y[idx++] = 0.4502229043567190;
    q->y[idx++] = 0.0497770956432810;
    q->y[idx++] = 0.4502229043567190;
    q->y[idx++] = 0.0497770956432810;
    q->y[idx++] = 0.4502229043567190;
    q->y[idx++] = 0.1837304473985499;
    q->y[idx++] = 0.3162695526014501;
    q->y[idx++] = 0.1837304473985499;
    q->y[idx++] = 0.3162695526014501;
    q->y[idx++] = 0.1837304473985499;
    q->y[idx++] = 0.3162695526014501;
    q->y[idx++] = 0.2319010893971509;
    q->y[idx++] = 0.0229177878448171;
    q->y[idx++] = 0.2319010893971509;
    q->y[idx++] = 0.2319010893971509;
    q->y[idx++] = 0.5132800333608811;
    q->y[idx++] = 0.2319010893971509;
    q->y[idx++] = 0.0229177878448171;
    q->y[idx++] = 0.5132800333608811;
    q->y[idx++] = 0.2319010893971509;
    q->y[idx++] = 0.5132800333608811;
    q->y[idx++] = 0.2319010893971509;
    q->y[idx++] = 0.0229177878448171;
    q->y[idx++] = 0.0379700484718286;
    q->y[idx++] = 0.7303134278075384;
    q->y[idx++] = 0.0379700484718286;
    q->y[idx++] = 0.0379700484718286;
    q->y[idx++] = 0.1937464752488044;
    q->y[idx++] = 0.0379700484718286;
    q->y[idx++] = 0.7303134278075384;
    q->y[idx++] = 0.1937464752488044;
    q->y[idx++] = 0.0379700484718286;
    q->y[idx++] = 0.1937464752488044;
    q->y[idx++] = 0.0379700484718286;
    q->y[idx++] = 0.7303134278075384;

    idx = 0;
    q->z[idx++] = 0.2500000000000000;
    q->z[idx++] = 0.1274709365666390;
    q->z[idx++] = 0.1274709365666390;
    q->z[idx++] = 0.6175871903000830;
    q->z[idx++] = 0.1274709365666390;
    q->z[idx++] = 0.0320788303926323;
    q->z[idx++] = 0.0320788303926323;
    q->z[idx++] = 0.9037635088221031;
    q->z[idx++] = 0.0320788303926323;
    q->z[idx++] = 0.0497770956432810;
    q->z[idx++] = 0.0497770956432810;
    q->z[idx++] = 0.4502229043567190;
    q->z[idx++] = 0.4502229043567190;
    q->z[idx++] = 0.4502229043567190;
    q->z[idx++] = 0.0497770956432810;
    q->z[idx++] = 0.1837304473985499;
    q->z[idx++] = 0.1837304473985499;
    q->z[idx++] = 0.3162695526014501;
    q->z[idx++] = 0.3162695526014501;
    q->z[idx++] = 0.3162695526014501;
    q->z[idx++] = 0.1837304473985499;
    q->z[idx++] = 0.2319010893971509;
    q->z[idx++] = 0.2319010893971509;
    q->z[idx++] = 0.0229177878448171;
    q->z[idx++] = 0.2319010893971509;
    q->z[idx++] = 0.2319010893971509;
    q->z[idx++] = 0.5132800333608811;
    q->z[idx++] = 0.5132800333608811;
    q->z[idx++] = 0.2319010893971509;
    q->z[idx++] = 0.0229177878448171;
    q->z[idx++] = 0.0229177878448171;
    q->z[idx++] = 0.5132800333608811;
    q->z[idx++] = 0.2319010893971509;
    q->z[idx++] = 0.0379700484718286;
    q->z[idx++] = 0.0379700484718286;
    q->z[idx++] = 0.7303134278075384;
    q->z[idx++] = 0.0379700484718286;
    q->z[idx++] = 0.0379700484718286;
    q->z[idx++] = 0.1937464752488044;
    q->z[idx++] = 0.1937464752488044;
    q->z[idx++] = 0.0379700484718286;
    q->z[idx++] = 0.7303134278075384;
    q->z[idx++] = 0.7303134278075384;
    q->z[idx++] = 0.1937464752488044;
    q->z[idx++] = 0.0379700484718286;

    idx = 0;
    q->w[idx++] = -0.2359620398477559;
    q->w[idx++] = 0.0244878963560563;
    q->w[idx++] = 0.0244878963560563;
    q->w[idx++] = 0.0244878963560563;
    q->w[idx++] = 0.0244878963560563;
    q->w[idx++] = 0.0039485206398261;
    q->w[idx++] = 0.0039485206398261;
    q->w[idx++] = 0.0039485206398261;
    q->w[idx++] = 0.0039485206398261;
    q->w[idx++] = 0.0263055529507371;
    q->w[idx++] = 0.0263055529507371;
    q->w[idx++] = 0.0263055529507371;
    q->w[idx++] = 0.0263055529507371;
    q->w[idx++] = 0.0263055529507371;
    q->w[idx++] = 0.0263055529507371;
    q->w[idx++] = 0.0829803830550590;
    q->w[idx++] = 0.0829803830550590;
    q->w[idx++] = 0.0829803830550590;
    q->w[idx++] = 0.0829803830550590;
    q->w[idx++] = 0.0829803830550590;
    q->w[idx++] = 0.0829803830550590;
    q->w[idx++] = 0.0254426245481024;
    q->w[idx++] = 0.0254426245481024;
    q->w[idx++] = 0.0254426245481024;
    q->w[idx++] = 0.0254426245481024;
    q->w[idx++] = 0.0254426245481024;
    q->w[idx++] = 0.0254426245481024;
    q->w[idx++] = 0.0254426245481024;
    q->w[idx++] = 0.0254426245481024;
    q->w[idx++] = 0.0254426245481024;
    q->w[idx++] = 0.0254426245481024;
    q->w[idx++] = 0.0254426245481024;
    q->w[idx++] = 0.0254426245481024;
    q->w[idx++] = 0.0134324384376852;
    q->w[idx++] = 0.0134324384376852;
    q->w[idx++] = 0.0134324384376852;
    q->w[idx++] = 0.0134324384376852;
    q->w[idx++] = 0.0134324384376852;
    q->w[idx++] = 0.0134324384376852;
    q->w[idx++] = 0.0134324384376852;
    q->w[idx++] = 0.0134324384376852;
    q->w[idx++] = 0.0134324384376852;
    q->w[idx++] = 0.0134324384376852;
    q->w[idx++] = 0.0134324384376852;
    q->w[idx++] = 0.0134324384376852;
}

static SFEM_INLINE void box_tet_quadrature(const quadrature_t *const q,
                                           const geom_t x_min,
                                           const geom_t y_min,
                                           const geom_t z_min,
                                           const geom_t x_max,
                                           const geom_t y_max,
                                           const geom_t z_max,
                                           const geom_t x[4],
                                           const geom_t y[4],
                                           const geom_t z[4],
                                           quadrature_t *const q_box,
                                           quadrature_t *const q_tet) {
    real_t p0[3], p1[3], p2[3], p3[3];
    real_t qp[3];

    const real_t x_range = x_max - x_min;
    const real_t y_range = y_max - y_min;
    const real_t z_range = z_max - z_min;

    // Move to reference cube
    p0[0] = (x[0] - x_min) / x_range;
    p0[1] = (y[0] - y_min) / y_range;
    p0[2] = (z[0] - z_min) / z_range;

    p1[0] = (x[1] - x_min) / x_range;
    p1[1] = (y[1] - y_min) / y_range;
    p1[2] = (z[1] - z_min) / z_range;

    p2[0] = (x[2] - x_min) / x_range;
    p2[1] = (y[2] - y_min) / y_range;
    p2[2] = (z[2] - z_min) / z_range;

    p3[0] = (x[3] - x_min) / x_range;
    p3[1] = (y[3] - y_min) / y_range;
    p3[2] = (z[3] - z_min) / z_range;

    // Create local coordinate system for tet4
    for (int d = 0; d < 3; ++d) {
        p1[d] -= p0[d];
        p2[d] -= p0[d];
        p3[d] -= p0[d];
    }

    // Generate quadrature points and test for containment
    q_box->size = 0;
    real_t measure = 0;
    for (int k = 0; k < q->size; k++) {
        int discard = 0;
        for (int d = 0; d < 3; ++d) {
            qp[d] = p0[d] + (q->x[k] * p1[d]) + (q->y[k] * p2[d]) + (q->z[k] * p3[d]);
            discard += qp[d] < -1e-16 || qp[d] > (1 + 1e-16);
        }

        if (discard) continue;

        measure += q->w[k];

        const int qidx = q_box->size;
        q_box->x[qidx] = qp[0];
        q_box->y[qidx] = qp[1];
        q_box->z[qidx] = qp[2];
        q_box->w[qidx] = q->w[k];

        q_tet->x[qidx] = q->x[k];
        q_tet->y[qidx] = q->y[k];
        q_tet->z[qidx] = q->z[k];
        q_tet->w[qidx] = q->w[k];

        q_box->size++;
    }

    q_tet->size = q_box->size;
}

static SFEM_INLINE void box_gather(const count_t x,
                                   const count_t y,
                                   const count_t z,
                                   const count_t *SFEM_RESTRICT stride,
                                   const real_t *SFEM_RESTRICT box_field,
                                   real_t *SFEM_RESTRICT box_nodal_values) {
    const count_t xl = x * stride[0];
    const count_t xr = (x + 1) * stride[0];

    const count_t yl = y * stride[1];
    const count_t yr = (y + 1) * stride[1];

    const count_t zl = z * stride[2];
    const count_t zr = (z + 1) * stride[2];

    // z-bottom
    box_nodal_values[0] = box_field[xl + yl + zl];
    box_nodal_values[1] = box_field[xr + yl + zl];
    box_nodal_values[2] = box_field[xl + yr + zl];
    box_nodal_values[3] = box_field[xr + yr + zl];

    // z-top
    box_nodal_values[4] = box_field[xl + yl + zr];
    box_nodal_values[5] = box_field[xr + yl + zr];
    box_nodal_values[6] = box_field[xl + yr + zr];
    box_nodal_values[7] = box_field[xr + yr + zr];
}

static SFEM_INLINE void tet4_scatter_add(const idx_t *const SFEM_RESTRICT tet_dofs,
                                         const real_t *const SFEM_RESTRICT tet_nodal_values,
                                         real_t *const SFEM_RESTRICT field) {
    for (int d = 0; d < 4; d++) {
        const idx_t node = tet_dofs[d];
        field[node] += tet_nodal_values[d];
    }
}

static SFEM_INLINE void l2_assemble(const quadrature_t *q_box,
                                    const quadrature_t *q_tet,
                                    const real_t *SFEM_RESTRICT box_nodal_values,
                                    real_t *SFEM_RESTRICT tet_nodal_values,
                                    real_t *SFEM_RESTRICT tet_nodal_weights) {
    memset(tet_nodal_values, 0, 4 * sizeof(real_t));
    memset(tet_nodal_weights, 0, 4 * sizeof(real_t));

    for (int k = 0; k < q_box->size; k++) {
        const real_t xk = q_box->x[k];
        const real_t yk = q_box->y[k];
        const real_t zk = q_box->z[k];
        const real_t dV = q_tet->w[k] / 6;

        real_t value = 0;

        const real_t mx = (1 - xk);
        const real_t my = (1 - yk);
        const real_t mz = (1 - zk);

        // z-bottom
        value += mx * my * mz * box_nodal_values[0];
        value += xk * my * mz * box_nodal_values[1];
        value += mx * yk * mz * box_nodal_values[2];
        value += xk * yk * mz * box_nodal_values[3];

        // z-top
        value += mx * my * zk * box_nodal_values[4];
        value += xk * my * zk * box_nodal_values[5];
        value += mx * yk * zk * box_nodal_values[6];
        value += xk * yk * zk * box_nodal_values[7];

        // Scale by quadrature weight
        value *= dV;

        real_t f0k = (1 - q_tet->x[k] - q_tet->y[k] - q_tet->z[k]);

        tet_nodal_values[0] += f0k * value;
        tet_nodal_values[1] += q_tet->x[k] * value;
        tet_nodal_values[2] += q_tet->y[k] * value;
        tet_nodal_values[3] += q_tet->z[k] * value;

        tet_nodal_weights[0] += f0k * dV;
        tet_nodal_weights[1] += q_tet->x[k] * dV;
        tet_nodal_weights[2] += q_tet->y[k] * dV;
        tet_nodal_weights[3] += q_tet->z[k] * dV;
    }
}

void resample_box_to_tetra_mesh(const count_t n[3],
                                const count_t stride[3],
                                const real_t *const SFEM_RESTRICT box_field,
                                const ptrdiff_t n_elements,
                                const ptrdiff_t n_nodes,
                                idx_t **const SFEM_RESTRICT elems,
                                geom_t **const SFEM_RESTRICT xyz,
                                real_t *const SFEM_RESTRICT mesh_field) {
    memset(mesh_field, 0, n_nodes * sizeof(real_t));

    real_t *weight_field = (real_t *)malloc(n_nodes * sizeof(real_t));
    memset(weight_field, 0, n_nodes * sizeof(real_t));

    quadrature_t q_ref;
    quadrature_t q_box;
    quadrature_t q_tet;

    {
        quadrature_create_tet_4_order_1(&q_ref);
        // quadrature_create_tet_4_order_2(&q_ref);
        // quadrature_create_tet_4_order_6(&q_ref);
        quadrature_create(&q_box, q_ref.size);
        quadrature_create(&q_tet, q_ref.size);
    }

    geom_t xe[4], ye[4], ze[4];
    real_t box_nodal_values[8];
    real_t tet_nodal_values[4];
    real_t tet_nodal_weights[4];
    idx_t tet_dofs[4];

    const geom_t lmargin = 0;
    const geom_t rmargin = 1;

    for (ptrdiff_t e = 0; e < n_elements; e++) {
        for (int d = 0; d < 4; d++) {
            const idx_t node = elems[d][e];
            tet_dofs[d] = node;
            xe[d] = xyz[0][node];
            ye[d] = xyz[1][node];
            ze[d] = xyz[2][node];
        }

        const count_t x_min = MAX(0, floor(array_min(4, xe)));
        const count_t y_min = MAX(0, floor(array_min(4, ye)));
        const count_t z_min = MAX(0, floor(array_min(4, ze)));

        const count_t x_max = MIN(n[0], ceil(array_max(4, xe)));
        const count_t y_max = MIN(n[1], ceil(array_max(4, ye)));
        const count_t z_max = MIN(n[2], ceil(array_max(4, ze)));

        printf("-------------------------\n");
        printf("min %ld %ld %ld\n", (long)x_min, (long)y_min, (long)z_min);
        printf("max %ld %ld %ld\n", (long)x_max, (long)y_max, (long)z_max);

        for (count_t z = z_min; z < z_max; z++) {
            for (count_t y = y_min; y < y_max; y++) {
                for (count_t x = x_min; x < x_max; x++) {
                    box_tet_quadrature(&q_ref,
                                       x - lmargin,
                                       y - lmargin,
                                       z - lmargin,
                                       x + rmargin,
                                       y + rmargin,
                                       z + rmargin,
                                       xe,
                                       ye,
                                       ze,
                                       &q_box,
                                       &q_tet);

                    if (!q_box.size) {
                        // No intersection
                        continue;
                    }

                    box_gather(x, y, z, stride, box_field, box_nodal_values);
                    l2_assemble(&q_box, &q_tet, box_nodal_values, tet_nodal_values, tet_nodal_weights);

                    tet4_scatter_add(tet_dofs, tet_nodal_values, mesh_field);
                    tet4_scatter_add(tet_dofs, tet_nodal_weights, weight_field);
                }
            }
        }
    }

    // Normalize projection
    for (ptrdiff_t i = 0; i < n_nodes; ++i) {
        real_t w = weight_field[i];
        printf("%g\n", w);
        assert(w != 0.);
        if (w == 0) {
            mesh_field[i] = 0;
        } else {
            mesh_field[i] /= w;
        }
    }

    // Clean-up
    {
        quadrature_destroy(&q_ref);
        quadrature_destroy(&q_box);
        quadrature_destroy(&q_tet);
        free(weight_field);
    }
}

void resample_box_to_tetra_mesh_buggy(const count_t n[3],
                                      const count_t stride[3],
                                      const real_t *const SFEM_RESTRICT box_field,
                                      const ptrdiff_t n_elements,
                                      const ptrdiff_t n_nodes,
                                      idx_t **const SFEM_RESTRICT elems,
                                      geom_t **const SFEM_RESTRICT xyz,
                                      real_t *const SFEM_RESTRICT mesh_field) {
    geom_t *zbi_min = (geom_t *)malloc(n_elements * sizeof(geom_t));
    geom_t *zbi_max = (geom_t *)malloc(n_elements * sizeof(geom_t));
    cell_list_1D_t cl;

    memset(mesh_field, 0, n_nodes * sizeof(real_t));

    real_t *weight_field = (real_t *)malloc(n_nodes * sizeof(real_t));
    memset(weight_field, 0, n_nodes * sizeof(real_t));

    bounding_intervals(n_elements, 4, elems, xyz[2], zbi_min, zbi_max);
    cell_list_1D_create(&cl, n_elements, zbi_min, zbi_max);

    quadrature_t q_ref;
    quadrature_t q_box;
    quadrature_t q_tet;

    {
        quadrature_create_tet_4_order_2(&q_ref);
        quadrature_create(&q_box, q_ref.size);
        quadrature_create(&q_tet, q_ref.size);
    }

    geom_t xe[4], ye[4], ze[4];
    real_t box_nodal_values[8];
    real_t tet_nodal_values[4];
    real_t tet_nodal_weights[4];
    idx_t tet_dofs[4];

    const geom_t lmargin = 0;
    const geom_t rmargin = 1;

    for (count_t z = 0; z < n[2] - 1; z++) {
        cell_list_1D_query_t q = cell_list_1D_query(&cl, z - lmargin, z + rmargin);
        assert(q.begin >= 0);
        assert(q.end <= cl.n_entries);

        for (ptrdiff_t k = q.begin; k < q.end; k++) {
            const ptrdiff_t e = cl.idx[k];

            for (int d = 0; d < 4; d++) {
                const idx_t node = elems[d][e];
                tet_dofs[d] = node;
                xe[d] = xyz[0][node];
                ye[d] = xyz[1][node];
                ze[d] = xyz[2][node];
            }

            for (ptrdiff_t y = 0; y < n[1] - 1; y++) {
                for (ptrdiff_t x = 0; x < n[0] - 1; x++) {
                    const geom_t x_min = x - lmargin;
                    const geom_t y_min = y - lmargin;
                    const geom_t z_min = z - lmargin;

                    const geom_t x_max = x + rmargin;
                    const geom_t y_max = y + rmargin;
                    const geom_t z_max = z + rmargin;

                    box_tet_quadrature(&q_ref, x_min, y_min, z_min, x_max, y_max, z_max, xe, ye, ze, &q_box, &q_tet);

                    // No intersection
                    if (!q_box.size) continue;

                    box_gather(x, y, z, stride, box_field, box_nodal_values);
                    l2_assemble(&q_box, &q_tet, box_nodal_values, tet_nodal_values, tet_nodal_weights);

                    tet4_scatter_add(tet_dofs, tet_nodal_values, mesh_field);
                    tet4_scatter_add(tet_dofs, tet_nodal_weights, weight_field);
                }
            }
        }
    }

    // Normalize projection
    for (ptrdiff_t i = 0; i < n_nodes; ++i) {
        real_t w = weight_field[i];
        printf("%g\n", w);
        assert(w != 0.);
        if (w == 0) {
            mesh_field[i] = 0;
        } else {
            mesh_field[i] /= w;
        }
    }

    // Clean-up
    {
        quadrature_destroy(&q_ref);
        quadrature_destroy(&q_box);
        quadrature_destroy(&q_tet);

        free(zbi_min);
        free(zbi_max);
        free(weight_field);
        cell_list_1D_destroy(&cl);
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (argc < 6) {
        if (!rank) {
            fprintf(
                stderr, "usage: %s <nx> <ny> <nz> <field.raw> <mesh_folder> [output_path=./mesh_field.raw]", argv[0]);
        }

        return EXIT_FAILURE;
    }

    count_t n[3];
    count_t stride[3];

    n[0] = atol(argv[1]);
    n[1] = atol(argv[2]);
    n[2] = atol(argv[3]);
    const ptrdiff_t size_field = n[0] * n[1] * n[2];

    stride[0] = 1;
    stride[1] = n[0];
    stride[2] = n[0] * n[1];

    const char *field_path = argv[4];
    const char *mesh_folder = argv[5];

    const char *output_path = "./mesh_field.raw";

    if (argc > 6) {
        output_path = argv[6];
    }

    if (!rank) {
        fprintf(stderr,
                "usage: %s %ld %ld %ld %s %s %s\n",
                argv[0],
                (long)n[0],
                (long)n[1],
                (long)n[2],
                field_path,
                mesh_folder,
                output_path);
    }

    double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    real_t *box_field;
    ptrdiff_t field_n_local, field_n_global;

    if (strcmp(field_path, "demo") == 0) {
        box_field = (real_t *)malloc(size_field * sizeof(real_t));

        geom_t point[3] = {0, 0, 0};
        for (ptrdiff_t z = 0; z < n[2]; ++z) {
            point[2] = z / (1.0 * n[2]);
            for (ptrdiff_t y = 0; y < n[1]; ++y) {
                point[1] = y / (1.0 * n[1]);
                for (ptrdiff_t x = 0; x < n[0]; ++x) {
                    point[0] = x / (1.0 * n[0]);
                    // box_field[z * stride[2] + y * stride[1] + x * stride[0]] =
                    //     point[0] * point[0] + point[1] * point[1] + point[2] * point[2];

                    box_field[z * stride[2] + y * stride[1] + x * stride[0]] = x * (z == 0) * (y == 0);

                    // Constant function
                    // box_field[z * stride[2] + y * stride[1] + x * stride[0]] = 1;
                }
            }
        }

    } else {
        array_read(comm, field_path, SFEM_MPI_REAL_T, (void **)&box_field, &field_n_local, &field_n_global);
        assert(size_field == field_n_global);
    }

    mesh_t mesh;
    if (mesh_read(comm, mesh_folder, &mesh)) {
        return EXIT_FAILURE;
    }

    real_t *mesh_field = (real_t *)malloc(mesh.nnodes * sizeof(real_t));
    memset(mesh_field, 0, mesh.nnodes * sizeof(real_t));

    // FIXME! Implement parallel version!
    resample_box_to_tetra_mesh(
        n, stride, box_field, mesh.nelements, mesh.nnodes, mesh.elements, mesh.points, mesh_field);

    // FIXME! Implement write field with mesh!
    array_write(comm, output_path, SFEM_MPI_REAL_T, (void *)mesh_field, mesh.nnodes, mesh.nnodes);

    // Free resources
    mesh_destroy(&mesh);
    free(box_field);
    free(mesh_field);

    double tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
