#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "array_dtof.h"
#include "matrixio_array.h"
#include "matrixio_crs.h"
#include "utils.h"

#include "sortreduce.h"

#include "crs_graph.h"
#include "read_mesh.h"
#include "sfem_base.h"
#include "sfem_mesh_write.h"

#include "grid.h"

typedef struct {
    real_t shift[3];
    real_t scaling[3];
} affine_transform_t;

void affine_transform_init(affine_transform_t *const trafo) {
    trafo->shift[0] = 0;
    trafo->shift[1] = 0;
    trafo->shift[2] = 0;

    trafo->scaling[0] = 1;
    trafo->scaling[1] = 1;
    trafo->scaling[2] = 1;
}

void affine_transform_copy(const affine_transform_t *const src, affine_transform_t *const dest) {
    dest->shift[0] = src->shift[0];
    dest->shift[1] = src->shift[1];
    dest->shift[2] = src->shift[2];
    dest->scaling[0] = src->scaling[0];
    dest->scaling[1] = src->scaling[1];
    dest->scaling[2] = src->scaling[2];
}

static SFEM_INLINE void affine_transform_apply(const affine_transform_t *const trafo,
                                               const real_t *const in,
                                               real_t *out) {
    for (int d = 0; d < 3; ++d) {
        out[d] = trafo->shift[d] + trafo->scaling[d] * in[d];
    }
}

static SFEM_INLINE void affine_transform_apply_inverse(const affine_transform_t *const trafo,
                                                       const real_t *const in,
                                                       real_t *out) {
    for (int d = 0; d < 3; ++d) {
        out[d] = (in[d] - trafo->shift[d]) / trafo->scaling[d];
    }
}

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

static SFEM_INLINE real_t array_min_r(const ptrdiff_t n, const real_t *a) {
    real_t ret = a[0];
    for (ptrdiff_t i = 1; i < n; ++i) {
        ret = MIN(ret, a[i]);
    }

    return ret;
}

static SFEM_INLINE real_t array_max_r(const ptrdiff_t n, const real_t *const SFEM_RESTRICT a) {
    real_t ret = a[0];
    for (ptrdiff_t i = 1; i < n; ++i) {
        ret = MAX(ret, a[i]);
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

    // cell_list_1D_print(cl);
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

void quadrature_print(const quadrature_t *const q) {
    printf("---------------------\n");
    for (int k = 0; k < q->size; k++) {
        printf("%g ", (double)q->x[k]);
        printf("%g ", (double)q->y[k]);
        printf("%g ", (double)q->z[k]);
        printf("\n");
    }

    for (int k = 0; k < q->size; k++) {
        printf("%g ", (double)q->w[k]);
    }
    printf("\n");
    printf("---------------------\n");
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
                                           const real_t x_min,
                                           const real_t y_min,
                                           const real_t z_min,
                                           const real_t x_max,
                                           const real_t y_max,
                                           const real_t z_max,
                                           const real_t x[4],
                                           const real_t y[4],
                                           const real_t z[4],
                                           quadrature_t *const q_box,
                                           quadrature_t *const q_tet) {
    real_t p0[3], p1[3], p2[3], p3[3];
    real_t qp[3];

    const real_t x_range = x_max - x_min;
    const real_t y_range = y_max - y_min;
    const real_t z_range = z_max - z_min;

    assert(x_range > 0);
    assert(y_range > 0);
    assert(z_range > 0);

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
    for (int k = 0; k < q->size; k++) {
        int discard = 0;
        for (int d = 0; d < 3; ++d) {
            qp[d] = p0[d] + (q->x[k] * p1[d]) + (q->y[k] * p2[d]) + (q->z[k] * p3[d]);
            discard += qp[d] < -1e-16 || qp[d] > (1 + 1e-16);
        }

        if (discard) continue;

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
        const real_t dV = q_tet->w[k] / 6.0;

        assert(xk >= -1e-16);
        assert(xk <= 1 + 1e-16);

        assert(yk >= -1e-16);
        assert(yk <= 1 + 1e-16);

        assert(zk >= -1e-16);
        assert(zk <= 1 + 1e-16);

        assert(q_tet->x[k] >= -1e-16);
        assert(q_tet->x[k] <= 1 + 1e-16);

        assert(q_tet->y[k] >= -1e-16);
        assert(q_tet->y[k] <= 1 + 1e-16);

        assert(q_tet->z[k] >= -1e-16);
        assert(q_tet->z[k] <= 1 + 1e-16);

        real_t value = 0;

        const real_t mx = (1.0 - xk);
        const real_t my = (1.0 - yk);
        const real_t mz = (1.0 - zk);

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

        real_t f0k = (1.0 - q_tet->x[k] - q_tet->y[k] - q_tet->z[k]);
#if 1
        real_t df0k = f0k;
        real_t df1k = q_tet->x[k];
        real_t df2k = q_tet->y[k];
        real_t df3k = q_tet->z[k];
#else
        real_t df0k = 3 * f0k - q_tet->x[k] - q_tet->y[k] - q_tet->z[k];
        real_t df1k = -f0k + 3 * q_tet->x[k] - q_tet->y[k] - q_tet->z[k];
        real_t df2k = -f0k - q_tet->x[k] + 3 * q_tet->y[k] - q_tet->z[k];
        real_t df3k = -f0k - q_tet->x[k] - q_tet->y[k] + 3 * q_tet->z[k];
#endif

        tet_nodal_values[0] += df0k * value;
        tet_nodal_values[1] += df1k * value;
        tet_nodal_values[2] += df2k * value;
        tet_nodal_values[3] += df3k * value;

        tet_nodal_weights[0] += df0k * dV;
        tet_nodal_weights[1] += df1k * dV;
        tet_nodal_weights[2] += df2k * dV;
        tet_nodal_weights[3] += df3k * dV;
    }
}

void resample_box_to_tetra_mesh_unique(const count_t n[3],
                                       const count_t stride[3],
                                       const affine_transform_t *const trafo,
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
    quadrature_t q_physical;

    {
        // quadrature_create_tet_4_order_1(&q_ref);
        // quadrature_create_tet_4_order_2(&q_ref);
        quadrature_create_tet_4_order_6(&q_ref);
        quadrature_create(&q_physical, q_ref.size);
    }

    uint8_t *found = (uint8_t *)malloc(q_ref.size * sizeof(uint8_t));

    real_t xe[4], ye[4], ze[4];
    real_t box_nodal_values[8];
    real_t tet_nodal_values[4];
    real_t tet_nodal_weights[4];
    idx_t tet_dofs[4];
    real_t aabb_min[3], aabb_max[3];

    for (ptrdiff_t e = 0; e < n_elements; e++) {
        for (int d = 0; d < 4; d++) {
            const idx_t node = elems[d][e];
            tet_dofs[d] = node;
            xe[d] = xyz[0][node];
            ye[d] = xyz[1][node];
            ze[d] = xyz[2][node];
        }

        aabb_min[0] = array_min_r(4, xe);
        aabb_min[1] = array_min_r(4, ye);
        aabb_min[2] = array_min_r(4, ze);

        aabb_max[0] = array_max_r(4, xe);
        aabb_max[1] = array_max_r(4, ye);
        aabb_max[2] = array_max_r(4, ze);

        affine_transform_apply_inverse(trafo, aabb_min, aabb_min);
        affine_transform_apply_inverse(trafo, aabb_max, aabb_max);

        count_t grid_n[3];
        grid_n[0] = n[0] - 1;
        grid_n[1] = n[1] - 1;
        grid_n[2] = n[2] - 1;

        const count_t x_min = MAX(0, floor(aabb_min[0] * grid_n[0]));
        const count_t y_min = MAX(0, floor(aabb_min[1] * grid_n[1]));
        const count_t z_min = MAX(0, floor(aabb_min[2] * grid_n[2]));

        const count_t x_max = MIN(grid_n[0], ceil(aabb_max[0] * grid_n[0]));
        const count_t y_max = MIN(grid_n[1], ceil(aabb_max[1] * grid_n[1]));
        const count_t z_max = MIN(grid_n[2], ceil(aabb_max[2] * grid_n[2]));

        {
            // Generate quadrature points in physical coordinates
            for (int k = 0; k < q_ref.size; k++) {
                const real_t phi0 = (1.0 - q_ref.x[k] - q_ref.y[k] - q_ref.z[k]);
                const real_t phi1 = q_ref.x[k];
                const real_t phi2 = q_ref.y[k];
                const real_t phi3 = q_ref.z[k];

                q_physical.x[k] = phi0 * xe[0] + phi1 * xe[1] + phi2 * xe[2] + phi3 * xe[3];
                q_physical.y[k] = phi0 * ye[0] + phi1 * ye[1] + phi2 * ye[2] + phi3 * ye[3];
                q_physical.z[k] = phi0 * ze[0] + phi1 * ze[1] + phi2 * ze[2] + phi3 * ze[3];
                q_physical.w[k] = q_ref.w[k];
            }
        }

        memset(found, 0, q_ref.size * sizeof(uint8_t));
        memset(tet_nodal_values, 0, 4 * sizeof(real_t));
        memset(tet_nodal_weights, 0, 4 * sizeof(real_t));

        ptrdiff_t all_nqp = 0;
        for (count_t z = z_min; z < z_max; z++) {
            for (count_t y = y_min; y < y_max; y++) {
                for (count_t x = x_min; x < x_max; x++) {
                    aabb_min[0] = x;
                    aabb_min[1] = y;
                    aabb_min[2] = z;

                    aabb_max[0] = (x + 1);
                    aabb_max[1] = (y + 1);
                    aabb_max[2] = (z + 1);

                    for (int d = 0; d < 3; ++d) {
                        aabb_min[d] /= grid_n[d];
                        aabb_max[d] /= grid_n[d];
                    }

                    affine_transform_apply(trafo, aabb_min, aabb_min);
                    affine_transform_apply(trafo, aabb_max, aabb_max);

                    box_gather(x, y, z, stride, box_field, box_nodal_values);

                    real_t qp[3];
                    for (int k = 0; k < q_ref.size; ++k) {
                        if (found[k]) continue;

                        qp[0] = (q_physical.x[k] - aabb_min[0]) / (aabb_max[0] - aabb_min[0]);
                        qp[1] = (q_physical.y[k] - aabb_min[1]) / (aabb_max[1] - aabb_min[1]);
                        qp[2] = (q_physical.z[k] - aabb_min[2]) / (aabb_max[2] - aabb_min[2]);

                        int discard = 0;
                        for (int d = 0; d < 3; ++d) {
                            discard += qp[d] < -1e-16 || qp[d] > (1 + 1e-16);
                        }

                        if (discard) continue;
                        found[k] = 1;
                        all_nqp++;

                        const real_t dV = q_physical.w[k] / 6.0;
                        real_t value = 0;

                        {  // Box
                            const real_t xk = qp[0];
                            const real_t yk = qp[1];
                            const real_t zk = qp[2];

                            assert(xk >= 0);
                            assert(xk <= 1);

                            assert(yk >= 0);
                            assert(yk <= 1);

                            assert(zk >= 0);
                            assert(zk <= 1);

                            const real_t mx = (1.0 - xk);
                            const real_t my = (1.0 - yk);
                            const real_t mz = (1.0 - zk);

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
                        }

                        {  // Tet
                            real_t f0k = (1.0 - q_ref.x[k] - q_ref.y[k] - q_ref.z[k]);

#ifdef SFEM_P1_MULTIPLIER
                            real_t df0k = f0k;
                            real_t df1k = q_ref.x[k];
                            real_t df2k = q_ref.y[k];
                            real_t df3k = q_ref.z[k];
#else
                            real_t df0k = 4 * f0k - q_ref.x[k] - q_ref.y[k] - q_ref.z[k];
                            real_t df1k = -f0k + 4 * q_ref.x[k] - q_ref.y[k] - q_ref.z[k];
                            real_t df2k = -f0k - q_ref.x[k] + 4 * q_ref.y[k] - q_ref.z[k];
                            real_t df3k = -f0k - q_ref.x[k] - q_ref.y[k] + 4 * q_ref.z[k];
#endif

                            tet_nodal_values[0] += df0k * value;
                            tet_nodal_values[1] += df1k * value;
                            tet_nodal_values[2] += df2k * value;
                            tet_nodal_values[3] += df3k * value;

                            tet_nodal_weights[0] += df0k * dV;
                            tet_nodal_weights[1] += df1k * dV;
                            tet_nodal_weights[2] += df2k * dV;
                            tet_nodal_weights[3] += df3k * dV;
                        }
                    }
                }
            }
        }

        for (int k = 0; k < q_physical.size; ++k) {
            assert(found[k]);
        }

        assert(all_nqp > 0);

        tet4_scatter_add(tet_dofs, tet_nodal_values, mesh_field);
        tet4_scatter_add(tet_dofs, tet_nodal_weights, weight_field);
    }

    // Normalize projection
    for (ptrdiff_t i = 0; i < n_nodes; ++i) {
        real_t w = weight_field[i];
        // printf("%g\n", w);
        assert(w != 0.);
        if (w != 0) {
            mesh_field[i] /= w;
        } else {
            mesh_field[i] = 0;
        }
    }

    // Clean-up
    {
        quadrature_destroy(&q_ref);
        quadrature_destroy(&q_physical);
        free(weight_field);
        free(found);
    }
}

void resample_box_to_tetra_mesh(const count_t n[3],
                                const count_t stride[3],
                                const affine_transform_t *const trafo,
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
        // quadrature_create_tet_4_order_1(&q_ref);
        quadrature_create_tet_4_order_2(&q_ref);
        // quadrature_create_tet_4_order_6(&q_ref);
        quadrature_create(&q_box, q_ref.size);
        quadrature_create(&q_tet, q_ref.size);
    }

    real_t xe[4], ye[4], ze[4];
    real_t box_nodal_values[8];
    real_t tet_nodal_values[4];
    real_t tet_nodal_weights[4];
    idx_t tet_dofs[4];

    real_t aabb_min[3], aabb_max[3];

    for (ptrdiff_t e = 0; e < n_elements; e++) {
        for (int d = 0; d < 4; d++) {
            const idx_t node = elems[d][e];
            tet_dofs[d] = node;
            xe[d] = xyz[0][node];
            ye[d] = xyz[1][node];
            ze[d] = xyz[2][node];
        }

        aabb_min[0] = array_min_r(4, xe);
        aabb_min[1] = array_min_r(4, ye);
        aabb_min[2] = array_min_r(4, ze);

        aabb_max[0] = array_max_r(4, xe);
        aabb_max[1] = array_max_r(4, ye);
        aabb_max[2] = array_max_r(4, ze);

        affine_transform_apply_inverse(trafo, aabb_min, aabb_min);
        affine_transform_apply_inverse(trafo, aabb_max, aabb_max);

        count_t grid_n[3];
        grid_n[0] = n[0] - 1;
        grid_n[1] = n[1] - 1;
        grid_n[2] = n[2] - 1;

        const count_t x_min = MAX(0, floor(aabb_min[0] * grid_n[0]));
        const count_t y_min = MAX(0, floor(aabb_min[1] * grid_n[1]));
        const count_t z_min = MAX(0, floor(aabb_min[2] * grid_n[2]));

        const count_t x_max = MIN(grid_n[0], ceil(aabb_max[0] * grid_n[0]));
        const count_t y_max = MIN(grid_n[1], ceil(aabb_max[1] * grid_n[1]));
        const count_t z_max = MIN(grid_n[2], ceil(aabb_max[2] * grid_n[2]));

        ptrdiff_t all_nqp = 0;
        for (count_t z = z_min; z < z_max; z++) {
            for (count_t y = y_min; y < y_max; y++) {
                for (count_t x = x_min; x < x_max; x++) {
                    aabb_min[0] = x;
                    aabb_min[1] = y;
                    aabb_min[2] = z;

                    aabb_max[0] = (x + 1);
                    aabb_max[1] = (y + 1);
                    aabb_max[2] = (z + 1);

                    for (int d = 0; d < 3; ++d) {
                        aabb_min[d] /= grid_n[d];
                        aabb_max[d] /= grid_n[d];
                    }

                    affine_transform_apply(trafo, aabb_min, aabb_min);
                    affine_transform_apply(trafo, aabb_max, aabb_max);

                    box_tet_quadrature(&q_ref,
                                       aabb_min[0],
                                       aabb_min[1],
                                       aabb_min[2],
                                       aabb_max[0],
                                       aabb_max[1],
                                       aabb_max[2],
                                       xe,
                                       ye,
                                       ze,
                                       &q_box,
                                       &q_tet);

                    all_nqp += q_box.size;
                    if (!q_box.size) {
                        // No intersection
                        continue;
                    }

                    box_gather(x, y, z, stride, box_field, box_nodal_values);
                    l2_assemble(
                        &q_box, &q_tet, box_nodal_values, tet_nodal_values, tet_nodal_weights);

                    tet4_scatter_add(tet_dofs, tet_nodal_values, mesh_field);
                    tet4_scatter_add(tet_dofs, tet_nodal_weights, weight_field);
                }
            }
        }

        assert(all_nqp > 0);
    }

    // Normalize projection
    for (ptrdiff_t i = 0; i < n_nodes; ++i) {
        real_t w = weight_field[i];
        // printf("%g\n", w);
        assert(w != 0.);
        if (w != 0) {
            mesh_field[i] /= w;
        } else {
            mesh_field[i] = 0;
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

// FIXME
void resample_box_to_tetra_mesh_cell_list(const count_t n[3],
                                          const count_t stride[3],
                                          const affine_transform_t *const trafo,
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

    real_t xe[4], ye[4], ze[4];
    real_t box_nodal_values[8];
    real_t tet_nodal_values[4];
    real_t tet_nodal_weights[4];
    idx_t tet_dofs[4];

    real_t aabb_min[3], aabb_max[3];

    for (count_t z = 0; z < n[2] - 1; z++) {
        aabb_min[2] = trafo->shift[2] + trafo->scaling[2] * z;
        aabb_max[2] = trafo->shift[2] + trafo->scaling[2] * (z + 1);

        cell_list_1D_query_t q = cell_list_1D_query(&cl, aabb_min[2], aabb_max[2]);
        assert(q.begin >= 0);
        assert(q.end <= cl.n_entries);

        for (ptrdiff_t k = q.begin; k < q.end; k++) {
            const ptrdiff_t e = cl.idx[k];
            assert(e < n_elements);
            assert(e >= 0);

            for (int d = 0; d < 4; d++) {
                const idx_t node = elems[d][e];
                tet_dofs[d] = node;
                xe[d] = xyz[0][node];
                ye[d] = xyz[1][node];
                ze[d] = xyz[2][node];
            }

            for (ptrdiff_t y = 0; y < n[1] - 1; y++) {
                for (ptrdiff_t x = 0; x < n[0] - 1; x++) {
                    aabb_min[0] = x;
                    aabb_min[1] = y;
                    aabb_min[2] = z;

                    aabb_max[0] = (x + 1);
                    aabb_max[1] = (y + 1);
                    aabb_max[2] = (z + 1);

                    affine_transform_apply(trafo, aabb_min, aabb_min);
                    affine_transform_apply(trafo, aabb_max, aabb_max);

                    box_tet_quadrature(&q_ref,
                                       aabb_min[0],
                                       aabb_min[1],
                                       aabb_min[2],
                                       aabb_max[0],
                                       aabb_max[1],
                                       aabb_max[2],
                                       xe,
                                       ye,
                                       ze,
                                       &q_box,
                                       &q_tet);

                    // No intersection
                    if (!q_box.size) continue;

                    box_gather(x, y, z, stride, box_field, box_nodal_values);
                    l2_assemble(
                        &q_box, &q_tet, box_nodal_values, tet_nodal_values, tet_nodal_weights);

                    tet4_scatter_add(tet_dofs, tet_nodal_values, mesh_field);
                    tet4_scatter_add(tet_dofs, tet_nodal_weights, weight_field);
                }
            }
        }
    }

    // Normalize projection
    for (ptrdiff_t i = 0; i < n_nodes; ++i) {
        real_t w = weight_field[i];
        // printf("%g\n", w);
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
            fprintf(stderr,
                    "usage: %s <nx> <ny> <nz> <field.raw> <mesh_folder> "
                    "[output_path=./mesh_field.raw]\n",
                    argv[0]);
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

    int SFEM_READ_FP32 = 0;
    SFEM_READ_ENV(SFEM_READ_FP32, atoi);

    printf("Env:\nSFEM_READ_FP32=%d\n", SFEM_READ_FP32);

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

    mesh_t mesh;
    if (mesh_read(comm, mesh_folder, &mesh)) {
        return EXIT_FAILURE;
    }

    affine_transform_t trafo;
    affine_transform_init(&trafo);

    gridz_t grid;
    gridz_create(&grid, comm, n[0], n[1], n[2], 0);

    if (strcmp(field_path, "demo") == 0) {
        box_field = (real_t *)malloc(size_field * sizeof(real_t));

        for (int d = 0; d < 3; ++d) {
            trafo.shift[d] = array_min(mesh.nnodes, mesh.points[d]);
            trafo.scaling[d] = array_max(mesh.nnodes, mesh.points[d]);
        }

        CATCH_MPI_ERROR(
            MPI_Allreduce(MPI_IN_PLACE, trafo.shift, 3, SFEM_MPI_REAL_T, MPI_MIN, comm));
        CATCH_MPI_ERROR(
            MPI_Allreduce(MPI_IN_PLACE, trafo.scaling, 3, SFEM_MPI_REAL_T, MPI_MAX, comm));

        for (int d = 0; d < 3; ++d) {
            trafo.scaling[d] -= trafo.shift[d];
        }

        if (!rank) {
            printf("grid %ld %ld %ld\n", (long)n[0], (long)n[1], (long)n[2]);
            printf("trafo\nshift: %g %g %g\n",
                   (double)trafo.shift[0],
                   (double)trafo.shift[1],
                   (double)trafo.shift[2]);
            printf("scaling: %g %g %g\n",
                   (double)trafo.scaling[0],
                   (double)trafo.scaling[1],
                   (double)trafo.scaling[2]);
        }

        geom_t point[3] = {0, 0, 0};
        for (ptrdiff_t z = 0; z < grid.extent[2]; ++z) {
            point[2] = (grid.z_begin + z) / (1.0 * grid.z_global_extent);

            for (ptrdiff_t y = 0; y < grid.extent[1]; ++y) {
                point[1] = y / (1.0 * grid.extent[1]);

                for (ptrdiff_t x = 0; x < grid.extent[0]; ++x) {
                    point[0] = x / (1.0 * grid.extent[0]);
                    box_field[z * stride[2] + y * stride[1] + x * stride[0]] =
                        (point[0] * point[1] * point[2]);
                }
            }
        }

    } else {
        box_field = malloc(grid.local_size * sizeof(real_t));

        if (SFEM_READ_FP32) {
            gridz_read_field(&grid, field_path, MPI_FLOAT, (void *)box_field);

            float *float_field = ((float *)box_field);

            for (ptrdiff_t i = grid.local_size - 1; i >= 0; --i) {
                box_field[i] = float_field[i];
            }

        } else {
            gridz_read_field(&grid, field_path, SFEM_MPI_REAL_T, (void *)box_field);
        }

        for (int d = 0; d < 3; ++d) {
            trafo.shift[d] = 0;
            trafo.scaling[d] = (n[d] - 1);
        }
    }

    real_t *mesh_field = (real_t *)malloc(mesh.nnodes * sizeof(real_t));
    memset(mesh_field, 0, mesh.nnodes * sizeof(real_t));

    double tack = MPI_Wtime();

    if (size > 1) {
        // Parallel transfer

        /////////////////////////////////////////////////////////////////////////
        // Locate portion of grid data
        /////////////////////////////////////////////////////////////////////////

        real_t aabb_min[3], aabb_max[3];
        for (int d = 0; d < 3; ++d) {
            aabb_min[d] = array_min(mesh.nnodes, mesh.points[d]);
            aabb_max[d] = array_max(mesh.nnodes, mesh.points[d]);
        }

        affine_transform_apply_inverse(&trafo, aabb_min, aabb_min);
        affine_transform_apply_inverse(&trafo, aabb_max, aabb_max);

        count_t grid_n[3];
        grid_n[0] = grid.extent[0] - 1;
        grid_n[1] = grid.extent[1] - 1;
        grid_n[2] = grid.z_global_extent - 1;

        const count_t x_min = MAX(0, floor(aabb_min[0] * grid_n[0]));
        const count_t y_min = MAX(0, floor(aabb_min[1] * grid_n[1]));
        const count_t z_min = MAX(0, floor(aabb_min[2] * grid_n[2]));

        const count_t x_max = MIN(grid_n[0], ceil(aabb_max[0] * grid_n[0]));
        const count_t y_max = MIN(grid_n[1], ceil(aabb_max[1] * grid_n[1]));
        const count_t z_max = MIN(grid_n[2], ceil(aabb_max[2] * grid_n[2]));

        const count_t x_extent = x_max - x_min;
        const count_t y_extent = y_max - y_min;
        const count_t z_extent = z_max - z_min;

        // assert(remote_grid_size > 0);

        /////////////////////////////////////////////////////////////////////////
        // Prepare grid data exchange
        /////////////////////////////////////////////////////////////////////////

        ptrdiff_t *ownership_ranges = malloc((size + 1) * sizeof(ptrdiff_t));
        gridz_z_ownership_ranges(&grid, ownership_ranges);

        const int start_rank = lower_bound(z_min, ownership_ranges, size + 1);
        const int end_rank = lower_bound(z_max - 1, ownership_ranges, size + 1);

        assert(start_rank < size);
        assert(end_rank < size);

        int *recv_starts = (int *)malloc(size * sizeof(int));
        int *recv_count = (int *)malloc(size * sizeof(int));

        memset(recv_starts, 0, size * sizeof(int));
        memset(recv_count, 0, size * sizeof(int));

        int *send_starts = (int *)malloc(size * sizeof(int));
        int *send_count = (int *)malloc(size * sizeof(int));

        ptrdiff_t check_z_extent = 0;
        for (int r = start_rank; r <= end_rank; r++) {
            const int s = (int)MAX(ownership_ranges[r], z_min) - ownership_ranges[r];
            const ptrdiff_t e = MIN(ownership_ranges[r + 1], z_max) - ownership_ranges[r];
            recv_starts[r] = s;
            recv_count[r] = (int)(e - s);
            check_z_extent += recv_count[r];
            assert(recv_count[r] >= 0);
        }

        assert(check_z_extent == z_extent);

        MPI_Alltoall(recv_starts, 1, MPI_INT, send_starts, 1, MPI_INT, comm);
        MPI_Alltoall(recv_count, 1, MPI_INT, send_count, 1, MPI_INT, comm);

        /////////////////////////////////////////////////////////////////////////
        // Exchange grid data
        /////////////////////////////////////////////////////////////////////////

        // const count_t remote_grid_size = x_extent * y_extent * z_extent;

        // FIXME? For coding convenience we are over-communicating in the x-y plane
        const count_t remote_grid_size = grid.extent[0] * grid.extent[1] * z_extent;
        real_t *remote_grid_field = (real_t *)malloc(remote_grid_size * sizeof(real_t));

        MPI_Alltoallv(box_field,
                      send_count,
                      send_starts,
                      SFEM_MPI_REAL_T,
                      remote_grid_field,
                      recv_count,
                      recv_starts,
                      SFEM_MPI_REAL_T,
                      comm);

        /////////////////////////////////////////////////////////////////////////
        // Shift affine transform for sub-region of grid data
        /////////////////////////////////////////////////////////////////////////

        affine_transform_t subregion_trafo;
        affine_transform_copy(&trafo, &subregion_trafo);
        subregion_trafo.shift[2] = trafo.shift[2] + (trafo.scaling[2] * z_min);

        count_t subregion_n[3];

        subregion_n[0] = grid.extent[0];
        subregion_n[1] = grid.extent[1];
        subregion_n[2] = z_extent;

        /////////////////////////////////////////////////////////////////////////
        // Transfer data
        /////////////////////////////////////////////////////////////////////////

        resample_box_to_tetra_mesh_unique(subregion_n,
                                          stride,
                                          &subregion_trafo,
                                          remote_grid_field,
                                          mesh.nelements,
                                          mesh.nnodes,
                                          mesh.elements,
                                          mesh.points,
                                          mesh_field);

        /////////////////////////////////////////////////////////////////////////
        // Clean-up
        /////////////////////////////////////////////////////////////////////////

        free(ownership_ranges);
        free(recv_starts);
        free(recv_count);
        free(send_starts);
        free(send_count);
        free(remote_grid_field);
    } else {
        // Serial transfer

        // resample_box_to_tetra_mesh(
        resample_box_to_tetra_mesh_unique(
            // resample_box_to_tetra_mesh_cell_list(
            n,
            stride,
            &trafo,
            box_field,
            mesh.nelements,
            mesh.nnodes,
            mesh.elements,
            mesh.points,
            mesh_field);
    }

    tack = MPI_Wtime() - tack;

    if (!rank) {
        printf("----------------------------------------\n");
        printf("resample_box_to_tetra_mesh:\t%g seconds\n", tack);
    }

    {
        real_t min_field = array_min_r(size_field, box_field);
        real_t max_field = array_max_r(size_field, box_field);

        printf("input field (%g, %g)\n", (double)min_field, (double)max_field);
    }

    {
        real_t min_field = array_min_r(mesh.nnodes, mesh_field);
        real_t max_field = array_max_r(mesh.nnodes, mesh_field);

        printf("output field (%g, %g)\n", (double)min_field, (double)max_field);
    }

    mesh_write_nodal_field(&mesh, output_path, SFEM_MPI_REAL_T, (void *)mesh_field);

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
