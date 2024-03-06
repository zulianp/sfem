#include "mesh_utils.h"

idx_t** allocate_elements(const int nxe, const ptrdiff_t n_elements) {
    idx_t** elements = malloc(nxe * sizeof(idx_t*));
    for (int d = 0; d < nxe; d++) {
        elements[d] = malloc(n_elements * sizeof(idx_t));
    }

    return elements;
}

void free_elements(const int nxe, idx_t** elements) {
    for (int d = 0; d < nxe; d++) {
        free(elements[d]);
    }

    free(elements);
}

void select_elements(const int nxe,
                     const ptrdiff_t nselected,
                     const element_idx_t* const idx,
                     idx_t** const SFEM_RESTRICT elements,
                     idx_t** const SFEM_RESTRICT selection) {
    for (int d = 0; d < nxe; d++) {
        for (ptrdiff_t i = 0; i < nselected; i++) {
            selection[d][i] = elements[d][idx[i]];
        }
    }
}

geom_t** allocate_points(const int dim, const ptrdiff_t n_points) {
    geom_t** points = malloc(dim * sizeof(geom_t*));
    for (int d = 0; d < dim; d++) {
        points[d] = malloc(n_points * sizeof(geom_t));
    }

    return points;
}

void free_points(const int dim, geom_t** points) {
    for (int d = 0; d < dim; d++) {
        free(points[d]);
    }

    free(points);
}

void select_points(const int dim,
                   const ptrdiff_t n_points,
                   const idx_t* idx,
                   geom_t** const points,
                   geom_t** const selection) {
    for (int d = 0; d < dim; d++) {
        for (ptrdiff_t i = 0; i < n_points; i++) {
            selection[d][i] = points[d][idx[i]];
        }
    }
}

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

void minmax(const ptrdiff_t n, const geom_t* const SFEM_RESTRICT x, geom_t* xmin, geom_t* xmax) {
    *xmin = x[0];
    *xmax = x[0];
    for (ptrdiff_t i = 1; i < n; i++) {
        *xmin = MIN(*xmin, x[i]);
        *xmax = MAX(*xmax, x[i]);
    }
}
