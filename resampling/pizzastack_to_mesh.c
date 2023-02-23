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
                               geom_t *const SFEM_RESTRICT x,
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

static SFEM_INLINE geom_t array_max(const ptrdiff_t n, const geom_t *a) {
    geom_t ret = a[0];
    for (ptrdiff_t i = 1; i < n; ++i) {
        ret = MAX(ret, a[i]);
    }

    return ret;
}

static SFEM_INLINE geom_t array_max_range(const ptrdiff_t n, const geom_t *start, const geom_t *end) {
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

static SFEM_INLINE cell_list_1D_query_t cell_list_1D_query(const cell_list_1D_t *cl,
                                                           const geom_t bi_min,
                                                           const geom_t bi_max) {
    cell_list_1D_query_t ret;
    ret.begin = MAX(0, floor(cl->scaling * (bi_min + cl->shift)));
    ret.end = MIN(ceil(cl->scaling * (bi_max + cl->shift)) + 1, cl->n_cells);
    return ret;
}

void cell_list_1D_create(cell_list_1D_t *cl,
                         const ptrdiff_t n,
                         const geom_t *SFEM_RESTRICT bi_min,
                         const geom_t *SFEM_RESTRICT bi_max) {
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

void cell_list_1D_destroy(cell_list_1D_t *cl) {
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

void quadrature_create(quadrature_t *q, const int size) {
    q->size = size;
    q->x = (real_t *)malloc(size * sizeof(real_t));
    q->y = (real_t *)malloc(size * sizeof(real_t));
    q->z = (real_t *)malloc(size * sizeof(real_t));
    q->w = (real_t *)malloc(size * sizeof(real_t));
}

void quadrature_destroy(quadrature_t *q) {
    q->size = 0;
    free(q->x);
    free(q->y);
    free(q->z);
    free(q->w);
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
    real_t box_min[3];
    real_t box_range[3];
    real_t x0[3], x1[3], x2[3], x3[3];
    real_t qp[3];

    box_min[0] = x_min;
    box_min[1] = y_min;
    box_min[2] = z_min;

    box_range[0] = x_max - x_min;
    box_range[1] = y_max - y_min;
    box_range[2] = z_max - z_min;

    // Move to reference cube
    x0[0] = (x[0] - box_min[0]) / box_range[0];
    x0[1] = (y[0] - box_min[1]) / box_range[1];
    x0[2] = (z[0] - box_min[2]) / box_range[2];

    x1[0] = (x[1] - box_min[0]) / box_range[0];
    x1[1] = (y[1] - box_min[1]) / box_range[1];
    x1[2] = (z[1] - box_min[2]) / box_range[2];

    x2[0] = (x[2] - box_min[0]) / box_range[0];
    x2[1] = (y[2] - box_min[1]) / box_range[1];
    x2[2] = (z[2] - box_min[2]) / box_range[2];

    x3[0] = (x[3] - box_min[0]) / box_range[0];
    x3[1] = (y[3] - box_min[1]) / box_range[1];
    x3[2] = (z[3] - box_min[2]) / box_range[2];

    // Create local coordinate system for tet4
    for (int d = 0; d < 3; ++d) {
        x1[d] -= x0[d];
        x2[d] -= x0[d];
        x3[d] -= x0[d];
    }

    // Generate quadrature points and test for containment
    q_box->size = 0;
    real_t measure = 0;
    for (int k = 0; k < q->size; k++) {
        int discard = 0;
        for (int d = 0; d < 3; ++d) {
            qp[d] = x0[d] + q->x[k] * x1[d] + q->y[k] * x2[d] + q->z[k] * x2[d];
            discard += qp[d] < 0 || qp[d] > 1;
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

void resample_box_to_tetra_mesh(const count_t n[3],
                                const count_t ld[3],
                                const real_t *SFEM_RESTRICT box_field,
                                const ptrdiff_t n_elements,
                                const ptrdiff_t n_nodes,
                                idx_t **const SFEM_RESTRICT elems,
                                geom_t **const SFEM_RESTRICT xyz,
                                real_t *const SFEM_RESTRICT mesh_field) {
    geom_t *zbi_min = (geom_t *)malloc(n_elements * sizeof(geom_t));
    geom_t *zbi_max = (geom_t *)malloc(n_elements * sizeof(geom_t));
    cell_list_1D_t cl;

    memset(mesh_field, 0, n_nodes * sizeof(real_t));

    real_t * weight_field = (real_t * )malloc(n_nodes * sizeof(real_t));
    memset(weight_field, 0, n_nodes * sizeof(real_t));


    bounding_intervals(n_elements, 4, elems, xyz[2], zbi_min, zbi_max);
    cell_list_1D_create(&cl, n_elements, zbi_min, zbi_max);

    quadrature_t q_ref;
    quadrature_t q_box;
    quadrature_t q_tet;

    int n_qp = 1;
    quadrature_create(&q_ref, n_qp);
    quadrature_create(&q_box, n_qp);
    quadrature_create(&q_tet, n_qp);

    q_ref.x[0] = 0.25;
    q_ref.y[0] = 0.25;
    q_ref.z[0] = 0.25;
    q_ref.w[0] = 1.0 / 6.;

    geom_t xe[4], ye[4], ze[4];
    real_t box_nodal_values[8];
    real_t tet_nodal_values[4];

    for (count_t z = 0; z < n[2]; z++) {
        cell_list_1D_query_t q = cell_list_1D_query(&cl, z - 0.5, z + 0.5);
        assert(q.begin >= 0);
        assert(q.end <= cl.n_entries);

        printf("query %ld: ", (long)z);
        for (ptrdiff_t k = q.begin; k < q.end; k++) {
            const ptrdiff_t e = cl.idx[k];

            for (int d = 0; d < 4; d++) {
                idx_t node = elems[d][e];
                xe[d] = xyz[0][node];
                ye[d] = xyz[1][node];
                ze[d] = xyz[2][node];
            }

            for (ptrdiff_t y = 0; y < n[1]; y++) {
                for (ptrdiff_t x = 0; x < n[0]; x++) {
                    // const geom_t x_min = x - 0.5;
                    // const geom_t y_min = y - 0.5;
                    // const geom_t z_min = z - 0.5;

                    // const geom_t x_max = x + 0.5;
                    // const geom_t y_max = y + 0.5;
                    // const geom_t z_max = z + 0.5;

                    const geom_t x_min = x;
                    const geom_t y_min = y;
                    const geom_t z_min = z;

                    const geom_t x_max = x + 1;
                    const geom_t y_max = y + 1;
                    const geom_t z_max = z + 1;

                    box_tet_quadrature(&q_ref, x_min, y_min, z_min, x_max, y_max, z_max, xe, ye, ze, &q_box, &q_tet);

                    // No intersection
                    if (!q_box.size) continue;

                    // z-bottom
                    box_nodal_values[0] = box_field[x * ld[0] + y * ld[1] + z * ld[2]];
                    box_nodal_values[1] = box_field[(x + 1) * ld[0] + y * ld[1] + z * ld[2]];
                    box_nodal_values[2] = box_field[x * ld[0] + (y + 1) * ld[1] + z * ld[2]];
                    box_nodal_values[3] = box_field[(x + 1) * ld[0] + (y + 1) * ld[1] + z * ld[2]];

                    // z-top
                    box_nodal_values[4] = box_field[x * ld[0] + y * ld[1] + (z + 1) * ld[2]];
                    box_nodal_values[5] = box_field[(x + 1) * ld[0] + y * ld[1] + (z + 1) * ld[2]];
                    box_nodal_values[6] = box_field[x * ld[0] + (y + 1) * ld[1] + (z + 1) * ld[2]];
                    box_nodal_values[7] = box_field[(x + 1) * ld[0] + (y + 1) * ld[1] + (z + 1) * ld[2]];

                    memset(tet_nodal_values, 0, sizeof(tet_nodal_values));

                    real_t measure = 0;
                    for (int k = 0; k < q_box.size; ++k) {
                        const real_t xk = q_box.x[k];
                        const real_t yk = q_box.y[k];
                        const real_t zk = q_box.z[k];

                        real_t value = 0;
                        //  z-bottom
                        value += (1 - xk) * (1 - yk) * (1 - zk) * box_nodal_values[0];
                        value += xk * (1 - yk) * (1 - zk) * box_nodal_values[1];
                        value += (1 - xk) * yk * (1 - zk) * box_nodal_values[2];
                        value += xk * yk * (1 - zk) * box_nodal_values[3];

                        //  z-top
                        value += (1 - xk) * (1 - yk) * zk * box_nodal_values[4];
                        value += xk * (1 - yk) * zk * box_nodal_values[5];
                        value += (1 - xk) * yk * zk * box_nodal_values[6];
                        value += xk * yk * zk * box_nodal_values[7];

                        // Scale by quadrature weight
                        value *= q_tet.w[k];

                        tet_nodal_values[0] += (1 - q_tet.x[k] - q_tet.y[k] - q_tet.z[k]) * value;
                        tet_nodal_values[1] += q_tet.x[k] * value;
                        tet_nodal_values[2] += q_tet.y[k] * value;
                        tet_nodal_values[3] += q_tet.z[k] * value;

                        measure += q_tet.w[k];
                    }


                    for (int d = 0; d < 4; d++) {
                        idx_t node = elems[d][e];
                        mesh_field[node] += tet_nodal_values[d];
                        weight_field[node] += measure;
                    }
                }
            }
        }
    }

    for(ptrdiff_t i = 0; i < n_nodes; ++i) {
        mesh_field[i] /= weight_field[i];
    }

    quadrature_destroy(&q_ref);
    quadrature_destroy(&q_box);
    quadrature_destroy(&q_tet);

    free(zbi_min);
    free(zbi_max);
    free(weight_field);
    cell_list_1D_destroy(&cl);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (argc < 6) {
        if (!rank) {
            fprintf(stderr, "usage: %s <nx> <ny> <nz> <field.raw> <mesh_folder> [output_path=./mesh_field.raw]", argv[0]);
        }

        return EXIT_FAILURE;
    }

    count_t n[3];
    count_t ld[3];

    n[0] = atol(argv[1]);
    n[1] = atol(argv[2]);
    n[2] = atol(argv[3]);
    const ptrdiff_t size_field = n[0] * n[1] * n[2];

    ld[0] = 0;
    ld[1] = n[0];
    ld[2] = n[1] * n[1];

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
                    box_field[z * ld[2] + y * ld[1] + x * ld[0]] = point[2] * point[2];
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

    resample_box_to_tetra_mesh(n, ld, box_field, mesh.nelements, mesh.nnodes, mesh.elements, mesh.points, mesh_field);

    // FIXME!
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
