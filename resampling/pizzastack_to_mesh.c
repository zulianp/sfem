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
} cell_list_1D;


void cell_list_1D_print(cell_list_1D *cl)
{   
    printf("shift: %g\n", (double) cl->shift);
    printf("scaling: %g\n", (double) cl->scaling);
    printf("n_cells: %ld\n", (long) cl->n_cells);
    printf("n_entries: %ld\n", (long) cl->n_entries);

    printf("---------------------\n");
    for(ptrdiff_t i = 0; i < cl->n_cells; i++) {
        ptrdiff_t begin = cl->cell_ptr[i];
        ptrdiff_t end = cl->cell_ptr[i+1];

        assert(end <= cl->n_entries);

        printf("%ld)\n", (long)i);
        for(ptrdiff_t k = begin; k < end; k++) {
            printf("%ld ", (long)cl->idx[k]);
        }
        printf("\n");
    }

    printf("---------------------\n");

    for(ptrdiff_t i = 0; i < cl->n_entries; i++) {
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

static SFEM_INLINE cell_list_1D_query_t cell_list_1D_query(const cell_list_1D *cl,
                                                           const geom_t bi_min,
                                                           const geom_t bi_max) {
    cell_list_1D_query_t ret;
    ret.begin = MAX(0, floor(cl->scaling * (bi_min + cl->shift)));
    ret.end = MIN(ceil(cl->scaling * (bi_max + cl->shift)) + 1, cl->n_cells);
    return ret;
}

void cell_list_1D_create(cell_list_1D *cl,
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
        idx[zhisto[cell]++] = i;
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

void cell_list_1D_destroy(cell_list_1D *cl) {
    free(cl->cell_ptr);
    free(cl->idx);

    cl->shift = 0;
    cl->scaling = 0;
    cl->n_cells = 0;
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
    cell_list_1D cl;

    bounding_intervals(n_elements, 4, elems, xyz[2], zbi_min, zbi_max);
    cell_list_1D_create(&cl, n_elements, zbi_min, zbi_max);

    for (count_t z = 0; z < n[2]; z++) {
        cell_list_1D_query_t q = cell_list_1D_query(&cl, z - 0.5, z + 0.5);
        assert(q.begin >= 0);
        assert(q.end <=  cl.n_entries);

        printf("query %ld: ", (long)z);
        for (ptrdiff_t k = q.begin; k < q.end; k++) {
            printf("%ld -> %ld, ", k, cl.idx[k]);
        }

        printf("\n");
    }

    free(zbi_min);
    free(zbi_max);
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
            fprintf(stderr, "usage: %s <nx> <ny> <nz> <field.raw> <mesh_folder> [output_folder=./]", argv[0]);
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

    const char *output_folder = "./";

    if (argc > 6) {
        output_folder = argv[6];
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
                output_folder);
    }

    double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    real_t *box_field;
    ptrdiff_t field_n_local, field_n_global;

    if(strcmp(field_path, "demo") == 0) {
        box_field = (real_t*)malloc(size_field * sizeof(real_t));

        geom_t point[3] = {0, 0, 0};
        for(ptrdiff_t z = 0; z < n[2]; ++z) {   
            point[2] = z / (1.0 * n[2]); 
            for(ptrdiff_t y = 0; y < n[1]; ++y) {
                point[1] = y / (1.0 * n[1]);
                for(ptrdiff_t x = 0; x < n[0]; ++x) {
                    point[0] = x / (1.0 * n[0]);
                    box_field[z*ld[2] + y*ld[1] + x*ld[0]] = point[2] * point[2];
                }
            }
        }

    } else {
        array_read(comm, field_path, SFEM_MPI_REAL_T, (void**)&box_field, &field_n_local, &field_n_global);
        assert(size_field == field_n_global);
    }

    mesh_t mesh;
    if (mesh_read(comm, mesh_folder, &mesh)) {
        return EXIT_FAILURE;
    }

    real_t *mesh_field = (real_t *)malloc(mesh.nnodes * sizeof(real_t));
    memset(mesh_field, 0, mesh.nnodes * sizeof(real_t));

    resample_box_to_tetra_mesh(n, ld, box_field, mesh.nelements, mesh.nnodes, mesh.elements, mesh.points, mesh_field);

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
