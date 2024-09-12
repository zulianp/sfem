#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "matrixio_array.h"
#include "matrixio_ndarray.h"

#include "read_mesh.h"
#include "sfem_mesh_write.h"
#include "sfem_resample_gap.h"

#include "point_triangle_distance.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define SDF_SQRT sqrtf
#define SDF_CEIL ceilf

int write_metadata(const char* meta_data_path,
                   const char* data_path,
                   const ptrdiff_t* size,
                   const geom_t* origin,
                   const geom_t* delta) {
    FILE* file = fopen(meta_data_path, "w");
    if (!file) return EXIT_FAILURE;
    fprintf(file, "nx: %ld\n", size[0]);
    fprintf(file, "ny: %ld\n", size[1]);
    fprintf(file, "nz: %ld\n", size[2]);
    fprintf(file, "block_size: 1\n");
    fprintf(file, "type: float\n");
    fprintf(file, "ox: %.15f\n", origin[0]);
    fprintf(file, "oy: %.15f\n", origin[1]);
    fprintf(file, "oz: %.15f\n", origin[2]);
    fprintf(file, "dx: %.15f\n", delta[0]);
    fprintf(file, "dy: %.15f\n", delta[1]);
    fprintf(file, "dz: %.15f\n", delta[2]);
    fprintf(file, "path: %s\n", data_path);
    fclose(file);
    return EXIT_SUCCESS;
}

static SFEM_INLINE void minmax(const ptrdiff_t n,
                               const geom_t* const SFEM_RESTRICT x,
                               geom_t* xmin,
                               geom_t* xmax) {
    *xmin = x[0];
    *xmax = x[0];
    for (ptrdiff_t i = 1; i < n; i++) {
        *xmin = MIN(*xmin, x[i]);
        *xmax = MAX(*xmax, x[i]);
    }
}

static ptrdiff_t array_sum_ptrdiff_t(const ptrdiff_t n, const ptrdiff_t* const SFEM_RESTRICT a) {
    ptrdiff_t tot = 0;
    for (ptrdiff_t i = 0; i < n; i++) {
        tot += a[i];
    }

    return tot;
}

static SFEM_INLINE void cross3(const geom_t* const SFEM_RESTRICT u,
                               const geom_t* const SFEM_RESTRICT v,
                               geom_t* const SFEM_RESTRICT n) {
    n[0] = u[1] * v[2] - u[2] * v[1];
    n[1] = u[2] * v[0] - u[0] * v[2];
    n[2] = u[0] * v[1] - u[1] * v[0];
}

static SFEM_INLINE void diff3(const geom_t* const SFEM_RESTRICT p,
                              const geom_t* const SFEM_RESTRICT q,
                              geom_t* const SFEM_RESTRICT diff) {
    diff[0] = p[0] - q[0];
    diff[1] = p[1] - q[1];
    diff[2] = p[2] - q[2];
}

static SFEM_INLINE void normalize(geom_t* const vec3) {
    const geom_t len = SDF_SQRT(vec3[0] * vec3[0] + vec3[1] * vec3[1] + vec3[2] * vec3[2]);
    vec3[0] /= len;
    vec3[1] /= len;
    vec3[2] /= len;
}

static void compute_vertex_pseudo_normals_3(const ptrdiff_t nelements,
                                            const ptrdiff_t nnodes,
                                            idx_t** const SFEM_RESTRICT elements,
                                            geom_t** const SFEM_RESTRICT points,
                                            geom_t** const SFEM_RESTRICT normals) {
    const double tick = MPI_Wtime();

    for (int d = 0; d < 3; d++) {
        memset(normals[d], 0, nnodes * sizeof(geom_t));
    }

#pragma omp parallel
    {
#pragma omp for
        for (ptrdiff_t e = 0; e < nelements; e++) {
            const idx_t i0 = elements[0][e];
            const idx_t i1 = elements[1][e];
            const idx_t i2 = elements[2][e];

            const geom_t u[3] = {points[0][i1] - points[0][i0],
                                 points[1][i1] - points[1][i0],
                                 points[2][i1] - points[2][i0]};

            const geom_t v[3] = {points[0][i2] - points[0][i0],
                                 points[1][i2] - points[1][i0],
                                 points[2][i2] - points[2][i0]};
            geom_t n[3];
            cross3(u, v, n);
            normalize(n);

            for (int d = 0; d < 3; d++) {
#pragma omp atomic update
                normals[d][i0] += n[d];

#pragma omp atomic update
                normals[d][i1] += n[d];

#pragma omp atomic update
                normals[d][i2] += n[d];
            }
        }
    }

#pragma omp parallel
    {
#pragma omp for
        for (ptrdiff_t i = 0; i < nnodes; i++) {
            geom_t n[3] = {
                normals[0][i],
                normals[1][i],
                normals[2][i],
            };

            normalize(n);
            normals[0][i] = n[0];
            normals[1][i] = n[1];
            normals[2][i] = n[2];
        }
    }

    const double tock = MPI_Wtime();
    printf("mesh_to_sdf.c: compute_vertex_pseudo_normals_3:\t\t\t%g seconds\n", tock - tick);
}

ptrdiff_t select_submesh(const ptrdiff_t nelements,
                         const int nodesxelem,
                         const ptrdiff_t nnodes,
                         idx_t** const SFEM_RESTRICT elements,
                         geom_t** const SFEM_RESTRICT points,
                         const geom_t* SFEM_RESTRICT box_min,
                         const geom_t* SFEM_RESTRICT box_max) {
    const double tick = MPI_Wtime();

    short* is_node_inside = (short*)malloc(nnodes * sizeof(short));
    for (ptrdiff_t i = 0; i < nnodes; i++) {
        const geom_t x = points[0][i];
        const geom_t y = points[1][i];
        const geom_t z = points[2][i];

        is_node_inside[i] = x > box_min[0] && y > box_min[1] && z > box_min[2] && x < box_max[0] &&
                            y < box_max[1] && z < box_max[2];
    }

    ptrdiff_t removed_elements = 0;
    for (ptrdiff_t i = 0; i < nelements; i++) {
        int contained = 0;
        for (int de = 0; de < nodesxelem; de++) {
            idx_t nn = elements[de][i];
            contained += is_node_inside[nn];
        }

        if (!contained) {
            for (int de = 0; de < nodesxelem; de++) {
                // Remove element
                elements[de][i] = -1;
            }

            removed_elements++;
        }
    }

    free(is_node_inside);

    for (int de = 0; de < nodesxelem; de++) {
        ptrdiff_t n_valid = 0;
        for (ptrdiff_t i = 0; i < nelements; i++) {
            if (elements[de][i] == -1) continue;
            elements[de][n_valid++] = elements[de][i];
        }

        assert(n_valid == (nelements - removed_elements));
    }

    const double tock = MPI_Wtime();
    printf("mesh_to_sdf.c: select_submesh:\t\t\t%g seconds\n", tock - tick);
    return nelements - removed_elements;
}

void compute_sdf_brute_force(const ptrdiff_t nelements,
                             idx_t** const SFEM_RESTRICT elements,
                             geom_t** const SFEM_RESTRICT points,
                             geom_t** const SFEM_RESTRICT normals,
                             const ptrdiff_t* SFEM_RESTRICT size,
                             const ptrdiff_t* SFEM_RESTRICT stride,
                             const geom_t* SFEM_RESTRICT origin,
                             const geom_t* SFEM_RESTRICT delta,
                             geom_t* const SFEM_RESTRICT sdf) {
    static const geom_t infty = 1e8;

    const double tick = MPI_Wtime();

#pragma omp parallel
    {
#pragma omp for
        for (ptrdiff_t k = 0; k < size[2]; k++) {
            for (ptrdiff_t j = 0; j < size[1]; j++) {
                for (ptrdiff_t i = 0; i < size[0]; i++) {
                    geom_t e_min = infty;
                    geom_t e_sign = 1;

                    // Brute force
                    for (ptrdiff_t e = 0; e < nelements; e++) {
                        geom_t temp = infty;

                        const idx_t i0 = elements[0][e];
                        const idx_t i1 = elements[1][e];
                        const idx_t i2 = elements[2][e];

                        const geom_t p[3] = {origin[0] + i * delta[0],
                                             origin[1] + j * delta[1],
                                             origin[2] + k * delta[2]};

                        const geom_t x[3] = {points[0][i0], points[0][i1], points[0][i2]};
                        const geom_t y[3] = {points[1][i0], points[1][i1], points[1][i2]};
                        const geom_t z[3] = {points[2][i0], points[2][i1], points[2][i2]};

                        geom_t n[3] = {0., 0., 0.};

                        point_triangle_distance_result_t result;
                        point_triangle_distance(p, x, y, z, &result);

                        geom_t diff[3];
                        diff3(p, result.point, diff);
                        const geom_t d =
                            SDF_SQRT(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);

                        if (d < e_min) {
                            e_min = d;

                            if (d == 0) {
                                e_sign = 1;
                            } else {
                                const geom_t phi1 = result.s;
                                const geom_t phi2 = result.t;
                                const geom_t phi0 = 1.0 - phi1 - phi2;

                                n[0] = phi0 * normals[0][i0] + phi1 * normals[0][i1] +
                                       phi2 * normals[0][i2];

                                n[1] = phi0 * normals[1][i0] + phi1 * normals[1][i1] +
                                       phi2 * normals[1][i2];

                                n[2] = phi0 * normals[2][i0] + phi1 * normals[2][i1] +
                                       phi2 * normals[2][i2];

                                normalize(n);
                                e_sign = signbit(dot3(diff, n)) ? -1 : 1;
                            }
                        }
                    }

                    sdf[k * stride[2] + j * stride[1] + i * stride[0]] = e_sign * e_min;
                }
            }
        }
    }

    const double tock = MPI_Wtime();
    printf("mesh_to_sdf.c: compute_sdf:\t\t\t%g seconds\n", tock - tick);
}

void compute_sdf(const ptrdiff_t nelements,
                 idx_t** const SFEM_RESTRICT elements,
                 geom_t** const SFEM_RESTRICT points,
                 geom_t** const SFEM_RESTRICT normals,
                 const ptrdiff_t* SFEM_RESTRICT size,
                 const ptrdiff_t* SFEM_RESTRICT stride,
                 const geom_t* SFEM_RESTRICT origin,
                 const geom_t* SFEM_RESTRICT delta,
                 geom_t* const SFEM_RESTRICT sdf) {
    static const geom_t infty = 1e8;

    const double tick = MPI_Wtime();

    const idx_t en0 = elements[0][0];

    geom_t box_min[3] = {points[0][en0], points[1][en0], points[2][en0]};
    geom_t box_max[3] = {points[0][en0], points[1][en0], points[2][en0]};

    // Max element extent
    geom_t hmax = box_max[0] - box_min[0];

    {
        // Compute mesh bounding-box and max element extent
        for (ptrdiff_t e = 0; e < nelements; e++) {
            for (int d = 0; d < 3; d++) {
                geom_t min_x = points[d][elements[0][e]];
                geom_t max_x = min_x;

                for (int ni = 0; ni < 3; ni++) {
                    geom_t x = points[d][elements[ni][e]];

                    box_min[d] = MIN(box_min[d], x);
                    box_max[d] = MAX(box_max[d], x);

                    min_x = MIN(min_x, x);
                    max_x = MAX(max_x, x);
                }

                hmax = MAX(hmax, max_x - min_x);
            }
        }
    }

#define SUBDIVISION_LEVELS 1
    geom_t grid_spacing[SUBDIVISION_LEVELS][3];
    int grid_size[SUBDIVISION_LEVELS][3];
    ptrdiff_t* element_count[SUBDIVISION_LEVELS];

    // Fine level arrays!
    int* g_finest_size = grid_size[SUBDIVISION_LEVELS - 1];
    geom_t* g_finest_spacing = grid_spacing[SUBDIVISION_LEVELS - 1];
    ptrdiff_t* g_finest_el_count;
    ptrdiff_t g_finest_stride[3] = {1, -1, -1};

    geom_t multiple = 2;
    geom_t H = multiple * hmax * pow(2, SUBDIVISION_LEVELS - 1);

    {  // Construct grids
        for (int l = 0; l < SUBDIVISION_LEVELS; l++) {
            ptrdiff_t nbins = 1;
            for (int d = 0; d < 3; d++) {
                grid_size[l][d] = MAX(1, ((box_max[d] - box_min[d]) / H) * pow(2, l)) + 1;
                grid_spacing[l][d] = (box_max[d] - box_min[d]) / (grid_size[l][d] - 1);
                nbins *= grid_size[l][d];
            }

            element_count[l] = calloc(nbins, sizeof(ptrdiff_t));
        }

        g_finest_stride[1] = g_finest_size[0];
        g_finest_stride[2] = g_finest_size[0] * g_finest_size[1];
        g_finest_el_count = element_count[SUBDIVISION_LEVELS - 1];
    }

    {
        // Count element contained in cells
        for (ptrdiff_t e = 0; e < nelements; e++) {
            int start[3] = {0, 0, 0};
            int end[3] = {g_finest_size[0], g_finest_size[1], g_finest_size[2]};

            const idx_t i0 = elements[0][e];
            geom_t e_box_min[3] = {points[0][i0], points[1][i0], points[2][i0]};
            geom_t e_box_max[3] = {points[0][i0], points[1][i0], points[2][i0]};

            for (int d = 0; d < 3; d++) {
                for (int ni = 1; ni < 3; ni++) {
                    e_box_min[d] = MIN(e_box_min[d], points[d][elements[ni][e]]);
                    e_box_max[d] = MAX(e_box_max[d], points[d][elements[ni][e]]);
                }
            }

            for (int d = 0; d < 3; d++) {
                const geom_t h = g_finest_spacing[d];
                ptrdiff_t i_begin = (e_box_min[d] - box_min[d]) / h;
                start[d] = MAX(start[d], i_begin);

                ptrdiff_t i_end = floorf((e_box_max[d] - box_min[d]) / h) + 1;
                end[d] = MIN(end[d], i_end);
            }

            for (int k = start[2]; k < end[2]; k++) {
                for (int j = start[1]; j < end[1]; j++) {
                    for (int i = start[0]; i < end[0]; i++) {
                        ptrdiff_t cell_idx = i * g_finest_stride[0] + j * g_finest_stride[1] +
                                             k * g_finest_stride[2];

                        g_finest_el_count[cell_idx]++;
                    }
                }
            }

            // {
            //     // Only min
            //     ptrdiff_t cell_idx = start[0] * g_finest_stride[0] + start[1] * g_finest_stride[1] +
            //                          start[2] * g_finest_stride[2];
            //     g_finest_el_count[cell_idx]++;
            // }
        }
    }

    const ptrdiff_t g_finest_nnodes = grid_size[SUBDIVISION_LEVELS - 1][0] *
                                      grid_size[SUBDIVISION_LEVELS - 1][1] *
                                      grid_size[SUBDIVISION_LEVELS - 1][2];

    ptrdiff_t* g_finest_cell_ptr = malloc((g_finest_nnodes + 1) * sizeof(ptrdiff_t));
    g_finest_cell_ptr[0] = 0;

    for (ptrdiff_t i = 0; i < g_finest_nnodes; i++) {
        g_finest_cell_ptr[i + 1] = g_finest_cell_ptr[i] + g_finest_el_count[i];
    }

    ptrdiff_t* g_finest_cell_idx = calloc(g_finest_cell_ptr[g_finest_nnodes], sizeof(ptrdiff_t));

    {
        // FIll cell list
        for (ptrdiff_t e = 0; e < nelements; e++) {
            int start[3] = {0, 0, 0};
            int end[3] = {g_finest_size[0], g_finest_size[1], g_finest_size[2]};

            geom_t e_box_min[3] = {
                points[0][elements[0][e]], points[1][elements[0][e]], points[2][elements[0][e]]};

            geom_t e_box_max[3] = {
                points[0][elements[0][e]], points[1][elements[0][e]], points[2][elements[0][e]]};

            for (int d = 0; d < 3; d++) {
                for (int ni = 1; ni < 3; ni++) {
                    e_box_min[d] = MIN(e_box_min[d], points[d][elements[ni][e]]);
                    e_box_max[d] = MAX(e_box_max[d], points[d][elements[ni][e]]);
                }
            }

            for (int d = 0; d < 3; d++) {
                const geom_t h = g_finest_spacing[d];
                ptrdiff_t i_begin = (e_box_min[d] - box_min[d]) / h;
                start[d] = MAX(start[d], i_begin);

                ptrdiff_t i_end = floorf((e_box_max[d] - box_min[d]) / h) + 1;
                end[d] = MIN(end[d], i_end);
            }

            for (int k = start[2]; k < end[2]; k++) {
                for (int j = start[1]; j < end[1]; j++) {
                    for (int i = start[0]; i < end[0]; i++) {
                        ptrdiff_t cell_idx = i * g_finest_stride[0] + j * g_finest_stride[1] +
                                             k * g_finest_stride[2];

                        g_finest_cell_idx[g_finest_cell_ptr[cell_idx]++] = e;
                    }
                }
            }

            // {
            //     // Only min
            //     ptrdiff_t cell_idx = start[0] * g_finest_stride[0] + start[1] * g_finest_stride[1] +
            //                          start[2] * g_finest_stride[2];

            //     g_finest_cell_idx[g_finest_cell_ptr[cell_idx]++] = e;
            // }
        }
    }

    // Reset cell ptr
    g_finest_cell_ptr[0] = 0;
    for (ptrdiff_t i = 0; i < g_finest_nnodes; i++) {
        g_finest_cell_ptr[i + 1] = g_finest_cell_ptr[i] + g_finest_el_count[i];
    }

    printf("size (%ld, %ld, %ld)\n", size[0], size[1], size[2]);
    printf("detection grid (%d, %d, %d)\n", g_finest_size[0], g_finest_size[1], g_finest_size[2]);
    printf("num cell_idx %ld\n", g_finest_cell_ptr[g_finest_nnodes]);

    for (ptrdiff_t i = 0; i < g_finest_nnodes; i++) {
        printf("%ld ", g_finest_el_count[i]);
    }

    printf("\n");

    {
        const double tock = MPI_Wtime();
        printf("mesh_to_sdf.c: compute_sdf (cell-list prep)\t\t\t%g seconds\n", tock - tick);
    }

#pragma omp parallel
    {
#pragma omp for
        for (ptrdiff_t k = 0; k < size[2]; k++) {
            for (ptrdiff_t j = 0; j < size[1]; j++) {
                for (ptrdiff_t i = 0; i < size[0]; i++) {
                    const geom_t p[3] = {origin[0] + i * delta[0],
                                         origin[1] + j * delta[1],
                                         origin[2] + k * delta[2]};

                    int start[3] = {0, 0, 0};
                    int end[3] = {0, 0, 0};

                    for (int d = 0; d < 3; d++) {
                        geom_t xi = (p[d] - box_min[d]) / (g_finest_spacing[d]);
                        start[d] = MAX(0, floorf(xi));
                        end[d] = MIN(g_finest_size[d], ceilf(xi) + 1);
                    }

                    int search_radius = 1000000;
                    for (int d = 0; d < 3; d++) {
                        // Expand to the left
                        {
                            int extent[3] = {0, 0, 0};
                            while (start[d] - extent[d] > 0) {
                                const ptrdiff_t cell_idx =
                                    (start[0] - extent[0]) * g_finest_stride[0] +
                                    (start[1] - extent[1]) * g_finest_stride[1] +
                                    (start[2] - extent[2]) * g_finest_stride[2];

                                extent[d]++;
                                if (g_finest_el_count[cell_idx] != 0 ||
                                    extent[d] >= search_radius) {
                                    search_radius = MIN(extent[d], search_radius);
                                    break;
                                }
                            }
                        }

                        // Expand to the right
                        {
                            int extent[3] = {0, 0, 0};
                            while (end[d] + extent[d] < g_finest_size[d] - 1) {
                                const ptrdiff_t cell_idx =
                                    (end[0] + extent[0]) * g_finest_stride[0] +
                                    (end[1] + extent[1]) * g_finest_stride[1] +
                                    (end[2] + extent[2]) * g_finest_stride[2];
                                extent[d] += 1;
                                if (g_finest_el_count[cell_idx] != 0 ||
                                    extent[d] >= search_radius) {
                                    search_radius = MIN(extent[d], search_radius);
                                    break;
                                }
                            }
                        }
                    }

                    for (int d = 0; d < 3; d++) {
                        start[d] = MAX(0, start[d] - search_radius);
                        end[d] = MIN(end[d] + search_radius, g_finest_size[d]);
                        assert(end[d] - start[d] >= 1);
                    }

                    geom_t e_min = infty;
                    geom_t e_sign = 1;

                    for (int kk = start[2]; kk < end[2]; ++kk) {
                        for (int jj = start[1]; jj < end[1]; ++jj) {
                            for (int ii = start[0]; ii < end[0]; ++ii) {
                                ptrdiff_t cell_idx = ii * g_finest_stride[0] +
                                                     jj * g_finest_stride[1] +
                                                     kk * g_finest_stride[2];

                                for (ptrdiff_t m = g_finest_cell_ptr[cell_idx];
                                     m < g_finest_cell_ptr[cell_idx + 1];
                                     m++) {
                                    ptrdiff_t e = g_finest_cell_idx[m];

                                    geom_t temp = infty;

                                    const idx_t i0 = elements[0][e];
                                    const idx_t i1 = elements[1][e];
                                    const idx_t i2 = elements[2][e];

                                    const geom_t x[3] = {
                                        points[0][i0], points[0][i1], points[0][i2]};
                                    const geom_t y[3] = {
                                        points[1][i0], points[1][i1], points[1][i2]};
                                    const geom_t z[3] = {
                                        points[2][i0], points[2][i1], points[2][i2]};

                                    geom_t n[3] = {0., 0., 0.};

                                    point_triangle_distance_result_t result;
                                    point_triangle_distance(p, x, y, z, &result);

                                    geom_t diff[3];
                                    diff3(p, result.point, diff);
                                    const geom_t d = SDF_SQRT(
                                        diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);

                                    if (d < e_min) {
                                        e_min = d;

                                        if (d == 0) {
                                            e_sign = 1;
                                        } else {
                                            const geom_t phi1 = result.s;
                                            const geom_t phi2 = result.t;
                                            const geom_t phi0 = 1.0 - phi1 - phi2;

                                            n[0] = phi0 * normals[0][i0] + phi1 * normals[0][i1] +
                                                   phi2 * normals[0][i2];

                                            n[1] = phi0 * normals[1][i0] + phi1 * normals[1][i1] +
                                                   phi2 * normals[1][i2];

                                            n[2] = phi0 * normals[2][i0] + phi1 * normals[2][i1] +
                                                   phi2 * normals[2][i2];

                                            normalize(n);
                                            e_sign = signbit(dot3(diff, n)) ? -1 : 1;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    sdf[k * stride[2] + j * stride[1] + i * stride[0]] = e_sign * e_min;
                }
            }
        }
    }

    for (int l = 0; l < SUBDIVISION_LEVELS; l++) {
        free(element_count[l]);
    }

    free(g_finest_cell_idx);

    const double tock = MPI_Wtime();
    printf("mesh_to_sdf.c: compute_sdf:\t\t\t%g seconds\n", tock - tick);
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (argc != 5) {
        fprintf(stderr, "usage: %s <mesh> <hmax> <margin> <output_folder>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char* SFEM_BOXED_MESH = 0;
    SFEM_READ_ENV(SFEM_BOXED_MESH, );

    geom_t SFEM_SCALE_BOX = 1;
    SFEM_READ_ENV(SFEM_SCALE_BOX, atof);

    if (!rank) {
        printf(
            "SFEM_BOXED_MESH=%s\n"
            "SFEM_SCALE_BOX=%f\n",
            SFEM_BOXED_MESH,
            SFEM_SCALE_BOX);
    }

    double tick = MPI_Wtime();

    const char* folder = argv[1];
    const geom_t hmax = atof(argv[2]);
    const geom_t margin = atof(argv[3]);
    const char* output_folder = argv[4];

    {
        struct stat st = {0};
        if (stat(output_folder, &st) == -1) {
            mkdir(output_folder, 0700);
        }
    }

    mesh_t mesh;
    if (mesh_read(comm, folder, &mesh)) {
        return EXIT_FAILURE;
    }

    geom_t origin[3], box_max[3];

    {  // AABB
        if (SFEM_BOXED_MESH) {
            // FIXME we do not actually need to read the mesh! only the points!
            mesh_t boxed_mesh;
            if (mesh_read(comm, SFEM_BOXED_MESH, &boxed_mesh)) {
                return EXIT_FAILURE;
            }

            for (int d = 0; d < mesh.spatial_dim; d++) {
                minmax(boxed_mesh.nnodes, boxed_mesh.points[d], &origin[d], &box_max[d]);
            }

            mesh_destroy(&boxed_mesh);
        } else {
            for (int d = 0; d < mesh.spatial_dim; d++) {
                minmax(mesh.nnodes, mesh.points[d], &origin[d], &box_max[d]);
            }
        }

        if (margin != 0) {
            for (int d = 0; d < mesh.spatial_dim; d++) {
                origin[d] -= margin;
                box_max[d] += margin;
            }
        }

        if (SFEM_SCALE_BOX != 1) {
            for (int d = 0; d < mesh.spatial_dim; d++) {
                const geom_t pmean = (origin[d] + box_max[d]) / 2;
                geom_t ppmin = origin[d] - pmean;
                geom_t ppmax = box_max[d] - pmean;
                ppmin *= SFEM_SCALE_BOX;
                ppmax *= SFEM_SCALE_BOX;
                origin[d] = ppmin + pmean;
                box_max[d] = ppmax + pmean;
            }
        }
    }

    // Remove elements we do not need!
    mesh.nelements = select_submesh(mesh.nelements,
                                    elem_num_nodes(mesh.element_type),
                                    mesh.nnodes,
                                    mesh.elements,
                                    mesh.points,
                                    origin,
                                    box_max);

    geom_t** normals = malloc(mesh.spatial_dim * sizeof(geom_t*));
    for (int d = 0; d < 3; d++) {
        normals[d] = malloc(mesh.nnodes * sizeof(geom_t));
    }

    compute_vertex_pseudo_normals_3(
        mesh.nelements, mesh.nnodes, mesh.elements, mesh.points, normals);

    const geom_t x_range = box_max[0] - origin[0];
    const geom_t y_range = box_max[1] - origin[1];
    const geom_t z_range = box_max[2] - origin[2];

    ptrdiff_t nx = SDF_CEIL((x_range) / hmax) + 1;
    ptrdiff_t ny = SDF_CEIL((y_range) / hmax) + 1;
    ptrdiff_t nz = SDF_CEIL((z_range) / hmax) + 1;

    ptrdiff_t nglobal[3] = {nx, ny, nz};
    ptrdiff_t stride[3] = {1, nx, nx * ny};

    geom_t delta[3] = {x_range / (nx - 1.), y_range / (ny - 1.), z_range / (nz - 1.)};

    ptrdiff_t sdf_size = nglobal[0] * nglobal[1] * nglobal[2];
    geom_t* sdf = malloc(sdf_size * sizeof(geom_t));

    // compute_sdf
        compute_sdf_brute_force
        //
        (mesh.nelements, mesh.elements, mesh.points, normals, nglobal, stride, origin, delta, sdf);

    const ptrdiff_t nelements = mesh.nelements;
    const ptrdiff_t nnodes = mesh.nnodes;

    char data_path[2048];
    sprintf(data_path, "%s/sdf.float32.raw", output_folder);

    array_write(comm, data_path, SFEM_MPI_GEOM_T, sdf, sdf_size, sdf_size);

    if (!rank) {
        char meta_data_path[2048];
        sprintf(meta_data_path, "%s/metadata_sdf.float32.yml", output_folder);
        write_metadata(meta_data_path, data_path, nglobal, origin, delta);
    }

    // Free resources
    {
        free(sdf);
        mesh_destroy(&mesh);
    }

    double tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("mesh_to_sdf.c: #elements %ld #nodes %ld #grid (%ld x %ld x %ld)\n",
               (long)nelements,
               (long)nnodes,
               nglobal[0],
               nglobal[1],
               nglobal[2]);
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
