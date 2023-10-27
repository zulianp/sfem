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
#define SDF_SQRT sqrt
#define SDF_CEIL ceil

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

                        const geom_t gpx = origin[0] + i * delta[0];
                        const geom_t gpy = origin[1] + j * delta[1];
                        const geom_t gpz = origin[2] + k * delta[2];
                        const geom_t p[3] = {gpx, gpy, gpz};

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
                                if (dot3(diff, n) < 0) {
                                    e_sign = -1;
                                } else {
                                    e_sign = 1;
                                }
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

    geom_t delta[3] = {
        x_range / (nx - 1.), 
        y_range / (ny - 1.), 
        z_range / (nz - 1.)
    };

    ptrdiff_t sdf_size = nglobal[0] * nglobal[1] * nglobal[2];
    geom_t* sdf = malloc(sdf_size * sizeof(geom_t));

    compute_sdf(
        mesh.nelements, mesh.elements, mesh.points, normals, nglobal, stride, origin, delta, sdf);

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
