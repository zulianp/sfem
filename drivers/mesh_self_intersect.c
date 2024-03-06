#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "array_dtof.h"
#include "matrixio_array.h"
#include "matrixio_crs.h"
#include "utils.h"

#include "crs_graph.h"
#include "read_mesh.h"
#include "sfem_base.h"
#include "sfem_mesh_write.h"
#include "sfem_vec.h"

#include "extract_surface_graph.h"

#include "sfem_defs.h"

#include "argsort.h"

#include "adj_table.h"

static SFEM_INLINE void normalize3(real_t* const vec3) {
    const real_t len = sqrt(vec3[0] * vec3[0] + vec3[1] * vec3[1] + vec3[2] * vec3[2]);
    vec3[0] /= len;
    vec3[1] /= len;
    vec3[2] /= len;
}

static SFEM_INLINE real_t dot3(const real_t* const a3, const real_t* const b3) {
    return a3[0] * b3[0] + a3[1] * b3[1] + a3[2] * b3[2];
}

static SFEM_INLINE void normal3(const idx_t i0,
                                const idx_t i1,
                                const idx_t i2,
                                geom_t** const SFEM_RESTRICT xyz,
                                real_t* const SFEM_RESTRICT n) {
    real_t u[3] = {xyz[0][i1] - xyz[0][i0], xyz[1][i1] - xyz[1][i0], xyz[2][i1] - xyz[2][i0]};
    real_t v[3] = {xyz[0][i2] - xyz[0][i0], xyz[1][i2] - xyz[1][i0], xyz[2][i2] - xyz[2][i0]};

    normalize3(u);
    normalize3(v);

    n[0] = u[1] * v[2] - u[2] * v[1];
    n[1] = u[2] * v[0] - u[0] * v[2];
    n[2] = u[0] * v[1] - u[1] * v[0];

    normalize3(n);
}

static SFEM_INLINE void box3(const idx_t i0,
                             const idx_t i1,
                             const idx_t i2,
                             const idx_t i3,
                             geom_t** const SFEM_RESTRICT xyz,
                             geom_t* const SFEM_RESTRICT bmin,
                             geom_t* const SFEM_RESTRICT bmax) {
    for (int d = 0; d < 3; d++) {
        bmin[d] = xyz[d][i0];
        bmax[d] = xyz[d][i0];
    }

    for (int d = 0; d < 3; d++) {
        bmin[d] = MIN(bmin[d], xyz[d][i1]);
        bmax[d] = MAX(bmax[d], xyz[d][i1]);

        bmin[d] = MIN(bmin[d], xyz[d][i2]);
        bmax[d] = MAX(bmax[d], xyz[d][i2]);

        bmin[d] = MIN(bmin[d], xyz[d][i3]);
        bmax[d] = MAX(bmax[d], xyz[d][i3]);
    }
}

static SFEM_INLINE int box_intersect3(const geom_t* const SFEM_RESTRICT bmin1,
                                      const geom_t* const SFEM_RESTRICT bmax1,
                                      const geom_t* const SFEM_RESTRICT bmin2,
                                      const geom_t* const SFEM_RESTRICT bmax2) {
    int outside = 0;
    for (int d = 0; d < 3; d++) {
        outside += bmax1[d] < bmin2[d];
        outside += bmax2[d] < bmin1[d];
    }

    return outside == 0;
}

static SFEM_INLINE void tet4_hpoly3(const idx_t i0,
                                    const idx_t i1,
                                    const idx_t i2,
                                    const idx_t i3,
                                    geom_t** const SFEM_RESTRICT xyz,
                                    real_t* const SFEM_RESTRICT xnormal,
                                    real_t* const SFEM_RESTRICT ynormal,
                                    real_t* const SFEM_RESTRICT znormal,
                                    real_t* const SFEM_RESTRICT d) {
    real_t n[3];
    real_t p[3];

    // Using ordering of Exodus element model

    // f0
    normal3(i0, i1, i3, xyz, n);

    xnormal[0] = n[0];
    ynormal[0] = n[1];
    znormal[0] = n[2];

    p[0] = xyz[0][i0];
    p[1] = xyz[1][i0];
    p[2] = xyz[2][i0];

    d[0] = dot3(n, p);

    // f1
    normal3(i1, i2, i3, xyz, n);

    xnormal[1] = n[0];
    ynormal[1] = n[1];
    znormal[1] = n[2];

    p[0] = xyz[0][i1];
    p[1] = xyz[1][i1];
    p[2] = xyz[2][i1];

    d[1] = dot3(n, p);

    // f2
    normal3(i1, i3, i2, xyz, n);

    xnormal[2] = n[0];
    ynormal[2] = n[1];
    znormal[2] = n[2];

    p[0] = xyz[0][i2];
    p[1] = xyz[1][i2];
    p[2] = xyz[2][i2];

    d[2] = dot3(n, p);

    // f3
    normal3(i1, i2, i0, xyz, n);

    xnormal[3] = n[0];
    ynormal[3] = n[1];
    znormal[3] = n[2];

    p[0] = xyz[0][i3];
    p[1] = xyz[1][i3];
    p[2] = xyz[2][i3];

    d[3] = dot3(n, p);
}

static SFEM_INLINE int tet4_hpoly3_detect_isect(const idx_t i0,
                                                const idx_t i1,
                                                const idx_t i2,
                                                const idx_t i3,
                                                geom_t** const SFEM_RESTRICT xyz,
                                                const real_t* const SFEM_RESTRICT xnormal,
                                                const real_t* const SFEM_RESTRICT ynormal,
                                                const real_t* const SFEM_RESTRICT znormal,
                                                const real_t* const SFEM_RESTRICT d) {
    real_t p[3][4];
    uint8_t outside[4] = {0, 0, 0, 0};

#pragma unroll(3)
    for (int d = 0; d < 3; d++) {
        p[d][0] = xyz[d][i0];
        p[d][1] = xyz[d][i1];
        p[d][2] = xyz[d][i2];
        p[d][3] = xyz[d][i3];
    }

#pragma unroll(4)
    for (int t = 0; t < 4; t++) {
#pragma unroll(4)
        for (int i = 0; i < 4; i++) {
            real_t dist = p[0][i] * xnormal[t] + p[1][i] * ynormal[t] + p[2][i] * znormal[t];
            outside[i] += dist > d[t];
        }
    }

    for (int i = 0; i < 4; i++) {
        if (outside[i] == 4) return 0;
    }

    return 1;
}

static void mesh_self_intersect_3D(const ptrdiff_t nelements,
                                   const ptrdiff_t nnodes,
                                   idx_t** const SFEM_RESTRICT elements,
                                   geom_t** const SFEM_RESTRICT xyz,
                                   idx_t** const SFEM_RESTRICT intersection_elements,
                                   geom_t** const SFEM_RESTRICT intersection_xyz,
                                   idx_t** const SFEM_RESTRICT removed_elements) {

    ptrdiff_t intersection_count = 0;
    ptrdiff_t intersection_list_size = MAX(64, nelements / 32);

    ptrdiff_t * ii = malloc(intersection_list_size * sizeof(ptrdiff_t));
    ptrdiff_t * jj = malloc(intersection_list_size * sizeof(ptrdiff_t));

    // Brute force
    for (ptrdiff_t e1 = 0; e1 < nelements; e1++) {
        geom_t bmin1[3], bmax1[3];
        real_t hpoly_x[4], hpoly_y[4], hpoly_z[4];
        real_t hpoly_d[4];

        tet4_hpoly3(elements[0][e1],
                    elements[1][e1],
                    elements[2][e1],
                    elements[3][e1],
                    xyz,
                    hpoly_x,
                    hpoly_y,
                    hpoly_z,
                    hpoly_d);

        box3(elements[0][e1], elements[1][e1], elements[2][e1], elements[3][e1], xyz, bmin1, bmax1);
        
        for (ptrdiff_t e2 = e1 + 1; e2 < nelements; e2++) {
            geom_t bmin2[3], bmax2[3];

            box3(elements[0][e2],
                 elements[1][e2],
                 elements[2][e2],
                 elements[3][e2],
                 xyz,
                 bmin2,
                 bmax2);

            // Cheap
            if (!box_intersect3(bmin1, bmax1, bmin2, bmax2)) {
                continue;
            }

            if (!tet4_hpoly3_detect_isect(elements[0][e2],
                                          elements[1][e2],
                                          elements[2][e2],
                                          elements[3][e2],
                                          xyz,
                                          hpoly_x,
                                          hpoly_y,
                                          hpoly_z,
                                          hpoly_d)) {
                continue;
            }

            if(intersection_count >= intersection_list_size) {
                intersection_list_size *= 2;
                ii = realloc(ii, intersection_list_size * sizeof(ptrdiff_t));
                jj = realloc(jj, intersection_list_size * sizeof(ptrdiff_t));
            }

            // Update intersection list
            ii[intersection_count] = e1;
            jj[intersection_count] = e2;
            intersection_count++;
        }
    }

    // TODO
    // Compute intersection and (stich it) to the orginal mesh (i.e., match node ids)
    // Keep track of elements to be removed
    for(ptrdiff_t i = 0; i < intersection_count; i++) {
        ptrdiff_t e1 = ii[i];
        ptrdiff_t e2 = jj[i];
    }

    free(ii);
    free(jj);
}

void mesh_self_intersect(const ptrdiff_t nelements,
                         const ptrdiff_t nnodes,
                         idx_t** const SFEM_RESTRICT elements,
                         geom_t** const SFEM_RESTRICT xyz,
                         idx_t** const SFEM_RESTRICT intersection_elements,
                         geom_t** const SFEM_RESTRICT intersection_xyz) {
    double tick = MPI_Wtime();

    double tock = MPI_Wtime();
    printf("mesh_self_intersect.c: mesh_self_intersect\t%g seconds\n", tock - tick);
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (argc != 4) {
        if (!rank) {
            fprintf(stderr, "usage: %s <folder> <thickness> <output_folder>\n", argv[0]);
        }

        return EXIT_FAILURE;
    }

    const geom_t thickness = atof(argv[2]);
    const char* output_folder = argv[3];

    {
        struct stat st = {0};
        if (stat(output_folder, &st) == -1) {
            mkdir(output_folder, 0700);
        }
    }

    double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    const char* folder = argv[1];

    mesh_t mesh;
    if (mesh_read(comm, folder, &mesh)) {
        return EXIT_FAILURE;
    }

    if (mesh.element_type != TRI3 || mesh.element_type == TET4) {
        fprintf(stderr, "This code only supports mesh with element type TRI3 or TET4\n");
        return EXIT_FAILURE;
    }

    mesh_t intersection_mesh;
    mesh_init(&intersection_mesh);

    intersection_mesh.comm = mesh.comm;
    intersection_mesh.mem_space = mesh.mem_space;

    intersection_mesh.spatial_dim = mesh.spatial_dim;
    intersection_mesh.element_type = WEDGE6;

    intersection_mesh.nelements = mesh.nelements;
    intersection_mesh.nnodes = mesh.nnodes * 2;
    intersection_mesh.n_owned_elements = intersection_mesh.nelements;

    intersection_mesh.node_mapping = 0;
    intersection_mesh.element_mapping = 0;
    intersection_mesh.node_owner = 0;

    int nnxe_intersection_mesh = elem_num_nodes(intersection_mesh.element_type);
    intersection_mesh.elements = malloc(nnxe_intersection_mesh * sizeof(idx_t*));
    for (int d = 0; d < nnxe_intersection_mesh; d++) {
        intersection_mesh.elements[d] = malloc(intersection_mesh.nelements * sizeof(idx_t));
    }

    intersection_mesh.points = malloc(intersection_mesh.spatial_dim * sizeof(geom_t*));
    for (int d = 0; d < intersection_mesh.spatial_dim; d++) {
        intersection_mesh.points[d] = malloc(intersection_mesh.nnodes * sizeof(geom_t));
    }

    mesh_self_intersect(mesh.nelements,
                        mesh.nnodes,
                        mesh.elements,
                        mesh.points,
                        intersection_mesh.elements,
                        intersection_mesh.points);

    mesh_write(output_folder, &intersection_mesh);

    if (!rank) {
        printf("----------------------------------------\n");
        printf("Volume: #elements %ld #nodes %ld\n", (long)mesh.nelements, (long)mesh.nnodes);
        printf("Surface: #elements %ld #nodes %ld\n",
               (long)intersection_mesh.nelements,
               (long)intersection_mesh.nnodes);
    }

    // Clean-up
    mesh_destroy(&mesh);
    mesh_destroy(&intersection_mesh);

    double tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
