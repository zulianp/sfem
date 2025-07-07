#include <math.h>
#include <stdint.h>
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

#include "sfem_defs.h"

#include "argsort.h"

#include "sfem_API.hpp"

static SFEM_INLINE void normal(real_t u[3], real_t v[3], real_t *n) {
    const real_t u_len = sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
    const real_t v_len = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);

    for (int d = 0; d < 3; ++d) {
        u[d] /= u_len;
        v[d] /= v_len;
    }

    n[0] = (u[1] * v[2]) - (u[2] * v[1]);
    n[1] = (u[2] * v[0]) - (u[0] * v[2]);
    n[2] = (u[0] * v[1]) - (u[1] * v[0]);

    const real_t n_len = sqrt((n[0] * n[0]) + (n[1] * n[1]) + (n[2] * n[2]));

    for (int d = 0; d < 3; ++d) {
        n[d] /= n_len;
    }
}

static SFEM_INLINE void normal2(real_t p0[2], real_t p1[2], real_t *n) {
    n[0] = -p1[1] + p0[1];
    n[1] = p1[0] - p0[0];

    const real_t len_n = sqrt(n[0] * n[0] + n[1] * n[1]);
    n[0] /= len_n;
    n[1] /= len_n;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (argc != 7) {
        if (!rank) {
            fprintf(stderr,
                    "usage: %s <folder> <x> <y> <z> <angle_threshold> <selection.raw>\n",
                    argv[0]);
        }

        return EXIT_FAILURE;
    }

    const char *folder = argv[1];
    const geom_t roi[3] = {(geom_t)atof(argv[2]), (geom_t)atof(argv[3]), (geom_t)atof(argv[4])};
    const geom_t angle_threshold = atof(argv[5]);
    const char *path_selection = argv[6];

    if (!rank) {
        printf("%s %s %g %g %g %g %s\n",
               argv[0],
               folder,
               (double)roi[0],
               (double)roi[1],
               (double)roi[2],
               (double)angle_threshold,
               path_selection);
    }

    double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    auto mesh = sfem::Mesh::create_from_file(sfem::Communicator::wrap(comm), folder);


    const char* SFEM_ELEMENT_TYPE = type_to_string(mesh->element_type());
    SFEM_READ_ENV(SFEM_ELEMENT_TYPE, );
    mesh->set_element_type(type_from_string(SFEM_ELEMENT_TYPE));

    int nxe = mesh->n_nodes_per_element();


    ///////////////////////////////////////////////////////////////////////////////
    // Find approximately closest elemenent
    ///////////////////////////////////////////////////////////////////////////////

    idx_t closest_element = SFEM_IDX_INVALID;
    real_t closest_sq_dist = 1000000;

    auto elements = mesh->elements()->data();
    auto points = mesh->points()->data();
    const int spatial_dim = mesh->spatial_dimension();

    for (ptrdiff_t e = 0; e < mesh->n_elements(); ++e) {
        geom_t element_sq_dist = 1000000;

        for (int n = 0; n < nxe; ++n) {
            const idx_t node = elements[n][e];

            geom_t sq_dist = 0.;
            for (int d = 0; d < spatial_dim; ++d) {
                const real_t m_x = points[d][node];
                const real_t roi_x = roi[d];
                const real_t diff = m_x - roi_x;
                sq_dist += diff * diff;
            }

            element_sq_dist = MIN(element_sq_dist, sq_dist);
        }

        if (element_sq_dist < closest_sq_dist) {
            closest_sq_dist = element_sq_dist;
            closest_element = e;
        }
    }

    if (closest_element == SFEM_IDX_INVALID) {
        SFEM_ERROR("Invalid set up! for mesh #nelements %ld #nodes %ld\n", mesh->n_elements(), mesh->n_nodes());
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Create dual-graph for navigating neighboring elements
    ///////////////////////////////////////////////////////////////////////////////

    int element_type_hack = mesh->element_type();
    if (mesh->element_type() == TRI6) {
        element_type_hack = TRI3;
    }

    if (mesh->element_type() == EDGE3) {
        element_type_hack = EDGE2;
    }

    count_t *adj_ptr = 0;
    element_idx_t *adj_idx = 0;
    create_dual_graph(
        mesh->n_elements(), mesh->n_nodes(), element_type_hack, elements, &adj_ptr, &adj_idx);

    uint8_t *selected = (uint8_t *)malloc(mesh->n_elements() * sizeof(uint8_t));
    memset(selected, 0, mesh->n_elements() * sizeof(uint8_t));

    ptrdiff_t size_queue = (mesh->n_elements() + 1);
    ptrdiff_t *elem_queue = (ptrdiff_t *)malloc(size_queue * sizeof(ptrdiff_t));

    elem_queue[0] = closest_element;
    for (ptrdiff_t e = 1; e < size_queue; ++e) {
        elem_queue[e] = SFEM_PTRDIFF_INVALID;
    }

    // Next slot
    ptrdiff_t next_slot = 1;

    ///////////////////////////////////////////////////////////////////////////////
    // Create marker for different faces based on dihedral angles
    ///////////////////////////////////////////////////////////////////////////////

    if (element_type_hack == EDGE2) {
        for (ptrdiff_t q = 0; elem_queue[q] >= 0; q = (q + 1) % size_queue) {
            const ptrdiff_t e = elem_queue[q];

            if (selected[e]) continue;

            real_t n[2];
            {
                const idx_t idx0 = elements[0][e];
                const idx_t idx1 = elements[1][e];

                real_t p0[2];
                real_t p1[2];
                for (int d = 0; d < 2; ++d) {
                    p0[d] = points[d][idx0];
                    p1[d] = points[d][idx1];
                }

                normal2(p0, p1, n);
                // printf("------------------\n");
                // printf("%g %g\n", n[0], n[1]);
            }

            const count_t e_begin = adj_ptr[e];
            const count_t e_end = adj_ptr[e + 1];

            real_t current_thres = angle_threshold;

            for (count_t k = e_begin; k < e_end; ++k) {
                const idx_t e_adj = adj_idx[k];

                if (selected[e_adj]) {
                    continue;
                }

                real_t cos_angle;
                {
                    const idx_t idx0 = elements[0][e_adj];
                    const idx_t idx1 = elements[1][e_adj];

                    real_t p0a[2];
                    real_t p1a[2];
                    real_t na[2];

                    for (int d = 0; d < 2; ++d) {
                        p0a[d] = points[d][idx0];
                        p1a[d] = points[d][idx1];
                    }

                    normal2(p0a, p1a, na);
                    cos_angle = fabs((n[0] * na[0]) + (n[1] * na[1]));

                    // printf("%g %g (%g)\n", na[0], na[1], cos_angle);
                }

                if (cos_angle > angle_threshold) {
                    elem_queue[next_slot++ % size_queue] = e_adj;
                }
            }

            selected[e] = 1;
            elem_queue[q] = SFEM_PTRDIFF_INVALID;
        }
    } else {
        for (ptrdiff_t q = 0; elem_queue[q] >= 0; q = (q + 1) % size_queue) {
            const ptrdiff_t e = elem_queue[q];

            if (selected[e]) continue;

            real_t n[3];
            {
                idx_t idx0 = elements[0][e];
                idx_t idx1 = elements[1][e];
                idx_t idx2 = elements[2][e];

                real_t u[3];
                real_t v[3];
                for (int d = 0; d < 3; ++d) {
                    u[d] = points[d][idx1] - points[d][idx0];
                    v[d] = points[d][idx2] - points[d][idx0];
                }

                normal(u, v, n);
            }

            const count_t e_begin = adj_ptr[e];
            const count_t e_end = adj_ptr[e + 1];

            real_t current_thres = angle_threshold;

            for (count_t k = e_begin; k < e_end; ++k) {
                const idx_t e_adj = adj_idx[k];

                if (selected[e_adj]) {
                    continue;
                }

                real_t cos_angle;
                {
                    const idx_t idx0 = elements[0][e_adj];
                    const idx_t idx1 = elements[1][e_adj];
                    const idx_t idx2 = elements[2][e_adj];

                    real_t ua[3];
                    real_t va[3];
                    real_t na[3];

                    for (int d = 0; d < 3; ++d) {
                        ua[d] = points[d][idx1] - points[d][idx0];
                        va[d] = points[d][idx2] - points[d][idx0];
                    }

                    normal(ua, va, na);
                    cos_angle = fabs((n[0] * na[0]) + (n[1] * na[1]) + (n[2] * na[2]));
                }

                if (cos_angle > angle_threshold) {
                    elem_queue[next_slot++ % size_queue] = e_adj;
                }
            }

            selected[e] = 1;
            elem_queue[q] = SFEM_PTRDIFF_INVALID;
        }
    }

    printf("num_selected = %ld / %ld\n", (long)next_slot, (long)mesh->n_elements());

    int SFEM_EXPORT_COLOR = 0;
    SFEM_READ_ENV(SFEM_EXPORT_COLOR, atoi);

    if (SFEM_EXPORT_COLOR) {
        array_write(comm, "color.raw", MPI_CHAR, selected, mesh->n_elements(), mesh->n_elements());
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Create element index
    ///////////////////////////////////////////////////////////////////////////////

    ptrdiff_t n_selected = 0;
    for (ptrdiff_t i = 0; i < mesh->n_elements(); i++) {
        n_selected += selected[i] == 1;
    }

    idx_t *indices = (idx_t *)malloc(n_selected * sizeof(idx_t));
    for (ptrdiff_t i = 0, n_inserted = 0; i < mesh->n_elements(); i++) {
        if (selected[i]) {
            indices[n_inserted++] = i;
        }
    }

    array_write(comm, path_selection, SFEM_MPI_IDX_T, indices, n_selected, n_selected);

    free(selected);
    free(indices);

    double tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
