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
    sfem::Context context(argc, argv);
    MPI_Comm      comm = context.comm();

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

#if 0

    if (argc != 7) {
        if (!rank) {
            fprintf(stderr, "usage: %s <folder> <x> <y> <z> <angle_threshold> <output_folder>\n", argv[0]);
        }

        return EXIT_FAILURE;
    }

    const char  *path_mesh       = argv[1];
    const geom_t roi[3]          = {(geom_t)atof(argv[2]), (geom_t)atof(argv[3]), (geom_t)atof(argv[4])};
    const geom_t angle_threshold = atof(argv[5]);
    const char  *path_output     = argv[6];

    if (!rank) {
        fprintf(stderr,
                "%s %s %g %g %g %g %s\n",
                argv[0],
                path_mesh,
                (double)roi[0],
                (double)roi[1],
                (double)roi[2],
                (double)angle_threshold,
                path_output);
    }

    double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    auto mesh = sfem::Mesh::create_from_file(comm, path_mesh);

    ///////////////////////////////////////////////////////////////////////////////
    // Extract buffers and values
    ///////////////////////////////////////////////////////////////////////////////

    const ptrdiff_t n_elements = mesh->n_elements();
    const ptrdiff_t n_nodes    = mesh->n_nodes();
    auto            elements   = mesh->elements()->data();
    auto            points     = mesh->points()->data();
    const int       nsxe       = elem_num_sides(mesh->element_type());

    ///////////////////////////////////////////////////////////////////////////////
    // Create skin sideset
    ///////////////////////////////////////////////////////////////////////////////

    // Reduce cost of computation by exploiting low-order representation
    int element_type_hack = mesh->element_type();
    switch (element_type_hack) {
        case TRI6: {
            element_type_hack = TRI3;
            break;
        }
        case TET10: {
            element_type_hack = TET4;
            break;
        }
        case EDGE3: {
            element_type_hack = EDGE2;
            break;
        }
        default:
            break;
    }

    enum ElemType st   = side_type(element_type_hack);
    const int     nnxs = elem_num_nodes(st);
    const int     dim  = mesh->spatial_dimension();

    ptrdiff_t      n_surf_elements = 0;
    element_idx_t *parent_buff     = 0;
    int16_t       *side_idx_buff   = 0;

    if (extract_skin_sideset(
                n_elements, n_nodes, element_type_hack, elements, &n_surf_elements, &parent_buff, &side_idx_buff) !=
        SFEM_SUCCESS) {
        SFEM_ERROR("Failed to extract skin!\n");
    }
    ///////////////////////////////////////////////////////////////////////////////

    auto parent   = sfem::manage_host_buffer<element_idx_t>(n_surf_elements, parent_buff);
    auto side_idx = sfem::manage_host_buffer<int16_t>(n_surf_elements, side_idx_buff);

    auto local_side_table = sfem::create_host_buffer<int>(nsxe * nnxs);
    fill_local_side_table(element_type_hack, local_side_table->data());

    ///////////////////////////////////////////////////////////////////////////////
    // Find approximately closest elemenent
    ///////////////////////////////////////////////////////////////////////////////

    element_idx_t *surf_parents = parent->data();
    int16_t       *surf_idx     = side_idx->data();
    auto           lst          = local_side_table->data();

    idx_t  closest_element = SFEM_IDX_INVALID;
    real_t closest_sq_dist = 1000000;

    for (ptrdiff_t e = 0; e < n_surf_elements; ++e) {
        geom_t        element_sq_dist = 1000000;
        element_idx_t sp              = surf_parents[e];
        int16_t       s               = surf_idx[e];

        for (int n = 0; n < nn; n++) {
            idx_t node = elems[lst[s * nnxs + n]][e];

            geom_t sq_dist = 0.;
            for (int d = 0; d < dim; ++d) {
                const real_t m_x   = points[d][node];
                const real_t roi_x = roi[d];
                const real_t diff  = m_x - roi_x;
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
        SFEM_ERROR("Invalid set up! for mesh #nelements %ld #nodes %ld\n", n_elements, n_nodes);
    }

    // std::shared_ptr<Buffer<element_idx_t>> parent;
    // std::shared_ptr<Buffer<int16_t>>       lfi;

    uint8_t *selected = (uint8_t *)malloc(n_elements * sizeof(uint8_t));
    memset(selected, 0, n_elements * sizeof(uint8_t));

    ptrdiff_t  size_queue = (n_elements + 1);
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
                const idx_t idx0 = mesh.elements[0][e];
                const idx_t idx1 = mesh.elements[1][e];

                real_t p0[2];
                real_t p1[2];
                for (int d = 0; d < 2; ++d) {
                    p0[d] = mesh.points[d][idx0];
                    p1[d] = mesh.points[d][idx1];
                }

                normal2(p0, p1, n);
                // printf("------------------\n");
                // printf("%g %g\n", n[0], n[1]);
            }

            const count_t e_begin = adj_ptr[e];
            const count_t e_end   = adj_ptr[e + 1];

            real_t current_thres = angle_threshold;

            for (count_t k = e_begin; k < e_end; ++k) {
                const idx_t e_adj = adj_idx[k];

                if (selected[e_adj]) {
                    continue;
                }

                real_t cos_angle;
                {
                    const idx_t idx0 = mesh.elements[0][e_adj];
                    const idx_t idx1 = mesh.elements[1][e_adj];

                    real_t p0a[2];
                    real_t p1a[2];
                    real_t na[2];

                    for (int d = 0; d < 2; ++d) {
                        p0a[d] = mesh.points[d][idx0];
                        p1a[d] = mesh.points[d][idx1];
                    }

                    normal2(p0a, p1a, na);
                    cos_angle = fabs((n[0] * na[0]) + (n[1] * na[1]));

                    // printf("%g %g (%g)\n", na[0], na[1], cos_angle);
                }

                if (cos_angle > angle_threshold) {
                    elem_queue[next_slot++ % size_queue] = e_adj;
                }
            }

            selected[e]   = 1;
            elem_queue[q] = SFEM_PTRDIFF_INVALID;
        }
    } else {
        for (ptrdiff_t q = 0; elem_queue[q] >= 0; q = (q + 1) % size_queue) {
            const ptrdiff_t e = elem_queue[q];

            if (selected[e]) continue;

            real_t n[3];
            {
                idx_t idx0 = mesh.elements[0][e];
                idx_t idx1 = mesh.elements[1][e];
                idx_t idx2 = mesh.elements[2][e];

                real_t u[3];
                real_t v[3];
                for (int d = 0; d < 3; ++d) {
                    u[d] = mesh.points[d][idx1] - mesh.points[d][idx0];
                    v[d] = mesh.points[d][idx2] - mesh.points[d][idx0];
                }

                normal(u, v, n);
            }

            const count_t e_begin = adj_ptr[e];
            const count_t e_end   = adj_ptr[e + 1];

            real_t current_thres = angle_threshold;

            for (count_t k = e_begin; k < e_end; ++k) {
                const idx_t e_adj = adj_idx[k];

                if (selected[e_adj]) {
                    continue;
                }

                real_t cos_angle;
                {
                    const idx_t idx0 = mesh.elements[0][e_adj];
                    const idx_t idx1 = mesh.elements[1][e_adj];
                    const idx_t idx2 = mesh.elements[2][e_adj];

                    real_t ua[3];
                    real_t va[3];
                    real_t na[3];

                    for (int d = 0; d < 3; ++d) {
                        ua[d] = mesh.points[d][idx1] - mesh.points[d][idx0];
                        va[d] = mesh.points[d][idx2] - mesh.points[d][idx0];
                    }

                    normal(ua, va, na);
                    cos_angle = fabs((n[0] * na[0]) + (n[1] * na[1]) + (n[2] * na[2]));
                }

                if (cos_angle > angle_threshold) {
                    elem_queue[next_slot++ % size_queue] = e_adj;
                }
            }

            selected[e]   = 1;
            elem_queue[q] = SFEM_PTRDIFF_INVALID;
        }
    }

    printf("num_selected = %ld / %ld\n", (long)next_slot, (long)mesh.nelements);

    int SFEM_EXPORT_COLOR = 0;
    SFEM_READ_ENV(SFEM_EXPORT_COLOR, atoi);

    if (SFEM_EXPORT_COLOR) {
        array_write(comm, "color.raw", MPI_CHAR, selected, mesh.nelements, mesh.nelements);
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Create element index
    ///////////////////////////////////////////////////////////////////////////////

    ptrdiff_t n_selected = 0;
    for (ptrdiff_t i = 0; i < mesh.nelements; i++) {
        n_selected += selected[i] == 1;
    }

    idx_t *indices = (idx_t *)malloc(n_selected * sizeof(idx_t));
    for (ptrdiff_t i = 0, n_inserted = 0; i < mesh.nelements; i++) {
        if (selected[i]) {
            indices[n_inserted++] = i;
        }
    }

    array_write(comm, path_output, SFEM_MPI_IDX_T, indices, n_selected, n_selected);

    free(selected);
    free(indices);
    mesh_destroy(&mesh);
    double tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
#endif
}
