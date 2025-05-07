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
    {
        MPI_Comm comm = context.comm();

        int rank, size;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);

        if (argc != 7) {
            if (!rank) {
                fprintf(stderr, "usage: %s <folder> <x> <y> <z> <angle_threshold> <output_folder>\n", argv[0]);
            }

            return EXIT_FAILURE;
        }

        const char  *path_mesh       = argv[1];
        const geom_t roi[3]          = {(geom_t)atof(argv[2]), (geom_t)atof(argv[3]), (geom_t)atof(argv[4])};
        const geom_t angle_threshold = atof(argv[5]);
        std::string  output_folder   = argv[6];

        if (!rank) {
            fprintf(stderr,
                    "%s %s %g %g %g %g %s\n",
                    argv[0],
                    path_mesh,
                    (double)roi[0],
                    (double)roi[1],
                    (double)roi[2],
                    (double)angle_threshold,
                    output_folder.c_str());
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
        enum ElemType element_type_hack = mesh->element_type();
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
        const int     ns   = elem_num_sides(element_type_hack);

        ptrdiff_t      n_surf_elements = 0;
        element_idx_t *parent_buff     = 0;
        int16_t       *side_idx_buff   = 0;
        element_idx_t *table_buff      = 0;

        create_element_adj_table(n_elements, n_nodes, element_type_hack, elements, &table_buff);

        if (extract_sideset_from_adj_table(
                    element_type_hack, n_elements, table_buff, &n_surf_elements, &parent_buff, &side_idx_buff) != SFEM_SUCCESS) {
            SFEM_ERROR("Failed to extract extract_sideset_from_adj_table!\n");
        }

        auto parent              = sfem::manage_host_buffer<element_idx_t>(n_surf_elements, parent_buff);
        auto side_idx            = sfem::manage_host_buffer<int16_t>(n_surf_elements, side_idx_buff);
        auto table               = sfem::manage_host_buffer<element_idx_t>(n_surf_elements * nsxe, table_buff);
        auto element_mapping_ptr = sfem::create_host_buffer<element_idx_t>(n_elements + 1);

        auto local_side_table = sfem::create_host_buffer<int>(nsxe * nnxs);
        fill_local_side_table(element_type_hack, local_side_table->data());

        ///////////////////////////////////////////////////////////////////////////////
        // Find approximately closest elemenent
        ///////////////////////////////////////////////////////////////////////////////

        element_idx_t *surf_parents = parent->data();
        int16_t       *surf_idx     = side_idx->data();
        auto           lst          = local_side_table->data();

        element_idx_t closest_element = SFEM_ELEMENT_IDX_INVALID;
        int16_t       closest_side    = -1;
        real_t        closest_sq_dist = 1000000;

        auto emap_ptr = element_mapping_ptr->data();

#pragma omp parallel for
        for (ptrdiff_t e = 0; e < n_surf_elements; ++e) {
            geom_t        element_sq_dist = 1000000;
            element_idx_t sp              = surf_parents[e];
            int16_t       s               = surf_idx[e];

#pragma omp atomic update
            emap_ptr[sp + 1]++;

            for (int n = 0; n < nnxs; n++) {
                idx_t node = elements[lst[s * nnxs + n]][e];

                geom_t sq_dist = 0.;
                for (int d = 0; d < dim; ++d) {
                    const real_t m_x   = points[d][node];
                    const real_t roi_x = roi[d];
                    const real_t diff  = m_x - roi_x;
                    sq_dist += diff * diff;
                }

                element_sq_dist = MIN(element_sq_dist, sq_dist);
            }

#pragma omp critical
            {
                if (element_sq_dist < closest_sq_dist) {
                    closest_sq_dist = element_sq_dist;
                    closest_element = e;
                    closest_side    = s;
                }
            }
        }

        for (ptrdiff_t i = 0; i < n_elements; i++) {
            emap_ptr[i + 1] += emap_ptr[i];
        }

        ptrdiff_t nmaps               = emap_ptr[n_elements];
        auto      element_mapping_idx = sfem::create_host_buffer<ptrdiff_t>(nmaps);

        assert(nmaps < table->size());

        auto emap_idx = element_mapping_idx->data();

        {
            auto book_keeping = sfem::create_host_buffer<element_idx_t>(n_elements);
            auto bk           = book_keeping->data();
            for (ptrdiff_t e = 0; e < n_surf_elements; ++e) {
                element_idx_t sp                  = surf_parents[e];
                emap_idx[emap_ptr[sp] + bk[sp]++] = e;
            }
        }

#ifndef NDEBUG
        for(ptrdiff_t i = 0; i < nmaps; i++) {
            assert(emap_idx[i] < table->size());
        }
#endif

        if (closest_element == SFEM_IDX_INVALID) {
            SFEM_ERROR("Invalid set up! for mesh #nelements %ld #nodes %ld\n", n_elements, n_nodes);
        }

        auto adj      = table->data();
        auto selected = sfem::create_host_buffer<uint8_t>(table->size());
        auto eselect  = selected->data();

        ptrdiff_t size_queue    = (n_elements + 1);
        auto      element_queue = sfem::create_host_buffer<ptrdiff_t>(size_queue);
        auto      equeue        = element_queue->data();

        equeue[0] = closest_element;
        for (ptrdiff_t e = 1; e < size_queue; ++e) {
            equeue[e] = SFEM_PTRDIFF_INVALID;
        }

        // Next slot
        ptrdiff_t next_slot = 1;

        ///////////////////////////////////////////////////////////////////////////////
        // Create marker for different faces based on dihedral angles
        ///////////////////////////////////////////////////////////////////////////////

        if (dim == 2) {
            for (ptrdiff_t q = 0; equeue[q] >= 0; q = (q + 1) % size_queue) {
                const ptrdiff_t e = equeue[q];

                element_idx_t sp = surf_parents[e];
                int16_t       s  = surf_idx[e];

                if (eselect[e]) continue;

                real_t n[2];
                {
                    const idx_t idx0 = elements[lst[s * nnxs + 0]][sp];
                    const idx_t idx1 = elements[lst[s * nnxs + 1]][sp];

                    real_t p0[2];
                    real_t p1[2];
                    for (int d = 0; d < 2; ++d) {
                        p0[d] = points[d][idx0];
                        p1[d] = points[d][idx1];
                    }

                    normal2(p0, p1, n);
                }

                real_t current_thres = angle_threshold;
                for (int neigh = 0; neigh < nsxe; neigh++) {
                    const element_idx_t neigh_sp = adj[sp * nsxe + neigh];
                    if (neigh_sp == SFEM_ELEMENT_IDX_INVALID) continue;

                    for (int k = emap_ptr[neigh_sp]; k < emap_ptr[neigh_sp + 1]; k++) {
                        const element_idx_t neigh_e = emap_idx[k];

                        if (neigh_e == SFEM_ELEMENT_IDX_INVALID || eselect[neigh_e]) continue;

                        int16_t neigh_s = surf_idx[neigh_e];

                        real_t cos_angle;
                        {
                            const idx_t idx0 = elements[lst[neigh_s * nnxs + 0]][neigh_sp];
                            const idx_t idx1 = elements[lst[neigh_s * nnxs + 1]][neigh_sp];

                            real_t p0a[2];
                            real_t p1a[2];
                            real_t na[2];

                            for (int d = 0; d < 2; ++d) {
                                p0a[d] = points[d][idx0];
                                p1a[d] = points[d][idx1];
                            }

                            normal2(p0a, p1a, na);
                            cos_angle = fabs((n[0] * na[0]) + (n[1] * na[1]));
                        }

                        if (cos_angle > angle_threshold) {
                            equeue[next_slot++ % size_queue] = neigh_e;
                        }
                    }
                }

                eselect[e] = 1;
                equeue[q]  = SFEM_PTRDIFF_INVALID;
            }
        } else {
            for (ptrdiff_t q = 0; equeue[q] >= 0; q = (q + 1) % size_queue) {
                const ptrdiff_t e = equeue[q];

                element_idx_t sp = surf_parents[e];
                int16_t       s  = surf_idx[e];

                if (eselect[e]) continue;

                real_t n[3];
                {
                    const idx_t idx0 = elements[lst[s * nnxs + 0]][sp];
                    const idx_t idx1 = elements[lst[s * nnxs + 1]][sp];
                    const idx_t idx2 = elements[lst[s * nnxs + 2]][sp];

                    real_t u[3];
                    real_t v[3];
                    for (int d = 0; d < 3; ++d) {
                        u[d] = points[d][idx1] - points[d][idx0];
                        v[d] = points[d][idx2] - points[d][idx0];
                    }

                    normal(u, v, n);
                }

                real_t current_thres = angle_threshold;
                for (int neigh = 0; neigh < nsxe; neigh++) {
                    const element_idx_t neigh_sp = adj[sp * nsxe + neigh];
                    if (neigh_sp == SFEM_ELEMENT_IDX_INVALID) continue;

                    for (int k = emap_ptr[neigh_sp]; k < emap_ptr[neigh_sp + 1]; k++) {
                        const element_idx_t neigh_e = emap_idx[k];
                        assert(neigh_e < selected->size());
                        if (neigh_e == SFEM_ELEMENT_IDX_INVALID || eselect[neigh_e]) continue;

                        int16_t neigh_s = surf_idx[neigh_e];

                        real_t cos_angle;
                        {
                            const idx_t idx0 = elements[lst[neigh_s * nnxs + 0]][neigh_sp];
                            const idx_t idx1 = elements[lst[neigh_s * nnxs + 1]][neigh_sp];
                            const idx_t idx2 = elements[lst[neigh_s * nnxs + 2]][neigh_sp];

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
                            equeue[next_slot++ % size_queue] = neigh_e;
                        }
                    }
                }

                eselect[e] = 1;
                equeue[q]  = SFEM_PTRDIFF_INVALID;
            }
        }

        ///////////////////////////////////////////////////////////////////////////////
        // Create sidesets
        ///////////////////////////////////////////////////////////////////////////////

        ptrdiff_t n_selected = 0;
        for (ptrdiff_t i = 0; i < n_surf_elements; i++) {
            n_selected += eselect[i] == 1;
        }

        auto sideset_parent = sfem::create_host_buffer<element_idx_t>(n_selected);
        auto sideset_lfi    = sfem::create_host_buffer<int16_t>(n_selected);

        auto ssp   = sideset_parent->data();
        auto sslfi = sideset_lfi->data();

        for (ptrdiff_t e = 0, n_inserted = 0; e < n_surf_elements; ++e) {
            if (eselect[e]) {
                sslfi[n_inserted] = surf_idx[e];
                ssp[n_inserted++] = surf_parents[e];
            }
        }

        sfem::create_directory(output_folder.c_str());
        sideset_parent->to_file((output_folder + "/parent.raw").c_str());
        sideset_lfi->to_file((output_folder + "/lfi.int16.raw").c_str());

        int SFEM_DEBUG = 0;
        SFEM_READ_ENV(SFEM_DEBUG, atoi);

        if (SFEM_DEBUG) {
            auto debug_elements = sfem::create_host_buffer<idx_t>(elem_num_nodes(side_type(mesh->element_type())), n_selected);

            if (extract_surface_from_sideset(mesh->element_type(),
                                             elements,
                                             n_selected,
                                             sideset_parent->data(),
                                             sideset_lfi->data(),
                                             debug_elements->data()) != SFEM_SUCCESS) {
                SFEM_ERROR("Unable to extract surface from sideset!\n");
            }

            sfem::create_directory((output_folder + "/surf").c_str());
            debug_elements->to_files((output_folder + "/surf/i%d.raw").c_str());
        }

        double tock = MPI_Wtime();

        if (!rank) {
            printf("----------------------------------------\n");
            printf("TTS:\t\t\t%g seconds\n", tock - tick);
        }
    }

    return SFEM_SUCCESS;
}
