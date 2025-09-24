#include "sfem_P1toP2.hpp"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "utils.h"

#include "crs_graph.h"
#include "sfem_base.h"
#include "sfem_defs.h"

#include "sortreduce.h"
#include "sfem_glob.hpp"

#include "sfem_API.hpp"
#include "sfem_Mesh.hpp"

namespace sfem {

    namespace {
        std::shared_ptr<Mesh> handle_failure(count_t *rowptr, idx_t *colidx, idx_t *p2idx) {
            if (p2idx) free(p2idx);
            if (rowptr) free(rowptr);
            if (colidx) free(colidx);
            return nullptr;
        }
    }  // namespace

    std::shared_ptr<Mesh> convert_p1_mesh_to_p2(const std::shared_ptr<Mesh> &p1_mesh) {
        if (!p1_mesh) {
            return nullptr;
        }

        const ptrdiff_t n_elements = p1_mesh->n_elements();
        const ptrdiff_t n_nodes    = p1_mesh->n_nodes();
        const int       p1_nxe     = elem_num_nodes(p1_mesh->element_type());

        count_t *rowptr = nullptr;
        idx_t   *colidx = nullptr;

        build_crs_graph_for_elem_type(p1_mesh->element_type(),
                                      n_elements,
                                      n_nodes,
                                      p1_mesh->elements()->data(),
                                      &rowptr,
                                      &colidx);

        if (!rowptr || !colidx) {
            return handle_failure(rowptr, colidx, nullptr);
        }

        const count_t nnz = rowptr[n_nodes];

        idx_t *p2idx = static_cast<idx_t *>(malloc(nnz * sizeof(idx_t)));
        if (!p2idx) {
            return handle_failure(rowptr, colidx, nullptr);
        }
        memset(p2idx, 0, nnz * sizeof(idx_t));

        ptrdiff_t p2_nodes = 0;
        idx_t     next_id  = n_nodes;
        for (ptrdiff_t i = 0; i < n_nodes; i++) {
            const count_t begin = rowptr[i];
            const count_t end   = rowptr[i + 1];

            for (count_t k = begin; k < end; k++) {
                const idx_t j = colidx[k];

                if (i < j) {
                    p2_nodes += 1;
                    p2idx[k] = next_id++;
                }
            }
        }

        const int          spatial_dim      = p1_mesh->spatial_dimension();
        const enum ElemType p2_element_type = elem_higher_order(p1_mesh->element_type());
        const int          p2_nxe           = elem_num_nodes(p2_element_type);

        auto p2_elements_buffer = sfem::create_host_buffer<idx_t>(p2_nxe, n_elements);
        auto p2_points_buffer   = sfem::create_host_buffer<geom_t>(spatial_dim, n_nodes + p2_nodes);

        if (!p2_elements_buffer || !p2_points_buffer) {
            return handle_failure(rowptr, colidx, p2idx);
        }

        auto p2_elements = p2_elements_buffer->data();
        auto p2_points   = p2_points_buffer->data();

        auto p1_elements = p1_mesh->elements()->data();
        auto p1_points   = p1_mesh->points()->data();

        for (int d = 0; d < p1_nxe; d++) {
            memcpy(p2_elements[d], p1_elements[d], n_elements * sizeof(idx_t));
        }

        for (int d = p1_nxe; d < p2_nxe; d++) {
            memset(p2_elements[d], 0, n_elements * sizeof(idx_t));
        }

        for (int d = 0; d < spatial_dim; d++) {
            memcpy(p2_points[d], p1_points[d], n_nodes * sizeof(geom_t));
        }

        if (p1_mesh->element_type() == TET4) {
            for (ptrdiff_t e = 0; e < n_elements; e++) {
                idx_t row[6];
                row[0] = MIN(p2_elements[0][e], p2_elements[1][e]);
                row[1] = MIN(p2_elements[1][e], p2_elements[2][e]);
                row[2] = MIN(p2_elements[0][e], p2_elements[2][e]);
                row[3] = MIN(p2_elements[0][e], p2_elements[3][e]);
                row[4] = MIN(p2_elements[1][e], p2_elements[3][e]);
                row[5] = MIN(p2_elements[2][e], p2_elements[3][e]);

                idx_t key[6];
                key[0] = MAX(p2_elements[0][e], p2_elements[1][e]);
                key[1] = MAX(p2_elements[1][e], p2_elements[2][e]);
                key[2] = MAX(p2_elements[0][e], p2_elements[2][e]);
                key[3] = MAX(p2_elements[0][e], p2_elements[3][e]);
                key[4] = MAX(p2_elements[1][e], p2_elements[3][e]);
                key[5] = MAX(p2_elements[2][e], p2_elements[3][e]);

                for (int l = 0; l < 6; l++) {
                    const idx_t   r         = row[l];
                    const count_t row_begin = rowptr[r];
                    const count_t len_row   = rowptr[r + 1] - row_begin;
                    const idx_t  *cols      = &colidx[row_begin];
                    const idx_t   k         = find_idx_binary_search(key[l], cols, len_row);
                    p2_elements[l + p1_nxe][e] = p2idx[row_begin + k];
                }
            }
        } else if (p1_mesh->element_type() == TRI3) {
            for (ptrdiff_t e = 0; e < n_elements; e++) {
                idx_t row[3];
                row[0] = MIN(p2_elements[0][e], p2_elements[1][e]);
                row[1] = MIN(p2_elements[1][e], p2_elements[2][e]);
                row[2] = MIN(p2_elements[0][e], p2_elements[2][e]);

                idx_t key[3];
                key[0] = MAX(p2_elements[0][e], p2_elements[1][e]);
                key[1] = MAX(p2_elements[1][e], p2_elements[2][e]);
                key[2] = MAX(p2_elements[0][e], p2_elements[2][e]);

                for (int l = 0; l < 3; l++) {
                    const idx_t   r         = row[l];
                    const count_t row_begin = rowptr[r];
                    const count_t len_row   = rowptr[r + 1] - row_begin;
                    const idx_t  *cols      = &colidx[row_begin];
                    const idx_t   k         = find_idx_binary_search(key[l], cols, len_row);
                    p2_elements[l + p1_nxe][e] = p2idx[row_begin + k];
                }
            }
        } else {
            fprintf(stderr, "convert_p1_mesh_to_p2: unsupported element_type %d\n", p1_mesh->element_type());
            return handle_failure(rowptr, colidx, p2idx);
        }

        for (ptrdiff_t i = 0; i < n_nodes; i++) {
            const count_t begin = rowptr[i];
            const count_t end   = rowptr[i + 1];

            for (count_t k = begin; k < end; k++) {
                const idx_t j = colidx[k];

                if (i < j) {
                    const idx_t nidx = p2idx[k];

                    for (int d = 0; d < spatial_dim; d++) {
                        const geom_t xi = p2_points[d][i];
                        const geom_t xj = p2_points[d][j];
                        p2_points[d][nidx] = (xi + xj) / 2;
                    }
                }
            }
        }

        int SFEM_MAP_TO_SPHERE = 0;
        SFEM_READ_ENV(SFEM_MAP_TO_SPHERE, atoi);
        if (SFEM_MAP_TO_SPHERE) {
            float SFEM_SPERE_RADIUS = 0.5f;
            SFEM_READ_ENV(SFEM_SPERE_RADIUS, atof);

            double SFEM_SPERE_TOL = 1e-8;
            SFEM_READ_ENV(SFEM_SPERE_TOL, atof);

            for (ptrdiff_t i = 0; i < n_nodes; i++) {
                const count_t begin = rowptr[i];
                const count_t end   = rowptr[i + 1];

                for (count_t k = begin; k < end; k++) {
                    const idx_t j = colidx[k];

                    if (i < j) {
                        const idx_t nidx = p2idx[k];

                        geom_t r1 = 0;
                        geom_t r2 = 0;
                        geom_t mr = 0;

                        for (int d = 0; d < spatial_dim; d++) {
                            const geom_t xi  = p2_points[d][i];
                            const geom_t xj  = p2_points[d][j];
                            r1 += xi * xi;
                            r2 += xj * xj;

                            const geom_t mxi = p2_points[d][nidx];
                            mr += mxi * mxi;
                        }

                        r1 = sqrt(r1);
                        r2 = sqrt(r2);
                        mr = r1 / sqrt(mr);

                        if (fabs(r1 - SFEM_SPERE_RADIUS) < SFEM_SPERE_TOL &&
                            fabs(r2 - SFEM_SPERE_RADIUS) < SFEM_SPERE_TOL) {
                            for (int d = 0; d < spatial_dim; d++) {
                                p2_points[d][nidx] *= mr;
                            }
                        }
                    }
                }
            }
        }

        auto p2_mesh = std::make_shared<Mesh>(p1_mesh->comm(),
                                              spatial_dim,
                                              p2_element_type,
                                              n_elements,
                                              p2_elements_buffer,
                                              n_nodes + p2_nodes,
                                              p2_points_buffer);

        free(p2idx);
        free(rowptr);
        free(colidx);

        return p2_mesh;
    }

}  // namespace sfem
