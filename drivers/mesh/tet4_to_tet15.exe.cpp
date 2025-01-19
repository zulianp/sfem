#include "sfem_API.hpp"

#include "sortreduce.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (argc != 3) {
        if (!rank) {
            fprintf(stderr, "usage: %s <tet4_mesh> <output_tet15_mesh>\n", argv[0]);
        }

        return EXIT_FAILURE;
    }

    auto        tet4_mesh     = sfem::Mesh::create_from_file(comm, argv[1]);
    const char *output_folder = argv[2];

    const ptrdiff_t n_elements = tet4_mesh->n_elements();
    const ptrdiff_t n_nodes    = tet4_mesh->n_nodes();
    auto            edge_graph = tet4_mesh->node_to_node_graph_upper_triangular();
    auto            side_table = tet4_mesh->half_face_table();
    int             nsxe       = elem_num_sides(tet4_mesh->element_type());

    auto side_ids = sfem::create_host_buffer<idx_t>(side_table->size());

    const ptrdiff_t id_offset      = tet4_mesh->n_nodes() + edge_graph->nnz();
    ptrdiff_t       n_unique_sides = 0;
    {
        auto st = side_table->data();
        auto si = side_ids->data();
        for (ptrdiff_t e = 0; e < n_elements; e++) {
            for (int s = 0; s < nsxe; s++) {
                const ptrdiff_t     offset = e * nsxe + s;
                const element_idx_t neigh  = st[offset];
                if (neigh == -1 || e < neigh) {
                    // Create new id
                    si[offset] = id_offset + n_unique_sides;
                    n_unique_sides++;
                } else if (neigh != -1) {
                    // Search id in neighbor
                    for (int sneigh = 0; sneigh < nsxe; sneigh++) {
                        const ptrdiff_t offset_neigh = neigh * nsxe + sneigh;
                        if (st[offset_neigh] == e) {
                            si[offset_neigh] = si[offset];
                        }
                    }
                }
            }
        }
    }

    auto            tet15_elements = sfem::create_host_buffer<idx_t>(15, n_elements);
    const ptrdiff_t n_edges        = edge_graph->nnz();

    {
        // Element connectivity

        auto rowptr      = edge_graph->rowptr()->data();
        auto colidx      = edge_graph->colidx()->data();
        auto si          = side_ids->data();
        auto tet4_elems  = tet4_mesh->elements()->data();
        auto tet15_elems = tet15_elements->data();

        for (ptrdiff_t e = 0; e < n_elements; e++) {
            // Nodes
            idx_t ii[15] = {tet4_elems[0][e], tet4_elems[1][e], tet4_elems[2][e], tet4_elems[3][e]};

            // Edges
            {
                idx_t row[6];
                row[0] = MIN(tet4_elems[0][e], tet4_elems[1][e]);
                row[1] = MIN(tet4_elems[1][e], tet4_elems[2][e]);
                row[2] = MIN(tet4_elems[0][e], tet4_elems[2][e]);
                row[3] = MIN(tet4_elems[0][e], tet4_elems[3][e]);
                row[4] = MIN(tet4_elems[1][e], tet4_elems[3][e]);
                row[5] = MIN(tet4_elems[2][e], tet4_elems[3][e]);

                idx_t key[6];
                key[0] = MAX(tet4_elems[0][e], tet4_elems[1][e]);
                key[1] = MAX(tet4_elems[1][e], tet4_elems[2][e]);
                key[2] = MAX(tet4_elems[0][e], tet4_elems[2][e]);
                key[3] = MAX(tet4_elems[0][e], tet4_elems[3][e]);
                key[4] = MAX(tet4_elems[1][e], tet4_elems[3][e]);
                key[5] = MAX(tet4_elems[2][e], tet4_elems[3][e]);

                for (int l = 0; l < 6; l++) {
                    const idx_t   r         = row[l];
                    const count_t row_begin = rowptr[r];
                    const count_t len_row   = rowptr[r + 1] - row_begin;
                    const idx_t  *cols      = &colidx[row_begin];
                    const idx_t   k         = find_idx_binary_search(key[l], cols, len_row);
                    ii[4 + l]               = row_begin + k + n_nodes;
                }
            }

            // Faces
            {
                ii[10] = si[e * nsxe + 0];
                ii[11] = si[e * nsxe + 1];
                ii[12] = si[e * nsxe + 2];
                ii[13] = si[e * nsxe + 3];
            }

            // Volume
            ii[14] = n_nodes + n_edges + n_unique_sides + e;

            for (int node = 0; node < 15; node++) {
                tet15_elems[node][e] = ii[node];
            }
        }
    }

    // Corners + Edges + Sides + Volume
    const ptrdiff_t tet15_n_nodes = n_nodes + n_edges + n_unique_sides + n_elements;
    auto            tet15_points  = sfem::create_host_buffer<geom_t>(3, tet15_n_nodes);

    {
        auto rowptr = edge_graph->rowptr()->data();
        auto colidx = edge_graph->colidx()->data();

        auto tet4_pts    = tet4_mesh->points()->data();
        auto tet15_pts   = tet15_points->data();
        auto tet15_elems = tet15_elements->data();

        // Edge points
        for (ptrdiff_t i = 0; i < n_nodes; i++) {
            const count_t row_begin = rowptr[i];
            const count_t len_row   = rowptr[i + 1] - row_begin;
            const idx_t  *cols      = &colidx[row_begin];

            for (int d = 0; d < 3; d++) {
                tet15_pts[d][i] = tet4_pts[d][i];
            }

            for (count_t k = 0; k < len_row; k++) {
                const idx_t j    = cols[k];
                const idx_t edge = n_nodes + row_begin + k;

                for (int d = 0; d < 3; d++) {
                    tet15_pts[d][edge] = (tet4_pts[d][i] + tet4_pts[d][j]) / 2;
                }
            }
        }

        // Rest of the nodes
        for (ptrdiff_t e = 0; e < n_elements; e++) {
            // Collect indices
            // Nodes
            idx_t ii[15] = {tet15_elems[0][e], tet15_elems[1][e], tet15_elems[2][e], tet15_elems[3][e]};

            // Faces
            {
                ii[10] = tet15_elems[10][e];
                ii[11] = tet15_elems[11][e];
                ii[12] = tet15_elems[12][e];
                ii[13] = tet15_elems[13][e];
            }

            // Volume
            ii[14] = tet15_elems[14][e];

            // Compute points
            for (int d = 0; d < 3; d++) {
                tet15_pts[d][ii[10]] = (tet4_pts[d][ii[0]] + tet4_pts[d][ii[1]] + tet4_pts[d][ii[3]]) / 3;
                tet15_pts[d][ii[11]] = (tet4_pts[d][ii[1]] + tet4_pts[d][ii[2]] + tet4_pts[d][ii[3]]) / 3;
                tet15_pts[d][ii[12]] = (tet4_pts[d][ii[0]] + tet4_pts[d][ii[3]] + tet4_pts[d][ii[2]]) / 3;
                tet15_pts[d][ii[13]] = (tet4_pts[d][ii[0]] + tet4_pts[d][ii[2]] + tet4_pts[d][ii[1]]) / 3;
            }

            for (int d = 0; d < 3; d++) {
                tet15_pts[d][ii[14]] = (tet4_pts[d][ii[0]] + tet4_pts[d][ii[1]] + tet4_pts[d][ii[2]] + +tet4_pts[d][ii[3]]) / 4;
            }
        }
    }

    // Output
    sfem::create_directory(output_folder);

    std::string path_output_format = output_folder;
    path_output_format += "/i%d.raw";
    tet15_elements->to_files(path_output_format.c_str());

    path_output_format = output_folder;
    path_output_format += "/x%d.raw";
    tet15_points->to_files(path_output_format.c_str());

    return MPI_Finalize();
}
