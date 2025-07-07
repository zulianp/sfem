#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "array_dtof.h"
#include "matrixio_array.h"
#include "matrixio_crs.h"
#include "mpi.h"
#include "utils.h"

#include "crs_graph.h"
#include "read_mesh.h"
#include "sfem_base.h"
#include "sfem_defs.h"
#include "sfem_mesh_write.h"

#include "sortreduce.h"
#include "sfem_glob.hpp"
#include "sfem_API.hpp"

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (argc != 3) {
        if (!rank) {
            fprintf(stderr, "usage: %s <folder> <output_folder>\n", argv[0]);
        }

        return EXIT_FAILURE;
    }

    const char *output_folder = argv[2];

    if (!rank) {
        printf("%s %s %s\n", argv[0], argv[1], output_folder);
    }

    double tick = MPI_Wtime();
    sfem::create_directory(output_folder);

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    const char *folder = argv[1];
    // char path[1024 * 10];

    auto mesh = sfem::Mesh::create_from_file(sfem::Communicator::wrap(comm), folder);

    int nnxe = elem_num_nodes(mesh->element_type());

    local_idx_t **lelements = (local_idx_t**)malloc(nnxe * sizeof(local_idx_t *));

    for (int d = 0; d < nnxe; d++) {
        lelements[d] = (local_idx_t*)calloc(mesh->n_elements(), sizeof(local_idx_t));
    }

    ptrdiff_t max_rep_limit = 1 << (8 * sizeof(local_idx_t) - 1);
    ptrdiff_t SFEM_ELEMENT_BLOCK_SIZE = max_rep_limit;
    SFEM_READ_ENV(SFEM_ELEMENT_BLOCK_SIZE, atol);

    // max_block_size <= 32'768
    ptrdiff_t max_block_size = MIN(SFEM_ELEMENT_BLOCK_SIZE, mesh->n_elements());
    ptrdiff_t num_blocks = MAX((mesh->n_elements() + max_block_size - 1) / max_block_size, 1);

    local_idx_t *node_lidx = (local_idx_t*)malloc(mesh->n_nodes() * sizeof(local_idx_t));
    int *nodes_in_block = (int*)calloc(num_blocks, sizeof(int));

    for (ptrdiff_t i = 0; i < mesh->n_nodes(); i++) {
        node_lidx[i] = SFEM_LOCAL_IDX_INVALID;
    }

    auto elements = mesh->elements()->data();
    for (ptrdiff_t e_offset = 0, block_num = 0; e_offset < mesh->n_elements();
         e_offset += max_block_size, block_num++) {
        const ptrdiff_t next_offset = e_offset + MIN(max_block_size, mesh->n_elements() - e_offset);

        local_idx_t last_index = 0;
        for (ptrdiff_t e = e_offset; e < next_offset; e++) {
            for (int d = 0; d < nnxe; d++) {
                const idx_t node = elements[d][e];

                if (node_lidx[node] == SFEM_LOCAL_IDX_INVALID) {
                    // Create index
                    node_lidx[node] = last_index++;
                    nodes_in_block[block_num]++;

                    assert(node_lidx[node] < nodes_in_block[block_num]);
                }

                lelements[d][e] = node_lidx[node];
            }
        }

        for (ptrdiff_t e = e_offset; e < next_offset; e++) {
            for (int d = 0; d < nnxe; d++) {
                const idx_t node = elements[d][e];
                node_lidx[node] = SFEM_LOCAL_IDX_INVALID;
            }
        }
    }

    int violated_max_count = 0;
    ptrdiff_t new_num_nodes = 0;
    for (int b = 0; b < num_blocks; b++) {
        new_num_nodes += nodes_in_block[b];
        violated_max_count += nodes_in_block[b] > max_rep_limit;
    }

    if(violated_max_count) {
        fprintf(stderr, "Error: violated max rep count\n");
        for (int b = 0; b < num_blocks; b++) {
            fprintf(stderr, "%d < %ld\n", nodes_in_block[b], max_rep_limit);
        }

        MPI_Abort(comm, SFEM_FAILURE);
    }

    idx_t *node_mapping = (idx_t *)calloc(new_num_nodes, sizeof(idx_t));

    ptrdiff_t lnode_offset = 0;
    for (ptrdiff_t e_offset = 0, block_num = 0; e_offset < mesh->n_elements();
         e_offset += max_block_size, block_num++) {
        const ptrdiff_t block_size = MIN(max_block_size, mesh->n_elements() - e_offset);
        const ptrdiff_t next_offset = e_offset + block_size;

        for (ptrdiff_t e = e_offset; e < next_offset; e++) {
            for (int d = 0; d < nnxe; d++) {
                local_idx_t lnode = lelements[d][e];

                assert(lnode < nodes_in_block[block_num]);

                const ptrdiff_t offset = lnode_offset + lnode;
                assert(offset < new_num_nodes);
                node_mapping[offset] = elements[d][e];
            }
        }

        lnode_offset += nodes_in_block[block_num];
    }

    if (0) {
        lnode_offset = 0;
        for (ptrdiff_t e_offset = 0, block_num = 0; e_offset < mesh->n_elements();
             e_offset += max_block_size, block_num++) {
            ptrdiff_t block_size = MIN(max_block_size, mesh->n_elements() - e_offset);
            ptrdiff_t next_offset = e_offset + block_size;

            printf("block %ld) [%ld, %ld) #elements %ld #nodes %d\n",
                   block_num,
                   e_offset,
                   next_offset,
                   block_size,
                   nodes_in_block[block_num]);

            local_idx_t last_index = 0;
            for (ptrdiff_t e = e_offset; e < next_offset; e++) {
                for (int d = 0; d < nnxe; d++) {
                    printf("%" d_LOCAL_IDX_T " ", lelements[d][e]);
                }

                printf("\t-> ");

                for (int d = 0; d < nnxe; d++) {
                    printf("%" d_IDX_T " ", node_mapping[lnode_offset + lelements[d][e]]);
                }

                printf("\n");
            }

            lnode_offset += nodes_in_block[block_num];
        }
    }

    // int * node_membership_count = calloc(mesh.nnodes, sizeof(int));

    ///////////////////////////////////////////////////////////////////////////////

    if (!rank) {
        printf("----------------------------------------\n");
        printf("mesh_to_blocks.c: #elements %ld #nodes %ld #blocks %ld #node_mapping %ld\n",
               (long)mesh->n_elements(),
               (long)mesh->n_nodes(),
               (long)num_blocks,
               (long)new_num_nodes);
        printf("----------------------------------------\n");
    }

    // Clean up

    for (int d = 0; d < nnxe; d++) {
        free(lelements[d]);
    }

    free(lelements);
    // free(b_size);
    free(node_lidx);
    free(node_mapping);

    double tock = MPI_Wtime();
    if (!rank) {
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
