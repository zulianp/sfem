#include "sfem_mesh.h"
#include "sortreduce.h"
#include "sfem_defs.h"

#include <string.h>

void mesh_aura(const mesh_t *mesh, mesh_t *aura) {
    MPI_Comm comm = mesh->comm;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    const ptrdiff_t n_owned_elements = mesh->n_owned_elements;
    const ptrdiff_t n_shared_elements = mesh->n_shared_elements;

    const int nnxe = elem_num_nodes(mesh->element_type);
    idx_t **shared_elements = (idx_t **)malloc(nnxe * sizeof(idx_t *));

    count_t *send_displs = (count_t *)malloc((size + 1) * sizeof(count_t));
    memset(send_displs, 0, (size + 1) * sizeof(count_t));
    idx_t *owner = malloc(nnxe * sizeof(idx_t));

    ptrdiff_t n_send_elements = 0;
    for (ptrdiff_t es = 0; es < n_shared_elements; es++) {
        for (int d = 0; d < nnxe; d++) {
            owner[d] = mesh->node_owner[mesh->elements[d][n_owned_elements + es]];
        }

        int n_owners = sortreduce(owner, nnxe);
        for (int i = 0; i < n_owners; i++) {
            send_displs[owner[i] + 1]++;
            n_send_elements++;
        }
    }

    for (int r = 0; r < size; r++) {
        send_displs[r + 1] += send_displs[r];
    }

    for (int d = 0; d < nnxe; d++) {
        shared_elements[d] = (idx_t *)malloc(n_send_elements * sizeof(idx_t));
    }

    count_t *send_count = (count_t *)malloc((size + 1) * sizeof(count_t));
    memset(send_count, 0, (size + 1) * sizeof(count_t));

    for (ptrdiff_t es = 0; es < n_shared_elements; es++) {
        for (int d = 0; d < nnxe; d++) {
            owner[d] = mesh->node_owner[mesh->elements[d][n_owned_elements + es]];
        }

        int n_owners = sortreduce(owner, nnxe);

        for (int i = 0; i < n_owners; i++) {
        	int owner_rank = owner[i];
            for (int d = 0; d < nnxe; d++) {
                const idx_t nidx = mesh->node_mapping[mesh->elements[d][n_owned_elements + es]];
            	shared_elements[d][send_displs[owner_rank] + send_count[owner_rank]] = nidx;
            }

            send_count[owner_rank]++;
        }
    }

    // Store in output mesh
    aura->comm = comm;
    // aura->elements

    free(owner);
    free(send_displs);
    free(send_count);
}
