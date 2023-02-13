
#include "read_mesh.h"

#include "../matrix.io/matrixio_array.h"
#include "../matrix.io/utils.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <mpi.h>

#include "sortreduce.h"

int mesh_read_generic(MPI_Comm comm, const int nnodesxelem, const int ndims, const char *folder, mesh_t *mesh) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (size > 1) {
        double tick = MPI_Wtime();
        ///////////////////////////////////////////////////////////////

        MPI_Datatype mpi_geom_t = SFEM_MPI_GEOM_T;
        MPI_Datatype mpi_idx_t = SFEM_MPI_IDX_T;

        // ///////////////////////////////////////////////////////////////

        ptrdiff_t n_local_elements = 0, n_elements = 0;
        ptrdiff_t n_local_nodes = 0, n_nodes = 0;

        char path[1024 * 10];

        idx_t **elems = (idx_t **)malloc(sizeof(idx_t *) * nnodesxelem);

        {
            idx_t *idx = 0;

            for (int d = 0; d < nnodesxelem; ++d) {
                sprintf(path, "%s/i%d.raw", folder, d);
                array_read(comm, path, mpi_idx_t, (void **)&idx, &n_local_elements, &n_elements);
                elems[d] = idx;
            }
        }

        idx_t *unique_idx = (idx_t *)malloc(sizeof(idx_t) * n_local_elements * nnodesxelem);
        for (int d = 0; d < nnodesxelem; ++d) {
            memcpy(&unique_idx[d * n_local_elements], elems[d], sizeof(idx_t) * n_local_elements);
        }

        ptrdiff_t n_unique = sortreduce(unique_idx, n_local_elements * nnodesxelem);

        ////////////////////////////////////////////////////////////////////////////////
        // Read coordinates
        ////////////////////////////////////////////////////////////////////////////////

        geom_t **xyz = (geom_t **)malloc(sizeof(geom_t *) * ndims);

        static const char *str_xyz = "xyzt";

        for (int d = 0; d < ndims; ++d) {
            sprintf(path, "%s/%c.raw", folder, str_xyz[d]);
            array_read(comm, path, mpi_geom_t, (void **)&xyz[d], &n_local_nodes, &n_nodes);
        }

        ////////////////////////////////////////////////////////////////////////////////

        idx_t *input_node_partitions = (idx_t *)malloc(sizeof(idx_t) * (size + 1));
        memset(input_node_partitions, 0, sizeof(idx_t) * (size + 1));
        input_node_partitions[rank + 1] = n_local_nodes;

        CATCH_MPI_ERROR(MPI_Allreduce(MPI_IN_PLACE, &input_node_partitions[1], size, SFEM_MPI_IDX_T, MPI_SUM, comm));

        for (int r = 0; r < size; ++r) {
            input_node_partitions[r + 1] += input_node_partitions[r];
        }

        idx_t *gather_node_count = malloc(size * sizeof(idx_t));
        memset(gather_node_count, 0, size * sizeof(idx_t));

        int *owner_rank = malloc(n_unique * sizeof(int));
        memset(owner_rank, 0, n_unique * sizeof(int));

        for (ptrdiff_t i = 0; i < n_unique; ++i) {
            idx_t idx = unique_idx[i];
            ptrdiff_t owner = MIN(size - 1, idx / n_local_nodes);

            if (input_node_partitions[owner] > idx) {
                assert(owner < size);
                while (input_node_partitions[--owner] > idx) {
                    assert(owner < size);
                }
            } else if (input_node_partitions[owner + 1] < idx) {
                assert(owner < size);

                while (input_node_partitions[(++owner + 1)] < idx) {
                    assert(owner < size);
                }
            }

            assert(owner < size);
            assert(owner >= 0);

            assert(input_node_partitions[owner] <= idx);
            assert(input_node_partitions[owner + 1] > idx);

            gather_node_count[owner]++;
            owner_rank[i] = owner;
        }

        idx_t *scatter_node_count = malloc(size * sizeof(idx_t));
        memset(scatter_node_count, 0, size * sizeof(idx_t));

        CATCH_MPI_ERROR(
            MPI_Alltoall(gather_node_count, 1, SFEM_MPI_IDX_T, scatter_node_count, 1, SFEM_MPI_IDX_T, comm));

        idx_t *gather_node_displs = malloc((size + 1) * sizeof(idx_t));
        idx_t *scatter_node_displs = malloc((size + 1) * sizeof(idx_t));

        gather_node_displs[0] = 0;
        scatter_node_displs[0] = 0;

        for (int i = 0; i < size; i++) {
            gather_node_displs[i + 1] = gather_node_displs[i] + gather_node_count[i];
        }

        for (int i = 0; i < size; i++) {
            scatter_node_displs[i + 1] = scatter_node_displs[i] + scatter_node_count[i];
        }

        ptrdiff_t size_send_list = scatter_node_displs[size];

        idx_t *send_list = malloc(sizeof(idx_t) * size_send_list);
        memset(send_list, 0, sizeof(idx_t) * size_send_list);

        CATCH_MPI_ERROR(MPI_Alltoallv(unique_idx,
                                      gather_node_count,
                                      gather_node_displs,
                                      SFEM_MPI_IDX_T,
                                      send_list,
                                      scatter_node_count,
                                      scatter_node_displs,
                                      SFEM_MPI_IDX_T,
                                      comm));

        ///////////////////////////////////////////////////////////////////////

        // Remove offset
        for (ptrdiff_t i = 0; i < size_send_list; ++i) {
            send_list[i] -= input_node_partitions[rank];
        }

        ///////////////////////////////////////////////////////////////////////
        // Exchange points

        geom_t *sendx = (geom_t *)malloc(size_send_list * sizeof(geom_t));
        geom_t **part_xyz = (geom_t **)malloc(sizeof(geom_t *) * ndims);

        for (int d = 0; d < ndims; ++d) {
            // Fill buffer
            for (ptrdiff_t i = 0; i < size_send_list; ++i) {
                sendx[i] = xyz[d][send_list[i]];
            }

            geom_t *recvx = (geom_t *)malloc(n_unique * sizeof(geom_t));
            CATCH_MPI_ERROR(MPI_Alltoallv(sendx,
                                          scatter_node_count,
                                          scatter_node_displs,
                                          SFEM_MPI_GEOM_T,
                                          recvx,
                                          gather_node_count,
                                          gather_node_displs,
                                          SFEM_MPI_GEOM_T,
                                          comm));
            part_xyz[d] = recvx;

            // Free space
            free(xyz[d]);
        }

        ///////////////////////////////////////////////////////////////////////
        // Determine owners
        int *node_owner = (idx_t *)malloc(n_unique * sizeof(idx_t));
        
        {   
            int *decide_node_owner = (int*)malloc(n_local_nodes * sizeof(int));

            int *send_node_owner = (int *)malloc(size_send_list * sizeof(int));

            for (ptrdiff_t i = 0; i < n_local_nodes; ++i) {
                decide_node_owner[i] = size;
            }

            for (int r = 0; r < size; ++r) {
                idx_t begin = scatter_node_displs[r];
                idx_t end = scatter_node_displs[r + 1];

                for (idx_t i = begin; i < end; ++i) {
                    decide_node_owner[send_list[i]] = MIN(decide_node_owner[send_list[i]], r);
                }
            }

            for (int r = 0; r < size; ++r) {
                idx_t begin = scatter_node_displs[r];
                idx_t end = scatter_node_displs[r + 1];

                for (idx_t i = begin; i < end; ++i) {
                    send_node_owner[i] = decide_node_owner[send_list[i]];
                }
            }

            CATCH_MPI_ERROR(MPI_Alltoallv(send_node_owner,
                                          scatter_node_count,
                                          scatter_node_displs,
                                          MPI_INT,
                                          node_owner,
                                          gather_node_count,
                                          gather_node_displs,
                                          MPI_INT,
                                          comm));

            free(decide_node_owner);
            free(send_node_owner);
        }

        ///////////////////////////////////////////////////////////////////////
        // Localize element index
        for (ptrdiff_t d = 0; d < nnodesxelem; ++d) {
            for (ptrdiff_t e = 0; e < n_local_elements; ++e) {
                elems[d][e] = find_idx_binary_search(elems[d][e], unique_idx, n_unique);
            }
        }

        ///////////////////////////////////////////////////////////////////////
        // MPI_Barrier(comm);
        // for (int r = 0; r < size; ++r) {
        //     if (r == rank) {
        //         printf("[%d] elements\n", rank);

        //         for (ptrdiff_t e = 0; e < n_local_elements; ++e) {
        //             for (ptrdiff_t d = 0; d < nnodesxelem; ++d) {
        //                 printf(
        //                     "%d (%d, %d), ", (int)elems[d][e], (int)unique_idx[elems[d][e]], (int)node_owner[elems[d][e]]);
        //             }

        //             printf("\n");
        //         }

        //         printf("\n");
        //         printf("------------------\n");

        //         fflush(stdout);
        //     }

        //     sleep(0.1);

        //     MPI_Barrier(comm);
        // }

        ///////////////////////////////////////////////////////////////////////
        // Free space
        // free(unique_idx);
        free(sendx);
        free(send_list);
        free(input_node_partitions);

        free(scatter_node_count);
        free(gather_node_count);
        free(gather_node_displs);
        free(scatter_node_displs);

        ///////////////////////////////////////////////////////////////////////
        mesh->comm = comm;
        mesh->mem_space = SFEM_MEM_SPACE_HOST;

        
        mesh->spatial_dim = 3;
        mesh->element_type = 4;

        mesh->elements = elems;
        mesh->points = part_xyz;
        mesh->nelements = n_local_elements;
        mesh->nnodes = n_unique;

        // Original indexing
        mesh->mapping = unique_idx;
        mesh->node_owner = node_owner;

        double tock = MPI_Wtime();
        return 0;
    } else {
        // Serial fallback
        ///////////////////////////////////////////////////////////////

        MPI_Datatype mpi_geom_t = SFEM_MPI_GEOM_T;
        MPI_Datatype mpi_idx_t = SFEM_MPI_IDX_T;

        // ///////////////////////////////////////////////////////////////

        ptrdiff_t n_local_elements = 0, n_elements = 0;
        ptrdiff_t n_local_nodes = 0, n_nodes = 0;

        char path[1024 * 10];

        idx_t **elems = (idx_t **)malloc(sizeof(idx_t *) * nnodesxelem);

        {
            idx_t *idx = 0;

            for (int d = 0; d < nnodesxelem; ++d) {
                sprintf(path, "%s/i%d.raw", folder, d);
                array_read(comm, path, mpi_idx_t, (void **)&idx, &n_local_elements, &n_elements);
                elems[d] = idx;
            }
        }

        ////////////////////////////////////////////////////////////////////////////////
        // Read coordinates
        ////////////////////////////////////////////////////////////////////////////////

        geom_t **xyz = (geom_t **)malloc(sizeof(geom_t *) * ndims);

        static const char *str_xyz = "xyzt";

        for (int d = 0; d < ndims; ++d) {
            sprintf(path, "%s/%c.raw", folder, str_xyz[d]);
            array_read(comm, path, mpi_geom_t, (void **)&xyz[d], &n_local_nodes, &n_nodes);
        }

        mesh->comm = comm;

        mesh->mem_space = SFEM_MEM_SPACE_HOST;

        mesh->spatial_dim = ndims;
        mesh->element_type = nnodesxelem;

        mesh->nelements = n_local_elements;
        mesh->nnodes = n_local_nodes;

        mesh->elements = elems;
        mesh->points = xyz;

        mesh->mapping = 0;
        mesh->node_owner = 0;

        return 0;
    }
}

static ptrdiff_t read_file(MPI_Comm comm, const char *path, void **data) {
    MPI_Status status;
    MPI_Offset nbytes;
    MPI_File file;
    CATCH_MPI_ERROR(MPI_File_open(comm, path, MPI_MODE_RDONLY, MPI_INFO_NULL, &file));
    CATCH_MPI_ERROR(MPI_File_get_size(file, &nbytes));
    *data = malloc(nbytes);

    CATCH_MPI_ERROR(MPI_File_read_at_all(file, 0, *data, nbytes, MPI_CHAR, &status));
    return nbytes;
}

int serial_read_tet_mesh(const char *folder, ptrdiff_t *nelements, idx_t *elems[4], ptrdiff_t *nnodes, geom_t *xyz[3]) {
    char path[1024 * 10];

    {
        sprintf(path, "%s/x.raw", folder);
        ptrdiff_t x_nnodes = read_file(MPI_COMM_SELF, path, (void **)&xyz[0]);

        sprintf(path, "%s/y.raw", folder);
        ptrdiff_t y_nnodes = read_file(MPI_COMM_SELF, path, (void **)&xyz[1]);

        sprintf(path, "%s/z.raw", folder);
        ptrdiff_t z_nnodes = read_file(MPI_COMM_SELF, path, (void **)&xyz[2]);

        assert(x_nnodes == y_nnodes);
        assert(x_nnodes == z_nnodes);

        x_nnodes /= sizeof(geom_t);
        assert(x_nnodes * sizeof(geom_t) == y_nnodes);
        *nnodes = x_nnodes;
    }

    {
        sprintf(path, "%s/i0.raw", folder);
        ptrdiff_t nindex0 = read_file(MPI_COMM_SELF, path, (void **)&elems[0]);

        sprintf(path, "%s/i1.raw", folder);
        ptrdiff_t nindex1 = read_file(MPI_COMM_SELF, path, (void **)&elems[1]);

        sprintf(path, "%s/i2.raw", folder);
        ptrdiff_t nindex2 = read_file(MPI_COMM_SELF, path, (void **)&elems[2]);

        sprintf(path, "%s/i3.raw", folder);
        ptrdiff_t nindex3 = read_file(MPI_COMM_SELF, path, (void **)&elems[3]);

        assert(nindex0 == nindex1);
        assert(nindex3 == nindex2);

        nindex0 /= sizeof(idx_t);
        assert(nindex0 * sizeof(idx_t) == nindex1);
        *nelements = nindex0;
    }

    return 0;
}

int mesh_surf_read(MPI_Comm comm, const char *folder, mesh_t *mesh)
{
    int nnodesxelem = 3;
    int ndims = 3;
    return mesh_read_generic(comm, nnodesxelem, ndims, folder, mesh);
}

int mesh_read(MPI_Comm comm, const char *folder, mesh_t *mesh) {
    int nnodesxelem = 4;
    int ndims = 3;
    return mesh_read_generic(comm, nnodesxelem, ndims, folder, mesh);
}
