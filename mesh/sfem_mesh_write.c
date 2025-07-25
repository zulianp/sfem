#include "sfem_mesh_write.h"

#include <stdint.h>
#include <sys/stat.h>
#include <sys/types.h>
// #include <unistd.h>

#include <assert.h>
#include <stdio.h>
#include <string.h>

// matrix.io
#include "matrixio_array.h"
#include "utils.h"
#include "sfem_glob.h"

int write_raw_array(const char *path, const int type_size, const void *data, const ptrdiff_t n_elements) {
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        SFEM_ERROR("write_raw_array: Unable to write file %s\n", path);
    }
    fwrite(data, type_size, n_elements, fp);
    fclose(fp);
    return SFEM_SUCCESS;
}

int mesh_write_serial(const char     *path,
                      enum ElemType   element_type,
                      const ptrdiff_t n_elements,
                      idx_t **const   elements,
                      const int       spatial_dim,
                      const ptrdiff_t n_nodes,
                      geom_t **const  points) {
    static const char *str_xyz = "xyzt";
    char               output_path[2048];

    create_directory(path);

    for (int d = 0; d < spatial_dim; ++d) {
        snprintf(output_path, sizeof(output_path), "%s/%c.raw", path, str_xyz[d]);
        write_raw_array(output_path, sizeof(geom_t), points[d], n_nodes);
    }

    const int nxe = elem_num_nodes(element_type);
    for (int d = 0; d < nxe; ++d) {
        snprintf(output_path, sizeof(output_path), "%s/i%d.raw", path, d);
        write_raw_array(output_path, sizeof(idx_t), elements[d], n_elements);
    }

    snprintf(output_path, sizeof(output_path), "%s/meta.yaml", path);
    FILE *meta_file = fopen(output_path, "w");
    if (!meta_file) {
        SFEM_ERROR("mesh_write: Unable to write file %s\n", output_path);
    }

    fprintf(meta_file, "# SFEM mesh meta file (generated by sfem_mesh_write.c)\n");
    fprintf(meta_file, "spatial_dimension: %d\n", spatial_dim);
    fprintf(meta_file, "elem_num_nodes: %d\n", nxe);
    fprintf(meta_file, "element_type: %s\n", type_to_string(element_type));
    fprintf(meta_file, "elements:\n");

    for (int d = 0; d < nxe; ++d) {
        fprintf(meta_file, "- i%d: i%d.raw\n", d, d);
    }

    fprintf(meta_file, "points:\n");
    for (int d = 0; d < spatial_dim; d++) {
        fprintf(meta_file, "- %c: %c.raw\n", str_xyz[d], str_xyz[d]);
    }

    fprintf(meta_file, "rpath: true\n");
    fclose(meta_file);

    return SFEM_SUCCESS;
}

int mesh_multiblock_write_serial(
                      const char     *path,
                      const uint16_t n_blocks,
                      const char     *block_names[],
                      enum ElemType   element_types[],
                      const ptrdiff_t n_elements[],
                      idx_t **const   elements[],
                      const int       spatial_dim,
                      const ptrdiff_t n_nodes,
                      geom_t **const  points) {

    if(n_blocks == 1) {
        return mesh_write_serial(path, element_types[0], n_elements[0], elements[0], spatial_dim, n_nodes, points);
    }

    static const char *str_xyz = "xyzt";
    char               output_path[2048];

    create_directory(path);

    snprintf(output_path, sizeof(output_path), "%s/blocks", path);
    create_directory(output_path);

    for (int d = 0; d < spatial_dim; ++d) {
        snprintf(output_path, sizeof(output_path), "%s/%c.raw", path, str_xyz[d]);
        write_raw_array(output_path, sizeof(geom_t), points[d], n_nodes);
    }

    for (int b = 0; b < n_blocks; ++b) {
        snprintf(output_path, sizeof(output_path), "%s/blocks/%d", path, b);
        create_directory(output_path);
        const int nxe = elem_num_nodes(element_types[b]);
        for (int d = 0; d < nxe; ++d) {
            snprintf(output_path, sizeof(output_path), "%s/blocks/%d/i%d.raw", path, b, d);
            write_raw_array(output_path, sizeof(idx_t), elements[b][d], n_elements[b]);
        }
    }

    snprintf(output_path, sizeof(output_path), "%s/meta.yaml", path);
    FILE *meta_file = fopen(output_path, "w");
    if (!meta_file) {
        SFEM_ERROR("mesh_write: Unable to write file %s\n", output_path);
    }

    fprintf(meta_file, "# SFEM mesh meta file (generated by sfem_mesh_write.c)\n");
    fprintf(meta_file, "spatial_dimension: %d\n", spatial_dim);

    fprintf(meta_file, "n_blocks: %d\n", n_blocks);
    fprintf(meta_file, "blocks:\n");
   
    for (int b = 0; b < n_blocks; ++b) {
        int nxe = elem_num_nodes(element_types[b]);
        fprintf(meta_file, "- name: %s\n", block_names[b]);
        fprintf(meta_file, "  element_type: %s\n", type_to_string(element_types[b]));
        fprintf(meta_file, "  elem_num_nodes: %d\n", nxe);
        fprintf(meta_file, "  n_elements: %ld\n", n_elements[b]);
        fprintf(meta_file, "  elements:\n");
        for (int d = 0; d < nxe; ++d) {
            fprintf(meta_file, "  - i%d: blocks/%d/i%d.raw\n", d, b, d);
        }
    }

    fprintf(meta_file, "points:\n");
    for (int d = 0; d < spatial_dim; d++) {
        fprintf(meta_file, "- %c: %c.raw\n", str_xyz[d], str_xyz[d]);
    }

    fprintf(meta_file, "rpath: true\n");
    fclose(meta_file);

    return SFEM_SUCCESS;
}

int mesh_write(const char *path, const mesh_t *mesh) {
    // TODO
    // MPI_Comm comm = mesh->comm;
    MPI_Comm comm = MPI_COMM_SELF;

    int rank, size;
    MPI_Comm_rank(mesh->comm, &rank);
    MPI_Comm_size(mesh->comm, &size);

    static const char *str_xyz = "xyzt";

    char folder[2048];
    char output_path[2048];

    create_directory(path);

    ptrdiff_t nelements;
    if (size > 1) {
        snprintf(folder, sizeof(folder), "%s/%d", path, rank);
        nelements = mesh->n_owned_elements;
    } else {
        snprintf(folder, sizeof(folder), "%s", path);
        nelements = mesh->nelements;
    }

    create_directory(folder);

    if (!rank) {
        printf("Writing mesh in %s\n", folder);
    }

    // if (size == 1) {
    for (int d = 0; d < mesh->spatial_dim; ++d) {
        snprintf(output_path, sizeof(output_path), "%s/%c.raw", folder, str_xyz[d]);
        array_write(comm, output_path, SFEM_MPI_GEOM_T, mesh->points[d], mesh->nnodes, mesh->nnodes);
    }

    const int nxe = elem_num_nodes(mesh->element_type);
    for (int d = 0; d < nxe; ++d) {
        snprintf(output_path, sizeof(output_path), "%s/i%d.raw", folder, d);
        array_write(comm, output_path, SFEM_MPI_IDX_T, mesh->elements[d], nelements, nelements);
    }

    if (mesh->node_mapping) {
        snprintf(output_path, sizeof(output_path), "%s/node_mapping.raw", folder);
        array_write(comm, output_path, SFEM_MPI_IDX_T, mesh->node_mapping, mesh->nnodes, mesh->nnodes);
    }

    if (mesh->element_mapping) {
        snprintf(output_path, sizeof(output_path), "%s/element_mapping.raw", folder);
        array_write(comm, output_path, SFEM_MPI_IDX_T, mesh->element_mapping, nelements, nelements);
    }

    if (mesh->node_owner) {
        snprintf(output_path, sizeof(output_path), "%s/node_owner.raw", folder);
        array_write(comm, output_path, SFEM_MPI_IDX_T, mesh->node_owner, mesh->nnodes, mesh->nnodes);
    }

    snprintf(output_path, sizeof(output_path), "%s/meta.yaml", path);
    FILE *meta_file = fopen(output_path, "w");
    if (!meta_file) {
        SFEM_ERROR("mesh_write: Unable to write file %s\n", output_path);
    }

    fprintf(meta_file, "# SFEM mesh meta file (generated by sfem_mesh_write.c)\n");
    fprintf(meta_file, "spatial_dimension: %d\n", mesh->spatial_dim);
    fprintf(meta_file, "elem_num_nodes: %d\n", nxe);
    fprintf(meta_file, "element_type: %s\n", type_to_string(mesh->element_type));
    fprintf(meta_file, "elements:\n");

    for (int d = 0; d < nxe; ++d) {
        fprintf(meta_file, "- i%d: i%d.raw\n", d, d);
    }

    fprintf(meta_file, "points:\n");
    for (int d = 0; d < mesh->spatial_dim; d++) {
        fprintf(meta_file, "- %c: %c.raw\n", str_xyz[d], str_xyz[d]);
    }

    fprintf(meta_file, "rpath: true\n");
    fclose(meta_file);

    return SFEM_SUCCESS;
    // } else {
    //     // TODO
    //     assert(0);
    //     return 1;
    // }
}

int mesh_write_nodal_field(MPI_Comm           comm,
                           const ptrdiff_t    n_owned_nodes,
                           const idx_t *const node_mapping,
                           const char        *path,
                           MPI_Datatype       data_type,
                           const void *const  data) {
    // get MPI rank
    int mpi_rank;
    MPI_Comm_rank(comm, &mpi_rank);

    count_t n_global_nodes = n_owned_nodes;
    MPI_CATCH_ERROR(MPI_Allreduce(MPI_IN_PLACE, &n_global_nodes, 1, SFEM_MPI_COUNT_T, MPI_SUM, comm));

    if (!node_mapping) {
#ifndef NDEBUG
        int size;
        MPI_Comm_size(comm, &size);
        assert(size == 1);
#endif

        if (mpi_rank == 0) printf("%s:%d: Writing using array_write\n", __FILE__, __LINE__);

        return array_write(comm, path, data_type, data, n_owned_nodes, n_global_nodes);

    } else {
        if (mpi_rank == 0) printf("%s:%d: Writing using write_mapped_field\n", __FILE__, __LINE__);

        return write_mapped_field(comm, path, n_owned_nodes, n_global_nodes, node_mapping, data_type, data);
    }
}

int write_mapped_field(MPI_Comm           comm,
                       const char        *output_path,
                       const ptrdiff_t    n_local,
                       const ptrdiff_t    n_global,
                       const idx_t *const mapping,
                       MPI_Datatype       data_type,
                       const void *const  data_in) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    const uint8_t *const data = (const uint8_t *const)data_in;

    int type_size;
    MPI_CATCH_ERROR(MPI_Type_size(data_type, &type_size));

    const ptrdiff_t local_output_size_no_remainder = n_global / size;
    const ptrdiff_t begin                          = (n_global / size) * rank;

    ptrdiff_t local_output_size = local_output_size_no_remainder;
    if (rank == size - 1) {
        local_output_size = n_global - begin;
    }

    int *send_count = (int *)malloc((size) * sizeof(int));
    memset(send_count, 0, (size) * sizeof(int));

    for (ptrdiff_t i = 0; i < n_local; ++i) {
        const idx_t idx       = mapping[i];
        int         dest_rank = MIN(size - 1, idx / local_output_size_no_remainder);
        send_count[dest_rank]++;
    }

    int *recv_count = (int *)malloc((size) * sizeof(int));
    MPI_CATCH_ERROR(MPI_Alltoall(send_count, 1, SFEM_MPI_IDX_T, recv_count, 1, SFEM_MPI_IDX_T, comm));

    int     *send_displs  = (int *)malloc(size * sizeof(int));
    int     *recv_displs  = (int *)malloc(size * sizeof(int));
    count_t *book_keeping = (count_t *)calloc(size, sizeof(count_t));

    send_displs[0] = 0;
    recv_displs[0] = 0;

    // Create data displacements for sending
    for (int i = 0; i < size - 1; ++i) {
        send_displs[i + 1] = send_displs[i] + send_count[i];
    }

    // Create data displacements for receiving
    for (int i = 0; i < size - 1; ++i) {
        recv_displs[i + 1] = recv_displs[i] + recv_count[i];
    }

    const ptrdiff_t total_recv = recv_displs[size - 1] + recv_count[size - 1];

    idx_t *send_list = (idx_t *)malloc(n_local * sizeof(idx_t));

    ptrdiff_t n_buff                      = MAX(n_local, local_output_size);
    uint8_t  *send_data_and_final_storage = (uint8_t *)malloc(n_buff * type_size);

    // Pack data and indices
    for (ptrdiff_t i = 0; i < n_local; ++i) {
        const idx_t idx       = mapping[i];
        int         dest_rank = MIN(size - 1, idx / local_output_size_no_remainder);
        assert(dest_rank < size);

        // Put index and data into buffers
        const ptrdiff_t offset = send_displs[dest_rank] + book_keeping[dest_rank];
        send_list[offset]      = idx;
        memcpy((void *)&send_data_and_final_storage[offset * type_size], (void *)&data[i * type_size], type_size);

        book_keeping[dest_rank]++;
    }

    idx_t   *recv_list = (idx_t *)malloc(local_output_size * sizeof(idx_t));
    uint8_t *recv_data = (uint8_t *)malloc(local_output_size * type_size);

    ///////////////////////////////////
    // Send indices
    ///////////////////////////////////

    MPI_CATCH_ERROR(MPI_Alltoallv(
            send_list, send_count, send_displs, SFEM_MPI_IDX_T, recv_list, recv_count, recv_displs, SFEM_MPI_IDX_T, comm));

    ///////////////////////////////////
    // Send data
    ///////////////////////////////////

    MPI_CATCH_ERROR(MPI_Alltoallv(send_data_and_final_storage,
                                  send_count,
                                  send_displs,
                                  data_type,
                                  recv_data,
                                  recv_count,
                                  recv_displs,
                                  data_type,
                                  comm));

    if (0) {
        for (int r = 0; r < size; r++) {
            MPI_Barrier(comm);

            if (r == rank) {
                printf("[%d]\n", rank);
                printf("\nsend_count\n");
                for (int i = 0; i < size; i++) {
                    printf("%d ", send_count[i]);
                }

                printf("\nsend_displs\n");
                for (int i = 0; i < size; i++) {
                    printf("%d ", send_displs[i]);
                }

                printf("\nrecv_count\n");
                for (int i = 0; i < size; i++) {
                    printf("%d ", recv_count[i]);
                }

                printf("\nrecv_displs\n");
                for (int i = 0; i < size; i++) {
                    printf("%d ", recv_displs[i]);
                }

                printf("\n");

                idx_t min_idx = mapping[0];
                idx_t max_idx = mapping[0];
                for (ptrdiff_t i = 0; i < n_local; ++i) {
                    const idx_t idx = mapping[i];
                    min_idx         = MIN(min_idx, idx);
                    max_idx         = MAX(max_idx, idx);
                }

                printf("[%d, %d]\n", min_idx, max_idx);
                printf("%ld == %ld\n", total_recv, local_output_size);

                for (ptrdiff_t recv_rank = 0; recv_rank < size; ++recv_rank) {
                    if (recv_rank != rank) {
                        for (int i = 0; i < send_count[recv_rank]; i++) {
                            printf("%d ", (int)send_list[send_displs[recv_rank] + i]);
                        }
                    }
                }

                printf("\n");

                for (ptrdiff_t i = 0; i < local_output_size; ++i) {
                    ptrdiff_t dest = recv_list[i] - begin;

                    if (dest < 0 || dest >= local_output_size) {
                        printf("%d not in [%ld, %ld)\n", recv_list[i], begin, begin + local_output_size);
                    }
                }

                fflush(stdout);
            }

            MPI_Barrier(comm);
        }
    }

    ///////////////////////////////////
    // Unpack indexed data
    ///////////////////////////////////

    for (ptrdiff_t i = 0; i < local_output_size; ++i) {
        ptrdiff_t dest = recv_list[i] - begin;
        assert(dest >= 0);
        assert(dest < local_output_size);
        memcpy((void *)&send_data_and_final_storage[dest * type_size], (void *)&recv_data[i * type_size], type_size);
    }

    array_write(comm, output_path, data_type, (void *)send_data_and_final_storage, local_output_size, n_global);

    ///////////////////////////////////
    // Clean-up
    ///////////////////////////////////
    free(send_count);
    free(send_displs);
    free(recv_count);
    free(recv_displs);
    free(book_keeping);
    free(send_list);
    free(recv_list);
    free(recv_data);
    free(send_data_and_final_storage);
    return 0;
}

int mesh_write_nodal_field_deprecated(const mesh_t *mesh, const char *path, MPI_Datatype data_type, const void *const data) {
    return mesh_write_nodal_field(mesh->comm, mesh->n_owned_nodes, mesh->node_mapping, path, data_type, data);
}
