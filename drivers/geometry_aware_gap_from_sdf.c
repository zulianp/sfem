#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "matrixio_array.h"
#include "matrixio_ndarray.h"

#include "mesh_aura.h"
#include "read_mesh.h"
#include "sfem_mesh_write.h"
#include "sfem_resample_gap.h"

#include "mass.h"

#include "crs_graph.h"
#include "extract_sharp_features.h"

#include "mesh_utils.h"

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (argc != 13) {
        fprintf(stderr,
                "usage: %s <mesh_folder> <nx> <ny> <nz> <ox> <oy> <oz> <dx> <dy> <dz> "
                "<data.float32.raw> <output_folder>\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    geom_t SFEM_ANGLE_THRESHOLD = 0.15;
    SFEM_READ_ENV(SFEM_ANGLE_THRESHOLD, atof);

    int SFEM_SUPERIMPOSE = 0;
    SFEM_READ_ENV(SFEM_SUPERIMPOSE, atoi);

    double tick = MPI_Wtime();

    const char* folder = argv[1];
    ptrdiff_t nglobal[3] = {atol(argv[2]), atol(argv[3]), atol(argv[4])};
    geom_t origin[3] = {atof(argv[5]), atof(argv[6]), atof(argv[7])};
    geom_t delta[3] = {atof(argv[8]), atof(argv[9]), atof(argv[10])};
    const char* data_path = argv[11];
    const char* output_folder = argv[12];

    struct stat st = {0};
    if (stat(output_folder, &st) == -1) {
        mkdir(output_folder, 0700);
    }

    mesh_t mesh;
    if (mesh_read(comm, folder, &mesh)) {
        return EXIT_FAILURE;
    }

    ptrdiff_t n = nglobal[0] * nglobal[1] * nglobal[2];
    geom_t* sdf = 0;
    ptrdiff_t nlocal[3];

    {
        double ndarray_read_tick = MPI_Wtime();

        if (ndarray_create_from_file(
                comm, data_path, SFEM_MPI_GEOM_T, 3, (void**)&sdf, nlocal, nglobal)) {
            return EXIT_FAILURE;
        }

        double ndarray_read_tock = MPI_Wtime();

        if (!rank) {
            printf("[%d] ndarray_create_from_file %g (seconds)\n",
                   rank,
                   ndarray_read_tock - ndarray_read_tick);
        }
    }

    // X is contiguous
    ptrdiff_t stride[3] = {1, nlocal[0], nlocal[0] * nlocal[1]};

    if (size > 1) {
        geom_t* psdf;
        sdf_view(comm,
                 mesh.nnodes,
                 mesh.points[2],
                 nlocal,
                 nglobal,
                 stride,
                 origin,
                 delta,
                 sdf,
                 &psdf,
                 &nlocal[2],
                 &origin[2]);

        free(sdf);
        sdf = psdf;
    }

    // Edge graph for resampling on sharp edges
    ptrdiff_t n_sharp_edges = 0;
    idx_t* e0 = 0;
    idx_t* e1 = 0;

    // Indices for interpolating on corner nodes
    ptrdiff_t n_corners = 0;
    idx_t* corners = 0;

    // Selector for resampling on faces
    ptrdiff_t n_disconnected_elements = 0;
    element_idx_t* disconnected_elements = 0;

    {  // Extract sharp features!
        {
            count_t* rowptr = 0;
            idx_t* colidx = 0;
            build_crs_graph_for_elem_type(
                mesh.element_type, mesh.nelements, mesh.nnodes, mesh.elements, &rowptr, &colidx);

            extract_sharp_edges(mesh.element_type,
                                mesh.nelements,
                                mesh.nnodes,
                                mesh.elements,
                                mesh.points,
                                // CRS-graph (node to node)
                                rowptr,
                                colidx,
                                SFEM_ANGLE_THRESHOLD,
                                &n_sharp_edges,
                                &e0,
                                &e1);

            free(rowptr);
            free(colidx);
        }

        extract_disconnected_faces(mesh.element_type,
                                   mesh.nelements,
                                   mesh.nnodes,
                                   mesh.elements,
                                   n_sharp_edges,
                                   e0,
                                   e1,
                                   &n_disconnected_elements,
                                   &disconnected_elements);

        n_sharp_edges = extract_sharp_corners(
            mesh.nnodes, n_sharp_edges, e0, e1, &n_corners, &corners, !SFEM_SUPERIMPOSE);
    }

    // Quantities of interest
    real_t* g = calloc(mesh.nnodes, sizeof(real_t));
    real_t* xnormal = calloc(mesh.nnodes, sizeof(real_t));
    real_t* ynormal = calloc(mesh.nnodes, sizeof(real_t));
    real_t* znormal = calloc(mesh.nnodes, sizeof(real_t));
    real_t* mass_vector = calloc(mesh.nnodes, sizeof(real_t));
    {
        double resample_tick = MPI_Wtime();

        if (n_sharp_edges) {  // BEAM2 integral
            idx_t* edges[2] = {e0, e1};
            resample_gap_local(
                // Mesh
                BEAM2,
                n_sharp_edges,
                mesh.nnodes,
                edges,
                mesh.points,
                // SDF
                nlocal,
                stride,
                origin,
                delta,
                sdf,
                // Output
                g,
                xnormal,
                ynormal,
                znormal);

            assemble_lumped_mass(
                BEAM2, n_sharp_edges, mesh.nnodes, edges, mesh.points, mass_vector);
        }

        if (SFEM_SUPERIMPOSE) {
            resample_gap_local(
                // Mesh
                shell_type(mesh.element_type),
                mesh.nelements,
                mesh.nnodes,
                mesh.elements,
                mesh.points,
                // SDF
                nlocal,
                stride,
                origin,
                delta,
                sdf,
                // Output
                g,
                xnormal,
                ynormal,
                znormal);

            assemble_lumped_mass(shell_type(mesh.element_type),
                                 mesh.nelements,
                                 mesh.nnodes,
                                 mesh.elements,
                                 mesh.points,
                                 mass_vector);

        } else {
            if (n_disconnected_elements) {  // Faces
                int nxe = elem_num_nodes(mesh.element_type);
                idx_t** selected_elements = allocate_elements(nxe, n_disconnected_elements);
                select_elements(nxe,
                                n_disconnected_elements,
                                disconnected_elements,
                                mesh.elements,
                                selected_elements);

                resample_gap_local(
                    // Mesh
                    shell_type(mesh.element_type),
                    n_disconnected_elements,
                    mesh.nnodes,
                    selected_elements,
                    mesh.points,
                    // SDF
                    nlocal,
                    stride,
                    origin,
                    delta,
                    sdf,
                    // Output
                    g,
                    xnormal,
                    ynormal,
                    znormal);

                assemble_lumped_mass(shell_type(mesh.element_type),
                                     n_disconnected_elements,
                                     mesh.nnodes,
                                     selected_elements,
                                     mesh.points,
                                     mass_vector);

                free_elements(nxe, selected_elements);
            }
        }

        if (n_corners) {  // Nodes
            geom_t** corner_points = allocate_points(mesh.spatial_dim, n_corners);
            select_points(mesh.spatial_dim, n_corners, corners, mesh.points, corner_points);

            real_t* p_g = calloc(n_corners, sizeof(real_t));
            real_t* p_xnormal = calloc(n_corners, sizeof(real_t));
            real_t* p_ynormal = calloc(n_corners, sizeof(real_t));
            real_t* p_znormal = calloc(n_corners, sizeof(real_t));

            interpolate_gap(
                // Mesh
                n_corners,
                corner_points,
                // SDF
                nlocal,
                stride,
                origin,
                delta,
                sdf,
                // Output
                p_g,
                p_xnormal,
                p_ynormal,
                p_znormal);

            // Add to complete array
            for (ptrdiff_t i = 0; i < n_corners; i++) {
                g[corners[i]] += p_g[i];
                xnormal[corners[i]] += p_xnormal[i];
                ynormal[corners[i]] += p_ynormal[i];
                znormal[corners[i]] += p_znormal[i];
                mass_vector[corners[i]] += 1;
            }

            free_points(mesh.spatial_dim, corner_points);
            free(p_g);
            free(p_xnormal);
            free(p_ynormal);
            free(p_znormal);
        }

        if (size > 1) {
            // // exchange ghost nodes and add contribution
            send_recv_t slave_to_master;
            mesh_create_nodal_send_recv(&mesh, &slave_to_master);

            ptrdiff_t count = mesh_exchange_master_buffer_count(&slave_to_master);
            real_t* real_buffer = malloc(count * sizeof(real_t));

            exchange_add(&mesh, &slave_to_master, mass_vector, real_buffer);
            exchange_add(&mesh, &slave_to_master, g, real_buffer);
            exchange_add(&mesh, &slave_to_master, xnormal, real_buffer);
            exchange_add(&mesh, &slave_to_master, ynormal, real_buffer);
            exchange_add(&mesh, &slave_to_master, znormal, real_buffer);

            free(real_buffer);
        }

        // divide by the mass vector
        for (ptrdiff_t i = 0; i < mesh.n_owned_nodes; i++) {
            if (mass_vector[i] == 0) {
                fprintf(stderr,
                        "Found 0 mass at %ld, info (%ld, %ld)\n",
                        i,
                        mesh.n_owned_nodes,
                        mesh.nnodes);
            }

            assert(mass_vector[i] != 0);
            g[i] /= mass_vector[i];
        }

        for (ptrdiff_t i = 0; i < mesh.n_owned_nodes; i++) {
            const real_t xn = xnormal[i];
            const real_t yn = ynormal[i];
            const real_t zn = znormal[i];
            const real_t ln = sqrt(xn * xn + yn * yn + zn * zn);

            xnormal[i] /= ln;
            ynormal[i] /= ln;
            znormal[i] /= ln;
        }

        free(mass_vector);

        double resample_tock = MPI_Wtime();

        if (!rank) {
            printf("[%d] resample %g (seconds)\n", rank, resample_tock - resample_tick);
        }
    }

    // Write result to disk
    {
        double io_tick = MPI_Wtime();

        char path[1024 * 10];
        sprintf(path, "%s/gap.float64.raw", output_folder);
        mesh_write_nodal_field(&mesh, path, SFEM_MPI_REAL_T, g);

        sprintf(path, "%s/xnormal.float64.raw", output_folder);
        mesh_write_nodal_field(&mesh, path, SFEM_MPI_REAL_T, xnormal);

        sprintf(path, "%s/ynormal.float64.raw", output_folder);
        mesh_write_nodal_field(&mesh, path, SFEM_MPI_REAL_T, ynormal);

        sprintf(path, "%s/znormal.float64.raw", output_folder);
        mesh_write_nodal_field(&mesh, path, SFEM_MPI_REAL_T, znormal);

        if (0) {
            for (ptrdiff_t i = 0; i < mesh.nnodes; i++) {
                g[i] = rank;
            }

            sprintf(path, "%s/rank.float64.raw", output_folder);
            mesh_write_nodal_field(&mesh, path, SFEM_MPI_REAL_T, g);
        }

        double io_tock = MPI_Wtime();

        if (!rank) {
            printf("[%d] write %g (seconds)\n", rank, io_tock - io_tick);
        }
    }

    ptrdiff_t nelements = mesh.nelements;
    ptrdiff_t nnodes = mesh.nnodes;

    // Free resources
    {
        free(sdf);
        free(g);
        free(xnormal);
        free(ynormal);
        free(znormal);
        mesh_destroy(&mesh);
    }

    double tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld #grid (%ld x %ld x %ld)\n",
               (long)nelements,
               (long)nnodes,
               nglobal[0],
               nglobal[1],
               nglobal[2]);
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
