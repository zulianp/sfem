#include "sfem_base.h"

#include "sfem_cuda_base.h"

#include "FE3D_phase_field_for_fracture_kernels.h"
#include "Tet4_impl.cu"

static const int block_size = 128;
static const int n_max_blocks = 500;
static const int n_vars = (fe_spatial_dim + 1);
static const int n_vars_squared = n_vars * n_vars;

class CudaWorkspace {
public:
    // Mesh
    idx_t *hh_elems[fe_n_nodes];
    idx_t **hd_elems[fe_n_nodes];
    idx_t **d_elems{nullptr};

    // Buffers
    geom_t *he_xyz{nullptr};
    geom_t *de_xyz{nullptr};

    real_t *de_matrix{nullptr};
    real_t *de_vector{nullptr};

    real_t **hde_u{nullptr};
    real_t **de_u{nullptr};
    real_t *de_c{nullptr};

    // CRS-matrix
    count_t *d_rowptr{nullptr};
    idx_t *d_colidx{nullptr};

    // (c, u) x (c, u)
    real_t **hd_values{nullptr};
    real_t **d_values{nullptr};

    cudaStream_t upload, compute, download;
    ptrdiff_t nbatch{0}, n_blocks{0};

    ~CudaWorkspace() {
        SFEM_NVTX_SCOPE("~CudaWorkspace");

        destroy_fe();
        destroy_streams();
        destroy_crs_view();
    }

    void create_sol_buffs()
    {   
        hde_u = (real_t **)malloc(fe_spatial_dim * sizeof(real_t *));
        for(int d = 0; d < fe_spatial_dim; d++) {
            SFEM_CUDA_CHECK(cudaMalloc(&hde_u[d], fe_n_nodes * nbatch * sizeof(real_t)));
        }

        SFEM_CUDA_CHECK(cudaMalloc(&de_c, fe_n_nodes * nbatch * sizeof(real_t)));
        cudaMemcpy(de_c, hd_elems, fe_n_nodes * sizeof(idx_t *), cudaMemcpyHostToDevice);
    }

    void destroy_sol_buffs()
    {
        for(int d =0; d < fe_spatial_dim; d++) {
            SFEM_CUDA_CHECK(cudaFree(hde_u[d]));
        }

        free(hde_u);
        SFEM_CUDA_CHECK(cudaFree(de_u));
    }

    void create_streams() {
        SFEM_NVTX_SCOPE("create_streams");

        cudaStreamCreate(&upload);
        cudaStreamCreate(&compute);
        cudaStreamCreate(&download);
    }

    void destroy_streams() {
        SFEM_NVTX_SCOPE("destroy_streams");

        cudaStreamDestroy(upload);
        cudaStreamDestroy(compute);
        cudaStreamDestroy(download);
    }

    void create_fe(ptrdiff_t nelements) {
        SFEM_NVTX_SCOPE("create_fe");

        nbatch = std::min(block_size * n_max_blocks, nelements);
        n_blocks = std::max(ptrdiff_t(1), (nbatch + block_size - 1) / block_size);

        SFEM_CUDA_CHECK(
            cudaMallocHost(&he_xyz, fe_spatial_dim * fe_n_nodes * nbatch * sizeof(geom_t)));
        SFEM_CUDA_CHECK(cudaMalloc(&de_xyz, fe_spatial_dim * fe_n_nodes * nbatch * sizeof(geom_t)));
        SFEM_CUDA_CHECK(
            cudaMalloc(&d_jacobian, fe_manifold_dim * fe_spatial_dim * nbatch * sizeof(real_t)));
        SFEM_CUDA_CHECK(cudaMalloc(&de_matrix, fe_n_nodes * fe_n_nodes * nbatch * sizeof(real_t)));

        for (int d = 0; d < fe_n_nodes; d++) {
            SFEM_CUDA_CHECK(cudaMallocHost(&hh_elems[d], nbatch * sizeof(idx_t)));
        }

        // Allocate space for indices
        for (int d = 0; d < fe_n_nodes; d++) {
            SFEM_CUDA_CHECK(cudaMalloc(&hd_elems[d], nbatch * sizeof(idx_t)));
        }

        SFEM_CUDA_CHECK(cudaMalloc(&d_elems, fe_n_nodes * sizeof(idx_t *)));
        cudaMemcpy(d_elems, hd_elems, fe_n_nodes * sizeof(idx_t *), cudaMemcpyHostToDevice);
    }

    void destroy_fe() {
        SFEM_NVTX_SCOPE("destroy_fe");

        {  // Free resources on CPU
            cudaFreeHost(he_xyz);

            for (int d = 0; d < fe_n_nodes; d++) {
                SFEM_CUDA_CHECK(cudaFreeHost(hh_elems[d]));
            }
        }

        {  // Free resources on GPU
            SFEM_CUDA_CHECK(cudaFree(de_xyz));
            SFEM_CUDA_CHECK(cudaFree(de_matrix));
            SFEM_CUDA_CHECK(cudaFree(d_fff));

            for (int d = 0; d < fe_n_nodes; d++) {
                SFEM_CUDA_CHECK(cudaFree(hd_elems[d]));
            }

            SFEM_CUDA_CHECK(cudaFree(d_elems));
        }
    }

    void create_crs_view(const ptrdiff_t nnodes,
                         const count_t *const SFEM_RESTRICT rowptr,
                         const idx_t *const SFEM_RESTRICT colidx) {
        SFEM_NVTX_SCOPE("create_crs_view");

        crs_graph_device_create(nnodes, rowptr[nnodes], &d_rowptr, &d_colidx);
        crs_graph_host_to_device(nnodes, rowptr[nnodes], rowptr, colidx, d_rowptr, d_colidx);

        hd_values = (real_t **)malloc(n_vars_squared * sizeof(real_t *));
        SFEM_CUDA_CHECK(cudaMalloc(&d_values, n_vars_squared * sizeof(real_t *)));

        for (int d = 0; d < n_vars_squared; d++) {
            SFEM_CUDA_CHECK(cudaMallocHost(&hd_values[d], rowptr[nnodes] * sizeof(real_t)));
        }

        SFEM_CUDA_CHECK(cudaMemcpy(
            d_values, hd_values, rowptr[nnodes] * sizeof(idx_t *), cudaMemcpyHostToDevice));
    }

    void destroy_crs_view() {
        SFEM_NVTX_SCOPE("destroy_crs_view");
        crs_graph_device_free(d_rowptr, d_colidx);

        for (int d = 0; d < n_vars_squared; d++) {
            SFEM_CUDA_CHECK(cudaFree(hd_values[d]));
        }

        free(hd_values);
        SFEM_CUDA_CHECK(cudaFree(d_values));
    }
};

extern "C" void phase_field_for_fracture_assemble_hessian(const ptrdiff_t nelements,
                                                          const ptrdiff_t nnodes,
                                                          idx_t **const SFEM_RESTRICT elems,
                                                          geom_t **const SFEM_RESTRICT xyz,
                                                          const real_t mu,
                                                          const real_t lambda,
                                                          const real_t Gc,
                                                          const real_t ls,
                                                          real_t *const SFEM_RESTRICT c,
                                                          real_t *const SFEM_RESTRICT u,
                                                          const count_t *const SFEM_RESTRICT rowptr,
                                                          const idx_t *const SFEM_RESTRICT colidx,
                                                          real_t **const SFEM_RESTRICT values) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    CudaWorkspace w;
    w.create_fe(nelements);
    w.create_streams();

    static const int UPLOAD_POINTS = 0;
    static const int UPLOAD_ELEMENTS = 1;
    std::vector<cudaEvent_t> uploads(2);

    static const int DOWNLOAD_MATRIX = 0;
    std::vector<cudaEvent_t> downloads(1);

    static const int COMPUTE_JACOBIAN = 0;
    static const int COMPUTE_ELEMENTAL_MATRICES = 1;
    static const int COMPUTE_LOCAL_TO_GLOBAL = 2;
    std::vector<cudaEvent_t> computes(3);

    for (auto &e : uploads) {
        cudaEventCreate(&e);
    }

    for (auto &e : downloads) {
        cudaEventCreate(&e);
    }

    for (auto &e : computes) {
        cudaEventCreate(&e);
    }

    ptrdiff_t last_n = 0;
    ptrdiff_t last_element_offset = 0;
    for (ptrdiff_t element_offset = 0; element_offset < nelements; element_offset += nbatch) {
        ptrdiff_t n = MIN(nbatch, nelements - element_offset);

        /////////////////////////////////////////////////////////
        // Packing (stream 0)
        /////////////////////////////////////////////////////////

        if (last_n) {
            cudaStreamWaitEvent(w.compute, computes[COMPUTE_JACOBIAN], 0);
            cudaStreamWaitEvent(w.upload, uploads[UPLOAD_POINTS], 0);
        }

        Tet4_host_pack_elements(n, element_offset, elems, xyz, w.he_xyz);

        /////////////////////////////////////////////////////////
        // Local to global (stream 3)
        /////////////////////////////////////////////////////////

        if (last_n) {
            // Make sure we have the elemental matrices and dof indices
            cudaStreamWaitEvent(w.upload, uploads[UPLOAD_ELEMENTS], 0);

            // Do this here to let the main kernel overlap with the packing
            Tet4_matrix_local_to_global_kernel<<<n_blocks, block_size, 0, w.compute>>>(
                last_n, d_elems, de_matrix, d_rowptr, d_colidx, d_values);

            cudaEventRecord(w.compute, computes[COMPUTE_JACOBIAN]);

            SFEM_DEBUG_SYNCHRONIZE();
        }

        /////////////////////////////////////////////////////////
        // XYZ HtoD (stream 0)
        /////////////////////////////////////////////////////////

        SFEM_CUDA_CHECK(cudaMemcpyAsync(
            de_xyz, he_xyz, fe_spatial_dim * fe_subparam_n_nodes * n * sizeof(geom_t), cudaMemcpyHostToDevice, w.upload));
        cudaEventRecord(w.upload, uploads[UPLOAD_POINTS]);

        SFEM_DEBUG_SYNCHRONIZE();
        /////////////////////////////////////////////////////////
        // Jacobian computations (stream 1)
        /////////////////////////////////////////////////////////

        // Make sure we have the new XYZ coordinates
        cudaStreamWaitEvent(w.upload, uploads[UPLOAD_POINTS], 0);
        Tet4_jacobian_inverse<<<n_blocks, block_size, 0, stream[1]>>>(n, de_xyz, d_fff);

        SFEM_DEBUG_SYNCHRONIZE();
        /////////////////////////////////////////////////////////
        // DOF indices HtoD (stream 2)
        /////////////////////////////////////////////////////////

        // Ensure that previous HtoD is completed
        if (last_n) {
            cudaStreamWaitEvent(w.upload, uploads[UPLOAD_ELEMENTS], 0);
        }

        {
            SFEM_NVTX_SCOPE("copy-elements-to-pinned-memory");
            //  Copy elements to host-pinned memory
            for (int e_node = 0; e_node < fe_n_nodes; e_node++) {
                memcpy(hh_elems[e_node], &elems[e_node][element_offset], n * sizeof(idx_t));
            }
        }

        // Make sure local to global has ended
        if (last_n) {
            cudaStreamWaitEvent(w.compute, computes[COMPUTE_LOCAL_TO_GLOBAL], 0);
        }

        {
            SFEM_NVTX_SCOPE("copy-elements-to-device");
            for (int e_node = 0; e_node < fe_subparam_n_nodes; e_node++) {
                SFEM_CUDA_CHECK(cudaMemcpyAsync(hd_elems[e_node],
                                                hh_elems[e_node],
                                                n * sizeof(idx_t),
                                                cudaMemcpyHostToDevice,
                                                w.upload));
            }

            cudaEventRecord(w.upload, uploads[UPLOAD_ELEMENTS]);
        }

        SFEM_DEBUG_SYNCHRONIZE();
        /////////////////////////////////////////////////////////
        // Assemble elemental matrices (stream 1)
        /////////////////////////////////////////////////////////
        {
            Tet4_phase_field_for_fracture_assemble_hessian_kernel<<<n_blocks, block_size, 0, w.compute>>>(
                n, d_jacobian, mu, lambda, Gc, ls, w.de_u, w.de_c, de_matrix);
            cudaEventRecord(w.compute, computes[COMPUTE_ELEMENTAL_MATRICES]);
        }

        SFEM_DEBUG_SYNCHRONIZE();
        /////////////////////////////////////////////////////////

        last_n = n;
        last_element_offset = element_offset;
    }

    /////////////////////////////////////////////////////////
    // Local to global (stream 3)
    /////////////////////////////////////////////////////////

    if (last_n) {
        // Make sure we have the elemental matrices and dof indices
        cudaStreamWaitEvent(w.upload, uploads[UPLOAD_ELEMENTS], 0);

        // Do this here to let the main kernel overlap with the packing
        Tet4_matrix_local_to_global_kernel<<<n_blocks, block_size, 0, w.compute>>>(
            last_n, w.d_elems, w.de_matrix, w.d_rowptr, w.d_colidx, w.d_values);

        SFEM_DEBUG_SYNCHRONIZE();
        cudaStreamSynchronize(w.compute);
    }

    {
        SFEM_NVTX_SCOPE("downloads");
        SFEM_CUDA_CHECK(
            cudaMemcpy(values, d_values, rowptr[nnodes] * sizeof(real_t), cudaMemcpyDeviceToHost));
    }

    for (auto &e : uploads) {
        cudaEventDestroy(e);
    }

    for (auto &e : downloads) {
        cudaEventDestroy(e);
    }

    for (auto &e : compute) {
        cudaEventDestroy(e);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
