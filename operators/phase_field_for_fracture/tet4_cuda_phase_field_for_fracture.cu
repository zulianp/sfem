#include "sfem_base.h"

#include "sfem_cuda_base.h"

#include "FE3D_phase_field_for_fracture_kernels.h"
#include "Tet4_impl.cu"

#include "cuda_crs.h"

#include <algorithm>
#include <vector>


#define MIN(a, b) ((b) < (a) ? (b) : (a))

static const int block_size = 128;
static const ptrdiff_t n_max_blocks = 500;
static const int n_vars = (fe_spatial_dim + 1);
static const int n_vars_squared = n_vars * n_vars;
static const int n_quad_points = 27;

class CudaWorkspace {
public:
    // Mesh
    idx_t *hh_elems[fe_n_nodes];
    idx_t **hd_elems[fe_n_nodes];
    idx_t **d_elems{nullptr};

    // Buffers
    geom_t *he_xyz{nullptr};
    geom_t *de_xyz{nullptr};
    real_t *d_jacobian_inverse;

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


    real_t *d_qx, *d_qy, *d_qz, *d_qw;

    cudaStream_t upload, compute, download;
    ptrdiff_t nbatch{0}, n_blocks{0};

    ~CudaWorkspace() {
        SFEM_NVTX_SCOPE("~CudaWorkspace");

        destroy_fe();
        destroy_streams();
        destroy_crs_view();
    }

    void create_sol_buffs() {
        hde_u = (real_t **)malloc(fe_spatial_dim * sizeof(real_t *));
        for (int d = 0; d < fe_spatial_dim; d++) {
            SFEM_CUDA_CHECK(cudaMalloc(&hde_u[d], fe_n_nodes * nbatch * sizeof(real_t)));
        }

        SFEM_CUDA_CHECK(cudaMalloc(&de_c, fe_n_nodes * nbatch * sizeof(real_t)));
        cudaMemcpy(de_c, hd_elems, fe_n_nodes * sizeof(idx_t *), cudaMemcpyHostToDevice);
    }

    void destroy_sol_buffs() {
        for (int d = 0; d < fe_spatial_dim; d++) {
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
            cudaMalloc(&d_jacobian_inverse, fe_manifold_dim * fe_spatial_dim * nbatch * sizeof(real_t)));
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
            SFEM_CUDA_CHECK(cudaFree(d_jacobian_inverse));

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

SFEM_DEVICE_KERNEL void Tet4_phase_field_for_fracture_assemble_hessian_kernel(
    const ptrdiff_t nelements,
    real_t *const SFEM_RESTRICT jacobian_inverse,
    const real_t *const SFEM_RESTRICT qx,
    const real_t *const SFEM_RESTRICT qy,
    const real_t *const SFEM_RESTRICT qz,
    const real_t *const SFEM_RESTRICT qw,
    const real_t mu,
    const real_t lambda,
    const real_t Gc,
    const real_t ls,
    const real_t *const SFEM_RESTRICT c,
    const real_t *const SFEM_RESTRICT ux,
    const real_t *const SFEM_RESTRICT uy,
    const real_t *const SFEM_RESTRICT uz,
    real_t *const SFEM_RESTRICT de_matrix) 
{
    real_t mat[n_vars_squared];

#ifdef __NVCC__
    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements;
         e += blockDim.x * gridDim.x)
#else
    for (ptrdiff_t e = 0; e < nelements; e++)
#endif
    {
        const real_t det_jac = 1. / Tet4_mk_det_3(nelements, &jacobian_inverse[e]);
        real_t fun[fe_n_nodes];
        real_t grad[fe_spatial_dim][fe_n_nodes];

        real_t test_grad[fe_spatial_dim], trial_grad[fe_spatial_dim];

        // TODO
        real_t s_grad_phase[fe_spatial_dim];
        real_t s_grad_disp[fe_spatial_dim * fe_spatial_dim];

        // Constant gradient
        Tet4_mk_partial_x(0, 0, 0, nelements, &jacobian_inverse[e], 1, grad[0]);
        Tet4_mk_partial_y(0, 0, 0, nelements, &jacobian_inverse[e], 1, grad[1]);
        Tet4_mk_partial_z(0, 0, 0, nelements, &jacobian_inverse[e], 1, grad[2]);

        for (int i = 0; i < fe_n_nodes; i++) {
            for (int j = 0; j < fe_n_nodes; j++) {
                for (int k = 0; k < n_quad_points; k++) {
                    Tet4_mk_fun(qx[k], qy[k], qz[k], 1, fun);

                    real_t s_phase = 0;

#pragma unroll(fe_spatial_dim)
                    for (int d = 0; d < fe_spatial_dim; d++) {
                        test_grad[d] = grad[d][i];
                        trial_grad[d] = grad[d][j];
                    }

                    FE3D_phase_field_for_fracture_hessian(mu,
                                                          lambda,
                                                          Gc,
                                                          ls,
                                                          fe_reference_measure * qw[k],
                                                          det_jac,
                                                          fun[i],
                                                          test_grad,
                                                          fun[j],
                                                          trial_grad,
                                                          s_phase,
                                                          s_grad_phase,
                                                          s_grad_disp,
                                                          mat);
                }

                // point mat to element mat
                const static int nn = fe_n_nodes * fe_n_nodes;
                for(int d1 = 0; d1 < n_vars; d1++) {
#pragma unroll(n_vars)
                    for(int d2 = 0; d2 < n_vars; d2++) {
                        ptrdiff_t idx = (d1 * n_vars + d2) * nn + i * fe_n_nodes + j;
                        de_matrix[idx * nelements] = mat[d1 * n_vars + d2];
                    }
                }
            }
        }
    }
}

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
    for (ptrdiff_t element_offset = 0; element_offset < nelements; element_offset += w.nbatch) {
        ptrdiff_t n = MIN(w.nbatch, nelements - element_offset);

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
            Tet4_block_matrix_local_to_global_kernel<n_vars, n_vars><<<w.n_blocks, block_size, 0, w.compute>>>(
                last_n, w.d_elems, last_n, w.de_matrix, w.d_rowptr, w.d_colidx, w.d_values);

            cudaEventRecord(computes[COMPUTE_JACOBIAN], w.compute);

            SFEM_DEBUG_SYNCHRONIZE();
        }

        /////////////////////////////////////////////////////////
        // XYZ HtoD (stream 0)
        /////////////////////////////////////////////////////////

        SFEM_CUDA_CHECK(cudaMemcpyAsync(w.de_xyz,
                                        w.he_xyz,
                                        fe_spatial_dim * fe_subparam_n_nodes * n * sizeof(geom_t),
                                        cudaMemcpyHostToDevice,
                                        w.upload));
        cudaEventRecord(uploads[UPLOAD_POINTS], w.upload);

        SFEM_DEBUG_SYNCHRONIZE();
        /////////////////////////////////////////////////////////
        // Jacobian computations (stream 1)
        /////////////////////////////////////////////////////////

        // Make sure we have the new XYZ coordinates
        cudaStreamWaitEvent(w.upload, uploads[UPLOAD_POINTS], 0);
        Tet4_jacobian_inverse_kernel<<<w.n_blocks, block_size, 0, w.compute>>>(n, w.de_xyz, w.d_jacobian_inverse);

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
                memcpy(w.hh_elems[e_node], &elems[e_node][element_offset], n * sizeof(idx_t));
            }
        }

        // Make sure local to global has ended
        if (last_n) {
            cudaStreamWaitEvent(w.compute, computes[COMPUTE_LOCAL_TO_GLOBAL], 0);
        }

        {
            SFEM_NVTX_SCOPE("copy-elements-to-device");
            for (int e_node = 0; e_node < fe_subparam_n_nodes; e_node++) {
                SFEM_CUDA_CHECK(cudaMemcpyAsync(w.hd_elems[e_node],
                                                w.hh_elems[e_node],
                                                n * sizeof(idx_t),
                                                cudaMemcpyHostToDevice,
                                                w.upload));
            }

            cudaEventRecord(uploads[UPLOAD_ELEMENTS], w.upload);
        }

        SFEM_DEBUG_SYNCHRONIZE();
        /////////////////////////////////////////////////////////
        // Assemble elemental matrices (stream 1)
        /////////////////////////////////////////////////////////
        {
            Tet4_phase_field_for_fracture_assemble_hessian_kernel<<<w.n_blocks,
                                                                    block_size,
                                                                    0,
                                                                    w.compute>>>(
                n, 
                w.d_jacobian_inverse, 
                w.d_qx, 
                w.d_qy, 
                w.d_qz, 
                w.d_qw,
                mu, lambda, Gc, ls, 
                w.de_c, 
                w.de_u[0],
                w.de_u[1],
                w.de_u[2],
                w.de_matrix);
            cudaEventRecord(computes[COMPUTE_ELEMENTAL_MATRICES], w.compute);
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
        Tet4_block_matrix_local_to_global_kernel<n_vars,n_vars><<<w.n_blocks, block_size, 0, w.compute>>>(
            last_n, w.d_elems, last_n, w.de_matrix, w.d_rowptr, w.d_colidx, w.d_values);

        SFEM_DEBUG_SYNCHRONIZE();
        cudaStreamSynchronize(w.compute);
    }

    {
        SFEM_NVTX_SCOPE("downloads");
        SFEM_CUDA_CHECK(
            cudaMemcpy(values, w.d_values, rowptr[nnodes] * sizeof(real_t), cudaMemcpyDeviceToHost));
    }

    for (auto &e : uploads) {
        cudaEventDestroy(e);
    }

    for (auto &e : downloads) {
        cudaEventDestroy(e);
    }

    for (auto &e : computes) {
        cudaEventDestroy(e);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
