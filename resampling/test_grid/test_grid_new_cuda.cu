#include <cuda_profiler_api.h>
#include <stdio.h>
#include <vector>

int getSPcores(cudaDeviceProp devProp) {
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major) {
        case 2:  // Fermi
            if (devProp.minor == 1)
                cores = mp * 48;
            else
                cores = mp * 32;
            break;
        case 3:  // Kepler
            cores = mp * 192;
            break;
        case 5:  // Maxwell
            cores = mp * 128;
            break;
        case 6:  // Pascal
            if ((devProp.minor == 1) || (devProp.minor == 2))
                cores = mp * 128;
            else if (devProp.minor == 0)
                cores = mp * 64;
            else
                printf("Unknown device type\n");
            break;
        case 7:  // Volta and Turing
            if ((devProp.minor == 0) || (devProp.minor == 5))
                cores = mp * 64;
            else
                printf("Unknown device type\n");
            break;
        case 8:  // Ampere
            if (devProp.minor == 0)
                cores = mp * 64;
            else if (devProp.minor == 6)
                cores = mp * 128;
            else if (devProp.minor == 9)
                cores = mp * 128;  // ada lovelace
            else
                printf("Unknown device type\n");
            break;
        case 9:  // Hopper
            if (devProp.minor == 0)
                cores = mp * 128;
            else
                printf("Unknown device type\n");
            break;
        default:
            printf("Unknown device type\n");
            break;
    }
    return cores;
}

#include "test_grid_new.h"

/**
 * @brief Calculates the nearest grid coordinates (floor values) for a given point.
 *
 * This function calculates the nearest grid coordinates (floor values) for a given point (x, y)
 * based on the grid origin (x_zero, y_zero) and grid spacing (delta_x, delta_y). The calculated
 * grid coordinates are stored in the variables i and j.
 *
 * @param x_zero The x-coordinate of the grid origin.
 * @param y_zero The y-coordinate of the grid origin.
 * @param delta_x The spacing between grid points along the x-axis.
 * @param delta_y The spacing between grid points along the y-axis.
 * @param x The x-coordinate of the point for which nearest grid coordinates are to be calculated.
 * @param y The y-coordinate of the point for which nearest grid coordinates are to be calculated.
 * @param i Reference to the variable where the calculated x-coordinate of the nearest grid point
 * will be stored.
 * @param j Reference to the variable where the calculated y-coordinate of the nearest grid point
 * will be stored.
 */
__device__ void get_nearest_coordinates_floor_cu(const double x_zero,
                                                 const double y_zero,
                                                 const double delta_x,
                                                 const double delta_y,
                                                 const double x,
                                                 const double y,
                                                 int& i,
                                                 int& j) {
    //
    i = int(floor((x - x_zero) / delta_x));
    j = int(floor((y - y_zero) / delta_y));
}

/**
 * @brief Calculates the nearest grid coordinates (ceil values) for a given point.
 *
 * This function calculates the nearest grid coordinates (ceil values) for a given point (x, y)
 * based on the grid origin (x_zero, y_zero) and grid spacing (delta_x, delta_y). The calculated
 * grid coordinates are stored in the variables i and j.
 *
 * @param x_zero The x-coordinate of the grid origin.
 * @param y_zero The y-coordinate of the grid origin.
 * @param delta_x The spacing between grid points along the x-axis.
 * @param delta_y The spacing between grid points along the y-axis.
 * @param x The x-coordinate of the point for which nearest grid coordinates are to be calculated.
 * @param y The y-coordinate of the point for which nearest grid coordinates are to be calculated.
 * @param i Reference to the variable where the calculated x-coordinate of the nearest grid point
 * will be stored.
 * @param j Reference to the variable where the calculated y-coordinate of the nearest grid point
 * will be stored.
 */
__device__ void get_nearest_coordinates_ceil_cu(const double x_zero,
                                                const double y_zero,
                                                const double delta_x,
                                                const double delta_y,
                                                const double x,
                                                const double y,
                                                int& i,
                                                int& j) {
    //
    i = int(ceil((x - x_zero) / delta_x));
    j = int(ceil((y - y_zero) / delta_y));
}

/**
 * @brief Calculates the domain boundaries for a given domain number within a stripe.
 *
 * This function calculates the minimum and maximum values of the x and y coordinates
 * for a specific domain within a stripe. The domain number is used to determine the
 * position of the domain within the stripe.
 *
 * @param ds The domains_stripe struct containing information about the stripe.
 * @param domain_nr The number of the domain within the stripe.
 * @param x_min The minimum x coordinate of the domain.
 * @param y_min The minimum y coordinate of the domain.
 * @param x_max The maximum x coordinate of the domain.
 * @param y_max The maximum y coordinate of the domain.
 */
__device__ void get_domain_from_stripe_cu(const domains_stripe& ds,
                                          const size_t domain_nr,
                                          double& x_min,
                                          double& y_min,
                                          double& x_max,
                                          double& y_max) {
    //
    x_min = ds.x_min + domain_nr * ds.side_x;
    y_min = ds.y_min;

    x_max = x_min + ds.side_x;
    y_max = y_min + ds.side_y;
}

/**
 * @brief Performs the quadrature for a single stripe.
 *
 * @param Qs Pointer to the array where the calculated quadrature values will be stored.
 * @param gg The global_grid_type struct containing information about the global grid.
 * @param qr The quadrature_rule struct containing information about the quadrature rule.
 * @param qr_nodes_nr_  The number of nodes in the quadrature rule.
 * @param ds The domains_stripe struct containing information about the stripe.
 * @return void
 */
__device__ void perform_quadrature_global_stripe(double* Qs,                  //
                                                 const global_grid_type& gg,  //
                                                 const quadrature_rule& qr,   //
                                                 const size_t qr_nodes_nr_,   //
                                                 const domains_stripe& ds) {  //
                                                                              //
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t domain_nr = threadIdx.x;
    // printf("domain_nr: %lu, id %lu \n", domain_nr, id);

    // printf("domain_nr: %lu, id %lu \n", domain_nr, id);
    // return ;

    // Qs[id] = 1.0;

    // return ;

    // return;
    // const size_t stripe_nr = blockIdx.x;

    //    const size_t domain_nr = threadIdx.x;

    // if (domain_nr >= ds.nr_domains) {
    //     return;
    // }

    // double Ql = 0.0;

    // for (size_t i = 0; i < ds.nr_domains; ++i) {
    double x_d_min, y_d_min, x_d_max, y_d_max;
    get_domain_from_stripe_cu(ds, domain_nr, x_d_min, y_d_min, x_d_max, y_d_max);

    const double volume = (x_d_max - x_d_min) * (y_d_max - y_d_min);  // 3 * ds_nr_domains

    double Qs_i = 0.0;

    const size_t qr_size = qr_nodes_nr_;

    for (size_t q_i = 0; q_i < qr_size; ++q_i) {
        //
        // 2 * 3 * qr_size * ds.nr_domains
        const double x_Q = (qr.x_nodes_ptr_cu[q_i]) * (x_d_max - x_d_min) + x_d_min;
        const double y_Q = (qr.y_nodes_ptr_cu[q_i]) * (y_d_max - y_d_min) + y_d_min;

        // const double x_Q = q_i * (x_d_max - x_d_min) + x_d_min;
        //         const double y_Q = q_i * (y_d_max - y_d_min) + y_d_min;
        //     Qs_i = x_Q + y_Q;

        //     continue;

        // std::cout << "x_Q: " << x_Q << " y_Q: " << y_Q << std::endl;

        int i_local, j_local;

        // Qs_i += 2.0 * qr.weights_ptr_cu[q_i] * volume;
        // continue;

        get_nearest_coordinates_floor_cu(gg.x_zero,  //
                                         gg.y_zero,  //
                                         gg.delta,
                                         gg.delta,
                                         x_Q,
                                         y_Q,
                                         i_local,
                                         j_local);

        /////// data trasfer 4 * 8 * qr_size * dsnr_domains
        const double f1 = gg.grid_ptr_cu[i_local * gg.x_size + j_local];
        const double f2 = gg.grid_ptr_cu[i_local * gg.y_size + j_local + 1];
        const double f3 = gg.grid_ptr_cu[(i_local + 1) * gg.y_size + j_local];
        const double f4 = gg.grid_ptr_cu[(i_local + 1) * gg.y_size + j_local + 1];

        // const double f1 = 0.0 , f2 = 0.0, f3 = 0.0, f4 = 0.0;

        // std::cout << "i_local: " << i_local << " j_local: " << j_local << std::endl;
        // std::cout << "f1: " << f1 << " f2: " << f2 << " f3: " << f3 << " f4: " << f4
        //           << std::endl;

        // std::cout << std::endl;

        // check if qs is correct
        // if (x_Q < x_d_min || x_Q > x_d_max || y_Q < y_d_min || y_Q > y_d_max) {
        //     std::cout << "x_Q: " << x_Q << " y_Q: " << y_Q << std::endl;
        //     std::cout << "x_d_min: " << x_d_min << " x_d_max: " << x_d_max << std::endl;
        //     std::cout << "y_d_min: " << y_d_min << " y_d_max: " << y_d_max << std::endl;
        //     return false;
        // }

        const double x1 = gg.x_zero + i_local * gg.delta;        // 2 * qr_size * ds_nr_domains
        const double x2 = gg.x_zero + (i_local + 1) * gg.delta;  // 3 * qr_size * ds_nr_domains
        const double y1 = gg.y_zero + j_local * gg.delta;        // 1 * qr_size * ds_nr_domains
        const double y2 = gg.y_zero + (j_local + 1) * gg.delta;  // 3 * qr_size * ds_nr_domains

        // std::cout << "x1: " << x1 << " x2: " << x2 << " y1: " << y1 << " y2: " << y2
        //           << std::endl;

        // 5 * 4 * qr_size * ds.nr_domains
        const double w11 = (x2 - x_Q) * (y2 - y_Q) / (gg.delta * gg.delta);
        const double w12 = (x2 - x_Q) * (y_Q - y1) / (gg.delta * gg.delta);
        const double w21 = (x_Q - x1) * (y2 - y_Q) / (gg.delta * gg.delta);
        const double w22 = (x_Q - x1) * (y_Q - y1) / (gg.delta * gg.delta);

        // std::cout << "w11: " << w11 << " w12: " << w12 << " w21: " << w21 << " w22: " << w22
        //           << std::endl;

        // 7 * qr_size * ds.nr_domains
        const double f_Q = w11 * f1 + w12 * f2 + w21 * f3 + w22 * f4;

        // Qs_i += f_Q * volume;

        // 3 * qr_size * ds.nr_domains
        // data transfer 8 * qr_size * ds.nr_domains
        Qs_i += f_Q * qr.weights_ptr_cu[q_i] * volume;

        // std::cout << "---gg" << std::endl;
    }

    // Qs[i] = Qs_i;
    // }

    Qs[id] = Qs_i;

    return;
}

/**
 * @brief Kernel function to perform the quadrature in the global domain.
 *
 * @param Qs
 * @param gg
 * @param qr
 * @param qr_nodes_nr_
 * @return __global__
 */
__global__ void perform_quadrature_global_stripe_kernel(double* Qs,                            //
                                                        const global_grid_type gg,             //
                                                        const quadrature_rule qr,              //
                                                        const size_t qr_nodes_nr_,             //
                                                        const domains_stripe* ds_vector,       //
                                                        const size_t nr_of_domains_stripes) {  //
    //
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t domain_nr = threadIdx.x;
    const size_t stripe_nr = blockIdx.x;

    perform_quadrature_global_stripe(Qs, gg, qr, qr_nodes_nr_, ds_vector[stripe_nr]);
}

struct local_grid_cuda_type {
    double delta; /**< The grid spacing. */

    double x_d_min; /**< The x-coordinate of the domain grid origin. */
    double y_d_min; /**< The y-coordinate of the domain grid origin. */

    double x_d_max; /**< The maximum x-coordinate domain of the grid. */
    double y_d_max; /**< The maximum y-coordinate domain of the grid. */

    double x_grid_min; /**< The min x-coordinate in the local grid. */
    double y_grid_min; /**< The min y-coordinate in the local grid. */

    double x_grid_max; /**< The maximum x-coordinate in the local grid. */
    double y_grid_max; /**< The maximum y-coordinate in the local grid. */

    size_t x_size; /**< The number of grid points in the x-direction. */
    size_t y_size; /**< The number of grid points in the y-direction. */
};

/**
 * @brief
 *
 * @param Qs
 * @param gg
 * @param qr
 * @param qr_nodes_nr_
 * @param ds_vector
 * @param nr_of_domains_stripes
 * @return __global__
 */
__global__ void perform_quadrature_local_stripe_kernel(double* Qs,                            //
                                                       const global_grid_type gg,             //
                                                       const quadrature_rule qr,              //
                                                       const size_t qr_nodes_nr_,             //
                                                       const domains_stripe* ds_vector,       //
                                                       const size_t nr_of_domains_stripes) {  //
    //
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t domain_nr = threadIdx.x;
    const size_t stripe_nr = blockIdx.x;

    struct local_grid_cuda_type lg;

    lg.delta = gg.delta;
    double x_d_min, y_d_min, x_d_max, y_d_max;
    x_d_min = ds_vector[stripe_nr].x_min;
    x_d_max = ds_vector[stripe_nr].x_min +
              ds_vector[stripe_nr].nr_domains * ds_vector[stripe_nr].side_x;

    y_d_min = ds_vector[stripe_nr].y_min;
    y_d_max = ds_vector[stripe_nr].y_min + ds_vector[stripe_nr].side_y;

    lg.x_d_min = x_d_min;
    lg.y_d_min = y_d_min;
    lg.x_d_max = x_d_max;
    lg.y_d_max = y_d_max;

    int i_local_min, j_local_min;
    get_nearest_coordinates_floor_cu(gg.x_zero,  //
                                     gg.y_zero,  //
                                     gg.delta,
                                     gg.delta,
                                     x_d_min,
                                     y_d_min,
                                     i_local_min,
                                     j_local_min);

    int i_local_max, j_local_max;
    get_nearest_coordinates_ceil_cu(gg.x_zero,  //
                                    gg.y_zero,  //
                                    gg.delta,
                                    gg.delta,
                                    x_d_max,
                                    y_d_max,
                                    i_local_max,
                                    j_local_max);

    lg.x_grid_min = gg.x_zero + i_local_min * gg.delta;
    lg.y_grid_min = gg.y_zero + j_local_min * gg.delta;

    lg.x_grid_max = gg.x_zero + i_local_max * gg.delta;
    lg.y_grid_max = gg.y_zero + j_local_max * gg.delta;

    lg.x_size = i_local_max - i_local_min + 1;
    lg.y_size = j_local_max - j_local_min + 1;

    __shared__ double local_grid_shared[1024];

    // perform_quadrature_global_stripe(Qs, gg, qr, qr_nodes_nr_, ds_vector[stripe_nr]);
}

/**
 * @brief Copies the global grid to the device.
 *
 * @param gg
 * @return true
 * @return false
 */
bool copy_global_grid_to_device(global_grid_type& gg) {
    //
    cudaError e1 = cudaMalloc((void**)&gg.grid_ptr_cu,  //
                              (unsigned long)gg.grid.size() * sizeof(double));

    cudaError e2 = cudaMemcpy(gg.grid_ptr_cu,
                              &gg.grid[0],
                              (unsigned long)gg.grid.size() * sizeof(double),
                              cudaMemcpyHostToDevice);

    if (e1 != cudaSuccess || e2 != cudaSuccess) {
        printf("Error Alloccating and copying global grid to device\n");
        printf("Error code e1: %s\n", cudaGetErrorString(e1));
        printf("Error code e2: %s\n", cudaGetErrorString(e2));
        return false;
    }

    return true;
}

/**
 * @brief
 *
 * @param gg
 * @return true
 * @return false
 */
bool free_global_grid_on_device(global_grid_type& gg) {
    cudaError e1 = cudaFree(gg.grid_ptr_cu);

    if (e1 != cudaSuccess) {
        printf("!!!!! Error freeing global grid on device\n");
        printf("!!!!! Error code: %s\n", cudaGetErrorString(e1));
        return false;
    }

    gg.grid_ptr_cu = nullptr;

    return true;
}

/**
 * @brief Copies the global grid to the device.
 *
 * @param qr
 * @return true
 * @return false
 */
bool copy_quadrature_rule_to_device(const quadrature_rule& qr) {
    cudaError e1 = cudaMalloc((void**)&qr.x_nodes_ptr_cu,  //
                              (unsigned long)qr.x_nodes.size() * sizeof(double));

    cudaError e2 = cudaMalloc((void**)&qr.y_nodes_ptr_cu,  //
                              (unsigned long)qr.y_nodes.size() * sizeof(double));

    cudaError e3 = cudaMalloc((void**)&qr.weights_ptr_cu,  //
                              (unsigned long)qr.weights.size() * sizeof(double));

    cudaError e4 = cudaMemcpy(qr.x_nodes_ptr_cu,
                              &qr.x_nodes[0],
                              (unsigned long)qr.x_nodes.size() * sizeof(double),
                              cudaMemcpyHostToDevice);

    cudaError e5 = cudaMemcpy(qr.y_nodes_ptr_cu,
                              &qr.y_nodes[0],
                              (unsigned long)qr.y_nodes.size() * sizeof(double),
                              cudaMemcpyHostToDevice);

    cudaError e6 = cudaMemcpy(qr.weights_ptr_cu,
                              &qr.weights[0],
                              (unsigned long)qr.weights.size() * sizeof(double),
                              cudaMemcpyHostToDevice);

    printf("qr.x_nodes.size(): %lu\n", qr.x_nodes.size());
    printf("qr.y_nodes.size(): %lu\n", qr.y_nodes.size());
    printf("qr.weights.size(): %lu\n", qr.weights.size());

    if (e1 != cudaSuccess || e2 != cudaSuccess ||  //
        e3 != cudaSuccess || e4 != cudaSuccess ||  //
        e5 != cudaSuccess || e6 != cudaSuccess) {  //

        printf("Error Copying quadrature rule to device\n");
        printf("Error code e1: %s\n", cudaGetErrorString(e1));
        printf("Error code e2: %s\n", cudaGetErrorString(e2));
        printf("Error code e3: %s\n", cudaGetErrorString(e3));
        printf("Error code e4: %s\n", cudaGetErrorString(e4));
        printf("Error code e5: %s\n", cudaGetErrorString(e5));
        printf("Error code e6: %s\n", cudaGetErrorString(e6));

        return false;
    }

    return true;
}

/**
 * @brief Frees the quadrature rule on the device.
 *
 * @param qr
 * @return true
 * @return false
 */
bool free_quadrature_rule_on_device(quadrature_rule& qr) {
    cudaError e1 = cudaFree(qr.x_nodes_ptr_cu);
    cudaError e2 = cudaFree(qr.y_nodes_ptr_cu);
    cudaError e3 = cudaFree(qr.weights_ptr_cu);

    if (e1 != cudaSuccess || e2 != cudaSuccess || e3 != cudaSuccess) {
        return false;
    }

    qr.x_nodes_ptr_cu = nullptr;
    qr.y_nodes_ptr_cu = nullptr;
    qr.weights_ptr_cu = nullptr;

    return true;
}

void print_GPU_info() {
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    printf("\n");

    printf("Device:                %s\n", deviceProp.name);
    printf("CUDA Capability:       %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("Memory available:      %.3lf Gbytes\n",
           (double)deviceProp.totalGlobalMem / (double)(1024 * 1024 * 1024));
    printf("Number of SMs:         %d\n", deviceProp.multiProcessorCount);
    printf("Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
    printf("Max threads per SM:    %d\n", deviceProp.maxThreadsPerMultiProcessor);
    printf("Shared memory per SM:  %d\n", deviceProp.sharedMemPerMultiprocessor);
    printf("Number of SP:          %d\n", getSPcores(deviceProp));
    printf("Warp size:             %d\n", deviceProp.warpSize);
    printf("Max lane per SM:       %d\n", getSPcores(deviceProp) / deviceProp.multiProcessorCount);

    printf("\n");
}

/**
 * @brief
 *
 */
extern "C" int test_grid_cuda(global_grid_type& gg,                      //
                              quadrature_rule& qr,                       //
                              std::vector<domains_stripe>& ds_vector) {  //
    print_GPU_info();

    copy_global_grid_to_device(gg);
    copy_quadrature_rule_to_device(qr);

    domains_stripe* ds_vector_cu = nullptr;

    cudaError e1 = cudaMalloc((void**)&ds_vector_cu, ds_vector.size() * sizeof(domains_stripe));
    cudaError e2 = cudaMemcpy(ds_vector_cu,
                              &ds_vector[0],
                              ds_vector.size() * sizeof(domains_stripe),
                              cudaMemcpyHostToDevice);
    if (e1 != cudaSuccess || e2 != cudaSuccess) {
        printf("Error allocating memory for ds_vector_cu\n");
        printf("Error code e1: %s\n", cudaGetErrorString(e1));
        printf("Error code e2: %s\n", cudaGetErrorString(e2));
        return 1;
    }

    size_t nr_domains_tot = ds_vector.size() * ds_vector[0].nr_domains;

    double *Qs, *Qs_cu;
    cudaMalloc((void**)&Qs_cu, nr_domains_tot * sizeof(double));
    Qs = (double*)malloc(nr_domains_tot * sizeof(double));

    ///// Kernel here /////
    const size_t nr_stripes = ds_vector.size();
    const size_t nr_domains_per_stripe = ds_vector[0].nr_domains;

    printf("\nnr_stripes:            %lu\n", nr_stripes);
    printf("nr_domains_per_stripe: %lu\n\n", nr_domains_per_stripe);

    printf("Nr of qr nodes: %lu\n", qr.x_nodes.size());

    printf("Nr of threads: %lu\n", nr_stripes * nr_domains_per_stripe);
    printf("Nr of domains: %lu\n", nr_domains_tot);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaDeviceSynchronize();

    cudaEventRecord(start);

    /// start kernel ///
    perform_quadrature_global_stripe_kernel<<<nr_stripes, nr_domains_per_stripe>>>(  //
            Qs_cu,                                                                   //
            gg,                                                                      //
            qr,                                                                      //
            qr.weights.size(),                                                       //
            ds_vector_cu,                                                            //
            ds_vector.size());                                                       //

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // get error code
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "KERNEL ERROR: %s\n", cudaGetErrorString(error));
        return 1;
    }

    cudaDeviceSynchronize();

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Time elapsed: %f ms\n", milliseconds);

    ///// end kernel /////

    cudaMemcpy(Qs, Qs_cu, nr_domains_tot * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(Qs_cu);

    free_global_grid_on_device(gg);
    free_quadrature_rule_on_device(qr);

    cudaFree(ds_vector_cu);

    double Q_tot = 0.0;
    for (size_t i = 0; i < nr_domains_tot; ++i) {
        Q_tot += Qs[i];
    }

    free(Qs);

    printf("\n+++++ Global Q_tot GPU : %f   \n\n", Q_tot);

    return 0;
}