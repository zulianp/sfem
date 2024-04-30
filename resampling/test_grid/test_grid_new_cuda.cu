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
__device__ __forceinline__ void get_nearest_coordinates_floor_cu(const double x_zero,
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
__device__ __forceinline__ void get_nearest_coordinates_ceil_cu(const double x_zero,
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
__device__ __forceinline__                                                 //
        void                                                               //
        perform_quadrature_global_stripe(double* Qs,                       //
                                         const global_grid_cuda_type& gg,  //
                                         const quadrature_rule_cuda& qr,   //
                                         const size_t qr_nodes_nr_,        //
                                         const domains_stripe& ds) {       //
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
        // 2 * 3 * qnr * dnr
        const double x_Q = (qr.x_nodes_ptr_cu[q_i]) * (x_d_max - x_d_min) + x_d_min;
        const double y_Q = (qr.y_nodes_ptr_cu[q_i]) * (y_d_max - y_d_min) + y_d_min;

        int i_local, j_local;

        // dnr * qnr * 8
        get_nearest_coordinates_floor_cu(gg.x_zero,  //
                                         gg.y_zero,  //
                                         gg.delta,
                                         gg.delta,
                                         x_Q,
                                         y_Q,
                                         i_local,
                                         j_local);

        /////// data trasfer: 4 * 8 * qnr * dnr
        // const size_t i1 = INDEX_ABS(gg.y_size, i_local, j_local);
        // const size_t i2 = INDEX_ABS(gg.y_size, i_local, j_local + 1);
        // const size_t i3 = INDEX_ABS(gg.y_size, i_local + 1, j_local);
        // const size_t i4 = INDEX_ABS(gg.y_size, i_local + 1, j_local + 1);

        const size_t i1 = XY_INDEX(gg.x_size, gg.y_size, i_local, j_local);
        const size_t i2 = XY_INDEX(gg.x_size, gg.y_size, i_local, j_local + 1);
        const size_t i3 = XY_INDEX(gg.x_size, gg.y_size, i_local + 1, j_local);
        const size_t i4 = XY_INDEX(gg.x_size, gg.y_size, i_local + 1, j_local + 1);

        const double f1 = gg.grid_ptr_cu[i1];
        const double f2 = gg.grid_ptr_cu[i2];
        const double f3 = gg.grid_ptr_cu[i3];
        const double f4 = gg.grid_ptr_cu[i4];

        const double x1 = gg.x_zero + i_local * gg.delta;        // 2 * qnr * dnr
        const double x2 = gg.x_zero + (i_local + 1) * gg.delta;  // 3 * qnr * dnr
        const double y1 = gg.y_zero + j_local * gg.delta;        // 1 * qnr * dnr
        const double y2 = gg.y_zero + (j_local + 1) * gg.delta;  // 3 * qnr * dnr

        // std::cout << "x1: " << x1 << " x2: " << x2 << " y1: " << y1 << " y2: " << y2
        //           << std::endl;

        // 5 * 4 * qnr * dnr
        const double w11 = (x2 - x_Q) * (y2 - y_Q) / (gg.delta * gg.delta);
        const double w12 = (x2 - x_Q) * (y_Q - y1) / (gg.delta * gg.delta);
        const double w21 = (x_Q - x1) * (y2 - y_Q) / (gg.delta * gg.delta);
        const double w22 = (x_Q - x1) * (y_Q - y1) / (gg.delta * gg.delta);

        // 7 * qnr * dnr
        const double f_Q = w11 * f1 + w12 * f2 + w21 * f3 + w22 * f4;

        // Qs_i += f_Q * volume;

        // 3 * qnr * dnr
        // data transfer 8 * qne * dnr
        Qs_i += f_Q * qr.weights_ptr_cu[q_i] * volume;

        // std::cout << "---gg" << std::endl;
    }

    // Qs[i] = Qs_i;
    // }

    Qs[id] = Qs_i;

    return;
}

__device__ void foo(double* Qs, double* local_grid_shared) {
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    Qs[id] = local_grid_shared[id % blockDim.x];
}

/**
 * @brief Performs the quadrature for a single stripe.
 *
 * @param Qs
 * @param gg
 * @param qr
 * @param qr_nodes_nr_
 * @param ds
 * @return __device__
 */
__device__ __noinline__                                                  //
        void                                                             //
        perform_quadrature_local_stripe(double* Qs,                      //
                                        double* local_grid_shared,       //
                                        const local_grid_cuda_type lg,   //
                                        const quadrature_rule_cuda& qr,  //
                                        const size_t qr_nodes_nr_,       //
                                        const domains_stripe& ds) {      //
                                                                         //
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t domain_nr = threadIdx.x;

    // for (size_t i = 0; i < ds.nr_domains; ++i) {
    double x_d_min, y_d_min, x_d_max, y_d_max;
    get_domain_from_stripe_cu(ds, domain_nr, x_d_min, y_d_min, x_d_max, y_d_max);

    const size_t lg_size = lg.x_size * lg.y_size;

    // printf("Domain %d: x_min: %f, x_max: %f, y_min: %f, y_max: %f  -- GPU\n",
    //        domain_nr,
    //        x_d_min,
    //        x_d_max,
    //        y_d_min,
    //        y_d_max);

    const double volume = (x_d_max - x_d_min) * (y_d_max - y_d_min);  // 3 * ds_nr_domains

    double Qs_i = 0.0;

    const size_t qr_size = qr_nodes_nr_;

    for (size_t q_i = 0; q_i < qr_size; ++q_i) {
        //
        // 2 * 3 * qnr * dnr
        const double x_Q = (qr.x_nodes_ptr_cu[q_i]) * (x_d_max - x_d_min) + x_d_min;
        const double y_Q = (qr.y_nodes_ptr_cu[q_i]) * (y_d_max - y_d_min) + y_d_min;

        int i_local, j_local;

        // dnr * qnr * 8
        get_nearest_coordinates_floor_cu(lg.x_grid_min,  //
                                         lg.y_grid_min,  //
                                         lg.delta,
                                         lg.delta,
                                         x_Q,
                                         y_Q,
                                         i_local,
                                         j_local);

        /////// data trasfer 4 * 8 * qnr * dnr

        // const size_t i1 = INDEX_ABS(lg.y_size, i_local, j_local);
        // const size_t i2 = INDEX_ABS(lg.y_size, i_local, j_local + 1);
        // const size_t i3 = INDEX_ABS(lg.y_size, i_local + 1, j_local);
        // const size_t i4 = INDEX_ABS(lg.y_size, i_local + 1, j_local + 1);

        const size_t i1 = XY_INDEX(lg.x_size, lg.y_size, i_local, j_local);
        const size_t i2 = XY_INDEX(lg.x_size, lg.y_size, i_local, j_local + 1);
        const size_t i3 = XY_INDEX(lg.x_size, lg.y_size, i_local + 1, j_local);
        const size_t i4 = XY_INDEX(lg.x_size, lg.y_size, i_local + 1, j_local + 1);

        // if (i1 >= 4096 or i2 >= 4096 or i3 >= 4096 or i4 >= 4096 or lg_size >= 4096) {
        //     printf("Error: i1: %lu, i2: %lu, i3: %lu, i4: %lu, lg_size: %lu\n",
        //            i1,
        //            i2,
        //            i3,
        //            i4,
        //            lg_size);
        //     // interupt cuda kernel
        //     return;
        // }

        const double f1 = local_grid_shared[i1];
        const double f2 = local_grid_shared[i2];
        const double f3 = local_grid_shared[i3];
        const double f4 = local_grid_shared[i4];

        // const double f1 = 0.0, f2 = 0.0, f3 = 0.0, f4 = 0.0;

        // if (q_i == 1)
        // printf("f1: %f, f2: %f, f3: %f, f4: %f\n", f1, f2, f3, f4);

        const double x1 = lg.x_grid_min + i_local * lg.delta;        // 2 * qnr * dnr
        const double x2 = lg.x_grid_min + (i_local + 1) * lg.delta;  // 3 * qnr * dnr
        const double y1 = lg.y_grid_min + j_local * lg.delta;        // 1 * qnr * dnr
        const double y2 = lg.y_grid_min + (j_local + 1) * lg.delta;  // 3 * qnr * dnr

        //  if (q_i == 1)
        // printf("x1: %f, x2: %f, y1: %f, y2: %f\n", x1, x2, y1, y2);

        // 5 * 4 * qnr * dnr
        const double w11 = (x2 - x_Q) * (y2 - y_Q) / (lg.delta * lg.delta);
        const double w12 = (x2 - x_Q) * (y_Q - y1) / (lg.delta * lg.delta);
        const double w21 = (x_Q - x1) * (y2 - y_Q) / (lg.delta * lg.delta);
        const double w22 = (x_Q - x1) * (y_Q - y1) / (lg.delta * lg.delta);

        // if (q_i == 1)
        // printf("w11: %f, w12: %f, w21: %f, w22: %f\n", w11, w12, w21, w22);

        // 7 * qnr * dnr
        const double f_Q = w11 * f1 + w12 * f2 + w21 * f3 + w22 * f4;

        // // Qs_i += f_Q * volume;
        // if (q_i == 1)
        // printf("f_Q: %f ", f_Q);

        // if (q_i == 1)
        // printf("volume: %f\n", volume);

        // 3 * qnr * dnr
        // data transfer 8 * qne * dnr
        Qs_i += f_Q * qr.weights_ptr_cu[q_i] * volume;

        // if (q_i == 2 and domain_nr == 25) {
        //     printf("w: %f, f_Q: %f, volume: %f, Qs_i: %f, q_i: %lu, domanin_nr: %lu\n",
        //            qr.weights_ptr_cu[q_i],
        //            f_Q,
        //            volume,
        //            Qs_i,
        //            q_i,
        //            domain_nr);
        //     printf("f1: %f, f2: %f, f3: %f, f4: %f\n", f1, f2, f3, f4);
        //     printf("x1: %f, x2: %f, y1: %f, y2: %f\n", x1, x2, y1, y2);
        //     printf("w11: %f, w12: %f, w21: %f, w22: %f\n", w11, w12, w21, w22);
        //     printf("i_local: %d, j_local: %d\n", i_local, j_local);
        //     printf("lg.x_grid_min: %f, lg.y_grid_min: %f, lg.delta: %f\n",
        //            lg.x_grid_min,
        //            lg.y_grid_min,
        //            lg.delta);
        //     printf("lg.x_grid_max: %f, lg.y_grid_max: %f\n", lg.x_grid_max, lg.y_grid_max);
        //     printf("lg.x_size: %lu, lg.y_size: %lu\n", lg.x_size, lg.y_size);
        // }
        // printf("Qs_i: %lf, q_i: %lu, domanin_nr: %lu\n", Qs_i, q_i, domain_nr);

        // std::cout << "---gg" << std::endl;
    }

    // printf("\n");

    // Qs[i] = Qs_i;
    // }

    // printf("Qs_i: %f\n", Qs_i);

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
                                                        const global_grid_cuda_type gg,        //
                                                        const quadrature_rule_cuda qr,         //
                                                        const size_t qr_nodes_nr_,             //
                                                        const domains_stripe* ds_vector,       //
                                                        const size_t nr_of_domains_stripes) {  //
    //
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t domain_nr = threadIdx.x;
    const size_t stripe_nr = blockIdx.x;

    perform_quadrature_global_stripe(Qs, gg, qr, qr_nodes_nr_, ds_vector[stripe_nr]);
}

/**
 * @brief Kernel function to perform the quadrature in the local domain.
 *
 * @param Qs
 * @param gg
 * @param qr
 * @param qr_nodes_nr_
 * @param ds_vector
 * @param nr_of_domains_stripes
 * @return __global__
 */
__global__                                                                            //
        void                                                                          //
        perform_quadrature_local_stripe_kernel(double* Qs,                            //
                                               const global_grid_cuda_type gg,        //
                                               const quadrature_rule_cuda qr,         //
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
    y_d_min = ds_vector[stripe_nr].y_min;

    x_d_max = x_d_min + double(ds_vector[stripe_nr].nr_domains) * ds_vector[stripe_nr].side_x;
    y_d_max = y_d_min + ds_vector[stripe_nr].side_y;

    lg.x_d_min = x_d_min;
    lg.y_d_min = y_d_min;

    lg.x_d_max = x_d_max;
    lg.y_d_max = y_d_max;

    int i_global_min;
    int j_global_min;
    get_nearest_coordinates_floor_cu(gg.x_zero,  //
                                     gg.y_zero,  //
                                     gg.delta,
                                     gg.delta,
                                     x_d_min,
                                     y_d_min,
                                     i_global_min,
                                     j_global_min);

    int i_global_max;
    int j_global_max;
    get_nearest_coordinates_ceil_cu(gg.x_zero,  //
                                    gg.y_zero,  //
                                    gg.delta,
                                    gg.delta,
                                    x_d_max,
                                    y_d_max,
                                    i_global_max,
                                    j_global_max);

    lg.x_grid_min = gg.x_zero + i_global_min * gg.delta;
    lg.y_grid_min = gg.y_zero + j_global_min * gg.delta;

    lg.x_grid_max = gg.x_zero + i_global_max * gg.delta;
    lg.y_grid_max = gg.y_zero + j_global_max * gg.delta;

    lg.x_size = size_t(i_global_max - i_global_min + 1);
    lg.y_size = size_t(j_global_max - j_global_min + 1);

#define LOCAL_GRID_SIZE 4096

    __shared__ double local_grid_shared[LOCAL_GRID_SIZE];
    // __shared__ double local_grid_shared_b[LOCAL_GRID_SIZE];

    int i = 0, j = 0;

    int i_global = i_global_min;
    int j_global = j_global_min;

    const int local_grid_size = lg.x_size * lg.y_size;
    // printf("local_grid_size: %d\n", local_grid_size);

    if (local_grid_size > LOCAL_GRID_SIZE) {
        printf("Error: local_grid_size > LOCAL_GRID_SIZE\n");
        printf("local_grid_size: %d\n", local_grid_size);
        printf("LOCAL_GRID_SIZE: %d\n", LOCAL_GRID_SIZE);
        return;
    }

    int cnt = 0;

    const int block_dim = blockDim.x;
    const int thread_id = threadIdx.x;

    while (true) {
        int index_abs_local = cnt * block_dim + thread_id;

#if Y_MAJOR == 1
        i = index_abs_local % lg.x_size;
        j = index_abs_local / lg.x_size;
#elif X_MAJOR == 1
        i = index_abs_local / lg.y_size;
        j = index_abs_local % lg.y_size;
#else
#error "Either Y_MAJOR or X_MAJOR should be defined"
#endif

        if (index_abs_local >= local_grid_size) break;

        i_global = i_global_min + i;
        j_global = j_global_min + j;

        const size_t i_global_abs = XY_INDEX(gg.x_size, gg.y_size, i_global, j_global);
        const size_t i_local_abs = XY_INDEX(lg.x_size, lg.y_size, i, j);

        local_grid_shared[i_local_abs] = gg.grid_ptr_cu[i_global_abs];

        cnt += 1;
    }

    __syncthreads();

    //     if (threadIdx.x == 0) {
    //         for (size_t i = 0; i < lg.x_size; ++i) {
    //             for (size_t j = 0; j < lg.y_size; ++j) {
    //                 printf("%ld %ld %f, ", i, j, local_grid_shared[i * lg.y_size + j]);
    //             }
    //             printf("\n");
    //         }
    //     }

    // __syncthreads();

    // foo(Qs, local_grid_shared);

    perform_quadrature_local_stripe(Qs,  //
                                    local_grid_shared,
                                    lg,
                                    qr,
                                    qr_nodes_nr_,
                                    ds_vector[stripe_nr]);

    // printf("shared: %f\n", local_grid_shared_b[0]);
}

__global__ void axpy(double* x, double* y, double a, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) y[i] = a * x[i] + y[i];
}

/**
 * @brief Frees the global grid on the device.
 *
 * @param gg The global grid to be freed.
 * @return true if the memory was successfully freed.
 * @return false if an error occurred while freeing the memory.
 */
bool copy_global_grid_to_device(const global_grid_type& gg, global_grid_cuda_type& gg_dev) {
    cudaError e1 = cudaMalloc((void**)&gg_dev.grid_ptr_cu,  //
                              (unsigned long)gg.grid.size() * sizeof(double));

    cudaError e2 = cudaMemcpy(gg_dev.grid_ptr_cu,
                              &gg.grid[0],
                              (unsigned long)gg.grid.size() * sizeof(double),
                              cudaMemcpyHostToDevice);

    if (e1 != cudaSuccess || e2 != cudaSuccess) {
        printf("!!!!!  Error Alloccating and copying global grid to device\n");
        printf("!!!!!  Error code e1: %s\n", cudaGetErrorString(e1));
        printf("!!!!!  Error code e2: %s\n", cudaGetErrorString(e2));
        return false;
    }

    gg_dev.delta = gg.delta;
    gg_dev.x_zero = gg.x_zero;
    gg_dev.y_zero = gg.y_zero;
    gg_dev.x_size = gg.x_size;
    gg_dev.y_size = gg.y_size;

    return true;
}

/**
 * @brief Frees the global grid on the device.
 *
 * @param gg The global grid to be freed.
 * @return true if the memory was successfully freed.
 * @return false if an error occurred while freeing the memory.
 */
bool free_global_grid_on_device(global_grid_cuda_type& gg) {
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
 * @param qr The quadrature rule to be copied.
 * @param qr_dev The quadrature rule on the device.
 * @return true if the quadrature rule was successfully copied to the device.
 * @return false if an error occurred while copying the quadrature rule to the device.
 */
bool copy_quadrature_rule_to_device_cu(const quadrature_rule& qr, quadrature_rule_cuda& qr_dev) {
    //
    cudaError e1 = cudaMalloc((void**)&qr_dev.x_nodes_ptr_cu,  //
                              (unsigned long)qr.x_nodes.size() * sizeof(double));

    cudaError e2 = cudaMalloc((void**)&qr_dev.y_nodes_ptr_cu,  //
                              (unsigned long)qr.y_nodes.size() * sizeof(double));

    cudaError e3 = cudaMalloc((void**)&qr_dev.weights_ptr_cu,  //
                              (unsigned long)qr.weights.size() * sizeof(double));

    cudaError e4 = cudaMemcpy(qr_dev.x_nodes_ptr_cu,
                              &qr.x_nodes[0],
                              (unsigned long)qr.x_nodes.size() * sizeof(double),
                              cudaMemcpyHostToDevice);

    cudaError e5 = cudaMemcpy(qr_dev.y_nodes_ptr_cu,
                              &qr.y_nodes[0],
                              (unsigned long)qr.y_nodes.size() * sizeof(double),
                              cudaMemcpyHostToDevice);

    cudaError e6 = cudaMemcpy(qr_dev.weights_ptr_cu,
                              &qr.weights[0],
                              (unsigned long)qr.weights.size() * sizeof(double),
                              cudaMemcpyHostToDevice);

    printf("\nqr.x_nodes.size(): %lu\n", qr.x_nodes.size());
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
 * @param qr_dev reference to the quadrature rule on the device.
 * @return true if the memory was successfully freed.
 * @return false if an error occurred while freeing the memory.
 */
bool free_quadrature_rule_on_device(quadrature_rule_cuda& qr_dev) {
    cudaError e1 = cudaFree(qr_dev.x_nodes_ptr_cu);
    cudaError e2 = cudaFree(qr_dev.y_nodes_ptr_cu);
    cudaError e3 = cudaFree(qr_dev.weights_ptr_cu);

    if (e1 != cudaSuccess || e2 != cudaSuccess || e3 != cudaSuccess) {
        printf("Error freeing quadrature rule on device\n");
        printf("Error code e1: %s\n", cudaGetErrorString(e1));
        printf("Error code e2: %s\n", cudaGetErrorString(e2));
        printf("Error code e3: %s\n", cudaGetErrorString(e3));

        return false;
    }

    qr_dev.x_nodes_ptr_cu = nullptr;
    qr_dev.y_nodes_ptr_cu = nullptr;
    qr_dev.weights_ptr_cu = nullptr;

    return true;
}

/**
 * @brief Prints the GPU information.
 */
void print_GPU_info() {
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    printf("\n");

    printf("Device:                %s\n", deviceProp.name);
    printf("CUDA Capability:       %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("PCI Bus ID:            %d\n", deviceProp.pciBusID);
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
 * @param Q_global
 * @param gg
 * @param qr
 * @param ds_vector
 * @return int
 */
int test_global(double& Q_global,                          //
                global_grid_type& gg,                      //
                quadrature_rule& qr,                       //
                std::vector<domains_stripe>& ds_vector) {  //

    global_grid_cuda_type gg_dev;
    quadrature_rule_cuda qr_dev;

    copy_global_grid_to_device(gg, gg_dev);
    // copy_quadrature_rule_to_device(qr);
    copy_quadrature_rule_to_device_cu(qr, qr_dev);

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
    // cudaProfilerStart();

    cudaEventRecord(start);

    /// start kernel ///
    perform_quadrature_global_stripe_kernel<<<nr_stripes, nr_domains_per_stripe>>>(  //
            Qs_cu,                                                                   //
            gg_dev,                                                                  //
            qr_dev,                                                                  //
            qr.weights.size(),                                                       //
            ds_vector_cu,                                                            //
            ds_vector.size());                                                       //

    // axpy<<<nr_domains_tot, 1024>>>(Qs_cu, Qs_cu, 1.0, nr_domains_tot);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // cudaProfilerStop();

    // get error code
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "KERNEL ERROR: %s\n", cudaGetErrorString(error));
        return 1;
    }

    cudaDeviceSynchronize();

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Global Time elapsed: %f ms\n", milliseconds);

    ///// end kernel /////

    cudaMemcpy(Qs, Qs_cu, nr_domains_tot * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(Qs_cu);

    free_global_grid_on_device(gg_dev);
    free_quadrature_rule_on_device(qr_dev);

    cudaFree(ds_vector_cu);

    double Q_tot = 0.0;
    for (size_t i = 0; i < nr_domains_tot; ++i) {
        Q_tot += Qs[i];
    }

    free(Qs);

    printf("\n+++++ Global Q_tot GPU : %f   \n\n", Q_tot);
    const double flops =
            q_flops(nr_stripes, nr_domains_per_stripe, qr.weights.size(), milliseconds / 1000.0);
    printf("Global CUDA GFLOPS: %f\n\n", flops / (1024.0 * 1024.0 * 1024.0));

    Q_global = Q_tot;

    return 0;
}

/**
 * @brief
 *
 * @param Q_global
 * @param gg
 * @param qr
 * @param ds_vector
 * @return int
 */
int test_local(double& Q_local,                           //
               global_grid_type& gg,                      //
               quadrature_rule& qr,                       //
               std::vector<domains_stripe>& ds_vector) {  //

    global_grid_cuda_type gg_dev;
    quadrature_rule_cuda qr_dev;

    copy_global_grid_to_device(gg, gg_dev);
    copy_quadrature_rule_to_device_cu(qr, qr_dev);

    // copy_quadrature_rule_to_device(qr);

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

    printf("--------------------------------\n");
    printf("--------------------------------\n");

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

    /* start kernel  */
    perform_quadrature_local_stripe_kernel<<<nr_stripes, nr_domains_per_stripe>>>(  //
            Qs_cu,                                                                  //
            gg_dev,                                                                 //
            qr_dev,                                                                 //
            qr.weights.size(),                                                      //
            ds_vector_cu,                                                           //
            ds_vector.size());                                                      //

    // axpy<<<nr_domains_tot, 1024>>>(Qs_cu, Qs_cu, 1.0, nr_domains_tot);

    cudaEventSynchronize(stop);
    cudaEventRecord(stop);

    // get error code
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "KERNEL ERROR: %s\n", cudaGetErrorString(error));
        return 1;
    }

    cudaDeviceSynchronize();

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Local Time elapsed: %f ms\n", milliseconds);

    ///// end kernel /////

    cudaMemcpy(Qs, Qs_cu, nr_domains_tot * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(Qs_cu);

    free_global_grid_on_device(gg_dev);
    free_quadrature_rule_on_device(qr_dev);

    cudaFree(ds_vector_cu);

    double Q_tot = 0.0;
    for (size_t i = 0; i < nr_domains_tot; ++i) {
        Q_tot += Qs[i];
    }

    free(Qs);

    printf("--------------------------------\n");
    printf("\n+++++ Local Q_tot GPU : %f   \n\n", Q_tot);

    const double flops =
            q_flops(nr_stripes, nr_domains_per_stripe, qr.weights.size(), milliseconds / 1000.0);
    printf("Local CUDA GFLOPS: %f\n", flops / (1024.0 * 1024.0 * 1024.0));

    Q_local = Q_tot;

    return 0;
}

/**
 * @brief
 *
 */
extern "C" int test_grid_cuda(global_grid_type& gg,                      //
                              quadrature_rule& qr,                       //
                              std::vector<domains_stripe>& ds_vector) {  //
                                                                         //
    print_GPU_info();

    double Q_global = 0.0;
    test_global(Q_global, gg, qr, ds_vector);

    double Q_local = 0.0;
    test_local(Q_local, gg, qr, ds_vector);

    printf("\n");
    printf("Result *************************************************************\n");
    printf("Q_global: %f\n", Q_global);
    printf("Q_local:  %f\n", Q_local);
    printf("Q_global - Q_local: %f\n", Q_global - Q_local);
    printf("Q_global / Q_local: %f\n", Q_global / Q_local);
    printf("\n");
    printf("\n");

    return 0;
}