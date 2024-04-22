#include <stdio.h>

#include "test_grid_new.h"

__device__ void get_nearest_coordinates_floor_cu(const double x_zero,  //
                                                 const double y_zero,
                                                 const double delta_x,
                                                 const double delta_y,
                                                 const double x,
                                                 const double y,
                                                 int& i,
                                                 int& j) {
    i = static_cast<int>(floor((x - x_zero) / delta_x));
    j = static_cast<int>(floor((y - y_zero) / delta_y));
}

__device__ void get_domain_from_stripe_cu(const domains_stripe& ds,
                                          const size_t domain_nr,
                                          double& x_min,
                                          double& y_min,
                                          double& x_max,
                                          double& y_max) {
    x_min = ds.x_min + domain_nr * ds.side_x;
    y_min = ds.y_min;

    x_max = x_min + ds.side_x;
    y_max = y_min + ds.side_y;
}

__device__ void perform_quadrature_global_stripe(double* Qs,                  //
                                                 const global_grid_type& gg,  //
                                                 const quadrature_rule& qr,   //
                                                 const size_t qr_nodes_nr_,   //
                                                 const domains_stripe& ds) {  //
    //
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t domain_nr = threadIdx.x;
    const size_t stripe_nr = blockIdx.x;

    double Ql = 0.0;

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

        // std::cout << "x_Q: " << x_Q << " y_Q: " << y_Q << std::endl;

        int i_local, j_local;

        get_nearest_coordinates_floor_cu(gg.x_zero,  //
                                         gg.y_zero,  //
                                         gg.delta,
                                         gg.delta,
                                         x_Q,
                                         y_Q,
                                         i_local,
                                         j_local);

        // data trasfer 4 * 8 * qr_size * dsnr_domains
        const double f1 = gg.grid_ptr_cu[i_local * gg.x_size + j_local];
        const double f2 = gg.grid_ptr_cu[i_local * gg.y_size + j_local + 1];
        const double f3 = gg.grid_ptr_cu[(i_local + 1) * gg.y_size + j_local];
        const double f4 = gg.grid_ptr_cu[(i_local + 1) * gg.y_size + j_local + 1];

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

extern "C" int test_grid() {
    printf("Hello from test_grid_new_cuda.cu\n");
    return 0;
}