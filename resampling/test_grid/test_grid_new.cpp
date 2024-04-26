#include <chrono>
#include <cmath>
#include <fstream>  // Include the necessary header file
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <thread>
#include <tuple>
#include <valarray>
#include <vector>

#include "test_grid_new.h"

/**
 * @brief Calculates the number of bytes required to store a global grid.
 *
 * @param gg The global grid to calculate the size of.
 * @return size_t The number of bytes required to store the global grid.
 */
inline size_t global_grid_bytes(const global_grid_type& gg) {
    return gg.grid.size() * sizeof(double);
}

/**
 * @brief Function to get the nearest coordinates based on given parameters.
 *
 * @param x_zero The x-coordinate of the origin.
 * @param y_zero The y-coordinate of the origin.
 * @param delta_x The x-coordinate spacing.
 * @param delta_y The y-coordinate spacing.
 * @param x The x-coordinate for which nearest coordinates are to be found.
 * @param y The y-coordinate for which nearest coordinates are to be found.
 * @param ceil_flag Flag indicating whether to round up the coordinates.
 * @return std::tuple<int, int> The nearest coordinates as a tuple of integers.
 */
inline std::tuple<int, int> get_nearest_coordinates(const double x_zero,  //
                                                    const double y_zero,
                                                    const double delta_x,
                                                    const double delta_y,
                                                    const double x,
                                                    const double y,
                                                    const bool ceil_flag) {
    if (ceil_flag) {
        return std::make_tuple(static_cast<int>(std::ceil((x - x_zero) / delta_x)),
                               static_cast<int>(std::ceil((y - y_zero) / delta_y)));
    } else {
        return std::make_tuple(static_cast<int>(std::floor((x - x_zero) / delta_x)),
                               static_cast<int>(std::floor((y - y_zero) / delta_y)));
    }
}

/**
 * Calculates the nearest grid coordinates (floor values) for a given point (x, y)
 * based on the grid origin (x_zero, y_zero) and grid spacing (delta_x, delta_y).
 *
 * @param x_zero The x-coordinate of the grid origin.
 * @param y_zero The y-coordinate of the grid origin.
 * @param delta_x The spacing between grid points along the x-axis.
 * @param delta_y The spacing between grid points along the y-axis.
 * @param x The x-coordinate of the point for which nearest grid coordinates are to be calculated.
 * @param y The y-coordinate of the point for which nearest grid coordinates are to be calculated.
 * @param i Reference to the variable to store the nearest grid coordinate along the x-axis.
 * @param j Reference to the variable to store the nearest grid coordinate along the y-axis.
 */
inline void get_nearest_coordinates_floor(const double x_zero,  //
                                          const double y_zero,
                                          const double delta_x,
                                          const double delta_y,
                                          const double x,
                                          const double y,
                                          int& i,
                                          int& j) {
    i = static_cast<int>(std::floor((x - x_zero) / delta_x));
    j = static_cast<int>(std::floor((y - y_zero) / delta_y));
}

/**
 * Calculates the nearest coordinates (i, j) rounded up to the next integer values,
 * based on the given parameters.
 *
 * @param x_zero The starting x-coordinate of the grid.
 * @param y_zero The starting y-coordinate of the grid.
 * @param delta_x The spacing between adjacent grid points along the x-axis.
 * @param delta_y The spacing between adjacent grid points along the y-axis.
 * @param x The x-coordinate for which to find the nearest grid point.
 * @param y The y-coordinate for which to find the nearest grid point.
 * @param i [out] The nearest x-coordinate rounded up to the next integer value.
 * @param j [out] The nearest y-coordinate rounded up to the next integer value.
 */
inline void get_nearest_coordinates_ceil(const double x_zero,  //
                                         const double y_zero,
                                         const double delta_x,
                                         const double delta_y,
                                         const double x,
                                         const double y,
                                         int& i,
                                         int& j) {
    i = static_cast<int>(std::ceil((x - x_zero) / delta_x));
    j = static_cast<int>(std::ceil((y - y_zero) / delta_y));
}

/**
 * @brief Creates a global grid and populates it with values computed by the provided function.
 *
 * This function creates a global grid of type `global_grid_type` and populates it with values
 * computed by the provided function `f`. The grid is defined by the given parameters: `delta`,
 * `x_zero`, `y_zero`, `x_max`, and `y_max`.
 *
 * @param gg The global grid to be created and populated.
 * @param delta The spacing between grid points.
 * @param x_zero The starting x-coordinate of the grid.
 * @param y_zero The starting y-coordinate of the grid.
 * @param x_max The maximum x-coordinate of the grid.
 * @param y_max The maximum y-coordinate of the grid.
 * @param f The function used to compute the values for each grid point.
 * @return `true` if the grid was successfully created and populated, `false` otherwise.
 */
bool make_gloal_grid(global_grid_type& gg,
                     const double delta,
                     const double x_zero,
                     const double y_zero,
                     const double x_max,
                     const double y_max,
                     std::function<double(double, double)> f) {
    gg.delta = delta;
    gg.x_zero = x_zero;
    gg.y_zero = y_zero;
    gg.x_max = x_max;
    gg.y_max = y_max;

    if (x_max <= x_zero || y_max <= y_zero || delta <= 0.0) {
        return false;
    }

    gg.x_size = static_cast<size_t>(std::ceil((x_max - x_zero) / delta));
    gg.y_size = static_cast<size_t>(std::ceil((y_max - y_zero) / delta));

    gg.grid.resize(gg.x_size * gg.y_size);

    for (size_t i = 0; i < gg.x_size; ++i) {
        for (size_t j = 0; j < gg.y_size; ++j) {
            gg.grid[i * gg.y_size + j] = f(x_zero + i * delta, y_zero + j * delta);
        }
    }

    return true;
}

/**
 * @brief Retrieves a local grid from a global grid within specified bounds.
 *
 * This function takes a global grid and extracts a local grid within the specified bounds.
 * The local grid is defined by the minimum and maximum x and y coordinates.
 *
 * @param lg The local grid to be populated.
 * @param gg The global grid from which the local grid is extracted.
 * @param x_d_min The minimum x coordinate of the local grid.
 * @param y_d_min The minimum y coordinate of the local grid.
 * @param x_d_max The maximum x coordinate of the local grid.
 * @param y_d_max The maximum y coordinate of the local grid.
 * @return True if the local grid is successfully retrieved, false otherwise.
 */
bool get_local_grid(local_grid_type& lg,
                    const global_grid_type& gg,
                    const double x_d_min,
                    const double y_d_min,
                    const double x_d_max,
                    const double y_d_max) {
    lg.delta = gg.delta;

    lg.x_d_min = x_d_min;
    lg.y_d_min = y_d_min;

    lg.x_d_max = x_d_max;
    lg.y_d_max = y_d_max;

    if (x_d_max <= x_d_min || y_d_max <= y_d_min) {
        return false;
    }

    // get x y min coordinates from global grid
    auto [i_min, j_min] = get_nearest_coordinates(gg.x_zero,  //
                                                  gg.y_zero,
                                                  gg.delta,
                                                  gg.delta,
                                                  x_d_min,
                                                  y_d_min,
                                                  false);

    // get x y max coordinates from global grid
    auto [i_max, j_max] = get_nearest_coordinates(gg.x_zero,  //
                                                  gg.y_zero,
                                                  gg.delta,
                                                  gg.delta,
                                                  x_d_max,
                                                  y_d_max,
                                                  true);

    lg.x_grid_min = gg.x_zero + i_min * gg.delta;
    lg.y_grid_min = gg.y_zero + j_min * gg.delta;

    lg.x_grid_max = gg.x_zero + i_max * gg.delta;
    lg.y_grid_max = gg.y_zero + j_max * gg.delta;

    lg.x_size = static_cast<size_t>(i_max - i_min + 1);
    lg.y_size = static_cast<size_t>(j_max - j_min + 1);

    lg.grid.resize(lg.x_size * lg.y_size);

    int i_global = i_min;
    int j_global = j_min;

    for (size_t i = 0; i < (size_t)lg.x_size; ++i) {

        double* lg_ptr = &lg.grid[i * lg.y_size];
        const double* gg_ptr = &gg.grid[i_global * gg.y_size];

        for (size_t j = 0; j < (size_t)lg.y_size; ++j) {
            // lg.grid[index_local] = gg.grid[index_global];
            lg_ptr[j] = gg_ptr[j_global];

            j_global++;
        }
        j_global = j_min;
        i_global++;
        // std::cout << std::endl;
    }

    // for (size_t i = 0; i < lg.x_size; ++i) {
    //     for (size_t j = 0; j < lg.y_size; ++j) {
    //         printf("%ld %ld %f, ", i, j, lg.grid[i * lg.y_size + j]);
    //     }
    //     printf("\n");
    // }


    return true;
}

/**
 * @brief Creates a local grid from a global grid within a specified stripe.
 *
 * @param lg
 * @param gg
 * @param ds
 */
void make_local_grid_from_stripe(local_grid_type& lg,
                                 const global_grid_type& gg,
                                 const domains_stripe& ds) {
    double x_d_min = ds.x_min;
    double y_d_min = ds.y_min;
    double x_d_max = x_d_min + double(ds.nr_domains) * ds.side_x;
    double y_d_max = y_d_min + ds.side_y;

    get_local_grid(lg, gg, x_d_min, y_d_min, x_d_max, y_d_max);
}

/**
 * @brief Creates a domains_stripe object with the specified parameters.
 *
 * @param ds The domains_stripe object to be created.
 * @param x_min The minimum x-coordinate of the domains_stripe.
 * @param y_min The minimum y-coordinate of the domains_stripe.
 * @param nr_domains The number of domains in the stripe.
 * @param side_x The width of each domain in the x-direction.
 * @param side_y The height of the domain in the y-direction.
 * @return true if the domains_stripe object is successfully created, false otherwise.
 */
bool make_domains_stripe(domains_stripe& ds,
                         const double x_min,
                         const double y_min,
                         const size_t nr_domains,
                         const double side_x,
                         const double side_y) {
    ds.x_min = x_min;
    ds.y_min = y_min;
    ds.nr_domains = nr_domains;
    ds.side_x = side_x;
    ds.side_y = side_y;

    ds.x_max = x_min + nr_domains * side_x;
    ds.y_max = y_min + side_y;

    return true;
}

/**
 * @brief Creates a set of stripes from a global grid.
 *
 * This function creates a set of stripes from a given global grid. Each stripe is defined by its
 * minimum and maximum coordinates in the x and y directions. The number of stripes and the number
 * of domains per stripe are specified as input parameters. The size of each stripe is determined by
 * the side lengths in the x and y directions.
 *
 * @param stripes The output vector of domains_stripes representing the set of stripes.
 * @param gg The global grid from which the stripes are created.
 * @param nr_stripes The number of stripes to create.
 * @param nr_domains_per_stripe The number of domains per stripe.
 * @param side_x The side length in the x direction.
 * @param side_y The side length in the y direction.
 * @return True if the stripes are successfully created, false otherwise.
 */
bool make_stripes_set_from_global_grid(std::vector<domains_stripe>& stripes,
                                       const global_grid_type& gg,
                                       const double start_x,
                                       const double start_y,
                                       const size_t nr_stripes,
                                       const size_t nr_domains_per_stripe,
                                       const double side_x,
                                       const double side_y) {
    stripes.resize(nr_stripes);

    const double x_min = start_x;
    const double y_min = start_y;

    // check if the start coordinates are within the global grid
    if (x_min < gg.x_zero || y_min < gg.y_zero || x_min > gg.x_max || y_min > gg.y_max) {
        return false;
    }

    // const double x_max = gg.x_max;
    // const double y_max = gg.y_max;

    size_t i = 0, j = 0;

    for (size_t n = 0; n < nr_stripes; ++n) {
        double x_min_stripe = x_min + i * side_x * nr_domains_per_stripe;
        double y_min_stripe = y_min + j * side_y;

        double x_max_stripe = x_min_stripe + nr_domains_per_stripe * side_x;
        double y_max_stripe = y_min_stripe + side_y;

        i += 1;

        if (x_max_stripe > gg.x_max) {
            i = 0;
            j++;

            x_min_stripe = x_min + i * side_x * nr_domains_per_stripe;
            y_min_stripe = y_min + j * side_y;

            x_max_stripe = x_min_stripe + nr_domains_per_stripe * side_x;
            y_max_stripe = y_min_stripe + side_y;

            i += 1;
        }

        if (y_max_stripe > gg.y_max) {
            return false;
        }

        make_domains_stripe(stripes[n],  //
                            x_min_stripe,
                            y_min_stripe,
                            nr_domains_per_stripe,
                            side_x,
                            side_y);
    }

    return true;
}

/**
 * @brief Calculates the domain coordinates based on the given stripe and domain number.
 *
 * @param ds The domains_stripe object containing the stripe information.
 * @param domain_nr The domain number.
 * @param x_min The minimum x-coordinate of the domain (output parameter).
 * @param y_min The minimum y-coordinate of the domain (output parameter).
 * @param x_max The maximum x-coordinate of the domain (output parameter).
 * @param y_max The maximum y-coordinate of the domain (output parameter).
 */
inline void get_domain_from_stripe(const domains_stripe& ds,
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

/**
 * Prints the values of a local grid.
 *
 * @param lg The local grid to be printed.
 */
void print_local_grid(local_grid_type& lg) {
    for (size_t i = 0; i < lg.x_size; ++i) {
        for (size_t j = 0; j < lg.y_size; ++j) {
            std::cout << "X: " << lg.x_grid_min + i * lg.delta
                      << " Y: " << lg.y_grid_min + j * lg.delta << ": v:";
            // std::cout << "i: " << i << " j: " << j << " value: ";
            std::cout << lg.grid[i * lg.y_size + j] << " || ";
        }
        std::cout << std::endl;
    }
}

/**
 * @brief Prints the information of a local grid.
 *
 * This function prints the delta, minimum and maximum values of x and y coordinates,
 * minimum and maximum values of x and y grid indices, and the size of the local grid.
 *
 * @param lg The local grid to print the information of.
 */
void print_local_grid_info(const local_grid_type& lg) {
    int width = 13;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << std::setw(width) << "delta:" << std::setw(width) << lg.delta << std::endl;
    std::cout << std::setw(width) << "Nr of points:" << std::setw(width) << lg.x_size * lg.y_size
              << std::endl;
    std::cout << std::setw(width) << "x_d_min:" << std::setw(width) << lg.x_d_min
              << std::setw(width) << "y_d_min:" << std::setw(width) << lg.y_d_min << std::endl;
    std::cout << std::setw(width) << "x_d_max:" << std::setw(width) << lg.x_d_max
              << std::setw(width) << "y_d_max:" << std::setw(width) << lg.y_d_max << std::endl;
    std::cout << std::setw(width) << "x_grid_min:" << std::setw(width) << lg.x_grid_min
              << std::setw(width) << "y_grid_min:" << std::setw(width) << lg.y_grid_min
              << std::endl;
    std::cout << std::setw(width) << "x_grid_max:" << std::setw(width) << lg.x_grid_max
              << std::setw(width) << "y_grid_max:" << std::setw(width) << lg.y_grid_max
              << std::endl;
    std::cout << std::setw(width) << "x_size:" << std::setw(width) << lg.x_size << std::setw(width)
              << "y_size:" << std::setw(width) << lg.y_size << std::endl;
}

/**
 * @brief Generates a Monte Carlo quadrature rule.
 *
 * This function generates a Monte Carlo quadrature rule by randomly sampling points within a given
 * range. The quadrature rule is stored in the provided `quadrature_rule` object.
 *
 * @param qr The `quadrature_rule` object to store the generated quadrature rule.
 * @param n The number of points to sample.
 * @param x_min The minimum x-coordinate of the sampling range.
 * @param y_min The minimum y-coordinate of the sampling range.
 * @param x_max The maximum x-coordinate of the sampling range.
 * @param y_max The maximum y-coordinate of the sampling range.
 * @return `true` if the quadrature rule is successfully generated, `false` otherwise.
 */
bool make_MC_quadrature_rule(quadrature_rule& qr,
                             const size_t n,
                             const double x_min,
                             const double y_min,
                             const double x_max,
                             const double y_max) {
    qr.x_min = x_min;
    qr.y_min = y_min;
    qr.x_max = x_max;
    qr.y_max = y_max;

    qr.weights.resize(n);
    qr.x_nodes.resize(n);
    qr.y_nodes.resize(n);

    for (size_t i = 0; i < n; ++i) {
        qr.weights[i] = 1.0 / n;
        qr.x_nodes[i] = x_min + (x_max - x_min) * ((double)rand() / (double)RAND_MAX);
        qr.y_nodes[i] = y_min + (y_max - y_min) * ((double)rand() / (double)RAND_MAX);
    }

    return true;
}

/**
 * @brief Performs quadrature on a local grid using a given quadrature rule.
 *
 * This function calculates the quadrature of a local grid by evaluating the function values at the
 * quadrature nodes and applying the corresponding weights. The quadrature is performed within the
 * specified domain defined by the minimum and maximum values of x and y.
 *
 * @param lg The local grid to perform quadrature on.
 * @param qr The quadrature rule to use.
 * @param x_d_min The minimum x value of the domain.
 * @param y_d_min The minimum y value of the domain.
 * @param x_d_max The maximum x value of the domain.
 * @param y_d_max The maximum y value of the domain.
 * @return The result of the quadrature.
 */
double perform_quadrature_local(const local_grid_type& lg,  //
                                const quadrature_rule& qr,  //
                                const double x_d_min,       //
                                const double y_d_min,       //
                                const double x_d_max,       //
                                const double y_d_max) {     //
    //
    double result_Q = 0.0;

    const double volume = (x_d_max - x_d_min) * (y_d_max - y_d_min);

    for (size_t i = 0; i < qr.x_nodes.size(); ++i) {
        const double x_Q = (qr.x_nodes[i] + x_d_min) / (x_d_max - x_d_min);
        const double y_Q = (qr.y_nodes[i] + y_d_min) / (y_d_max - y_d_min);

        auto [i_local, j_local] = get_nearest_coordinates(lg.x_grid_min,  //
                                                          lg.y_grid_min,
                                                          lg.delta,
                                                          lg.delta,
                                                          x_Q,
                                                          y_Q,
                                                          false);

        const double f1 = lg.grid[i_local * lg.y_size + j_local];
        const double f2 = lg.grid[i_local * lg.y_size + j_local + 1];
        const double f3 = lg.grid[(i_local + 1) * lg.y_size + j_local];
        const double f4 = lg.grid[(i_local + 1) * lg.y_size + j_local + 1];

        // const double delta_x = lg.x_grid_max - lg.x_grid_min;
        // const double delta_y = lg.y_grid_max - lg.y_grid_min;

        const double x1 = lg.x_grid_min + i_local * lg.delta;
        const double x2 = lg.x_grid_min + (i_local + 1) * lg.delta;
        const double y1 = lg.y_grid_min + j_local * lg.delta;
        const double y2 = lg.y_grid_min + (j_local + 1) * lg.delta;

        // std::cout << "Volume: " << volume << std::endl;
        // std::cout << "x1: " << x1 << " x2: " << x2 << " y1: " << y1 << " y2: " << y2 <<
        // std::endl; std::cout << "f1: " << f1 << " f2: " << f2 << " f3: " << f3 << " f4: " << f4
        // << std::endl; std::cout << "x_Q: " << x_Q << " y_Q: " << y_Q << std::endl ;

        const double w11 = (x2 - x_Q) * (y2 - y_Q) / (lg.delta * lg.delta);
        const double w12 = (x2 - x_Q) * (y_Q - y1) / (lg.delta * lg.delta);
        const double w21 = (x_Q - x1) * (y2 - y_Q) / (lg.delta * lg.delta);
        const double w22 = (x_Q - x1) * (y_Q - y1) / (lg.delta * lg.delta);

        const double f_Q = w11 * f1 + w12 * f2 + w21 * f3 + w22 * f4;

        // std::cout << "w11: " << w11 << " w12: " << w12 << " w21: " << w21 << " w22: " << w22
        //           << std::endl;
        // std::cout << "f_Q: " << f_Q << std::endl << std::endl;

        result_Q += f_Q * qr.weights[i] * volume;
    }

    return result_Q;
}

/**
 * @brief Performs quadrature on local stripes.
 *
 * This function calculates the quadrature on local stripes using the provided
 * local grid, quadrature rule, and domains stripe. It updates the `Qs` array
 * with the calculated values.
 *
 * @param Qs The array to store the calculated quadrature values.
 * @param lg The local grid.
 * @param qr The quadrature rule.
 * @param ds The domains stripe.
 * @return True if the quadrature calculation is successful, false otherwise.
 */
bool perform_quadrature_local_stripe(std::valarray<double>& Qs,
                                     const local_grid_type& lg,
                                     const quadrature_rule& qr,
                                     const domains_stripe& ds) {
    //
    Qs.resize(ds.nr_domains, 0.0);
    Qs = 0.0;

    for (size_t i = 0; i < ds.nr_domains; ++i) {
        // std::cout << "Domain: " << i << std::endl;?

        double x_d_min, y_d_min, x_d_max, y_d_max;
        get_domain_from_stripe(ds, i, x_d_min, y_d_min, x_d_max, y_d_max);  // 4 * ds_nr_domains

        // std::cout.precision(6);

        // std::cout << "Domain: " << i << ", x_d_min: " << x_d_min << ", x_d_max: " << x_d_max
        //           << ", y_d_min: " << y_d_min << ", y_d_max: " << y_d_max << std::endl;

        const double volume = (x_d_max - x_d_min) * (y_d_max - y_d_min);  // 3 * ds_nr_domains

        double Qs_i = 0.0;

        const size_t qr_size = qr.x_nodes.size();

        for (size_t q_i = 0; q_i < qr_size; ++q_i) {
            //
            // 2 * 3 * qr_size * ds.nr_domains
            const double x_Q = (qr.x_nodes[q_i]) * (x_d_max - x_d_min) + x_d_min;
            const double y_Q = (qr.y_nodes[q_i]) * (y_d_max - y_d_min) + y_d_min;

            // std::cout << "x_Q: " << x_Q << " y_Q: " << y_Q << std::endl;

            int i_local, j_local;

            // dnr * qnr * 8
            get_nearest_coordinates_floor(lg.x_grid_min,  //
                                          lg.y_grid_min,  //
                                          lg.delta,
                                          lg.delta,
                                          x_Q,
                                          y_Q,
                                          i_local,
                                          j_local);

            // data trasfer 4 * 8 * qr_size * ds.nr_domains
            const double f1 = lg.grid[i_local * lg.y_size + j_local];
            const double f2 = lg.grid[i_local * lg.y_size + j_local + 1];
            const double f3 = lg.grid[(i_local + 1) * lg.y_size + j_local];
            const double f4 = lg.grid[(i_local + 1) * lg.y_size + j_local + 1];

            // if (q_i == 1)
            //             std::cout << "f1: " << f1 << ", f2: " << f2 << ", f3: " << f3 << ", f4: "
            //             << f4
            //                       << std::endl;

            // std::cout << "i_local: " << i_local << " j_local: " << j_local << std::endl;
            // std::cout << "f1: " << f1 << " f2: " << f2 << " f3: " << f3 << " f4: " << f4
            //           << std::endl;

            // std::cout << std::endl;

            // chack if qs is correct
            // if (x_Q < x_d_min || x_Q > x_d_max || y_Q < y_d_min || y_Q > y_d_max) {
            //     std::cout << "x_Q: " << x_Q << " y_Q: " << y_Q << std::endl;
            //     std::cout << "x_d_min: " << x_d_min << " x_d_max: " << x_d_max << std::endl;
            //     std::cout << "y_d_min: " << y_d_min << " y_d_max: " << y_d_max << std::endl;
            //     return false;
            // }

            const double x1 = lg.x_grid_min + i_local * lg.delta;  // 2 * qr_size * ds_nr_domains
            const double x2 =
                    lg.x_grid_min + (i_local + 1) * lg.delta;      // 3 * qr_size * ds_nr_domains
            const double y1 = lg.y_grid_min + j_local * lg.delta;  // 1 * qr_size * ds_nr_domains
            const double y2 =
                    lg.y_grid_min + (j_local + 1) * lg.delta;  // 3 * qr_size * ds_nr_domains

            // if (q_i == 1)
            // std::cout << "x1: " << x1 << ", x2: " << x2 << ", y1: " << y1 << ", y2: " << y2
            //           << std::endl;

            // 5 * 4 * qr_size * ds.nr_domains
            const double w11 = (x2 - x_Q) * (y2 - y_Q) / (lg.delta * lg.delta);
            const double w12 = (x2 - x_Q) * (y_Q - y1) / (lg.delta * lg.delta);
            const double w21 = (x_Q - x1) * (y2 - y_Q) / (lg.delta * lg.delta);
            const double w22 = (x_Q - x1) * (y_Q - y1) / (lg.delta * lg.delta);
            // std::cout << "w11: " << w11 << " w12: " << w12 << " w21: " << w21 << " w22: " << w22
            //           << std::endl;

            // if (q_i == 1)
            // std::cout << "w11: " << w11 << ", w12: " << w12 << ", w21: " << w21 << ", w22: " <<
            // w22
            //           << std::endl;

            // 7 * qr_size * ds.nr_domains
            const double f_Q = w11 * f1 + w12 * f2 + w21 * f3 + w22 * f4;

            // if (q_i == 1)
            // std::cout << "f_Q: " << f_Q << " " ;

            // if (q_i == 1)
            // std::cout << "volume: " << volume << std::endl;

            // 3 * qr_size * ds.nr_domains
            // data transfer 8 * qr_size * ds.nr_domains
            Qs_i += f_Q * qr.weights[q_i] * volume;

            // if (q_i == 2 and i == 25) {
            //     printf("w: %f, f_Q: %f, volume: %f, Qs_i: %f, q_i: %lu, domanin_nr: %lu\n",
            //            qr.weights[q_i],
            //            f_Q,
            //            volume,
            //            Qs_i,
            //            q_i,
            //            i);
            //     printf("f1: %f, f2: %f, f3: %f, f4: %f\n", f1, f2, f3, f4);
            //     printf("x1: %f, x2: %f, y1: %f, y2: %f\n", x1, x2, y1, y2);
            //     printf("w11: %f, w12: %f, w21: %f, w22: %f\n", w11, w12, w21, w22);
            //     printf("i_local: %d, j_local: %d\n", i_local, j_local);
            //     printf("lg.x_grid_min: %f, lg.y_grid_min: %f, lg.delta: %f\n", lg.x_grid_min, lg.y_grid_min, lg.delta); 
            //     printf("lg.x_grid_max: %f, lg.y_grid_max: %f\n", lg.x_grid_max, lg.y_grid_max);
            // printf("lg.x_size: %lu, lg.y_size: %lu\n", lg.x_size, lg.y_size);
            // }

            // std::cout << "---ll" << std::endl;
        }
        // std::cout << std::endl;

        // std::cout << "Qs_i: " << Qs_i << std::endl;

        Qs[i] = Qs_i;
    }

    return true;
}

/**
 * @brief Performs local quadratures on a set of stripes.
 *
 * This function calculates the local quadratures for each stripe in the given set of stripes.
 * It uses the provided global grid and quadrature rule to perform the calculations.
 * The results are stored in the output array Qs, where each element corresponds to a stripe.
 *
 * @param Qs The output array to store the results of the local quadratures.
 * @param stripes The set of stripes on which to perform the local quadratures.
 * @param gg The global grid used for the calculations.
 * @param qr The quadrature rule used for the calculations.
 * @return Returns true if the local quadratures were performed successfully, false otherwise.
 */
bool perform_local_quadratures_stripe_set(std::valarray<double>& Qs,
                                          const std::vector<domains_stripe>& stripes,
                                          const global_grid_type& gg,
                                          const quadrature_rule& qr) {
    const size_t nr_stripes = stripes.size();
    Qs.resize(nr_stripes, 0.0);

    double num_nodes_per_stripe = 0.0;

    for (size_t i = 0; i < nr_stripes; ++i) {
        local_grid_type lg_stripe;
        make_local_grid_from_stripe(lg_stripe, gg, stripes[i]);

        num_nodes_per_stripe += lg_stripe.x_size * lg_stripe.y_size;

        std::valarray<double> Qs_stripe;
        perform_quadrature_local_stripe(Qs_stripe, lg_stripe, qr, stripes[i]);

        Qs[i] = Qs_stripe.sum();
    }

    num_nodes_per_stripe /= double(nr_stripes);

    std::cout << "Number of nodes per stripe: " << num_nodes_per_stripe << std::endl;
    std::cout << "Number of nodes per domain: "
              << num_nodes_per_stripe / double(stripes[0].nr_domains) << std::endl;

    std::cout << std::endl;

    return true;
}

// std::mutex mtx;

/**
 * @brief Performs local quadratures on a range of stripes.
 *
 * This function calculates the local quadratures for a range of stripes within a global grid.
 * It iterates over the specified range of stripes, creates a local grid for each stripe,
 * performs the quadrature calculation, and stores the results in the output array `Qs`.
 *
 * @param Qs The output array to store the results of the quadrature calculations.
 * @param start_index The starting index of the range of stripes to process.
 * @param end_index The ending index (exclusive) of the range of stripes to process.
 * @param stripes The vector of stripes containing the domain information.
 * @param gg The global grid type.
 * @param qr The quadrature rule to use for the calculations.
 * @return Returns true if the quadratures were performed successfully, false otherwise.
 */
bool perform_local_quadratures_stripe_set_range(std::valarray<double>& Qs,
                                                const size_t start_index,  //
                                                const size_t end_index,    //
                                                const std::vector<domains_stripe>& stripes,
                                                const global_grid_type& gg,
                                                const quadrature_rule& qr) {
    //
    const size_t nr_stripes = end_index - start_index;
    Qs.resize(nr_stripes, 0.0);

    // mtx.lock();
    // std::cout << "Number of stripes: " << nr_stripes << std::endl;
    // std::cout << "Start index: " << start_index << std::endl;
    // std::cout << "End index: " << end_index << std::endl;
    // std::cout << std::endl;
    // mtx.unlock();

    // double num_nodes_per_stripe = 0.0;

    for (size_t i = start_index; i < end_index; ++i) {
        local_grid_type lg_stripe;
        make_local_grid_from_stripe(lg_stripe, gg, stripes[i]);

        // num_nodes_per_stripe += lg_stripe.x_size * lg_stripe.y_size;

        std::valarray<double> Qs_stripe;
        perform_quadrature_local_stripe(Qs_stripe, lg_stripe, qr, stripes[i]);

        Qs[i - start_index] = Qs_stripe.sum();
    }

    // num_nodes_per_stripe /= double(nr_stripes);

    // std::cout << "Number of nodes per stripe: " << num_nodes_per_stripe << std::endl;
    // std::cout << "Number of nodes per domain: "
    //           << num_nodes_per_stripe / double(stripes[0].nr_domains) << std::endl;

    // std::cout << std::endl;

    return true;
}

/**
 * @brief Performs quadrature on a global stripe of domains.
 *
 * This function calculates the quadrature of a global stripe of domains using the provided
 * global grid, quadrature rule, and domain stripe. The result is stored in the `Qs` array.
 *
 * @param Qs The array to store the quadrature results.
 * @param gg The global grid.
 * @param qr The quadrature rule.
 * @param ds The domain stripe.
 * @return True if the quadrature was successful, false otherwise.
 */
bool perform_quadrature_global_stripe(std::valarray<double>& Qs,
                                      const global_grid_type& gg,
                                      const quadrature_rule& qr,
                                      const domains_stripe& ds) {
    Qs.resize(ds.nr_domains, 0.0);
    Qs = 0.0;

    for (size_t i = 0; i < ds.nr_domains; ++i) {
        double x_d_min, y_d_min, x_d_max, y_d_max;
        get_domain_from_stripe(ds, i, x_d_min, y_d_min, x_d_max, y_d_max);

        const double volume = (x_d_max - x_d_min) * (y_d_max - y_d_min);  // 3 * ds_nr_domains

        // std::cout << "Domain: " << i << ", x_d_min: " << x_d_min << ", x_d_max: " << x_d_max
        //           << ", y_d_min: " << y_d_min << ", y_d_max: " << y_d_max << std::endl;

        double Qs_i = 0.0;

        const size_t qr_size = qr.x_nodes.size();

        for (size_t q_i = 0; q_i < qr_size; ++q_i) {
            //
            // 2 * 3 * qr_size * ds.nr_domains
            const double x_Q = (qr.x_nodes[q_i]) * (x_d_max - x_d_min) + x_d_min;
            const double y_Q = (qr.y_nodes[q_i]) * (y_d_max - y_d_min) + y_d_min;

            // std::cout << "x_Q: " << x_Q << " y_Q: " << y_Q << std::endl;

            int i_local, j_local;

            get_nearest_coordinates_floor(gg.x_zero,  //
                                          gg.y_zero,  //
                                          gg.delta,
                                          gg.delta,
                                          x_Q,
                                          y_Q,
                                          i_local,
                                          j_local);

            // data trasfer 4 * 8 * qr_size * dsnr_domains
            const double f1 = gg.grid[i_local * gg.x_size + j_local];
            const double f2 = gg.grid[i_local * gg.y_size + j_local + 1];
            const double f3 = gg.grid[(i_local + 1) * gg.y_size + j_local];
            const double f4 = gg.grid[(i_local + 1) * gg.y_size + j_local + 1];

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
            Qs_i += f_Q * qr.weights[q_i] * volume;

            // std::cout << "---gg" << std::endl;
        }

        Qs[i] = Qs_i;
    }

    return true;
}

/**
 * @brief Performs global quadratures on a set of stripes.
 *
 * @param Qs Resulting quadrature values.
 * @param stripes Set of stripes.
 * @param gg Global grid.
 * @param qr Quadrature rule.
 * @return true if the quadratures were performed successfully, false otherwise.
 */
bool perform_global_quadratures_stripe_set(std::valarray<double>& Qs,
                                           const std::vector<domains_stripe>& stripes,
                                           const global_grid_type& gg,
                                           const quadrature_rule& qr) {
    const size_t nr_stripes = stripes.size();
    Qs.resize(nr_stripes, 0.0);

    // double num_nodes_per_stripe = 0.0;

    for (size_t i = 0; i < nr_stripes; ++i) {
        std::valarray<double> Qs_stripe;

        perform_quadrature_global_stripe(Qs_stripe, gg, qr, stripes[i]);

        Qs[i] = Qs_stripe.sum();
    }

    return true;
}

/**
 * Performs global quadratures on a range of stripes and stores the results in a valarray.
 *
 * @param Qs The valarray to store the results of the quadratures.
 * @param start_index The starting index of the range of stripes.
 * @param end_index The ending index of the range of stripes.
 * @param stripes The vector of domains stripes to perform quadratures on.
 * @param gg The global grid type.
 * @param qr The quadrature rule to use for the quadratures.
 * @return True if the quadratures were performed successfully, false otherwise.
 */
bool perform_global_quadratures_stripe_set_range(std::valarray<double>& Qs,
                                                 const size_t start_index,
                                                 const size_t end_index,
                                                 const std::vector<domains_stripe>& stripes,
                                                 const global_grid_type& gg,
                                                 const quadrature_rule& qr) {
    const size_t nr_stripes = end_index - start_index;
    Qs.resize(nr_stripes, 0.0);

    for (size_t i = start_index; i < end_index; ++i) {
        std::valarray<double> Qs_stripe;

        perform_quadrature_global_stripe(Qs_stripe, gg, qr, stripes[i]);

        Qs[i - start_index] = Qs_stripe.sum();
    }

    return true;
}

struct problem_parameters {
    size_t quad_nodes_nr;
    double xy_max_domain;
    double xy_zero_domain;
    double delta_domain;
    unsigned int nr_stripes;
    int random_seed;
    int nr_domains_per_stripe;
    double side_x_stripe;
    double side_y_stripe;
    double start_x_stripes;
    double start_y_stripes;
    std::function<double(double, double)> fun;
};

/**
 * @brief Builds the problem based on the given parameters.
 *
 * @param gg
 * @param stripes
 * @param qr
 * @param prs
 */
int build_problem(global_grid_type& gg,
                  std::vector<domains_stripe>& stripes,
                  quadrature_rule& qr,
                  problem_parameters prs) {
    //
    srand(prs.random_seed);
    const size_t quad_nodes_nr = prs.quad_nodes_nr;
    const double x_min_QMC = 0.0;
    const double y_min_QMC = 0.0;
    const double x_max_QMC = 1.0;
    const double y_max_QMC = 1.0;

    make_MC_quadrature_rule(qr, quad_nodes_nr, x_min_QMC, y_min_QMC, x_max_QMC, y_max_QMC);

    // auto f = [](double x, double y) { return std::sin(x) + std::log(1.0 + (y + x) * 0.0000001);
    // };

    const double delta = prs.delta_domain;
    const double x_zero = prs.xy_zero_domain;
    const double y_zero = prs.xy_zero_domain;
    const double x_max = prs.xy_max_domain;
    const double y_max = prs.xy_max_domain;

    make_gloal_grid(gg, delta, x_zero, y_zero, x_max, y_max, prs.fun);

    const double gg_bytes = global_grid_bytes(gg);

    std::cout << std::endl;
    std::cout << "Global grid size: " << double(gg_bytes) / (1024.0 * 1024.0 * 1024.0) << " GBytes"
              << std::endl
              << std::endl;

    stripes.clear();
    const size_t nr_stripes = prs.nr_stripes;
    const size_t nr_domains_per_stripe = prs.nr_domains_per_stripe;
    const double start_x = prs.start_x_stripes;
    const double start_y = prs.start_y_stripes;
    const double side_x_stripe = prs.side_x_stripe;
    const double side_y_stripe = prs.side_y_stripe;

    const bool flag = make_stripes_set_from_global_grid(stripes,  //
                                                        gg,
                                                        start_x,
                                                        start_y,
                                                        nr_stripes,
                                                        nr_domains_per_stripe,
                                                        side_x_stripe,
                                                        side_y_stripe);

    std::cout << "Nr of stripes: " << stripes.size() << std::endl;
    std::cout << "Nr of quadrature nodes: " << quad_nodes_nr << std::endl;
    std::cout << std::endl;

    if (!flag) {
        std::cout << "Failed to create stripes" << std::endl;
        return 1;
    }

    const double volume_domain = side_x_stripe * side_y_stripe;
    const int nodes_per_stripe = int(volume_domain / (delta * delta));

    std::cout << "Volume of a domain: " << volume_domain << std::endl;
    std::cout << "Nodes per stripe:   " << nodes_per_stripe << std::endl;
    std::cout << std::endl;

    return 0;
}

/**
 * @brief Test function for the stripes.
 *
 * This function tests the creation of stripes and the quadrature calculation on local stripes.
 *
 * @param argc
 * @param argv
 * @return int 0 if the test is successful, 1 otherwise.
 */
int test_stripes(int argc,
                 char* argv[],
                 global_grid_type& gg,
                 std::vector<domains_stripe>& stripes,
                 quadrature_rule& qr) {
    //
    double Qg = 0.0;
    double Ql = 0.0;

    std::cout << std::fixed << std::setprecision(6);

    {
        std::cout << "++ Local quadratures:" << std::endl;
        std::cout << "-------------------------------------------------------" << std::endl
                  << std::endl;

        auto start = std::chrono::high_resolution_clock::now();

        std::valarray<double> Qs_stripes;
        perform_local_quadratures_stripe_set(Qs_stripes, stripes, gg, qr);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        std::cout.precision(16);
        std::cout << "Qs_stripes: " << Qs_stripes.sum() << std::endl;
        std::cout.precision(3);  // Set precision back to default
        Ql = Qs_stripes.sum();
        // for (size_t i = 0; i < Qs_stripes.size(); ++i) {
        //     std::cout << Qs_stripes[i] << " ";
        // }

        // for (size_t i = 0; i < Qs_stripes.size(); ++i) {
        //     std::cout << Qs_stripes[i] << " " << std::endl;
        // }

        const double seconds = double(duration.count()) / 1000000.0;

        std::cout << "Execution time: " << seconds << " seconds" << std::endl;

        const double nr_stripes = stripes.size();
        const double nr_domains_per_stripe = stripes[0].nr_domains;
        const double quad_nodes_nr = qr.x_nodes.size();

        const double tot_flop = nr_stripes * (7.0 * nr_domains_per_stripe +
                                              51.0 * nr_domains_per_stripe * quad_nodes_nr);
        const double flops = tot_flop / seconds;

        const double data_transfer =
                8.0 * nr_stripes * nr_domains_per_stripe * quad_nodes_nr * (4 + 1);

        std::cout << std::endl;

        std::cout << "Total GFLOP: " << flops / (1024.0 * 1024.0 * 1024.0) << std::endl;
        std::cout << "Total data transfer: " << data_transfer / (1024.0 * 1024.0 * 1024.0) << " GB"
                  << std::endl;

        std::cout << std::endl;
        std::cout << std::endl;
    }

    return 0;

    {
        std::cout << "-------------------------------------------------------" << std::endl
                  << std::endl;
        std::cout << "++ Global quadratures:" << std::endl;
        std::cout << "-------------------------------------------------------" << std::endl
                  << std::endl;

        auto start = std::chrono::high_resolution_clock::now();

        std::valarray<double> Qs_stripes;
        perform_global_quadratures_stripe_set(Qs_stripes, stripes, gg, qr);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        std::cout.precision(16);
        std::cout << "Qs_stripes: " << Qs_stripes.sum() << std::endl;
        std::cout.precision(6);  // Set precision back to default
        Qg = Qs_stripes.sum();

        const double seconds = double(duration.count()) / 1000000.0;

        std::cout << "Execution time: " << seconds << " seconds" << std::endl;

        const double nr_stripes = stripes.size();
        const double nr_domains_per_stripe = stripes[0].nr_domains;
        const double quad_nodes_nr = qr.x_nodes.size();

        const double tot_flop = nr_stripes * (7.0 * nr_domains_per_stripe +
                                              51.0 * nr_domains_per_stripe * quad_nodes_nr);

        const double flops = tot_flop / seconds;

        const double data_transfer =
                8.0 * nr_stripes * nr_domains_per_stripe * quad_nodes_nr * (4 + 1);

        std::cout << std::endl;

        std::cout << "Total GFLOP: " << flops / (1024.0 * 1024.0 * 1024.0) << std::endl;
        std::cout << "Total data transfer: " << data_transfer / (1024.0 * 1024.0 * 1024.0) << " GB"
                  << std::endl;
    }

    std::cout << std::endl;

    std::cout << "Ql: " << Ql << std::endl;
    std::cout << "Qg: " << Qg << std::endl;

    std::cout << "Error: " << std::scientific << std::abs(Ql - Qg) / std::abs(Ql) << std::endl;

    std::cout << std::endl;

    return 0;
}

/**
 * @brief Test function for the stripes using multiple threads.
 *
 * @param argc The number of command-line arguments.
 * @param argv The command-line arguments.
 * @return true if the test is successful, false otherwise.
 * @return false if the test is unsuccessful.
 */
bool test_stripes_mt(int argc,
                     char* argv[],
                     const unsigned int nr_threads,
                     global_grid_type& gg,
                     std::vector<domains_stripe>& stripes,
                     quadrature_rule& qr) {
    // //

    double Qg = 0.0;
    double Ql = 0.0;

    const unsigned int nr_stripes = stripes.size();
    const double nr_domains_per_stripe = stripes[0].nr_domains;
    const double quad_nodes_nr = qr.x_nodes.size();

    std::cout << std::fixed << std::setprecision(6);

    {
        std::cout << "++ Local quadratures:" << std::endl;
        std::cout << "-------------------------------------------------------" << std::endl
                  << std::endl;

        // std::valarray<double> Qs_stripes;

        std::vector<std::thread> threads(nr_threads);
        const size_t nr_stripes_per_thread = nr_stripes / nr_threads;
        const size_t tail_threads = nr_stripes % nr_threads;

        size_t start_index = 0;
        size_t end_index = nr_stripes_per_thread;

        std::valarray<double> Qs_threads;
        Qs_threads.resize(nr_threads, 0.0);

        auto start = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < nr_threads; ++i) {
            if (i == nr_threads - 1) {
                end_index += tail_threads;
            }

            threads[i] = std::thread([thread_index = i,  //
                                      &Qs_threads,       //
                                      &stripes,          //
                                      &gg,
                                      &qr,
                                      start_index,
                                      end_index]() {
                //
                std::valarray<double> Qs_stripes;
                perform_local_quadratures_stripe_set_range(Qs_stripes,  //
                                                           start_index,
                                                           end_index,
                                                           stripes,
                                                           gg,
                                                           qr);
                // std::cout << "Thread: " << thread_index << " Qs_stripes: " << Qs_stripes.sum()
                //           << std::endl;
                // std::cout << "Qs thread size: "  << Qs_threads.size() << std::endl;
                Qs_threads[thread_index] = Qs_stripes.sum();
                //
            });

            start_index = end_index;
            end_index += nr_stripes_per_thread;
        }

        for (size_t i = 0; i < nr_threads; ++i) {
            threads[i].join();
        }

        Ql = Qs_threads.sum();

        // std::cout << "Ql threads: " << Ql << std::endl;

        // perform_local_quadratures_stripe_set(Qs_stripes, stripes, gg, qr);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        std::cout.precision(16);
        std::cout << "Ql threads: " << Ql << std::endl;
        std::cout.precision(3);  // Set precision back to default

        const double seconds = double(duration.count()) / 1000000.0;

        std::cout << "Execution time: " << seconds << " seconds" << std::endl;

        const double tot_flop = nr_stripes * (7.0 * nr_domains_per_stripe +
                                              51.0 * nr_domains_per_stripe * quad_nodes_nr);
        const double flops = tot_flop / seconds;

        const double data_transfer =
                8.0 * nr_stripes * nr_domains_per_stripe * quad_nodes_nr * (4 + 2);

        std::cout << std::endl;

        std::cout << "Total GFLOP: " << flops / (1024.0 * 1024.0 * 1024.0) << std::endl;
        std::cout << "Total data transfer: " << data_transfer / (1024.0 * 1024.0 * 1024.0) << " GB"
                  << std::endl;

        std::cout << std::endl;
        std::cout << std::endl;
    }

    {
        std::cout << "-------------------------------------------------------" << std::endl;
        std::cout << "++ Global quadratures:" << std::endl;
        std::cout << "-------------------------------------------------------" << std::endl
                  << std::endl;

        std::vector<std::thread> threads(nr_threads);
        const size_t nr_stripes_per_thread = nr_stripes / nr_threads;
        const size_t tail_threads = nr_stripes % nr_threads;

        size_t start_index = 0;
        size_t end_index = nr_stripes_per_thread;

        std::valarray<double> Qs_threads;
        Qs_threads.resize(nr_threads, 0.0);

        auto start = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < nr_threads; ++i) {
            if (i == nr_threads - 1) {
                end_index += tail_threads;
            }

            threads[i] = std::thread([thread_index = i,  //
                                      &Qs_threads,       //
                                      &stripes,          //
                                      &gg,
                                      &qr,
                                      start_index,
                                      end_index]() {
                //
                std::valarray<double> Qs_stripes;
                perform_global_quadratures_stripe_set_range(Qs_stripes,  //
                                                            start_index,
                                                            end_index,
                                                            stripes,
                                                            gg,
                                                            qr);
                // std::cout << "Thread: " << thread_index << " Qs_stripes: " << Qs_stripes.sum()
                //           << std::endl;
                // std::cout << "Qs thread size: "  << Qs_threads.size() << std::endl;
                Qs_threads[thread_index] = Qs_stripes.sum();
                //
            });

            start_index = end_index;
            end_index += nr_stripes_per_thread;
        }

        for (size_t i = 0; i < nr_threads; ++i) {
            threads[i].join();
        }

        Qg = Qs_threads.sum();

        // std::cout << "Ql threads: " << Ql << std::endl;

        // perform_local_quadratures_stripe_set(Qs_stripes, stripes, gg, qr);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        std::cout.precision(16);
        std::cout << "Qg threads: " << Ql << std::endl;
        std::cout.precision(3);  // Set precision back to default

        const double seconds = double(duration.count()) / 1000000.0;

        std::cout << "Global Execution time: " << seconds << " seconds" << std::endl;

        const double tot_flop = nr_stripes * (7.0 * nr_domains_per_stripe +
                                              51.0 * nr_domains_per_stripe * quad_nodes_nr);
        const double flops = tot_flop / seconds;

        const double data_transfer =
                8.0 * nr_stripes * nr_domains_per_stripe * quad_nodes_nr * (4 + 1);

        std::cout << std::endl;

        std::cout << "Global Total GFLOP: " << flops / (1024.0 * 1024.0 * 1024.0) << std::endl;
        std::cout << "Global Total data transfer: " << data_transfer / (1024.0 * 1024.0 * 1024.0)
                  << " GB" << std::endl;

        std::cout << std::endl;
        std::cout << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Ql: " << Ql << std::endl;
    std::cout << "Qg: " << Qg << std::endl;
    std::cout << "Error: " << std::scientific << std::abs(Ql - Qg) / std::abs(Ql) << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << std::endl;

    return true;
}

extern "C" int test_grid_cuda(global_grid_type& gg,  //
                              quadrature_rule& qr,   //
                              std::vector<domains_stripe>& ds_vector);

/**
 * @brief
 *
 * @param argc
 * @param argv
 * @return int
 */
int main(int argc, char* argv[]) {
    // test_grid_cuda();
    // return 0;

    global_grid_type gg;
    std::vector<domains_stripe> stripes;
    quadrature_rule qr;

    problem_parameters prs = {
            .quad_nodes_nr = 120,
            .xy_max_domain = 1000.0,
            .xy_zero_domain = 0.0,
            .delta_domain = 0.06,
            .nr_stripes = 22020,
            .random_seed = 0,
            .nr_domains_per_stripe = 32,
            .side_x_stripe = 0.52,
            .side_y_stripe = 0.52,
            .start_x_stripes = 0.111,
            .start_y_stripes = 0.122,
            .fun = [](double x,
                      double y) { return std::sin(x) + std::log(1.0 + (y + x) * 0.0000001); },
    };

    std::cout << "-------------------------------------------------------" << std::endl
              << "-------------------------------------------------------" << std::endl;
    std::cout << "Building problem" << std::endl;
    std::cout << "-------------------------------------------------------" << std::endl
              << "-------------------------------------------------------" << std::endl;

    build_problem(gg, stripes, qr, prs);

    // size_t quad_nodes_nr = 130;
    size_t nr_threads = 18;

    int a = 0, b = 0, c = 0;

    std::cout << "-------------------------------------------------------" << std::endl
              << "-------------------------------------------------------" << std::endl
              << "-------------------------------------------------------" << std::endl;

    c = test_grid_cuda(gg, qr, stripes);

    // const double xy_max = 3500.0;
    // const unsigned int nr_stripes = 620200;

    a = test_stripes(argc, argv, gg, stripes, qr);

    std::cout << "-------------------------------------------------------" << std::endl
              << "-------------------------------------------------------" << std::endl
              << "-------------------------------------------------------" << std::endl;

    return a && b && c;

    b = test_stripes_mt(argc, argv, nr_threads, gg, stripes, qr);

    return b && a && c;

    // global_grid_type gg;

    // auto f = [](double x, double y) { return x * x + y * y; };

    // const double delta = 0.1;
    // const double x_zero = 0.0;
    // const double y_zero = 0.0;
    // const double x_max = 100.0;
    // const double y_max = 100.0;

    // make_gloal_grid(gg, delta, x_zero, y_zero, x_max, y_max, f);

    // local_grid_type lg;

    // get_local_grid(lg, gg, 1.0, 1.0, 2.0, 2.0);

    // print_local_grid_info(lg);

    // // print_local_grid(lg);

    // quadrature_rule qr;

    // srand(time(NULL));

    // const size_t quad_nodes_nr = 5;
    // const double x_min_QMC = 0.0;
    // const double y_min_QMC = 0.0;
    // const double x_max_QMC = 1.0;
    // const double y_max_QMC = 1.0;

    // make_MC_quadrature_rule(qr, quad_nodes_nr, x_min_QMC, y_min_QMC, x_max_QMC, y_max_QMC);

    // // std::cout << "Quadrature rule:" << std::endl;
    // // for (size_t i = 0; i < quad_nodes_nr; ++i) {
    // //     std::cout << "X: " << qr.x_nodes[i] << " Y: " << qr.y_nodes[i] << " W: " <<
    // // qr.weights[i]
    // //               << std::endl;
    // // }

    // auto start = std::chrono::high_resolution_clock::now();

    // const double Q = perform_quadrature_local(lg, qr, 1.0, 1.0, 2.0, 2.0);

    // auto end = std::chrono::high_resolution_clock::now();

    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // std::cout << std::endl;
    // std::cout << "Execution time: " << (duration.count() / 1000.0) << " milliseconds" <<
    // std::endl; std::cout << std::fixed << std::setprecision(6); std::cout << "Q: " << Q <<
    // std::endl;

    // std::cout << "-------------------------------------------------------" << std::endl
    //           << std::endl;

    // std::cout << std::setprecision(6);

    // domains_stripe ds;
    // const double x_min_stripe = 0.511;
    // const double y_min_stripe = 0.522;
    // const size_t nr_domains = 2;
    // const double side_x_stripe = 0.52;
    // const double side_y_stripe = 0.52;

    // make_domains_stripe(ds, x_min_stripe, y_min_stripe, nr_domains, side_x_stripe,
    // side_y_stripe);

    // local_grid_type lg_stripe;
    // make_local_grid_from_stripe(lg_stripe, gg, ds);

    // print_local_grid_info(lg_stripe);

    // std::valarray<double> Qs;
    // perform_quadrature_local_stripe(Qs, lg_stripe, qr, ds);

    // std::cout << std::endl;
    // std::cout << std::setprecision(16);
    // std::cout << "Qs: " << Qs.sum() << std::endl;
    // std::cout << std::setprecision(6);
    // for (size_t i = 0; i < Qs.size(); ++i) {
    //     std::cout << Qs[i] << " ";
    // }

    // std::cout << std::endl;
    // std::cout << std::endl;

    // perform_quadrature_global_stripe(Qs, gg, qr, ds);

    // std::cout << std::endl;
    // std::cout << "Qs: " << Qs.sum() << std::endl;
    // for (size_t i = 0; i < Qs.size(); ++i) {
    //     std::cout << Qs[i] << " ";
    // }

    return 0;
}