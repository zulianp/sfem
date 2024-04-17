#include <chrono>
#include <cmath>
#include <fstream>  // Include the necessary header file
#include <functional>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <valarray>
#include <vector>

/**
 * @brief Represents a global grid with specified dimensions and properties.
 */
struct global_grid_type {
    std::valarray<double> grid; /**< The grid values. */

    double delta; /**< The grid spacing. */

    double x_zero; /**< The x-coordinate of the grid origin. */
    double y_zero; /**< The y-coordinate of the grid origin. */

    double x_max; /**< The maximum x-coordinate of the grid. */
    double y_max; /**< The maximum y-coordinate of the grid. */

    size_t x_size; /**< The number of grid points in the x-direction. */
    size_t y_size; /**< The number of grid points in the y-direction. */
};

/**
 * @brief Represents a local grid with specified dimensions and properties.
 */
struct local_grid_type {
    std::valarray<double> grid; /**< The grid values. */

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
 * @brief Represents a stripe of domains in a grid.
 *
 * This struct defines the boundaries and properties of a stripe of domains in a grid.
 * It includes the minimum and maximum values of the x and y coordinates, the number of domains,
 * and the side lengths of each domain.
 */
struct domains_stripe {
    double x_min; /**< The minimum x-coordinate of the stripe. */
    double y_min; /**< The minimum y-coordinate of the stripe. */
    double x_max; /**< The maximum x-coordinate of the stripe. */
    double y_max; /**< The maximum y-coordinate of the stripe. */

    size_t nr_domains; /**< The number of domains in the stripe. */
    double side_x;     /**< The side length of a single domain in the x-direction. */
    double side_y;     /**< The side length of a single domain in the y-direction. */
};

struct quadrature_rule {
    std::valarray<double> weights; /**< The quadrature weights. */
    std::valarray<double> x_nodes; /**< The x-coordinates of the quadrature nodes. */
    std::valarray<double> y_nodes; /**< The y-coordinates of the quadrature nodes. */

    double x_min; /**< The minimum x-coordinate of the domain. */
    double y_min; /**< The minimum y-coordinate of the domain. */
    double x_max; /**< The maximum x-coordinate of the domain. */
    double y_max; /**< The maximum y-coordinate of the domain. */
};

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
    auto [i_min, j_min] = get_nearest_coordinates(
            gg.x_zero, gg.y_zero, gg.delta, gg.delta, x_d_min, y_d_min, false);

    // get x y max coordinates from global grid
    auto [i_max, j_max] = get_nearest_coordinates(
            gg.x_zero, gg.y_zero, gg.delta, gg.delta, x_d_max, y_d_max, true);

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
        for (size_t j = 0; j < (size_t)lg.y_size; ++j) {
            const size_t index_global = i_global * gg.y_size + j_global;
            const size_t index_local = i * lg.y_size + j;

            // if (index_global >= gg.grid.size()) {
            //     std::cout << "index_global: " << index_global << " gg.grid.size(): " <<
            //     gg.grid.size()
            //               << std::endl;
            //     return false;
            // }

            // if (index_local >= lg.grid.size()) {
            //     std::cout << "index_local: " << index_local << " lg.grid.size(): " <<
            //     lg.grid.size()
            //               << std::endl;
            //     return false;
            // }

            lg.grid[index_local] = gg.grid[index_global];
            // std::cout << "i: " << i << " j: " << j << " value: " << gg.grid[i_global * gg.y_size
            // + j_global]
            //           << std::endl;
            j_global++;
        }
        j_global = j_min;
        i_global++;
        // std::cout << std::endl;
    }

    return true;
}

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
        get_domain_from_stripe(ds, i, x_d_min, y_d_min, x_d_max, y_d_max);

        // std::cout << "Domain: " << i << ", x_d_min: " << x_d_min << ", x_d_max: " << x_d_max
        //           << ", y_d_min: " << y_d_min << ", y_d_max: " << y_d_max << std::endl;

        const double volume = (x_d_max - x_d_min) * (y_d_max - y_d_min);

        for (size_t q_i = 0; q_i < qr.x_nodes.size(); ++q_i) {
            const double x_Q = (qr.x_nodes[q_i]) * (x_d_max - x_d_min) + x_d_min;
            const double y_Q = (qr.y_nodes[q_i]) * (y_d_max - y_d_min) + y_d_min;

            // std::cout << "x_Q: " << x_Q << " y_Q: " << y_Q << std::endl;

            auto [i_local, j_local] = get_nearest_coordinates(lg.x_d_min,  //
                                                              lg.y_d_min,  //
                                                              lg.delta,
                                                              lg.delta,
                                                              x_Q,
                                                              y_Q,
                                                              false);

            const double f1 = lg.grid[i_local * lg.y_size + j_local];
            const double f2 = lg.grid[i_local * lg.y_size + j_local + 1];
            const double f3 = lg.grid[(i_local + 1) * lg.y_size + j_local];
            const double f4 = lg.grid[(i_local + 1) * lg.y_size + j_local + 1];

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

            const double x1 = lg.x_d_min + i_local * lg.delta;
            const double x2 = lg.x_d_min + (i_local + 1) * lg.delta;
            const double y1 = lg.y_d_min + j_local * lg.delta;
            const double y2 = lg.y_d_min + (j_local + 1) * lg.delta;

            const double w11 = (x2 - x_Q) * (y2 - y_Q) / (lg.delta * lg.delta);
            const double w12 = (x2 - x_Q) * (y_Q - y1) / (lg.delta * lg.delta);
            const double w21 = (x_Q - x1) * (y2 - y_Q) / (lg.delta * lg.delta);
            const double w22 = (x_Q - x1) * (y_Q - y1) / (lg.delta * lg.delta);

            const double f_Q = w11 * f1 + w12 * f2 + w21 * f3 + w22 * f4;

            Qs[i] += f_Q * qr.weights[q_i] * volume;
        }
    }

    return true;
}

/**
 * @brief
 *
 * @param argc
 * @param argv
 * @return int
 */
int main(int argc, char* argv[]) {
    global_grid_type gg;

    auto f = [](double x, double y) { return x * x + y * y; };

    const double delta = 0.1;
    const double x_zero = 0.0;
    const double y_zero = 0.0;
    const double x_max = 10.0;
    const double y_max = 10.0;

    make_gloal_grid(gg, delta, x_zero, y_zero, x_max, y_max, f);

    local_grid_type lg;

    get_local_grid(lg, gg, 1.0, 1.0, 2.0, 2.0);

    print_local_grid_info(lg);

    // print_local_grid(lg);

    quadrature_rule qr;

    srand(time(NULL));

    const size_t quad_nodes_nr = 10;
    const double x_min_QMC = 0.0;
    const double y_min_QMC = 0.0;
    const double x_max_QMC = 1.0;
    const double y_max_QMC = 1.0;
    make_MC_quadrature_rule(qr, quad_nodes_nr, x_min_QMC, y_min_QMC, x_max_QMC, y_max_QMC);

    // std::cout << "Quadrature rule:" << std::endl;
    // for (size_t i = 0; i < quad_nodes_nr; ++i) {
    //     std::cout << "X: " << qr.x_nodes[i] << " Y: " << qr.y_nodes[i] << " W: " << qr.weights[i]
    //               << std::endl;
    // }

    auto start = std::chrono::high_resolution_clock::now();

    const double Q = perform_quadrature_local(lg, qr, 1.0, 1.0, 2.0, 2.0);

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << std::endl;
    std::cout << "Execution time: " << (duration.count() / 1000.0) << " milliseconds" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Q: " << Q << std::endl;

    {  // Test stripe
        domains_stripe ds;
        const double x_min_stripe = 0.511;
        const double y_min_stripe = 0.522;
        const size_t nr_domains = 10;
        const double side_x_stripe = 0.52;
        const double side_y_stripe = 0.52;

        make_domains_stripe(
                ds, x_min_stripe, y_min_stripe, nr_domains, side_x_stripe, side_y_stripe);

        local_grid_type lg_stripe;
        make_local_grid_from_stripe(lg_stripe, gg, ds);

        print_local_grid_info(lg_stripe);

        std::valarray<double> Qs;
        perform_quadrature_local_stripe(Qs, lg_stripe, qr, ds);

        std::cout << std::endl;
        std::cout << "Qs: ";
        for (size_t i = 0; i < Qs.size(); ++i) {
            std::cout << Qs[i] << " ";
        }
        std::cout << std::endl;
        std::cout << std::endl;
    }

    return 0;
}