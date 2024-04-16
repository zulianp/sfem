#include <cmath>
#include <fstream>  // Include the necessary header file
#include <functional>
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

    double x_d_min; /**< The x-coordinate of the grid origin. */
    double y_d_min; /**< The y-coordinate of the grid origin. */

    double x_d_max; /**< The maximum x-coordinate of the grid. */
    double y_d_max; /**< The maximum y-coordinate of the grid. */

    double x_grid_min; /**< The x-coordinate of the grid origin in the global grid. */
    double y_grid_min; /**< The y-coordinate of the grid origin in the global grid. */

    double x_grid_max; /**< The maximum x-coordinate of the grid in the global grid. */
    double y_grid_max; /**< The maximum y-coordinate of the grid in the global grid. */

    size_t x_size; /**< The number of grid points in the x-direction. */
    size_t y_size; /**< The number of grid points in the y-direction. */
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
                    global_grid_type& gg,
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
            //     std::cout << "index_global: " << index_global << " gg.grid.size(): " << gg.grid.size()
            //               << std::endl;
            //     return false;
            // }

            // if (index_local >= lg.grid.size()) {
            //     std::cout << "index_local: " << index_local << " lg.grid.size(): " << lg.grid.size()
            //               << std::endl;
            //     return false;
            // }

            lg.grid[index_local] = gg.grid[index_global];
            // std::cout << "i: " << i << " j: " << j << " value: " << gg.grid[i_global * gg.y_size + j_global]
            //           << std::endl;
            j_global++;
        }
        j_global = j_min;
        i_global++;
        // std::cout << std::endl;
    }

    return true;
}

/**
 * Prints the values of a local grid.
 *
 * @param lg The local grid to be printed.
 */
void print_local_grid(local_grid_type& lg) {
    for (size_t i = 0; i < lg.x_size; ++i) {
        for (size_t j = 0; j < lg.y_size; ++j) {
            std::cout << "X: " << lg.x_grid_min + i * lg.delta << " Y: " << lg.y_grid_min + j * lg.delta
                      << ": v:";
            // std::cout << "i: " << i << " j: " << j << " value: ";
            std::cout << lg.grid[i * lg.y_size + j] << " || ";
        }
        std::cout << std::endl;
    }
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

    print_local_grid(lg);
}