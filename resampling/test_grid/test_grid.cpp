#include <cmath>
#include <fstream>  // Include the necessary header file
#include <functional>
#include <iostream>
#include <tuple>
#include <valarray>
#include <vector>

/**
 * Generates a grid of values by evaluating a given function at each grid point.
 *
 * @param grid The output grid of values.
 * @param nx The number of grid points along the x-axis.
 * @param ny The number of grid points along the y-axis.
 * @param f The function to evaluate at each grid point.
 * @param delta The grid spacing.
 * @param xmin The minimum x-coordinate of the grid.
 * @param xmax The maximum x-coordinate of the grid.
 * @param ymin The minimum y-coordinate of the grid.
 * @param ymax The maximum y-coordinate of the grid.
 * @return True if the grid generation is successful, false otherwise.
 */
bool generate_grid(std::valarray<double> &grid,
                   int &nx,
                   int &ny,
                   std::function<double(double, double)> f,
                   const double delta,
                   const double xmin = 0,
                   const double xmax = 10,
                   const double ymin = 0,
                   const double ymax = 10) {
    if (delta <= 0) {
        return false;
    }

    if (xmin >= xmax or ymin >= ymax) {
        return false;
    }

    nx = std::ceil((xmax - xmin) / delta);
    ny = std::ceil((ymax - ymin) / delta);

    std::cout << "Grid size = " << (nx * ny) << std::endl;

    grid.resize(nx * ny);
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            const std::size_t index = i * ny + j;
            if (index >= grid.size()) {
                std::cerr << "Index out of range" << std::endl;
                std::cerr << "i = " << i << ", j = " << j << std::endl;
                std::cerr << "nx = " << nx << ", ny = " << ny << std::endl;
                std::cerr << "index = " << index << std::endl;
                std::cerr << "grid.size() = " << grid.size() << std::endl;
                return false;
            }
            grid[i * ny + j] = f(xmin + i * delta, ymin + j * delta);
        }
    }

    return true;
}

/**
 * @brief Returns the nearest coordinates to the given (x, y) point.
 *
 * This function calculates the nearest coordinates to the given (x, y) point based on the specified
 * delta and minimum values for x and y.
 *
 * @param x The x-coordinate of the point.
 * @param y The y-coordinate of the point.
 * @param delta The spacing between grid points.
 * @param xmin The minimum value for x.
 * @param ymin The minimum value for y.
 * @return A tuple containing the nearest x and y coordinates.
 */
inline std::tuple<int, int> get_nearest_coordinates(const double x,
                                                    const double y,
                                                    const double delta,
                                                    const double xmin = 0,
                                                    const double ymin = 0) {
    const int nx = std::ceil((x - xmin) / delta);
    const int ny = std::ceil((y - ymin) / delta);

    return std::make_tuple(nx, ny);
}

/**
 * @brief Copies a square region from a global grid to a local grid centered around a given point.
 *
 * @param grid The global grid to copy from.
 * @param nx The number of grid points in the x-direction.
 * @param ny The number of grid points in the y-direction.
 * @param x The x-coordinate of the center of the local grid.
 * @param y The y-coordinate of the center of the local grid.
 * @param square_side The side length of the square region to copy.
 * @param delta The grid spacing.
 * @param local_grid The local grid to copy to.
 * @param n The number of grid points in each dimension of the local grid.
 */
inline void copy_to_local_grid(const std::valarray<double> &grid,  //
                               const int nx,                       //
                               const int ny,                       //
                               const double x,                     //
                               const double y,                     //
                               const double square_side,           //
                               const double delta,                 //
                               std::valarray<double> &local_grid,  //
                               int &n) {                           //
    //
    const auto [i, j] = get_nearest_coordinates(x, y, delta);
    n = std::ceil(square_side / delta);

    local_grid.resize(n * n);

    for (int ii = 0; ii < n; ++ii) {
        for (int jj = 0; jj < n; ++jj) {
            local_grid[ii * n + jj] = grid[(i + ii) * ny + (j + jj)];
        }
    }
}

/**
 * Generates Monte Carlo nodes.
 *
 * @param n The number of nodes to generate.
 * @param xmin The minimum x-coordinate value.
 * @param xmax The maximum x-coordinate value.
 * @param ymin The minimum y-coordinate value.
 * @param ymax The maximum y-coordinate value.
 * @param x The array to store the x-coordinates of the generated nodes.
 * @param y The array to store the y-coordinates of the generated nodes.
 * @param w The array to store the weights of the generated nodes.
 */
inline void generate_MC_nodes(const int n,
                              const double xmin,
                              const double xmax,
                              const double ymin,
                              const double ymax,
                              std::valarray<double> &x,
                              std::valarray<double> &y,
                              std::valarray<double> &w) {
    x.resize(n);
    y.resize(n);
    w.resize(n);

    for (int i = 0; i < n; ++i) {
        x[i] = xmin + (xmax - xmin) * std::rand() / RAND_MAX;
        y[i] = ymin + (ymax - ymin) * std::rand() / RAND_MAX;
        w[i] = 1.0 / (double)n;
    }
}

void make_square_domaind_stripe(const unsigned int nr_squares,
                                const double side,
                                const double xmin,
                                const double ymin) {


                                    
                                }

/**
 * @brief The main function of the program.
 *
 * This function generates a grid, performs some calculations on the grid, and writes a local grid
 * to a file. It demonstrates the usage of various functions and variables.
 *
 * @param argc The number of command-line arguments.
 * @param argv An array of command-line arguments.
 * @return 0 on successful execution.
 */
int main(int argc, char **argv) {
    // Define the domain
    const double xmin = 0.0;
    const double xmax = 10.0;

    const double ymin = 0.0;
    const double ymax = 10.0;

    const double delta = 0.001;
    std::valarray<double> grid;

    auto f = [](double x, double y) { return x * x + y * y; };

    std::cout << "Generating grid..." << std::endl;
    std::cout << "delta = " << delta << std::endl;

    int nx = 0;
    int ny = 0;

    const bool flag = generate_grid(grid, nx, ny, f, delta, xmin, xmax, ymin, ymax);

    if (flag) {
        std::cout << "Grid generated successfully" << std::endl;
    } else {
        std::cout << "Failed to generate grid" << std::endl;
    }

    {
        const double x = 1.014;
        const double y = 1.10;

        const auto [i, j] = get_nearest_coordinates(x, y, delta);

        // int nx = std::ceil((xmax - xmin) / delta);
        int ny = std::ceil((ymax - ymin) / delta);

        std::cout << "x = " << x << ", y = " << y << std::endl;
        std::cout << "i = " << i << ", j = " << j << std::endl << std::endl;

        std::cout << "grid[i * ny + j]         = " << grid[i * ny + j] << std::endl;
        std::cout << "grid[(i+1) * ny + j]     = " << grid[(i + 1) * ny + j] << std::endl;
        std::cout << "grid[i * ny + (j+1)]     = " << grid[i * ny + (j + 1)] << std::endl;
        std::cout << "grid[(i+1) * ny + (j+1)] = " << grid[(i + 1) * ny + (j + 1)] << std::endl;
    }

    {
        std::valarray<double> local_grid;
        int n = 0;
        const double x = 2.014;
        const double y = 3.10;
        const double square_side = 1.0;

        copy_to_local_grid(grid, nx, ny, x, y, square_side, delta, local_grid, n);

        std::cout << "x = " << x << ", y = " << y << std::endl << std::endl;

        // Write local_grid to a file
        std::ofstream file("local_grid.txt");
        if (file.is_open()) {
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    file << local_grid[i * n + j] << " ";
                }
                file << std::endl;
            }
            file.close();
            std::cout << "local_grid written to file successfully" << std::endl;
        } else {
            std::cout << "Failed to open file for writing" << std::endl;
        }

        std::cout << "local_grid.size() = " << local_grid.size() << std::endl;
    }
    return 0;
}
