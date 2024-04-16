#include <cmath>
#include <fstream>  // Include the necessary header file
#include <functional>
#include <iostream>
#include <tuple>
#include <valarray>
#include <vector>


struct global_grid {
    std::valarray<double> grid;

    double delta;
    
    double x_zero;
    double y_zero;
    
    double x_max;
    double y_max;
};


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
    const int nx = std::floor((x - xmin) / delta);
    const int ny = std::floor((y - ymin) / delta);

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
        x[i] = xmin + (xmax - xmin) * double(std::rand()) / double(RAND_MAX);
        y[i] = ymin + (ymax - ymin) * double(std::rand()) / double(RAND_MAX);
        w[i] = 1.0 / (double)n;
    }
}

/**
 * @brief Creates a square domain with stripes.
 *
 * This function creates a square domain with stripes by specifying the number of squares, side
 * length, starting coordinates, and domain boundaries.
 *
 * @param nr_squares The number of squares in the domain.
 * @param side The side length of each square.
 * @param xstart The starting x-coordinate of the domain.
 * @param ystart The starting y-coordinate of the domain.
 * @param xmin The minimum x-coordinate of the domain.
 * @param xmax The maximum x-coordinate of the domain.
 * @param ymin The minimum y-coordinate of the domain.
 * @param ymax The maximum y-coordinate of the domain.
 * @param x_coords Reference to a valarray to store the x-coordinates of the squares.
 * @param y_coords Reference to a valarray to store the y-coordinates of the squares.
 * @return The number of squares created in the domain.
 */
unsigned int make_square_domaind_stripe(const unsigned int nr_squares,      //
                                        const double side,                  //
                                        const double xstart,                //
                                        const double ystart,                //
                                        const double xmin,                  //
                                        const double xmax,                  //
                                        const double ymin,                  //
                                        const double ymax,                  //
                                        std::valarray<double> &x_coords,    //
                                        std::valarray<double> &y_coords) {  //
    //
    // if (xstart < xmin or xstart + nr_squares * side > xmax or ystart < ymin or
    //     ystart + nr_squares * side > ymax) {
    //     std::cerr << "Invalid domain" << std::endl;
    //     return 0;
    // }

    x_coords.resize(nr_squares);
    y_coords.resize(nr_squares);

    unsigned int i = 0;
    unsigned int j = 0;
    unsigned int cnt = 0;

    while (true) {
        if (xstart + i * side + side > xmax) {
            i = 0;
            ++j;
        }

        if (ystart + j * side + side > ymax) {
            break;
        }

        x_coords[cnt] = xstart + i * side;
        y_coords[cnt] = ystart + j * side;

        if (cnt >= nr_squares) {
            break;
        }
        cnt++;
        i++;
    }

    return cnt;
}

/**
 * @brief Computes the integral of a function over a square domain using quadrature.
 *
 * This function computes the integral of a function over a square domain using quadrature. The
 * function is evaluated at the quadrature nodes and the integral is computed using the quadrature
 * weights.
 *
 * @param local_grid The local grid of values.
 * @param x_min_grid The minimum x-coordinate of the grid.
 * @param y_min_grid The minimum y-coordinate of the grid.
 * @param n The number of grid points in each dimension of the local grid.
 * @param x_min The minimum x-coordinate of the square domain.
 * @param y_min The minimum y-coordinate of the square domain.
 * @param side The side length of the square domain.
 * @param x_Q The x-coordinates of the quadrature nodes.
 * @param y_Q The y-coordinates of the quadrature nodes.
 * @param w_Q The weights of the quadrature nodes.
 * @return The computed integral value.
 */
double quadrture_local(const std::valarray<double> &local_grid,  //
                       const double x_min_grid,                  //
                       const double y_min_grid,                  //
                       const int n,                              //
                       const double x_min,                       //
                       const double y_min,                       //
                       const double side,                        //
                       const std::valarray<double> &x_Q,         //
                       const std::valarray<double> &y_Q,         //
                       const std::valarray<double> &w_Q) {       //
    //
    // const int n_nodes = n * n;
    unsigned int Q_nodes = x_Q.size();

    double Q = 0.0;

    if (Q_nodes != y_Q.size() or Q_nodes != w_Q.size()) {
        return false;
    }

    const double delta_side_X = side / std::sqrt(w_Q.size());
    const double delta_side_Y = side / std::sqrt(w_Q.size());

    for (unsigned int cnt = 0; cnt < Q_nodes; ++cnt) {
        const auto [i, j] = get_nearest_coordinates(x_Q[cnt],      //
                                                    y_Q[cnt],      //
                                                    delta_side_X,  //
                                                    x_min_grid,    //
                                                    y_min_grid);   //

        const double x1 = x_min_grid + i * delta_side_X;
        const double y1 = y_min_grid + j * delta_side_Y;

        const double x2 = x_min_grid + (i + 1) * delta_side_X;
        const double y2 = y_min_grid + (j + 1) * delta_side_Y;

        // std::cout << "X1 = " << x1 << ", Y1 = " << y1 << std::endl;
        // std::cout << "X2 = " << x2 << ", Y2 = " << y2 << std::endl;
        // std::cout << "XQ = " << x_Q[cnt] << ", YQ = " << y_Q[cnt] << std::endl;
        // std::cout << std::endl;

        const double f1 = local_grid[i * n + j];
        const double f2 = local_grid[(i + 1) * n + j];
        const double f3 = local_grid[i * n + (j + 1)];
        const double f4 = local_grid[(i + 1) * n + (j + 1)];

        const double delta_x = x2 - x1;
        const double delta_y = y2 - y1;

        const double w11 = (x2 - x_Q[cnt]) * (y2 - y_Q[cnt]) / (delta_x * delta_y);
        const double w12 = (x2 - x_Q[cnt]) * (y_Q[cnt] - y1) / (delta_x * delta_y);
        const double w21 = (x_Q[cnt] - x1) * (y2 - y_Q[cnt]) / (delta_x * delta_y);
        const double w22 = (x_Q[cnt] - x1) * (y_Q[cnt] - y1) / (delta_x * delta_y);

        const double fxy = f1 * w11 + f2 * w12 + f3 * w21 + f4 * w22;

        std::cout << "X1 = " << x1 << ", Y1 = " << y1 << std::endl;
        std::cout << "X2 = " << x2 << ", Y2 = " << y2 << std::endl;
        std::cout << "XQ = " << x_Q[cnt] << ", YQ = " << y_Q[cnt] << std::endl;
        std::cout << "f1 = " << f1 << ", f2 = " << f2 << ", f3 = " << f3 << ", f4 = " << f4
                  << std::endl;
        std::cout << "fxy = " << fxy << std::endl;
        std::cout << std::endl;

        Q += fxy * w_Q[cnt];
    }

    return Q;
}

/**
 * This function tests the generation of a grid and the creation of square domains.
 *
 * @param argc The number of command-line arguments.
 * @param argv An array of command-line arguments.
 * @return 0 indicating successful execution.
 */
int test_quadratures(int argc, char **argv) {
    // Define the domain
    const double xmin = 0.0;
    const double xmax = 10.0;

    const double ymin = 0.0;
    const double ymax = 10.0;

    const double delta = 0.04;
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

    std::valarray<double> x_coords, y_coords;
    const unsigned int nr_squares = 200;
    const double side = 0.42;
    const double xstart = 1.21;
    const double ystart = 1.33;

    const unsigned int n = make_square_domaind_stripe(nr_squares,  //
                                                      side,
                                                      xstart,
                                                      ystart,
                                                      xmin,
                                                      xmax,
                                                      ymin,
                                                      ymax,
                                                      x_coords,   // x-coordinates of the squares
                                                      y_coords);  // y-coordinates of the squares

    std::cout << "Number of squares created = " << n << std::endl;

    // for (unsigned int i = 0; i < n; ++i) {
    //     std::cout << "Square " << i << ": (" << x_coords[i] << ", " << y_coords[i] << ")"
    //               << std::endl;
    // }

    std::vector<std::valarray<double>> x_MC, y_MC, w_MC;

    std::valarray<double> x_MC_zero, y_MC_zero, w_MC_zero;

    unsigned int MC_nodes = 200*200;

    for (unsigned int i = 0; i < n; ++i) {
        // std::valarray<double> x, y, w;
        // generate_MC_nodes(MC_nodes,  //
        //                   x_coords[i],
        //                   x_coords[i] + side,
        //                   y_coords[i],
        //                   y_coords[i] + side,
        //                   x,
        //                   y,
        //                   w);
        // x_MC.push_back(x);
        // y_MC.push_back(y);
        // w_MC.push_back(w);
    }

    std::valarray<double> local_grid;
    int n_local = 0;

    copy_to_local_grid(grid, nx, ny, x_coords[0], y_coords[0], side, delta, local_grid, n_local);

    generate_MC_nodes(MC_nodes,            // numebr of nodes
                      x_coords[0],         // x_min of the square
                      x_coords[0] + side,  // x_max of the square
                      y_coords[0],         // y_min of the square
                      y_coords[0] + side,  // y_max of the square
                      x_MC_zero,           // x-coordinates of the nodes
                      y_MC_zero,           // y-coordinates of the nodes
                      w_MC_zero);          // weights of the nodes

    std::cout << "local_grid.size() = " << local_grid.size() << std::endl;
    std::cout << "x_MC_zero.size() = " << x_MC_zero.size() << std::endl;
    std::cout << "nx = " << nx << ", ny = " << ny << std::endl;
    std::cout << "n_local = " << n_local << std::endl;

    double Q = quadrture_local(
            local_grid,   //
            x_coords[0],  //
            y_coords[0],  //
            n_local,      //
            x_coords[0],  //
            y_coords[0],  //
            side,         //
            x_MC_zero,    // x-coordinates of the nodes are random in the square with no sorting
            y_MC_zero,    //
            w_MC_zero);

    std::cout << "Q = " << Q << std::endl;
    std::cout << std::endl;

    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;

    for (unsigned int ii = 0; ii < (unsigned int)n_local; ++ii) {
        for (unsigned int jj = 0; jj < (unsigned int)n_local; ++jj) {
            double x = x_coords[0] + ii * delta;
            double y = y_coords[0] + jj * delta;
            std::cout << x << " " << y << " " << local_grid[ii * n_local + jj] << " || ";
        }
        std::cout << std::endl;
    }

    return 0;
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
    return test_quadratures(argc, argv);

    // // Define the domain
    // const double xmin = 0.0;
    // const double xmax = 10.0;

    // const double ymin = 0.0;
    // const double ymax = 10.0;

    // const double delta = 0.001;
    // std::valarray<double> grid;

    // auto f = [](double x, double y) { return x * x + y * y; };

    // std::cout << "Generating grid..." << std::endl;
    // std::cout << "delta = " << delta << std::endl;

    // int nx = 0;
    // int ny = 0;

    // const bool flag = generate_grid(grid, nx, ny, f, delta, xmin, xmax, ymin, ymax);

    // if (flag) {
    //     std::cout << "Grid generated successfully" << std::endl;
    // } else {
    //     std::cout << "Failed to generate grid" << std::endl;
    // }

    // {
    //     const double x = 1.014;
    //     const double y = 1.10;

    //     const auto [i, j] = get_nearest_coordinates(x, y, delta);

    //     // int nx = std::ceil((xmax - xmin) / delta);
    //     int ny = std::ceil((ymax - ymin) / delta);

    //     std::cout << "x = " << x << ", y = " << y << std::endl;
    //     std::cout << "i = " << i << ", j = " << j << std::endl << std::endl;

    //     std::cout << "grid[i * ny + j]         = " << grid[i * ny + j] << std::endl;
    //     std::cout << "grid[(i+1) * ny + j]     = " << grid[(i + 1) * ny + j] << std::endl;
    //     std::cout << "grid[i * ny + (j+1)]     = " << grid[i * ny + (j + 1)] << std::endl;
    //     std::cout << "grid[(i+1) * ny + (j+1)] = " << grid[(i + 1) * ny + (j + 1)] << std::endl;
    // }

    // {
    //     std::valarray<double> local_grid;
    //     int n = 0;
    //     const double x = 2.014;
    //     const double y = 3.10;
    //     const double square_side = 1.0;

    //     copy_to_local_grid(grid, nx, ny, x, y, square_side, delta, local_grid, n);

    //     std::cout << "x = " << x << ", y = " << y << std::endl << std::endl;

    //     // Write local_grid to a file
    //     std::ofstream file("local_grid.txt");
    //     if (file.is_open()) {
    //         for (int i = 0; i < n; ++i) {
    //             for (int j = 0; j < n; ++j) {
    //                 file << local_grid[i * n + j] << " ";
    //             }
    //             file << std::endl;
    //         }
    //         file.close();
    //         std::cout << "local_grid written to file successfully" << std::endl;
    //     } else {
    //         std::cout << "Failed to open file for writing" << std::endl;
    //     }

    //     std::cout << "local_grid.size() = " << local_grid.size() << std::endl;
    // }
    return 0;
}
