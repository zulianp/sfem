
#ifndef TEST_GRID_NEW_H
#define TEST_GRID_NEW_H

#include <valarray>
/**
 * @brief Represents a global grid with specified dimensions and properties.
 */
struct global_grid_type {
    std::valarray<double> grid; /**< The grid values. */

    double *grid_ptr_cu = nullptr; /**< Pointer to the grid values for cuda device */

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

/**
 * @brief Quadrature rule for numerical integration.
 *
 */
struct quadrature_rule {
    std::valarray<double> weights; /**< The quadrature weights. */
    std::valarray<double> x_nodes; /**< The x-coordinates of the quadrature nodes. */
    std::valarray<double> y_nodes; /**< The y-coordinates of the quadrature nodes. */

    double *weights_ptr_cu = nullptr; /**< Pointer to the quadrature weights for cuda device */
    double *x_nodes_ptr_cu = nullptr; /**< Pointer to the x-coordinates of the quadrature nodes for
                               cuda device */
    double *y_nodes_ptr_cu = nullptr; /**< Pointer to the y-coordinates of the quadrature nodes for
                              cuda device */

    double x_min; /**< The minimum x-coordinate of the domain. */
    double y_min; /**< The minimum y-coordinate of the domain. */
    double x_max; /**< The maximum x-coordinate of the domain. */
    double y_max; /**< The maximum y-coordinate of the domain. */
};

#endif  // TEST_GRID_NEW_H