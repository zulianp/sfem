#include <cmath>
#include <functional>
#include <iostream>
#include <tuple>
#include <valarray>
#include <vector>

bool generate_grid(std::valarray<double> &grid,
                   std::function<double(double, double)> f,
                   const double delta,
                   const double xmin = 0,
                   const double xmax = 10,
                   const double ymin = 0,
                   const double ymax = 10) {
    if (delta <= 0) {
        return false;
    }

    if (xmin >= xmax || ymin >= ymax) {
        return false;
    }

    const int nx = std::ceil((xmax - xmin) / delta);
    const int ny = std::ceil((ymax - ymin) / delta);

    std::cout << "Grid size = " << (nx * ny) << std::endl;

    grid.resize(nx * ny);
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            const int index = i * ny + j;
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

std::tuple<int, int> get_nearest_coordinates(const double x,
                                             const double y,
                                             const double delta,
                                             const double xmin = 0,
                                             const double ymin = 0) {
    const int nx = std::ceil((x - xmin) / delta);
    const int ny = std::ceil((y - ymin) / delta);

    return std::make_tuple(nx, ny);
}

int main() {
    const double xmin = 0;
    const double xmax = 10;
    const double ymin = 0;
    const double ymax = 10;

    const double delta = 0.001;
    std::valarray<double> grid;

    auto f = [](double x, double y) { return x * x + y * y; };

    std::cout << "Generating grid..." << std::endl;
    std::cout << "delta = " << delta << std::endl;

    const bool flag = generate_grid(grid, f, delta, xmin, xmax, ymin, ymax);

    if (flag) {
        std::cout << "Grid generated successfully" << std::endl;
    } else {
        std::cout << "Failed to generate grid" << std::endl;
    }

    {
        const double x = 1.014;
        const double y = 1.10;

        const auto [i, j] = get_nearest_coordinates(x, y, delta);

        int nx = std::ceil((xmax - xmin) / delta);
        int ny = std::ceil((ymax - ymin) / delta);

        std::cout << "x = " << x << ", y = " << y << std::endl;
        std::cout << "i = " << i << ", j = " << j << std::endl << std::endl;

        std::cout << "grid[i * ny + j] = " << grid[i * ny + j] << std::endl;
        std::cout << "grid[(i+1) * ny + j] = " << grid[(i + 1) * ny + j] << std::endl;
        std::cout << "grid[i * ny + (j+1)] = " << grid[i * ny + (j + 1)] << std::endl;
        std::cout << "grid[(i+1) * ny + (j+1)] = " << grid[(i + 1) * ny + (j + 1)] << std::endl;
    }
    return 0;
}
