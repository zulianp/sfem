#include "sfem_SELL.hpp"
#include "sfem_test.hpp"

#include <cmath>
#include <cstdint>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

using smesh::count_t;
using smesh::half_t;
using smesh::idx_t;

// SellMatrix against the difference form computed in double: a 27-point graph of 3x3 blocks on an
// n^3 grid (rows sorted, every row with its diagonal), random off-diagonal blocks, symmetric
// (K_ji = K_ij^T) or not.

namespace {

    struct Graph {
        ptrdiff_t            n = 0;
        std::vector<count_t> rp;
        std::vector<idx_t>   ci;
    };

    Graph grid_graph(const int m) {
        Graph g;
        g.n = (ptrdiff_t)m * m * m;
        g.rp.assign((size_t)g.n + 1, 0);
        for (int z = 0; z < m; ++z)
            for (int y = 0; y < m; ++y)
                for (int x = 0; x < m; ++x) {
                    const idx_t i = (idx_t)((z * m + y) * m + x);
                    for (int dz = -1; dz <= 1; ++dz)
                        for (int dy = -1; dy <= 1; ++dy)
                            for (int dx = -1; dx <= 1; ++dx) {
                                const int X = x + dx, Y = y + dy, Z = z + dz;
                                if (X < 0 || Y < 0 || Z < 0 || X >= m || Y >= m || Z >= m) continue;
                                g.ci.push_back((idx_t)((Z * m + Y) * m + X));
                            }
                    g.rp[(size_t)i + 1] = (count_t)g.ci.size();
                }
        return g;
    }

    double rnd(unsigned &s) {
        s = s * 1664525u + 1013904223u;
        return (double)(s >> 8) / (double)(1u << 24) - 0.5;
    }

    /// Values on (rp, ci), blocks row-major; symmetric: K_ji = K_ij^T.
    std::vector<float> values(const Graph &g, const bool symmetric) {
        std::vector<float> v(g.ci.size() * 9);
        unsigned           s = 7;
        for (ptrdiff_t i = 0; i < g.n; ++i)
            for (count_t k = g.rp[i]; k < g.rp[i + 1]; ++k) {
                const idx_t j = g.ci[k];
                if (symmetric && j < i) continue;
                for (int e = 0; e < 9; ++e) v[(size_t)k * 9 + e] = (float)rnd(s);
                if (symmetric && j > i) {
                    count_t kt = g.rp[j];
                    while (g.ci[kt] != (idx_t)i) ++kt;
                    for (int r = 0; r < 3; ++r)
                        for (int c = 0; c < 3; ++c) v[(size_t)kt * 9 + c * 3 + r] = v[(size_t)k * 9 + r * 3 + c];
                }
            }
        return v;
    }

    /// y_i = sum_{j != i} K_ij (x_j - x_i), and the diagonal -sum_{j != i} diag(K_ij), in double.
    void reference(const Graph &g, const std::vector<float> &v, const std::vector<double> &x, std::vector<double> &y, std::vector<double> &d) {
        y.assign((size_t)g.n * 3, 0);
        d.assign((size_t)g.n * 3, 0);
        for (ptrdiff_t i = 0; i < g.n; ++i)
            for (count_t k = g.rp[i]; k < g.rp[i + 1]; ++k) {
                const idx_t j = g.ci[k];
                if (j == i) continue;
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 3; ++c) y[(size_t)i * 3 + r] += (double)v[(size_t)k * 9 + r * 3 + c] * (x[(size_t)j * 3 + c] - x[(size_t)i * 3 + c]);
                    d[(size_t)i * 3 + r] -= (double)v[(size_t)k * 9 + r * 3 + r];
                }
            }
    }

    double rel(const std::vector<double> &a, const std::vector<double> &b) {
        double num = 0, den = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            num += (a[i] - b[i]) * (a[i] - b[i]);
            den += b[i] * b[i];
        }
        return std::sqrt(num / den);
    }

    template <class M>
    int check(const char *const label, const bool symmetric, const double bound) {
        const Graph               g = grid_graph(12);  // 1728 rows: four windows of 512
        const std::vector<float>  v = values(g, symmetric);
        std::vector<double>       x((size_t)g.n * 3), y_ref, d_ref;
        unsigned                  s = 11;
        for (auto &e : x) e = rnd(s);
        reference(g, v, x, y_ref, d_ref);

        M m(g.n, g.rp.data(), g.ci.data());
        m.encode(g.rp.data(), g.ci.data(), v.data());
        std::vector<double> y((size_t)g.n * 3, 0), d((size_t)g.n * 3, 0);
        m.template apply<float, false>(x.data(), y.data());
        m.add_diagonal(d.data());
        const double ey = rel(y, y_ref), ed = rel(d, d_ref);
        printf("%-30s apply %.3e, diagonal %.3e (bound %.0e), %zu bytes\n", label, ey, ed, bound, m.bytes());
        SFEM_TEST_ASSERT(ey < bound);
        SFEM_TEST_ASSERT(ed < bound);

        // Twice with the same threads: the same bits (no atomics, fixed orders).
        std::vector<double> again((size_t)g.n * 3, 0);
        m.template apply<float, false>(x.data(), again.data());
        SFEM_TEST_ASSERT(again == y);
#ifdef _OPENMP
        // The difference form sums each row alone, in its own order: the thread count does not enter.
        if constexpr (!M::upper) {
            const int threads = omp_get_max_threads();
            omp_set_num_threads(1);
            std::vector<double> one((size_t)g.n * 3, 0);
            m.template apply<float, false>(x.data(), one.data());
            omp_set_num_threads(threads);
            SFEM_TEST_ASSERT(one == y);
        }
#endif
        return SFEM_TEST_SUCCESS;
    }

}  // namespace

using sfem::SellForm;
using sfem::SellMatrix;

int test_symmetric_difference() {
    SFEM_TEST_ASSERT((check<SellMatrix<float, SellForm::Difference, 16, idx_t, true>>("f32 difference", true, 1e-6) == SFEM_TEST_SUCCESS));
    SFEM_TEST_ASSERT((check<SellMatrix<half_t, SellForm::Difference, 16, idx_t, true>>("f16 difference", true, 2e-3) == SFEM_TEST_SUCCESS));
    SFEM_TEST_ASSERT((check<SellMatrix<int8_t, SellForm::Difference, 16, idx_t, true>>("i8 difference", true, 3e-2) == SFEM_TEST_SUCCESS));
    return SFEM_TEST_SUCCESS;
}

int test_symmetric_upper() {
    SFEM_TEST_ASSERT((check<SellMatrix<float, SellForm::Upper, 16, uint16_t>>("f32 upper", true, 1e-6) == SFEM_TEST_SUCCESS));
    SFEM_TEST_ASSERT((check<SellMatrix<half_t, SellForm::Upper, 16, uint16_t>>("f16 upper", true, 2e-3) == SFEM_TEST_SUCCESS));
    SFEM_TEST_ASSERT((check<SellMatrix<int8_t, SellForm::Upper, 16, uint16_t>>("i8 upper", true, 3e-2) == SFEM_TEST_SUCCESS));
    return SFEM_TEST_SUCCESS;
}

int test_unsymmetric() {
    SFEM_TEST_ASSERT((check<SellMatrix<float, SellForm::Difference, 16, idx_t, false>>("f32 unsymmetric", false, 1e-6) == SFEM_TEST_SUCCESS));
    SFEM_TEST_ASSERT((check<SellMatrix<half_t, SellForm::Difference, 16, idx_t, false>>("f16 unsymmetric", false, 2e-3) == SFEM_TEST_SUCCESS));
    SFEM_TEST_ASSERT((check<SellMatrix<int8_t, SellForm::Difference, 16, idx_t, false>>("i8 unsymmetric", false, 3e-2) == SFEM_TEST_SUCCESS));
    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_symmetric_difference);
    SFEM_RUN_TEST(test_symmetric_upper);
    SFEM_RUN_TEST(test_unsymmetric);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
