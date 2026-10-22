#include "sfem_test.hpp"

#include "sfem_DSSBXYZE.hpp"
#include "sfem_dss_bxyze_laplacian.hpp"
#include "smesh_mesh.hpp"
#include "smesh_semistructured.hpp"

#include <cmath>
#include <limits>
#include <vector>

static double tolerance() { return 512 * std::numeric_limits<scalar_t>::epsilon(); }

static int test_affine_metrics_and_padding() {
    // Independent affine cells: no nx/ny/nz or structured connectivity needed.
    const ptrdiff_t ne = 5;
    const int B = 4;
    for (const int level : {1, 2, 4, 8}) {
        sfem_dss_bxyze_layout_t layout;
        SFEM_TEST_EQ(sfem_dss_bxyze_create_layout(level, ne, B, 2, &layout), SFEM_SUCCESS);
        const int N = layout.n;
        const int N3 = N * N * N;
        std::vector<idx_t> connectivity((size_t)N3 * ne);
        std::vector<const idx_t *> elements((size_t)N3);
        std::vector<geom_t> x((size_t)N3 * ne), y(x.size()), z(x.size());
        for (int v = 0; v < N3; ++v) {
            elements[v] = &connectivity[(size_t)v * ne];
        }

        double expected[5][6];
        for (ptrdiff_t e = 0; e < ne; ++e) {
            // Upper triangular J, followed by a rigid rotation on odd cells.
            const double a = 1 + 0.25 * e, b = 0.75 + 0.125 * e, c = 0.5 + 0.125 * e;
            const double s = 0.25, t = -0.125, h = 0.125;
            const double i00 = 1 / a, i01 = -s / (a * b), i02 = (s * h - t * b) / (a * b * c);
            const double i11 = 1 / b, i12 = -h / (b * c), i22 = 1 / c;
            const double det = a * b * c;
            expected[e][0] = det * (i00 * i00 + i01 * i01 + i02 * i02);
            expected[e][1] = det * (i01 * i11 + i02 * i12);
            expected[e][2] = det * i02 * i22;
            expected[e][3] = det * (i11 * i11 + i12 * i12);
            expected[e][4] = det * i12 * i22;
            expected[e][5] = det * i22 * i22;

            for (int iz = 0; iz < N; ++iz) {
                for (int iy = 0; iy < N; ++iy) {
                    for (int ix = 0; ix < N; ++ix) {
                        const int v = ix + N * (iy + N * iz);
                        const idx_t gid = (idx_t)(e * N3 + v);
                        connectivity[(size_t)v * ne + e] = gid;
                        const double xi = (double)ix / level, eta = (double)iy / level, zeta = (double)iz / level;
                        const double px = a * xi + s * eta + t * zeta;
                        const double py = b * eta + h * zeta;
                        x[gid] = (geom_t)(4 * e + ((e % 2) ? -py : px));
                        y[gid] = (geom_t)((e % 2) ? px : py);
                        z[gid] = (geom_t)(c * zeta);
                    }
                }
            }
        }

        const geom_t *points[3] = {x.data(), y.data(), z.data()};
        const ptrdiff_t nvalues = sfem_dss_bxyze_affine_geometry_n_values(&layout);
        SFEM_TEST_EQ(nvalues, (ptrdiff_t)48);
        std::vector<scalar_t> geometry((size_t)nvalues, (scalar_t)-777);
        SFEM_TEST_EQ(sfem_dss_bxyze_affine_geometry(&layout, elements.data(), points, geometry.data()), SFEM_SUCCESS);

        for (ptrdiff_t e = 0; e < layout.nblocks * B; ++e) {
            for (int c = 0; c < 6; ++c) {
                const ptrdiff_t offset = ((e / B) * 6 + c) * B + e % B;
                const double reference = e < ne ? expected[e][c] : 0;
                SFEM_TEST_APPROXEQ((double)geometry[offset], reference, tolerance());
            }
        }
    }
    return SFEM_TEST_SUCCESS;
}

static int test_mesh_setup_reuses_cartesian_layout() {
    const int level = 4, B = 3;
    auto macro_mesh = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 2, 2, 1, 0, 0, 0, 1, 1, 1);
    auto mesh = smesh::to_semistructured(level, macro_mesh, false, false);
    sfem_dss_bxyze_laplacian_cartesian_layout_t layout;
    SFEM_TEST_EQ(sfem_dss_bxyze_laplacian_cartesian_create_layout(level, 2, 2, 1, B, 2, &layout), SFEM_SUCCESS);

    sfem::DSSBXYZEGeometry geometry;
    SFEM_TEST_EQ(sfem::dss_bxyze_affine_geometry(mesh, layout, &geometry), SFEM_SUCCESS);
    SFEM_TEST_EQ(geometry.layout.nelements, layout.nelements);
    SFEM_TEST_EQ(geometry.layout.execution_block_size, 2);
    SFEM_TEST_EQ(geometry.block_stride, (ptrdiff_t)18);
    SFEM_TEST_EQ(geometry.nvalues, (ptrdiff_t)36);
    SFEM_TEST_EQ((ptrdiff_t)geometry.values->size(), geometry.nvalues);
    const double expected[6] = {1, 0, 0, 1, 0, 0.25};
    for (ptrdiff_t e = 0; e < layout.nelements; ++e) {
        for (int c = 0; c < 6; ++c) {
            const ptrdiff_t offset = ((e / B) * 6 + c) * B + e % B;
            SFEM_TEST_APPROXEQ((double)geometry.values->data()[offset], expected[c], tolerance());
        }
    }
    SFEM_TEST_EQ(sfem::dss_bxyze_affine_geometry(mesh, geometry.layout, &geometry), SFEM_SUCCESS);

    sfem_dss_bxyze_layout_t mismatch;
    SFEM_TEST_EQ(sfem_dss_bxyze_create_layout(level, 3, B, 2, &mismatch), SFEM_SUCCESS);
    SFEM_TEST_EQ(sfem::dss_bxyze_affine_geometry(mesh, mismatch, &geometry), SFEM_FAILURE);
    SFEM_TEST_ASSERT(!geometry.values);
    return SFEM_TEST_SUCCESS;
}

static int test_empty_invalid_and_degenerate_geometry() {
    sfem_dss_bxyze_layout_t layout;
    SFEM_TEST_EQ(sfem_dss_bxyze_create_layout(1, 0, 4, 2, &layout), SFEM_SUCCESS);
    SFEM_TEST_EQ(sfem_dss_bxyze_affine_geometry_n_values(&layout), (ptrdiff_t)0);
    SFEM_TEST_EQ(sfem_dss_bxyze_affine_geometry(&layout, nullptr, nullptr, nullptr), SFEM_SUCCESS);
    SFEM_TEST_EQ(sfem_dss_bxyze_affine_geometry(nullptr, nullptr, nullptr, nullptr), SFEM_FAILURE);
    SFEM_TEST_EQ(sfem_dss_bxyze_create_layout(1, 1, 0, 1, &layout), SFEM_FAILURE);

    SFEM_TEST_EQ(sfem_dss_bxyze_create_layout(1, 1, 4, 2, &layout), SFEM_SUCCESS);
    idx_t ids[8];
    const idx_t *elements[8];
    geom_t x[8], y[8], z[8] = {0};
    for (int v = 0; v < 8; ++v) {
        ids[v] = v;
        elements[v] = &ids[v];
        x[v] = (geom_t)(v % 2);
        y[v] = (geom_t)((v / 2) % 2);
    }
    const geom_t *points[3] = {x, y, z};
    scalar_t geometry[24];
    SFEM_TEST_EQ(sfem_dss_bxyze_affine_geometry(&layout, elements, points, geometry), SFEM_FAILURE);
    for (int v = 0; v < 8; ++v) {
        z[v] = (geom_t)-(v / 4);
    }
    SFEM_TEST_EQ(sfem_dss_bxyze_affine_geometry(&layout, elements, points, geometry), SFEM_FAILURE);
    layout.nblocks = 2;
    SFEM_TEST_EQ(sfem_dss_bxyze_affine_geometry_n_values(&layout), (ptrdiff_t)-1);
    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_affine_metrics_and_padding);
    SFEM_RUN_TEST(test_mesh_setup_reuses_cartesian_layout);
    SFEM_RUN_TEST(test_empty_invalid_and_degenerate_geometry);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
