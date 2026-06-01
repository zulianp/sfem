#include "sfem_test.hpp"

#include "integrations/smesh/sccd_smesh_CCD.hpp"
#include "sfem_aliases.hpp"
#include "smesh_env.hpp"
#include "smesh_mesh.hpp"
#include "ssdf.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace {

    using sfem::idx_t;
    using sfem::real_t;

    std::shared_ptr<smesh::Mesh> make_two_body_surface_mesh() {
        auto points = smesh::create_host_buffer<smesh::geom_t>(3, 16);

        const smesh::geom_t coords[16][3] = {{0.00, 0.00, 0.00},
                                             {1.00, 0.00, 0.00},
                                             {1.00, 0.40, 0.00},
                                             {0.00, 0.40, 0.00},
                                             {0.00, 0.00, 1.00},
                                             {1.00, 0.00, 1.00},
                                             {1.00, 0.40, 1.00},
                                             {0.00, 0.40, 1.00},
                                             {0.25, 1.05, 0.25},
                                             {0.75, 1.05, 0.25},
                                             {0.75, 1.45, 0.25},
                                             {0.25, 1.45, 0.25},
                                             {0.25, 1.05, 0.75},
                                             {0.75, 1.05, 0.75},
                                             {0.75, 1.45, 0.75},
                                             {0.25, 1.45, 0.75}};

        for (ptrdiff_t i = 0; i < 16; ++i) {
            points->data()[0][i] = coords[i][0];
            points->data()[1][i] = coords[i][1];
            points->data()[2][i] = coords[i][2];
        }

        auto elements = smesh::create_host_buffer<smesh::idx_t>(3, 24);

        const smesh::idx_t tris[24][3] = {{0, 2, 1},    {0, 3, 2},    {4, 5, 6},    {4, 6, 7},    {0, 1, 5},   {0, 5, 4},
                                          {3, 7, 6},    {3, 6, 2},    {0, 4, 7},    {0, 7, 3},    {1, 2, 6},   {1, 6, 5},
                                          {8, 10, 9},   {8, 11, 10},  {12, 13, 14}, {12, 14, 15}, {8, 9, 13},  {8, 13, 12},
                                          {11, 15, 14}, {11, 14, 10}, {8, 12, 15},  {8, 15, 11},  {9, 10, 14}, {9, 14, 13}};

        for (ptrdiff_t i = 0; i < 24; ++i) {
            elements->data()[0][i] = tris[i][0];
            elements->data()[1][i] = tris[i][1];
            elements->data()[2][i] = tris[i][2];
        }

        return std::make_shared<smesh::Mesh>(smesh::Communicator::self(), smesh::TRI3, elements, points);
    }

    bool is_top_node(const idx_t node) { return node >= 8; }

    bool is_top_face(const idx_t face) { return face >= 12; }

    std::array<real_t, 3> point_at_time(const smesh::SharedBuffer<real_t*>& p0,
                                        const smesh::SharedBuffer<real_t*>& p1,
                                        const idx_t                         node,
                                        const real_t                        t) {
        return {p0->data()[0][node] + t * (p1->data()[0][node] - p0->data()[0][node]),
                p0->data()[1][node] + t * (p1->data()[1][node] - p0->data()[1][node]),
                p0->data()[2][node] + t * (p1->data()[2][node] - p0->data()[2][node])};
    }

    real_t normalize(const real_t dx, const real_t dy, const real_t dz, real_t& nx, real_t& ny, real_t& nz) {
        const real_t distance = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (distance > real_t(0)) {
            nx = dx / distance;
            ny = dy / distance;
            nz = dz / distance;
        } else {
            nx = ny = nz = 0;
        }

        return distance;
    }

    void append_vec(std::vector<real_t>& out, const real_t x, const real_t y, const real_t z) {
        out.push_back(x);
        out.push_back(y);
        out.push_back(z);
    }

    template <typename T>
    int write_vector(const smesh::Path& path, const std::vector<T>& values) {
        auto buffer = smesh::create_host_buffer<T>(values.size());
        std::copy(values.begin(), values.end(), buffer->data());
        return buffer->to_file(path);
    }

    smesh::Path timestep_path(const smesh::Path& dir, const char* const name, const int component, const int step) {
        char      buffer[2048];
        const int nchars = snprintf(buffer,
                                    sizeof(buffer),
                                    "%s.%d.%06d.%s",
                                    name,
                                    component,
                                    step,
                                    std::string(smesh::TypeToString<real_t>::value()).c_str());
        SFEM_UNUSED(nchars);
        return dir / buffer;
    }

    smesh::Path timestep_path(const smesh::Path& dir, const char* const name, const int step) {
        char      buffer[2048];
        const int nchars = snprintf(
                buffer, sizeof(buffer), "%s.%06d.%s", name, step, std::string(smesh::TypeToString<real_t>::value()).c_str());
        SFEM_UNUSED(nchars);
        return dir / buffer;
    }

    template <typename T>
    smesh::Path timestep_typed_path(const smesh::Path& dir, const char* const name, const int step) {
        char      buffer[2048];
        const int nchars =
                snprintf(buffer, sizeof(buffer), "%s.%06d.%s", name, step, std::string(smesh::TypeToString<T>::value()).c_str());
        SFEM_UNUSED(nchars);
        return dir / buffer;
    }

    template <typename T>
    smesh::Path typed_path(const smesh::Path& dir, const char* const name) {
        return dir / (std::string(name) + "." + std::string(smesh::TypeToString<T>::value()));
    }

}  // namespace

int test_grads() {
    auto surface = make_two_body_surface_mesh();
    SFEM_TEST_ASSERT(surface != nullptr);

    auto p0 = smesh::astype<real_t>(surface->points());
    auto p1 = smesh::astype<real_t>(surface->points());

    const real_t translation_y       = smesh::Env::read("SFEM_DISTANCE_GRADIENT_TRANSLATION_Y", real_t(-0.9));
    const real_t edge_gradient_scale = smesh::Env::read("SFEM_DISTANCE_GRADIENT_EDGE_SCALE", real_t(0.05));
    for (ptrdiff_t i = 0; i < surface->n_nodes(); ++i) {
        if (is_top_node(i)) {
            p1->data()[1][i] += translation_y;
        }
    }

    auto                       ccd = sccd::CCD<real_t>::create(surface);
    smesh::SharedBuffer<idx_t> vertex_overlap;
    smesh::SharedBuffer<idx_t> face_overlap;
    smesh::SharedBuffer<idx_t> edge0_overlap;
    smesh::SharedBuffer<idx_t> edge1_overlap;

    ccd->broad_phase(p0, p1, vertex_overlap, face_overlap, edge0_overlap, edge1_overlap);

    const smesh::Path output_dir(smesh::Env::read_string("SFEM_DISTANCE_GRADIENT_OUTPUT", "distance_gradients"));
    if (!output_dir.exists()) {
        SFEM_TEST_ASSERT(output_dir.make_dir());
    }

    SFEM_TEST_ASSERT(surface->write(output_dir / "surface") == SFEM_SUCCESS);

    const smesh::Path fields_dir = output_dir / "fields";
    if (!fields_dir.exists()) {
        SFEM_TEST_ASSERT(fields_dir.make_dir());
    }

    std::vector<idx_t>  pt_ids;
    std::vector<idx_t>  ee_ids;
    std::vector<real_t> pt_values;
    std::vector<real_t> ee_values;

    std::vector<smesh::geom_t> xdmf_points(surface->n_nodes() * 3);
    for (ptrdiff_t i = 0; i < surface->n_nodes(); ++i) {
        xdmf_points[3 * i + 0] = surface->points()->data()[0][i];
        xdmf_points[3 * i + 1] = surface->points()->data()[1][i];
        xdmf_points[3 * i + 2] = surface->points()->data()[2][i];
    }

    auto               surface_elements = surface->block(0)->elements();
    std::vector<idx_t> xdmf_triangles(surface->n_elements() * 3);
    for (ptrdiff_t i = 0; i < surface->n_elements(); ++i) {
        xdmf_triangles[3 * i + 0] = surface_elements->data()[0][i];
        xdmf_triangles[3 * i + 1] = surface_elements->data()[1][i];
        xdmf_triangles[3 * i + 2] = surface_elements->data()[2][i];
    }

    SFEM_TEST_ASSERT(write_vector(typed_path<smesh::geom_t>(output_dir, "xdmf_points"), xdmf_points) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(write_vector(typed_path<idx_t>(output_dir, "xdmf_triangles"), xdmf_triangles) == SFEM_SUCCESS);

    const int       n_steps = std::max(1, smesh::Env::read("SFEM_DISTANCE_GRADIENT_STEPS", 8));
    auto            faces   = surface->block(0)->elements();
    auto            edges   = ccd->edges();
    const ptrdiff_t n_nodes = surface->n_nodes();

    for (int step = 0; step <= n_steps; ++step) {
        const real_t t = real_t(step) / real_t(n_steps);

        std::vector<real_t> disp[3];
        std::vector<real_t> pt_distance(n_nodes, real_t(1e20));
        std::vector<real_t> pt_grad[3];
        std::vector<real_t> pt_closest[3];
        std::vector<real_t> ee_distance(n_nodes, real_t(1e20));
        std::vector<real_t> ee_grad[3];
        std::vector<real_t> ee_closest_points;
        std::vector<real_t> ee_closest_grad;
        std::vector<real_t> ee_closest_distance;
        std::vector<idx_t>  ee_closest_indices;
        std::vector<real_t> ee_gradient_line_points;
        std::vector<idx_t>  ee_gradient_line_indices;
        std::vector<real_t> ee_gradient_line_distance;

        for (int d = 0; d < 3; ++d) {
            disp[d].resize(n_nodes);
            pt_grad[d].assign(n_nodes, 0);
            pt_closest[d].assign(n_nodes, 0);
            ee_grad[d].assign(n_nodes, 0);
        }

        for (ptrdiff_t i = 0; i < n_nodes; ++i) {
            for (int d = 0; d < 3; ++d) {
                disp[d][i] = t * (p1->data()[d][i] - p0->data()[d][i]);
            }
        }

        for (ptrdiff_t i = 0; i < vertex_overlap->size(); ++i) {
            const idx_t v = vertex_overlap->data()[i];
            const idx_t f = face_overlap->data()[i];
            if (is_top_node(v) == is_top_face(f)) continue;

            const idx_t f0 = faces->data()[0][f];
            const idx_t f1 = faces->data()[1][f];
            const idx_t f2 = faces->data()[2][f];

            const auto p = point_at_time(p0, p1, v, t);
            const auto a = point_at_time(p0, p1, f0, t);
            const auto b = point_at_time(p0, p1, f1, t);
            const auto c = point_at_time(p0, p1, f2, t);

            real_t u = 0, w = 0;
            ssdf::point_triangle_closest_point_param<real_t>(
                    p[0], p[1], p[2], a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2], &u, &w);

            const real_t b0 = 1 - u - w;
            const real_t cx = b0 * a[0] + u * b[0] + w * c[0];
            const real_t cy = b0 * a[1] + u * b[1] + w * c[1];
            const real_t cz = b0 * a[2] + u * b[2] + w * c[2];

            real_t       nx = 0, ny = 0, nz = 0;
            const real_t distance = normalize(p[0] - cx, p[1] - cy, p[2] - cz, nx, ny, nz);

            pt_ids.push_back(step);
            pt_ids.push_back(v);
            pt_ids.push_back(f);

            pt_values.push_back(t);
            pt_values.push_back(distance);
            append_vec(pt_values, p[0], p[1], p[2]);
            append_vec(pt_values, cx, cy, cz);
            append_vec(pt_values, nx, ny, nz);
            append_vec(pt_values, -b0 * nx, -b0 * ny, -b0 * nz);
            append_vec(pt_values, -u * nx, -u * ny, -u * nz);
            append_vec(pt_values, -w * nx, -w * ny, -w * nz);

            if (distance < pt_distance[v]) {
                pt_distance[v]   = distance;
                pt_grad[0][v]    = nx;
                pt_grad[1][v]    = ny;
                pt_grad[2][v]    = nz;
                pt_closest[0][v] = cx;
                pt_closest[1][v] = cy;
                pt_closest[2][v] = cz;
            }
        }

        for (ptrdiff_t i = 0; i < edge0_overlap->size(); ++i) {
            const idx_t e0 = edge0_overlap->data()[i];
            const idx_t e1 = edge1_overlap->data()[i];

            const idx_t a0i = edges->data()[0][e0];
            const idx_t a1i = edges->data()[1][e0];
            const idx_t b0i = edges->data()[0][e1];
            const idx_t b1i = edges->data()[1][e1];
            if (is_top_node(a0i) == is_top_node(b0i)) continue;

            const auto a0 = point_at_time(p0, p1, a0i, t);
            const auto a1 = point_at_time(p0, p1, a1i, t);
            const auto b0 = point_at_time(p0, p1, b0i, t);
            const auto b1 = point_at_time(p0, p1, b1i, t);

            real_t s = 0, r = 0;
            ssdf::edge_to_edge_closest_points<real_t>(
                    a0[0], a0[1], a0[2], a1[0], a1[1], a1[2], b0[0], b0[1], b0[2], b1[0], b1[1], b1[2], &s, &r);

            const real_t ax = (1 - s) * a0[0] + s * a1[0];
            const real_t ay = (1 - s) * a0[1] + s * a1[1];
            const real_t az = (1 - s) * a0[2] + s * a1[2];
            const real_t bx = (1 - r) * b0[0] + r * b1[0];
            const real_t by = (1 - r) * b0[1] + r * b1[1];
            const real_t bz = (1 - r) * b0[2] + r * b1[2];

            real_t       nx = 0, ny = 0, nz = 0;
            const real_t distance = normalize(ax - bx, ay - by, az - bz, nx, ny, nz);

            append_vec(ee_closest_points, ax, ay, az);
            append_vec(ee_closest_grad, nx, ny, nz);
            ee_closest_distance.push_back(distance);
            {
                const idx_t line_start = static_cast<idx_t>(ee_gradient_line_points.size() / 3);
                append_vec(ee_gradient_line_points, ax, ay, az);
                append_vec(ee_gradient_line_points,
                           ax + edge_gradient_scale * nx,
                           ay + edge_gradient_scale * ny,
                           az + edge_gradient_scale * nz);
                ee_gradient_line_indices.push_back(line_start);
                ee_gradient_line_indices.push_back(line_start + 1);
                ee_gradient_line_distance.push_back(distance);
                ee_gradient_line_distance.push_back(distance);
            }

            append_vec(ee_closest_points, bx, by, bz);
            append_vec(ee_closest_grad, -nx, -ny, -nz);
            ee_closest_distance.push_back(distance);
            {
                const idx_t line_start = static_cast<idx_t>(ee_gradient_line_points.size() / 3);
                append_vec(ee_gradient_line_points, bx, by, bz);
                append_vec(ee_gradient_line_points,
                           bx - edge_gradient_scale * nx,
                           by - edge_gradient_scale * ny,
                           bz - edge_gradient_scale * nz);
                ee_gradient_line_indices.push_back(line_start);
                ee_gradient_line_indices.push_back(line_start + 1);
                ee_gradient_line_distance.push_back(distance);
                ee_gradient_line_distance.push_back(distance);
            }

            ee_ids.push_back(step);
            ee_ids.push_back(e0);
            ee_ids.push_back(e1);

            ee_values.push_back(t);
            ee_values.push_back(distance);
            append_vec(ee_values, ax, ay, az);
            append_vec(ee_values, bx, by, bz);
            append_vec(ee_values, (1 - s) * nx, (1 - s) * ny, (1 - s) * nz);
            append_vec(ee_values, s * nx, s * ny, s * nz);
            append_vec(ee_values, -(1 - r) * nx, -(1 - r) * ny, -(1 - r) * nz);
            append_vec(ee_values, -r * nx, -r * ny, -r * nz);

            const idx_t  edge_nodes[4] = {a0i, a1i, b0i, b1i};
            const real_t weights[4]    = {1 - s, s, -(1 - r), -r};
            for (int local = 0; local < 4; ++local) {
                const idx_t node = edge_nodes[local];
                if (distance < ee_distance[node]) {
                    ee_distance[node] = distance;
                    ee_grad[0][node]  = weights[local] * nx;
                    ee_grad[1][node]  = weights[local] * ny;
                    ee_grad[2][node]  = weights[local] * nz;
                }
            }
        }

        for (ptrdiff_t i = 0; i < n_nodes; ++i) {
            if (pt_distance[i] == real_t(1e20)) pt_distance[i] = 0;
            if (ee_distance[i] == real_t(1e20)) ee_distance[i] = 0;
        }

        for (int d = 0; d < 3; ++d) {
            SFEM_TEST_ASSERT(write_vector(timestep_path(fields_dir, "disp", d, step), disp[d]) == SFEM_SUCCESS);
            SFEM_TEST_ASSERT(write_vector(timestep_path(fields_dir, "pt_grad", d, step), pt_grad[d]) == SFEM_SUCCESS);
            SFEM_TEST_ASSERT(write_vector(timestep_path(fields_dir, "pt_closest", d, step), pt_closest[d]) == SFEM_SUCCESS);
            SFEM_TEST_ASSERT(write_vector(timestep_path(fields_dir, "ee_grad", d, step), ee_grad[d]) == SFEM_SUCCESS);
        }

        std::vector<real_t> disp_vec(3 * n_nodes);
        std::vector<real_t> pt_grad_vec(3 * n_nodes);
        std::vector<real_t> pt_closest_vec(3 * n_nodes);
        std::vector<real_t> ee_grad_vec(3 * n_nodes);
        for (ptrdiff_t i = 0; i < n_nodes; ++i) {
            for (int d = 0; d < 3; ++d) {
                disp_vec[3 * i + d]       = disp[d][i];
                pt_grad_vec[3 * i + d]    = pt_grad[d][i];
                pt_closest_vec[3 * i + d] = pt_closest[d][i];
                ee_grad_vec[3 * i + d]    = ee_grad[d][i];
            }
        }

        SFEM_TEST_ASSERT(write_vector(timestep_path(fields_dir, "disp_vec", step), disp_vec) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(write_vector(timestep_path(fields_dir, "pt_grad_vec", step), pt_grad_vec) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(write_vector(timestep_path(fields_dir, "pt_closest_vec", step), pt_closest_vec) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(write_vector(timestep_path(fields_dir, "ee_grad_vec", step), ee_grad_vec) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(write_vector(timestep_path(fields_dir, "ee_closest_points", step), ee_closest_points) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(write_vector(timestep_path(fields_dir, "ee_closest_grad", step), ee_closest_grad) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(write_vector(timestep_path(fields_dir, "ee_closest_distance", step), ee_closest_distance) ==
                         SFEM_SUCCESS);
        SFEM_TEST_ASSERT(write_vector(timestep_path(fields_dir, "ee_gradient_line_points", step), ee_gradient_line_points) ==
                         SFEM_SUCCESS);
        SFEM_TEST_ASSERT(write_vector(timestep_typed_path<idx_t>(fields_dir, "ee_gradient_line_indices", step),
                                      ee_gradient_line_indices) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(write_vector(timestep_path(fields_dir, "ee_gradient_line_distance", step), ee_gradient_line_distance) ==
                         SFEM_SUCCESS);

        ee_closest_indices.resize(ee_closest_distance.size());
        for (size_t i = 0; i < ee_closest_indices.size(); ++i) {
            ee_closest_indices[i] = static_cast<idx_t>(i);
        }
        SFEM_TEST_ASSERT(write_vector(timestep_typed_path<idx_t>(fields_dir, "ee_closest_indices", step), ee_closest_indices) ==
                         SFEM_SUCCESS);

        SFEM_TEST_ASSERT(write_vector(timestep_path(fields_dir, "pt_distance", step), pt_distance) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(write_vector(timestep_path(fields_dir, "ee_distance", step), ee_distance) == SFEM_SUCCESS);
    }

    SFEM_TEST_ASSERT(vertex_overlap->size() + edge0_overlap->size() > 0);
    SFEM_TEST_ASSERT(!pt_values.empty());
    SFEM_TEST_ASSERT(!ee_values.empty());

    const idx_t pt_value_cols = 20;
    const idx_t ee_value_cols = 20;
    const idx_t id_cols       = 3;

    const std::vector<idx_t> pt_shape    = {static_cast<idx_t>(pt_values.size() / pt_value_cols), pt_value_cols};
    const std::vector<idx_t> ee_shape    = {static_cast<idx_t>(ee_values.size() / ee_value_cols), ee_value_cols};
    const std::vector<idx_t> id_shape    = {static_cast<idx_t>(pt_ids.size() / id_cols), id_cols};
    const std::vector<idx_t> ee_id_shape = {static_cast<idx_t>(ee_ids.size() / id_cols), id_cols};

    SFEM_TEST_ASSERT(write_vector(typed_path<idx_t>(output_dir, "point_triangle_ids"), pt_ids) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(write_vector(typed_path<real_t>(output_dir, "point_triangle_values"), pt_values) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(write_vector(typed_path<idx_t>(output_dir, "point_triangle_ids_shape"), id_shape) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(write_vector(typed_path<idx_t>(output_dir, "point_triangle_values_shape"), pt_shape) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(write_vector(typed_path<idx_t>(output_dir, "edge_edge_ids"), ee_ids) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(write_vector(typed_path<real_t>(output_dir, "edge_edge_values"), ee_values) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(write_vector(typed_path<idx_t>(output_dir, "edge_edge_ids_shape"), ee_id_shape) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(write_vector(typed_path<idx_t>(output_dir, "edge_edge_values_shape"), ee_shape) == SFEM_SUCCESS);

    return SFEM_TEST_SUCCESS;
}

int main(int argc, char* argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_grads);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
