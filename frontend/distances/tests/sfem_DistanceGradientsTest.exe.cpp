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
#include <limits>
#include <string>
#include <vector>

namespace {

    using sfem::idx_t;
    using sfem::real_t;

    struct DistanceGradientOptions {
        real_t      translation_y       = -0.9;
        real_t      edge_gradient_scale = 1;
        real_t      top_body_y_min      = 1.0;
        int         nx                  = 5;
        int         n_steps             = 8;
        std::string output_dir          = "distance_gradients";

        static DistanceGradientOptions from_env() {
            DistanceGradientOptions opts;
            opts.translation_y       = smesh::Env::read("SFEM_DISTANCE_GRADIENT_TRANSLATION_Y", opts.translation_y);
            opts.edge_gradient_scale = smesh::Env::read("SFEM_DISTANCE_GRADIENT_EDGE_SCALE", opts.edge_gradient_scale);
            opts.top_body_y_min      = smesh::Env::read("SFEM_DISTANCE_GRADIENT_TOP_BODY_Y_MIN", opts.top_body_y_min);
            opts.nx                  = std::max(1, smesh::Env::read("SFEM_DISTANCE_GRADIENT_NX", opts.nx));
            opts.n_steps             = std::max(1, smesh::Env::read("SFEM_DISTANCE_GRADIENT_STEPS", opts.n_steps));
            opts.output_dir          = smesh::Env::read_string("SFEM_DISTANCE_GRADIENT_OUTPUT", opts.output_dir);
            return opts;
        }
    };

    std::shared_ptr<smesh::Mesh> make_two_body_surface_mesh(const DistanceGradientOptions& opts) {
        auto mesh1 = smesh::Mesh::create_tet4_cube(smesh::Communicator::self(),
                                                   static_cast<ptrdiff_t>(opts.nx),
                                                   std::max<ptrdiff_t>(1, opts.nx / 5),
                                                   static_cast<ptrdiff_t>(opts.nx),
                                                   0,
                                                   0.8,
                                                   0,
                                                   1,
                                                   1,
                                                   1);
        auto mesh2 = smesh::Mesh::create_tet4_cube(smesh::Communicator::self(),
                                                   std::max<ptrdiff_t>(1, opts.nx / 2),
                                                   static_cast<ptrdiff_t>(opts.nx),
                                                   std::max<ptrdiff_t>(1, opts.nx / 2),
                                                   0.25,
                                                   1.1,
                                                   0.25,
                                                   0.75,
                                                   1.9,
                                                   0.75);

        return smesh::skin(smesh::concatenate(mesh1, mesh2));
    }

    std::vector<unsigned char> classify_top_nodes(const std::shared_ptr<smesh::Mesh>& surface, const real_t top_body_y_min) {
        std::vector<unsigned char> ret(surface->n_nodes(), 0);
        auto                       points = surface->points()->data();
        for (ptrdiff_t i = 0; i < surface->n_nodes(); ++i) {
            ret[i] = points[1][i] > top_body_y_min ? 1 : 0;
        }

        return ret;
    }

    bool is_top_face(const smesh::SharedBuffer<idx_t*>& faces, const std::vector<unsigned char>& top_nodes, const idx_t face) {
        return top_nodes[faces->data()[0][face]] && top_nodes[faces->data()[1][face]] && top_nodes[faces->data()[2][face]];
    }

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

    void expand_bbox(const std::array<real_t, 3>& p, std::array<real_t, 3>& lo, std::array<real_t, 3>& hi) {
        for (int d = 0; d < 3; ++d) {
            lo[d] = std::min(lo[d], p[d]);
            hi[d] = std::max(hi[d], p[d]);
        }
    }

    bool aabb_overlap(const std::array<real_t, 3>& a_lo,
                      const std::array<real_t, 3>& a_hi,
                      const std::array<real_t, 3>& b_lo,
                      const std::array<real_t, 3>& b_hi) {
        for (int d = 0; d < 3; ++d) {
            if (a_hi[d] < b_lo[d] || b_hi[d] < a_lo[d]) {
                return false;
            }
        }

        return true;
    }

    void reset_bbox(std::array<real_t, 3>& lo, std::array<real_t, 3>& hi) {
        lo = {std::numeric_limits<real_t>::max(), std::numeric_limits<real_t>::max(), std::numeric_limits<real_t>::max()};
        hi = {-std::numeric_limits<real_t>::max(), -std::numeric_limits<real_t>::max(), -std::numeric_limits<real_t>::max()};
    }

    void node_swept_bbox(const smesh::SharedBuffer<real_t*>& p0,
                         const smesh::SharedBuffer<real_t*>& p1,
                         const idx_t                         node,
                         const real_t                        t0,
                         const real_t                        t1,
                         std::array<real_t, 3>&              lo,
                         std::array<real_t, 3>&              hi) {
        reset_bbox(lo, hi);
        expand_bbox(point_at_time(p0, p1, node, t0), lo, hi);
        expand_bbox(point_at_time(p0, p1, node, t1), lo, hi);
    }

    void element_swept_bbox(const smesh::SharedBuffer<idx_t*>&  elements,
                            const int                           n_nodes_per_element,
                            const idx_t                         element,
                            const smesh::SharedBuffer<real_t*>& p0,
                            const smesh::SharedBuffer<real_t*>& p1,
                            const real_t                        t0,
                            const real_t                        t1,
                            std::array<real_t, 3>&              lo,
                            std::array<real_t, 3>&              hi) {
        reset_bbox(lo, hi);
        for (int enode = 0; enode < n_nodes_per_element; ++enode) {
            const idx_t node = elements->data()[enode][element];
            expand_bbox(point_at_time(p0, p1, node, t0), lo, hi);
            expand_bbox(point_at_time(p0, p1, node, t1), lo, hi);
        }
    }

    void append_bbox_lines(const std::array<real_t, 3>& lo,
                           const std::array<real_t, 3>& hi,
                           const idx_t                  kind,
                           const idx_t                  id,
                           std::vector<real_t>&         points,
                           std::vector<idx_t>&          indices,
                           std::vector<idx_t>&          kinds,
                           std::vector<idx_t>&          ids) {
        const idx_t base = static_cast<idx_t>(points.size() / 3);
        append_vec(points, lo[0], lo[1], lo[2]);
        append_vec(points, hi[0], lo[1], lo[2]);
        append_vec(points, hi[0], hi[1], lo[2]);
        append_vec(points, lo[0], hi[1], lo[2]);
        append_vec(points, lo[0], lo[1], hi[2]);
        append_vec(points, hi[0], lo[1], hi[2]);
        append_vec(points, hi[0], hi[1], hi[2]);
        append_vec(points, lo[0], hi[1], hi[2]);

        const idx_t edges[12][2] = {
                {0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6}, {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
        for (int e = 0; e < 12; ++e) {
            indices.push_back(base + edges[e][0]);
            indices.push_back(base + edges[e][1]);
            kinds.push_back(kind);
            ids.push_back(id);
        }
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
    const auto opts = DistanceGradientOptions::from_env();

    auto surface = make_two_body_surface_mesh(opts);
    SFEM_TEST_ASSERT(surface != nullptr);
    SFEM_TEST_ASSERT(surface->block(0)->n_nodes_per_element() == 3);

    auto top_nodes = classify_top_nodes(surface, opts.top_body_y_min);
    SFEM_TEST_ASSERT(std::any_of(top_nodes.begin(), top_nodes.end(), [](const unsigned char v) { return v != 0; }));
    SFEM_TEST_ASSERT(std::any_of(top_nodes.begin(), top_nodes.end(), [](const unsigned char v) { return v == 0; }));

    auto p0 = smesh::astype<real_t>(surface->points());
    auto p1 = smesh::astype<real_t>(surface->points());

    for (ptrdiff_t i = 0; i < surface->n_nodes(); ++i) {
        if (top_nodes[i]) {
            p1->data()[1][i] += opts.translation_y;
        }
    }

    auto                       ccd = sccd::CCD<real_t>::create(surface);
    smesh::SharedBuffer<idx_t> vertex_overlap;
    smesh::SharedBuffer<idx_t> face_overlap;
    smesh::SharedBuffer<idx_t> edge0_overlap;
    smesh::SharedBuffer<idx_t> edge1_overlap;

    ccd->broad_phase(p0, p1, vertex_overlap, face_overlap, edge0_overlap, edge1_overlap);

    const smesh::Path output_dir(opts.output_dir);
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

    auto            faces   = surface->block(0)->elements();
    auto            edges   = ccd->edges();
    const ptrdiff_t n_nodes = surface->n_nodes();

    for (int step = 0; step <= opts.n_steps; ++step) {
        const real_t t = real_t(step) / real_t(opts.n_steps);

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
        std::vector<real_t> bbox_points;
        std::vector<idx_t>  bbox_indices;
        std::vector<idx_t>  bbox_kind;
        std::vector<idx_t>  bbox_id;

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

        const real_t t_prev = step == 0 ? t : real_t(step - 1) / real_t(opts.n_steps);
        for (ptrdiff_t f = 0; f < surface->n_elements(); ++f) {
            std::array<real_t, 3> lo;
            std::array<real_t, 3> hi;
            element_swept_bbox(faces, 3, f, p0, p1, t_prev, t, lo, hi);
            append_bbox_lines(lo, hi, 0, static_cast<idx_t>(f), bbox_points, bbox_indices, bbox_kind, bbox_id);
        }

        for (ptrdiff_t e = 0; e < static_cast<ptrdiff_t>(edges->extent(1)); ++e) {
            std::array<real_t, 3> lo;
            std::array<real_t, 3> hi;
            element_swept_bbox(edges, 2, e, p0, p1, t_prev, t, lo, hi);
            append_bbox_lines(lo, hi, 1, static_cast<idx_t>(e), bbox_points, bbox_indices, bbox_kind, bbox_id);
        }

        for (ptrdiff_t i = 0; i < vertex_overlap->size(); ++i) {
            const idx_t v = vertex_overlap->data()[i];
            const idx_t f = face_overlap->data()[i];
            if (static_cast<bool>(top_nodes[v]) == is_top_face(faces, top_nodes, f)) continue;

            std::array<real_t, 3> v_lo;
            std::array<real_t, 3> v_hi;
            std::array<real_t, 3> f_lo;
            std::array<real_t, 3> f_hi;
            node_swept_bbox(p0, p1, v, t_prev, t, v_lo, v_hi);
            element_swept_bbox(faces, 3, f, p0, p1, t_prev, t, f_lo, f_hi);
            if (!aabb_overlap(v_lo, v_hi, f_lo, f_hi)) continue;

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
            if (static_cast<bool>(top_nodes[a0i]) == static_cast<bool>(top_nodes[b0i])) continue;

            std::array<real_t, 3> e0_lo;
            std::array<real_t, 3> e0_hi;
            std::array<real_t, 3> e1_lo;
            std::array<real_t, 3> e1_hi;
            element_swept_bbox(edges, 2, e0, p0, p1, t_prev, t, e0_lo, e0_hi);
            element_swept_bbox(edges, 2, e1, p0, p1, t_prev, t, e1_lo, e1_hi);
            if (!aabb_overlap(e0_lo, e0_hi, e1_lo, e1_hi)) continue;

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
                           ax + opts.edge_gradient_scale * nx,
                           ay + opts.edge_gradient_scale * ny,
                           az + opts.edge_gradient_scale * nz);
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
                           bx - opts.edge_gradient_scale * nx,
                           by - opts.edge_gradient_scale * ny,
                           bz - opts.edge_gradient_scale * nz);
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
        SFEM_TEST_ASSERT(write_vector(timestep_path(fields_dir, "bbox_points", step), bbox_points) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(write_vector(timestep_typed_path<idx_t>(fields_dir, "bbox_indices", step), bbox_indices) ==
                         SFEM_SUCCESS);
        SFEM_TEST_ASSERT(write_vector(timestep_typed_path<idx_t>(fields_dir, "bbox_kind", step), bbox_kind) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(write_vector(timestep_typed_path<idx_t>(fields_dir, "bbox_id", step), bbox_id) == SFEM_SUCCESS);

        ee_closest_indices.resize(ee_closest_distance.size());
        for (size_t i = 0; i < ee_closest_indices.size(); ++i) {
            ee_closest_indices[i] = static_cast<idx_t>(i);
        }
        SFEM_TEST_ASSERT(write_vector(timestep_typed_path<idx_t>(fields_dir, "ee_closest_indices", step), ee_closest_indices) ==
                         SFEM_SUCCESS);

        SFEM_TEST_ASSERT(write_vector(timestep_path(fields_dir, "pt_distance", step), pt_distance) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(write_vector(timestep_path(fields_dir, "ee_distance", step), ee_distance) == SFEM_SUCCESS);
    }

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
