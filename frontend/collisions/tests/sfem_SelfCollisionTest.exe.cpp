#include "sfem_test.hpp"

#include "sfem_SelfCollisions.hpp"

#include "sfem_aliases.hpp"
#include "sfem_context.hpp"
#include "smesh_mesh.hpp"

#include <algorithm>
#include <array>
#include <utility>
#include <vector>

namespace {

    std::shared_ptr<smesh::Mesh> make_surface_mesh() {
        auto points = smesh::create_host_buffer<smesh::geom_t>(3, 6);

        points->data()[0][0] = 0.0;
        points->data()[1][0] = 0.0;
        points->data()[2][0] = 0.0;

        points->data()[0][1] = 1.0;
        points->data()[1][1] = 0.0;
        points->data()[2][1] = 0.0;

        points->data()[0][2] = 0.0;
        points->data()[1][2] = 1.0;
        points->data()[2][2] = 0.0;

        points->data()[0][3] = 2.0;
        points->data()[1][3] = 0.25;
        points->data()[2][3] = -0.25;

        points->data()[0][4] = 2.0;
        points->data()[1][4] = 1.25;
        points->data()[2][4] = -0.25;

        points->data()[0][5] = 2.0;
        points->data()[1][5] = 0.25;
        points->data()[2][5] = 0.75;

        auto elements          = smesh::create_host_buffer<smesh::idx_t>(3, 2);
        elements->data()[0][0] = 0;
        elements->data()[1][0] = 1;
        elements->data()[2][0] = 2;

        elements->data()[0][1] = 3;
        elements->data()[1][1] = 4;
        elements->data()[2][1] = 5;

        return std::make_shared<smesh::Mesh>(smesh::Communicator::self(), smesh::TRI3, elements, points);
    }

    std::vector<smesh::SharedBuffer<real_t>> make_displacement(const ptrdiff_t n_nodes) {
        std::vector<smesh::SharedBuffer<real_t>> disp(3);
        for (int d = 0; d < 3; ++d) {
            disp[d] = smesh::create_host_buffer<real_t>(n_nodes);
            for (ptrdiff_t i = 0; i < n_nodes; ++i) {
                disp[d]->data()[i] = 0;
            }
        }

        return disp;
    }

    std::array<const real_t*, 3> make_displacement_view(const std::vector<smesh::SharedBuffer<real_t>>& disp) {
        return {disp[0]->data(), disp[1]->data(), disp[2]->data()};
    }

    std::vector<std::pair<smesh::idx_t, smesh::idx_t>> collect_pairs(const sfem::CollisionPairs& pairs) {
        std::vector<std::pair<smesh::idx_t, smesh::idx_t>> ret(pairs.first->size());
        for (size_t i = 0; i < ret.size(); ++i) {
            ret[i] = {pairs.first->data()[i], pairs.second->data()[i]};
        }

        std::sort(ret.begin(), ret.end());
        return ret;
    }

}  // namespace

int test_self_collisions_find_candidates() {
    auto surface = make_surface_mesh();
    SFEM_TEST_ASSERT(surface != nullptr);
    SFEM_TEST_EQ(surface->spatial_dimension(), 3);
    SFEM_TEST_EQ(surface->element_type(0), smesh::TRI3);

    auto self_collisions = sfem::SelfCollisions::create(surface);
    SFEM_TEST_ASSERT(self_collisions != nullptr);

    auto disp0 = make_displacement(surface->n_nodes());
    auto disp1 = make_displacement(surface->n_nodes());
    auto disp0_view = make_displacement_view(disp0);
    auto disp1_view = make_displacement_view(disp1);

    self_collisions->find(1, disp0_view.data(), disp1_view.data());
    const size_t baseline_vertex_face = self_collisions->vertex_to_face().first->size();
    const size_t baseline_edge_edge   = self_collisions->edge_to_edge().first->size();

    SFEM_TEST_EQ(self_collisions->vertex_to_face().second->size(), baseline_vertex_face);
    SFEM_TEST_EQ(self_collisions->edge_to_edge().second->size(), baseline_edge_edge);

    for (smesh::idx_t i = 3; i < 6; ++i) {
        disp1[0]->data()[i] = -1.75;
    }

    self_collisions->find(1, disp0_view.data(), disp1_view.data());

    const auto vertex_face_pairs = collect_pairs(self_collisions->vertex_to_face());
    const auto edge_edge_pairs   = collect_pairs(self_collisions->edge_to_edge());

    const std::vector<std::pair<smesh::idx_t, smesh::idx_t>> expected_edge_edge = {
            {0, 5}, {0, 7}, {5, 11}, {5, 14}, {5, 15}, {5, 16}, {7, 11}, {7, 14}, {7, 15}, {7, 16}, {9, 14}, {9, 16}};

    SFEM_TEST_EQ(vertex_face_pairs.size(), (size_t)0);
    SFEM_TEST_EQ(edge_edge_pairs.size(), expected_edge_edge.size());
    SFEM_TEST_ASSERT(edge_edge_pairs.size() + vertex_face_pairs.size() > baseline_vertex_face + baseline_edge_edge);
    for (size_t i = 0; i < expected_edge_edge.size(); ++i) {
        SFEM_TEST_EQ(edge_edge_pairs[i].first, expected_edge_edge[i].first);
        SFEM_TEST_EQ(edge_edge_pairs[i].second, expected_edge_edge[i].second);
    }

    return SFEM_TEST_SUCCESS;
}

int main(int argc, char* argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_self_collisions_find_candidates);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
