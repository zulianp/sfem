#include "sfem_test.hpp"

#include "sfem_SelfCollisions.hpp"

#include "sfem_aliases.hpp"
#include "sfem_context.hpp"
#include "smesh_mesh.hpp"

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

    auto elements = smesh::create_host_buffer<smesh::idx_t>(3, 2);
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

    self_collisions->find(1, disp0, disp1);
    const size_t baseline_vertex_face = self_collisions->vertex_to_face().first->size();
    const size_t baseline_edge_edge   = self_collisions->edge_to_edge().first->size();

    SFEM_TEST_EQ(self_collisions->vertex_to_face().second->size(), baseline_vertex_face);
    SFEM_TEST_EQ(self_collisions->edge_to_edge().second->size(), baseline_edge_edge);

    for (smesh::idx_t i = 3; i < 6; ++i) {
        disp1[0]->data()[i] = -1.75;
    }

    self_collisions->find(1, disp0, disp1);

    const size_t n_vertex_face = self_collisions->vertex_to_face().first->size();
    const size_t n_edge_edge   = self_collisions->edge_to_edge().first->size();

    SFEM_TEST_ASSERT(n_vertex_face + n_edge_edge > baseline_vertex_face + baseline_edge_edge);
    SFEM_TEST_EQ(self_collisions->vertex_to_face().second->size(), n_vertex_face);
    SFEM_TEST_EQ(self_collisions->edge_to_edge().second->size(), n_edge_edge);

    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_self_collisions_find_candidates);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
