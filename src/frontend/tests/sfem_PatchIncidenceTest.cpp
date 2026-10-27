// The elements incident on each node, and where the node sits in each.
//
// A node-centric patch kernel contracts against the basis function belonging
// to the node it is visiting, and it finds that basis function by presenting
// the element with the node's slot brought to the front.  So the slot has to
// be right: a wrong one contracts against a different basis function and
// produces a plausible, wrong residual with nothing to indicate it.

#include "sfem_test.hpp"

#include "sfem_API.hpp"
#include "sfem_PatchIncidence.hpp"

#include <set>
#include <vector>

namespace {

    int test_every_incidence_names_the_right_slot() {
        auto mesh = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 3, 3, 3);
        auto patch = sfem::build_patch_incidence(mesh);
        SFEM_TEST_ASSERT(patch != nullptr);

        const auto      elements = mesh->elements(0)->data();
        const int       nxe      = mesh->block(0)->n_nodes_per_element();
        const ptrdiff_t n_nodes  = patch->n_nodes();
        SFEM_TEST_ASSERT(n_nodes == mesh->n_nodes());

        for (ptrdiff_t node = 0; node < n_nodes; ++node) {
            const count_t begin = patch->node_ptr->data()[node];
            const count_t end   = patch->node_ptr->data()[node + 1];
            SFEM_TEST_ASSERT(begin <= end);
            for (count_t k = begin; k < end; ++k) {
                const auto element = patch->element->data()[k];
                const int  slot    = (int)patch->element_local->data()[k];
                SFEM_TEST_ASSERT(slot >= 0 && slot < nxe);
                // The property the kernel rests on.
                SFEM_TEST_ASSERT((ptrdiff_t)elements[slot][element] == node);
            }
        }
        return SFEM_TEST_SUCCESS;
    }

    /// Every element appears once per node it carries, and no more.
    ///
    /// Without this the test above would pass for an incidence that had
    /// dropped elements: each surviving entry would still name a correct slot,
    /// and a node's residual would simply be missing contributions.
    int test_the_incidence_is_complete() {
        auto mesh  = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 3, 3, 3);
        auto patch = sfem::build_patch_incidence(mesh);

        const int       nxe       = mesh->block(0)->n_nodes_per_element();
        const ptrdiff_t nelements = mesh->n_elements();
        SFEM_TEST_ASSERT(patch->n_incidences() == nelements * nxe);

        // And each (element, slot) pair occurs exactly once across all nodes.
        std::vector<int> seen((size_t)nelements * nxe, 0);
        for (count_t k = 0; k < (count_t)patch->n_incidences(); ++k) {
            const auto element = patch->element->data()[k];
            const int  slot    = (int)patch->element_local->data()[k];
            seen[(size_t)element * nxe + slot] += 1;
        }
        for (size_t i = 0; i < seen.size(); ++i) {
            SFEM_TEST_ASSERT(seen[i] == 1);
        }
        return SFEM_TEST_SUCCESS;
    }

    /// An interior node of a hex mesh touches eight elements.
    ///
    /// A shape check on the adjacency itself, so the two tests above cannot
    /// both pass on an incidence that is self-consistent and wrong.
    int test_an_interior_node_has_the_expected_valence() {
        auto mesh  = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 4, 4, 4);
        auto patch = sfem::build_patch_incidence(mesh);

        int interior = 0;
        for (ptrdiff_t node = 0; node < patch->n_nodes(); ++node) {
            const count_t valence =
                    patch->node_ptr->data()[node + 1] - patch->node_ptr->data()[node];
            SFEM_TEST_ASSERT(valence >= 1);
            SFEM_TEST_ASSERT(valence <= 8);
            interior += (valence == 8);
        }
        // A 4x4x4 cube of hexes has 3x3x3 interior nodes.
        SFEM_TEST_ASSERT(interior == 27);
        return SFEM_TEST_SUCCESS;
    }

}  // namespace

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_every_incidence_names_the_right_slot);
    SFEM_RUN_TEST(test_the_incidence_is_complete);
    SFEM_RUN_TEST(test_an_interior_node_has_the_expected_valence);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
