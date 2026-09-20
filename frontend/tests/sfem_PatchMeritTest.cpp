// The node-centric merit kernel, against the merit computed the long way.
//
// Everything about this kernel so far has been structural: the shape of the
// emitted text, the consistency of the incidence, the parity of a permutation.
// None of that says it computes `1/2 * ||g + R(x + alpha*h)||^2`.  Three places
// in it can produce a plausible wrong number without any of the earlier checks
// noticing -- the orientation, which decides which basis function each element
// contracts against; the seeding, which decides whether the square is over the
// whole residual or one operator's share; and the per-step lanes, which decide
// whether a step reads its own alpha.  This is the test for that.

#include "sfem_test.hpp"

#include "sfem_API.hpp"
#include "sfem_Function.hpp"
#include "sfem_GeneratedMooneyRivlinKelvinVoigt_c_abi.hpp"
#include "sfem_OpFactory.hpp"
#include "sfem_PatchIncidence.hpp"

#include "reference/tet4_q1.hpp"
#include "reference/quad_tet_q1.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

    //: What agreement between the two paths is worth asking for.
    //:
    //: Not round-off in `real_t`.  The two carry the element geometry at
    //: different precisions: the assembled path reads an adjugate cached as
    //: `geom_t`, which this tree builds as `float`, while the patch kernel
    //: forms the Jacobian in `s_t` from the same `float` coordinates and keeps
    //: it there.  So they agree to about `float` epsilon, and asking for more
    //: would be asking the patch kernel to reproduce the cached value's
    //: rounding rather than to be right.
    static const real_t kGeometryTolerance = real_t(1e-6);

    static const real_t kMu       = real_t(3);
    static const real_t kLambda   = real_t(1);
    static const real_t kEtaS     = real_t(0.05);
    static const real_t kEtaB     = real_t(0.01);
    static const real_t kDtShift  = real_t(2);
    static const int    kBlock    = 3;
    //: At most the kernel's vector width, which is what the caller rounds to.
    static const int    kSteps    = 8;

    void seed(const ptrdiff_t n, const int salt, real_t *const v) {
        for (ptrdiff_t i = 0; i < n; ++i) {
            v[i] = real_t(1e-3) * std::sin(real_t(0.5 * (i + 1) + salt));
        }
    }

    struct Fixture {
        std::shared_ptr<sfem::Mesh>          mesh;
        std::shared_ptr<sfem::FunctionSpace> space;
        std::shared_ptr<sfem::Function>      f;
        std::shared_ptr<sfem::Op>            op;
        std::shared_ptr<sfem::PatchIncidence> patch;
        sfem::SharedBuffer<real_t>           x, h, previous;
        ptrdiff_t                            ndofs{0};
    };

    Fixture make(const int resolution = 3) {
        Fixture fx;
        fx.mesh  = sfem::Mesh::create_tet4_cube(
                sfem::Communicator::self(), resolution, resolution, resolution, 0, 0, 0, 1, 1, 1);
        fx.space = sfem::FunctionSpace::create(fx.mesh, kBlock);
        fx.ndofs = fx.space->n_dofs();

        fx.op = sfem::Factory::create_op(fx.space, "GeneratedMooneyRivlinKelvinVoigt");
        if (!fx.op) { fprintf(stderr, "no MooneyRivlinKelvinVoigt op\n"); std::abort(); }
        // Affine by construction for a P1 simplex, and the kernel reads the
        // cached adjugate, so it has to be built.
        fx.op->set_option("ASSUME_AFFINE", true);
        for (const auto &block : fx.mesh->blocks()) {
            fx.op->set_value_in_block(block->name(), "mu", kMu);
            fx.op->set_value_in_block(block->name(), "lmbda", kLambda);
            fx.op->set_value_in_block(block->name(), "eta_s", kEtaS);
            fx.op->set_value_in_block(block->name(), "eta_b", kEtaB);
            fx.op->set_value_in_block(block->name(), "u_dt_shift", kDtShift);
        }

        fx.previous = sfem::create_host_buffer<real_t>(fx.ndofs);
        fx.x        = sfem::create_host_buffer<real_t>(fx.ndofs);
        fx.h        = sfem::create_host_buffer<real_t>(fx.ndofs);
        seed(fx.ndofs, 9, fx.previous->data());
        seed(fx.ndofs, 1, fx.x->data());
        seed(fx.ndofs, 2, fx.h->data());
        fx.op->set_field("previous", fx.previous, 0);

        fx.f = sfem::Function::create(fx.space);
        fx.f->add_operator(fx.op);
        fx.patch = sfem::build_patch_incidence(fx.mesh);
        return fx;
    }

    /// `1/2 * ||accumulator + R(x + alpha*h)||^2`, assembled and reduced.
    real_t merit_the_long_way(Fixture &fx, const real_t alpha, const real_t *const accumulator) {
        auto stepped  = sfem::create_host_buffer<real_t>(fx.ndofs);
        auto residual = sfem::create_host_buffer<real_t>(fx.ndofs);
        for (ptrdiff_t i = 0; i < fx.ndofs; ++i) {
            stepped->data()[i] = fx.x->data()[i] + alpha * fx.h->data()[i];
        }
        SFEM_TEST_ASSERT(fx.f->gradient(stepped->data(), residual->data()) == SFEM_SUCCESS);
        real_t acc = 0;
        for (ptrdiff_t i = 0; i < fx.ndofs; ++i) {
            const real_t value = accumulator[i] + residual->data()[i];
            acc += value * value;
        }
        return real_t(0.5) * acc;
    }

    int run_patch_kernel(Fixture &fx, const std::vector<real_t> &steps,
                         const real_t *const accumulator, real_t *const merit) {
        const void *grad_ref[3] = {
                sfem::codegen::ref_tet4_q1<real_t>::grad_ref_x(),
                sfem::codegen::ref_tet4_q1<real_t>::grad_ref_y(),
                sfem::codegen::ref_tet4_q1<real_t>::grad_ref_z(),
        };
        return mooney_rivlin_kelvin_voigt_total_merit_patch_3d_a_msoa(
                smesh::TET4,
                smesh::TypeToEnum<real_t>::value(),
                fx.patch->n_nodes(),
                fx.patch->node_ptr->data(),
                fx.patch->element->data(),
                fx.patch->element_local->data(),
                fx.mesh->elements(0)->data(),
                const_cast<const geom_t *const *>(fx.mesh->points()->data()),
                sfem::codegen::ref_tet4_q1<real_t>::shape(),
                grad_ref,
                sfem::codegen::quad_tet_q1<real_t>::q_weight(),
                kEtaB, kEtaS, kLambda, kMu, kDtShift,
                (int)steps.size(),
                steps.data(),
                fx.x->data(),
                fx.h->data(),
                fx.previous->data(),
                accumulator,
                merit);
    }

    /// The merit at every sampled step, against the assembled one.
    int test_the_patch_merit_matches_the_assembled_merit() {
        auto fx = make();
        std::vector<real_t> steps(kSteps);
        for (int k = 0; k < kSteps; ++k) {
            steps[k] = real_t(-1) + real_t(2.0 * k) / real_t(kSteps - 1);
        }
        std::vector<real_t> zero((size_t)fx.ndofs, 0);
        std::vector<real_t> merit((size_t)kSteps, 0);

        SFEM_TEST_ASSERT(run_patch_kernel(fx, steps, zero.data(), merit.data()) == SFEM_SUCCESS);

        for (int k = 0; k < kSteps; ++k) {
            const real_t reference = merit_the_long_way(fx, steps[k], zero.data());
            SFEM_TEST_ASSERT(reference > 0);
            const real_t error = std::abs(merit[k] - reference) / reference;
            if (error > kGeometryTolerance) {
                fprintf(stderr,
                        "step %d (alpha %g): patch %.17g against assembled %.17g, rel %g\n",
                        k, (double)steps[k], (double)merit[k], (double)reference, (double)error);
            }
            SFEM_TEST_ASSERT(error <= kGeometryTolerance);
        }
        return SFEM_TEST_SUCCESS;
    }

    /// The accumulator really is inside the square.
    ///
    /// Seeding `rho` with it is what makes the merit the system's rather than
    /// this operator's share.  With a non-zero accumulator the answer must move,
    /// and must still match the long way round.
    int test_the_accumulator_is_inside_the_square() {
        auto fx = make();
        std::vector<real_t> steps{real_t(0), real_t(0.25)};
        std::vector<real_t> zero((size_t)fx.ndofs, 0);
        std::vector<real_t> g((size_t)fx.ndofs, 0);
        seed(fx.ndofs, 5, g.data());

        std::vector<real_t> without((size_t)steps.size(), 0);
        std::vector<real_t> with((size_t)steps.size(), 0);
        SFEM_TEST_ASSERT(run_patch_kernel(fx, steps, zero.data(), without.data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(run_patch_kernel(fx, steps, g.data(), with.data()) == SFEM_SUCCESS);

        for (size_t k = 0; k < steps.size(); ++k) {
            const real_t reference = merit_the_long_way(fx, steps[k], g.data());
            SFEM_TEST_ASSERT(std::abs(with[k] - reference) <= kGeometryTolerance * reference);
            // Otherwise the seeding could be a no-op and the test above would
            // still pass.
            SFEM_TEST_ASSERT(std::abs(with[k] - without[k]) > kGeometryTolerance * with[k]);
        }
        return SFEM_TEST_SUCCESS;
    }

    /// Each lane reads its own alpha.
    ///
    /// A kernel that broadcast one step across the lanes, or indexed `steps`
    /// by the element, would still match at a single step.  Asking for the
    /// same alpha twice in different lanes and for distinct alphas in between
    /// catches both.
    int test_every_lane_carries_its_own_step() {
        auto fx = make();
        std::vector<real_t> steps{real_t(0.25), real_t(-0.5), real_t(0.25), real_t(1)};
        std::vector<real_t> zero((size_t)fx.ndofs, 0);
        std::vector<real_t> merit((size_t)steps.size(), 0);
        SFEM_TEST_ASSERT(run_patch_kernel(fx, steps, zero.data(), merit.data()) == SFEM_SUCCESS);

        // Repeated alphas agree exactly; distinct ones do not.
        SFEM_TEST_ASSERT(merit[0] == merit[2]);
        SFEM_TEST_ASSERT(std::abs(merit[0] - merit[1]) > real_t(1e-12) * merit[0]);
        SFEM_TEST_ASSERT(std::abs(merit[0] - merit[3]) > real_t(1e-12) * merit[0]);
        for (size_t k = 0; k < steps.size(); ++k) {
            const real_t reference = merit_the_long_way(fx, steps[k], zero.data());
            SFEM_TEST_ASSERT(std::abs(merit[k] - reference) <= kGeometryTolerance * reference);
        }
        return SFEM_TEST_SUCCESS;
    }

}  // namespace

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_the_patch_merit_matches_the_assembled_merit);
    SFEM_RUN_TEST(test_the_accumulator_is_inside_the_square);
    SFEM_RUN_TEST(test_every_lane_carries_its_own_step);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
