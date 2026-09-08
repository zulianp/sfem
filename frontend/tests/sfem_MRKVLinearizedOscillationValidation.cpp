#include "sfem_test.hpp"

#include "sfem_API.hpp"
#include "sfem_Function.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace {

    void set_material_parameter(const std::shared_ptr<sfem::Op>   &op,
                                const std::shared_ptr<sfem::Mesh> &mesh,
                                const char *const                  name,
                                const real_t                       value) {
        for (const auto &block : mesh->blocks()) {
            op->set_value_in_block(block->name(), name, value);
        }
    }

    real_t dot(const ptrdiff_t n, const real_t *const SFEM_RESTRICT x, const real_t *const SFEM_RESTRICT y) {
        real_t ret = 0;
#pragma omp parallel for reduction(+ : ret)
        for (ptrdiff_t i = 0; i < n; ++i) {
            ret += x[i] * y[i];
        }
        return ret;
    }

    real_t mass_dot(const ptrdiff_t n,
                    const real_t *const SFEM_RESTRICT mass,
                    const real_t *const SFEM_RESTRICT x,
                    const real_t *const SFEM_RESTRICT y) {
        real_t ret = 0;
#pragma omp parallel for reduction(+ : ret)
        for (ptrdiff_t i = 0; i < n; ++i) {
            ret += mass[i] * x[i] * y[i];
        }
        return ret;
    }

    void fill_shear_mode(const std::shared_ptr<sfem::Mesh> &mesh, const real_t L, real_t *const SFEM_RESTRICT phi) {
        const ptrdiff_t n_nodes = mesh->n_nodes();
        auto            points  = mesh->points()->data();
        const geom_t   *x       = points[0];
        const real_t    pi      = std::acos(real_t(-1));

#pragma omp parallel for
        for (ptrdiff_t node = 0; node < n_nodes; ++node) {
            phi[3 * node + 0] = 0;
            phi[3 * node + 1] = std::sin(pi * x[node] / L);
            phi[3 * node + 2] = 0;
        }
    }

    void scale_field(const ptrdiff_t n,
                     const real_t    scale,
                     const real_t *const SFEM_RESTRICT in,
                     real_t *const SFEM_RESTRICT       out) {
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n; ++i) {
            out[i] = scale * in[i];
        }
    }

    real_t exact_underdamped_q(const real_t q0,
                               const real_t v0,
                               const real_t omega0,
                               const real_t delta,
                               const real_t t) {
        const real_t omega_d = std::sqrt(omega0 * omega0 - delta * delta);
        return std::exp(-delta * t) *
               (q0 * std::cos(omega_d * t) + ((v0 + delta * q0) / omega_d) * std::sin(omega_d * t));
    }

    int check_linearized_shear_oscillator(const smesh::ElemType element_type) {
        constexpr real_t L     = 1.0;
        constexpr real_t area  = 1.0;
        constexpr real_t mu    = 3.0;
        constexpr real_t rho   = 2.0;
        constexpr real_t eta_s = 0.02;
        constexpr real_t amp   = 1e-4;
        constexpr real_t v_amp = 1e-7;

        const int nx = element_type == smesh::HEX27 ? 8 : 24;
        auto      mesh =
                sfem::Mesh::create_cube(sfem::Communicator::self(), element_type, nx, 1, 1, 0, 0, 0, L, 1, 1);
        auto space = sfem::FunctionSpace::create(mesh, 3);

        const ptrdiff_t ndofs = space->n_dofs();
        auto            phi   = sfem::create_host_buffer<real_t>(ndofs);
        auto            state = sfem::create_host_buffer<real_t>(ndofs);
        auto            prev  = sfem::create_host_buffer<real_t>(ndofs);
        auto            grad  = sfem::create_host_buffer<real_t>(ndofs);
        auto            mass  = sfem::create_host_buffer<real_t>(ndofs);

        fill_shear_mode(mesh, L, phi->data());

        auto lumped_mass = sfem::create_op(space, "LumpedMass", sfem::EXECUTION_SPACE_HOST);
        SFEM_TEST_ASSERT(lumped_mass != nullptr);
        SFEM_TEST_ASSERT(lumped_mass->initialize() == SFEM_SUCCESS);
        std::fill(mass->data(), mass->data() + ndofs, real_t(0));
        SFEM_TEST_ASSERT(lumped_mass->hessian_diag(nullptr, mass->data()) == SFEM_SUCCESS);

        auto op = sfem::create_op(space, "GeneratedMooneyRivlinKelvinVoigtNewmark", sfem::EXECUTION_SPACE_HOST);
        SFEM_TEST_ASSERT(op != nullptr);

        set_material_parameter(op, mesh, "mu", mu);
        set_material_parameter(op, mesh, "lmbda", real_t(0));
        set_material_parameter(op, mesh, "eta_s", real_t(0));
        set_material_parameter(op, mesh, "eta_b", real_t(0));
        set_material_parameter(op, mesh, "newmark_velocity_alpha", real_t(0));

        std::fill(prev->data(), prev->data() + ndofs, real_t(0));
        scale_field(ndofs, amp, phi->data(), state->data());
        std::fill(grad->data(), grad->data() + ndofs, real_t(0));
        op->set_field("previous", prev, 0);
        SFEM_TEST_ASSERT(op->gradient(state->data(), grad->data()) == SFEM_SUCCESS);

        const real_t modal_stiffness = dot(ndofs, phi->data(), grad->data()) / amp;

        set_material_parameter(op, mesh, "mu", mu);
        set_material_parameter(op, mesh, "lmbda", real_t(0));
        set_material_parameter(op, mesh, "eta_s", eta_s);
        set_material_parameter(op, mesh, "eta_b", real_t(0));
        set_material_parameter(op, mesh, "newmark_velocity_alpha", real_t(0));

        std::fill(state->data(), state->data() + ndofs, real_t(0));
        scale_field(ndofs, v_amp, phi->data(), prev->data());
        std::fill(grad->data(), grad->data() + ndofs, real_t(0));
        SFEM_TEST_ASSERT(op->gradient(state->data(), grad->data()) == SFEM_SUCCESS);

        const real_t modal_damping = dot(ndofs, phi->data(), grad->data()) / v_amp;
        const real_t modal_mass    = rho * mass_dot(ndofs, mass->data(), phi->data(), phi->data());

        const real_t pi                  = std::acos(real_t(-1));
        const real_t continuum_stiffness = 4 * mu * area * pi * pi / (2 * L);
        const real_t continuum_damping   = eta_s * area * pi * pi / (2 * L);
        const real_t continuum_mass      = rho * area * L / 2;

        const real_t rel_k = std::abs(modal_stiffness - continuum_stiffness) / continuum_stiffness;
        const real_t rel_c = std::abs(modal_damping - continuum_damping) / continuum_damping;
        const real_t rel_m = std::abs(modal_mass - continuum_mass) / continuum_mass;

        std::printf("linearized oscillator %s: m=%.8e k=%.8e c=%.8e rel(m,k,c)=(%.3e, %.3e, %.3e)\n",
                    sfem::type_to_string(element_type),
                    (double)modal_mass,
                    (double)modal_stiffness,
                    (double)modal_damping,
                    (double)rel_m,
                    (double)rel_k,
                    (double)rel_c);

        SFEM_TEST_ASSERT(modal_mass > 0);
        SFEM_TEST_ASSERT(modal_stiffness > 0);
        SFEM_TEST_ASSERT(modal_damping > 0);
        SFEM_TEST_ASSERT(rel_m < real_t(2e-2));
        SFEM_TEST_ASSERT(rel_k < real_t(5e-2));
        SFEM_TEST_ASSERT(rel_c < real_t(5e-2));

        const real_t omega0 = std::sqrt(modal_stiffness / modal_mass);
        const real_t delta  = modal_damping / (2 * modal_mass);
        SFEM_TEST_ASSERT(delta < omega0);

        const real_t beta  = 0.25;
        const real_t gamma = 0.5;
        const real_t T     = 2 * pi / omega0;
        const real_t dt    = T / 2000;
        const int    steps = 2000;

        const real_t alpha_a = 1 / (beta * dt * dt);
        const real_t alpha_v = gamma / (beta * dt);

        real_t q = 1;
        real_t v = 0;
        real_t a = -(modal_damping * v + modal_stiffness * q) / modal_mass;

        real_t max_error = 0;
        for (int step = 1; step <= steps; ++step) {
            const real_t q_hat = q + dt * v + dt * dt * (real_t(0.5) - beta) * a;
            const real_t z     = v + dt * (1 - gamma) * a - alpha_v * q_hat;
            const real_t q_new =
                    (modal_mass * alpha_a * q_hat - modal_damping * z) /
                    (modal_stiffness + modal_damping * alpha_v + modal_mass * alpha_a);
            const real_t v_new = alpha_v * q_new + z;
            const real_t a_new = alpha_a * (q_new - q_hat);

            q = q_new;
            v = v_new;
            a = a_new;

            const real_t t       = step * dt;
            const real_t q_exact = exact_underdamped_q(real_t(1), real_t(0), omega0, delta, t);
            max_error            = std::max(max_error, std::abs(q - q_exact));
        }

        std::printf("linearized oscillator %s: omega0=%.8e delta=%.8e max_q_error=%.8e\n",
                    sfem::type_to_string(element_type),
                    (double)omega0,
                    (double)delta,
                    (double)max_error);

        SFEM_TEST_ASSERT(max_error < real_t(2e-5));
        return SFEM_TEST_SUCCESS;
    }

}  // namespace

int test_linearized_shear_oscillator_hex8() { return check_linearized_shear_oscillator(smesh::HEX8); }

int test_linearized_shear_oscillator_hex27() { return check_linearized_shear_oscillator(smesh::HEX27); }

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_linearized_shear_oscillator_hex8);
    SFEM_RUN_TEST(test_linearized_shear_oscillator_hex27);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
