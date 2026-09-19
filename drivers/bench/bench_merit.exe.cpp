// The merit -- `Function::value` -- measured where it actually runs.
//
// `bench_linear_elasticity.exe` times the merit on the host across the standard
// and packed layouts.  This one exists for the axis that one cannot express:
// the device.  There is no packed device kernel to compare against -- the
// generated device tree carries no packed variant, because the thread-local
// scratch the packed traversal needs has no device meaning -- so the comparison
// here is host against device for the standard layout.
//
// Both merit reductions are measured, because they are different computations:
//
//   element-wise  an energy or a recovered potential, summed over elements by
//                 the objective kernels.  It does not assemble a residual.
//   node-wise     `1/2*||R||^2` over the residual the Function assembles, so it
//                 pays for a `gradient` and then reduces.  On the host that
//                 reduction is deterministic by construction; on the device it
//                 is the vendor's dot product, so the two agree to a tolerance
//                 rather than bit for bit, and that difference is reported
//                 rather than hidden behind a pass.
#include "sfem_API.hpp"
#include "sfem_Function.hpp"
#include "sfem_OpFactory.hpp"

#include <cmath>
#include <limits>
#include <cstdio>
#include <string>
#include <vector>

namespace {

    void print_rate(const char     *name,
                    const double    elapsed,
                    const ptrdiff_t nelements,
                    const ptrdiff_t ndofs,
                    const int       repeat,
                    const real_t    value) {
        const double seconds_per_call = elapsed / repeat;
        printf("%-34s %12.6e %14.3f %14.3f %26.17g\n",
               name,
               seconds_per_call,
               1e-6 * static_cast<double>(nelements) / seconds_per_call,
               1e-6 * static_cast<double>(ndofs) / seconds_per_call,
               static_cast<double>(value));
        fflush(stdout);
    }

    double time_value(const std::shared_ptr<sfem::Function> &f,
                      const real_t *const                    x,
                      const int                              warmup,
                      const int                              repeat,
                      real_t                                &value_out) {
        for (int i = 0; i < warmup; ++i) {
            real_t warm = 0;
            if (f->has_energy_merit()) f->energy_merit(x, &warm); else f->residual_merit(x, &warm);
        }
        sfem::device_synchronize();
        const double t0 = MPI_Wtime();
        real_t       value = 0;
        for (int i = 0; i < repeat; ++i) {
            value = 0;
            if (f->has_energy_merit()) f->energy_merit(x, &value); else f->residual_merit(x, &value);
        }
        sfem::device_synchronize();
        const double elapsed = MPI_Wtime() - t0;
        value_out            = value;
        return elapsed;
    }

    // The gradient beside the merit, because the two fail differently and the
    // difference is the diagnosis.  An element-wise merit never assembles a
    // residual, so if the merit disagrees with the host while the gradient
    // agrees, the objective path is at fault rather than the device kernels in
    // general -- and the other way round if both disagree.
    real_t gradient_norm(const std::shared_ptr<sfem::Function> &f,
                         const real_t *const                    x,
                         const sfem::ExecutionSpace             es) {
        const ptrdiff_t ndofs    = f->space()->n_dofs();
        auto            residual = sfem::create_buffer<real_t>(ndofs, es);
        if (f->gradient(x, residual->data()) != SFEM_SUCCESS) {
            return std::numeric_limits<real_t>::quiet_NaN();
        }
        // Ordered against the kernels that filled it -- see
        // `Function::node_wise_merit` for what reading early looks like.
        sfem::device_synchronize();
        auto         blas = sfem::blas<real_t>(es);
        const real_t norm = blas->norm2(ndofs, residual->data());

        // Cross-check the device reduction against a host sum of the very same
        // buffer.  If these disagree the reduction is at fault; if they agree
        // and both differ from the host operator, the kernel is.  Guessing
        // between those two cost several rounds.
        if (es == sfem::EXECUTION_SPACE_DEVICE && smesh::Env::read("SFEM_CHECK_REDUCTION", 0)) {
            auto   back = smesh::to_host(residual);
            double acc  = 0;
            for (ptrdiff_t i = 0; i < ndofs; ++i) {
                acc += back->data()[i] * back->data()[i];
            }
            // Round-trip the very same numbers into a *fresh* device buffer and
            // reduce that.  If the round trip reduces correctly while the
            // original does not, the two are not the same memory -- which is
            // the only remaining way cuBLAS and the host can disagree about a
            // buffer that has been fully synchronised.
            auto again = smesh::to_device(back);
            sfem::device_synchronize();
            const double n_again = static_cast<double>(blas->norm2(ndofs, again->data()));
            printf("# reduction: device norm2 %23.16e  host sum %23.16e  round-tripped norm2 %23.16e\n",
                   static_cast<double>(norm), std::sqrt(acc), n_again);
            printf("# residual is_ptr_device=%d  round-trip is_ptr_device=%d  ndofs=%td\n",
                   (int)sfem::is_ptr_device(residual->data()),
                   (int)sfem::is_ptr_device(again->data()),
                   ndofs);
            fflush(stdout);
        }
        return norm;
    }

#ifdef SFEM_ENABLE_CUDA
    // The BLAS layer on its own, with no generated kernel involved.  A buffer
    // of ones has `norm2 == sqrt(n)` and `dot == n` exactly.  Run before and
    // after the generated kernels, it says whether they left the CUDA context
    // in a state the library can no longer work in -- which is the difference
    // between "the reduction is wrong" and "everything after that kernel is
    // wrong".
    void blas_self_test(const char *const when, const ptrdiff_t ndofs) {
        auto ones = sfem::create_buffer<real_t>(ndofs, sfem::EXECUTION_SPACE_DEVICE);
        auto twos = sfem::create_buffer<real_t>(ndofs, sfem::EXECUTION_SPACE_DEVICE);
        auto blas = sfem::blas<real_t>(sfem::EXECUTION_SPACE_DEVICE);
        blas->values(ndofs, real_t(1), ones->data());
        blas->values(ndofs, real_t(2), twos->data());
        sfem::device_synchronize();
        const double n2 = static_cast<double>(blas->norm2(ndofs, ones->data()));
        // Two distinct buffers: `dot(x, x)` hands cuBLAS the same pointer as
        // both operands, which it does not support, and that is a property of
        // the call rather than of the context it runs in.
        const double dd = static_cast<double>(blas->dot(ndofs, ones->data(), twos->data())) / 2.0;
        // `twos` on its own.  If the first buffer reduces correctly and the
        // second does not, the fault is in allocating or filling the second --
        // not in the reduction that reads them both.
        const double n2b = static_cast<double>(blas->norm2(ndofs, twos->data()));
        auto         bk  = smesh::to_host(twos);
        double       hs  = 0;
        for (ptrdiff_t i = 0; i < ndofs; ++i) hs += bk->data()[i];
        printf("# blas self-test (%s) n=%td: norm2(ones) %23.16e (want %23.16e)  norm2(twos) %23.16e (want %23.16e)\n",
               when, ndofs, n2, std::sqrt((double)ndofs), n2b, 2.0 * std::sqrt((double)ndofs));
        printf("#   dot %23.16e (want %23.16e)  host sum of twos %23.16e (want %23.16e)  ones@%p twos@%p\n",
               dd, (double)ndofs, hs, 2.0 * (double)ndofs, (void *)ones->data(), (void *)twos->data());
        fflush(stdout);
    }
#endif

}  // namespace

int main(int argc, char *argv[]) {
    sfem::Context context(argc, argv);

    const int   resolution = smesh::Env::read("SFEM_BASE_RESOLUTION", 48);
    const int   warmup     = smesh::Env::read("SFEM_WARMUP", 2);
    const int   repeat     = smesh::Env::read("SFEM_REPEAT", 5);
    const auto  element    = smesh::type_from_string(smesh::Env::read_string("SFEM_ELEM_TYPE", "HEX8").c_str());
    const std::string op_name = smesh::Env::read_string("SFEM_OPERATOR", "GeneratedLinearElasticity");
    const int         block   = smesh::Env::read("SFEM_BLOCK_SIZE", 3);

    auto mesh = sfem::Mesh::create_cube(sfem::Communicator::self(),
                                        static_cast<smesh::ElemType>(element),
                                        resolution,
                                        resolution,
                                        resolution,
                                        0, 0, 0, 1, 1, 1);
    auto fs   = sfem::FunctionSpace::create(mesh, block);

    const ptrdiff_t nelements = mesh->n_elements();
    const ptrdiff_t ndofs     = fs->n_dofs();

    auto x_host = sfem::create_host_buffer<real_t>(ndofs);
    auto previous_host = sfem::create_host_buffer<real_t>(ndofs);
    for (ptrdiff_t i = 0; i < ndofs; ++i) {
        previous_host->data()[i] = static_cast<real_t>(1e-3 * ((i % 7) + 1));
        x_host->data()[i]        = previous_host->data()[i] + static_cast<real_t>(1e-4 * ((i % 5) + 1));
    }

    // A Newmark operator refuses to run without the state it steps from, and
    // saying so is the whole point of that refusal -- so the bench supplies it
    // rather than restricting itself to operators that do not ask.
    const bool needs_previous = smesh::Env::read("SFEM_NEEDS_PREVIOUS", op_name.find("Newmark") != std::string::npos);

    printf("#elements %td\n#nodes %td\n#dofs %td\n", nelements, mesh->n_nodes(), ndofs);

#ifdef SFEM_ENABLE_CUDA
    // The BLAS layer on its own, with no generated kernel anywhere near it.
    // A buffer of ones has `norm2 == sqrt(n)` and `dot == n` exactly, so if
    // these disagree the reduction is broken independently of anything this
    // benchmark is measuring.
    if (smesh::Env::read("SFEM_CHECK_BLAS", 0)) {
        blas_self_test("before any generated kernel", ndofs);
    }
#endif
    printf("%-34s %12s %14s %14s %26s\n", "kernel", "[s]", "[MElem/s]", "[MDOF/s]", "value");
    printf("--------------------------------------------------------------------------------------------------------\n");

    real_t host_value    = 0;
    real_t host_gradient = 0;
    {
        auto op = sfem::create_op(fs, op_name.c_str(), sfem::EXECUTION_SPACE_HOST);
        if (!op) {
            SFEM_ERROR("no host operator %s\n", op_name.c_str());
        }
        if (op->initialize() != SFEM_SUCCESS) {
            SFEM_ERROR("could not initialize host %s\n", op_name.c_str());
        }
        if (needs_previous) {
            op->set_field("previous", previous_host, 0);
            op->update(previous_host->data(), x_host->data());
        }
        auto f = sfem::Function::create(fs);
        f->add_operator(op);
        if (!needs_previous) {
            f->update(x_host->data());
        }
        const double elapsed = time_value(f, x_host->data(), warmup, repeat, host_value);
        print_rate(f->has_energy_merit() ? "host_energy_merit" : "host_residual_merit",
                   elapsed, nelements, ndofs, repeat, host_value);
        host_gradient = gradient_norm(f, x_host->data(), sfem::EXECUTION_SPACE_HOST);
        printf("%-34s %12s %14s %14s %26.17g\n", "host_gradient_norm", "", "", "", host_gradient);
    }

#ifdef SFEM_ENABLE_CUDA
    {
        const std::string device_name = "gpu:" + op_name;
        auto              op          = sfem::create_op(fs, device_name.c_str(), sfem::EXECUTION_SPACE_DEVICE);
        if (!op) {
            printf("# no device operator %s -- skipping the device arm\n", device_name.c_str());
        } else {
            if (op->initialize() != SFEM_SUCCESS) {
                SFEM_ERROR("could not initialize device %s\n", device_name.c_str());
            }
            auto x_device = smesh::to_device(x_host);
            // Held for as long as the operator is, not just for the `if`.
            // `update(previous, current)` keeps the raw pointer and drops the
            // buffer it was handed (`impl_->previous_buffer.reset()`), so the
            // only owner left is this variable -- and a block-scoped one frees
            // the device memory the kernels are about to read.
            smesh::SharedBuffer<real_t> previous_device;
            if (needs_previous) {
                previous_device = smesh::to_device(previous_host);
                op->set_field("previous", previous_device, 0);
                op->update(previous_device->data(), x_device->data());
            }
            auto f = sfem::Function::create(fs);
            f->add_operator(op);
            if (!needs_previous) {
                f->update(x_device->data());
            }

            real_t       device_value = 0;
            const double elapsed      = time_value(f, x_device->data(), warmup, repeat, device_value);
            print_rate(f->has_energy_merit() ? "device_energy_merit" : "device_residual_merit",
                       elapsed, nelements, ndofs, repeat, device_value);

            // The device reduction is the vendor's dot product for the node-wise
            // path, so this is a tolerance rather than an equality -- and saying
            // which it is, with the number, beats asserting a pass.
            const double scale    = std::fabs(static_cast<double>(host_value));
            const double relative = scale > 0 ? std::fabs(static_cast<double>(device_value - host_value)) / scale : 0;
            printf("# device against host: relative difference %.3e\n", relative);

            // Does the answer vary *within* one process?  A race gives a
            // different number each call; a wrong-but-stable number points at
            // initialisation or state instead.  The distinction is the whole
            // diagnosis and it costs eight calls to make.
            // Where is the residual, and how exactly does it differ?  A few bad
            // entries point at particular nodes; uniform garbage points at the
            // kernel.  Both beat guessing from a norm.
            if (smesh::Env::read("SFEM_DUMP_RESIDUAL", 0)) {
                auto r_dev  = sfem::create_buffer<real_t>(ndofs, sfem::EXECUTION_SPACE_DEVICE);
                auto r_host = sfem::create_host_buffer<real_t>(ndofs);
                printf("# residual buffer is_ptr_device=%d, x is_ptr_device=%d\n",
                       (int)sfem::is_ptr_device(r_dev->data()),
                       (int)sfem::is_ptr_device(x_device->data()));
                f->gradient(x_device->data(), r_dev->data());
                sfem::device_synchronize();
                auto r_back = smesh::to_host(r_dev);

                auto h_op = sfem::create_op(fs, op_name.c_str(), sfem::EXECUTION_SPACE_HOST);
                h_op->initialize();
                auto hf = sfem::Function::create(fs);
                hf->add_operator(h_op);
                hf->update(x_host->data());
                hf->gradient(x_host->data(), r_host->data());

                ptrdiff_t bad = 0, first_bad = -1;
                double    worst = 0;
                for (ptrdiff_t i = 0; i < ndofs; ++i) {
                    const double a = r_back->data()[i], b = r_host->data()[i];
                    const double d = std::fabs(a - b);
                    if (d > 1e-10 * (1.0 + std::fabs(b))) {
                        if (first_bad < 0) first_bad = i;
                        ++bad;
                    }
                    if (d > worst) worst = d;
                }
                printf("# residual entries differing: %td of %td, first at %td, worst |diff| %.6e\n",
                       bad, ndofs, first_bad, worst);
                for (ptrdiff_t i = 0; i < 6 && i < ndofs; ++i) {
                    printf("#   [%td] device %23.16e  host %23.16e\n", i, r_back->data()[i], r_host->data()[i]);
                }
                fflush(stdout);
            }

            if (smesh::Env::read("SFEM_DUMP_REPEATS", 0)) {
                for (int i = 0; i < 8; ++i) {
                    real_t v = 0;
                    if (f->has_energy_merit()) f->energy_merit(x_device->data(), &v); else f->residual_merit(x_device->data(), &v);
                    sfem::device_synchronize();
                    const real_t g = gradient_norm(f, x_device->data(), sfem::EXECUTION_SPACE_DEVICE);
                    printf("# repeat %d  merit %26.17g  gradient %26.17g\n",
                           i, static_cast<double>(v), static_cast<double>(g));
                    fflush(stdout);
                }
            }

            if (smesh::Env::read("SFEM_CHECK_BLAS", 0)) {
                blas_self_test("after the generated merit", ndofs);
            }

            const real_t device_gradient = gradient_norm(f, x_device->data(), sfem::EXECUTION_SPACE_DEVICE);
            printf("%-34s %12s %14s %14s %26.17g\n", "device_gradient_norm", "", "", "", device_gradient);
            const double gscale = std::fabs(static_cast<double>(host_gradient));
            printf("# gradient device against host: relative difference %.3e\n",
                   gscale > 0 ? std::fabs(static_cast<double>(device_gradient - host_gradient)) / gscale : 0.0);
        }
    }
#else
    printf("# built without CUDA -- device arm unavailable\n");
#endif

    return SFEM_SUCCESS;
}
