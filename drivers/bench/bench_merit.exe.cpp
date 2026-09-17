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
            f->value(x, &warm);
        }
        sfem::device_synchronize();
        const double t0 = MPI_Wtime();
        real_t       value = 0;
        for (int i = 0; i < repeat; ++i) {
            value = 0;
            f->value(x, &value);
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
        auto blas = sfem::blas<real_t>(es);
        return std::sqrt(blas->dot(ndofs, residual->data(), residual->data()));
    }

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
        print_rate(f->reduces_node_wise() ? "host_node_wise_merit" : "host_element_wise_merit",
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
            if (needs_previous) {
                auto previous_device = smesh::to_device(previous_host);
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
            print_rate(f->reduces_node_wise() ? "device_node_wise_merit" : "device_element_wise_merit",
                       elapsed, nelements, ndofs, repeat, device_value);

            // The device reduction is the vendor's dot product for the node-wise
            // path, so this is a tolerance rather than an equality -- and saying
            // which it is, with the number, beats asserting a pass.
            const double scale    = std::fabs(static_cast<double>(host_value));
            const double relative = scale > 0 ? std::fabs(static_cast<double>(device_value - host_value)) / scale : 0;
            printf("# device against host: relative difference %.3e\n", relative);

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
