#include "sfem_Function.hpp"

#include <stddef.h>

#include "matrixio_array.h"
#include "utils.h"

#include "crs_graph.h"
#include "sfem_defs.h"
#include "sfem_logger.h"
#include "sfem_mesh.h"

#include "boundary_condition.h"
#include "boundary_condition_io.h"

#include "dirichlet.h"
#include "integrate_values.h"
#include "neumann.h"

#include <sys/stat.h>
// #include <sys/wait.h>
#include <cstddef>
#include <fstream>
#include <functional>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <sstream>
#include <vector>

// Mesh
#include "adj_table.h"
#include "hex8_fff.h"
#include "hex8_jacobian.h"
#include "sfem_hex8_mesh_graph.h"
#include "sshex8.h"
#include "sshex8_mesh.h"

// Multigrid
#include "sfem_prolongation_restriction.h"

// C++ includes
#include "sfem_CRSGraph.hpp"
#include "sfem_SemiStructuredMesh.hpp"
#include "sfem_Tracer.hpp"
#include "sfem_glob.hpp"

#ifdef SFEM_ENABLE_RYAML

#if defined(RYML_SINGLE_HEADER)  // using the single header directly in the executable
#define RYML_SINGLE_HDR_DEFINE_NOW
#include <ryml_all.hpp>
#elif defined(RYML_SINGLE_HEADER_LIB)  // using the single header from a library
#include <ryml_all.hpp>
#else
#include <ryml.hpp>
// <ryml_std.hpp> is needed if interop with std containers is
// desired; ryml itself does not use any STL container.
// For this sample, we will be using std interop, so...
#include <c4/format.hpp>  // needed for the examples below
#include <ryml_std.hpp>   // optional header, provided for std:: interop
#endif

#include <sstream>
#endif

#include <map>

#include "sfem_DirichletConditions.hpp"
#include "sfem_NeumannConditions.hpp"

namespace sfem {

    class Output::Impl {
    public:
        std::shared_ptr<FunctionSpace> space;
        bool                           AoS_to_SoA{false};
        std::string                    output_dir{"."};
        std::string                    file_format{"%s/%s.raw"};
        std::string                    time_dependent_file_format{"%s/%s.%09d.raw"};
        size_t                         export_counter{0};
        logger_t                       time_logger;
        Impl() { log_init(&time_logger); }
        ~Impl() { log_destroy(&time_logger); }
    };

    void Output::enable_AoS_to_SoA(const bool val) { impl_->AoS_to_SoA = val; }

    Output::Output(const std::shared_ptr<FunctionSpace> &space) : impl_(std::make_unique<Impl>()) {
        impl_->space = space;

        const char *SFEM_OUTPUT_DIR = ".";
        SFEM_READ_ENV(SFEM_OUTPUT_DIR, );
        impl_->output_dir = SFEM_OUTPUT_DIR;
    }

    Output::~Output() = default;

    void Output::clear() { impl_->export_counter = 0; }

    void Output::set_output_dir(const char *path) { impl_->output_dir = path; }

    int Output::write(const char *name, const real_t *const x) {
        SFEM_TRACE_SCOPE("Output::write");

        MPI_Comm comm = impl_->space->mesh_ptr()->comm()->get();
        sfem::create_directory(impl_->output_dir.c_str());

        const int block_size = impl_->space->block_size();
        if (impl_->AoS_to_SoA && block_size > 1) {
            ptrdiff_t n_blocks = impl_->space->n_dofs() / block_size;

            auto buff = create_host_buffer<real_t>(n_blocks);
            auto bb   = buff->data();

            char path[2048];
            for (int b = 0; b < block_size; b++) {
                for (ptrdiff_t i = 0; i < n_blocks; i++) {
                    bb[i] = x[i * block_size + b];
                }

                char b_name[1024];
                snprintf(b_name, sizeof(b_name), "%s.%d", name, b);
                snprintf(path, sizeof(path), impl_->file_format.c_str(), impl_->output_dir.c_str(), b_name);
                if (array_write(comm, path, SFEM_MPI_REAL_T, buff->data(), n_blocks, n_blocks)) {
                    return SFEM_FAILURE;
                }
            }

        } else {
            char path[2048];
            snprintf(path, sizeof(path), impl_->file_format.c_str(), impl_->output_dir.c_str(), name);
            if (array_write(comm, path, SFEM_MPI_REAL_T, x, impl_->space->n_dofs(), impl_->space->n_dofs())) {
                return SFEM_FAILURE;
            }
        }

        return SFEM_SUCCESS;
    }

    void Output::log_time(const real_t t) {
        if (log_is_empty(&impl_->time_logger)) {
            char path[2048];
            snprintf(path, sizeof(path), "%s/time.txt", impl_->output_dir.c_str());
            log_create_file(&impl_->time_logger, path, "w");
        }

        log_write_double(&impl_->time_logger, t);
    }

    int Output::write_time_step(const char *name, const real_t t, const real_t *const x) {
        SFEM_TRACE_SCOPE("Output::write_time_step");

        auto      space      = impl_->space;
        auto      mesh       = space->mesh_ptr();
        const int block_size = space->block_size();

        sfem::create_directory(impl_->output_dir.c_str());

        char path[2048];

        if (impl_->AoS_to_SoA && block_size > 1) {
            ptrdiff_t n_blocks = space->n_dofs() / block_size;

            auto buff = create_host_buffer<real_t>(n_blocks);
            auto bb   = buff->data();

            for (int b = 0; b < block_size; b++) {
                for (ptrdiff_t i = 0; i < n_blocks; i++) {
                    bb[i] = x[i * block_size + b];
                }

                char b_name[1024];
                snprintf(b_name, sizeof(b_name), "%s.%d", name, b);
                snprintf(path,
                         sizeof(path),
                         impl_->time_dependent_file_format.c_str(),
                         impl_->output_dir.c_str(),
                         b_name,
                         impl_->export_counter++);

                if (array_write(mesh->comm()->get(), path, SFEM_MPI_REAL_T, buff->data(), n_blocks, n_blocks)) {
                    return SFEM_FAILURE;
                }
            }

        } else {
            snprintf(path,
                     sizeof(path),
                     impl_->time_dependent_file_format.c_str(),
                     impl_->output_dir.c_str(),
                     name,
                     impl_->export_counter++);

            if (array_write(mesh->comm()->get(), path, SFEM_MPI_REAL_T, x, space->n_dofs(), space->n_dofs())) {
                return SFEM_FAILURE;
            }
        }

        return SFEM_SUCCESS;
    }

    class Function::Impl {
    public:
        std::shared_ptr<FunctionSpace>           space;
        std::vector<std::shared_ptr<Op>>         ops;
        std::vector<std::shared_ptr<Constraint>> constraints;

        std::shared_ptr<Output> output;
        bool                    handle_constraints{true};
    };

    ExecutionSpace Function::execution_space() const {
        ExecutionSpace ret = EXECUTION_SPACE_INVALID;

        for (auto op : impl_->ops) {
            assert(ret == EXECUTION_SPACE_INVALID || ret == op->execution_space());
            ret = op->execution_space();
        }

        return ret;
    }

    void Function::describe(std::ostream &os) const {
        os << "n_dofs: " << impl_->space->n_dofs() << "\n";
        os << "n_ops: " << impl_->ops.size() << "\n";
        os << "n_constraints: " << impl_->constraints.size() << "\n";
    }

    Function::Function(const std::shared_ptr<FunctionSpace> &space) : impl_(std::make_unique<Impl>()) {
        impl_->space  = space;
        impl_->output = std::make_shared<Output>(space);
    }

    std::shared_ptr<FunctionSpace> Function::space() { return impl_->space; }

    Function::~Function() {}

    void Function::remove_operator(const std::shared_ptr<Op> &op) {
        auto it = std::find(impl_->ops.begin(), impl_->ops.end(), op);

        if (it == impl_->ops.end()) {
            SFEM_ERROR("remove_operator: op does not exist!");
        }

        impl_->ops.erase(it);
    }

    void Function::add_operator(const std::shared_ptr<Op> &op) { impl_->ops.push_back(op); }
    void Function::add_constraint(const std::shared_ptr<Constraint> &c) { impl_->constraints.push_back(c); }

    void Function::clear_constraints() { impl_->constraints.clear(); }

    void Function::add_dirichlet_conditions(const std::shared_ptr<DirichletConditions> &c) { add_constraint(c); }

    int Function::constaints_mask(mask_t *mask) {
        SFEM_TRACE_SCOPE("Function::constaints_mask");

        for (auto &c : impl_->constraints) {
            c->mask(mask);
        }

        return SFEM_FAILURE;
    }

    std::shared_ptr<CRSGraph> Function::crs_graph() const { return impl_->space->dof_to_dof_graph(); }

    int Function::hessian_crs(const real_t *const  x,
                              const count_t *const rowptr,
                              const idx_t *const   colidx,
                              real_t *const        values) {
        SFEM_TRACE_SCOPE("Function::hessian_crs");

        for (auto &op : impl_->ops) {
            if (op->hessian_crs(x, rowptr, colidx, values) != SFEM_SUCCESS) {
                std::cerr << "Failed hessian_crs in op: " << op->name() << "\n";
                return SFEM_FAILURE;
            }
        }

        if (impl_->handle_constraints) {
            for (auto &c : impl_->constraints) {
                c->hessian_crs(x, rowptr, colidx, values);
            }
        }

        return SFEM_SUCCESS;
    }

    int Function::hessian_bsr(const real_t *const  x,
                              const count_t *const rowptr,
                              const idx_t *const   colidx,
                              real_t *const        values) {
        SFEM_TRACE_SCOPE("Function::hessian_bsr");

        for (auto &op : impl_->ops) {
            if (op->hessian_bsr(x, rowptr, colidx, values) != SFEM_SUCCESS) {
                std::cerr << "Failed hessian_bsr in op: " << op->name() << "\n";
                return SFEM_FAILURE;
            }
        }

        if (impl_->handle_constraints) {
            for (auto &c : impl_->constraints) {
                c->hessian_bsr(x, rowptr, colidx, values);
            }
        }

        return SFEM_SUCCESS;
    }

    int Function::hessian_bcrs_sym(const real_t *const  x,
                                   const count_t *const rowptr,
                                   const idx_t *const   colidx,
                                   const ptrdiff_t      block_stride,
                                   real_t **const       diag_values,
                                   real_t **const       off_diag_values) {
        SFEM_TRACE_SCOPE("Function::hessian_bcrs_sym");
        for (auto &op : impl_->ops) {
            if (op->hessian_bcrs_sym(x, rowptr, colidx, block_stride, diag_values, off_diag_values) != SFEM_SUCCESS) {
                std::cerr << "Failed hessian_bcrs_sym in op: " << op->name() << "\n";
                return SFEM_FAILURE;
            }
        }
        return SFEM_SUCCESS;
    }

    int Function::hessian_crs_sym(const real_t *const  x,
                                  const count_t *const rowptr,
                                  const idx_t *const   colidx,
                                  real_t *const        diag_values,
                                  real_t *const        off_diag_values) {
        SFEM_TRACE_SCOPE("Function::hessian_crs_sym");
        for (auto &op : impl_->ops) {
            if (op->hessian_crs_sym(x, rowptr, colidx, diag_values, off_diag_values) != SFEM_SUCCESS) {
                std::cerr << "Failed hessian_crs_sym in op: " << op->name() << "\n";
                return SFEM_FAILURE;
            }
        }
        return SFEM_SUCCESS;
    }

    int Function::hessian_diag(const real_t *const x, real_t *const values) {
        SFEM_TRACE_SCOPE("Function::hessian_diag");
        for (auto &op : impl_->ops) {
            if (op->hessian_diag(x, values) != SFEM_SUCCESS) {
                std::cerr << "Failed hessian_diag in op: " << op->name() << "\n";
                return SFEM_FAILURE;
            }
        }

        if (impl_->handle_constraints) {
            for (auto &c : impl_->constraints) {
                c->apply_value(1, values);
            }
        }

        return SFEM_SUCCESS;
    }

    int Function::hessian_block_diag_sym(const real_t *const x, real_t *const values) {
        SFEM_TRACE_SCOPE("Function::hessian_block_diag_sym");

        for (auto &op : impl_->ops) {
            if (op->hessian_block_diag_sym(x, values) != SFEM_SUCCESS) {
                std::cerr << "Failed hessian_diag in op: " << op->name() << "\n";
                return SFEM_FAILURE;
            }
        }

        return SFEM_SUCCESS;
    }

    int Function::gradient(const real_t *const x, real_t *const out) {
        SFEM_TRACE_SCOPE("Function::gradient");

        for (auto &op : impl_->ops) {
            if (op->gradient(x, out) != SFEM_SUCCESS) {
                std::cerr << "Failed gradient in op: " << op->name() << "\n";
                return SFEM_FAILURE;
            }
        }

        if (impl_->handle_constraints) {
            constraints_gradient(x, out);
        }

        return SFEM_SUCCESS;
    }

    int Function::apply(const real_t *const x, const real_t *const h, real_t *const out) {
        SFEM_TRACE_SCOPE("Function::apply");

        for (auto &op : impl_->ops) {
            if (op->apply(x, h, out) != SFEM_SUCCESS) {
                std::cerr << "Failed apply in op: " << op->name() << "\n";
                return SFEM_FAILURE;
            }
        }

        if (impl_->handle_constraints) {
            copy_constrained_dofs(h, out);
        }

        return SFEM_SUCCESS;
    }

    std::shared_ptr<Operator<real_t>> Function::linear_op_variant(const std::vector<std::pair<std::string, int>> &options) {
        std::vector<std::shared_ptr<Op>> cloned_ops;

        for (auto &op : impl_->ops) {
            auto c = op->clone();

            for (auto p : options) {
                c->set_option(p.first, p.second);
            }

            cloned_ops.push_back(c);
        }

        return sfem::make_op<real_t>(
                this->space()->n_dofs(),
                this->space()->n_dofs(),
                [=](const real_t *const x, real_t *const y) {
                    for (auto op : cloned_ops) {
                        if (op->apply(nullptr, x, y) != SFEM_SUCCESS) {
                            std::cerr << "Failed apply in op: " << op->name() << "\n";
                            assert(false);
                        }
                    }

                    if (impl_->handle_constraints) {
                        copy_constrained_dofs(x, y);
                    }
                },
                this->execution_space());
    }

    int Function::value(const real_t *x, real_t *const out) {
        SFEM_TRACE_SCOPE("Function::value");

        for (auto &op : impl_->ops) {
            if (op->value(x, out) != SFEM_SUCCESS) {
                std::cerr << "Failed value in op: " << op->name() << "\n";
                return SFEM_FAILURE;
            }
        }

        for (auto &c : impl_->constraints) {
            c->value(x, out);
        }

        return SFEM_SUCCESS;
    }

    int Function::value_steps(const real_t *x, const real_t *h, const int nsteps, const real_t *const steps, real_t *const out) {
        SFEM_TRACE_SCOPE("Function::value_steps");
        for (auto &op : impl_->ops) {
            if (op->value_steps(x, h, nsteps, steps, out) != SFEM_SUCCESS) {
                std::cerr << "Failed value_steps in op: " << op->name() << "\n";
                return SFEM_FAILURE;
            }
        }

        for (auto &c : impl_->constraints) {
            c->value_steps(x, h, nsteps, steps, out);
        }
        return SFEM_SUCCESS;
    }

    int Function::apply_constraints(real_t *const x) {
        SFEM_TRACE_SCOPE("Function::apply_constraints");

        for (auto &c : impl_->constraints) {
            c->apply(x);
        }
        return SFEM_SUCCESS;
    }

    int Function::constraints_gradient(const real_t *const x, real_t *const g) {
        SFEM_TRACE_SCOPE("Function::constraints_gradient");

        for (auto &c : impl_->constraints) {
            c->gradient(x, g);
        }
        return SFEM_SUCCESS;
    }

    int Function::apply_zero_constraints(real_t *const x) {
        SFEM_TRACE_SCOPE("Function::apply_zero_constraints");

        for (auto &c : impl_->constraints) {
            c->apply_zero(x);
        }
        return SFEM_SUCCESS;
    }

    int Function::set_value_to_constrained_dofs(const real_t val, real_t *const x) {
        SFEM_TRACE_SCOPE("Function::set_value_to_constrained_dofs");

        for (auto &c : impl_->constraints) {
            c->apply_value(val, x);
        }
        return SFEM_SUCCESS;
    }

    int Function::copy_constrained_dofs(const real_t *const src, real_t *const dest) {
        SFEM_TRACE_SCOPE("Function::copy_constrained_dofs");

        for (auto &c : impl_->constraints) {
            c->copy_constrained_dofs(src, dest);
        }
        return SFEM_SUCCESS;
    }

    int Function::report_solution(const real_t *const x) {
        SFEM_TRACE_SCOPE("Function::report_solution");

        return impl_->output->write("out", x);
    }

    int Function::initial_guess(real_t *const x) { return SFEM_SUCCESS; }

    int Function::set_output_dir(const char *path) {
        impl_->output->set_output_dir(path);
        return SFEM_SUCCESS;
    }

    int Function::update(const real_t *const x) {
        SFEM_TRACE_SCOPE("Function::update");
        for (auto &op : impl_->ops) {
            op->update(x);
        }
        return SFEM_SUCCESS;
    }

    std::shared_ptr<Output> Function::output() { return impl_->output; }

    std::shared_ptr<Function> Function::derefine(const bool dirichlet_as_zero) {
        return derefine(impl_->space->derefine(), dirichlet_as_zero);
    }

    std::shared_ptr<Function> Function::derefine(const std::shared_ptr<FunctionSpace> &space, const bool dirichlet_as_zero) {
        SFEM_TRACE_SCOPE("Function::derefine");
        auto ret = std::make_shared<Function>(space);

        for (auto &o : impl_->ops) {
            auto dop = o->derefine_op(space);
            if (!dop->is_no_op()) {
                ret->impl_->ops.push_back(dop);
            }
        }

        for (auto &c : impl_->constraints) {
            ret->impl_->constraints.push_back(c->derefine(space, dirichlet_as_zero));
        }

        ret->impl_->handle_constraints = impl_->handle_constraints;

        return ret;
    }

    std::shared_ptr<Function> Function::lor() { return lor(impl_->space->lor()); }
    std::shared_ptr<Function> Function::lor(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("Function::lor");

        auto ret = std::make_shared<Function>(space);

        for (auto &o : impl_->ops) {
            ret->impl_->ops.push_back(o->lor_op(space));
        }

        for (auto &c : impl_->constraints) {
            ret->impl_->constraints.push_back(c);
        }

        ret->impl_->handle_constraints = impl_->handle_constraints;

        return ret;
    }

    std::shared_ptr<Buffer<idx_t *>> mesh_connectivity_from_file(const std::shared_ptr<Communicator> &comm, const char *folder) {
        char pattern[1024 * 10];
        snprintf(pattern, sizeof(pattern), "%s/i*.raw", folder);

        std::shared_ptr<Buffer<idx_t *>> ret;

        auto files   = sfem::find_files(pattern);
        int  n_files = files.size();

        idx_t **data = (idx_t **)malloc(n_files * sizeof(idx_t *));

        ptrdiff_t local_size = SFEM_PTRDIFF_INVALID;
        ptrdiff_t size       = SFEM_PTRDIFF_INVALID;

        printf("n_files (%d):\n", n_files);
        int err = 0;
        for (int np = 0; np < n_files; np++) {
            printf("%s\n", files[np].c_str());

            char path[1024 * 10];
            snprintf(path, sizeof(path), "%s/i%d.raw", folder, np);

            idx_t *idx = 0;
            err |= array_create_from_file(comm->get(), path, SFEM_MPI_IDX_T, (void **)&idx, &local_size, &size);

            data[np] = idx;
        }

        ret = std::make_shared<Buffer<idx_t *>>(
                n_files,
                local_size,
                data,
                [](int n, void **data) {
                    for (int i = 0; i < n; i++) {
                        free(data[i]);
                    }

                    free(data);
                },
                MEMORY_SPACE_HOST);

        assert(!err);

        return ret;
    }

}  // namespace sfem
