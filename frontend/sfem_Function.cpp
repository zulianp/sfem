#include "sfem_Function.hpp"

#include <algorithm>
#include <stddef.h>
#include <vector>

#include "utils.h"

#include "sfem_defs.hpp"
#include "sfem_logger.hpp"
#include "sfem_openmp_blas.hpp"
#include "sfem_API.hpp"
#include "smesh_glob.hpp"
#include "smesh_mesh.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_cuda_blas.hpp"
#endif

#include "boundary_condition.hpp"
#include "boundary_condition_io.hpp"

#include "dirichlet.hpp"
#include "integrate_values.hpp"
#include "neumann.hpp"

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

#include "hex8_fff.hpp"
#include "hex8_jacobian.hpp"
//

#include "smesh_semistructured.hpp"

#include "smesh_common.hpp"
#include "smesh_glob.hpp"
#include "smesh_restriction.hpp"
#include "smesh_sshex8.hpp"

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
        std::shared_ptr<smesh::Output> smesh_output;
        std::shared_ptr<FunctionSpace> space;
        bool                           AoS_to_SoA{false};
        smesh::Path                    output_dir{smesh::Path(".")};
        std::string                    file_format{"%s/%s.%s"};
        std::string                    time_dependent_file_format{"%s/%s.%09d.%s"};
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
        impl_->output_dir = smesh::Path(SFEM_OUTPUT_DIR);

        impl_->smesh_output = smesh::Output::create(space->mesh_ptr(), smesh::Path(impl_->output_dir));
    }

    Output::~Output() = default;

    void Output::clear() { impl_->export_counter = 0; }

    void Output::set_output_dir(const smesh::Path &path) {
        impl_->smesh_output = smesh::Output::create(impl_->space->mesh_ptr(), smesh::Path(path));
        impl_->output_dir   = path;
    }

    int Output::write(const char *name, const real_t *const x) {
        SFEM_TRACE_SCOPE("Output::write");

        smesh::create_directory(impl_->output_dir.c_str());

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
                impl_->smesh_output->write_nodal(b_name, smesh::TypeToEnum<real_t>::value(), bb, 1);
            }

        } else {
            impl_->smesh_output->write_nodal(name, smesh::TypeToEnum<real_t>::value(), x, impl_->space->block_size());
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
        const int block_size = space->block_size();

        smesh::create_directory(impl_->output_dir.c_str());

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
                         impl_->export_counter++,
                         smesh::str(smesh::TypeToEnum<real_t>::value()).c_str());

                if (buff->to_file(smesh::Path(path))) {
                    return SFEM_FAILURE;
                }
            }

        } else {
            snprintf(path,
                     sizeof(path),
                     impl_->time_dependent_file_format.c_str(),
                     impl_->output_dir.c_str(),
                     name,
                     impl_->export_counter++,
                     smesh::str(smesh::TypeToEnum<real_t>::value()).c_str());

            auto out = Buffer<real_t>::wrap(space->n_dofs(), const_cast<real_t *>(x));
            if (out->to_file(smesh::Path(path))) {
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
        //! Reused across `residual_merit` calls so a line search allocates
        //! nothing; a caller may supply its own through the overload.
        SharedBuffer<real_t>                     merit_accumulator;

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

    bool Function::is_linear() const {
        for (const auto &op : impl_->ops) {
            if (op && !op->is_linear()) {
                return false;
            }
        }
        return true;
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

    void Function::add_operator(const std::shared_ptr<Op> &op) {
        // An operator with no potential mandates the residual merit, and that
        // merit is a norm of the whole assembled residual -- so it is the one
        // that finishes the sum, and it is kept last for `merit_contractor` to
        // find.  Two of them cannot be composed: `1/2*||R_a + R_b||^2` is
        // neither operator's number, and neither can compute it alone.
        const bool mandates_reduction = !op->energy_or_potential_based();

        if (mandates_reduction && !impl_->ops.empty() && !impl_->ops.back()->energy_or_potential_based()) {
            SFEM_ERROR(
                    "Function::add_operator: \"%s\" has no potential and neither does \"%s\", so "
                    "both would have to reduce the same residual; compose them into one operator\n",
                    op->name(),
                    impl_->ops.back()->name());
            return;
        }

        if (!mandates_reduction && !impl_->ops.empty() && !impl_->ops.back()->energy_or_potential_based()) {
            // The reducing operator stays last whatever order the caller adds
            // in, so a driver does not have to know the rule.
            impl_->ops.insert(impl_->ops.end() - 1, op);
            return;
        }

        impl_->ops.push_back(op);
    }
    void Function::add_constraint(const std::shared_ptr<Constraint> &c) { impl_->constraints.push_back(c); }

    void Function::clear_constraints() { impl_->constraints.clear(); }

    void Function::add_dirichlet_conditions(const std::shared_ptr<DirichletConditions> &c) { add_constraint(c); }

    int Function::constraints_mask(mask_t *mask) {
        SFEM_TRACE_SCOPE("Function::constraints_mask");

        int err = SFEM_SUCCESS;
        for (auto &c : impl_->constraints) {
            err += c->mask(mask);
        }

        return err == SFEM_SUCCESS ? SFEM_SUCCESS : SFEM_FAILURE;
    }

    std::shared_ptr<CRSGraph> Function::crs_graph() const { return impl_->space->dof_to_dof_graph(); }

    double Function::flops() const {
        double ret = 0;
        for (const auto &op : impl_->ops) {
            ret += op->flops();
        }

        return ret;
    }

    double Function::flops_value() const {
        double ret = 0;
        for (const auto &op : impl_->ops) {
            ret += op->flops_value();
        }

        return ret;
    }

    double Function::flops_gradient() const {
        double ret = 0;
        for (const auto &op : impl_->ops) {
            ret += op->flops_gradient();
        }

        return ret;
    }

    double Function::flops_apply() const {
        double ret = 0;
        for (const auto &op : impl_->ops) {
            ret += op->flops_apply();
        }

        return ret;
    }

    size_t Function::memory_traffic_bytes() const {
        size_t ret = 0;
        for (const auto &op : impl_->ops) {
            ret += op->memory_traffic_bytes();
        }

        return ret;
    }

    size_t Function::memory_traffic_bytes_value() const {
        size_t ret = 0;
        for (const auto &op : impl_->ops) {
            ret += op->memory_traffic_bytes_value();
        }

        return ret;
    }

    size_t Function::memory_traffic_bytes_gradient() const {
        size_t ret = 0;
        for (const auto &op : impl_->ops) {
            ret += op->memory_traffic_bytes_gradient();
        }

        return ret;
    }

    size_t Function::memory_traffic_bytes_apply() const {
        size_t ret = 0;
        for (const auto &op : impl_->ops) {
            ret += op->memory_traffic_bytes_apply();
        }

        return ret;
    }

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

    int Function::hessian_dia(const real_t *const x,
                              const int *const    diag_offsets,
                              const ptrdiff_t     ndiag,
                              real_t *const       values) {
        SFEM_TRACE_SCOPE("Function::hessian_dia");

        for (auto &op : impl_->ops) {
            if (op->hessian_dia(x, diag_offsets, ndiag, values) != SFEM_SUCCESS) {
                std::cerr << "Failed hessian_dia in op: " << op->name() << "\n";
                return SFEM_FAILURE;
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
                std::cerr << "Failed hessian_block_diag_sym in op: " << op->name() << "\n";
                return SFEM_FAILURE;
            }
        }

        return SFEM_SUCCESS;
    }

    int Function::gradient(const real_t *const x, real_t *const out, const ElementScope scope) {
        SFEM_TRACE_SCOPE("Function::gradient");

        for (auto &op : impl_->ops) {
            if (op->gradient(x, out, scope) != SFEM_SUCCESS) {
                std::cerr << "Failed gradient in op: " << op->name() << "\n";
                return SFEM_FAILURE;
            }
        }

        if (scope == ElementScope::ALL && impl_->handle_constraints) {
            constraints_gradient(x, out);
        }

        return SFEM_SUCCESS;
    }

    bool Function::inexact_supported() const {
        for (auto &op : impl_->ops) {
            if (op->inexact_supported()) return true;
        }
        return false;
    }

    int Function::inexact_update(const real_t *const x) {
        SFEM_TRACE_SCOPE("Function::inexact_update");

        for (auto &op : impl_->ops) {
            if (!op->inexact_supported()) continue;
            if (op->inexact_update(x) != SFEM_SUCCESS) {
                std::cerr << "Failed inexact_update in op: " << op->name() << "\n";
                return SFEM_FAILURE;
            }
        }
        return SFEM_SUCCESS;
    }

    int Function::inexact_apply(const real_t *const h, real_t *const out) {
        SFEM_TRACE_SCOPE("Function::inexact_apply");

        for (auto &op : impl_->ops) {
            // An operator with no stored tangent is applied exactly.  Its state
            // is whatever its own `update` last saw, which is the state this
            // function's tangent was assembled at -- a Newton step drives both
            // from the same iterate.
            const int status = op->inexact_supported() ? op->inexact_apply(h, out)
                                                       : op->apply(nullptr, h, out, ElementScope::ALL);
            if (status != SFEM_SUCCESS) {
                std::cerr << "Failed inexact_apply in op: " << op->name() << "\n";
                return SFEM_FAILURE;
            }
        }

        if (impl_->handle_constraints) {
            copy_constrained_dofs(h, out);
        }

        return SFEM_SUCCESS;
    }

    int Function::apply(const real_t *const x, const real_t *const h, real_t *const out, const ElementScope scope) {
        SFEM_TRACE_SCOPE("Function::apply");

        for (auto &op : impl_->ops) {
            if (op->apply(x, h, out, scope) != SFEM_SUCCESS) {
                std::cerr << "Failed apply in op: " << op->name() << "\n";
                return SFEM_FAILURE;
            }
        }

        if (scope == ElementScope::ALL && impl_->handle_constraints) {
            copy_constrained_dofs(h, out);
        }

        return SFEM_SUCCESS;
    }

    int Function::apply_scope_flat_range(const real_t *const x,
                                         const real_t *const h,
                                         real_t *const       out,
                                         const ElementScope  scope,
                                         const ptrdiff_t     flat_begin,
                                         const ptrdiff_t     flat_end) {
        SFEM_TRACE_SCOPE("Function::apply_scope_flat_range");

        for (auto &op : impl_->ops) {
            if (op->apply_scope_flat_range(x, h, out, scope, flat_begin, flat_end) != SFEM_SUCCESS) {
                std::cerr << "Failed apply_scope_flat_range in op: " << op->name() << "\n";
                return SFEM_FAILURE;
            }
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

    /// The one operator that moves with the state, or null when the count is
    /// anything but one.
    ///
    /// One is the case worth having: everything else is assembled once into the
    /// accumulator and this operator is asked to finish the sum at every trial
    /// step.  With none, the residual does not move and the merit is one number.
    /// With several, no single operator can be handed the rest -- each would
    /// need the others evaluated at the same step -- so the caller falls back to
    /// assembling everything per step, which is correct and slow.  It is not an
    /// error: such a `Function` is usually built for its gradient, or for the
    /// energy merit, and refusing it here would refuse those too.
    static std::shared_ptr<Op> merit_contractor(const std::vector<std::shared_ptr<Op>> &ops) {
        std::shared_ptr<Op> found;
        for (auto &op : ops) {
            if (!op->residual_depends_on_state()) {
                continue;
            }
            if (found) {
                return nullptr;
            }
            found = op;
        }
        return found;
    }

    int Function::energy_merit(const real_t       *x,
                               const real_t       *h,
                               const int           nsteps,
                               const real_t *const steps,
                               real_t *const       out) {
        SFEM_TRACE_SCOPE("Function::energy_merit");

        for (auto &op : impl_->ops) {
            if (!op->energy_or_potential_based()) {
                // Reported and refused rather than aborted: asking whether a
                // system has an energy is a fair question, and a caller that
                // asked wrongly can fall back to `residual_merit`, which every
                // system has.  Named, because the caller's next question is
                // always which operator.
                fprintf(stderr,
                        "Function::energy_merit: operator \"%s\" has no potential, so this system "
                        "has no energy to sum; use residual_merit instead\n",
                        op->name());
                return SFEM_FAILURE;
            }
        }

        for (auto &op : impl_->ops) {
            if (op->value_steps(x, h, nsteps, steps, out) != SFEM_SUCCESS) {
                std::cerr << "Failed value_steps in op: " << op->name() << "\n";
                return SFEM_FAILURE;
            }
        }

        for (auto &c : impl_->constraints) {
            if (c->value_steps(x, h, nsteps, steps, out) != SFEM_SUCCESS) {
                return SFEM_FAILURE;
            }
        }

        return SFEM_SUCCESS;
    }

    bool Function::has_energy_merit() const {
        for (auto &op : impl_->ops) {
            if (!op->energy_or_potential_based()) {
                return false;
            }
        }
        return true;
    }

    int Function::energy_merit(const real_t *x, real_t *const out) {
        // `x` stands in for the direction: at a step of length zero the
        // increment is unused and `x + 0 * x` is `x` exactly in IEEE.
        const real_t at_x = 0;
        return energy_merit(x, x, 1, &at_x, out);
    }

    int Function::residual_merit(const real_t *x, real_t *const out) {
        const real_t at_x = 0;
        return residual_merit(x, x, 1, &at_x, out);
    }

    int Function::residual_merit(const real_t       *x,
                                 const real_t       *h,
                                 const int           nsteps,
                                 const real_t *const steps,
                                 real_t *const       out) {
        const ptrdiff_t ndofs = impl_->space->n_dofs();
        if (!impl_->merit_accumulator || impl_->merit_accumulator->size() != (size_t)ndofs) {
            impl_->merit_accumulator = create_buffer<real_t>(ndofs, execution_space());
        }
        return residual_merit(x, h, nsteps, steps, impl_->merit_accumulator->data(), out);
    }

    int Function::residual_merit(const real_t       *x,
                                 const real_t       *h,
                                 const int           nsteps,
                                 const real_t *const steps,
                                 real_t *const       accumulator,
                                 real_t *const       out) {
        SFEM_TRACE_SCOPE("Function::residual_merit");

        const ptrdiff_t      ndofs = impl_->space->n_dofs();
        const ExecutionSpace es    = execution_space();
        auto                 blas  = sfem::blas<real_t>(es);

        // Explicitly, not by relying on the allocator: a caller's buffer is
        // reused across steps and a `create_buffer` one is zero only on its
        // first use.
        blas->zeros(ndofs, accumulator);

        auto contractor = merit_contractor(impl_->ops);

        if (!contractor && !impl_->ops.empty()) {
            bool any_moves = false;
            for (auto &op : impl_->ops) {
                any_moves = any_moves || op->residual_depends_on_state();
            }
            if (any_moves) {
                // Several operators move with the state, so nothing can be
                // hoisted: assemble the whole residual at each step.  Correct,
                // and the shape this interface exists to avoid.
                auto residual = create_buffer<real_t>(ndofs, es);
                auto stepped  = create_buffer<real_t>(ndofs, es);
                for (int step = 0; step < nsteps; ++step) {
                    blas->zeros(ndofs, residual->data());
                    blas->zaxpby(ndofs, 1, x, steps[step], h, stepped->data());
                    if (gradient(stepped->data(), residual->data()) != SFEM_SUCCESS) {
                        return SFEM_FAILURE;
                    }
                    if (es == EXECUTION_SPACE_DEVICE) {
                        sfem::device_synchronize();
                        const real_t norm = blas->norm2(ndofs, residual->data());
                        out[step] += real_t(0.5) * norm * norm;
                    } else {
                        out[step] += real_t(0.5) *
                                     squared_residual_norm(ndofs, impl_->space->block_size(), residual->data());
                    }
                }
                return SFEM_SUCCESS;
            }
        }

        // Everything that does not move with the state, once.  This is the
        // saving: today every operator is re-assembled at every trial step, and
        // a traction does not depend on the step at all.
        for (auto &op : impl_->ops) {
            if (op == contractor) {
                continue;
            }
            if (op->gradient(x, accumulator) != SFEM_SUCCESS) {
                std::cerr << "Failed gradient in op: " << op->name() << "\n";
                return SFEM_FAILURE;
            }
        }

        if (impl_->handle_constraints) {
            if (constraints_gradient(x, accumulator) != SFEM_SUCCESS) {
                return SFEM_FAILURE;
            }
        }

        if (!contractor) {
            // Nothing moves with the state, so the residual is the accumulator
            // at every step and the merit is one number repeated.
            const real_t merit =
                    real_t(0.5) * squared_residual_norm(ndofs, impl_->space->block_size(), accumulator);
            for (int step = 0; step < nsteps; ++step) {
                out[step] += merit;
            }
            return SFEM_SUCCESS;
        }

        // The sampling contraction closes the node sums inside a patch of
        // elements, and a patch is built within one mesh block.  A node on a
        // block boundary therefore has incident elements the patch never sees,
        // so its residual is still partial when the square is taken -- which is
        // silently wrong rather than loudly wrong, the merit simply coming out
        // too small.  Inter-block patches are the general answer and are not
        // built yet; until they are, abort rather than return a number that
        // looks like a merit.  Checked before the call, so the generated kernel
        // carries no test for it.
        //
        // Asked of the contractor and not of the space, because only a
        // patch-wise contraction has the problem: an operator that answers
        // false assembles the whole residual at each trial step through
        // `Op::gradient`, which spans blocks like any other assembly, and a
        // multi-block mesh must keep working there.
        if (contractor->contracts_residual_merit() && impl_->space->is_multi_block()) {
            SFEM_ERROR(
                    "Function::residual_merit: the sampled residual merit contracts patch-wise "
                    "and patches do not span mesh blocks, so a node shared between the %zu blocks "
                    "of this space would be squared before its sum is complete; inter-block "
                    "patches are not implemented yet\n",
                    impl_->space->n_blocks());
        }

        return contractor->residual_merit_steps(x, h, nsteps, steps, accumulator, out);
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

    int Function::set_output_dir(const smesh::Path &path) {
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

        for (size_t i = 0; i < impl_->ops.size(); i++) {
            auto &o = impl_->ops[i];
            if (o->is_no_op()) {
                continue;
            }
            auto dop = o->derefine_op(space);
            if (!dop) {
                SFEM_ERROR("derefine_op returned nullptr");
                return nullptr;
            }
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
        (void)comm;

        char pattern[1024 * 10];
        snprintf(pattern, sizeof(pattern), "%s/i*.raw", folder);

        std::shared_ptr<Buffer<idx_t *>> ret;

        auto files   = smesh::find_files(pattern);
        int  n_files = files.size();

        idx_t **data = (idx_t **)malloc(n_files * sizeof(idx_t *));

        ptrdiff_t local_size = SFEM_PTRDIFF_INVALID;

        printf("n_files (%d):\n", n_files);
        int err = 0;
        for (int np = 0; np < n_files; np++) {
            printf("%s\n", files[np].c_str());

            char path[1024 * 10];
            snprintf(path, sizeof(path), "%s/i%d.raw", folder, np);

            idx_t *idx = 0;
            err |= (smesh::array_read_convert_from_extension<idx_t>(smesh::Path(path), &idx, &local_size) != SMESH_SUCCESS);

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
