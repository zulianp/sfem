#include "sfem_Laplacian.hpp"
#include "sfem_Tracer.hpp"

#include "sfem_defs.h"
#include "sfem_logger.h"
#include "sfem_mesh.h"

#include "laplacian.h"

#include "sfem_CRSGraph.hpp"
#include "sfem_FunctionSpace.hpp"
#include "sfem_Mesh.hpp"

#include "sfem_Parameters.hpp"

#include <map>
#include <string>

namespace sfem {

    struct OpDomain {
    public:
        enum ElemType    element_type;
        SharedBlock      block;
        SharedParameters parameters;
    };

    class MultiDomainOp {
    public:
        std::map<std::string, OpDomain> domains;

        MultiDomainOp(const std::shared_ptr<FunctionSpace> &space, const std::vector<std::string> &block_names) {
            if (block_names.empty()) {
                for (auto &block : space->mesh_ptr()->blocks()) {
                    domains[block->name()] = OpDomain{block->element_type(), block, std::make_shared<Parameters>()};
                }
            } else {
                for (auto &block_name : block_names) {
                    auto block = space->mesh_ptr()->find_block(block_name);
                    if (!block) {
                        SFEM_ERROR("Block %s not found", block_name.c_str());
                    }
                    domains[block_name] = OpDomain{block->element_type(), block, std::make_shared<Parameters>()};
                }
            }
        }

        int iterate(const std::function<int(const OpDomain &)> &func) {
            for (auto &domain : domains) {
                int err = func(domain.second);
                if (err != SFEM_SUCCESS) {
                    return err;
                }
            }
            return SFEM_SUCCESS;
        }

        void override_element_types(const std::vector<enum ElemType> &element_types) {
            size_t i = 0;
            for (auto &domain : domains) {
                assert(i < element_types.size());
                domain.second.element_type = element_types[i];
                i++;
            }
        }

        std::shared_ptr<MultiDomainOp> lor_op(const std::shared_ptr<FunctionSpace> &space,
                                              const std::vector<std::string>       &block_names) {
            auto ret = std::make_shared<MultiDomainOp>(space, block_names);

            for (auto &domain : ret->domains) {
                domain.second.element_type = macro_type_variant(domain.second.element_type);
            }

            return ret;
        }

        std::shared_ptr<MultiDomainOp> derefine_op(const std::shared_ptr<FunctionSpace> &space,
                                                   const std::vector<std::string>       &block_names) {
            auto ret = std::make_shared<MultiDomainOp>(space, block_names);

            for (auto &domain : ret->domains) {
                domain.second.element_type = macro_base_elem(domain.second.element_type);
            }

            return ret;
        }

        void print_info() {
            for (auto &domain : domains) {
                printf("Domain %s: %s\n", domain.first.c_str(), domain.second.block->name().c_str());
            }
        }
    };

    class OpTracer {
    public:
        std::shared_ptr<FunctionSpace> space;
        std::string                    name;
        long                           calls{0};
        double                         total_time{0};

        OpTracer(const std::shared_ptr<FunctionSpace> &space, const std::string &name) : space(space), name(name) {}
        ~OpTracer() {
            if (calls) {
                printf("%s called %ld times. Total: %g [s], "
                       "Avg: %g [s], TP %g [MDOF/s]\n",
                       name.c_str(),
                       calls,
                       total_time,
                       total_time / calls,
                       1e-6 * space->n_dofs() / (total_time / calls));
            }
        }

        class ScopedCapture {
        public:
            OpTracer &profiler;
            ScopedCapture(OpTracer &profiler) : profiler(profiler) { start_time = MPI_Wtime(); }

            ~ScopedCapture() {
                double end_time = MPI_Wtime();
                profiler.calls++;
                profiler.total_time += end_time - start_time;
            }

        private:
            double start_time;
        };
    };

#if SFEM_PRINT_THROUGHPUT
#define SFEM_OP_CAPTURE() OpTracer::ScopedCapture __sfem_op_capture(*impl_->op_profiler);
#else
#define SFEM_OP_CAPTURE()
#endif

    class Laplacian::Impl {
    public:
        std::shared_ptr<FunctionSpace> space;  ///< Function space for the operator
        std::shared_ptr<MultiDomainOp> domains;
#if SFEM_PRINT_THROUGHPUT
        std::unique_ptr<OpTracer> op_profiler;
#endif
        Impl(const std::shared_ptr<FunctionSpace> &space) : space(space) {
#if SFEM_PRINT_THROUGHPUT
            op_profiler = std::make_unique<OpTracer>(space, "Laplacian::apply");
#endif
        }
        ~Impl() {}

        void print_info() { domains->print_info(); }

        int iterate(const std::function<int(const OpDomain &)> &func) { return domains->iterate(func); }
    };

    int Laplacian::initialize(const std::vector<std::string> &block_names) {
        SFEM_TRACE_SCOPE("Laplacian::initialize");
        impl_->domains = std::make_shared<MultiDomainOp>(impl_->space, block_names);
        return SFEM_SUCCESS;
    }

    std::unique_ptr<Op> Laplacian::create(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("Laplacian::create");

        assert(1 == space->block_size());

        auto ret = std::make_unique<Laplacian>(space);
        return ret;
    }

    std::shared_ptr<Op> Laplacian::lor_op(const std::shared_ptr<FunctionSpace> &space) {
        auto ret            = std::make_shared<Laplacian>(space);
        ret->impl_->domains = impl_->domains->lor_op(space, {});
        return ret;
    }

    std::shared_ptr<Op> Laplacian::derefine_op(const std::shared_ptr<FunctionSpace> &space) {
        auto ret            = std::make_shared<Laplacian>(space);
        ret->impl_->domains = impl_->domains->derefine_op(space, {});
        return ret;
    }

    Laplacian::Laplacian(const std::shared_ptr<FunctionSpace> &space) : impl_(std::make_unique<Impl>(space)) {}

    Laplacian::~Laplacian() = default;

    int Laplacian::hessian_crs(const real_t *const  x,
                               const count_t *const rowptr,
                               const idx_t *const   colidx,
                               real_t *const        values) {
        SFEM_TRACE_SCOPE("Laplacian::hessian_crs");

        auto mesh  = impl_->space->mesh_ptr();
        auto graph = impl_->space->dof_to_dof_graph();
        int  err   = SFEM_SUCCESS;

        impl_->iterate([&](const OpDomain &domain) {
            return laplacian_crs(domain.element_type,
                                 domain.block->n_elements(),
                                 mesh->n_nodes(),
                                 domain.block->elements()->data(),
                                 mesh->points()->data(),
                                 graph->rowptr()->data(),
                                 graph->colidx()->data(),
                                 values);
        });

        return err;
    }

    int Laplacian::hessian_crs_sym(const real_t *const  x,
                                   const count_t *const rowptr,
                                   const idx_t *const   colidx,
                                   real_t *const        diag_values,
                                   real_t *const        off_diag_values) {
        SFEM_TRACE_SCOPE("Laplacian::hessian_crs_sym");

        auto mesh = impl_->space->mesh_ptr();
        int  err  = SFEM_SUCCESS;

        impl_->iterate([&](const OpDomain &domain) {
            return laplacian_crs_sym(domain.element_type,
                                     domain.block->n_elements(),
                                     mesh->n_nodes(),
                                     domain.block->elements()->data(),
                                     mesh->points()->data(),
                                     rowptr,
                                     colidx,
                                     diag_values,
                                     off_diag_values);
        });

        return err;
    }

    int Laplacian::hessian_diag(const real_t *const /*x*/, real_t *const values) {
        SFEM_TRACE_SCOPE("Laplacian::hessian_diag");

        auto mesh = impl_->space->mesh_ptr();
        int  err  = SFEM_SUCCESS;

        impl_->iterate([&](const OpDomain &domain) {
            return laplacian_diag(domain.element_type,
                                  domain.block->n_elements(),
                                  mesh->n_nodes(),
                                  domain.block->elements()->data(),
                                  mesh->points()->data(),
                                  values);
        });

        return err;
    }

    int Laplacian::gradient(const real_t *const x, real_t *const out) {
        SFEM_TRACE_SCOPE("Laplacian::gradient");

        auto mesh = impl_->space->mesh_ptr();
        return impl_->iterate([&](const OpDomain &domain) {
            return laplacian_assemble_gradient(domain.element_type,
                                               domain.block->n_elements(),
                                               mesh->n_nodes(),
                                               domain.block->elements()->data(),
                                               mesh->points()->data(),
                                               x,
                                               out);
        });
    }

    int Laplacian::apply(const real_t *const /*x*/, const real_t *const h, real_t *const out) {
        SFEM_TRACE_SCOPE("Laplacian::apply");
        SFEM_OP_CAPTURE();

        auto mesh = impl_->space->mesh_ptr();
        return impl_->iterate([&](const OpDomain &domain) {
            return laplacian_apply(domain.element_type,
                                   domain.block->n_elements(),
                                   mesh->n_nodes(),
                                   domain.block->elements()->data(),
                                   mesh->points()->data(),
                                   h,
                                   out);
        });
    }

    int Laplacian::value(const real_t *x, real_t *const out) {
        SFEM_TRACE_SCOPE("Laplacian::value");

        auto mesh = impl_->space->mesh_ptr();
        return impl_->iterate([&](const OpDomain &domain) {
            return laplacian_assemble_value(domain.element_type,
                                            domain.block->n_elements(),
                                            mesh->n_nodes(),
                                            domain.block->elements()->data(),
                                            mesh->points()->data(),
                                            x,
                                            out);
        });
    }

    int Laplacian::report(const real_t *const) { return SFEM_SUCCESS; }

    std::shared_ptr<Op> Laplacian::clone() const {
        SFEM_ERROR("IMPLEMENT ME!\n");
        return nullptr;
    }

    void Laplacian::set_value_in_block(const std::string &block_name, const std::string &var_name, const real_t value) {
        auto mesh  = impl_->space->mesh_ptr();
        auto block = mesh->find_block(block_name);
        if (!block) {
            SFEM_ERROR("Block %s not found", block_name.c_str());
            return;
        }

        // TODO: Implement block-specific value setting
        // This would typically involve setting material properties or boundary conditions
        // specific to the named block and variable
    }

    void Laplacian::override_element_types(const std::vector<enum ElemType> &element_types) {
        impl_->domains->override_element_types(element_types);
    }
}  // namespace sfem
