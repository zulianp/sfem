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

    class Laplacian::Impl {
    public:
        std::shared_ptr<FunctionSpace>  space;  ///< Function space for the operator
        std::map<std::string, OpDomain> domains;

        long   calls{0};       ///< Number of apply() calls for performance tracking
        double total_time{0};  ///< Total time spent in apply() for performance tracking

        Impl(const std::shared_ptr<FunctionSpace> &space) : space(space) {}
        ~Impl() {
            if (SFEM_PRINT_THROUGHPUT && calls) {
                printf("Laplacian::apply called %ld times. Total: %g [s], "
                       "Avg: %g [s], TP %g [MDOF/s]\n",
                       calls,
                       total_time,
                       total_time / calls,
                       1e-6 * space->n_dofs() / (total_time / calls));
            }
        }

        void print_info() {
            for (auto &domain : domains) {
                printf("Domain %s: %s\n", domain.first.c_str(), domain.second.block->name().c_str());
            }
        }
    };

    int Laplacian::initialize(const std::vector<std::string> &block_names) {
        SFEM_TRACE_SCOPE("Laplacian::initialize");

        auto mesh = impl_->space->mesh_ptr();

        if (block_names.empty()) {
            // All blocks
            for (auto &block : mesh->blocks()) {
                impl_->domains[block->name()] = OpDomain{block->element_type(), block, std::make_shared<Parameters>()};
            }
        } else {
            // Specific blocks
            for (auto block_name : block_names) {
                auto block = mesh->find_block(block_name);
                if (!block) {
                    SFEM_ERROR("Block %s not found", block_name.c_str());
                    return SFEM_FAILURE;
                }

                impl_->domains[block_name] = OpDomain{block->element_type(), block, std::make_shared<Parameters>()};
            }
        }

        return SFEM_SUCCESS;
    }

    std::unique_ptr<Op> Laplacian::create(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("Laplacian::create");

        assert(1 == space->block_size());

        auto ret                  = std::make_unique<Laplacian>(space);
        return ret;
    }

    std::shared_ptr<Op> Laplacian::lor_op(const std::shared_ptr<FunctionSpace> &space) {
        auto ret = std::make_shared<Laplacian>(space);
        ret->impl_->domains = impl_->domains;

        for (auto &domain : ret->impl_->domains) {
            domain.second.element_type = macro_type_variant(domain.second.element_type);
        }

        return ret;
    }

    std::shared_ptr<Op> Laplacian::derefine_op(const std::shared_ptr<FunctionSpace> &space) {
        auto ret = std::make_shared<Laplacian>(space);

        for (auto &domain : ret->impl_->domains) {
            domain.second.element_type = macro_base_elem(domain.second.element_type);
        }
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

        for (auto &domain : impl_->domains) {
            auto &d = domain.second;
            err |= laplacian_crs(d.element_type,
                                 d.block->n_elements(),
                                 mesh->n_nodes(),
                                 d.block->elements()->data(),
                                 mesh->points()->data(),
                                 graph->rowptr()->data(),
                                 graph->colidx()->data(),
                                 values);
        }

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

        for (auto &domain : impl_->domains) {
            auto &d = domain.second;
            err |= laplacian_crs_sym(d.element_type,
                                     d.block->n_elements(),
                                     mesh->n_nodes(),
                                     d.block->elements()->data(),
                                     mesh->points()->data(),
                                     rowptr,
                                     colidx,
                                     diag_values,
                                     off_diag_values);
        }

        return err;
    }

    int Laplacian::hessian_diag(const real_t *const /*x*/, real_t *const values) {
        SFEM_TRACE_SCOPE("Laplacian::hessian_diag");

        auto mesh = impl_->space->mesh_ptr();
        int  err  = SFEM_SUCCESS;

        for (auto &domain : impl_->domains) {
            auto d = domain.second;
            err |= laplacian_diag(d.element_type,
                                  d.block->n_elements(),
                                  mesh->n_nodes(),
                                  d.block->elements()->data(),
                                  mesh->points()->data(),
                                  values);
        }

        return err;
    }

    int Laplacian::gradient(const real_t *const x, real_t *const out) {
        SFEM_TRACE_SCOPE("Laplacian::gradient");

        auto mesh = impl_->space->mesh_ptr();

        int err = SFEM_SUCCESS;

        for (auto &domain : impl_->domains) {
            auto d = domain.second;
            err |= laplacian_assemble_gradient(d.element_type,
                                               d.block->n_elements(),
                                               mesh->n_nodes(),
                                               d.block->elements()->data(),
                                               mesh->points()->data(),
                                               x,
                                               out);
        }

        return err;
    }

    int Laplacian::apply(const real_t *const /*x*/, const real_t *const h, real_t *const out) {
        SFEM_TRACE_SCOPE("Laplacian::apply");

        auto   mesh = impl_->space->mesh_ptr();
        double tick = MPI_Wtime();

        int err = SFEM_SUCCESS;
        for (auto &domain : impl_->domains) {
            auto d = domain.second;
            err |= laplacian_apply(d.element_type,
                                   d.block->n_elements(),
                                   mesh->n_nodes(),
                                   d.block->elements()->data(),
                                   mesh->points()->data(),
                                   h,
                                   out);
        }

        double tock = MPI_Wtime();
        impl_->total_time += (tock - tick);
        impl_->calls++;
        return err;
    }

    int Laplacian::value(const real_t *x, real_t *const out) {
        SFEM_TRACE_SCOPE("Laplacian::value");

        auto mesh = impl_->space->mesh_ptr();

        int err = SFEM_SUCCESS;
        for (auto &domain : impl_->domains) {
            auto d = domain.second;
            err |= laplacian_assemble_value(d.element_type,
                                            d.block->n_elements(),
                                            mesh->n_nodes(),
                                            d.block->elements()->data(),
                                            mesh->points()->data(),
                                            x,
                                            out);
        }

        return err;
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
        size_t i    = 0;
        for (auto &domain : impl_->domains) {
            assert(i < element_types.size());
            domain.second.element_type = element_types[i];
            i++;
        }
    }
}  // namespace sfem
