#include "sfem_ContactConditions.hpp"

#include "boundary_condition.h"
#include "boundary_condition_io.h"
#include "dirichlet.h"
#include "neumann.h"
#include "sfem_prolongation_restriction.h"

namespace sfem {
    class ContactConditions::Impl {
    public:
        std::shared_ptr<FunctionSpace> space;

        ~Impl() {
            if (conditions) {
                for (int i = 0; i < n_conditions; i++) {
                    free(conditions[i].idx);
                }

                free(conditions);
            }
        }

        int n_conditions{0};
        boundary_condition_t *conditions{nullptr};
    };

    std::shared_ptr<FunctionSpace> ContactConditions::space() { return impl_->space; }

    int ContactConditions::n_conditions() const { return impl_->n_conditions; }
    void *ContactConditions::impl_conditions() { return (void *)impl_->conditions; }

    ContactConditions::ContactConditions(const std::shared_ptr<FunctionSpace> &space)
        : impl_(std::make_unique<Impl>()) {
        impl_->space = space;
    }

    std::shared_ptr<Constraint> ContactConditions::derefine(
            const std::shared_ptr<FunctionSpace> &coarse_space,
            const bool as_zero) const {
        auto mesh = (mesh_t *)impl_->space->mesh().impl_mesh();
        auto et = (enum ElemType)impl_->space->element_type();

        const ptrdiff_t max_coarse_idx =
                max_node_id(coarse_space->element_type(), mesh->nelements, mesh->elements);

        auto coarse = std::make_shared<ContactConditions>(coarse_space);

        coarse->impl_->conditions =
                (boundary_condition_t *)malloc(impl_->n_conditions * sizeof(boundary_condition_t));
        coarse->impl_->n_conditions = 0;

        for (int i = 0; i < impl_->n_conditions; i++) {
            ptrdiff_t coarse_local_size = 0;
            idx_t *coarse_indices = nullptr;
            real_t *coarse_values = nullptr;
            hierarchical_create_coarse_indices(max_coarse_idx,
                                               impl_->conditions[i].local_size,
                                               impl_->conditions[i].idx,
                                               &coarse_local_size,
                                               &coarse_indices);

            if (!as_zero && impl_->conditions[i].values) {
                coarse_values = (real_t *)malloc(coarse_local_size * sizeof(real_t));

                hierarchical_collect_coarse_values(max_coarse_idx,
                                                   impl_->conditions[i].local_size,
                                                   impl_->conditions[i].idx,
                                                   impl_->conditions[i].values,
                                                   coarse_values);
            }

            long coarse_global_size = coarse_local_size;

            // MPI_CATCH_ERROR(
            // MPI_Allreduce(MPI_IN_PLACE, &coarse_global_size, 1, MPI_LONG, MPI_SUM, mesh->comm));

            if (as_zero) {
                boundary_condition_create(&coarse->impl_->conditions[coarse->impl_->n_conditions++],
                                          coarse_local_size,
                                          coarse_global_size,
                                          coarse_indices,
                                          impl_->conditions[i].component,
                                          0,
                                          nullptr);

            } else {
                boundary_condition_create(&coarse->impl_->conditions[coarse->impl_->n_conditions++],
                                          coarse_local_size,
                                          coarse_global_size,
                                          coarse_indices,
                                          impl_->conditions[i].component,
                                          impl_->conditions[i].value,
                                          coarse_values);
            }
        }

        return coarse;
    }

    std::shared_ptr<Constraint> ContactConditions::lor() const {
        assert(false);
        return nullptr;
    }

    ContactConditions::~ContactConditions() = default;

    void ContactConditions::add_condition(const ptrdiff_t local_size,
                                          const ptrdiff_t global_size,
                                          idx_t *const idx,
                                          const int component,
                                          const real_t value) {
        impl_->conditions = (boundary_condition_t *)realloc(
                impl_->conditions, (impl_->n_conditions + 1) * sizeof(boundary_condition_t));

        boundary_condition_create(&impl_->conditions[impl_->n_conditions],
                                  local_size,
                                  global_size,
                                  idx,
                                  component,
                                  value,
                                  nullptr);

        impl_->n_conditions++;
    }

    void ContactConditions::add_condition(const ptrdiff_t local_size,
                                          const ptrdiff_t global_size,
                                          idx_t *const idx,
                                          const int component,
                                          real_t *const values) {
        impl_->conditions = (boundary_condition_t *)realloc(
                impl_->conditions, (impl_->n_conditions + 1) * sizeof(boundary_condition_t));

        boundary_condition_create(&impl_->conditions[impl_->n_conditions],
                                  local_size,
                                  global_size,
                                  idx,
                                  component,
                                  0,
                                  values);

        impl_->n_conditions++;
    }

    std::shared_ptr<ContactConditions> ContactConditions::create_from_env(
            const std::shared_ptr<FunctionSpace> &space) {
        //
        auto dc = std::make_unique<ContactConditions>(space);

        char *SFEM_CONTACT_NODESET = 0;
        char *SFEM_CONTACT_VALUE = 0;
        char *SFEM_CONTACT_COMPONENT = 0;
        SFEM_READ_ENV(SFEM_CONTACT_NODESET, );
        SFEM_READ_ENV(SFEM_CONTACT_VALUE, );
        SFEM_READ_ENV(SFEM_CONTACT_COMPONENT, );

        auto mesh = (mesh_t *)space->mesh().impl_mesh();
        read_dirichlet_conditions(mesh,
                                  SFEM_CONTACT_NODESET,
                                  SFEM_CONTACT_VALUE,
                                  SFEM_CONTACT_COMPONENT,
                                  &dc->impl_->conditions,
                                  &dc->impl_->n_conditions);

        return dc;
    }

    int ContactConditions::apply(real_t *const x) {
        for (int i = 0; i < impl_->n_conditions; i++) {
            if (impl_->conditions[i].values) {
                constraint_nodes_to_values_vec(impl_->conditions[i].local_size,
                                               impl_->conditions[i].idx,
                                               impl_->space->block_size(),
                                               impl_->conditions[i].component,
                                               impl_->conditions[i].values,
                                               x);

            } else {
                constraint_nodes_to_value_vec(impl_->conditions[i].local_size,
                                              impl_->conditions[i].idx,
                                              impl_->space->block_size(),
                                              impl_->conditions[i].component,
                                              impl_->conditions[i].value,
                                              x);
            }
        }

        return SFEM_SUCCESS;
    }

    int ContactConditions::gradient(const real_t *const x, real_t *const g) {
        for (int i = 0; i < impl_->n_conditions; i++) {
            constraint_gradient_nodes_to_value_vec(impl_->conditions[i].local_size,
                                                   impl_->conditions[i].idx,
                                                   impl_->space->block_size(),
                                                   impl_->conditions[i].component,
                                                   impl_->conditions[i].value,
                                                   x,
                                                   g);
        }

        return SFEM_SUCCESS;
    }

    int ContactConditions::apply_value(const real_t value, real_t *const x) {
        for (int i = 0; i < impl_->n_conditions; i++) {
            constraint_nodes_to_value_vec(impl_->conditions[i].local_size,
                                          impl_->conditions[i].idx,
                                          impl_->space->block_size(),
                                          impl_->conditions[i].component,
                                          value,
                                          x);
        }

        return SFEM_SUCCESS;
    }

    int ContactConditions::copy_constrained_dofs(const real_t *const src, real_t *const dest) {
        for (int i = 0; i < impl_->n_conditions; i++) {
            constraint_nodes_copy_vec(impl_->conditions[i].local_size,
                                      impl_->conditions[i].idx,
                                      impl_->space->block_size(),
                                      impl_->conditions[i].component,
                                      src,
                                      dest);
        }

        return SFEM_SUCCESS;
    }

    int ContactConditions::hessian_crs(const real_t *const x,
                                       const count_t *const rowptr,
                                       const idx_t *const colidx,
                                       real_t *const values) {
        for (int i = 0; i < impl_->n_conditions; i++) {
            crs_constraint_nodes_to_identity_vec(impl_->conditions[i].local_size,
                                                 impl_->conditions[i].idx,
                                                 impl_->space->block_size(),
                                                 impl_->conditions[i].component,
                                                 1,
                                                 rowptr,
                                                 colidx,
                                                 values);
        }

        return SFEM_SUCCESS;
    }
}  // namespace sfem
