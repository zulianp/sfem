#include "sfem_ContactConditions_cuda.hpp"
#include "sfem_Function.hpp"

#include "boundary_condition.h"
#include "cu_boundary_condition.h"

namespace sfem {

    class GPUContactConditions final : public Constraint {
    public:
        std::shared_ptr<FunctionSpace> space;
        int n_conditions{0};
        boundary_condition_t *conditions{nullptr};
        std::shared_ptr<ContactConditions> h_contact_conditions;

        GPUContactConditions(const std::shared_ptr<ContactConditions> &dc)
            : space(dc->space()), h_contact_conditions(dc) {
            n_conditions = dc->n_conditions();
            auto *h_buffer = (boundary_condition_t *)dc->impl_conditions();

            conditions =
                    (boundary_condition_t *)malloc(n_conditions * sizeof(boundary_condition_t));

            for (int d = 0; d < n_conditions; d++) {
                boundary_conditions_host_to_device(&h_buffer[d], &conditions[d]);
            }
        }

        int apply(real_t *const x) override {
            for (int i = 0; i < n_conditions; i++) {
                if (conditions[i].values) {
                    d_constraint_nodes_to_values_vec(conditions[i].local_size,
                                                     conditions[i].idx,
                                                     space->block_size(),
                                                     conditions[i].component,
                                                     conditions[i].values,
                                                     x);

                } else {
                    d_constraint_nodes_to_value_vec(conditions[i].local_size,
                                                    conditions[i].idx,
                                                    space->block_size(),
                                                    conditions[i].component,
                                                    conditions[i].value,
                                                    x);
                }
            }

            return SFEM_SUCCESS;
        }

        int gradient(const real_t *const x, real_t *const g) override {
            for (int i = 0; i < n_conditions; i++) {
                d_constraint_gradient_nodes_to_value_vec(conditions[i].local_size,
                                                         conditions[i].idx,
                                                         space->block_size(),
                                                         conditions[i].component,
                                                         conditions[i].value,
                                                         x,
                                                         g);
            }

            return SFEM_SUCCESS;

            // assert(false);
            // return SFEM_FAILURE;
        }

        int apply_value(const real_t value, real_t *const x) override {
            for (int i = 0; i < n_conditions; i++) {
                d_constraint_nodes_to_value_vec(conditions[i].local_size,
                                                conditions[i].idx,
                                                space->block_size(),
                                                conditions[i].component,
                                                value,
                                                x);
            }

            return SFEM_SUCCESS;
        }

        int copy_constrained_dofs(const real_t *const src, real_t *const dest) override {
            for (int i = 0; i < n_conditions; i++) {
                d_constraint_nodes_copy_vec(conditions[i].local_size,
                                            conditions[i].idx,
                                            space->block_size(),
                                            conditions[i].component,
                                            src,
                                            dest);
            }

            return SFEM_SUCCESS;
        }

        int hessian_crs(const real_t *const x,
                        const count_t *const rowptr,
                        const idx_t *const colidx,
                        real_t *const values) override {
            for (int i = 0; i < n_conditions; i++) {
                cu_crs_constraint_nodes_to_identity_vec(conditions[i].local_size,
                                                        conditions[i].idx,
                                                        space->block_size(),
                                                        conditions[i].component,
                                                        1,
                                                        rowptr,
                                                        colidx,
                                                        values);
            }

            return SFEM_SUCCESS;
        }

        std::shared_ptr<Constraint> lor() const override {
            assert(false);
            return nullptr;
        }

        std::shared_ptr<Constraint> derefine(const std::shared_ptr<sfem::FunctionSpace> &space,
                                             bool as_zeros) const override {
            auto h_derefined = std::static_pointer_cast<ContactConditions>(
                    h_contact_conditions->derefine(space, as_zeros));
            return std::make_shared<GPUContactConditions>(h_derefined);
        }

        int mask(mask_t *mask) override
        {
            assert(false);
            return SFEM_FAILURE;
        }

        ~GPUContactConditions() { d_destroy_conditions(n_conditions, conditions); }
    };

    std::shared_ptr<Constraint> to_device(const std::shared_ptr<ContactConditions> &dc) {
        return std::make_shared<GPUContactConditions>(dc);
    }



}  // namespace sfem
