#ifndef SFEM_DIRICHLET_CONDITIONS_HPP
#define SFEM_DIRICHLET_CONDITIONS_HPP

#include <mpi.h>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "sfem_base.h"
#include "sfem_defs.h"
#include "sfem_mask.h"
#include "sfem_Buffer.hpp"
#include "sfem_ForwardDeclarations.hpp"
#include "sfem_Mesh.hpp"
#include "sfem_Sideset.hpp"
#include "sfem_Constraint.hpp"

namespace sfem {

    class DirichletConditions final : public Constraint {
    public:
        struct Condition {
            std::vector<std::shared_ptr<Sideset>> sidesets;  /// Maybe undefined in certain cases
            SharedBuffer<idx_t>      nodeset;
            SharedBuffer<real_t>     values;
            real_t                   value{0};
            int                      component{0};
        };

        DirichletConditions(const std::shared_ptr<FunctionSpace> &space);
        ~DirichletConditions();

        std::shared_ptr<FunctionSpace> space();
        std::vector<struct Condition> &conditions();

        static std::shared_ptr<DirichletConditions> create_from_env(const std::shared_ptr<FunctionSpace> &space);
        static std::shared_ptr<DirichletConditions> create_from_file(const std::shared_ptr<FunctionSpace> &space,
                                                                     const std::string &path);
        static std::shared_ptr<DirichletConditions> create_from_yaml(const std::shared_ptr<FunctionSpace> &space,
                                                                     std::string                           yaml);

        static std::shared_ptr<DirichletConditions> create(const std::shared_ptr<FunctionSpace> &space,
                                                           const std::vector<struct Condition>  &conditions);

        int apply(real_t *const x) override;
        int apply_value(const real_t value, real_t *const x) override;
        int copy_constrained_dofs(const real_t *const src, real_t *const dest) override;
        int mask(mask_t *mask) override;

        int value(const real_t *const x, real_t *const out) override;

        int gradient(const real_t *const x, real_t *const g) override;

        int hessian_crs(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override;

        int hessian_bsr(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override;

        void add_condition(const ptrdiff_t local_size,
                           const ptrdiff_t global_size,
                           idx_t *const    idx,
                           const int       component,
                           const real_t    value);

        void add_condition(const ptrdiff_t local_size,
                           const ptrdiff_t global_size,
                           idx_t *const    idx,
                           const int       component,
                           real_t *const   values);

        int   n_conditions() const;

        std::shared_ptr<Constraint> derefine(const std::shared_ptr<FunctionSpace> &coarse_space,
                                             const bool                            as_zero) const override;
        std::shared_ptr<Constraint> lor() const override;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace sfem

#endif // SFEM_DIRICHLET_CONDITIONS_HPP 