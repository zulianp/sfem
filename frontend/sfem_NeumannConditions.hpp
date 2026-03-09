#ifndef SFEM_NEUMANN_CONDITIONS_HPP
#define SFEM_NEUMANN_CONDITIONS_HPP

#include <mpi.h>
#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

#include "sfem_base.hpp"
#include "sfem_defs.hpp"
#include "sfem_aliases.hpp"
#include "sfem_ForwardDeclarations.hpp"
#include "smesh_mesh.hpp"

// Operator includes
#include "sfem_Op.hpp"


namespace sfem {

    class NeumannConditions final : public Op {
    public:
        struct Condition {
            smesh::ElemType            element_type { smesh::INVALID };
            std::vector<std::shared_ptr<Sideset>> sidesets;  /// Maybe empty in certain cases
            SharedBuffer<idx_t *>    surface;
            SharedBuffer<real_t>     values;
            real_t                   value{0};
            int                      component{0};
        };

        static std::shared_ptr<NeumannConditions> create_from_env(const std::shared_ptr<FunctionSpace> &space);

        static std::shared_ptr<NeumannConditions> create(const std::shared_ptr<FunctionSpace> &space,
                                                         const std::vector<struct Condition>  &conditions);

        std::shared_ptr<Op> derefine_op(const std::shared_ptr<FunctionSpace> &derefined_space) override;

        const char *name() const override;

        NeumannConditions(const std::shared_ptr<FunctionSpace> &space);
        ~NeumannConditions();

        int hessian_crs(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override;

        int gradient(const real_t *const x, real_t *const out) override;

        int apply(const real_t *const x, const real_t *const h, real_t *const out) override;

        int value(const real_t *x, real_t *const out) override;

        int hessian_diag(const real_t *const x, real_t *const values) override;
        int hessian_block_diag_sym(const real_t *const, real_t *const) override { return SFEM_SUCCESS; }

        inline bool is_linear() const override { return true; }
        ptrdiff_t  n_dofs_domain() const override;
        ptrdiff_t  n_dofs_image() const override;

        int                            n_conditions() const;
        std::shared_ptr<FunctionSpace> space();
        std::vector<struct Condition> &conditions();

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace sfem

#endif // SFEM_NEUMANN_CONDITIONS_HPP 