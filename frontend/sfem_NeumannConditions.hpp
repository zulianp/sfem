#ifndef SFEM_NEUMANN_CONDITIONS_HPP
#define SFEM_NEUMANN_CONDITIONS_HPP

#include <mpi.h>
#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

#include "sfem_base.h"
#include "sfem_defs.h"
#include "sfem_Buffer.hpp"
#include "sfem_ForwardDeclarations.hpp"
#include "sfem_Mesh.hpp"
#include "sfem_Sideset.hpp"

// Operator includes
#include "sfem_Op.hpp"


namespace sfem {

    class NeumannConditions final : public Op {
    public:
        struct Condition {
            enum ElemType            element_type { INVALID };
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

        inline bool is_linear() const override { return true; }

        int                            n_conditions() const;
        std::shared_ptr<FunctionSpace> space();
        std::vector<struct Condition> &conditions();

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace sfem

#endif // SFEM_NEUMANN_CONDITIONS_HPP 