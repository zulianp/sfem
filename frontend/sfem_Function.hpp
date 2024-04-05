#ifndef SFEM_FUNCTION_HPP
#define SFEM_FUNCTION_HPP

#include "isolver_function.h"
#include <memory>

namespace sfem {

    class Function;
    class Mesh;

    class Mesh {
    public:
        Mesh();
        ~Mesh();

        friend class Function;
    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    class Function {
    public:
        Function();
        ~Function();

        int create_matrix_crs(ptrdiff_t *nlocal,
                              ptrdiff_t *nglobal,
                              ptrdiff_t *nnz,
                              isolver_idx_t **rowptr,
                              isolver_idx_t **colidx);

        int hessian_crs(const isolver_scalar_t *const x,
                        const isolver_idx_t *const rowptr,
                        const isolver_idx_t *const colidx,
                        isolver_scalar_t *const values);

        int gradient(const isolver_scalar_t *const x, isolver_scalar_t *const out);
        int apply(const isolver_scalar_t *const x,
                  const isolver_scalar_t *const h,
                  isolver_scalar_t *const out);

        int apply_constraints(isolver_scalar_t *const x);
        int apply_zero_constraints(isolver_scalar_t *const x);
        int copy_constrained_dofs(const isolver_scalar_t *const src, isolver_scalar_t *const dest);
        int report_solution(const isolver_scalar_t *const x);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
}  // namespace sfem

#endif  // SFEM_FUNCTION_HPP
