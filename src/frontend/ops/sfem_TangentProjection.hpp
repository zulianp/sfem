#ifndef SFEM_TANGENT_PROJECTION_HPP
#define SFEM_TANGENT_PROJECTION_HPP

namespace sfem {

    /// Whether this build can project a tangent: LAPACK (SFEM_ENABLE_LAPACK).
    bool tangent_projection_available();

    /// A <- Q |Lambda| Q^T for the symmetric n x n matrix A = Q Lambda Q^T, n <= 9: positive
    /// semidefinite, and A itself where A already is. The absolute value rather than a clamp at
    /// zero keeps the curvature of a direction along which the energy is concave (Chen et al.,
    /// "Stabler Neo-Hookean Simulation: Absolute Eigenvalue Filtering for Projected Newton",
    /// SIGGRAPH 2024). The tangent dP/dF of a quadrature point is projected so; the element
    /// matrix, a weighted sum of B^T (dP/dF) B, is then positive semidefinite too.
    /// A positive definite A (a Cholesky factorisation exists) is returned as it is; any other
    /// goes through LAPACK dsyev, in double precision. Without LAPACK, or where dsyev fails, such
    /// an A becomes NaN, which the solve that reads it reports.
    void project_abs_symmetric(const int n, double *const A);
    void project_abs_symmetric(const int n, float *const A);

}  // namespace sfem

#endif  // SFEM_TANGENT_PROJECTION_HPP
