#ifndef PACKED_ELEMENTS_H
#define PACKED_ELEMENTS_H

#include "sfem_base.hpp"
#include "sshex8.hpp"


static SFEM_INLINE void sshex8_SoA_pack_elements(const int                      level,
                                                 scalar_t **const SFEM_RESTRICT eu,
                                                 scalar_t *const SFEM_RESTRICT  X) {
    int ledix = 0;
    for (int zi = 0; zi < level; zi++) {
        for (int yi = 0; yi < level; yi++) {
            for (int xi = 0; xi < level; xi++) {
                // Convert to standard HEX8 local ordering (see 3-4 and 6-7)
                int lev[8] = {// Bottom
                              sshex8_lidx(level, xi, yi, zi),
                              sshex8_lidx(level, xi + 1, yi, zi),
                              sshex8_lidx(level, xi + 1, yi + 1, zi),
                              sshex8_lidx(level, xi, yi + 1, zi),
                              // Top
                              sshex8_lidx(level, xi, yi, zi + 1),
                              sshex8_lidx(level, xi + 1, yi, zi + 1),
                              sshex8_lidx(level, xi + 1, yi + 1, zi + 1),
                              sshex8_lidx(level, xi, yi + 1, zi + 1)};

                scalar_t *Xex = &X[ledix * 24];
                scalar_t *Xey = &X[ledix * 24 + 8];
                scalar_t *Xez = &X[ledix * 24 + 16];

                for (int i = 0; i < 8; i++) {
                    int lidx = lev[i];
                    Xex[i]   = eu[0][lidx];
                    Xey[i]   = eu[1][lidx];
                    Xez[i]   = eu[2][lidx];
                }

                ledix++;
            }
        }
    }
}

static SFEM_INLINE void sshex8_SoA_unpack_add_elements(const int                           level,
                                                       const scalar_t *const SFEM_RESTRICT Y,
                                                       scalar_t **const SFEM_RESTRICT      v) {
    int ledix = 0;
    for (int zi = 0; zi < level; zi++) {
        for (int yi = 0; yi < level; yi++) {
            for (int xi = 0; xi < level; xi++) {
                // Convert to standard HEX8 local ordering (see 3-4 and 6-7)
                int lev[8] = {// Bottom
                              sshex8_lidx(level, xi, yi, zi),
                              sshex8_lidx(level, xi + 1, yi, zi),
                              sshex8_lidx(level, xi + 1, yi + 1, zi),
                              sshex8_lidx(level, xi, yi + 1, zi),
                              // Top
                              sshex8_lidx(level, xi, yi, zi + 1),
                              sshex8_lidx(level, xi + 1, yi, zi + 1),
                              sshex8_lidx(level, xi + 1, yi + 1, zi + 1),
                              sshex8_lidx(level, xi, yi + 1, zi + 1)};

                const scalar_t *const SFEM_RESTRICT Yex = &Y[ledix * 24];
                const scalar_t *const SFEM_RESTRICT Yey = &Y[ledix * 24 + 8];
                const scalar_t *const SFEM_RESTRICT Yez = &Y[ledix * 24 + 16];

                for (int i = 0; i < 8; i++) {
                    int lidx = lev[i];
                    v[0][lidx] += Yex[i];
                    v[1][lidx] += Yey[i];
                    v[2][lidx] += Yez[i];
                }

                ledix++;
            }
        }
    }
}

// BLAS function declaration
#ifdef SFEM_ENABLE_BLAS
#ifdef __cplusplus
extern "C" {
#endif
extern void dgemm_(const char   *transa,
                   const char   *transb,
                   const int    *m,
                   const int    *n,
                   const int    *k,
                   const double *alpha,
                   const double *a,
                   const int    *lda,
                   const double *b,
                   const int    *ldb,
                   const double *beta,
                   double       *c,
                   const int    *ldc);

extern void sgemm_(const char  *transa,
                   const char  *transb,
                   const int   *m,
                   const int   *n,
                   const int   *k,
                   const float *alpha,
                   const float *a,
                   const int   *lda,
                   const float *b,
                   const int   *ldb,
                   const float *beta,
                   float       *c,
                   const int   *ldc);
#ifdef __cplusplus
}
#endif

// SYMMETRIC ELEMENT MATRICES ONLY. The name is the warning: this routine computes
// different things in its two branches, and the difference cancels only when the element
// matrix equals its own transpose.
//
// The fallback evaluates Y_j[i] = sum_k element_matrix[i*K + k] * X_j[k], reading the
// element matrix row-major. This branch passes transa='N' with lda=k, so column-major
// dgemm reads the same buffer as its transpose and computes element_matrix^T * X_j. Both
// are dimensionally valid when m == k and neither errors, so nothing announces the
// disagreement -- it appears only in the values, and only for a non-symmetric matrix.
//
// Every current caller is safe: linear elasticity, the stencil element-matrix apply, and
// the NeoHookean tangents are all second derivatives of an energy and therefore
// symmetric. Anything else must use packed_elements_matmul_nonsym below, which computes
// element_matrix * X_j in both branches.
static SFEM_INLINE void packed_elements_matmul_sym(const int                           m,
                                               const int                           n,
                                               const int                           k,
                                               const scalar_t *const SFEM_RESTRICT element_matrix,
                                               const scalar_t *const SFEM_RESTRICT X,
                                               scalar_t *const SFEM_RESTRICT       Y) {
    char transa = 'N';
    char transb = 'N';
    int  ldm    = k;
    int  ldx    = k;
    int  ldy    = k;

    if (sizeof(scalar_t) == 8) {
        double alpha = 1;
        double beta  = 0;
        dgemm_(&transa,
               &transb,
               &m,
               &n,
               &k,
               &alpha,
               reinterpret_cast<const double *>(element_matrix),
               &ldm,
               reinterpret_cast<const double *>(X),
               &ldx,
               &beta,
               reinterpret_cast<double *>(Y),
               &ldy);
    } else {
        float alpha = 1;
        float beta  = 0;
        sgemm_(&transa,
               &transb,
               &m,
               &n,
               &k,
               &alpha,
               reinterpret_cast<const float *>(element_matrix),
               &ldm,
               reinterpret_cast<const float *>(X),
               &ldx,
               &beta,
               reinterpret_cast<float *>(Y),
               &ldy);
    }
}


// Any element matrix, symmetric or not: computes Y_j = element_matrix * X_j for each of
// the n right-hand sides, reading the matrix row-major, and gives the same answer whether
// or not BLAS is enabled. The only difference from _sym above is transa='T', which undoes
// the column-major reinterpretation of a row-major buffer.
static SFEM_INLINE void packed_elements_matmul_nonsym(const int                           m,
                                                      const int                           n,
                                                      const int                           k,
                                                      const scalar_t *const SFEM_RESTRICT element_matrix,
                                                      const scalar_t *const SFEM_RESTRICT X,
                                                      scalar_t *const SFEM_RESTRICT       Y) {
    char transa = 'T';
    char transb = 'N';
    int  ldm    = k;
    int  ldx    = k;
    int  ldy    = m;

    if (sizeof(scalar_t) == 8) {
        double alpha = 1;
        double beta  = 0;
        dgemm_(&transa, &transb, &m, &n, &k, &alpha,
               reinterpret_cast<const double *>(element_matrix), &ldm,
               reinterpret_cast<const double *>(X), &ldx, &beta,
               reinterpret_cast<double *>(Y), &ldy);
    } else {
        float alpha = 1;
        float beta  = 0;
        sgemm_(&transa, &transb, &m, &n, &k, &alpha,
               reinterpret_cast<const float *>(element_matrix), &ldm,
               reinterpret_cast<const float *>(X), &ldx, &beta,
               reinterpret_cast<float *>(Y), &ldy);
    }
}

#else

static SFEM_INLINE void packed_elements_matmul_sym(const int                           M,
                                               const int                           N,
                                               const int                           K,
                                               const scalar_t *const SFEM_RESTRICT element_matrix,
                                               const scalar_t *const SFEM_RESTRICT X,
                                               scalar_t *const SFEM_RESTRICT       Y) {
    for (int j = 0; j < N; j++) {
        scalar_t *const SFEM_RESTRICT       Yj = &Y[j * K];
        const scalar_t *const SFEM_RESTRICT Xj = &X[j * K];
        for (int i = 0; i < M; i++) {
            const scalar_t *const SFEM_RESTRICT element_matrix_i = &element_matrix[i * K];
            scalar_t                            acc              = 0;
            for (int k = 0; k < K; k++) {
                acc += element_matrix_i[k] * Xj[k];
            }
            Yj[i] = acc;
        }
    }
}

static SFEM_INLINE void packed_elements_matmul_nonsym(const int                           M,
                                                      const int                           N,
                                                      const int                           K,
                                                      const scalar_t *const SFEM_RESTRICT element_matrix,
                                                      const scalar_t *const SFEM_RESTRICT X,
                                                      scalar_t *const SFEM_RESTRICT       Y) {
    // The fallback already reads the element matrix row-major, so it needs no transpose
    // and the two routines coincide here.
    packed_elements_matmul_sym(M, N, K, element_matrix, X, Y);
}

// TODO SME version

#endif
#endif
