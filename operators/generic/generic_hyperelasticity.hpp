#ifndef SFEM_GENERIC_HYPERELASTICITY_HPP
#define SFEM_GENERIC_HYPERELASTICITY_HPP

#include "sfem_Op.hpp"

namespace sfem {

    struct Hyperelasticity3 {
        template <int NXE, typename MicroKernel, typename T>
        static int objective(MicroKernel                  micro_kernel,
                             const ptrdiff_t              nelements,
                             const ptrdiff_t              stride,
                             const ptrdiff_t              nnodes,
                             idx_t **const SFEM_RESTRICT  elements,
                             geom_t **const SFEM_RESTRICT points,
                             const ptrdiff_t              u_stride,
                             const T *const SFEM_RESTRICT ux,
                             const T *const SFEM_RESTRICT uy,
                             const T *const SFEM_RESTRICT uz,
                             const int                    is_element_wise,
                             T *const SFEM_RESTRICT       out) {
            const geom_t *const x = points[0];
            const geom_t *const y = points[1];
            const geom_t *const z = points[2];

#pragma omp parallel for
            for (ptrdiff_t i = 0; i < nelements; ++i) {
                idx_t ev[NXE];

                T lx[NXE];
                T ly[NXE];
                T lz[NXE];

                T edispx[NXE];
                T edispy[NXE];
                T edispz[NXE];

                T jacobian_adjugate[9];
                T jacobian_determinant = 0;

                for (int v = 0; v < NXE; ++v) {
                    ev[v] = elements[v][i * stride];
                }

                for (int d = 0; d < NXE; d++) {
                    lx[d] = x[ev[d]];
                    ly[d] = y[ev[d]];
                    lz[d] = z[ev[d]];
                }

                for (int v = 0; v < NXE; ++v) {
                    const ptrdiff_t idx = ev[v] * u_stride;
                    edispx[v]           = ux[idx];
                    edispy[v]           = uy[idx];
                    edispz[v]           = uz[idx];
                }

                T v = 0;
                micro_kernel(lx, ly, lz, edispx, edispy, edispz, &v);
                assert(v == v);

                if (is_element_wise) {
                    out[i] = v;
                } else {
#pragma omp atomic update
                    *out += v;
                }
            }

            if (*out != *out) {
                *out = 1e10;
            }

            return SFEM_SUCCESS;
        }

        template <int NXE, typename MicroKernel, typename T>
        int objective_steps(MicroKernel                  micro_kernel,
                            const ptrdiff_t              nelements,
                            const ptrdiff_t              stride,
                            const ptrdiff_t              nnodes,
                            idx_t **const SFEM_RESTRICT  elements,
                            geom_t **const SFEM_RESTRICT points,
                            const T                      mu,
                            const T                      lambda,
                            const ptrdiff_t              u_stride,
                            const T *const SFEM_RESTRICT ux,
                            const T *const SFEM_RESTRICT uy,
                            const T *const SFEM_RESTRICT uz,
                            const ptrdiff_t              inc_stride,
                            const T *const SFEM_RESTRICT incx,
                            const T *const SFEM_RESTRICT incy,
                            const T *const SFEM_RESTRICT incz,
                            const int                    nsteps,
                            const T *const SFEM_RESTRICT steps,
                            T *const SFEM_RESTRICT       out) {
            const geom_t *const x = points[0];
            const geom_t *const y = points[1];
            const geom_t *const z = points[2];

#pragma omp parallel
            {
                T *out_local = (T *)calloc(nsteps, sizeof(T));

#pragma omp for
                for (ptrdiff_t i = 0; i < nelements; ++i) {
                    idx_t ev[NXE];

                    T lx[NXE];
                    T ly[NXE];
                    T lz[NXE];

                    T edispx[NXE];
                    T edispy[NXE];
                    T edispz[NXE];

                    T eincx[NXE];
                    T eincy[NXE];
                    T eincz[NXE];

                    for (int v = 0; v < NXE; ++v) {
                        ev[v] = elements[v][i * stride];
                    }

                    for (int d = 0; d < NXE; d++) {
                        lx[d] = x[ev[d]];
                        ly[d] = y[ev[d]];
                        lz[d] = z[ev[d]];
                    }

                    for (int v = 0; v < NXE; ++v) {
                        const ptrdiff_t idx = ev[v] * u_stride;
                        edispx[v]           = ux[idx];
                        edispy[v]           = uy[idx];
                        edispz[v]           = uz[idx];
                    }

                    for (int v = 0; v < NXE; ++v) {
                        const ptrdiff_t idx = ev[v] * inc_stride;
                        eincx[v]            = incx[idx];
                        eincy[v]            = incy[idx];
                        eincz[v]            = incz[idx];
                    }

                    micro_kernel(lx, ly, lz, edispx, edispy, edispz, eincx, eincy, eincz, nsteps, steps, out_local);
                }

                for (int s = 0; s < nsteps; s++) {
#pragma omp atomic update
                    out[s] += out_local[s];
                }

                free(out_local);
            }

            for (int s = 0; s < nsteps; s++) {
                if (out[s] != out[s]) {
                    out[s] = 1e10;
                }
            }

            return SFEM_SUCCESS;
        }

        template <int NXE, typename MicroKernel, typename T>
        int gradient(MicroKernel                  micro_kernel,
                     const ptrdiff_t              nelements,
                     const ptrdiff_t              stride,
                     const ptrdiff_t              nnodes,
                     idx_t **const SFEM_RESTRICT  elements,
                     geom_t **const SFEM_RESTRICT points,
                     const ptrdiff_t              u_stride,
                     const T *const SFEM_RESTRICT ux,
                     const T *const SFEM_RESTRICT uy,
                     const T *const SFEM_RESTRICT uz,
                     const ptrdiff_t              out_stride,
                     T *const SFEM_RESTRICT       outx,
                     T *const SFEM_RESTRICT       outy,
                     T *const SFEM_RESTRICT       outz) {
            const geom_t *const x = points[0];
            const geom_t *const y = points[1];
            const geom_t *const z = points[2];

#pragma omp parallel for
            for (ptrdiff_t i = 0; i < nelements; ++i) {
                idx_t ev[NXE];

                T lx[NXE];
                T ly[NXE];
                T lz[NXE];

                T edispx[NXE];
                T edispy[NXE];
                T edispz[NXE];

                T eoutx[NXE] = {0};
                T eouty[NXE] = {0};
                T eoutz[NXE] = {0};

                T jacobian_adjugate[9];
                T jacobian_determinant = 0;

                for (int v = 0; v < NXE; ++v) {
                    ev[v] = elements[v][i * stride];
                }

                for (int d = 0; d < NXE; d++) {
                    lx[d] = x[ev[d]];
                    ly[d] = y[ev[d]];
                    lz[d] = z[ev[d]];
                }

                for (int v = 0; v < NXE; ++v) {
                    const ptrdiff_t idx = ev[v] * u_stride;
                    edispx[v]           = ux[idx];
                    edispy[v]           = uy[idx];
                    edispz[v]           = uz[idx];
                }

                micro_kernel(lx, ly, lz, edispx, edispy, edispz, eoutx, eouty, eoutz);

                for (int edof_i = 0; edof_i < NXE; edof_i++) {
                    const ptrdiff_t idx = ev[edof_i] * out_stride;

                    assert(eoutx[edof_i] == eoutx[edof_i]);
                    assert(eouty[edof_i] == eouty[edof_i]);
                    assert(eoutz[edof_i] == eoutz[edof_i]);

#pragma omp atomic update
                    outx[idx] += eoutx[edof_i];

#pragma omp atomic update
                    outy[idx] += eouty[edof_i];

#pragma omp atomic update
                    outz[idx] += eoutz[edof_i];
                }
            }

            return SFEM_SUCCESS;
        }

        template <int NXE, int IKMN_SIZE, typename MicroKernel, typename T>
        int hessian_partial_assembly(const ptrdiff_t                      nelements,
                                     const ptrdiff_t                      stride,
                                     idx_t **const SFEM_RESTRICT          elements,
                                     geom_t **const SFEM_RESTRICT         points,
                                     const ptrdiff_t                      u_stride,
                                     const T *const SFEM_RESTRICT         ux,
                                     const T *const SFEM_RESTRICT         uy,
                                     const T *const SFEM_RESTRICT         uz,
                                     metric_tensor_t *const SFEM_RESTRICT partial_assembly) {
            const geom_t *const x = points[0];
            const geom_t *const y = points[1];
            const geom_t *const z = points[2];

#pragma omp parallel for
            for (ptrdiff_t i = 0; i < nelements; ++i) {
                idx_t ev[NXE];
                T     element_ux[NXE];
                T     element_uy[NXE];
                T     element_uz[NXE];
                T     lx[NXE];
                T     ly[NXE];
                T     lz[NXE];

                T jacobian_adjugate[9];
                T jacobian_determinant = 0;

                for (int v = 0; v < NXE; ++v) {
                    ev[v] = elements[v][i * stride];
                }

                for (int v = 0; v < NXE; ++v) {
                    lx[v] = x[ev[v]];
                    ly[v] = y[ev[v]];
                    lz[v] = z[ev[v]];
                }

                for (int v = 0; v < NXE; ++v) {
                    const ptrdiff_t idx = ev[v] * u_stride;
                    element_ux[v]       = ux[idx];
                    element_uy[v]       = uy[idx];
                    element_uz[v]       = uz[idx];
                }

                micro_kernel(lx, ly, lz, element_ux, element_uy, element_uz, &partial_assembly[i * IKMN_SIZE]);
            }

            return SFEM_SUCCESS;
        }

        template <int NXE, int WIMPNSIZE, int IKMN_SIZE, typename WimpnKernel, typename MicroKernel, typename T>
        int partial_assembly_apply(WimpnKernel                                wimpn_kernel,
                                   MicroKernel                                micro_kernel,
                                   const ptrdiff_t                            nelements,
                                   const ptrdiff_t                            stride,
                                   idx_t **const SFEM_RESTRICT                elements,
                                   const metric_tensor_t *const SFEM_RESTRICT partial_assembly,
                                   const ptrdiff_t                            h_stride,
                                   const T *const                             hx,
                                   const T *const                             hy,
                                   const T *const                             hz,
                                   const ptrdiff_t                            out_stride,
                                   T *const                                   outx,
                                   T *const                                   outy,
                                   T *const                                   outz) {
            T Wimpn_compressed[WIMPNSIZE];
            wimpn_kernel(Wimpn_compressed);

#pragma omp parallel for
            for (ptrdiff_t i = 0; i < nelements; ++i) {
                idx_t ev[NXE];

                T element_hx[NXE];
                T element_hy[NXE];
                T element_hz[NXE];

                T eoutx[NXE];
                T eouty[NXE];
                T eoutz[NXE];

                for (int v = 0; v < NXE; ++v) {
                    ev[v] = elements[v][i * stride];
                }

                for (int v = 0; v < NXE; ++v) {
                    const ptrdiff_t idx = ev[v] * h_stride;
                    element_hx[v]       = hx[idx];
                    element_hy[v]       = hy[idx];
                    element_hz[v]       = hz[idx];
                }

                const metric_tensor_t *const pai = &partial_assembly[i * IKMN_SIZE];
                T                            S_ikmn[IKMN_SIZE];
                for (int k = 0; k < IKMN_SIZE; k++) {
                    S_ikmn[k] = pai[k];
                    assert(S_ikmn[k] == S_ikmn[k]);
                }

                micro_kernel(S_ikmn, Wimpn_compressed, element_hx, element_hy, element_hz, eoutx, eouty, eoutz);

                for (int edof_i = 0; edof_i < NXE; edof_i++) {
                    const ptrdiff_t idx = ev[edof_i] * out_stride;

                    assert(eoutx[edof_i] == eoutx[edof_i]);
                    assert(eouty[edof_i] == eouty[edof_i]);
                    assert(eoutz[edof_i] == eoutz[edof_i]);

#pragma omp atomic update
                    outx[idx] += eoutx[edof_i];

#pragma omp atomic update
                    outy[idx] += eouty[edof_i];

#pragma omp atomic update
                    outz[idx] += eoutz[edof_i];
                }
            }

            return SFEM_SUCCESS;
        }

        template <int NXE, int WIMPNSIZE, int IKMN_SIZE, typename WimpnKernel, typename MicroKernel, typename T>
        int compressed_partial_assembly_apply(WimpnKernel                             wimpn_kernel,
                                              MicroKernel                             micro_kernel,
                                              const ptrdiff_t                         nelements,
                                              const ptrdiff_t                         stride,
                                              idx_t **const SFEM_RESTRICT             elements,
                                              const compressed_t *const SFEM_RESTRICT partial_assembly,
                                              const scaling_t *const SFEM_RESTRICT    scaling,
                                              const ptrdiff_t                         h_stride,
                                              const T *const                          hx,
                                              const T *const                          hy,
                                              const T *const                          hz,
                                              const ptrdiff_t                         out_stride,
                                              T *const                                outx,
                                              T *const                                outy,
                                              T *const                                outz) {
            T Wimpn_compressed[WIMPNSIZE];
            wimpn_kernel(Wimpn_compressed);

#pragma omp parallel for
            for (ptrdiff_t i = 0; i < nelements; ++i) {
                idx_t ev[NXE];

                T element_hx[NXE];
                T element_hy[NXE];
                T element_hz[NXE];

                T eoutx[NXE];
                T eouty[NXE];
                T eoutz[NXE];

                for (int v = 0; v < NXE; ++v) {
                    ev[v] = elements[v][i * stride];
                }

                for (int v = 0; v < NXE; ++v) {
                    const ptrdiff_t idx = ev[v] * h_stride;
                    element_hx[v]       = hx[idx];
                    element_hy[v]       = hy[idx];
                    element_hz[v]       = hz[idx];
                }

                // Load and decompress low precision tensor
                const T                   s   = scaling[i];
                const compressed_t *const pai = &partial_assembly[i * IKMN_SIZE];
                T                         S_ikmn[IKMN_SIZE];
                for (int k = 0; k < IKMN_SIZE; k++) {
                    S_ikmn[k] = s * (T)(pai[k]);
                    assert(S_ikmn[k] == S_ikmn[k]);
                }

                micro_kernel(S_ikmn, Wimpn_compressed, element_hx, element_hy, element_hz, eoutx, eouty, eoutz);

                for (int edof_i = 0; edof_i < NXE; edof_i++) {
                    const ptrdiff_t idx = ev[edof_i] * out_stride;

                    assert(eoutx[edof_i] == eoutx[edof_i]);
                    assert(eouty[edof_i] == eouty[edof_i]);
                    assert(eoutz[edof_i] == eoutz[edof_i]);

#pragma omp atomic update
                    outx[idx] += eoutx[edof_i];

#pragma omp atomic update
                    outy[idx] += eouty[edof_i];

#pragma omp atomic update
                    outz[idx] += eoutz[edof_i];
                }
            }

            return SFEM_SUCCESS;
        }

        // Micro-kernel based Hessian diagonal assembly (AoS layout)
        // MicroKernel signature:
        //   mk(lx, ly, lz, ux, uy, uz, e_diag) where e_diag has size 3*NXE
        template <int NXE, typename MicroKernel, typename T>
        int hessian_diag(MicroKernel                  micro_kernel,
                         const ptrdiff_t              nelements,
                         const ptrdiff_t              stride,
                         const ptrdiff_t              nnodes,
                         idx_t **const SFEM_RESTRICT  elements,
                         geom_t **const SFEM_RESTRICT points,
                         const ptrdiff_t              u_stride,
                         const T *const SFEM_RESTRICT ux,
                         const T *const SFEM_RESTRICT uy,
                         const T *const SFEM_RESTRICT uz,
                         T *const SFEM_RESTRICT       out) {
            const geom_t *const x = points[0];
            const geom_t *const y = points[1];
            const geom_t *const z = points[2];

#pragma omp parallel for
            for (ptrdiff_t i = 0; i < nelements; ++i) {
                idx_t ev[NXE];

                T lx[NXE];
                T ly[NXE];
                T lz[NXE];

                T edispx[NXE];
                T edispy[NXE];
                T edispz[NXE];

                T ediag[3 * NXE];
                for (int k = 0; k < 3 * NXE; ++k) ediag[k] = 0;

                for (int v = 0; v < NXE; ++v) {
                    ev[v] = elements[v][i * stride];
                }

                for (int d = 0; d < NXE; d++) {
                    lx[d] = x[ev[d]];
                    ly[d] = y[ev[d]];
                    lz[d] = z[ev[d]];
                }

                for (int v = 0; v < NXE; ++v) {
                    const ptrdiff_t idx = ev[v] * u_stride;
                    edispx[v]           = ux[idx];
                    edispy[v]           = uy[idx];
                    edispz[v]           = uz[idx];
                }

                micro_kernel(lx, ly, lz, edispx, edispy, edispz, ediag);

                for (int edof_i = 0; edof_i < NXE; edof_i++) {
                    const ptrdiff_t idx = ev[edof_i] * 3;
#pragma omp atomic update
                    out[idx + 0] += ediag[3 * edof_i + 0];
#pragma omp atomic update
                    out[idx + 1] += ediag[3 * edof_i + 1];
#pragma omp atomic update
                    out[idx + 2] += ediag[3 * edof_i + 2];
                }
            }

            return SFEM_SUCCESS;
        }

        // Apply without precomputed partial assembly: compute S_ikmn on the fly
        // SKernel signature: mk_S(lx,ly,lz, ux,uy,uz, S_out[IKMN_SIZE])
        // WimpnKernel: fills Wimpn_compressed[WIMPNSIZE]
        // SdotHKernel: performs contraction to produce eoutx/y/z
        template <int NXE, int WIMPNSIZE, int IKMN_SIZE, typename SKernel, typename WimpnKernel, typename SdotHKernel, typename T>
        int apply_aos(SKernel                                   s_kernel,
                      WimpnKernel                               wimpn_kernel,
                      SdotHKernel                               sdot_kernel,
                      const ptrdiff_t                           nelements,
                      const ptrdiff_t                           stride,
                      idx_t **const SFEM_RESTRICT               elements,
                      geom_t **const SFEM_RESTRICT              points,
                      const ptrdiff_t                           u_stride,
                      const T *const                            ux,
                      const T *const                            uy,
                      const T *const                            uz,
                      const ptrdiff_t                           h_stride,
                      const T *const                            hx,
                      const T *const                            hy,
                      const T *const                            hz,
                      const ptrdiff_t                           out_stride,
                      T *const                                   outx,
                      T *const                                   outy,
                      T *const                                   outz) {
            const geom_t *const x = points[0];
            const geom_t *const y = points[1];
            const geom_t *const z = points[2];

            T Wimpn_compressed[WIMPNSIZE];
            wimpn_kernel(Wimpn_compressed);

#pragma omp parallel for
            for (ptrdiff_t i = 0; i < nelements; ++i) {
                idx_t ev[NXE];

                T lx[NXE];
                T ly[NXE];
                T lz[NXE];

                T e_ux[NXE];
                T e_uy[NXE];
                T e_uz[NXE];

                T e_hx[NXE];
                T e_hy[NXE];
                T e_hz[NXE];

                T eoutx[NXE];
                T eouty[NXE];
                T eoutz[NXE];

                for (int v = 0; v < NXE; ++v) {
                    ev[v] = elements[v][i * stride];
                }

                for (int d = 0; d < NXE; d++) {
                    lx[d] = x[ev[d]];
                    ly[d] = y[ev[d]];
                    lz[d] = z[ev[d]];
                }

                for (int v = 0; v < NXE; ++v) {
                    const ptrdiff_t iu = ev[v] * u_stride;
                    e_ux[v]             = ux[iu];
                    e_uy[v]             = uy[iu];
                    e_uz[v]             = uz[iu];

                    const ptrdiff_t ih = ev[v] * h_stride;
                    e_hx[v]             = hx[ih];
                    e_hy[v]             = hy[ih];
                    e_hz[v]             = hz[ih];
                }

                T S_ikmn[IKMN_SIZE];
                s_kernel(lx, ly, lz, e_ux, e_uy, e_uz, S_ikmn);

                sdot_kernel(S_ikmn, Wimpn_compressed, e_hx, e_hy, e_hz, eoutx, eouty, eoutz);

                for (int edof_i = 0; edof_i < NXE; edof_i++) {
                    const ptrdiff_t idx = ev[edof_i] * out_stride;

#pragma omp atomic update
                    outx[idx] += eoutx[edof_i];
#pragma omp atomic update
                    outy[idx] += eouty[edof_i];
#pragma omp atomic update
                    outz[idx] += eoutz[edof_i];
                }
            }

            return SFEM_SUCCESS;
        }

        // Note: no direct wrappers to external AoS C APIs here — this header 
        // is intended to provide reusable micro-kernel driven templates only.
    };
}  // namespace sfem

#endif
