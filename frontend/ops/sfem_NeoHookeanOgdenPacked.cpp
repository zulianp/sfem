#include "sfem_NeoHookeanOgdenPacked.hpp"
#include "sfem_Tracer.hpp"

#include "sfem_Env.hpp"
#include "sfem_defs.h"
#include "sfem_logger.h"
#include "sfem_macros.h"
#include "sfem_mesh.h"

#include "hex8_fff.h"
#include "hex8_laplacian_inline_cpu.h"
#include "hex8_neohookean_ogden.h"
#include "hex8_neohookean_ogden_local.h"
#include "hex8_partial_assembly_neohookean_inline.h"
#include "laplacian.h"
#include "line_quadrature.h"
#include "neohookean_ogden.h"
#include "tet10_laplacian_inline_cpu.h"
#include "tet10_neohookean_ogden_local.h"
#include "tet10_partial_assembly_neohookean_ogden_inline.h"
#include "tet4_inline_cpu.h"
#include "tet4_laplacian_inline_cpu.h"
#include "tet4_partial_assembly_neohookean_inline.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "sfem_CRSGraph.hpp"
#include "sfem_FunctionSpace.hpp"
#include "sfem_Mesh.hpp"

#include "sfem_ElasticityAssemblyData.hpp"
#include "sfem_MultiDomainOp.hpp"
#include "sfem_OpTracer.hpp"
#include "sfem_Packed.hpp"
#include "sfem_Parameters.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

using PackedIdxType = sfem::FunctionSpace::PackedIdxType;

struct HyperelasticityScratch {
    struct ThreadData {
        ptrdiff_t max_nodes_per_pack;
        real_t   *in[3];
        real_t   *out[3];
        geom_t   *px;
        geom_t   *py;
        geom_t   *pz;

        ThreadData(const ptrdiff_t max_nodes_per_pack) : max_nodes_per_pack(max_nodes_per_pack) {
            for (int d = 0; d < 3; d++) {
                in[d]  = (real_t *)malloc(max_nodes_per_pack * sizeof(real_t));
                out[d] = (real_t *)calloc(max_nodes_per_pack, sizeof(real_t));
            }
            px = (geom_t *)malloc(max_nodes_per_pack * sizeof(geom_t));
            py = (geom_t *)malloc(max_nodes_per_pack * sizeof(geom_t));
            pz = (geom_t *)malloc(max_nodes_per_pack * sizeof(geom_t));
        }

        ~ThreadData() {
            for (int d = 0; d < 3; d++) {
                free(in[d]);
                free(out[d]);
            }
            free(px);
            free(py);
            free(pz);
        }

        void reset() {
            for (int d = 0; d < 3; d++) {
                memset(out[d], 0, max_nodes_per_pack * sizeof(real_t));
            }
        }
    };

    std::vector<std::unique_ptr<ThreadData>> thread_data;

    real_t **in(int thread_id) { return thread_data[thread_id]->in; }
    real_t **out(int thread_id) { return thread_data[thread_id]->out; }
    geom_t  *px(int thread_id) { return thread_data[thread_id]->px; }
    geom_t  *py(int thread_id) { return thread_data[thread_id]->py; }
    geom_t  *pz(int thread_id) { return thread_data[thread_id]->pz; }

    void reset() {
#ifdef _OPENMP
#pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            thread_data[thread_id]->reset();
        }
#else
        thread_data[0]->reset();
#endif
    }

    HyperelasticityScratch(const ptrdiff_t max_nodes_per_pack) {
#ifdef _OPENMP
        int n_threads = omp_get_max_threads();
        thread_data.resize(n_threads);
#pragma omp parallel
        {
            int thread_id          = omp_get_thread_num();
            thread_data[thread_id] = std::make_unique<ThreadData>(max_nodes_per_pack);
        }
#else
        thread_data.emplace_back(std::make_unique<ThreadData>(max_nodes_per_pack));
#endif
    }

    ~HyperelasticityScratch() {}
};

template <typename pack_idx_t, int NXE, typename MicroKernel>
struct NeoHookeanOgdenPackedGradient {
    static int apply(const ptrdiff_t                      n_packs,
                     const ptrdiff_t                      n_elements_per_pack,
                     const ptrdiff_t                      n_elements,
                     const ptrdiff_t                      max_nodes_per_pack,
                     pack_idx_t **const SFEM_RESTRICT     elements,
                     geom_t **const SFEM_RESTRICT         points,
                     const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
                     const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
                     const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
                     const idx_t *const SFEM_RESTRICT     ghost_idx,
                     const real_t                         mu,
                     const real_t                         lambda,
                     const ptrdiff_t                      u_stride,
                     const real_t *const SFEM_RESTRICT    ux,
                     const real_t *const SFEM_RESTRICT    uy,
                     const real_t *const SFEM_RESTRICT    uz,
                     const ptrdiff_t                      out_stride,
                     real_t *const SFEM_RESTRICT          outx,
                     real_t *const SFEM_RESTRICT          outy,
                     real_t *const SFEM_RESTRICT          outz,
                     HyperelasticityScratch              &scratch) {
        const geom_t *const x = points[0];
        const geom_t *const y = points[1];
        const geom_t *const z = points[2];

#pragma omp parallel
        {
#ifdef _OPENMP
            int thread_id = omp_get_thread_num();
#else
            int thread_id = 0;
#endif
            real_t **in  = scratch.in(thread_id);
            real_t **out = scratch.out(thread_id);
            geom_t  *px  = scratch.px(thread_id);
            geom_t  *py  = scratch.py(thread_id);
            geom_t  *pz  = scratch.pz(thread_id);

            for (int d = 0; d < 3; d++) {
                memset(out[d], 0, max_nodes_per_pack * sizeof(real_t));
            }

#pragma omp for schedule(static)
            for (ptrdiff_t p = 0; p < n_packs; p++) {
                const ptrdiff_t                  e_start      = p * n_elements_per_pack;
                const ptrdiff_t                  e_end        = MIN(n_elements, (p + 1) * n_elements_per_pack);
                const ptrdiff_t                  n_contiguous = owned_nodes_ptr[p + 1] - owned_nodes_ptr[p];
                const ptrdiff_t                  n_shared     = n_shared_nodes[p];
                const ptrdiff_t                  n_ghost      = ghost_ptr[p + 1] - ghost_ptr[p];
                const ptrdiff_t                  n_not_shared = n_contiguous - n_shared;
                const idx_t *const SFEM_RESTRICT ghosts       = &ghost_idx[ghost_ptr[p]];

                for (ptrdiff_t k = 0; k < n_contiguous; k++) {
                    const ptrdiff_t idx = (owned_nodes_ptr[p] + k) * u_stride;
                    in[0][k]            = ux[idx];
                    in[1][k]            = uy[idx];
                    in[2][k]            = uz[idx];
                }

                memcpy(px, &x[owned_nodes_ptr[p]], n_contiguous * sizeof(geom_t));
                memcpy(py, &y[owned_nodes_ptr[p]], n_contiguous * sizeof(geom_t));
                memcpy(pz, &z[owned_nodes_ptr[p]], n_contiguous * sizeof(geom_t));

                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const ptrdiff_t gidx    = ghosts[k] * u_stride;
                    in[0][n_contiguous + k] = ux[gidx];
                    in[1][n_contiguous + k] = uy[gidx];
                    in[2][n_contiguous + k] = uz[gidx];
                }

                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const ptrdiff_t gidx = ghosts[k];
                    px[n_contiguous + k] = x[gidx];
                    py[n_contiguous + k] = y[gidx];
                    pz[n_contiguous + k] = z[gidx];
                }

                for (ptrdiff_t e = e_start; e < e_end; e++) {
                    pack_idx_t ev[NXE];
                    for (int v = 0; v < NXE; ++v) {
                        ev[v] = elements[v][e];
                    }

                    scalar_t element_u[3][NXE];
                    for (int d = 0; d < 3; d++) {
                        for (int v = 0; v < NXE; ++v) {
                            element_u[d][v] = in[d][ev[v]];
                        }
                    }

                    accumulator_t element_out[3][NXE];
                    scalar_t      lx[NXE];
                    scalar_t      ly[NXE];
                    scalar_t      lz[NXE];

                    for (int v = 0; v < NXE; ++v) {
                        lx[v] = px[ev[v]];
                        ly[v] = py[ev[v]];
                        lz[v] = pz[ev[v]];
                    }

                    for (int d = 0; d < 3; d++) {
                        for (int v = 0; v < NXE; ++v) {
                            element_out[d][v] = 0;
                        }
                    }

                    MicroKernel::apply(lx,
                                       ly,
                                       lz,
                                       mu,
                                       lambda,
                                       element_u[0],
                                       element_u[1],
                                       element_u[2],
                                       element_out[0],
                                       element_out[1],
                                       element_out[2]);

                    for (int d = 0; d < 3; d++) {
                        for (int v = 0; v < NXE; ++v) {
                            assert(element_out[d][v] == element_out[d][v]);
                            out[d][ev[v]] += element_out[d][v];
                        }
                    }
                }

                real_t *const SFEM_RESTRICT accx = &outx[owned_nodes_ptr[p] * out_stride];
                real_t *const SFEM_RESTRICT accy = &outy[owned_nodes_ptr[p] * out_stride];
                real_t *const SFEM_RESTRICT accz = &outz[owned_nodes_ptr[p] * out_stride];

                for (ptrdiff_t k = 0; k < n_not_shared; ++k) {
                    const ptrdiff_t idx = k * out_stride;
                    // No need for atomic there are no collisions
                    accx[idx] += out[0][k];
                    accy[idx] += out[1][k];
                    accz[idx] += out[2][k];

                    out[0][k] = 0;
                    out[1][k] = 0;
                    out[2][k] = 0;
                }

                for (ptrdiff_t k = n_not_shared; k < n_contiguous; ++k) {
                    const ptrdiff_t idx = k * out_stride;
#pragma omp atomic update
                    accx[idx] += out[0][k];
                    out[0][k] = 0;

#pragma omp atomic update
                    accy[idx] += out[1][k];
                    out[1][k] = 0;

#pragma omp atomic update
                    accz[idx] += out[2][k];
                    out[2][k] = 0;
                }

                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const ptrdiff_t idx = ghosts[k] * out_stride;
#pragma omp atomic update
                    outx[idx] += out[0][n_contiguous + k];
                    out[0][n_contiguous + k] = 0;
#pragma omp atomic update
                    outy[idx] += out[1][n_contiguous + k];
                    out[1][n_contiguous + k] = 0;
#pragma omp atomic update
                    outz[idx] += out[2][n_contiguous + k];
                    out[2][n_contiguous + k] = 0;
                }
            }
        }

        return SFEM_SUCCESS;
    }
};

struct Hex8MicroKernelGradient {
    static SFEM_INLINE void apply(const scalar_t *const SFEM_RESTRICT lx,
                                  const scalar_t *const SFEM_RESTRICT ly,
                                  const scalar_t *const SFEM_RESTRICT lz,
                                  const scalar_t                      mu,
                                  const scalar_t                      lambda,
                                  const scalar_t *const SFEM_RESTRICT edispx,
                                  const scalar_t *const SFEM_RESTRICT edispy,
                                  const scalar_t *const SFEM_RESTRICT edispz,
                                  scalar_t *const SFEM_RESTRICT       eoutx,
                                  scalar_t *const SFEM_RESTRICT       eouty,
                                  scalar_t *const SFEM_RESTRICT       eoutz) {
        static const int                     n_qp = line_q2_n;
        static const scalar_t *SFEM_RESTRICT qx   = line_q2_x;
        static const scalar_t *SFEM_RESTRICT qw   = line_q2_w;

        scalar_t jacobian_adjugate[9];
        scalar_t jacobian_determinant = 0;

        for (int kz = 0; kz < n_qp; kz++) {
            for (int ky = 0; ky < n_qp; ky++) {
                for (int kx = 0; kx < n_qp; kx++) {
                    hex8_adjugate_and_det(lx, ly, lz, qx[kx], qx[ky], qx[kz], jacobian_adjugate, &jacobian_determinant);
                    assert(jacobian_determinant == jacobian_determinant);
                    assert(jacobian_determinant != 0);

                    hex8_neohookean_ogden_grad(jacobian_adjugate,
                                               jacobian_determinant,
                                               qx[kx],
                                               qx[ky],
                                               qx[kz],
                                               qw[kx] * qw[ky] * qw[kz],
                                               mu,
                                               lambda,
                                               edispx,
                                               edispy,
                                               edispz,
                                               eoutx,
                                               eouty,
                                               eoutz);
                }
            }
        }
    }
};

struct Tet4MicroKernelGradient {
    static SFEM_INLINE void apply(const scalar_t *const SFEM_RESTRICT lx,
                                  const scalar_t *const SFEM_RESTRICT ly,
                                  const scalar_t *const SFEM_RESTRICT lz,
                                  const scalar_t                      mu,
                                  const scalar_t                      lambda,
                                  const scalar_t *const SFEM_RESTRICT edispx,
                                  const scalar_t *const SFEM_RESTRICT edispy,
                                  const scalar_t *const SFEM_RESTRICT edispz,
                                  scalar_t *const SFEM_RESTRICT       eoutx,
                                  scalar_t *const SFEM_RESTRICT       eouty,
                                  scalar_t *const SFEM_RESTRICT       eoutz) {}
};

struct Tet10MicroKernelGradient {
    static SFEM_INLINE void apply(const scalar_t *const SFEM_RESTRICT lx,
                                  const scalar_t *const SFEM_RESTRICT ly,
                                  const scalar_t *const SFEM_RESTRICT lz,
                                  const scalar_t                      mu,
                                  const scalar_t                      lambda,
                                  const scalar_t *const SFEM_RESTRICT edispx,
                                  const scalar_t *const SFEM_RESTRICT edispy,
                                  const scalar_t *const SFEM_RESTRICT edispz,
                                  scalar_t *const SFEM_RESTRICT       eoutx,
                                  scalar_t *const SFEM_RESTRICT       eouty,
                                  scalar_t *const SFEM_RESTRICT       eoutz) {
#define n_qp 35
        static real_t qw[n_qp] = {
                0.0021900463965388, 0.0021900463965388, 0.0021900463965388, 0.0021900463965388, 0.0143395670177665,
                0.0143395670177665, 0.0143395670177665, 0.0143395670177665, 0.0143395670177665, 0.0143395670177665,
                0.0143395670177665, 0.0143395670177665, 0.0143395670177665, 0.0143395670177665, 0.0143395670177665,
                0.0143395670177665, 0.0250305395686746, 0.0250305395686746, 0.0250305395686746, 0.0250305395686746,
                0.0250305395686746, 0.0250305395686746, 0.0479839333057554, 0.0479839333057554, 0.0479839333057554,
                0.0479839333057554, 0.0479839333057554, 0.0479839333057554, 0.0479839333057554, 0.0479839333057554,
                0.0479839333057554, 0.0479839333057554, 0.0479839333057554, 0.0479839333057554, 0.0931745731195340};

        static real_t qx[n_qp] = {
                0.0267367755543735, 0.9197896733368801, 0.0267367755543735, 0.0267367755543735, 0.7477598884818091,
                0.1740356302468940, 0.0391022406356488, 0.0391022406356488, 0.0391022406356488, 0.0391022406356488,
                0.1740356302468940, 0.7477598884818091, 0.1740356302468940, 0.7477598884818091, 0.0391022406356488,
                0.0391022406356488, 0.4547545999844830, 0.0452454000155172, 0.0452454000155172, 0.4547545999844830,
                0.4547545999844830, 0.0452454000155172, 0.2232010379623150, 0.5031186450145980, 0.2232010379623150,
                0.2232010379623150, 0.5031186450145980, 0.2232010379623150, 0.0504792790607720, 0.0504792790607720,
                0.0504792790607720, 0.5031186450145980, 0.2232010379623150, 0.2232010379623150, 0.2500000000000000};

        static real_t qy[n_qp] = {
                0.0267367755543735, 0.0267367755543735, 0.9197896733368801, 0.0267367755543735, 0.0391022406356488,
                0.0391022406356488, 0.7477598884818091, 0.1740356302468940, 0.0391022406356488, 0.0391022406356488,
                0.7477598884818091, 0.1740356302468940, 0.0391022406356488, 0.0391022406356488, 0.1740356302468940,
                0.7477598884818091, 0.0452454000155172, 0.4547545999844830, 0.0452454000155172, 0.4547545999844830,
                0.0452454000155172, 0.4547545999844830, 0.2232010379623150, 0.2232010379623150, 0.5031186450145980,
                0.0504792790607720, 0.0504792790607720, 0.0504792790607720, 0.2232010379623150, 0.5031186450145980,
                0.2232010379623150, 0.2232010379623150, 0.5031186450145980, 0.2232010379623150, 0.2500000000000000};

        static real_t qz[n_qp] = {
                0.0267367755543735, 0.0267367755543735, 0.0267367755543735, 0.9197896733368801, 0.0391022406356488,
                0.0391022406356488, 0.0391022406356488, 0.0391022406356488, 0.7477598884818091, 0.1740356302468940,
                0.0391022406356488, 0.0391022406356488, 0.7477598884818091, 0.1740356302468940, 0.7477598884818091,
                0.1740356302468940, 0.0452454000155172, 0.0452454000155172, 0.4547545999844830, 0.0452454000155172,
                0.4547545999844830, 0.4547545999844830, 0.0504792790607720, 0.0504792790607720, 0.0504792790607720,
                0.2232010379623150, 0.2232010379623150, 0.5031186450145980, 0.2232010379623150, 0.2232010379623150,
                0.5031186450145980, 0.2232010379623150, 0.2232010379623150, 0.5031186450145980, 0.2500000000000000};

        scalar_t jacobian_adjugate[9];
        scalar_t jacobian_determinant = 0;

        for (int k = 0; k < n_qp; k++) {
            tet10_adjugate_and_det(lx, ly, lz, qx[k], qy[k], qz[k], jacobian_adjugate, &jacobian_determinant);
            assert(jacobian_determinant == jacobian_determinant);
            assert(jacobian_determinant != 0);

            tet10_neohookean_ogden_grad(jacobian_adjugate,
                                        jacobian_determinant,
                                        qx[k],
                                        qy[k],
                                        qz[k],
                                        qw[k],
                                        mu,
                                        lambda,
                                        edispx,
                                        edispy,
                                        edispz,
                                        eoutx,
                                        eouty,
                                        eoutz);
        }

#undef n_qp
    }
};

template <typename pack_idx_t>
static int packed_neohookean_ogden_gradient(enum ElemType                        element_type,
                                            const ptrdiff_t                      n_packs,
                                            const ptrdiff_t                      n_elements_per_pack,
                                            const ptrdiff_t                      n_elements,
                                            const ptrdiff_t                      max_nodes_per_pack,
                                            pack_idx_t **const SFEM_RESTRICT     elements,
                                            geom_t **const SFEM_RESTRICT         points,
                                            const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
                                            const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
                                            const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
                                            const idx_t *const SFEM_RESTRICT     ghost_idx,
                                            const real_t                         mu,
                                            const real_t                         lambda,
                                            const ptrdiff_t                      u_stride,
                                            const real_t *const SFEM_RESTRICT    ux,
                                            const real_t *const SFEM_RESTRICT    uy,
                                            const real_t *const SFEM_RESTRICT    uz,
                                            const ptrdiff_t                      out_stride,
                                            real_t *const SFEM_RESTRICT          outx,
                                            real_t *const SFEM_RESTRICT          outy,
                                            real_t *const SFEM_RESTRICT          outz,
                                            HyperelasticityScratch              &scratch) {
    switch (element_type) {
        case HEX8:
            return NeoHookeanOgdenPackedGradient<PackedIdxType, 8, Hex8MicroKernelGradient>::apply(n_packs,
                                                                                                   n_elements_per_pack,
                                                                                                   n_elements,
                                                                                                   max_nodes_per_pack,
                                                                                                   elements,
                                                                                                   points,
                                                                                                   owned_nodes_ptr,
                                                                                                   n_shared_nodes,
                                                                                                   ghost_ptr,
                                                                                                   ghost_idx,
                                                                                                   mu,
                                                                                                   lambda,
                                                                                                   u_stride,
                                                                                                   ux,
                                                                                                   uy,
                                                                                                   uz,
                                                                                                   out_stride,
                                                                                                   outx,
                                                                                                   outy,
                                                                                                   outz,
                                                                                                   scratch);
        case TET4:
            return NeoHookeanOgdenPackedGradient<PackedIdxType, 4, Tet4MicroKernelGradient>::apply(n_packs,
                                                                                                   n_elements_per_pack,
                                                                                                   n_elements,
                                                                                                   max_nodes_per_pack,
                                                                                                   elements,
                                                                                                   points,
                                                                                                   owned_nodes_ptr,
                                                                                                   n_shared_nodes,
                                                                                                   ghost_ptr,
                                                                                                   ghost_idx,
                                                                                                   mu,
                                                                                                   lambda,
                                                                                                   u_stride,
                                                                                                   ux,
                                                                                                   uy,
                                                                                                   uz,
                                                                                                   out_stride,
                                                                                                   outx,
                                                                                                   outy,
                                                                                                   outz,
                                                                                                   scratch);
        case TET10:
            return NeoHookeanOgdenPackedGradient<PackedIdxType, 10, Tet10MicroKernelGradient>::apply(n_packs,
                                                                                                     n_elements_per_pack,
                                                                                                     n_elements,
                                                                                                     max_nodes_per_pack,
                                                                                                     elements,
                                                                                                     points,
                                                                                                     owned_nodes_ptr,
                                                                                                     n_shared_nodes,
                                                                                                     ghost_ptr,
                                                                                                     ghost_idx,
                                                                                                     mu,
                                                                                                     lambda,
                                                                                                     u_stride,
                                                                                                     ux,
                                                                                                     uy,
                                                                                                     uz,
                                                                                                     out_stride,
                                                                                                     outx,
                                                                                                     outy,
                                                                                                     outz,
                                                                                                     scratch);

        default: {
            SFEM_ERROR("packed_neohookean_ogden_gradient not implemented for type %s\n", type_to_string(element_type));
            return SFEM_FAILURE;
        }
    }
}

template <typename pack_idx_t, int NXE, typename MicroKernel>
struct HyperelasticityPackedApply {
    static int apply(const ptrdiff_t                            n_packs,
                     const ptrdiff_t                            n_elements_per_pack,
                     const ptrdiff_t                            n_elements,
                     const ptrdiff_t                            max_nodes_per_pack,
                     pack_idx_t **const SFEM_RESTRICT           elements,
                     const ptrdiff_t *const SFEM_RESTRICT       owned_nodes_ptr,
                     const ptrdiff_t *const SFEM_RESTRICT       n_shared_nodes,
                     const ptrdiff_t *const SFEM_RESTRICT       ghost_ptr,
                     const idx_t *const SFEM_RESTRICT           ghost_idx,
                     const metric_tensor_t *const SFEM_RESTRICT partial_assembly,
                     const scalar_t *const SFEM_RESTRICT        Wimpn_compressed,
                     const ptrdiff_t                            h_stride,
                     const real_t *const SFEM_RESTRICT          hx,
                     const real_t *const SFEM_RESTRICT          hy,
                     const real_t *const SFEM_RESTRICT          hz,
                     const ptrdiff_t                            out_stride,
                     real_t *const SFEM_RESTRICT                outx,
                     real_t *const SFEM_RESTRICT                outy,
                     real_t *const SFEM_RESTRICT                outz,
                     HyperelasticityScratch                    &scratch) {
#pragma omp parallel
        {
#ifdef _OPENMP
            int thread_id = omp_get_thread_num();
#else
            int thread_id = 0;
#endif
            real_t **in  = scratch.in(thread_id);
            real_t **out = scratch.out(thread_id);

            for (int d = 0; d < 3; d++) {
                memset(out[d], 0, max_nodes_per_pack * sizeof(real_t));
            }

#pragma omp for schedule(static)
            for (ptrdiff_t p = 0; p < n_packs; p++) {
                const ptrdiff_t                  e_start      = p * n_elements_per_pack;
                const ptrdiff_t                  e_end        = MIN(n_elements, (p + 1) * n_elements_per_pack);
                const ptrdiff_t                  n_contiguous = owned_nodes_ptr[p + 1] - owned_nodes_ptr[p];
                const ptrdiff_t                  n_shared     = n_shared_nodes[p];
                const ptrdiff_t                  n_ghost      = ghost_ptr[p + 1] - ghost_ptr[p];
                const ptrdiff_t                  n_not_shared = n_contiguous - n_shared;
                const idx_t *const SFEM_RESTRICT ghosts       = &ghost_idx[ghost_ptr[p]];

                for (ptrdiff_t k = 0; k < n_contiguous; k++) {
                    const ptrdiff_t idx = (owned_nodes_ptr[p] + k) * h_stride;
                    in[0][k]            = hx[idx];
                    in[1][k]            = hy[idx];
                    in[2][k]            = hz[idx];
                }

                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const ptrdiff_t gidx    = ghosts[k] * h_stride;
                    in[0][n_contiguous + k] = hx[gidx];
                    in[1][n_contiguous + k] = hy[gidx];
                    in[2][n_contiguous + k] = hz[gidx];
                }

                for (ptrdiff_t e = e_start; e < e_end; e++) {
                    pack_idx_t ev[NXE];
                    for (int v = 0; v < NXE; ++v) {
                        ev[v] = elements[v][e];
                    }

                    scalar_t element_u[3][NXE];
                    for (int d = 0; d < 3; d++) {
                        for (int v = 0; v < NXE; ++v) {
                            element_u[d][v] = in[d][ev[v]];
                        }
                    }

                    accumulator_t element_out[3][NXE];
                    for (int d = 0; d < 3; d++) {
                        for (int v = 0; v < NXE; ++v) {
                            element_out[d][v] = 0;
                        }
                    }

                    const metric_tensor_t *const pai = &partial_assembly[e * HEX8_S_IKMN_SIZE];
                    scalar_t                     S_ikmn[HEX8_S_IKMN_SIZE];
                    for (int k = 0; k < HEX8_S_IKMN_SIZE; k++) {
                        S_ikmn[k] = pai[k];
                        assert(S_ikmn[k] == S_ikmn[k]);
                    }

                    MicroKernel::apply(S_ikmn,
                                       Wimpn_compressed,
                                       element_u[0],
                                       element_u[1],
                                       element_u[2],
                                       element_out[0],
                                       element_out[1],
                                       element_out[2]);

                    for (int d = 0; d < 3; d++) {
                        for (int v = 0; v < NXE; ++v) {
                            assert(element_out[d][v] == element_out[d][v]);
                            out[d][ev[v]] += element_out[d][v];
                        }
                    }
                }

                real_t *const SFEM_RESTRICT accx = &outx[owned_nodes_ptr[p] * out_stride];
                real_t *const SFEM_RESTRICT accy = &outy[owned_nodes_ptr[p] * out_stride];
                real_t *const SFEM_RESTRICT accz = &outz[owned_nodes_ptr[p] * out_stride];

                for (ptrdiff_t k = 0; k < n_not_shared; ++k) {
                    const ptrdiff_t idx = k * out_stride;
                    // No need for atomic there are no collisions
                    accx[idx] += out[0][k];
                    accy[idx] += out[1][k];
                    accz[idx] += out[2][k];

                    out[0][k] = 0;
                    out[1][k] = 0;
                    out[2][k] = 0;
                }

                for (ptrdiff_t k = n_not_shared; k < n_contiguous; ++k) {
                    const ptrdiff_t idx = k * out_stride;
#pragma omp atomic update
                    accx[idx] += out[0][k];
                    out[0][k] = 0;

#pragma omp atomic update
                    accy[idx] += out[1][k];
                    out[1][k] = 0;

#pragma omp atomic update
                    accz[idx] += out[2][k];
                    out[2][k] = 0;
                }

                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const ptrdiff_t idx = ghosts[k] * out_stride;
#pragma omp atomic update
                    outx[idx] += out[0][n_contiguous + k];
                    out[0][n_contiguous + k] = 0;
#pragma omp atomic update
                    outy[idx] += out[1][n_contiguous + k];
                    out[1][n_contiguous + k] = 0;
#pragma omp atomic update
                    outz[idx] += out[2][n_contiguous + k];
                    out[2][n_contiguous + k] = 0;
                }
            }
        }
        return SFEM_SUCCESS;
    }
};

struct Hex8MicroKernelApply {
    inline static void apply(const scalar_t *const SFEM_RESTRICT S_ikmn,
                             const scalar_t *const SFEM_RESTRICT Wimpn,
                             const scalar_t *const SFEM_RESTRICT hx,
                             const scalar_t *const SFEM_RESTRICT hy,
                             const scalar_t *const SFEM_RESTRICT hz,
                             scalar_t *const SFEM_RESTRICT       outx,
                             scalar_t *const SFEM_RESTRICT       outy,
                             scalar_t *const SFEM_RESTRICT       outz) {
        hex8_SdotHdotG(S_ikmn, Wimpn, hx, hy, hz, outx, outy, outz);
    }
};

// struct Tet4MicroKernelApply {
//     inline static void apply(const scalar_t *const SFEM_RESTRICT S_ikmn,
//                              const scalar_t *const SFEM_RESTRICT Wimpn,
//                              const scalar_t *const SFEM_RESTRICT hx,
//                              const scalar_t *const SFEM_RESTRICT hy,
//                              const scalar_t *const SFEM_RESTRICT hz,
//                              scalar_t *const SFEM_RESTRICT       outx,
//                              scalar_t *const SFEM_RESTRICT       outy,
//                              scalar_t *const SFEM_RESTRICT       outz) {
//         tet4_SdotHdotG(S_ikmn, Wimpn, hx, hy, hz, outx, outy, outz);
//     }
// };

struct Tet10MicroKernelApply {
    inline static void apply(const scalar_t *const SFEM_RESTRICT S_ikmn,
                             const scalar_t *const SFEM_RESTRICT Wimpn,
                             const scalar_t *const SFEM_RESTRICT hx,
                             const scalar_t *const SFEM_RESTRICT hy,
                             const scalar_t *const SFEM_RESTRICT hz,
                             scalar_t *const SFEM_RESTRICT       outx,
                             scalar_t *const SFEM_RESTRICT       outy,
                             scalar_t *const SFEM_RESTRICT       outz) {
        tet10_SdotHdotG(S_ikmn, Wimpn, hx, hy, hz, outx, outy, outz);
    }
};
template <typename pack_idx_t>
static int packed_neohookean_ogden_apply(enum ElemType                              element_type,
                                         const ptrdiff_t                            n_packs,
                                         const ptrdiff_t                            n_elements_per_pack,
                                         const ptrdiff_t                            n_elements,
                                         const ptrdiff_t                            max_nodes_per_pack,
                                         pack_idx_t **const SFEM_RESTRICT           elements,
                                         const ptrdiff_t *const SFEM_RESTRICT       owned_nodes_ptr,
                                         const ptrdiff_t *const SFEM_RESTRICT       n_shared_nodes,
                                         const ptrdiff_t *const SFEM_RESTRICT       ghost_ptr,
                                         const idx_t *const SFEM_RESTRICT           ghost_idx,
                                         const metric_tensor_t *const SFEM_RESTRICT partial_assembly,
                                         const ptrdiff_t                            h_stride,
                                         const real_t *const SFEM_RESTRICT          hx,
                                         const real_t *const SFEM_RESTRICT          hy,
                                         const real_t *const SFEM_RESTRICT          hz,
                                         const ptrdiff_t                            out_stride,
                                         real_t *const SFEM_RESTRICT                outx,
                                         real_t *const SFEM_RESTRICT                outy,
                                         real_t *const SFEM_RESTRICT                outz,
                                         HyperelasticityScratch                    &scratch) {
    switch (element_type) {
        case HEX8: {
            scalar_t Wimpn_compressed[10];
            hex8_Wimpn_compressed(Wimpn_compressed);
            return HyperelasticityPackedApply<PackedIdxType, 8, Hex8MicroKernelApply>::apply(n_packs,
                                                                                             n_elements_per_pack,
                                                                                             n_elements,
                                                                                             max_nodes_per_pack,
                                                                                             elements,
                                                                                             owned_nodes_ptr,
                                                                                             n_shared_nodes,
                                                                                             ghost_ptr,
                                                                                             ghost_idx,
                                                                                             partial_assembly,
                                                                                             Wimpn_compressed,
                                                                                             h_stride,
                                                                                             hx,
                                                                                             hy,
                                                                                             hz,
                                                                                             out_stride,
                                                                                             outx,
                                                                                             outy,
                                                                                             outz,
                                                                                             scratch);
        }
            // case TET4: {
            // return HyperelasticityPackedApply<PackedIdxType, 4, Tet4MicroKernelApply>::apply(n_packs,
            //                                                                                  n_elements_per_pack,
            //                                                                                  n_elements,
            //                                                                                  max_nodes_per_pack,
            //                                                                                  elements,
            //                                                                                  owned_nodes_ptr,
            //                                                                                  n_shared_nodes,
            //                                                                                  ghost_ptr,
            //                                                                                  ghost_idx,
            //                                                                                  partial_assembly,
            //                                                                                  Wimpn_compressed,
            //                                                                                  h_stride,
            //                                                                                  hx,
            //                                                                                  hy,
            //                                                                                  hz,
            //                                                                                  out_stride,
            //                                                                                  outx,
            //                                                                                  outy,
            //                                                                                  outz,
            //                                                                                  scratch);
            // }
        case TET10: {
            scalar_t Wimpn_compressed[8];
            tet10_Wimpn_compressed(Wimpn_compressed);
            return HyperelasticityPackedApply<PackedIdxType, 10, Tet10MicroKernelApply>::apply(n_packs,
                                                                                               n_elements_per_pack,
                                                                                               n_elements,
                                                                                               max_nodes_per_pack,
                                                                                               elements,
                                                                                               owned_nodes_ptr,
                                                                                               n_shared_nodes,
                                                                                               ghost_ptr,
                                                                                               ghost_idx,
                                                                                               partial_assembly,
                                                                                               Wimpn_compressed,
                                                                                               h_stride,
                                                                                               hx,
                                                                                               hy,
                                                                                               hz,
                                                                                               out_stride,
                                                                                               outx,
                                                                                               outy,
                                                                                               outz,
                                                                                               scratch);
        }

        default: {
            SFEM_ERROR("packed_neohookean_ogden_apply not implemented for type %s\n", type_to_string(element_type));
            return SFEM_FAILURE;
        }
    }
}

namespace sfem {

    class NeoHookeanOgdenPacked::Impl {
    public:
        std::shared_ptr<FunctionSpace>                              space;
        std::shared_ptr<MultiDomainOp>                              domains;
        std::shared_ptr<Packed<PackedIdxType>>                      packed;
        real_t                                                      mu{1}, lambda{1};
        std::vector<std::shared_ptr<HyperelasticityScratch>>        scratch;
        std::vector<std::shared_ptr<struct ElasticityAssemblyData>> assembly_data;

#if SFEM_PRINT_THROUGHPUT
        std::unique_ptr<OpTracer> op_profiler;
#endif
        Impl(const std::shared_ptr<FunctionSpace> &space) : space(space) {
#if SFEM_PRINT_THROUGHPUT
            op_profiler = std::make_unique<OpTracer>(space, "NeoHookeanOgdenPacked::apply");
#endif
        }
        ~Impl() {}

        void print_info() { domains->print_info(); }

        int iterate(const std::function<int(const OpDomain &)> &func) { return domains->iterate(func); }
    };

    int NeoHookeanOgdenPacked::initialize(const std::vector<std::string> &block_names) {
        SFEM_TRACE_SCOPE("NeoHookeanOgdenPacked::initialize");
        impl_->domains = std::make_shared<MultiDomainOp>(impl_->space, block_names);

        if (!impl_->space->has_packed_mesh()) {
            fprintf(stderr, "[Warning] NeoHookeanOgdenPacked: Initializing packed mesh, outer states may be inconsistent!\n");
            impl_->space->initialize_packed_mesh();
            fprintf(stderr, "[Warning] NeoHookeanOgdenPacked: Packed mesh initialized\n");
        }
        impl_->packed = impl_->space->packed_mesh();

        impl_->scratch.resize(impl_->packed->n_blocks());
        for (int b = 0; b < impl_->packed->n_blocks(); b++) {
            impl_->scratch[b] = std::make_shared<HyperelasticityScratch>(impl_->packed->max_nodes_per_pack());
        }

        impl_->assembly_data.resize(impl_->packed->n_blocks());

        for (int b = 0; b < impl_->packed->n_blocks(); b++) {
            auto name   = impl_->packed->block_name(b);
            auto domain = impl_->domains->domains().find(name);
            if (domain == impl_->domains->domains().end()) {
                SFEM_ERROR("Domain %s not found", name.c_str());
            }

            domain->second.user_data = std::static_pointer_cast<void>(std::make_shared<int>(b));
        }

        for (int b = 0; b < impl_->packed->n_blocks(); b++) {
            auto name                                     = impl_->packed->block_name(b);
            auto domain                                   = impl_->domains->domains().find(name);
            impl_->assembly_data[b]                       = std::make_shared<struct ElasticityAssemblyData>();
            impl_->assembly_data[b]->use_partial_assembly = true;
            impl_->assembly_data[b]->use_compression      = false;
            impl_->assembly_data[b]->use_AoS              = true;
            impl_->assembly_data[b]->elements             = domain->second.block->elements();
            impl_->assembly_data[b]->elements_stride      = 1;
            impl_->assembly_data[b]->elements             = convert_host_buffer_to_fake_SoA(
                    domain->second.block->n_nodes_per_element(),
                    soa_to_aos(1, domain->second.block->n_nodes_per_element(), domain->second.block->elements()));
            impl_->assembly_data[b]->elements_stride = domain->second.block->n_nodes_per_element();
            impl_->assembly_data[b]->partial_assembly_buffer =
                    sfem::create_host_buffer<metric_tensor_t>(domain->second.block->n_elements() * HEX8_S_IKMN_SIZE);
        }

        return SFEM_SUCCESS;
    }

    std::unique_ptr<Op> NeoHookeanOgdenPacked::create(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("NeoHookeanOgdenPacked::create");
        auto ret = std::make_unique<NeoHookeanOgdenPacked>(space);
        return ret;
    }

    std::shared_ptr<Op> NeoHookeanOgdenPacked::lor_op(const std::shared_ptr<FunctionSpace> &space) {
        auto ret            = std::make_shared<NeoHookeanOgdenPacked>(space);
        ret->impl_->domains = impl_->domains->lor_op(space, {});
        return ret;
    }

    std::shared_ptr<Op> NeoHookeanOgdenPacked::derefine_op(const std::shared_ptr<FunctionSpace> &space) {
        auto ret            = std::make_shared<NeoHookeanOgdenPacked>(space);
        ret->impl_->domains = impl_->domains->derefine_op(space, {});
        return ret;
    }

    NeoHookeanOgdenPacked::NeoHookeanOgdenPacked(const std::shared_ptr<FunctionSpace> &space)
        : impl_(std::make_unique<Impl>(space)) {}

    NeoHookeanOgdenPacked::~NeoHookeanOgdenPacked() = default;

    int NeoHookeanOgdenPacked::hessian_crs(const real_t *const  x,
                                           const count_t *const rowptr,
                                           const idx_t *const   colidx,
                                           real_t *const        values) {
        // SFEM_TRACE_SCOPE("NeoHookeanOgdenPacked::hessian_crs");

        // auto mesh  = impl_->space->mesh_ptr();
        // auto graph = impl_->space->dof_to_dof_graph();
        // int  err   = SFEM_SUCCESS;

        // impl_->iterate([&](const OpDomain &domain) {
        //     return laplacian_crs(domain.element_type,
        //                          domain.block->n_elements(),
        //                          mesh->n_nodes(),
        //                          domain.block->elements()->data(),
        //                          mesh->points()->data(),
        //                          graph->rowptr()->data(),
        //                          graph->colidx()->data(),
        //                          values);
        // });

        // return err;
        SFEM_ERROR("NeoHookeanOgdenPacked::hessian_crs not implemented");
        return SFEM_FAILURE;
    }

    int NeoHookeanOgdenPacked::hessian_crs_sym(const real_t *const  x,
                                               const count_t *const rowptr,
                                               const idx_t *const   colidx,
                                               real_t *const        diag_values,
                                               real_t *const        off_diag_values) {
        SFEM_TRACE_SCOPE("NeoHookeanOgdenPacked::hessian_crs_sym");

        // auto mesh = impl_->space->mesh_ptr();
        // int  err  = SFEM_SUCCESS;

        // impl_->iterate([&](const OpDomain &domain) {
        //     return laplacian_crs_sym(domain.element_type,
        //                              domain.block->n_elements(),
        //                              mesh->n_nodes(),
        //                              domain.block->elements()->data(),
        //                              mesh->points()->data(),
        //                              rowptr,
        //                              colidx,
        //                              diag_values,
        //                              off_diag_values);
        // });

        // return err;
        SFEM_ERROR("NeoHookeanOgdenPacked::hessian_crs_sym not implemented");
        return SFEM_FAILURE;
    }

    int NeoHookeanOgdenPacked::hessian_diag(const real_t *const x, real_t *const values) {
        SFEM_TRACE_SCOPE("NeoHookeanOgdenPacked::hessian_diag");

        auto mesh = impl_->space->mesh_ptr();
        return impl_->iterate([&](const OpDomain &domain) -> int {
            return neohookean_ogden_diag_aos(domain.element_type,
                                             mesh->n_elements(),
                                             1,
                                             mesh->n_nodes(),
                                             domain.block->elements()->data(),
                                             mesh->points()->data(),
                                             this->impl_->mu,
                                             this->impl_->lambda,
                                             x,
                                             values);
        });
    }

    int NeoHookeanOgdenPacked::update(const real_t *const u) {
        SFEM_TRACE_SCOPE("NeoHookeanOgdenPacked::update");

        auto mesh = impl_->space->mesh_ptr();
        return impl_->iterate([&](const OpDomain &domain) {
            auto b             = *std::static_pointer_cast<int>(domain.user_data);
            auto assembly_data = impl_->assembly_data[b];

            int ok = neohookean_ogden_hessian_partial_assembly(domain.element_type,
                                                               domain.block->n_elements(),
                                                               assembly_data->elements_stride,
                                                               assembly_data->elements->data(),
                                                               mesh->points()->data(),
                                                               domain.parameters->get_real_value("mu", impl_->mu),
                                                               domain.parameters->get_real_value("lambda", impl_->lambda),
                                                               3,
                                                               &u[0],
                                                               &u[1],
                                                               &u[2],
                                                               assembly_data->partial_assembly_buffer->data());
            assembly_data->compress_partial_assembly(domain);

            return ok;
        });
    }

    int NeoHookeanOgdenPacked::gradient(const real_t *const x, real_t *const out) {
        SFEM_TRACE_SCOPE("NeoHookeanOgdenPacked::gradient");

        auto mesh   = impl_->space->mesh_ptr();
        auto points = mesh->points();
        return impl_->iterate([&](const OpDomain &domain) {
            auto b               = *std::static_pointer_cast<int>(domain.user_data);
            auto elements        = impl_->packed->elements(b);
            auto owned_nodes_ptr = impl_->packed->owned_nodes_ptr(b);
            auto n_shared        = impl_->packed->n_shared(b);
            auto ghost_ptr       = impl_->packed->ghost_ptr(b);
            auto ghost_idx       = impl_->packed->ghost_idx(b);

            const real_t mu     = domain.parameters->get_real_value("mu", impl_->mu);
            const real_t lambda = domain.parameters->get_real_value("lambda", impl_->lambda);

            auto scratch = impl_->scratch[b];

            auto max_nodes_per_pack = impl_->packed->max_nodes_per_pack();
            return packed_neohookean_ogden_gradient<PackedIdxType>(domain.element_type,
                                                                   impl_->packed->n_packs(b),
                                                                   impl_->packed->n_elements_per_pack(b),
                                                                   elements->extent(1),
                                                                   max_nodes_per_pack,
                                                                   elements->data(),
                                                                   points->data(),
                                                                   owned_nodes_ptr->data(),
                                                                   n_shared->data(),
                                                                   ghost_ptr->data(),
                                                                   ghost_idx->data(),
                                                                   mu,
                                                                   lambda,
                                                                   3,
                                                                   &x[0],
                                                                   &x[1],
                                                                   &x[2],
                                                                   3,
                                                                   &out[0],
                                                                   &out[1],
                                                                   &out[2],
                                                                   *scratch);
            return SFEM_FAILURE;
        });
    }

    int NeoHookeanOgdenPacked::apply(const real_t *const /*x*/, const real_t *const h, real_t *const out) {
        SFEM_TRACE_SCOPE("NeoHookeanOgdenPacked::apply");

        auto mesh   = impl_->space->mesh_ptr();
        auto points = mesh->points();
        return impl_->iterate([&](const OpDomain &domain) {
            auto b               = *std::static_pointer_cast<int>(domain.user_data);
            auto elements        = impl_->packed->elements(b);
            auto owned_nodes_ptr = impl_->packed->owned_nodes_ptr(b);
            auto n_shared        = impl_->packed->n_shared(b);
            auto ghost_ptr       = impl_->packed->ghost_ptr(b);
            auto ghost_idx       = impl_->packed->ghost_idx(b);

            auto assembly_data = impl_->assembly_data[b];

            auto scratch = impl_->scratch[b];

            auto max_nodes_per_pack = impl_->packed->max_nodes_per_pack();
            return packed_neohookean_ogden_apply<PackedIdxType>(domain.element_type,
                                                                impl_->packed->n_packs(b),
                                                                impl_->packed->n_elements_per_pack(b),
                                                                elements->extent(1),
                                                                max_nodes_per_pack,
                                                                elements->data(),
                                                                owned_nodes_ptr->data(),
                                                                n_shared->data(),
                                                                ghost_ptr->data(),
                                                                ghost_idx->data(),
                                                                assembly_data->partial_assembly_buffer->data(),
                                                                3,
                                                                &h[0],
                                                                &h[1],
                                                                &h[2],
                                                                3,
                                                                &out[0],
                                                                &out[1],
                                                                &out[2],
                                                                *scratch);
        });
    }

    int NeoHookeanOgdenPacked::value(const real_t *x, real_t *const out) {
        SFEM_TRACE_SCOPE("NeoHookeanOgdenPacked::value");

        auto mesh = impl_->space->mesh_ptr();
        return impl_->iterate([&](const OpDomain &domain) -> int {
            auto b             = *std::static_pointer_cast<int>(domain.user_data);
            auto assembly_data = impl_->assembly_data[b];
            return neohookean_ogden_objective_aos(domain.element_type,
                                                  domain.block->n_elements(),
                                                  assembly_data->elements_stride,
                                                  mesh->n_nodes(),
                                                  assembly_data->elements->data(),
                                                  mesh->points()->data(),
                                                  domain.parameters->get_real_value("mu", impl_->mu),
                                                  domain.parameters->get_real_value("lambda", impl_->lambda),
                                                  x,
                                                  false,
                                                  out);
        });
    }

    int NeoHookeanOgdenPacked::value_steps(const real_t       *x,
                                           const real_t       *h,
                                           const int           nsteps,
                                           const real_t *const steps,
                                           real_t *const       out) {
        SFEM_TRACE_SCOPE("NeoHookeanOgdenPacked::value_steps");

        auto mesh = impl_->space->mesh_ptr();
        return impl_->iterate([&](const OpDomain &domain) -> int {
            auto b             = *std::static_pointer_cast<int>(domain.user_data);
            auto assembly_data = impl_->assembly_data[b];

            return neohookean_ogden_objective_steps_aos(domain.element_type,
                                                        mesh->n_elements(),
                                                        assembly_data->elements_stride,
                                                        mesh->n_nodes(),
                                                        assembly_data->elements->data(),
                                                        mesh->points()->data(),
                                                        domain.parameters->get_real_value("mu", impl_->mu),
                                                        domain.parameters->get_real_value("lambda", impl_->lambda),
                                                        x,
                                                        h,
                                                        nsteps,
                                                        steps,
                                                        out);
        });
    }

    int NeoHookeanOgdenPacked::report(const real_t *const) { return SFEM_SUCCESS; }

    std::shared_ptr<Op> NeoHookeanOgdenPacked::clone() const {
        SFEM_ERROR("IMPLEMENT ME!\n");
        return nullptr;
    }

    void NeoHookeanOgdenPacked::set_value_in_block(const std::string &block_name,
                                                   const std::string &var_name,
                                                   const real_t       value) {
        impl_->domains->set_value_in_block(block_name, var_name, value);
    }

    void NeoHookeanOgdenPacked::override_element_types(const std::vector<enum ElemType> &element_types) {
        impl_->domains->override_element_types(element_types);
    }

    void NeoHookeanOgdenPacked::set_mu(const real_t mu) { impl_->mu = mu; }
    void NeoHookeanOgdenPacked::set_lambda(const real_t lambda) { impl_->lambda = lambda; }
}  // namespace sfem
