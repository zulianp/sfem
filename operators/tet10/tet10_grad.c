#include "tet10_grad.h"

// Microkernel
static SFEM_INLINE void tet10_grad_kernel(const real_t px0,
                                  const real_t px1,
                                  const real_t px2,
                                  const real_t px3,
                                  const real_t py0,
                                  const real_t py1,
                                  const real_t py2,
                                  const real_t py3,
                                  const real_t pz0,
                                  const real_t pz1,
                                  const real_t pz2,
                                  const real_t pz3,
                                  // Data
                                  const real_t *SFEM_RESTRICT f,
                                  // Output
                                  real_t *SFEM_RESTRICT dfdx,
                                  real_t *SFEM_RESTRICT dfdy,
                                  real_t *SFEM_RESTRICT dfdz) {
	//TODO
}

void tet10_grad(const ptrdiff_t nelements,
                const ptrdiff_t nnodes,
                idx_t **const SFEM_RESTRICT elems,
                geom_t **SFEM_RESTRICT xyz,
                const real_t *const SFEM_RESTRICT f,
                real_t *const SFEM_RESTRICT dfdx,
                real_t *const SFEM_RESTRICT dfdy,
                real_t *const SFEM_RESTRICT dfdz) {
    SFEM_UNUSED(nnodes);

    double tick = MPI_Wtime();

    idx_t ev[10];
    real_t element_f[10];
    real_t element_dfdx[4];
    real_t element_dfdy[4];
    real_t element_dfdz[4];

    for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            ev[v] = elems[v][i];
        }

        for (int v = 0; v < 10; ++v) {
            element_f[v] = f[ev[v]];
        }

        // Element indices
        const idx_t i0 = ev[0];
        const idx_t i1 = ev[1];
        const idx_t i2 = ev[2];
        const idx_t i3 = ev[3];

        tet10_grad_kernel(
            // X-coordinates
            xyz[0][i0],
            xyz[0][i1],
            xyz[0][i2],
            xyz[0][i3],
            // Y-coordinates
            xyz[1][i0],
            xyz[1][i1],
            xyz[1][i2],
            xyz[1][i3],
            // Z-coordinates
            xyz[2][i0],
            xyz[2][i1],
            xyz[2][i2],
            xyz[2][i3],
            // Data
            element_f,
            // Output
            element_dfdx,
            element_dfdy,
            element_dfdz);

        // Write cell data
        for (int v = 0; v < 4; ++v) {
            dfdx[e * 4 + v] = element_dfdx[v];
        }

        for (int v = 0; v < 4; ++v) {
            dfdy[e * 4 + v] = element_dfdy[v];
        }

        for (int v = 0; v < 4; ++v) {
            dfdz[e * 4 + v] = element_dfdz[v];
        }
    }

    double tock = MPI_Wtime();
    printf("cgrad.c: cgrad3\t%g seconds\n", tock - tick);
}
