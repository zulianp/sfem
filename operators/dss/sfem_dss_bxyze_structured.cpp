#include "sfem_dss_bxyze_structured.hpp"

#include <algorithm>
#include <array>

namespace {

constexpr int SFEM_DSS_MAX_LEVEL      = 8;
constexpr int SFEM_DSS_MAX_N          = SFEM_DSS_MAX_LEVEL + 1;
constexpr int SFEM_DSS_MAX_FACE_NODES = SFEM_DSS_MAX_N * SFEM_DSS_MAX_N;

bool valid_structured_shape(const int       level,
                            const ptrdiff_t nx,
                            const ptrdiff_t ny,
                            const ptrdiff_t nz,
                            const int       element_block_size) {
    return level >= 1 && level <= SFEM_DSS_MAX_LEVEL && nx > 0 && ny > 0 && nz > 0 && element_block_size > 0;
}

ptrdiff_t local_offset(const int n, const int element_block_size, const int ix, const int iy, const int iz) {
    return (ptrdiff_t)element_block_size *
           ((ptrdiff_t)ix + (ptrdiff_t)n * ((ptrdiff_t)iy + (ptrdiff_t)n * (ptrdiff_t)iz));
}

struct StructuredOffsets {
    ptrdiff_t                                      block_stride{0};
    std::array<ptrdiff_t, SFEM_DSS_MAX_FACE_NODES> z_left{};
    std::array<ptrdiff_t, SFEM_DSS_MAX_FACE_NODES> z_right{};
    std::array<ptrdiff_t, SFEM_DSS_MAX_FACE_NODES> y_left{};
    std::array<ptrdiff_t, SFEM_DSS_MAX_FACE_NODES> y_right{};
    std::array<ptrdiff_t, SFEM_DSS_MAX_FACE_NODES> x_left{};
    std::array<ptrdiff_t, SFEM_DSS_MAX_FACE_NODES> x_right{};
};

StructuredOffsets make_structured_offsets(const int n, const int bsize) {
    StructuredOffsets offsets;

    offsets.block_stride = (ptrdiff_t)bsize * (ptrdiff_t)n * (ptrdiff_t)n * (ptrdiff_t)n;

    for (int iy = 0; iy < n; ++iy) {
        for (int ix = 0; ix < n; ++ix) {
            const int face_node = ix + n * iy;
            offsets.z_left[face_node]  = local_offset(n, bsize, ix, iy, n - 1);
            offsets.z_right[face_node] = local_offset(n, bsize, ix, iy, 0);
        }
    }

    for (int iz = 0; iz < n; ++iz) {
        for (int ix = 0; ix < n; ++ix) {
            const int face_node = ix + n * iz;
            offsets.y_left[face_node]  = local_offset(n, bsize, ix, n - 1, iz);
            offsets.y_right[face_node] = local_offset(n, bsize, ix, 0, iz);
        }
    }

    for (int iz = 0; iz < n; ++iz) {
        for (int iy = 0; iy < n; ++iy) {
            const int face_node = iy + n * iz;
            offsets.x_left[face_node]  = local_offset(n, bsize, n - 1, iy, iz);
            offsets.x_right[face_node] = local_offset(n, bsize, 0, iy, iz);
        }
    }

    return offsets;
}

template <typename Scalar>
void exchange_linear_run(const int       bsize,
                         const ptrdiff_t block_stride,
                         const ptrdiff_t e_left,
                         const ptrdiff_t e_right,
                         const ptrdiff_t left_offset,
                         const ptrdiff_t right_offset,
                         const ptrdiff_t run_length,
                         Scalar *const SFEM_RESTRICT bxyze) {
    ptrdiff_t done = 0;

    while (done < run_length) {
        const ptrdiff_t le = e_left + done;
        const ptrdiff_t re = e_right + done;

        const ptrdiff_t leb = le / bsize;
        const ptrdiff_t reb = re / bsize;
        const int       lb  = (int)(le - leb * bsize);
        const int       rb  = (int)(re - reb * bsize);

        const ptrdiff_t remaining = run_length - done;
        const ptrdiff_t block_rem = std::min<ptrdiff_t>(bsize - lb, bsize - rb);
        const int       chunk     = (int)std::min(remaining, block_rem);

        Scalar *const SFEM_RESTRICT left  = &bxyze[block_stride * leb + left_offset + lb];
        Scalar *const SFEM_RESTRICT right = &bxyze[block_stride * reb + right_offset + rb];

#pragma omp simd
        for (int k = 0; k < chunk; ++k) {
            const Scalar s = left[k] + right[k];
            left[k]        = s;
            right[k]       = s;
        }

        done += chunk;
    }
}

template <typename Scalar>
void exchange_z_faces(const int       n,
                      const ptrdiff_t nx,
                      const ptrdiff_t ny,
                      const ptrdiff_t nz,
                      const int       bsize,
                      const StructuredOffsets &offsets,
                      Scalar *const SFEM_RESTRICT bxyze) {
    if (nz <= 1) {
        return;
    }

    const ptrdiff_t xy_stride = nx * ny;
    const ptrdiff_t x_tiles   = (nx + bsize - 1) / bsize;

#pragma omp for collapse(3) schedule(static)
    for (ptrdiff_t ez = 0; ez < nz - 1; ++ez) {
        for (ptrdiff_t ey = 0; ey < ny; ++ey) {
            for (ptrdiff_t tx = 0; tx < x_tiles; ++tx) {
                const ptrdiff_t x_begin    = tx * bsize;
                const ptrdiff_t run_length = std::min<ptrdiff_t>((ptrdiff_t)bsize, nx - x_begin);
                const ptrdiff_t e_left     = x_begin + nx * (ey + ny * ez);
                const ptrdiff_t e_right    = e_left + xy_stride;

                for (int iy = 0; iy < n; ++iy) {
                    for (int ix = 0; ix < n; ++ix) {
                        const int       face_node  = ix + n * iy;

                        exchange_linear_run(bsize,
                                            offsets.block_stride,
                                            e_left,
                                            e_right,
                                            offsets.z_left[face_node],
                                            offsets.z_right[face_node],
                                            run_length,
                                            bxyze);
                    }
                }
            }
        }
    }
}

template <typename Scalar>
void exchange_y_faces(const int       n,
                      const ptrdiff_t nx,
                      const ptrdiff_t ny,
                      const ptrdiff_t nz,
                      const int       bsize,
                      const StructuredOffsets &offsets,
                      Scalar *const SFEM_RESTRICT bxyze) {
    if (ny <= 1) {
        return;
    }

    const ptrdiff_t x_tiles = (nx + bsize - 1) / bsize;

#pragma omp for collapse(3) schedule(static)
    for (ptrdiff_t ez = 0; ez < nz; ++ez) {
        for (ptrdiff_t ey = 0; ey < ny - 1; ++ey) {
            for (ptrdiff_t tx = 0; tx < x_tiles; ++tx) {
                const ptrdiff_t x_begin    = tx * bsize;
                const ptrdiff_t run_length = std::min<ptrdiff_t>((ptrdiff_t)bsize, nx - x_begin);
                const ptrdiff_t e_left     = x_begin + nx * (ey + ny * ez);
                const ptrdiff_t e_right    = e_left + nx;

                for (int iz = 0; iz < n; ++iz) {
                    for (int ix = 0; ix < n; ++ix) {
                        const int       face_node  = ix + n * iz;

                        exchange_linear_run(bsize,
                                            offsets.block_stride,
                                            e_left,
                                            e_right,
                                            offsets.y_left[face_node],
                                            offsets.y_right[face_node],
                                            run_length,
                                            bxyze);
                    }
                }
            }
        }
    }
}

template <typename Scalar>
void exchange_x_faces(const int       n,
                      const ptrdiff_t nx,
                      const ptrdiff_t ny,
                      const ptrdiff_t nz,
                      const int       bsize,
                      const StructuredOffsets &offsets,
                      Scalar *const SFEM_RESTRICT bxyze) {
    if (nx <= 1) {
        return;
    }

    const ptrdiff_t n_interfaces = nx - 1;
    const ptrdiff_t x_tiles      = (n_interfaces + bsize - 1) / bsize;

#pragma omp for collapse(3) schedule(static)
    for (ptrdiff_t ez = 0; ez < nz; ++ez) {
        for (ptrdiff_t ey = 0; ey < ny; ++ey) {
            for (ptrdiff_t tx = 0; tx < x_tiles; ++tx) {
                const ptrdiff_t x_begin    = tx * bsize;
                const ptrdiff_t run_length = std::min<ptrdiff_t>((ptrdiff_t)bsize, n_interfaces - x_begin);
                const ptrdiff_t e_left     = x_begin + nx * (ey + ny * ez);
                const ptrdiff_t e_right    = e_left + 1;

                for (int iz = 0; iz < n; ++iz) {
                    for (int iy = 0; iy < n; ++iy) {
                        const int       face_node  = iy + n * iz;

                        exchange_linear_run(bsize,
                                            offsets.block_stride,
                                            e_left,
                                            e_right,
                                            offsets.x_left[face_node],
                                            offsets.x_right[face_node],
                                            run_length,
                                            bxyze);
                    }
                }
            }
        }
    }
}

template <typename Scalar>
int structured_inplace(const int       level,
                       const ptrdiff_t nx,
                       const ptrdiff_t ny,
                       const ptrdiff_t nz,
                       const int       element_block_size,
                       Scalar *const SFEM_RESTRICT bxyze) {
    if (!valid_structured_shape(level, nx, ny, nz, element_block_size) || !bxyze) {
        return SFEM_FAILURE;
    }

    const int n     = level + 1;
    const int bsize = element_block_size;

    const StructuredOffsets offsets = make_structured_offsets(n, bsize);

#pragma omp parallel
    {
        exchange_z_faces(n, nx, ny, nz, bsize, offsets, bxyze);
        exchange_y_faces(n, nx, ny, nz, bsize, offsets, bxyze);
        exchange_x_faces(n, nx, ny, nz, bsize, offsets, bxyze);
    }

    return SFEM_SUCCESS;
}

}  // namespace

int sfem_dss_bxyze_structured_inplace(const int       level,
                                      const ptrdiff_t nx,
                                      const ptrdiff_t ny,
                                      const ptrdiff_t nz,
                                      const int       element_block_size,
                                      real_t *const   bxyze) {
    return structured_inplace<real_t>(level, nx, ny, nz, element_block_size, bxyze);
}

int sfem_dss_bxyze_structured_inplace_real32(const int       level,
                                             const ptrdiff_t nx,
                                             const ptrdiff_t ny,
                                             const ptrdiff_t nz,
                                             const int       element_block_size,
                                             float *const    bxyze) {
    return structured_inplace<float>(level, nx, ny, nz, element_block_size, bxyze);
}

int sfem_dss_bxyze_structured_inplace_real64(const int       level,
                                             const ptrdiff_t nx,
                                             const ptrdiff_t ny,
                                             const ptrdiff_t nz,
                                             const int       element_block_size,
                                             double *const   bxyze) {
    return structured_inplace<double>(level, nx, ny, nz, element_block_size, bxyze);
}
