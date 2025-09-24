#include "sfem_SFC.hpp"



#include "argsort.h"

#include "sfem_API.hpp"
#include "sfem_Env.hpp"

#ifdef DSFEM_ENABLE_MPI_SORT
#include "mpi-sort.h"
#endif

#define SFEM_MPI_SFC_T MPI_UNSIGNED

typedef uint32_t sfc_t;

#define sort_function argsort_u32_element
#define fun_sfc morton3d
#define sfc_urange 1024u
// typedef idx_t element_idx_t;

// #define sort_function argsort_u32_ptrdiff_t
// typedef ptrdiff_t element_idx_t;

// https://mathworld.wolfram.com/FiedlerVector.html

// typedef uint64_t sfc_t;
// #define SFEM_MPI_SFC_T MPI_UNSIGNED_LONG
// #define sort_function argsort_u64

// Space-filling curve mapping function for 3D points
static SFEM_INLINE sfc_t morton3d(sfc_t x, sfc_t y, sfc_t z) {
    x &= 0x3FFFFFFF;
    y &= 0x3FFFFFFF;
    z &= 0x3FFFFFFF;
    x = (x | (x << 16)) & 0x030000FF;
    x = (x | (x << 8)) & 0x0300F00F;
    x = (x | (x << 4)) & 0x030C30C3;
    x = (x | (x << 2)) & 0x09249249;
    y = (y | (y << 16)) & 0x030000FF;
    y = (y | (y << 8)) & 0x0300F00F;
    y = (y | (y << 4)) & 0x030C30C3;
    y = (y | (y << 2)) & 0x09249249;
    z = (z | (z << 16)) & 0x030000FF;
    z = (z | (z << 8)) & 0x0300F00F;
    z = (z | (z << 4)) & 0x030C30C3;
    z = (z | (z << 2)) & 0x09249249;
    return x | (y << 1) | (z << 2);
}

static SFEM_INLINE sfc_t hilbert3d(sfc_t x, sfc_t y, sfc_t z) {
    sfc_t hilbert = 0;
    for (int i = 31; i >= 0; i--) {
        sfc_t mask = 1 << i;
        sfc_t rx = (x & mask) ? 1 : 0;
        sfc_t ry = (y & mask) ? 1 : 0;
        sfc_t rz = (z & mask) ? 1 : 0;
        hilbert = (hilbert << 3) | (rx ^ ry ^ rz);
        if (rz == 1) {
            hilbert = (hilbert << 2) | (rx ^ ry ^ 1);
            hilbert = (hilbert << 2) | (rx ^ 1);
            hilbert ^= 0xAAAAAAAB;
        } else {
            hilbert = (hilbert << 2) | (ry ^ rz);
            hilbert = (hilbert << 2) | rx;
            hilbert ^= 0x55555555;
        }
    }
    return hilbert;
}


namespace sfem {



    class SFC::Impl {
    public:
        int order_coordinate{-1};
    };

    SFC::SFC() : impl_(std::make_unique<Impl>()) {}
    SFC::~SFC() = default;

    std::shared_ptr<SFC> SFC::create_from_env() {
        auto ret = std::make_shared<SFC>();
        ret->impl_->order_coordinate  = sfem::Env::read<int>("SFEM_ORDER_WITH_COORDINATE", ret->impl_->order_coordinate);
        return ret;
    }

    int SFC::reorder(Mesh &mesh) {
        auto elements = mesh.elements()->data();
        auto points = mesh.points()->data();
        const ptrdiff_t n_owned_nodes = mesh.n_owned_nodes();
        const ptrdiff_t n_owned_elements = mesh.n_owned_elements();
        const int nxe = elem_num_nodes(mesh.element_type());
    
        auto sfc_buff = sfem::create_host_buffer<sfc_t>(n_owned_elements);
        auto sfc = sfc_buff->data();
    
        auto idx_buff = sfem::create_host_buffer<element_idx_t>(n_owned_elements);
        auto idx = idx_buff->data();
    
        geom_t box_min[3] = {0, 0, 0}, box_max[3] = {0, 0, 0}, box_extent[3] = {0, 0, 0};
    
        int spatial_dim = mesh.spatial_dimension();
        for (int coord = 0; coord < spatial_dim; coord++) {
            box_min[coord] = points[coord][0];
            box_max[coord] = points[coord][0];
    
            for (ptrdiff_t i = 0; i < n_owned_nodes; i++) {
                const geom_t x = points[coord][i];
                box_min[coord] = MIN(box_min[coord], x);
                box_max[coord] = MAX(box_max[coord], x);
            }
        }
    
        for (int d = 0; d < 3; d++) {
            box_extent[d] = box_max[d] - box_min[d];
        }
    
        int normalize_with_max_range = 1;
    
        if (normalize_with_max_range) {
            geom_t mm = box_extent[0];
    
            for (int coord = 0; coord < spatial_dim; coord++) {
                mm = MAX(mm, box_extent[coord]);
            }
    
            for (int coord = 0; coord < spatial_dim; coord++) {
                box_extent[coord] = mm;
            }
        }
    
        sfc_t urange[3] = {sfc_urange, sfc_urange, sfc_urange};
    
        if (impl_->order_coordinate != -1) {
            for (ptrdiff_t i = 0; i < n_owned_elements; i++) {
                geom_t b[3] = {0, 0, 0};
                const idx_t i0 = elements[0][i];
    
                for (int coord = 0; coord < spatial_dim; coord++) {
                    geom_t x = points[coord][i0];
                    x -= box_min[coord];
                    x /= box_extent[coord];
                    b[coord] = x;
                }
    
                for (int d = 1; d < nxe; d++) {
                    const idx_t ii = elements[d][i];
    
                    for (int coord = 0; coord < spatial_dim; coord++) {
                        geom_t x = points[coord][ii];
                        x -= box_min[coord];
                        x /= box_extent[coord];
                        b[coord] = MIN(b[coord], x);
                    }
                }
    
                // sfc[i] = (sfc_t)(b[2] * (geom_t)urange[2]);
                sfc[i] = b[impl_->order_coordinate] * urange[impl_->order_coordinate];
    
                // printf("%d -> %g %g %g %d\n", (int)i, (double)b[0], (double)b[1], (double)b[2],
                // (int)sfc[i]);
            }
        } else {
            for (ptrdiff_t i = 0; i < n_owned_elements; i++) {
                geom_t b[3] = {0, 0, 0};
                const idx_t i0 = elements[0][i];
    
                for (int coord = 0; coord < spatial_dim; coord++) {
                    geom_t x = points[coord][i0];
                    x -= box_min[coord];
                    x /= box_extent[coord];
                    b[coord] = x;
                }
    
                for (int d = 1; d < nxe; d++) {
                    const idx_t ii = elements[d][i];
    
                    for (int coord = 0; coord < spatial_dim; coord++) {
                        geom_t x = points[coord][ii];
                        x -= box_min[coord];
                        x /= box_extent[coord];
                        b[coord] = MIN(b[coord], x);
                    }
                }
    
                assert(b[0] >= 0);
                assert(b[0] <= 1);
    
                assert(b[1] >= 0);
                assert(b[1] <= 1);
    
                assert(b[2] >= 0);
                assert(b[2] <= 1);
    
                // sfc[i] = (sfc_t)(b[2] * (geom_t)urange[2]);
                sfc[i] = fun_sfc((sfc_t)(b[0] * (double)urange[0]),  //
                                 (sfc_t)(b[1] * (double)urange[1]),  //
                                 (sfc_t)(b[2] * (double)urange[2])   //
                );
    
                // printf("%d -> %g %g %g %d\n", (int)i, (double)b[0], (double)b[1], (double)b[2],
                // (int)sfc[i]);
            }
        }
    
    #ifdef DSFEM_ENABLE_MPI_SORT
    
        if (size > 1) {
            // TODO
    
            // MPI_Sort_bykey (
            //     void * sendkeys_destructive,
            //     void * sendvals_destructive,
            //     const int sendcount,
            //     MPI_Datatype keytype,
            //     MPI_Datatype valtype,
            //     void * recvkeys,
            //     void * recvvals,
            //     const int recvcount,
            //     comm);
        } else
    #endif
        {
            for (ptrdiff_t i = 0; i < n_owned_elements; i++) {
                idx[i] = i;
            }
    
            sort_function(n_owned_elements, sfc, idx);
    
            // idx_buff->print(std::cout);
    
            ptrdiff_t buff_size = MAX(n_owned_elements, n_owned_nodes) * sizeof(idx_t);
            void *buff = malloc(buff_size);
    
            // 1) rearrange elements
            {
                idx_t *elem_buff = (idx_t *)buff;
    
                for (int d = 0; d < nxe; d++) {
                    memcpy(elem_buff, elements[d], n_owned_elements * sizeof(idx_t));
                    for (ptrdiff_t i = 0; i < n_owned_elements; i++) {
                        elements[d][i] = elem_buff[idx[i]];
                    }
                }
    
                const char *SFEM_EXPORT_SFC = 0;
                SFEM_READ_ENV(SFEM_EXPORT_SFC, );
                if (SFEM_EXPORT_SFC) {
                    memcpy(elem_buff, sfc, n_owned_elements * sizeof(sfc_t));
                    for (ptrdiff_t i = 0; i < n_owned_elements; i++) {
                        sfc[i] = elem_buff[idx[i]];
                    }
    
                    array_write(mesh.comm()->get(),
                                SFEM_EXPORT_SFC,
                                SFEM_MPI_SFC_T,
                                sfc,
                                n_owned_elements,
                                n_owned_elements);
                }
            }
    
            // 2) rearrange element_mapping (if the case)
            // TODO
    
            // 3) rearrange nodes
            idx_t *node_buff = (idx_t *)buff;
    
            {
                memset(node_buff, 0, n_owned_nodes * sizeof(idx_t));
    
                idx_t next_node = 1;
                for (ptrdiff_t i = 0; i < n_owned_elements; i++) {
                    for (int d = 0; d < nxe; d++) {
                        idx_t i0 = elements[d][i];
    
                        if (!node_buff[i0]) {
                            node_buff[i0] = next_node++;
                            assert(next_node - 1 <= n_owned_nodes);
                        }
                    }
                }
    
                assert(next_node - 1 == n_owned_nodes);
    
                for (ptrdiff_t i = 0; i < n_owned_nodes; i++) {
                    assert(node_buff[i] > 0);
                    node_buff[i] -= 1;
                }
            }
    
            // Update e2n
            for (int d = 0; d < nxe; d++) {
                for (ptrdiff_t i = 0; i < n_owned_elements; i++) {
                    idx_t i0 = elements[d][i];
                    elements[d][i] = node_buff[i0];
                }
            }
    
            // update coordinates
            geom_t *x_buff = (geom_t*)malloc(n_owned_nodes * sizeof(geom_t));
            for (int d = 0; d < spatial_dim; d++) {
                memcpy(x_buff, points[d], n_owned_nodes * sizeof(geom_t));
    
                for (ptrdiff_t i = 0; i < n_owned_nodes; i++) {
                    points[d][node_buff[i]] = x_buff[i];
                }
            }
    
            // 4) rearrange (or create node_mapping (for relating data))
            // TODO
    
            free(buff);
            free(x_buff);
        }
    
        return SFEM_SUCCESS;
    }
}  // namespace sfem