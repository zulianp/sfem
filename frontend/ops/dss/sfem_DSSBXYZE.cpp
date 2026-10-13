#include "sfem_DSSBXYZE.hpp"

#include "sfem_dss_bxyze.hpp"
#include "smesh_semistructured.hpp"
#include "smesh_sshex8.hpp"

#include <cstdio>

namespace sfem {

int dss_global_to_bxyze(const SharedMesh &mesh,
                        const int         element_block_size,
                        const real_t *const global,
                        DSSBXYZEBuffer *const out) {
    if (!out) {
        std::fprintf(stderr, "dss_global_to_bxyze: output buffer descriptor is null\n");
        return SFEM_FAILURE;
    }

    *out = DSSBXYZEBuffer{};

    if (!mesh || !global || element_block_size <= 0) {
        std::fprintf(stderr, "dss_global_to_bxyze: invalid input\n");
        return SFEM_FAILURE;
    }

    if (mesh->n_blocks() != 1) {
        std::fprintf(stderr,
                     "dss_global_to_bxyze: expected exactly one mesh block, got %zu\n",
                     mesh->n_blocks());
        return SFEM_FAILURE;
    }

    const auto element_type = mesh->element_type(0);
    if (!smesh::is_semistructured_type(element_type) || !smesh::is_hex_ss_family(element_type)) {
        std::fprintf(stderr,
                     "dss_global_to_bxyze: expected semi-structured hex element type, got %s\n",
                     smesh::type_to_string(element_type));
        return SFEM_FAILURE;
    }

    auto block = mesh->block(0);
    if (!block) {
        std::fprintf(stderr, "dss_global_to_bxyze: mesh block 0 is null\n");
        return SFEM_FAILURE;
    }

    const int       level   = smesh::semistructured_level(*mesh);
    const ptrdiff_t ne      = block->n_elements();
    const ptrdiff_t nvalues = sfem_dss_bxyze_n_values(level, ne, element_block_size);

    if (nvalues < 0) {
        std::fprintf(stderr, "dss_global_to_bxyze: invalid BXYZE layout\n");
        return SFEM_FAILURE;
    }

    auto values = create_host_buffer<real_t>((size_t)nvalues);
    if (!values) {
        std::fprintf(stderr, "dss_global_to_bxyze: failed to allocate BXYZE buffer\n");
        return SFEM_FAILURE;
    }

    const int err = sfem_dss_global_to_bxyze(level,
                                             ne,
                                             element_block_size,
                                             block->elements()->data(),
                                             global,
                                             values->data());

    if (err != SFEM_SUCCESS) {
        return err;
    }

    out->level              = level;
    out->element_block_size = element_block_size;
    out->nelements          = ne;
    out->nvalues            = nvalues;
    out->values             = values;

    return SFEM_SUCCESS;
}

}  // namespace sfem
