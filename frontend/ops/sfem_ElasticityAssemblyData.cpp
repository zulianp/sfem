#include "sfem_ElasticityAssemblyData.hpp"

#include "sfem_macros.h"

// FIXME
#include "hex8_neohookean_ogden.h"
#include "tet4_neohookean_ogden.h"
#include "tet4_partial_assembly_neohookean_inline.h"

namespace sfem {

    int ElasticityAssemblyData::compress_partial_assembly(const OpDomain &domain) {
        auto mesh = domain.block;

        if (use_compression) {
            if (!compression_scaling) {
                compression_scaling         = sfem::create_host_buffer<scaling_t>(mesh->n_elements());
                partial_assembly_compressed = sfem::create_host_buffer<compressed_t>(mesh->n_elements() * TET4_S_IKMN_SIZE);
            }

            auto      cs         = compression_scaling->data();
            auto      pa         = partial_assembly_buffer->data();
            auto      pac        = partial_assembly_compressed->data();
            const ptrdiff_t n_elements = mesh->n_elements();
#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n_elements; i++) {
                auto pai = &pa[i * TET4_S_IKMN_SIZE];
                cs[i]    = pai[0];
                for (int v = 1; v < TET4_S_IKMN_SIZE; v++) {
                    cs[i] = MAX(cs[i], fabs(pai[v]));
                }
            }

            real_t max_scaling = 0;

#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n_elements; i++) {
                if (cs[i] > real_t(FP16_MAX)) {
                    max_scaling = MAX(max_scaling, cs[i]);
                    cs[i]       = real_t(cs[i] + 1e-8) / real_t(FP16_MAX);
                } else {
                    cs[i] = 1;
                }
            }

#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n_elements; i++) {
                auto pai  = &pa[i * TET4_S_IKMN_SIZE];
                auto paci = &pac[i * TET4_S_IKMN_SIZE];
                for (int v = 0; v < TET4_S_IKMN_SIZE; v++) {
                    paci[v] = (compressed_t)(pai[v] / cs[i]);

                    assert(cs[i] > 0);
                    // assert(std::isfinite(paci[v]));
                }
            }
        }

        return SFEM_SUCCESS;
    }

}  // namespace sfem
