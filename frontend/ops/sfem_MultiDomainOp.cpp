#include "sfem_MultiDomainOp.hpp"

#include "sfem_FunctionSpace.hpp"
#include "smesh_mesh.hpp"
#include "smesh_sshex8.hpp"
#include "sfem_logger.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_cuda_blas.hpp"
#endif

#include <vector>

namespace sfem {

    MultiDomainOp::MultiDomainOp(const std::shared_ptr<FunctionSpace> &space, const std::vector<std::string> &block_names) {
        if (block_names.empty()) {
            for (auto &block : space->mesh_ptr()->blocks()) {
                domains_[block->name()] =
                        OpDomain{block->element_type(), block, std::make_shared<Parameters>(), nullptr};
            }
        } else {
            for (auto &block_name : block_names) {
                auto block = space->mesh_ptr()->find_block(block_name);
                if (!block) {
                    SFEM_ERROR("Block %s not found", block_name.c_str());
                }
                domains_[block_name] =
                        OpDomain{block->element_type(), block, std::make_shared<Parameters>(), nullptr};
            }
        }
    }

    namespace {
        /// A semi-structured hex type above level 1, which a micro-element operator splits.
        bool splits_into_micro_elements(const smesh::ElemType type) {
            return smesh::is_semistructured_type(type) && smesh::is_hex_ss_family(type) &&
                   smesh::proteus_hex_micro_elements_per_dim(type) > 1;
        }

        /// The eight node rows of micro element (xi, yi, zi) of every level-L macro element, in
        /// HEX8 order, viewing `elements` (on the host or the device, wherever it lives).
        SharedBuffer<idx_t *> micro_element_rows(const int level, const int xi, const int yi, const int zi,
                                                 const SharedBuffer<idx_t *> &elements) {
            const int corners[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
            const ptrdiff_t nelements = elements->extent(1);

            std::vector<idx_t *> rows(elements->extent(0));
#ifdef SFEM_ENABLE_CUDA
            if (elements->mem_space() == MEMORY_SPACE_DEVICE) {
                copy_device_to_host(rows.size() * sizeof(idx_t *), elements->data(), rows.data());
            } else
#endif
            {
                std::copy(elements->data(), elements->data() + rows.size(), rows.begin());
            }

            std::vector<idx_t *> micro(8);
            for (int c = 0; c < 8; ++c) {
                micro[c] = rows[smesh::sshex8_lidx(level, xi + corners[c][0], yi + corners[c][1], zi + corners[c][2])];
            }

#ifdef SFEM_ENABLE_CUDA
            if (elements->mem_space() == MEMORY_SPACE_DEVICE) {
                auto dev = (idx_t **)d_buffer_alloc(8 * sizeof(idx_t *));
                copy_host_to_device(8 * sizeof(idx_t *), micro.data(), dev);
                return std::make_shared<Buffer<idx_t *>>(
                        8, nelements, dev, [keep_alive = elements](int, void **v) { d_buffer_destroy(v); },
                        MEMORY_SPACE_DEVICE);
            }
#endif
            auto host = (idx_t **)malloc(8 * sizeof(idx_t *));
            std::copy(micro.begin(), micro.end(), host);
            return std::make_shared<Buffer<idx_t *>>(
                    8, nelements, host, [keep_alive = elements](int, void **v) { free(v); }, elements->mem_space());
        }
    }  // namespace

    void MultiDomainOp::set_micro_elements(const bool micro) {
        micro_ = micro;
        micro_views_.clear();
    }

    const std::vector<OpDomain> &MultiDomainOp::micro_views(const std::string &name, const OpDomain &domain) {
        auto it = micro_views_.find(name);
        if (it != micro_views_.end()) return it->second;

        const int             level = smesh::proteus_hex_micro_elements_per_dim(domain.element_type);
        std::vector<OpDomain> views;
        views.reserve(level * level * level);
        for (int zi = 0; zi < level; ++zi) {
            for (int yi = 0; yi < level; ++yi) {
                for (int xi = 0; xi < level; ++xi) {
                    auto rows  = micro_element_rows(level, xi, yi, zi, domain.block->elements());
                    auto block = std::make_shared<smesh::Mesh::Block>(domain.block->name(), smesh::HEX8, rows);
                    views.push_back(OpDomain{smesh::HEX8, block, domain.parameters, nullptr});
                }
            }
        }
        return micro_views_[name] = std::move(views);
    }

    int MultiDomainOp::iterate(const std::function<int(const OpDomain &)> &func) {
        for (auto &domain : domains_) {
            if (micro_ && splits_into_micro_elements(domain.second.element_type)) {
                for (const auto &view : micro_views(domain.first, domain.second)) {
                    const int err = func(view);
                    if (err != SFEM_SUCCESS) return err;
                }
                continue;
            }
            int err = func(domain.second);
            if (err != SFEM_SUCCESS) {
                return err;
            }
        }
        return SFEM_SUCCESS;
    }

    int MultiDomainOp::require_supported_types(const char                                      *op_name,
                                               const std::function<bool(smesh::ElemType)> &has_kernel) const {
        for (const auto &kv : domains_) {
            const auto et = micro_ && splits_into_micro_elements(kv.second.element_type) ? smesh::HEX8
                                                                                           : kv.second.element_type;
            if (has_kernel(et)) {
                continue;
            }

            if (smesh::is_wedge_ss_family(et)) {
                SFEM_ERROR("%s: no kernel for WEDGE family on block '%s' (%s); hex-dominant apply is not implemented\n",
                           op_name,
                           kv.first.c_str(),
                           smesh::type_to_string(et));
            } else if (smesh::is_pyramid_ss_family(et)) {
                SFEM_ERROR("%s: no kernel for PYRAMID family on block '%s' (%s); hex-dominant apply is not implemented\n",
                           op_name,
                           kv.first.c_str(),
                           smesh::type_to_string(et));
            } else {
                SFEM_ERROR("%s: no kernel for block '%s' element type %s\n",
                           op_name,
                           kv.first.c_str(),
                           smesh::type_to_string(et));
            }
            return SFEM_FAILURE;
        }

        return SFEM_SUCCESS;
    }

    void MultiDomainOp::override_element_types(const std::vector<smesh::ElemType> &element_types) {
        micro_views_.clear();
        size_t i = 0;
        for (auto &domain : domains_) {
            assert(i < element_types.size());
            domain.second.element_type = element_types[i];
            i++;
        }
    }

    std::shared_ptr<MultiDomainOp> MultiDomainOp::lor(const std::shared_ptr<FunctionSpace> &space,
                                                         const std::vector<std::string>       &block_names) {
        auto ret = std::make_shared<MultiDomainOp>(space, block_names);

        for (auto &domain : ret->domains_) {
            domain.second.element_type = macro_type_variant(domain.second.element_type);
        }

        return ret;
    }

    std::shared_ptr<MultiDomainOp> MultiDomainOp::derefine(const std::shared_ptr<FunctionSpace> &space,
                                                              const std::vector<std::string>       &block_names) {
        auto ret = std::make_shared<MultiDomainOp>(space, block_names);

        for (auto &domain : ret->domains_) {
            domain.second.element_type = macro_base_elem(domain.second.element_type);
        }

        return ret;
    }

    void MultiDomainOp::print_info() {
        for (auto &domain : domains_) {
            printf("Domain %s: %s\n", domain.first.c_str(), domain.second.block->name().c_str());
        }
    }

    void MultiDomainOp::set_value_in_block(const std::string &block_name, const std::string &var_name, const real_t value) {
        auto block = domains_.find(block_name);
        if (block == domains_.end()) {
            SFEM_ERROR("Block %s not found", block_name.c_str());
        }
        block->second.parameters->set_value(var_name, value);
    }

}  // namespace sfem

