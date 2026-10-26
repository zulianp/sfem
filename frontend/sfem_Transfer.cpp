#include "sfem_Transfer.hpp"

#include "sfem_Function.hpp"
#include "sfem_FunctionSpace.hpp"
#include "smesh_exchange.hpp"
#include "smesh_prolongation.hpp"
#include "smesh_restrict.hpp"
#include "smesh_restriction.hpp"
#include "smesh_semistructured.hpp"
#include "smesh_sshex8.hpp"
#include "smesh_sshex8_prolongation.hpp"
#include "smesh_sshex8_restriction.hpp"
#include "smesh_ssquad4.hpp"
#include "smesh_ssquad4_prolongation.hpp"
#include "smesh_sstet4.hpp"
#include "smesh_sstet4_prolongation.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "smesh_device_buffer.hpp"
#include "smesh_sshex8_prolongation.cuh"
#include "smesh_sshex8_restriction.cuh"
#include "smesh_tet4_prolongation_restriction.cuh"
#endif

#include <cassert>

namespace sfem {

    namespace {

        smesh::ElemType transfer_source_family(const smesh::ElemType type) {
            return smesh::is_semistructured_type(type) ? smesh::ss_source_family(type) : type;
        }

        smesh::ElemType ss_block_family(const Mesh &mesh, const size_t b) {
            return smesh::ss_source_family(mesh.element_type(static_cast<smesh::block_idx_t>(b)));
        }

        bool ss_all_hex_family(const Mesh &mesh) {
            for (size_t b = 0; b < mesh.n_blocks(); ++b) {
                if (ss_block_family(mesh, b) != smesh::HEX8) {
                    return false;
                }
            }
            return true;
        }

        void assert_supported_ss_transfer_meshes(const Mesh &from, const Mesh &to) {
            if (from.n_blocks() != to.n_blocks()) {
                SFEM_ERROR("hierarchical transfer: from/to n_blocks mismatch (%zu vs %zu)\n", from.n_blocks(), to.n_blocks());
            }

            bool has_hex  = false;
            bool has_tet  = false;
            bool has_quad = false;
            for (size_t b = 0; b < from.n_blocks(); ++b) {
                const auto bid       = static_cast<smesh::block_idx_t>(b);
                const auto from_type = from.element_type(bid);
                const auto to_type   = to.element_type(bid);
                if (!smesh::is_semistructured_type(to_type)) {
                    SFEM_ERROR("hierarchical transfer: to-mesh block %zu is not semistructured (type %s)\n",
                               b,
                               smesh::type_to_string(to_type));
                }

                const auto from_family = transfer_source_family(from_type);
                const auto to_family   = smesh::ss_source_family(to_type);
                if (from_family != to_family) {
                    SFEM_ERROR("hierarchical transfer: from/to block %zu family mismatch (%s vs %s)\n",
                               b,
                               smesh::type_to_string(from_type),
                               smesh::type_to_string(to_type));
                }

                if (to_family != smesh::HEX8 && to_family != smesh::QUAD4 && to_family != smesh::TET4) {
                    SFEM_ERROR("hierarchical transfer: SS family %s is not implemented\n", smesh::type_to_string(to_family));
                }

                has_hex |= to_family == smesh::HEX8;
                has_tet |= to_family == smesh::TET4;
                has_quad |= to_family == smesh::QUAD4;
            }

            if (has_quad && (has_hex || has_tet)) {
                SFEM_ERROR("hierarchical transfer: mixed SS families with QUAD are not implemented\n");
            }
        }

        ptrdiff_t transfer_nelements(const std::shared_ptr<Mesh> &mesh, const std::shared_ptr<Mesh::Block> &block) {
            if (!block) {
                return 0;
            }
            return mesh_is_mpi(mesh) ? block->n_elements_owned() : block->n_elements();
        }

        std::shared_ptr<Operator<real_t>> wrap_prolongation_coarse_gather(const std::shared_ptr<FunctionSpace>    &from_space,
                                                                          const std::shared_ptr<FunctionSpace>    &to_space,
                                                                          const ExecutionSpace                     es,
                                                                          const std::shared_ptr<Operator<real_t>> &op) {
            auto from_mesh = from_space->mesh_ptr();
            if (!mesh_is_mpi(from_mesh)) {
                return op;
            }
            if (es == EXECUTION_SPACE_DEVICE) {
                SFEM_ERROR("create_hierarchical_prolongation: distributed DEVICE apply is not implemented\n");
                return op;
            }

            // Every owned fine node is incident to an owned macro (Restrict dual). Assignment
            // interpolant must not scatter_add: neighbor writes of the same node can disagree
            // when coarse ghosts are incomplete, and averaging then corrupts the owner value.
            auto      ex = smesh::Exchange::create_nodal(from_mesh, smesh::Exchange::ExchangeScope::GhostsAndAura);
            const int bs = from_space->block_size();
            (void)to_space;
            return make_op<real_t>(
                    op->rows(),
                    op->cols(),
                    [=](const real_t *const from, real_t *const to) {
                        ex->gather(const_cast<real_t *>(from), bs);
                        op->apply(from, to);
                    },
                    es);
        }

    }  // namespace

    SharedBuffer<idx_t> create_edge_idx(CRSGraph &crs_graph) {
        const ptrdiff_t rows        = crs_graph.n_nodes();
        auto            p2_vertices = create_host_buffer<idx_t>(crs_graph.nnz());

        smesh::build_p1_to_p2_edge_map(rows, crs_graph.rowptr()->data(), crs_graph.colidx()->data(), p2_vertices->data());

        return p2_vertices;
    }

    std::shared_ptr<CRSGraph> create_derefined_crs_graph(FunctionSpace &space) {
        auto et        = (smesh::ElemType)space.element_type();
        auto coarse_et = macro_base_elem(et);
        auto crs_graph = space.mesh_ptr()->create_node_to_node_graph(static_cast<smesh::ElemType>(coarse_et));
        return crs_graph;
    }

    bool mesh_is_mpi(const std::shared_ptr<Mesh> &mesh) {
        return mesh && mesh->is_distributed() && mesh->comm() && mesh->comm()->size() > 1;
    }

    std::shared_ptr<Operator<real_t>> create_hierarchical_prolongation(const std::shared_ptr<FunctionSpace> &from_space,
                                                                       const std::shared_ptr<FunctionSpace> &to_space,
                                                                       const ExecutionContext               &context) {
        const ExecutionSpace es      = context.space();
        const bool           to_ss   = to_space->has_semi_structured_mesh();
        const bool           from_ss = from_space->has_semi_structured_mesh();
        if (to_ss) {
            assert_supported_ss_transfer_meshes(*from_space->mesh_ptr(), *to_space->mesh_ptr());
        }
        if (!to_ss && to_space->mesh_ptr()->n_blocks() > 1) {
            SFEM_ERROR("create_hierarchical_prolongation: unstructured multi-block is not implemented\n");
        }

#ifdef SFEM_ENABLE_CUDA
        if (EXECUTION_SPACE_DEVICE == es) {
            if (to_ss) {
                if (!ss_all_hex_family(*to_space->mesh_ptr())) {
                    SFEM_ERROR("create_hierarchical_prolongation: DEVICE SS transfer is implemented for HEX-family only\n");
                }
                if (from_ss) {
                    return wrap_prolongation_coarse_gather(
                            from_space,
                            to_space,
                            es,
                            make_op<real_t>(
                                    to_space->n_dofs(),
                                    from_space->n_dofs(),
                                    [=](const real_t *const from, real_t *const to) {
                                        SFEM_TRACE_SCOPE("cu_sshex8_prolongate");

                                        auto     &from_ssm   = *from_space->mesh_ptr();
                                        auto     &to_ssm     = *to_space->mesh_ptr();
                                        const int to_level   = smesh::semistructured_level(to_ssm);
                                        const int from_level = smesh::semistructured_level(from_ssm);

                                        for (size_t b = 0; b < from_ssm.n_blocks(); ++b) {
                                            auto            from_b = from_ssm.block(b);
                                            auto            to_b   = to_ssm.block(b);
                                            const ptrdiff_t ne     = from_b->n_elements();
                                            if (ne == 0) {
                                                continue;
                                            }
                                            auto from_els = from_b->device_elements_SoA();
                                            auto to_els   = to_b->device_elements_SoA();
                                            if (from_els->mem_space() != smesh::MEMORY_SPACE_DEVICE ||
                                                to_els->mem_space() != smesh::MEMORY_SPACE_DEVICE) {
                                                SFEM_ERROR("Elements are not on the device");
                                            }
                                            smesh::cu_sshex8_prolongate(ne,
                                                                        from_level,
                                                                        1,
                                                                        from_els->data(),
                                                                        to_level,
                                                                        1,
                                                                        to_els->data(),
                                                                        from_space->block_size(),
                                                                        smesh::SMESH_DEFAULT,
                                                                        1,
                                                                        from,
                                                                        smesh::SMESH_DEFAULT,
                                                                        1,
                                                                        to,
                                                                        context.stream());
                                        }
                                    },
                                    context));
                }

                return wrap_prolongation_coarse_gather(from_space,
                                                       to_space,
                                                       es,
                                                       make_op<real_t>(
                                                               to_space->n_dofs(),
                                                               from_space->n_dofs(),
                                                               [=](const real_t *const from, real_t *const to) {
                                                                   SFEM_TRACE_SCOPE("cu_sshex8_hierarchical_prolongation");

                                                                   auto     &ssm   = *to_space->mesh_ptr();
                                                                   const int level = smesh::semistructured_level(ssm);
                                                                   for (size_t b = 0; b < ssm.n_blocks(); ++b) {
                                                                       auto            to_b = ssm.block(b);
                                                                       const ptrdiff_t ne   = to_b->n_elements();
                                                                       if (ne == 0) {
                                                                           continue;
                                                                       }
                                                                       auto elements = to_b->device_elements_SoA();
                                                                       if (elements->mem_space() != smesh::MEMORY_SPACE_DEVICE) {
                                                                           SFEM_ERROR("Elements are not on the device");
                                                                       }
                                                                       smesh::cu_sshex8_hierarchical_prolongation(
                                                                               level,
                                                                               ne,
                                                                               elements->data(),
                                                                               from_space->block_size(),
                                                                               smesh::SMESH_DEFAULT,
                                                                               1,
                                                                               from,
                                                                               smesh::SMESH_DEFAULT,
                                                                               1,
                                                                               to,
                                                                               context.stream());
                                                                   }
                                                               },
                                                               context));
            }

            if (to_space->mesh_ptr()->n_blocks() != 1) {
                SFEM_ERROR("create_hierarchical_prolongation: DEVICE unstructured multi-block is not implemented\n");
            }

            auto elements = to_space->mesh_ptr()->block(0)->device_elements_SoA();

            const ptrdiff_t n_elements = to_space->mesh_ptr()->n_elements(0);
            return wrap_prolongation_coarse_gather(
                    from_space,
                    to_space,
                    es,
                    make_op<real_t>(
                            to_space->n_dofs(),
                            from_space->n_dofs(),
                            [=](const real_t *const from, real_t *const to) {
                                SFEM_TRACE_SCOPE("smesh::cu_macrotet4_to_tet4_prolongation_element_based");

                                smesh::cu_macrotet4_to_tet4_prolongation_element_based(n_elements,
                                                                                       elements->data(),
                                                                                       from_space->block_size(),
                                                                                       smesh::SMESH_DEFAULT,
                                                                                       1,
                                                                                       from,
                                                                                       smesh::SMESH_DEFAULT,
                                                                                       1,
                                                                                       to,
                                                                                       context.stream());
                            },
                            context));

        } else
#endif
        {
            if (to_ss) {
                if (!from_ss) {
                    auto       from_mesh = from_space->mesh_ptr();
                    const bool mpi_from =
                            from_mesh && from_mesh->is_distributed() && from_mesh->comm() && from_mesh->comm()->size() > 1;
                    if (mpi_from && ss_all_hex_family(*to_space->mesh_ptr())) {
                        return wrap_prolongation_coarse_gather(
                                from_space,
                                to_space,
                                es,
                                make_op<real_t>(
                                        to_space->n_dofs(),
                                        from_space->n_dofs(),
                                        [=](const real_t *const from, real_t *const to) {
                                            SFEM_TRACE_SCOPE("sshex8_prolongate_from_hex8");

                                            auto     &from_m   = *from_space->mesh_ptr();
                                            auto     &to_ssm   = *to_space->mesh_ptr();
                                            const int to_level = smesh::semistructured_level(to_ssm);
                                            for (size_t b = 0; b < to_ssm.n_blocks(); ++b) {
                                                auto            from_b  = from_m.block(b);
                                                auto            to_b    = to_ssm.block(b);
                                                const ptrdiff_t ne_from = transfer_nelements(from_space->mesh_ptr(), from_b);
                                                const ptrdiff_t ne_to   = transfer_nelements(to_space->mesh_ptr(), to_b);
                                                const ptrdiff_t ne      = ne_from < ne_to ? ne_from : ne_to;
                                                if (ne == 0) {
                                                    continue;
                                                }
                                                idx_t *from_sshex[8];
                                                smesh::hex8_elements_as_sshex8_level1(from_b->elements()->data(), from_sshex);
                                                smesh::sshex8_prolongate(ne,
                                                                         1,
                                                                         1,
                                                                         from_sshex,
                                                                         to_level,
                                                                         1,
                                                                         to_b->elements()->data(),
                                                                         from_space->block_size(),
                                                                         from,
                                                                         to);
                                            }
                                        },
                                        EXECUTION_SPACE_HOST));
                    }

                    return wrap_prolongation_coarse_gather(
                            from_space,
                            to_space,
                            es,
                            make_op<real_t>(
                                    to_space->n_dofs(),
                                    from_space->n_dofs(),
                                    [=](const real_t *const from, real_t *const to) {
                                        SFEM_TRACE_SCOPE("ss_hierarchical_prolongation");

                                        auto     &ssm   = *to_space->mesh_ptr();
                                        const int level = smesh::semistructured_level(ssm);
                                        for (size_t b = 0; b < ssm.n_blocks(); ++b) {
                                            auto            to_b = ssm.block(b);
                                            const ptrdiff_t ne   = transfer_nelements(to_space->mesh_ptr(), to_b);
                                            if (ne == 0) {
                                                continue;
                                            }
                                            const auto fam = ss_block_family(ssm, b);
                                            if (fam == smesh::HEX8) {
                                                smesh::sshex8_hierarchical_prolongation(
                                                        level, ne, to_b->elements()->data(), from_space->block_size(), from, to);
                                            } else if (fam == smesh::QUAD4) {
                                                smesh::ssquad4_hierarchical_prolongation(
                                                        level, ne, to_b->elements()->data(), from_space->block_size(), from, to);
                                            } else {
                                                smesh::sstet4_hierarchical_prolongation(
                                                        level, ne, to_b->elements()->data(), from_space->block_size(), from, to);
                                            }
                                        }
                                    },
                                    EXECUTION_SPACE_HOST));
                }

                // Homogeneous HEX SS level 1 is converted to unstructured HEX8 in
                // FunctionSpace::derefine, so that case never takes SS-to-SS at from_level == 1.
                // Mixed HEX+TET and homogeneous QUAD/TET keep a level-1 SS coarse space;
                // ss*_prolongate(..., from_level=1, ...) is valid.
                assert(smesh::semistructured_level(*from_space->mesh_ptr()) >= 1);
                assert(!ss_all_hex_family(*from_space->mesh_ptr()) || smesh::semistructured_level(*from_space->mesh_ptr()) > 1);

                return wrap_prolongation_coarse_gather(
                        from_space,
                        to_space,
                        es,
                        make_op<real_t>(
                                to_space->n_dofs(),
                                from_space->n_dofs(),
                                [=](const real_t *const from, real_t *const to) {
                                    SFEM_TRACE_SCOPE("ss_prolongate");

                                    auto     &from_ssm   = *from_space->mesh_ptr();
                                    auto     &to_ssm     = *to_space->mesh_ptr();
                                    const int from_level = smesh::semistructured_level(from_ssm);
                                    const int to_level   = smesh::semistructured_level(to_ssm);
                                    for (size_t b = 0; b < from_ssm.n_blocks(); ++b) {
                                        auto            from_b  = from_ssm.block(b);
                                        auto            to_b    = to_ssm.block(b);
                                        const ptrdiff_t ne_from = transfer_nelements(from_space->mesh_ptr(), from_b);
                                        const ptrdiff_t ne_to   = transfer_nelements(to_space->mesh_ptr(), to_b);
                                        const ptrdiff_t ne      = ne_from < ne_to ? ne_from : ne_to;
                                        if (ne == 0) {
                                            continue;
                                        }
                                        const auto fam = ss_block_family(from_ssm, b);
                                        if (fam == smesh::HEX8) {
                                            smesh::sshex8_prolongate(ne,
                                                                     from_level,
                                                                     1,
                                                                     from_b->elements()->data(),
                                                                     to_level,
                                                                     1,
                                                                     to_b->elements()->data(),
                                                                     from_space->block_size(),
                                                                     from,
                                                                     to);
                                        } else if (fam == smesh::QUAD4) {
                                            smesh::ssquad4_prolongate(ne,
                                                                      from_level,
                                                                      1,
                                                                      from_b->elements()->data(),
                                                                      to_level,
                                                                      1,
                                                                      to_b->elements()->data(),
                                                                      from_space->block_size(),
                                                                      from,
                                                                      to);
                                        } else {
                                            smesh::sstet4_prolongate(ne,
                                                                     from_level,
                                                                     1,
                                                                     from_b->elements()->data(),
                                                                     to_level,
                                                                     1,
                                                                     to_b->elements()->data(),
                                                                     from_space->block_size(),
                                                                     from,
                                                                     to);
                                        }
                                    }
                                },
                                EXECUTION_SPACE_HOST));
            }

            return wrap_prolongation_coarse_gather(
                    from_space,
                    to_space,
                    es,
                    make_op<real_t>(
                            to_space->n_dofs(),
                            from_space->n_dofs(),
                            [=](const real_t *const from, real_t *const to) {
                                SFEM_TRACE_SCOPE("hierarchical_prolongation");

                                smesh::hierarchical_prolongation(
                                        from_space->element_type(),
                                        to_space->element_type(),
                                        transfer_nelements(to_space->mesh_ptr(), to_space->mesh_ptr()->block(0)),
                                        to_space->mesh_ptr()->elements(0)->data(),
                                        from_space->block_size(),
                                        from,
                                        to);
                            },
                            EXECUTION_SPACE_HOST));
        }
    }

    std::shared_ptr<Operator<real_t>> make_op(const std::shared_ptr<smesh::AbstractRestrict<real_t>> &op) {
        return make_op<real_t>(
                op->rows(),
                op->cols(),
                [=](const real_t *const from, real_t *const to) { op->apply(from, to); },
                op->execution_space());
    }

    std::shared_ptr<Operator<real_t>> create_hierarchical_restriction(const std::shared_ptr<FunctionSpace> &from_space,
                                                                      const std::shared_ptr<FunctionSpace> &to_space,
                                                                      const ExecutionSpace                  es) {
        auto op = smesh::Restrict<real_t>::create(from_space->mesh_ptr(), to_space->mesh_ptr(), es, from_space->block_size());
        return make_op<real_t>(
                from_space->n_dofs(),
                to_space->n_dofs(),
                [=](const real_t *const from, real_t *const to) { op->apply(from, to); },
                op->execution_space());
    }

    std::shared_ptr<Operator<real_t>> create_hierarchical_restriction_from_graph(const ptrdiff_t                  n_fine_nodes,
                                                                                 const int                        block_size,
                                                                                 const std::shared_ptr<CRSGraph> &crs_graph,
                                                                                 const SharedBuffer<idx_t>       &edges,
                                                                                 const ExecutionContext          &context) {
        SFEM_UNUSED(context);
        const ptrdiff_t n_coarse_nodes = crs_graph->n_nodes();

        ptrdiff_t rows = n_coarse_nodes * block_size;
        ptrdiff_t cols = n_fine_nodes * block_size;

#ifdef SFEM_ENABLE_CUDA
        if (context.space() == EXECUTION_SPACE_DEVICE) {
            auto d_edges     = smesh::to_device(edges);
            auto d_crs_graph = smesh::to_device(crs_graph);

            return make_op<real_t>(
                    rows,
                    cols,
                    [=](const real_t *const from, real_t *const to) {
                        // FIXME make it generic for all elements!
                        smesh::cu_macrotet4_to_tet4_restriction(n_coarse_nodes,
                                                                d_crs_graph->rowptr()->data(),
                                                                d_crs_graph->colidx()->data(),
                                                                d_edges->data(),
                                                                block_size,
                                                                smesh::SMESH_DEFAULT,
                                                                from,
                                                                smesh::SMESH_DEFAULT,
                                                                to,
                                                                context.stream());
                    },
                    context);
        }
#endif  // SFEM_ENABLE_CUDA

        return make_op<real_t>(
                rows,
                cols,
                [=](const real_t *const from, real_t *const to) {
                    smesh::hierarchical_restriction_with_edge_map(n_coarse_nodes,
                                                                  crs_graph->rowptr()->data(),
                                                                  crs_graph->colidx()->data(),
                                                                  edges->data(),
                                                                  block_size,
                                                                  from,
                                                                  to);
                },
                EXECUTION_SPACE_HOST);
    }

    std::shared_ptr<Operator<real_t>> create_hierarchical_prolongation_from_graph(const std::shared_ptr<Function> &function,
                                                                                  const std::shared_ptr<CRSGraph> &crs_graph,
                                                                                  const SharedBuffer<idx_t>       &edges,
                                                                                  const ExecutionContext          &context) {
        SFEM_UNUSED(context);
        const ptrdiff_t n_fine_nodes   = function->space()->mesh_ptr()->n_nodes();
        int             block_size     = function->space()->block_size();
        const ptrdiff_t n_coarse_nodes = crs_graph->n_nodes();

        ptrdiff_t rows = n_fine_nodes * block_size;
        ptrdiff_t cols = n_coarse_nodes * block_size;

#ifdef SFEM_ENABLE_CUDA
        if (context.space() == EXECUTION_SPACE_DEVICE) {
            auto d_edges     = to_device(edges);
            auto d_crs_graph = to_device(crs_graph);

            return make_op<real_t>(
                    rows,
                    cols,
                    [=](const real_t *const from, real_t *const to) {
                        // FIXME make it generic for all elements!
                        cu_tet4_to_macrotet4_prolongation(n_coarse_nodes,
                                                          d_crs_graph->rowptr()->data(),
                                                          d_crs_graph->colidx()->data(),
                                                          d_edges->data(),
                                                          block_size,
                                                          smesh::SMESH_DEFAULT,
                                                          from,
                                                          smesh::SMESH_DEFAULT,
                                                          to,
                                                          context.stream());

                        function->apply_zero_constraints(to);
                    },
                    context);
        }

#endif  // SFEM_ENABLE_CUDA

        return make_op<real_t>(
                rows,
                cols,
                [=](const real_t *const from, real_t *const to) {
                    smesh::hierarchical_prolongation_with_edge_map(n_coarse_nodes,
                                                                   crs_graph->rowptr()->data(),
                                                                   crs_graph->colidx()->data(),
                                                                   edges->data(),
                                                                   block_size,
                                                                   from,
                                                                   to);

                    function->apply_zero_constraints(to);
                },
                EXECUTION_SPACE_HOST);
    }

}  // namespace sfem
