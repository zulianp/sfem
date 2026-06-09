#include "sfem_test.hpp"

#include "sfem_FunctionSpace.hpp"

#include "integrations/smesh/sccd_smesh_CCD.hpp"

#include "sfem_aliases.hpp"
#include "sfem_context.hpp"
#include "smesh_mesh.hpp"
#include "smesh_sort.hpp"

#include "sfem_API.hpp"
#include "sfem_SelfContact.hpp"
#include "sfem_mask.hpp"

#include "bvh/bvh.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

using namespace sfem;

struct EnvOptions {
    int             demo;
    real_t          margin;
    int             outer_loops;
    int             inner_loops;
    int             nx;
    real_t          ytop;
    real_t          penalty;
    real_t          solver_tol;
    bool            enable_ccd;
    smesh::ElemType element_type;
    bool            enable_augmentation;
    real_t          toi_scale;
    real_t          search_radius;
    real_t          damping;
    int             output_frequency;

    static EnvOptions read() {
        return {
                smesh::Env::read("SFEM_DEMO", int(1)),
                smesh::Env::read("SFEM_MARGIN", real_t(0)),
                smesh::Env::read("SFEM_OUTER_LOOPS", int(1)),
                smesh::Env::read("SFEM_INNER_LOOPS", int(1000)),
                smesh::Env::read("SFEM_NX", int(10)),
                smesh::Env::read("SFEM_YTOP", real_t(-0.4)),
                smesh::Env::read("SFEM_PENALTY", real_t(10)),
                smesh::Env::read("SFEM_SOLVER_TOL", real_t(1e-6)),
                smesh::Env::read("SFEM_ENABLE_CCD", false),
                smesh::Env::read("SFEM_ELEM_TYPE", smesh::TET4),
                smesh::Env::read("SFEM_ENABLE_AUGMENTATION", false),
                smesh::Env::read("SFEM_TOI_SCALE", real_t(1)),
                smesh::Env::read("SFEM_SEARCH_RADIUS", real_t(0.1)),
                smesh::Env::read("SFEM_DAMPING", real_t(1)),
                smesh::Env::read("SFEM_OUTPUT_FREQUENCY", int(10)),
        };
    }

    void print(std::ostream& os) const {
        os << "EnvOptions:" << std::endl;
        os << "  demo: " << demo << std::endl;
        os << "  margin: " << margin << std::endl;
        os << "  outer_loops: " << outer_loops << std::endl;
        os << "  inner_loops: " << inner_loops << std::endl;
        os << "  nx: " << nx << std::endl;
        os << "  ytop: " << ytop << std::endl;
        os << "  penalty: " << penalty << std::endl;
        os << "  solver_tol: " << solver_tol << std::endl;
        os << "  enable_ccd: " << enable_ccd << std::endl;
        os << "  element_type: " << element_type << std::endl;
        os << "  enable_augmentation: " << enable_augmentation << std::endl;
        os << "  toi_scale: " << toi_scale << std::endl;
        os << "  search_radius: " << search_radius << std::endl;
    }
};

std::shared_ptr<Function> create_function(const EnvOptions& opts, const ExecutionSpace es, const EnvOptions& env) {
    if (env.demo) {
        const ptrdiff_t nx = opts.nx;

        geom_t y_bottom = 0.9;
        auto   mesh1    = smesh::Mesh::create_cube(
                Communicator::self(), env.element_type, nx, std::max<ptrdiff_t>(1, nx / 5), nx, 0, y_bottom, 0, 1, 1, 1);
        auto mesh2 = smesh::Mesh::create_cube(Communicator::self(),
                                              env.element_type,
                                              std::max<ptrdiff_t>(1, nx / 2),
                                              nx,
                                              std::max<ptrdiff_t>(1, nx / 2),
                                              0.25,
                                              1.1,
                                              0.25,
                                              0.75,
                                              1.9,
                                              0.75);

        auto mesh = smesh::concatenate(mesh1, mesh2);

        printf("Bulk: #nodes %zu #elements %zu\n", mesh->n_nodes(), mesh->n_elements());

        mesh->write(smesh::Path("contact_mesh"));

        auto top_ss =
                sfem::Sideset::create_from_selector(mesh, [=](const geom_t /*x*/, const geom_t y, const geom_t /*z*/) -> bool {
                    return y > (1.9 - 1e-4) && y < (1.9 + 1e-4);
                });

        auto left_ss = sfem::Sideset::create_from_selector(mesh, [=](const geom_t x, const geom_t /*y*/, const geom_t z) -> bool {
            return x > (-1e-4) && x < (1e-4) && z > 0.45 && z < 0.55;
        });

        auto right_ss =
                sfem::Sideset::create_from_selector(mesh, [=](const geom_t x, const geom_t /*y*/, const geom_t z) -> bool {
                    return x > (1 - 1e-4) && x < (1 + 1e-4) && z > 0.45 && z < 0.55;
                });

        const int dim   = mesh->spatial_dimension();
        auto      space = FunctionSpace::create(mesh, dim);

        auto op = create_op(space, "LinearElasticity", es);
        op->initialize();

        auto f = Function::create(space);
        f->add_operator(op);

        auto top_ns   = smesh::create_nodeset_from_sidesets(mesh, top_ss);
        auto left_ns  = smesh::create_nodeset_from_sidesets(mesh, left_ss);
        auto right_ns = smesh::create_nodeset_from_sidesets(mesh, right_ss);

        assert(top_ns != nullptr);
        assert(top_ns->size() > 0);

        DirichletConditions::Condition xtop{.sidesets = top_ss, .nodeset = top_ns, .value = 0, .component = 0};
        DirichletConditions::Condition ytop{.sidesets = top_ss, .nodeset = top_ns, .value = opts.ytop, .component = 1};
        DirichletConditions::Condition ztop{.sidesets = top_ss, .nodeset = top_ns, .value = 0, .component = 2};

        DirichletConditions::Condition xleft{.sidesets = left_ss, .nodeset = left_ns, .value = 0, .component = 0};
        DirichletConditions::Condition yleft{.sidesets = left_ss, .nodeset = left_ns, .value = 0, .component = 1};
        DirichletConditions::Condition zleft{.sidesets = left_ss, .nodeset = left_ns, .value = 0, .component = 2};

        DirichletConditions::Condition xright{.sidesets = right_ss, .nodeset = right_ns, .value = 0, .component = 0};
        DirichletConditions::Condition yright{.sidesets = right_ss, .nodeset = right_ns, .value = 0, .component = 1};
        DirichletConditions::Condition zright{.sidesets = right_ss, .nodeset = right_ns, .value = 0, .component = 2};

        auto conds =
                sfem::create_dirichlet_conditions(space, {xtop, ytop, ztop, xleft, yleft, zleft, xright, yright, zright}, es);
        f->add_constraint(conds);

        return f;
    } else {
        smesh::Path mesh_path{"./mesh"};
        smesh::Path dirichlet_path{"./case.yaml"};
        auto        mesh = smesh::Mesh::create_from_file(Communicator::self(), mesh_path);

        auto space = sfem::FunctionSpace::create(mesh, mesh->spatial_dimension());
        auto f     = sfem::Function::create(space);

        auto op = create_op(space, "LinearElasticity", es);
        op->initialize();
        f->add_operator(op);

        auto dirichlet_conditions = sfem::DirichletConditions::create_from_file(space, dirichlet_path);
        f->add_constraint(dirichlet_conditions);

        return f;
    }
}

struct ContactData {
    std::shared_ptr<smesh::Mesh>                     surface;
    std::shared_ptr<smesh::CRSGraph<count_t, idx_t>> graph;
    smesh::SharedBuffer<real_t>&                     values;
    smesh::SharedBuffer<real_t>&                     mass_vector;
    smesh::SharedBuffer<real_t*>&                    normals;
    smesh::SharedBuffer<real_t>&                     distances;
    smesh::SharedBuffer<real_t>&                     frozen_displacement;
    SharedBuffer<mask_t>                             constraints_mask;
    smesh::SharedBuffer<real_t>                      agumentation;
};

void remove_surface_elements_connected_to_constrained_nodes(const std::shared_ptr<smesh::Mesh>& surface,
                                                            const smesh::SharedBuffer<mask_t>&  constraints_mask,
                                                            const int                           block_size) {
    auto node_mapping = surface->node_mapping();
    assert(node_mapping);
    assert(constraints_mask);

    const ptrdiff_t            n_nodes           = surface->n_nodes();
    const idx_t* const         node_mapping_data = node_mapping->data();
    const mask_t* const        mask_data         = constraints_mask->data();
    std::vector<unsigned char> constrained_node(n_nodes);

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < n_nodes; ++i) {
        const ptrdiff_t dof = node_mapping_data[i] * block_size;
        bool            constrained{false};
        for (int d = 0; d < block_size; ++d) {
            constrained |= mask_get(dof + d, mask_data);
        }

        constrained_node[i] = constrained;
    }

    for (size_t b = 0; b < surface->n_blocks(); ++b) {
        auto                       block         = surface->block(b);
        const int                  nxe           = block->n_nodes_per_element();
        const ptrdiff_t            n_elements    = block->n_elements();
        auto                       elements      = block->elements();
        auto                       elements_data = elements->data();
        ptrdiff_t                  n_kept        = 0;
        std::vector<unsigned char> keep(n_elements);

#pragma omp parallel for reduction(+ : n_kept)
        for (ptrdiff_t e = 0; e < n_elements; ++e) {
            bool remove{false};
            for (int v = 0; v < nxe; ++v) {
                remove |= constrained_node[elements_data[v][e]];
            }

            keep[e] = !remove;
            n_kept += keep[e];
        }

        if (n_kept == n_elements) {
            continue;
        }

        auto filtered_elements      = smesh::create_host_buffer<idx_t>(nxe, n_kept);
        auto filtered_elements_data = filtered_elements->data();

        ptrdiff_t out = 0;
        for (ptrdiff_t e = 0; e < n_elements; ++e) {
            if (!keep[e]) {
                continue;
            }

            for (int v = 0; v < nxe; ++v) {
                filtered_elements_data[v][out] = elements_data[v][e];
            }

            ++out;
        }

        block->set_elements(filtered_elements);
    }

    std::vector<unsigned char> used_node(n_nodes);
    std::vector<idx_t>         old_to_new(n_nodes);
    ptrdiff_t                  n_used_nodes = 0;

    for (size_t b = 0; b < surface->n_blocks(); ++b) {
        auto            block         = surface->block(b);
        const int       nxe           = block->n_nodes_per_element();
        const ptrdiff_t n_elements    = block->n_elements();
        auto            elements_data = block->elements()->data();

        for (ptrdiff_t e = 0; e < n_elements; ++e) {
            for (int v = 0; v < nxe; ++v) {
                const idx_t node = elements_data[v][e];
                if (!used_node[node]) {
                    used_node[node]  = true;
                    old_to_new[node] = n_used_nodes++;
                }
            }
        }
    }

    if (n_used_nodes == n_nodes) {
        return;
    }

    const int dim              = surface->spatial_dimension();
    auto      points           = surface->points();
    auto      points_data      = points->data();
    auto      compact_points   = smesh::create_host_buffer<geom_t>(dim, n_used_nodes);
    auto      compact_mapping  = smesh::create_host_buffer<idx_t>(n_used_nodes);
    auto      compact_p_data   = compact_points->data();
    auto      compact_map_data = compact_mapping->data();

    for (ptrdiff_t i = 0; i < n_nodes; ++i) {
        if (!used_node[i]) {
            continue;
        }

        const idx_t new_node       = old_to_new[i];
        compact_map_data[new_node] = node_mapping_data[i];
        for (int d = 0; d < dim; ++d) {
            compact_p_data[d][new_node] = points_data[d][i];
        }
    }

    for (size_t b = 0; b < surface->n_blocks(); ++b) {
        auto            block         = surface->block(b);
        const int       nxe           = block->n_nodes_per_element();
        const ptrdiff_t n_elements    = block->n_elements();
        auto            elements_data = block->elements()->data();

#pragma omp parallel for
        for (ptrdiff_t e = 0; e < n_elements; ++e) {
            for (int v = 0; v < nxe; ++v) {
                elements_data[v][e] = old_to_new[elements_data[v][e]];
            }
        }
    }

    surface->set_points(compact_points);
    surface->set_node_mapping(compact_mapping);
}

void compute_penetration(ContactData& cd, const real_t* const disp, real_t* const penetration) {
    SFEM_TRACE_SCOPE("compute_penetration");
    const int dim    = cd.surface->spatial_dimension();
    auto      graph  = cd.graph;
    auto      values = cd.values;
    auto      rowptr = graph->rowptr()->data();
    auto      colidx = graph->colidx()->data();
    auto      vals   = values->data();
    ptrdiff_t n      = graph->rowptr()->size() - 1;

    auto d       = cd.distances->data();
    auto normals = cd.normals->data();
    auto disp0   = cd.frozen_displacement->data();
    auto nm      = cd.surface->node_mapping()->data();

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < n; i++) {
        const count_t lenrow = rowptr[i + 1] - rowptr[i];
        if (lenrow == 0) {
            penetration[i] = 0;
            continue;
        }

        const idx_t* const  row     = &colidx[rowptr[i]];
        const real_t* const weights = &vals[rowptr[i]];
        const ptrdiff_t     dof1    = nm[i] * dim;

        real_t normal_diff = 0;
        for (int d = 0; d < dim; d++) {
            real_t u2 = 0;
            for (count_t j = 0; j < lenrow; j++) {
                const ptrdiff_t dof2 = nm[row[j]] * dim + d;
                u2 += weights[j] * (disp[dof2] - disp0[dof2]);
            }

            normal_diff += normals[d][i] * (disp[dof1 + d] - disp0[dof1 + d] - u2);
        }

        penetration[i] = std::max(real_t(0), normal_diff - d[i]);
    }
}

void compute_macaulay_term_from_penetration(ContactData&        cd,
                                            const real_t        penalty,
                                            const real_t* const penetration,
                                            real_t* const       macaulay) {
    SFEM_TRACE_SCOPE("compute_macaulay_term_from_penetration");
    auto            rowptr = cd.graph->rowptr()->data();
    auto            aug    = cd.agumentation->data();
    const ptrdiff_t n      = cd.graph->rowptr()->size() - 1;

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < n; i++) {
        if (rowptr[i + 1] == rowptr[i]) {
            macaulay[i] = 0;
            continue;
        }

        macaulay[i] = std::max(penetration[i] + aug[i] / penalty, real_t(0));
    }
}

void displace_points(const std::shared_ptr<smesh::Mesh>&     surface,
                     const std::shared_ptr<Buffer<real_t>>&  displacement,
                     const std::shared_ptr<Buffer<real_t*>>& inout) {
    auto p = inout->data();
    auto u = displacement->data();
    auto m = surface->node_mapping()->data();

    const ptrdiff_t n   = surface->node_mapping()->size();
    const int       dim = surface->spatial_dimension();

    for (int d = 0; d < dim; d++) {
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n; i++) {
            p[d][i] += u[m[i] * dim + d];
        }
    }
}

void compute_macaulay_term(ContactData& cd, const real_t penalty, const real_t* const disp, real_t* const macaulay) {
    SFEM_TRACE_SCOPE("compute_macaulay_term");
    const int dim    = cd.surface->spatial_dimension();
    auto      graph  = cd.graph;
    auto      values = cd.values;
    auto      rowptr = graph->rowptr()->data();
    auto      colidx = graph->colidx()->data();
    auto      vals   = values->data();
    ptrdiff_t n      = graph->rowptr()->size() - 1;

    auto d       = cd.distances->data();
    auto aug     = cd.agumentation->data();
    auto normals = cd.normals->data();
    auto mass    = cd.mass_vector->data();
    auto disp0   = cd.frozen_displacement->data();

    auto nm = cd.surface->node_mapping()->data();

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < n; i++) {
        auto lenrow = rowptr[i + 1] - rowptr[i];
        if (lenrow == 0) {
            macaulay[i] = 0;
            continue;
        }

        auto row = &colidx[rowptr[i]];

        auto weights = &vals[rowptr[i]];

        real_t u1[3] = {0, 0, 0};
        for (int d = 0; d < dim; d++) {
            const ptrdiff_t dof = nm[i] * dim + d;
            u1[d]               = disp[dof] - disp0[dof];
        }

        real_t u2[3] = {0, 0, 0};
        for (int d = 0; d < dim; d++) {
            for (count_t j = 0; j < lenrow; j++) {
                const ptrdiff_t dof = nm[row[j]] * dim + d;
                u2[d] += weights[j] * (disp[dof] - disp0[dof]);
            }
        }

        const real_t g         = d[i];
        real_t       normal[3] = {0, 0, 0};
        for (int d = 0; d < dim; d++) {
            normal[d] = normals[d][i];
        }

        real_t diff[3] = {0, 0, 0};
        for (int d = 0; d < dim; d++) {
            diff[d] = u1[d] - u2[d];
        }

        real_t normal_diff = 0;
        for (int d = 0; d < dim; d++) {
            normal_diff += normal[d] * diff[d];
        }

        real_t pen       = normal_diff - g;
        real_t lagr_mult = aug[i];
        macaulay[i]      = std::max(pen + lagr_mult / penalty, real_t(0));
    }
}

void assemble_contact_gradient(ContactData& cd, const real_t penalty, const real_t* const macaulay, real_t* const grad) {
    SFEM_TRACE_SCOPE("assemble_contact_gradient");
    const int dim    = cd.surface->spatial_dimension();
    auto      graph  = cd.graph;
    auto      values = cd.values;
    auto      rowptr = graph->rowptr()->data();
    auto      colidx = graph->colidx()->data();
    auto      vals   = values->data();
    ptrdiff_t n      = graph->rowptr()->size() - 1;

    auto d       = cd.distances->data();
    auto aug     = cd.agumentation->data();
    auto normals = cd.normals->data();
    auto mass    = cd.mass_vector->data();

    auto nm = cd.surface->node_mapping()->data();

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < n; i++) {
        if (macaulay[i] == 0) continue;

        auto lenrow = rowptr[i + 1] - rowptr[i];
        if (lenrow == 0) continue;

        auto row     = &colidx[rowptr[i]];
        auto weights = &vals[rowptr[i]];

        real_t normal[3] = {0, 0, 0};
        for (int d = 0; d < dim; d++) {
            normal[d] = normals[d][i];
        }

        real_t force[3] = {0, 0, 0};
        for (int d = 0; d < dim; d++) {
            // Point-force we scale it by the mass-density at the contact point
            force[d] = mass[i] * penalty * macaulay[i] * normal[d];
        }

        for (int d = 0; d < dim; d++) {
#pragma omp atomic update
            grad[nm[i] * dim + d] += force[d];
        }

        for (int d = 0; d < dim; d++) {
            for (count_t j = 0; j < lenrow; j++) {
#pragma omp atomic update
                grad[nm[row[j]] * dim + d] -= force[d] * weights[j];
            }
        }
    }
}

void assemble_contact_hessian_diag(ContactData&                                     cd,
                                   const real_t                                     penalty,
                                   const real_t* const                              macaulay,
                                   const ptrdiff_t                                  diag_stride,
                                   real_t* const SFEM_RESTRICT* const SFEM_RESTRICT diag_values) {
    SFEM_TRACE_SCOPE("assemble_contact_hessian_diag");
    const int dim    = cd.surface->spatial_dimension();
    auto      graph  = cd.graph;
    auto      values = cd.values;
    auto      rowptr = graph->rowptr()->data();
    auto      colidx = graph->colidx()->data();
    auto      vals   = values->data();
    ptrdiff_t n      = graph->rowptr()->size() - 1;

    auto d       = cd.distances->data();
    auto aug     = cd.agumentation->data();
    auto normals = cd.normals->data();
    auto mass    = cd.mass_vector->data();

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < n; i++) {
        if (macaulay[i] == 0) continue;

        auto lenrow = rowptr[i + 1] - rowptr[i];
        if (lenrow == 0) continue;

        auto row     = &colidx[rowptr[i]];
        auto weights = &vals[rowptr[i]];

        real_t normal[3] = {0, 0, 0};
        for (int d = 0; d < dim; d++) {
            normal[d] = normals[d][i];
        }

        real_t nnT[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (int d1 = 0; d1 < dim; d1++) {
            for (int d2 = 0; d2 < dim; d2++) {
                nnT[d1 * dim + d2] = mass[i] * penalty * normal[d1] * normal[d2];
            }
        }

        // Assemble H11
        for (int d = 0; d < dim * dim; d++) {
#pragma omp atomic update
            diag_values[d][i * diag_stride] += nnT[d];
        }

        // Assemble H22
        for (int d = 0; d < dim * dim; d++) {
            for (count_t j = 0; j < lenrow; j++) {
#pragma omp atomic update
                diag_values[d][row[j] * diag_stride] += weights[j] * weights[j] * nnT[d];
            }
        }
    }
}

// Gather the diagonal values from the symmetric representation elast_diag_values (uses node mapping to read), add them to the
// diag_values, mask (uses node mapping to read) them for the constraint rows with an identiity row
void gather_combine_hessian_diag(ContactData&                                     cd,
                                 const mask_t* const                              is_constrained,
                                 const real_t* const                              elast_diag_values,
                                 const ptrdiff_t                                  diag_stride,
                                 real_t* const SFEM_RESTRICT* const SFEM_RESTRICT diag_values) {
    SFEM_TRACE_SCOPE("gather_combine_hessian_diag");
    const int       dim = cd.surface->spatial_dimension();
    const ptrdiff_t n   = cd.surface->node_mapping()->size();
    const idx_t*    nm  = cd.surface->node_mapping()->data();

    if (dim == 3) {
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n; ++i) {
            const ptrdiff_t global_node = nm[i];
            const real_t*   ed          = &elast_diag_values[global_node * 6];
            const ptrdiff_t local_node  = i * diag_stride;

            diag_values[0][local_node] += ed[0];
            diag_values[1][local_node] += ed[1];
            diag_values[2][local_node] += ed[2];
            diag_values[3][local_node] += ed[1];
            diag_values[4][local_node] += ed[3];
            diag_values[5][local_node] += ed[4];
            diag_values[6][local_node] += ed[2];
            diag_values[7][local_node] += ed[4];
            diag_values[8][local_node] += ed[5];

            const ptrdiff_t dof = global_node * 3;
            if (mask_get(dof, is_constrained)) {
                diag_values[0][local_node] = 1;
                diag_values[1][local_node] = 0;
                diag_values[2][local_node] = 0;
            }

            if (mask_get(dof + 1, is_constrained)) {
                diag_values[3][local_node] = 0;
                diag_values[4][local_node] = 1;
                diag_values[5][local_node] = 0;
            }

            if (mask_get(dof + 2, is_constrained)) {
                diag_values[6][local_node] = 0;
                diag_values[7][local_node] = 0;
                diag_values[8][local_node] = 1;
            }
        }
    } else {
        const int sym_block_size = (dim * (dim + 1)) / 2;

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n; ++i) {
            const ptrdiff_t global_node = nm[i];
            const real_t*   ed          = &elast_diag_values[global_node * sym_block_size];
            const ptrdiff_t local_node  = i * diag_stride;

            int s = 0;
            for (int d1 = 0; d1 < dim; ++d1) {
                diag_values[d1 * dim + d1][local_node] += ed[s++];
                for (int d2 = d1 + 1; d2 < dim; ++d2) {
                    const real_t e = ed[s++];
                    diag_values[d1 * dim + d2][local_node] += e;
                    diag_values[d2 * dim + d1][local_node] += e;
                }
            }

            const ptrdiff_t dof = global_node * dim;
            for (int d1 = 0; d1 < dim; ++d1) {
                if (!mask_get(dof + d1, is_constrained)) continue;
                for (int d2 = 0; d2 < dim; ++d2) {
                    diag_values[d1 * dim + d2][local_node] = (d1 == d2) ? 1 : 0;
                }
            }
        }
    }
}

void nljacobi(ContactData&                                 cd,
              const std::shared_ptr<sfem::Function>&       f,
              const std::shared_ptr<sfem::Buffer<real_t>>& x,
              const real_t                                 penalty,
              const int                                    n_loops,
              const real_t                                 solver_tol,
              const bool                                   enable_augmentation) {
    SFEM_TRACE_SCOPE("nljacobi");

    auto      space = f->space();
    auto      mesh  = space->mesh_ptr();
    const int dim   = mesh->spatial_dimension();
    assert(dim == 3);

    const ptrdiff_t ndofs          = space->n_dofs();
    const ptrdiff_t n_nodes        = ndofs / dim;
    const ptrdiff_t n_contact      = cd.surface->node_mapping()->size();
    const int       sym_block_size = (dim * (dim + 1)) / 2;
    const real_t    omega          = 1. / 3;
    auto            es             = f->execution_space();
    auto            blas           = sfem::blas<real_t>(es);

    auto material_grad     = sfem::create_buffer<real_t>(ndofs, es);
    auto elast_diag_values = sfem::create_buffer<real_t>(n_nodes * sym_block_size, es);
    auto contact_node_mask = sfem::create_buffer<mask_t>(mask_count(n_nodes), es);
    auto contact_grad      = sfem::create_buffer<real_t>(ndofs, es);
    auto penetration       = sfem::create_buffer<real_t>(n_contact, es);
    auto macaulay          = sfem::create_buffer<real_t>(n_contact, es);
    auto diag_values       = sfem::create_buffer<real_t>(dim * dim, n_contact, es);
    auto constraints_mask  = cd.constraints_mask;
    assert(constraints_mask);

    const idx_t* const nm = cd.surface->node_mapping()->data();
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < contact_node_mask->size(); ++i) {
        contact_node_mask->data()[i] = 0;
    }

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < n_contact; ++i) {
        mask_set(nm[i], contact_node_mask->data());
    }

    const mask_t* const mask         = constraints_mask->data();
    const mask_t* const contact_mask = contact_node_mask->data();
    real_t* const       xd           = x->data();

    // If the material is nonlinear should be inside the loop
    blas->values(elast_diag_values->size(), 0, elast_diag_values->data());
    f->hessian_block_diag_sym(x->data(), elast_diag_values->data());

    ptrdiff_t each = std::min(n_loops, 1);
    for (int loop = 0; loop < n_loops; ++loop) {
        blas->values(ndofs, 0, material_grad->data());

        f->gradient(x->data(), material_grad->data());

        const real_t* const eg = material_grad->data();
        const real_t* const ed = elast_diag_values->data();

        blas->values(ndofs, 0, contact_grad->data());
        for (int d = 0; d < dim * dim; ++d) {
            blas->values(n_contact, 0, diag_values->data()[d]);
        }

        compute_macaulay_term(cd, penalty, x->data(), macaulay->data());
        assemble_contact_gradient(cd, penalty, macaulay->data(), contact_grad->data());
        assemble_contact_hessian_diag(cd, penalty, macaulay->data(), 1, diag_values->data());
        gather_combine_hessian_diag(cd, constraints_mask->data(), elast_diag_values->data(), 1, diag_values->data());

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n_nodes; ++i) {
            if (mask_get(i, contact_mask)) continue;

            real_t a0 = ed[i * 6 + 0], a1 = ed[i * 6 + 1], a2 = ed[i * 6 + 2];
            real_t a3 = ed[i * 6 + 1], a4 = ed[i * 6 + 3], a5 = ed[i * 6 + 4];
            real_t a6 = ed[i * 6 + 2], a7 = ed[i * 6 + 4], a8 = ed[i * 6 + 5];

            const ptrdiff_t dof = i * 3;
            if (mask_get(dof, mask)) {
                a0 = 1;
                a1 = 0;
                a2 = 0;
            }

            if (mask_get(dof + 1, mask)) {
                a3 = 0;
                a4 = 1;
                a5 = 0;
            }

            if (mask_get(dof + 2, mask)) {
                a6 = 0;
                a7 = 0;
                a8 = 1;
            }

            const real_t g0 = eg[dof + 0];
            const real_t g1 = eg[dof + 1];
            const real_t g2 = eg[dof + 2];

            const real_t x0  = a4 * a8;
            const real_t x1  = a5 * a7;
            const real_t x2  = a1 * a5;
            const real_t x3  = a1 * a8;
            const real_t x4  = a2 * a4;
            const real_t det = a0 * x0 - a0 * x1 + a2 * a3 * a7 - a3 * x3 + a6 * x2 - a6 * x4;

            if (!std::isfinite(det) || det == 0) {
                if (std::isfinite(a0) && a0 != 0) xd[dof + 0] -= omega * g0 / a0;
                if (std::isfinite(a4) && a4 != 0) xd[dof + 1] -= omega * g1 / a4;
                if (std::isfinite(a8) && a8 != 0) xd[dof + 2] -= omega * g2 / a8;
                continue;
            }

            const real_t inv_det = 1 / det;

            const real_t i0 = inv_det * (x0 - x1);
            const real_t i1 = inv_det * (a2 * a7 - x3);
            const real_t i2 = inv_det * (x2 - x4);
            const real_t i3 = inv_det * (-a3 * a8 + a5 * a6);
            const real_t i4 = inv_det * (a0 * a8 - a2 * a6);
            const real_t i5 = inv_det * (-a0 * a5 + a2 * a3);
            const real_t i6 = inv_det * (a3 * a7 - a4 * a6);
            const real_t i7 = inv_det * (-a0 * a7 + a1 * a6);
            const real_t i8 = inv_det * (a0 * a4 - a1 * a3);

            xd[dof + 0] -= omega * (i0 * g0 + i1 * g1 + i2 * g2);
            xd[dof + 1] -= omega * (i3 * g0 + i4 * g1 + i5 * g2);
            xd[dof + 2] -= omega * (i6 * g0 + i7 * g1 + i8 * g2);
        }

        const real_t* const* const dv = diag_values->data();
        const real_t* const        cg = contact_grad->data();

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n_contact; ++i) {
            const ptrdiff_t local_node  = i;
            const ptrdiff_t global_node = nm[i];
            const ptrdiff_t dof         = global_node * 3;

            const real_t g0 = eg[dof + 0] + (mask_get(dof + 0, mask) ? 0 : cg[dof + 0]);
            const real_t g1 = eg[dof + 1] + (mask_get(dof + 1, mask) ? 0 : cg[dof + 1]);
            const real_t g2 = eg[dof + 2] + (mask_get(dof + 2, mask) ? 0 : cg[dof + 2]);

            if (g0 == 0 && g1 == 0 && g2 == 0) continue;

            const real_t a0 = dv[0][local_node], a1 = dv[1][local_node], a2 = dv[2][local_node];
            const real_t a3 = dv[3][local_node], a4 = dv[4][local_node], a5 = dv[5][local_node];
            const real_t a6 = dv[6][local_node], a7 = dv[7][local_node], a8 = dv[8][local_node];

            const real_t x0  = a4 * a8;
            const real_t x1  = a5 * a7;
            const real_t x2  = a1 * a5;
            const real_t x3  = a1 * a8;
            const real_t x4  = a2 * a4;
            const real_t det = a0 * x0 - a0 * x1 + a2 * a3 * a7 - a3 * x3 + a6 * x2 - a6 * x4;

            if (!std::isfinite(det) || det == 0) {
                if (std::isfinite(a0) && a0 != 0) xd[dof + 0] -= omega * g0 / a0;
                if (std::isfinite(a4) && a4 != 0) xd[dof + 1] -= omega * g1 / a4;
                if (std::isfinite(a8) && a8 != 0) xd[dof + 2] -= omega * g2 / a8;
                continue;
            }

            const real_t inv_det = 1 / det;

            const real_t i0 = inv_det * (x0 - x1);
            const real_t i1 = inv_det * (a2 * a7 - x3);
            const real_t i2 = inv_det * (x2 - x4);
            const real_t i3 = inv_det * (-a3 * a8 + a5 * a6);
            const real_t i4 = inv_det * (a0 * a8 - a2 * a6);
            const real_t i5 = inv_det * (-a0 * a5 + a2 * a3);
            const real_t i6 = inv_det * (a3 * a7 - a4 * a6);
            const real_t i7 = inv_det * (-a0 * a7 + a1 * a6);
            const real_t i8 = inv_det * (a0 * a4 - a1 * a3);

            xd[dof + 0] -= omega * (i0 * g0 + i1 * g1 + i2 * g2);
            xd[dof + 1] -= omega * (i3 * g0 + i4 * g1 + i5 * g2);
            xd[dof + 2] -= omega * (i6 * g0 + i7 * g1 + i8 * g2);
        }

        real_t* const aug = cd.agumentation->data();
        if ((loop + 1) % each == 0 && enable_augmentation) {
            compute_macaulay_term(cd, penalty, x->data(), macaulay->data());

            const real_t* const m = macaulay->data();
#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n_contact; ++i) {
                aug[i] = penalty * m[i];
            }
        }

        compute_penetration(cd, x->data(), penetration->data());

        real_t              penetration_norm2 = 0;
        real_t              lagr_mult_norm2   = 0;
        const real_t* const p                 = penetration->data();
#pragma omp parallel for reduction(+ : penetration_norm2, lagr_mult_norm2)
        for (ptrdiff_t i = 0; i < n_contact; ++i) {
            penetration_norm2 += p[i] * p[i];
            lagr_mult_norm2 += aug[i] * aug[i];
        }

        real_t              full_grad_norm2 = 0;
        const real_t* const mg              = material_grad->data();
#pragma omp parallel for reduction(+ : full_grad_norm2)
        for (ptrdiff_t i = 0; i < ndofs; ++i) {
            const real_t g = mg[i] + cg[i];
            full_grad_norm2 += g * g;
        }

        full_grad_norm2   = std::sqrt(full_grad_norm2);
        penetration_norm2 = std::sqrt(penetration_norm2);
        lagr_mult_norm2   = std::sqrt(lagr_mult_norm2);

        if (full_grad_norm2 < solver_tol && penetration_norm2 < solver_tol && lagr_mult_norm2 < solver_tol) {
            break;
        }

        if (loop % 100 == 0) {
            printf("%d) full_grad_norm = %g, penetration_norm = %g, lagr_mult_norm = %g\n",
                   loop,
                   full_grad_norm2,
                   penetration_norm2,
                   lagr_mult_norm2);
        }
    }
}

int test_two_body_contact() {
    const EnvOptions env = EnvOptions::read();
    env.print(std::cout);
    ptrdiff_t nx = env.nx;

    auto es   = ExecutionSpace::EXECUTION_SPACE_HOST;
    auto blas = sfem::blas<real_t>(es);

    auto      f     = create_function(env, es, env);
    auto      space = f->space();
    auto      mesh  = space->mesh_ptr();
    const int dim   = mesh->spatial_dimension();

    auto linear_op = sfem::create_linear_operator(MATRIX_FREE, f, nullptr, es);
    auto solver    = sfem::create_cg<real_t>(linear_op, es);
    solver->set_op(linear_op);
    solver->set_max_it(10000);
    solver->set_rtol(1e-4);
    solver->set_verbose(false);

    auto displacement     = sfem::create_buffer<real_t>(space->n_dofs(), es);
    auto rhs              = sfem::create_buffer<real_t>(space->n_dofs(), es);
    auto constraints_mask = sfem::create_buffer<mask_t>(mask_count(space->n_dofs()), es);

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < constraints_mask->size(); ++i) {
        constraints_mask->data()[i] = 0;
    }

    SFEM_TEST_ASSERT(f->constraints_mask(constraints_mask->data()) == SFEM_SUCCESS);

    f->apply_constraints(displacement->data());
    f->apply_constraints(rhs->data());

    solver->apply(rhs->data(), displacement->data());

    auto surface = skin(mesh);
    remove_surface_elements_connected_to_constrained_nodes(surface, constraints_mask, dim);
    surface->write(smesh::Path("contact_surface"));

    std::shared_ptr<sccd::CCD<real_t>> ccd;
    ccd = sccd::CCD<real_t>::create(surface);

    auto p0 = smesh::astype<real_t>(surface->points());
    auto p1 = smesh::astype<real_t>(surface->points());

    displace_points(surface, displacement, p1);

    real_t toi = 1;
    if (ccd) {
        ccd->find_earliest_impact_time(p0, p1, toi, 69, 1e-12);
    }

    printf("TOI: %g\n", toi);
    // SFEM_TEST_APPROXEQ(toi, 0.25, 1e-2);

    // toi *= 1.1;
    blas->scal(space->n_dofs(), env.toi_scale * toi, displacement->data());

    const real_t search_radius     = env.search_radius;
    const real_t search_radius_sqr = search_radius * search_radius;
    const real_t margin            = env.margin;

    auto contact_conditions = create_contact(space, surface, margin, search_radius_sqr, es);

    auto agumentation = sfem::create_buffer<real_t>(contact_conditions->mass_vector()->size(), es);

    real_t penalty          = env.penalty;
    auto   lagr_mult_normal = sfem::create_buffer<real_t>(space->n_dofs(), es);

    auto out = f->output();
    out->enable_AoS_to_SoA(true);
    out->set_output_dir(smesh::Path("contact_output"));

    const int outer_loops = env.outer_loops;
    const int inner_loops = env.inner_loops;

    contact_conditions->recompute(displacement);

    out->write_time_step("disp", 0, displacement->data());
    out->write_time_step("distance", 0, contact_conditions->distances_whole()->data());
    out->write_time_step("directors", 0, contact_conditions->directors()->data());
    out->write_time_step("lagr_mult_normal", 0, lagr_mult_normal->data());
    out->log_time(0);

    f->apply_constraints(displacement->data());
    f->apply_constraints(rhs->data());

    for (int outer = 0; outer < outer_loops; ++outer) {
        ContactData cd = {.surface             = surface,
                          .graph               = contact_conditions->graph(),
                          .values              = contact_conditions->values(),
                          .mass_vector         = contact_conditions->mass_vector(),
                          .normals             = contact_conditions->normals(),
                          .distances           = contact_conditions->distances(),
                          .frozen_displacement = contact_conditions->frozen_displacement(),
                          .constraints_mask    = constraints_mask,
                          .agumentation        = agumentation};

        nljacobi(cd, f, displacement, penalty, inner_loops, env.solver_tol, env.enable_augmentation);

        {
            const real_t* const u0 = contact_conditions->frozen_displacement()->data();
            real_t* const       u1 = displacement->data();
            const ptrdiff_t     n  = space->n_dofs();

#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n; ++i) {
                u1[i] = u0[i] + env.damping * (u1[i] - u0[i]);
            }
        }

        if (env.enable_ccd && ccd) {
            p0 = smesh::astype<real_t>(surface->points());
            displace_points(surface, contact_conditions->frozen_displacement(), p0);

            p1 = smesh::astype<real_t>(surface->points());
            displace_points(surface, displacement, p1);

            real_t ccd_toi = 1;
            ccd->find_earliest_impact_time(p0, p1, ccd_toi, 69, 1e-12);
            printf("CCD TOI: %g\n", ccd_toi);

            if (ccd_toi < 1) {
                const real_t* const u0 = contact_conditions->frozen_displacement()->data();
                real_t* const       u1 = displacement->data();
                const ptrdiff_t     n  = space->n_dofs();

#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n; ++i) {
                    u1[i] = u0[i] + ccd_toi * (u1[i] - u0[i]);
                }
            }
        }

        contact_conditions->recompute(displacement);

        blas->values(space->n_dofs(), 0, lagr_mult_normal->data());

        {
            const idx_t* const  node_mapping          = surface->node_mapping()->data();
            const real_t* const lagr_mult             = agumentation->data();
            const real_t* const normal_x              = contact_conditions->normals()->data()[0];
            const real_t* const normal_y              = contact_conditions->normals()->data()[1];
            const real_t* const normal_z              = contact_conditions->normals()->data()[2];
            real_t* const       lagr_mult_normal_data = lagr_mult_normal->data();
            const ptrdiff_t     n                     = surface->node_mapping()->size();

#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n; ++i) {
                const ptrdiff_t dof            = node_mapping[i] * 3;
                const real_t    lm             = lagr_mult[i];
                lagr_mult_normal_data[dof + 0] = lm * normal_x[i];
                lagr_mult_normal_data[dof + 1] = lm * normal_y[i];
                lagr_mult_normal_data[dof + 2] = lm * normal_z[i];
            }
        }

        if ((outer + 1) % env.output_frequency == 0) {
            out->write_time_step("disp", outer + 1, displacement->data());
            out->write_time_step("distance", outer + 1, contact_conditions->distances_whole()->data());
            out->write_time_step("directors", outer + 1, contact_conditions->directors()->data());
            out->write_time_step("lagr_mult_normal", outer + 1, lagr_mult_normal->data());
            out->log_time(outer + 1);
        }
    }

    return SFEM_TEST_SUCCESS;
}

int main(int argc, char* argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_two_body_contact);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
