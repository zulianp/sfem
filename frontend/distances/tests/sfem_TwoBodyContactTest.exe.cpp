#include "sfem_test.hpp"

#include "sfem_FunctionSpace.hpp"
// #include "sfem_SelfCollisions.hpp"

#include "integrations/smesh/sccd_smesh_CCD.hpp"

#include "sfem_aliases.hpp"
#include "sfem_context.hpp"
#include "smesh_mesh.hpp"

#include "sfem_API.hpp"
#include "sfem_mask.hpp"

#include "bvh/bvh.hpp"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

using namespace sfem;

struct TwoBodyContactOptions {
    int         demo             = 1;
    int         nx               = 5;
    real_t      omega            = 1. / 3;
    int         use_augmentation = 1;
    int         use_nljacobi     = 0;
    real_t      toi_scale        = 1.00;
    real_t      penalty          = 1000;
    int         outer_loops      = 30;
    int         inner_loops      = 100;
    real_t      contact_tol      = 1e-4;
    real_t      ytop             = -0.2;
    int         linear_max_it    = 10000;
    real_t      linear_rtol      = 1e-4;
    int         linear_verbose   = 0;
    int         cg_max_it        = 10000;
    real_t      cg_rtol          = 1e-4;
    int         cg_verbose       = 0;
    int         ccd_max_it       = 69;
    real_t      ccd_tol          = 1e-12;
    real_t      search_radius    = 0.001;
    int         adaptive_radius  = 0;
    real_t      margin           = 0;
    std::string demo_mesh_output = "contact_mesh";
    std::string mesh_path        = "./mesh";
    std::string dirichlet_path   = "./case.yaml";
    std::string output_dir       = "contact_output";

    static TwoBodyContactOptions from_env() {
        TwoBodyContactOptions opts;
        opts.demo             = smesh::Env::read("SFEM_DEMO", opts.demo);
        opts.nx               = smesh::Env::read("SFEM_NX", opts.nx);
        opts.omega            = smesh::Env::read("SFEM_OMEGA", opts.omega);
        opts.use_augmentation = smesh::Env::read("SFEM_USE_AUGMENTATION", opts.use_augmentation);
        opts.use_nljacobi     = smesh::Env::read("SFEM_USE_NLJACOBI", opts.use_nljacobi);
        opts.toi_scale        = smesh::Env::read("SFEM_TOI_SCALE", opts.toi_scale);
        opts.penalty          = smesh::Env::read("SFEM_PENALTY", opts.penalty);
        opts.outer_loops      = smesh::Env::read("SFEM_OUTER_LOOPS", opts.outer_loops);
        opts.inner_loops      = smesh::Env::read("SFEM_INNER_LOOPS", opts.inner_loops);
        opts.contact_tol      = smesh::Env::read("SFEM_CONTACT_TOL", opts.contact_tol);
        opts.ytop             = smesh::Env::read("SFEM_YTOP", opts.ytop);
        opts.linear_max_it    = smesh::Env::read("SFEM_LINEAR_MAX_IT", opts.linear_max_it);
        opts.linear_rtol      = smesh::Env::read("SFEM_LINEAR_RTOL", opts.linear_rtol);
        opts.linear_verbose   = smesh::Env::read("SFEM_LINEAR_VERBOSE", opts.linear_verbose);
        opts.cg_max_it        = smesh::Env::read("SFEM_CG_MAX_IT", opts.cg_max_it);
        opts.cg_rtol          = smesh::Env::read("SFEM_CG_RTOL", opts.cg_rtol);
        opts.cg_verbose       = smesh::Env::read("SFEM_CG_VERBOSE", opts.cg_verbose);
        opts.ccd_max_it       = smesh::Env::read("SFEM_CCD_MAX_IT", opts.ccd_max_it);
        opts.ccd_tol          = smesh::Env::read("SFEM_CCD_TOL", opts.ccd_tol);
        opts.search_radius    = smesh::Env::read("SFEM_SEARCH_RADIUS", opts.search_radius);
        opts.adaptive_radius  = smesh::Env::read("SMFEM_ADATIVE_RADIUS", opts.adaptive_radius);
        opts.demo_mesh_output = smesh::Env::read_string("SFEM_DEMO_MESH_OUTPUT", opts.demo_mesh_output);
        opts.mesh_path        = smesh::Env::read_string("SFEM_MESH_PATH", opts.mesh_path);
        opts.dirichlet_path   = smesh::Env::read_string("SFEM_DIRICHLET_PATH", opts.dirichlet_path);
        opts.output_dir       = smesh::Env::read_string("SFEM_OUTPUT_DIR", opts.output_dir);
        opts.margin           = smesh::Env::read("SFEM_MARGIN", opts.margin);
        return opts;
    }
};

std::shared_ptr<Function> create_function(const ptrdiff_t nx, const ExecutionSpace es, const TwoBodyContactOptions& opts) {
    if (opts.demo) {
        auto mesh1 =
                smesh::Mesh::create_tet4_cube(Communicator::self(), nx, std::max<ptrdiff_t>(1, nx / 5), nx, 0, 0.8, 0, 1, 1, 1);
        auto mesh2 = smesh::Mesh::create_tet4_cube(Communicator::self(),
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

        mesh->write(smesh::Path(opts.demo_mesh_output));

        auto top_ss =
                sfem::Sideset::create_from_selector(mesh, [=](const geom_t /*x*/, const geom_t y, const geom_t /*z*/) -> bool {
                    return y > (1.9 - 1e-4) && y < (1.9 + 1e-4);
                });

        // auto bottom_ss = sfem::Sideset::create_from_selector(
        //         mesh,
        //         [=](const geom_t /*x*/, const geom_t y, const geom_t /*z*/) -> bool { return y > (0.8 - 1e-4) && y < (0.8 +
        //         1e-4);
        //         });

        auto left_ss = sfem::Sideset::create_from_selector(
                mesh, [=](const geom_t x, const geom_t /*y*/, const geom_t /*z*/) -> bool { return x > (-1e-4) && x < (1e-4); });

        auto right_ss = sfem::Sideset::create_from_selector(
                mesh,
                [=](const geom_t x, const geom_t /*y*/, const geom_t /*z*/) -> bool { return x > (1 - 1e-4) && x < (1 + 1e-4); });

        const int dim   = mesh->spatial_dimension();
        auto      space = FunctionSpace::create(mesh, dim);

        auto op = create_op(space, "LinearElasticity", es);
        op->initialize();

        auto f = Function::create(space);
        f->add_operator(op);

        auto top_ns = smesh::create_nodeset_from_sidesets(mesh, top_ss);
        // auto bottom_ns = smesh::create_nodeset_from_sidesets(mesh, bottom_ss);
        auto left_ns  = smesh::create_nodeset_from_sidesets(mesh, left_ss);
        auto right_ns = smesh::create_nodeset_from_sidesets(mesh, right_ss);

        assert(top_ns != nullptr);
        // assert(bottom_ns != nullptr);
        assert(top_ns->size() > 0);
        // assert(bottom_ns->size() > 0);

        DirichletConditions::Condition xtop{.sidesets = top_ss, .nodeset = top_ns, .value = 0, .component = 0};
        DirichletConditions::Condition ytop{.sidesets = top_ss, .nodeset = top_ns, .value = opts.ytop, .component = 1};
        DirichletConditions::Condition ztop{.sidesets = top_ss, .nodeset = top_ns, .value = 0, .component = 2};

        DirichletConditions::Condition xleft{.sidesets = left_ss, .nodeset = left_ns, .value = 0, .component = 0};
        DirichletConditions::Condition yleft{.sidesets = left_ss, .nodeset = left_ns, .value = 0, .component = 1};
        DirichletConditions::Condition zleft{.sidesets = left_ss, .nodeset = left_ns, .value = 0, .component = 2};

        DirichletConditions::Condition xright{.sidesets = right_ss, .nodeset = right_ns, .value = 0, .component = 0};
        DirichletConditions::Condition yright{.sidesets = right_ss, .nodeset = right_ns, .value = 0, .component = 1};
        DirichletConditions::Condition zright{.sidesets = right_ss, .nodeset = right_ns, .value = 0, .component = 2};

        // DirichletConditions::Condition xbottom{.sidesets = bottom_ss, .nodeset = bottom_ns, .value = 0, .component = 0};
        // DirichletConditions::Condition ybottom{.sidesets = bottom_ss, .nodeset = bottom_ns, .value = 0, .component = 1};
        // DirichletConditions::Condition zbottom{.sidesets = bottom_ss, .nodeset = bottom_ns, .value = 0, .component = 2};

        auto conds =
                sfem::create_dirichlet_conditions(space, {xtop, ytop, ztop, xleft, yleft, zleft, xright, yright, zright}, es);
        // auto conds = sfem::create_dirichlet_conditions(space, {xtop, ytop, ztop, xbottom, ybottom, zbottom}, es);
        f->add_constraint(conds);

        return f;
    } else {
        smesh::Path mesh_path{opts.mesh_path};
        smesh::Path dirichlet_path{opts.dirichlet_path};
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
    smesh::SharedBuffer<real_t>                      agumentation;
};

std::shared_ptr<smesh::CRSGraph<count_t, idx_t>> create_contact_graph(const smesh::SharedBuffer<idx_t*>& elements,
                                                                      const smesh::SharedBuffer<idx_t>&  element_idx) {
    const ptrdiff_t npoints = element_idx->size();
    const int       nxe     = elements->extent(0);
    auto            rowptr  = sfem::create_host_buffer<count_t>(npoints + 1);
    auto            colidx  = sfem::create_host_buffer<idx_t>(npoints * nxe);

    auto elements_data    = elements->data();
    auto rowptr_data      = rowptr->data();
    auto colidx_data      = colidx->data();
    auto element_idx_data = element_idx->data();

    for (ptrdiff_t i = 0; i < npoints; i++) {
        ptrdiff_t offset   = element_idx_data[i] == -1 ? 0 : nxe;
        rowptr_data[i + 1] = offset + rowptr_data[i];
    }

    for (ptrdiff_t i = 0; i < npoints; i++) {
        const idx_t e = element_idx_data[i];
        if (e == -1) continue;
        for (int j = 0; j < nxe; j++) {
            colidx_data[rowptr_data[i] + j] = elements_data[j][e];
        }
    }

    return std::make_shared<smesh::CRSGraph<count_t, idx_t>>(rowptr, colidx);
}

void local_coordinates(const smesh::SharedBuffer<idx_t*>&     elements,
                       const smesh::SharedBuffer<real_t*>&    points,
                       const smesh::SharedBuffer<idx_t>&      element_idx,
                       const smesh::SharedBuffer<real_t*>&    c,
                       const smesh::CRSGraph<count_t, idx_t>& graph,
                       const smesh::SharedBuffer<real_t>&     values) {
    auto p   = c->data();
    auto pts = points->data();

    const ptrdiff_t           n   = c->extent(1);
    const idx_t* const* const idx = elements->data();

    SMESH_ASSERT(n == element_idx->size());

    auto rowptr = graph.rowptr()->data();
    auto colidx = graph.colidx()->data();
    auto vals   = values->data();

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < n; i++) {
        const ptrdiff_t e = element_idx->data()[i];
        if (e == -1) continue;

        const idx_t e0 = idx[0][e];
        const idx_t e1 = idx[1][e];
        const idx_t e2 = idx[2][e];

        const real_t x0 = pts[0][e0];
        const real_t x1 = pts[0][e1];
        const real_t x2 = pts[0][e2];
        const real_t y0 = pts[1][e0];
        const real_t y1 = pts[1][e1];
        const real_t y2 = pts[1][e2];
        const real_t z0 = pts[2][e0];
        const real_t z1 = pts[2][e1];
        const real_t z2 = pts[2][e2];

        const real_t p0 = p[0][i];
        const real_t p1 = p[1][i];
        const real_t p2 = p[2][i];

        const real_t v0x = x1 - x0;
        const real_t v0y = y1 - y0;
        const real_t v0z = z1 - z0;
        const real_t v1x = x2 - x0;
        const real_t v1y = y2 - y0;
        const real_t v1z = z2 - z0;
        const real_t v2x = p0 - x0;
        const real_t v2y = p1 - y0;
        const real_t v2z = p2 - z0;

        const real_t d00 = v0x * v0x + v0y * v0y + v0z * v0z;
        const real_t d01 = v0x * v1x + v0y * v1y + v0z * v1z;
        const real_t d11 = v1x * v1x + v1y * v1y + v1z * v1z;
        const real_t d20 = v2x * v0x + v2y * v0y + v2z * v0z;
        const real_t d21 = v2x * v1x + v2y * v1y + v2z * v1z;

        const real_t inv_det = 1 / (d00 * d11 - d01 * d01);
        const real_t w1      = (d11 * d20 - d01 * d21) * inv_det;
        const real_t w2      = (d00 * d21 - d01 * d20) * inv_det;
        const real_t w0      = 1 - w1 - w2;

        SMESH_ASSERT(std::abs(w0 + w1 + w2 - 1) < 1e-6);

        real_t ws[3]   = {w0, w1, w2};
        idx_t  keys[3] = {e0, e1, e2};

        const count_t row_offset = rowptr[i];
        idx_t*        row        = &colidx[row_offset];
        const int     lenrow     = rowptr[i + 1] - row_offset;

        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < lenrow; k++) {
                if (row[k] == keys[j]) {
                    vals[row_offset + k] = ws[j];
                    break;
                }
            }
        }
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

    // printf("-------------------------------------\n");
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

        // printf("%d) pen = (%g - %g) = %g, <pen + %g> = %g, (%g, %g, %g)\n",
        //        (int)i,
        //        normal_diff,
        //        g,
        //        pen,
        //        lagr_mult / penalty,
        //        macaulay[i],
        //        normal[0],
        //        normal[1],
        //        normal[2]);
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
              const TwoBodyContactOptions&                 opts) {
    SFEM_TRACE_SCOPE("nljacobi");

    auto      space = f->space();
    auto      mesh  = space->mesh_ptr();
    const int dim   = mesh->spatial_dimension();
    assert(dim == 3);

    const ptrdiff_t ndofs          = space->n_dofs();
    const ptrdiff_t n_nodes        = ndofs / dim;
    const ptrdiff_t n_contact      = cd.surface->node_mapping()->size();
    const int       sym_block_size = (dim * (dim + 1)) / 2;
    auto            es             = f->execution_space();
    auto            blas           = sfem::blas<real_t>(es);

    auto material_grad     = sfem::create_buffer<real_t>(ndofs, es);
    auto elast_diag_values = sfem::create_buffer<real_t>(n_nodes * sym_block_size, es);
    auto constraint_mask   = sfem::create_buffer<mask_t>(mask_count(ndofs), es);
    auto contact_grad      = sfem::create_buffer<real_t>(ndofs, es);
    auto macaulay          = sfem::create_buffer<real_t>(n_contact, es);
    auto diag_values       = sfem::create_buffer<real_t>(dim * dim, n_contact, es);

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < constraint_mask->size(); ++i) {
        constraint_mask->data()[i] = 0;
    }

    const idx_t* const nm = cd.surface->node_mapping()->data();
    f->constraints_mask(constraint_mask->data());

    const mask_t* const mask = constraint_mask->data();
    real_t* const       xd   = x->data();

    // If the material is nonlinear should be inside the loop
    blas->values(elast_diag_values->size(), 0, elast_diag_values->data());
    f->hessian_block_diag_sym(x->data(), elast_diag_values->data());

    for (int loop = 0; loop < opts.inner_loops; ++loop) {
        blas->values(ndofs, 0, material_grad->data());
        f->gradient(x->data(), material_grad->data());

        const real_t* const eg = material_grad->data();
        const real_t* const ed = elast_diag_values->data();

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n_nodes; ++i) {
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
                if (std::isfinite(a0) && a0 != 0) xd[dof + 0] -= opts.omega * g0 / a0;
                if (std::isfinite(a4) && a4 != 0) xd[dof + 1] -= opts.omega * g1 / a4;
                if (std::isfinite(a8) && a8 != 0) xd[dof + 2] -= opts.omega * g2 / a8;
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

            xd[dof + 0] -= opts.omega * (i0 * g0 + i1 * g1 + i2 * g2);
            xd[dof + 1] -= opts.omega * (i3 * g0 + i4 * g1 + i5 * g2);
            xd[dof + 2] -= opts.omega * (i6 * g0 + i7 * g1 + i8 * g2);
        }

        blas->values(ndofs, 0, contact_grad->data());
        for (int d = 0; d < dim * dim; ++d) {
            blas->values(n_contact, 0, diag_values->data()[d]);
        }

        compute_macaulay_term(cd, opts.penalty, x->data(), macaulay->data());
        assemble_contact_gradient(cd, opts.penalty, macaulay->data(), contact_grad->data());
        assemble_contact_hessian_diag(cd, opts.penalty, macaulay->data(), 1, diag_values->data());
        gather_combine_hessian_diag(cd, constraint_mask->data(), elast_diag_values->data(), 1, diag_values->data());

        const real_t* const* const dv = diag_values->data();
        const real_t* const        cg = contact_grad->data();

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n_contact; ++i) {
            const ptrdiff_t local_node  = i;
            const ptrdiff_t global_node = nm[i];
            const ptrdiff_t dof         = global_node * 3;

            const real_t g0 = mask_get(dof + 0, mask) ? 0 : cg[dof + 0];
            const real_t g1 = mask_get(dof + 1, mask) ? 0 : cg[dof + 1];
            const real_t g2 = mask_get(dof + 2, mask) ? 0 : cg[dof + 2];

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
                if (std::isfinite(a0) && a0 != 0) xd[dof + 0] -= opts.omega * g0 / a0;
                if (std::isfinite(a4) && a4 != 0) xd[dof + 1] -= opts.omega * g1 / a4;
                if (std::isfinite(a8) && a8 != 0) xd[dof + 2] -= opts.omega * g2 / a8;
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

            xd[dof + 0] -= opts.omega * (i0 * g0 + i1 * g1 + i2 * g2);
            xd[dof + 1] -= opts.omega * (i3 * g0 + i4 * g1 + i5 * g2);
            xd[dof + 2] -= opts.omega * (i6 * g0 + i7 * g1 + i8 * g2);
        }

        compute_macaulay_term(cd, opts.penalty, x->data(), macaulay->data());

        if (opts.use_augmentation) {
            real_t* const       aug = cd.agumentation->data();
            const real_t* const m   = macaulay->data();
#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n_contact; ++i) {
                aug[i] = opts.penalty * m[i];
            }
        }

        blas->values(ndofs, 0, material_grad->data());
        f->gradient(x->data(), material_grad->data());
        assemble_contact_gradient(cd, opts.penalty, macaulay->data(), material_grad->data());
        f->apply_zero_constraints(material_grad->data());

        const real_t grad_norm = blas->norm2(ndofs, material_grad->data());
        const bool   converged = grad_norm < opts.contact_tol;
        printf("nljacobi[%d] ||g|| = %g%s\n", loop, (double)grad_norm, converged ? " converged" : "");
        if (converged) break;
    }
}

void apply_contact_hessian(ContactData&        cd,
                           const real_t        penalty,
                           const real_t* const macaulay,
                           const mask_t* const is_constrained,
                           const real_t* const h,
                           real_t* const       out) {
    SFEM_TRACE_SCOPE("apply_contact_hessian");

    const int dim    = cd.surface->spatial_dimension();
    auto      graph  = cd.graph;
    auto      values = cd.values;
    auto      rowptr = graph->rowptr()->data();
    auto      colidx = graph->colidx()->data();
    auto      vals   = values->data();
    ptrdiff_t n      = graph->rowptr()->size() - 1;

    auto normals = cd.normals->data();
    auto mass    = cd.mass_vector->data();
    auto nm      = cd.surface->node_mapping()->data();

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < n; ++i) {
        if (macaulay[i] == 0) continue;

        const count_t lenrow = rowptr[i + 1] - rowptr[i];
        if (lenrow == 0) continue;

        const idx_t* const  row     = &colidx[rowptr[i]];
        const real_t* const weights = &vals[rowptr[i]];
        const ptrdiff_t     dof1    = nm[i] * dim;

        real_t diff[3] = {0, 0, 0};
        for (int d = 0; d < dim; ++d) {
            diff[d] = h[dof1 + d];
        }

        for (count_t j = 0; j < lenrow; ++j) {
            const ptrdiff_t dof2 = nm[row[j]] * dim;
            const real_t    w    = weights[j];
            for (int d = 0; d < dim; ++d) {
                diff[d] -= w * h[dof2 + d];
            }
        }

        real_t normal[3] = {0, 0, 0};
        real_t nd        = 0;
        for (int d = 0; d < dim; ++d) {
            normal[d] = normals[d][i];
            nd += normal[d] * diff[d];
        }

        const real_t scale = mass[i] * penalty * nd;
        for (int d = 0; d < dim; ++d) {
            const real_t force = scale * normal[d];

            if (!mask_get(dof1 + d, is_constrained)) {
#pragma omp atomic update
                out[dof1 + d] += force;
            }

            for (count_t j = 0; j < lenrow; ++j) {
                const ptrdiff_t dof2 = nm[row[j]] * dim + d;
                if (mask_get(dof2, is_constrained)) continue;
#pragma omp atomic update
                out[dof2] -= force * weights[j];
            }
        }
    }
}

void assemble_contact_hessian_diag(ContactData&        cd,
                                   const real_t        penalty,
                                   const real_t* const macaulay,
                                   const mask_t* const is_constrained,
                                   real_t* const       diag) {
    SFEM_TRACE_SCOPE("assemble_contact_hessian_diag");

    const int dim    = cd.surface->spatial_dimension();
    auto      graph  = cd.graph;
    auto      values = cd.values;
    auto      rowptr = graph->rowptr()->data();
    auto      colidx = graph->colidx()->data();
    auto      vals   = values->data();
    ptrdiff_t n      = graph->rowptr()->size() - 1;

    auto normals = cd.normals->data();
    auto mass    = cd.mass_vector->data();
    auto nm      = cd.surface->node_mapping()->data();

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < n; ++i) {
        if (macaulay[i] == 0) continue;

        const count_t lenrow = rowptr[i + 1] - rowptr[i];
        if (lenrow == 0) continue;

        const idx_t* const  row     = &colidx[rowptr[i]];
        const real_t* const weights = &vals[rowptr[i]];
        const ptrdiff_t     dof1    = nm[i] * dim;

        for (int d = 0; d < dim; ++d) {
            const real_t nd  = normals[d][i];
            const real_t val = mass[i] * penalty * nd * nd;

            if (!mask_get(dof1 + d, is_constrained)) {
#pragma omp atomic update
                diag[dof1 + d] += val;
            }

            for (count_t j = 0; j < lenrow; ++j) {
                const ptrdiff_t dof2 = nm[row[j]] * dim + d;
                if (mask_get(dof2, is_constrained)) continue;
#pragma omp atomic update
                diag[dof2] += weights[j] * weights[j] * val;
            }
        }
    }
}

void cg_solve(ContactData&                                 cd,
              const std::shared_ptr<sfem::Function>&       f,
              const std::shared_ptr<sfem::Buffer<real_t>>& x,
              const TwoBodyContactOptions&                 opts) {
    SFEM_TRACE_SCOPE("cg_solve");

    auto            space     = f->space();
    const ptrdiff_t ndofs     = space->n_dofs();
    const ptrdiff_t n_contact = cd.surface->node_mapping()->size();
    auto            es        = f->execution_space();
    auto            blas      = sfem::blas<real_t>(es);

    auto grad            = sfem::create_buffer<real_t>(ndofs, es);
    auto rhs             = sfem::create_buffer<real_t>(ndofs, es);
    auto dx              = sfem::create_buffer<real_t>(ndofs, es);
    auto diag            = sfem::create_buffer<real_t>(ndofs, es);
    auto macaulay        = sfem::create_buffer<real_t>(n_contact, es);
    auto constraint_mask = sfem::create_buffer<mask_t>(mask_count(ndofs), es);

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < constraint_mask->size(); ++i) {
        constraint_mask->data()[i] = 0;
    }

    f->constraints_mask(constraint_mask->data());

    auto linearized_op = sfem::make_op<real_t>(
            ndofs,
            ndofs,
            [=, &cd](const real_t* const from, real_t* const to) {
                blas->values(ndofs, 0, to);
                f->apply(x->data(), from, to);
                apply_contact_hessian(cd, opts.penalty, macaulay->data(), constraint_mask->data(), from, to);
                f->apply_zero_constraints(to);
            },
            es);

    auto preconditioner = sfem::create_inverse_diagonal_scaling(diag, es);
    auto solver         = sfem::create_cg<real_t>(linearized_op, es);
    solver->set_preconditioner_op(preconditioner);
    solver->set_max_it(std::min<int>(opts.cg_max_it, std::max<ptrdiff_t>(1, ndofs)));
    solver->set_rtol(opts.cg_rtol);
    solver->set_verbose(opts.cg_verbose != 0);

    for (int loop = 0; loop < opts.inner_loops; ++loop) {
        compute_macaulay_term(cd, opts.penalty, x->data(), macaulay->data());

        blas->values(ndofs, 0, grad->data());
        f->gradient(x->data(), grad->data());
        assemble_contact_gradient(cd, opts.penalty, macaulay->data(), grad->data());
        f->apply_zero_constraints(grad->data());

        real_t grad_norm = blas->norm2(ndofs, grad->data());
        if (grad_norm < opts.contact_tol) {
            printf("cgsolve[%d] ||g|| = %g converged\n", loop, (double)grad_norm);
            break;
        }

        blas->values(ndofs, 0, diag->data());
        f->hessian_diag(x->data(), diag->data());
        assemble_contact_hessian_diag(cd, opts.penalty, macaulay->data(), constraint_mask->data(), diag->data());

        real_t* const diag_data = diag->data();
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < ndofs; ++i) {
            if (!std::isfinite(diag_data[i]) || std::abs(diag_data[i]) < real_t(1e-14)) {
                diag_data[i] = 1;
            }
        }

        blas->axpby(ndofs, -1, grad->data(), 0, rhs->data());
        f->apply_zero_constraints(rhs->data());
        blas->values(ndofs, 0, dx->data());
        solver->apply(rhs->data(), dx->data());
        f->apply_zero_constraints(dx->data());
        blas->axpby(ndofs, 1, dx->data(), 1, x->data());
        f->apply_constraints(x->data());

        compute_macaulay_term(cd, opts.penalty, x->data(), macaulay->data());

        if (opts.use_augmentation) {
            real_t* const       aug = cd.agumentation->data();
            const real_t* const m   = macaulay->data();
#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n_contact; ++i) {
                aug[i] = opts.penalty * m[i];
            }
        }

        blas->values(ndofs, 0, grad->data());
        f->gradient(x->data(), grad->data());
        assemble_contact_gradient(cd, opts.penalty, macaulay->data(), grad->data());
        f->apply_zero_constraints(grad->data());

        grad_norm            = blas->norm2(ndofs, grad->data());
        const bool converged = grad_norm < opts.contact_tol;
        printf("cgsolve[%d] ||g|| = %g%s\n", loop, (double)grad_norm, converged ? " converged" : "");
        if (converged) break;
    }
}

int test_two_body_contact() {
    auto es   = ExecutionSpace::EXECUTION_SPACE_HOST;
    auto blas = sfem::blas<real_t>(es);

    const auto opts = TwoBodyContactOptions::from_env();

    auto      f     = create_function(opts.nx, es, opts);
    auto      space = f->space();
    auto      mesh  = space->mesh_ptr();
    const int dim   = mesh->spatial_dimension();

    auto linear_op = sfem::create_linear_operator(MATRIX_FREE, f, nullptr, es);
    auto solver    = sfem::create_cg<real_t>(linear_op, es);
    solver->set_op(linear_op);
    solver->set_max_it(opts.linear_max_it);
    solver->set_rtol(opts.linear_rtol);
    solver->set_verbose(opts.linear_verbose != 0);

    auto displacement = sfem::create_buffer<real_t>(space->n_dofs(), es);
    auto rhs          = sfem::create_buffer<real_t>(space->n_dofs(), es);

    f->apply_constraints(displacement->data());
    f->apply_constraints(rhs->data());

    solver->apply(rhs->data(), displacement->data());

    auto surface = skin(mesh);

    auto ccd = sccd::CCD<real_t>::create(surface);

    auto p0 = smesh::astype<real_t>(surface->points());
    auto p1 = smesh::astype<real_t>(surface->points());

    displace_points(surface, displacement, p1);

    real_t toi = 1;
    ccd->find_earliest_impact_time(p0, p1, toi, opts.ccd_max_it, opts.ccd_tol);

    printf("TOI: %g\n", toi);
    // SFEM_TEST_APPROXEQ(toi, 0.25, 1e-2);

    blas->scal(space->n_dofs(), opts.toi_scale * toi, displacement->data());

    const real_t search_radius     = opts.search_radius;
    const real_t search_radius_sqr = search_radius * search_radius;

    auto surface_elements = surface->block(0)->elements();
    auto npoints          = surface->n_nodes();
    auto nselements       = surface->n_elements();

    auto                                             closest_points      = sfem::create_buffer<real_t>(dim, npoints, es);
    auto                                             distances           = sfem::create_buffer<real_t>(npoints, es);
    auto                                             closest_triangles   = sfem::create_buffer<idx_t>(npoints, es);
    auto                                             distances_whole     = sfem::create_buffer<real_t>(space->n_dofs(), es);
    auto                                             directors           = sfem::create_buffer<real_t>(space->n_dofs(), es);
    auto                                             normals             = sfem::create_buffer<real_t>(dim, npoints, es);
    auto                                             frozen_displacement = sfem::create_buffer<real_t>(space->n_dofs(), es);
    auto                                             adaptive_radius_sqr = sfem::create_buffer<real_t>(npoints, es);
    std::shared_ptr<smesh::CRSGraph<count_t, idx_t>> graph;
    smesh::SharedBuffer<real_t>                      values;

    if (opts.adaptive_radius) {
        blas->values(space->n_dofs(), 0, frozen_displacement->data());
    }

    auto recompute_contact_conditions = [&]() {
        p1 = smesh::astype<real_t>(surface->points());
        displace_points(surface, displacement, p1);

        ptrdiff_t     radius_stride       = 0;
        const real_t* radius_squared_data = &search_radius_sqr;
        if (opts.adaptive_radius) {
            radius_stride                         = 1;
            radius_squared_data                   = adaptive_radius_sqr->data();
            const idx_t* const  node_mapping_data = surface->node_mapping()->data();
            const real_t* const u                 = displacement->data();
            const real_t* const u0                = frozen_displacement->data();
            real_t* const       r2                = adaptive_radius_sqr->data();
#pragma omp parallel for
            for (ptrdiff_t i = 0; i < npoints; i++) {
                const ptrdiff_t dof = node_mapping_data[i] * dim;
                const real_t    dx  = u[dof + 0] - u0[dof + 0];
                const real_t    dy  = u[dof + 1] - u0[dof + 1];
                const real_t    dz  = u[dof + 2] - u0[dof + 2];
                r2[i]               = std::max(1e-10, dx * dx + dy * dy + dz * dz);
            }
        }

        ssdf::closest_within_radius_bvh(npoints,
                                        p1->data()[0],
                                        p1->data()[1],
                                        p1->data()[2],
                                        nselements,
                                        surface_elements->data()[0],
                                        surface_elements->data()[1],
                                        surface_elements->data()[2],
                                        npoints,
                                        p1->data()[0],
                                        p1->data()[1],
                                        p1->data()[2],
                                        radius_stride,
                                        radius_squared_data,
                                        closest_triangles->data(),
                                        distances->data(),
                                        closest_points->data()[0],
                                        closest_points->data()[1],
                                        closest_points->data()[2],
                                        true);

        blas->values(space->n_dofs(), 0, distances_whole->data());
        blas->values(space->n_dofs(), 0, directors->data());
        for (int d = 0; d < dim; ++d) {
            blas->values(npoints, 0, normals->data()[d]);
        }

        auto node_mapping           = surface->node_mapping()->data();
        auto directors_data         = directors->data();
        auto distances_whole_data   = distances_whole->data();
        auto distances_data         = distances->data();
        auto closest_points_data    = closest_points->data();
        auto p1_data                = p1->data();
        auto closest_triangles_data = closest_triangles->data();
        auto normals_data           = normals->data();
        auto surface_elements_data  = surface_elements->data();
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < npoints; i++) {
            const idx_t tri = closest_triangles_data[i];
            if (tri == -1) {
                distances_data[i] = 1000;
                continue;
            }

            const idx_t e0 = surface_elements_data[0][tri];
            const idx_t e1 = surface_elements_data[1][tri];
            const idx_t e2 = surface_elements_data[2][tri];

            const real_t v0x = p1_data[0][e1] - p1_data[0][e0];
            const real_t v0y = p1_data[1][e1] - p1_data[1][e0];
            const real_t v0z = p1_data[2][e1] - p1_data[2][e0];

            const real_t v1x = p1_data[0][e2] - p1_data[0][e0];
            const real_t v1y = p1_data[1][e2] - p1_data[1][e0];
            const real_t v1z = p1_data[2][e2] - p1_data[2][e0];

            real_t tnx = v0y * v1z - v0z * v1y;
            real_t tny = v0z * v1x - v0x * v1z;
            real_t tnz = v0x * v1y - v0y * v1x;

            const real_t nn = std::sqrt(tnx * tnx + tny * tny + tnz * tnz);
            if (nn > 0) {
                tnx /= -nn;
                tny /= -nn;
                tnz /= -nn;
            } else {
                SFEM_ERROR("Triangle normal is zero\n");
            }

            real_t nx = 0, ny = 0, nz = 0;
            real_t dx = p1_data[0][i] - closest_points_data[0][i];
            real_t dy = p1_data[1][i] - closest_points_data[1][i];
            real_t dz = p1_data[2][i] - closest_points_data[2][i];
            real_t dn = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (dn > 0) {
                nx = dx / dn;
                ny = dy / dn;
                nz = dz / dn;
            } else {
                nx = tnx;
                ny = tny;
                nz = tnz;
            }

            const real_t    normal_dot  = tnx * nx + tny * ny + tnz * nz;
            const real_t    signed_dist = -dn * normal_dot - opts.margin;
            const ptrdiff_t dof         = node_mapping[i] * dim;

            distances_data[i]         = signed_dist;
            distances_whole_data[dof] = signed_dist;
            directors_data[dof + 0]   = -dx;
            directors_data[dof + 1]   = -dy;
            directors_data[dof + 2]   = -dz;
            normals_data[0][i]        = nx;
            normals_data[1][i]        = ny;
            normals_data[2][i]        = nz;
        }

        graph  = create_contact_graph(surface_elements, closest_triangles);
        values = sfem::create_buffer<real_t>(graph->nnz(), es);
        local_coordinates(surface_elements, p1, closest_triangles, closest_points, *graph, values);
        blas->copy(space->n_dofs(), displacement->data(), frozen_displacement->data());
    };

    auto trace_space = std::make_shared<FunctionSpace>(surface, 1);
    auto mass_vector = create_host_buffer<real_t>(trace_space->n_dofs());

    {
        auto bop = sfem::Factory::create_op(trace_space, "Mass");
        bop->initialize();

        auto ones = create_host_buffer<real_t>(trace_space->n_dofs());
        sfem::blas<real_t>(EXECUTION_SPACE_HOST)->values(trace_space->n_dofs(), 1, ones->data());
        bop->apply(nullptr, ones->data(), mass_vector->data());
    }

    auto agumentation = sfem::create_buffer<real_t>(trace_space->n_dofs(), es);
    blas->values(trace_space->n_dofs(), 0, agumentation->data());

    auto lagr_mult_normal = sfem::create_buffer<real_t>(space->n_dofs(), es);

    f->apply_constraints(displacement->data());
    f->apply_constraints(rhs->data());

    auto out = f->output();
    out->enable_AoS_to_SoA(true);
    out->set_output_dir(smesh::Path(opts.output_dir));

    out->write_time_step("disp", 0, displacement->data());
    out->write_time_step("distance", 0, distances_whole->data());
    out->write_time_step("directors", 0, directors->data());
    out->write_time_step("lagr_mult_normal", 0, lagr_mult_normal->data());
    out->log_time(0);

    for (int outer = 0; outer < opts.outer_loops; ++outer) {
        recompute_contact_conditions();

        ContactData cd = {.surface             = surface,
                          .graph               = graph,
                          .values              = values,
                          .mass_vector         = mass_vector,
                          .normals             = normals,
                          .distances           = distances,
                          .frozen_displacement = frozen_displacement,
                          .agumentation        = agumentation};

        if (opts.use_nljacobi) {
            nljacobi(cd, f, displacement, opts);
        } else {
            cg_solve(cd, f, displacement, opts);
        }

        {
            const idx_t* const  node_mapping          = surface->node_mapping()->data();
            const real_t* const lagr_mult             = agumentation->data();
            const real_t* const normal_x              = normals->data()[0];
            const real_t* const normal_y              = normals->data()[1];
            const real_t* const normal_z              = normals->data()[2];
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

        out->write_time_step("disp", outer + 1, displacement->data());
        out->write_time_step("distance", outer + 1, distances_whole->data());
        out->write_time_step("directors", outer + 1, directors->data());
        out->write_time_step("lagr_mult_normal", outer + 1, lagr_mult_normal->data());
        out->log_time(outer + 1);
    }

    return SFEM_TEST_SUCCESS;
}

int main(int argc, char* argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_two_body_contact);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
