#include "sfem_API.hpp"

#include "adj_table.h"

#include "sfem_glob.hpp"
#include "sfem_hex8_mesh_graph.h"

#include "hex8_inline_cpu.h"
#include "hex8_linear_elasticity.h"
#include "hex8_linear_elasticity_inline_cpu.h"
#include "line_quadrature.h"

namespace sfem {
    class HEX8Smooth final : public Op {
    public:
        std::shared_ptr<FunctionSpace> space;
        HEX8Smooth(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        const char *name() const override { return "HEX8Smooth"; }

        bool is_linear() const override { return true; }
        int  initialize() override { return SFEM_SUCCESS; }

        int gradient(const real_t *const x, real_t *const out) override {
            SFEM_ERROR("IMPLEMENT ME!");
            return SFEM_FAILURE;
        }
        int apply(const real_t *const x, const real_t *const h, real_t *const out) override {
            // FIXME Find out a good mesh smoothing operator
            return affine_hex8_linear_elasticity_apply(space->mesh_ptr()->n_elements(),
                                                       space->mesh_ptr()->n_nodes(),
                                                       space->mesh_ptr()->elements()->data(),
                                                       space->mesh_ptr()->points()->data(),
                                                       1.,
                                                       1.,
                                                       3,
                                                       &h[0],
                                                       &h[1],
                                                       &h[2],
                                                       3,
                                                       &out[0],
                                                       &out[1],
                                                       &out[2]);
        }

        int value(const real_t *x, real_t *const out) override {
            SFEM_ERROR("IMPLEMENT ME!");
            return SFEM_FAILURE;
        }

        int hessian_crs(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override {
            SFEM_ERROR("IMPLEMENT ME!");
            return SFEM_FAILURE;
        }
    };
}  // namespace sfem

std::shared_ptr<sfem::Buffer<real_t>> solve(const std::shared_ptr<sfem::Function> &f) {
    auto es        = f->execution_space();
    auto fs        = f->space();
    auto m         = fs->mesh_ptr();
    auto linear_op = sfem::create_linear_operator("MF", f, nullptr, es);
    auto cg        = sfem::create_cg<real_t>(linear_op, es);
    cg->verbose    = true;
    cg->set_max_it(1000);
    cg->set_op(linear_op);
    cg->set_rtol(1e-8);

    auto x   = sfem::create_buffer<real_t>(fs->n_dofs(), es);
    auto rhs = sfem::create_buffer<real_t>(fs->n_dofs(), es);

    f->apply_constraints(x->data());
    f->apply_constraints(rhs->data());

    cg->apply(rhs->data(), x->data());
    return x;
}

int smooth(const std::shared_ptr<sfem::Mesh> &m) {
    MPI_Comm comm = MPI_COMM_WORLD;
    auto     es   = sfem::EXECUTION_SPACE_HOST;

    const char *SFEM_EXECUTION_SPACE{nullptr};
    SFEM_READ_ENV(SFEM_EXECUTION_SPACE, );

    if (SFEM_EXECUTION_SPACE) {
        es = sfem::execution_space_from_string(SFEM_EXECUTION_SPACE);
    }

    auto fs = sfem::FunctionSpace::create(m, m->spatial_dimension());

    ptrdiff_t      n_surf_elements{0};
    element_idx_t *parent{nullptr};
    int16_t       *side_idx{nullptr};

    if (extract_skin_sideset(
                m->n_elements(), m->n_nodes(), m->element_type(), m->elements()->data(), &n_surf_elements, &parent, &side_idx) !=
        SFEM_SUCCESS) {
        SFEM_ERROR("Failed extract_skin_sideset!\n");
    }

    auto sideset = std::make_shared<sfem::Sideset>(m->comm(),
                                                   sfem::manage_host_buffer<element_idx_t>(n_surf_elements, parent),
                                                   sfem::manage_host_buffer<int16_t>(n_surf_elements, side_idx));

    auto sides = sfem::create_host_buffer<idx_t>(elem_num_nodes(side_type(m->element_type())), sideset->parent()->size());

    if (extract_surface_from_sideset(m->element_type(),
                                     m->elements()->data(),
                                     sideset->parent()->size(),
                                     sideset->parent()->data(),
                                     sideset->lfi()->data(),
                                     sides->data()) != SFEM_SUCCESS) {
        SFEM_ERROR("Failed extract_surface_from_sideset!\n");
    }

    idx_t    *idx          = nullptr;
    ptrdiff_t n_contiguous = -1;
    remap_elements_to_contiguous_index(sides->extent(1), sides->extent(0), sides->data(), &n_contiguous, &idx);
    auto nodeset = sfem::manage_host_buffer(n_contiguous, idx);

    auto sx = sfem::create_host_buffer<real_t>(nodeset->size());
    auto sy = sfem::create_host_buffer<real_t>(nodeset->size());
    auto sz = sfem::create_host_buffer<real_t>(nodeset->size());

    auto points = m->points()->data();
    for (ptrdiff_t i = 0; i < nodeset->size(); i++) {
        sx->data()[i] = points[0][idx[i]];
        sy->data()[i] = points[1][idx[i]];
        sz->data()[i] = points[2][idx[i]];
    }

    sfem::DirichletConditions::Condition s0{.sideset = sideset, .nodeset = nodeset, .values = sx, .component = 0};
    sfem::DirichletConditions::Condition s1{.sideset = sideset, .nodeset = nodeset, .values = sy, .component = 1};
    sfem::DirichletConditions::Condition s2{.sideset = sideset, .nodeset = nodeset, .values = sz, .component = 2};

    auto conds = sfem::create_dirichlet_conditions(fs, {s0, s1, s2}, es);
    auto f     = sfem::Function::create(fs);
    f->add_constraint(conds);
    f->add_operator(std::make_shared<sfem::HEX8Smooth>(fs));

    auto x = solve(f);

    {
        auto x_data = x->data();
        for (ptrdiff_t i = 0; i < m->n_nodes(); i++) {
            points[0][i] = x_data[i * 3 + 0];
            points[1][i] = x_data[i * 3 + 1];
            points[2][i] = x_data[i * 3 + 2];
        }
    }

    return SFEM_SUCCESS;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (argc != 3) {
        if (!rank) {
            fprintf(stderr, "usage: %s <input_mesh> <output_mesh>\n", argv[0]);
        }

        return EXIT_FAILURE;
    }

    auto        mesh          = sfem::Mesh::create_from_file(comm, argv[1]);
    const char *output_folder = argv[2];

    if (smooth(mesh) != SFEM_SUCCESS) {
        SFEM_ERROR("Unable to smooth!");
    }

    mesh->write(output_folder);

    return MPI_Finalize();
}
