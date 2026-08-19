#include <stdio.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "sfem_API.hpp"
#include "sfem_Function.hpp"
#include "sfem_ssgmg.hpp"
#include "smesh_env.hpp"
#include "smesh_glob.hpp"
#include "smesh_output.hpp"
#include "smesh_semistructured.hpp"
#include "smesh_sideset.hpp"

namespace {

sfem::SharedBuffer<idx_t> nodeset_from_sidesets(const std::shared_ptr<smesh::Mesh>                 &mesh,
                                                const std::vector<std::shared_ptr<smesh::Sideset>> &sidesets) {
    std::vector<idx_t> ids;
    for (const auto &ss : sidesets) {
        auto ns = smesh::create_nodeset_from_sideset(mesh, ss);
        if (!ns || ns->size() == 0) {
            continue;
        }
        auto d = ns->data();
        ids.insert(ids.end(), d, d + ns->size());
    }
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());

    auto out = sfem::create_host_buffer<idx_t>((ptrdiff_t)ids.size());
    if (!ids.empty()) {
        std::memcpy(out->data(), ids.data(), ids.size() * sizeof(idx_t));
    }
    return out;
}

real_t diffusion_for_block(const std::string &name, const real_t k_white, const real_t k_black) {
    if (name == "black") {
        return k_black;
    }
    return k_white;
}

int write_block_cell_fields(const std::shared_ptr<smesh::Mesh> &mesh,
                            const smesh::Path                  &output_dir,
                            const real_t                        k_white,
                            const real_t                        k_black) {
    const ptrdiff_t n_cells = mesh->n_elements();
    auto            k_cell  = sfem::create_host_buffer<real_t>(n_cells);
    auto            bid     = sfem::create_host_buffer<smesh::i32>(n_cells);

    // raw_to_db concatenates blocks in sorted directory-name order.
    std::vector<smesh::block_idx_t> order(mesh->n_blocks());
    for (size_t i = 0; i < order.size(); ++i) {
        order[i] = static_cast<smesh::block_idx_t>(i);
    }
    std::sort(order.begin(), order.end(), [&](const smesh::block_idx_t a, const smesh::block_idx_t b) {
        return mesh->block(a)->name() < mesh->block(b)->name();
    });

    ptrdiff_t offset = 0;
    auto      k_d    = k_cell->data();
    auto      b_d    = bid->data();
    for (const auto b : order) {
        auto            block = mesh->block(b);
        const real_t    k     = diffusion_for_block(block->name(), k_white, k_black);
        const ptrdiff_t ne    = block->n_elements();
        for (ptrdiff_t e = 0; e < ne; ++e, ++offset) {
            k_d[offset] = k;
            b_d[offset] = static_cast<smesh::i32>(b);
        }
    }
    if (offset != n_cells) {
        return SFEM_FAILURE;
    }

    auto out = smesh::Output::create(mesh, output_dir);
    if (out->write_elemental("k", k_cell) != SMESH_SUCCESS) {
        return SFEM_FAILURE;
    }
    if (out->write_elemental("block_id", bid) != SMESH_SUCCESS) {
        return SFEM_FAILURE;
    }
    return SFEM_SUCCESS;
}

}  // namespace

int solve_checkerboard_ssgmg(const std::shared_ptr<sfem::Communicator> &comm) {
    auto es = smesh::Env::read("SFEM_EXECUTION_SPACE", sfem::EXECUTION_SPACE_HOST);

    const int  SFEM_ELEMENT_REFINE_LEVEL = smesh::Env::read("SFEM_ELEMENT_REFINE_LEVEL", 4);
    const int  SFEM_BASE_RESOLUTION      = smesh::Env::read<int>("SFEM_BASE_RESOLUTION", 4);
    const auto SFEM_OUTPUT_DIR           = smesh::Env::read_string("SFEM_OUTPUT_DIR", "output_multiblock_ssgmg");
    const auto SFEM_OPERATOR             = smesh::Env::read_string("SFEM_OPERATOR", "Laplacian");

    geom_t Lx = 1;
    auto   m  = sfem::Mesh::create_hex8_checkerboard_cube(comm,
                                                       SFEM_BASE_RESOLUTION,
                                                       SFEM_BASE_RESOLUTION,
                                                       SFEM_BASE_RESOLUTION,
                                                       0,
                                                       0,
                                                       0,
                                                       Lx,
                                                       1,
                                                       1);

    m = smesh::to_semistructured(SFEM_ELEMENT_REFINE_LEVEL, m, true, false);

    auto fs = sfem::FunctionSpace::create(m, 1);
    auto f  = sfem::Function::create(fs);
    auto op = sfem::create_op(fs, SFEM_OPERATOR, es);
    op->initialize();
    const real_t k_white = smesh::Env::read("SFEM_DIFFUSION_WHITE", real_t(1));
    const real_t k_black = smesh::Env::read("SFEM_DIFFUSION_BLACK", real_t(10));
    if (SFEM_OPERATOR == "Laplacian") {
        op->set_value_in_block("white", "k", k_white);
        op->set_value_in_block("black", "k", k_black);
    }
    f->add_operator(op);

    auto bottom_ss = sfem::Sideset::create_from_selector(
            m, [](const geom_t /*x*/, const geom_t y, const geom_t /*z*/) -> bool { return y > -1e-5 && y < 1e-5; });

    auto right_ss = sfem::Sideset::create_from_selector(
            m, [=](const geom_t x, const geom_t /*y*/, const geom_t /*z*/) -> bool { return x > (Lx - 1e-5) && x < (Lx + 1e-5); });

    // Checkerboard skins span white+black; merge SS nodesets (plural extract is HEX8-only).
    sfem::DirichletConditions::Condition left{.nodeset = nodeset_from_sidesets(m, bottom_ss), .value = -1, .component = 0};
    sfem::DirichletConditions::Condition right{.nodeset = nodeset_from_sidesets(m, right_ss), .value = 1, .component = 0};

    auto conds = sfem::create_dirichlet_conditions(fs, {left, right}, es);
    f->add_constraint(conds);

    auto x   = sfem::create_buffer<real_t>(fs->n_dofs(), f->execution_space());
    auto rhs = sfem::create_buffer<real_t>(fs->n_dofs(), f->execution_space());
    f->apply_constraints(x->data());
    f->apply_constraints(rhs->data());

    auto mg = sfem::create_ssgmg(f, f->execution_space());
    if (!mg) {
        fprintf(stderr, "create_ssgmg failed\n");
        return SFEM_FAILURE;
    }
    mg->set_max_it(smesh::Env::read("SFEM_MG_MAX_IT", 40));
    mg->set_atol(smesh::Env::read("SFEM_MG_ATOL", real_t(1e-10)));
    if (mg->apply(rhs->data(), x->data()) != SFEM_SUCCESS) {
        fprintf(stderr, "ssgmg apply failed\n");
        return SFEM_FAILURE;
    }

    auto A  = sfem::create_linear_operator(sfem::op_type::MATRIX_FREE, f, nullptr, f->execution_space());
    auto ax = sfem::create_buffer<real_t>(fs->n_dofs(), f->execution_space());
    A->apply(x->data(), ax->data());

    const ptrdiff_t n_owned  = fs->n_owned_dofs();
    real_t          local_r2 = 0;
    real_t          local_b2 = 0;
    auto            rh  = smesh::to_host(rhs);
    auto            axh = smesh::to_host(ax);
    for (ptrdiff_t i = 0; i < n_owned; ++i) {
        const real_t r = rh->data()[i] - axh->data()[i];
        local_r2 += r * r;
        local_b2 += rh->data()[i] * rh->data()[i];
    }
    const real_t abs_res = std::sqrt(comm->sum(local_r2));
    const real_t rhs_nrm = std::sqrt(comm->sum(local_b2));
    const real_t rel_res = abs_res / (rhs_nrm + real_t(1e-16));
    if (comm->rank() == 0) {
        printf("ssgmg residual abs %g rel %g (||rhs|| %g)\n", (double)abs_res, (double)rel_res, (double)rhs_nrm);
    }

    const real_t abs_tol = sizeof(real_t) == sizeof(double) ? real_t(1e-6) : real_t(1e-4);
    if (!(abs_res < abs_tol || rel_res < abs_tol)) {
        if (comm->rank() == 0) {
            fprintf(stderr, "ssgmg residual too large\n");
        }
        return SFEM_FAILURE;
    }

    const smesh::Path output_dir(SFEM_OUTPUT_DIR);
    smesh::create_directory(output_dir);

    auto viz_mesh = m;
    if (fs->has_semi_structured_mesh()) {
        viz_mesh = smesh::sshex_to_hex8(m);
        if (!viz_mesh) {
            fprintf(stderr, "sshex_to_hex8 failed\n");
            return SFEM_FAILURE;
        }
    }
    if (viz_mesh->write(output_dir / "mesh") != SMESH_SUCCESS) {
        fprintf(stderr, "mesh write failed\n");
        return SFEM_FAILURE;
    }
    if (write_block_cell_fields(viz_mesh, output_dir, k_white, k_black) != SFEM_SUCCESS) {
        fprintf(stderr, "cell field write failed\n");
        return SFEM_FAILURE;
    }

    auto output = f->output();
    output->set_output_dir(output_dir);
#ifdef SFEM_ENABLE_CUDA
    if (x->mem_space() == sfem::MEMORY_SPACE_DEVICE) {
        output->write("x", smesh::to_host(x)->data());
        output->write("rhs", smesh::to_host(rhs)->data());
    } else
#endif
    {
        output->write("x", x->data());
        output->write("rhs", rhs->data());
    }

    printf("Wrote %s (dofs %ld)\n", output_dir.c_str(), (long)fs->n_dofs());
    return SFEM_SUCCESS;
}

int main(int argc, char *argv[]) {
    auto ctx = sfem::initialize(argc, argv);
    return solve_checkerboard_ssgmg(ctx->communicator());
}

