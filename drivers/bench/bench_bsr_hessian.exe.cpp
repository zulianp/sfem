// The BSR Hessian assembly of one operator on a cube, timed: the element kernel and the scatter
// into the node graph's 3x3 blocks, on the device or the host.
//
// Environment:
//   SFEM_OP=<yaml>                  the operator, e.g. "{type: GeneratedSaintVenantKirchhoff, mu: 1, lmbda: 2}"
//   SFEM_ELEM_TYPE=PROTEUS_HEX64    the cube's element type
//   SFEM_BASE_RESOLUTION=<n>        elements per side, default 10
//   SFEM_REPEAT=<n>                 timed assemblies after one untimed, default 5
//   SFEM_EXECUTION_SPACE=device     or host
//   SFEM_HIGH_ORDER=1               a semi-structured mesh (PROTEUS_*) as one Lagrange element per
//                                   macro element (the default), 0 as its micro elements

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>

#include "sfem_API.hpp"
#include "sfem_Function.hpp"

#include "smesh_env.hpp"

#include <ryml.hpp>
#include <ryml_std.hpp>

int main(int argc, char *argv[]) {
    sfem::Context context(argc, argv);
    {
        const std::string yaml       = smesh::Env::read_string("SFEM_OP", "{type: GeneratedSaintVenantKirchhoff, mu: 1, lmbda: 2}");
        const auto        type       = smesh::type_from_string(smesh::Env::read_string("SFEM_ELEM_TYPE", "PROTEUS_HEX64").c_str());
        const int         resolution = smesh::Env::read("SFEM_BASE_RESOLUTION", 10);
        const int         repeat     = smesh::Env::read("SFEM_REPEAT", 5);
        const auto        es = smesh::execution_space_from_string(smesh::Env::read_string("SFEM_EXECUTION_SPACE", "device").c_str());

        auto mesh = sfem::Mesh::create_cube(
                context.communicator(), static_cast<smesh::ElemType>(type), resolution, resolution, resolution, 0, 0, 0, 1, 1, 1);
        auto       fs   = sfem::FunctionSpace::create(mesh, 3);
        if (smesh::Env::read("SFEM_HIGH_ORDER", 1)) fs->set_semi_structured_basis(sfem::FunctionSpace::SemiStructuredBasis::HIGH_ORDER);
        ryml::Tree tree = ryml::parse_in_arena(ryml::to_csubstr(yaml));
        auto       op   = sfem::create_op_from_yaml(fs, tree.crootref(), es);
        if (!op) SFEM_ERROR("could not create %s\n", yaml.c_str());

        const ptrdiff_t ndofs = fs->n_dofs();
        auto            x     = sfem::create_host_buffer<real_t>(ndofs);
        {
            const geom_t *const *xyz = mesh->points()->data();
            for (ptrdiff_t i = 0; i < mesh->n_nodes(); ++i) {
                x->data()[i * 3 + 0] = real_t(0.05) * xyz[0][i] * xyz[1][i];
                x->data()[i * 3 + 1] = real_t(-0.03) * xyz[0][i] + real_t(0.02) * xyz[2][i];
                x->data()[i * 3 + 2] = real_t(0.04) * xyz[1][i] * xyz[2][i];
            }
        }

        auto       graph  = fs->element_node_graph();
        const auto nvals  = graph->nnz() * 9;
        auto       values = sfem::create_buffer<real_t>(nvals, es);
        auto       rowptr = es == sfem::EXECUTION_SPACE_DEVICE ? smesh::to_device(graph->rowptr()) : graph->rowptr();
        auto       colidx = es == sfem::EXECUTION_SPACE_DEVICE ? smesh::to_device(graph->colidx()) : graph->colidx();
        auto       u      = es == sfem::EXECUTION_SPACE_DEVICE ? smesh::to_device(x) : x;

        double best = std::numeric_limits<double>::max(), total = 0;
        for (int r = -1; r < repeat; ++r) {
            sfem::blas<real_t>(es)->zeros(nvals, values->data());
            sfem::device_synchronize();
            const double start = smesh::time_seconds();
            if (op->hessian_bsr(u->data(), rowptr->data(), colidx->data(), values->data()) != SFEM_SUCCESS) SFEM_ERROR("hessian_bsr failed\n");
            sfem::device_synchronize();
            const double elapsed = smesh::time_seconds() - start;
            if (r < 0) continue;
            best = std::min(best, elapsed);
            total += elapsed;
        }

        printf("op %s\nelement_type %s\n#elements %ld\n#nodes %ld\n#blocks %ld\n",
               yaml.c_str(),
               type_to_string(mesh->element_type(0)),
               (long)mesh->n_elements(),
               (long)mesh->n_nodes(),
               (long)graph->nnz());
        printf("hessian_bsr best %.6g s mean %.6g s, %.4g Melements/s\n",
               best,
               total / repeat,
               1e-6 * double(mesh->n_elements()) / best);
    }
    return SFEM_SUCCESS;
}
