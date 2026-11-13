// The matrix-free apply of one operator on a cube, timed, on the device or the host.
//
// Environment:
//   SFEM_OP_NAME=LinearElasticity   the operator (sfem::create_op)
//   SFEM_ELEM_TYPE=HEX8             the cube's element type
//   SFEM_BASE_RESOLUTION=<n>        elements per side, default 8
//   SFEM_ELEMENT_REFINE_LEVEL=<L>   a semi-structured mesh of level L (> 1), default 0
//   SFEM_REPEAT=<n>                 timed applies after one untimed, default 10
//   SFEM_EXECUTION_SPACE=device     or host

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>

#include "sfem_API.hpp"
#include "sfem_Function.hpp"

#include "smesh_env.hpp"

int main(int argc, char *argv[]) {
    sfem::Context context(argc, argv);
    {
        const std::string name       = smesh::Env::read_string("SFEM_OP_NAME", "LinearElasticity");
        const auto        type       = smesh::type_from_string(smesh::Env::read_string("SFEM_ELEM_TYPE", "HEX8").c_str());
        const int         resolution = smesh::Env::read("SFEM_BASE_RESOLUTION", 8);
        const int         level      = smesh::Env::read("SFEM_ELEMENT_REFINE_LEVEL", 0);
        const int         repeat     = smesh::Env::read("SFEM_REPEAT", 10);
        const auto        es = smesh::execution_space_from_string(smesh::Env::read_string("SFEM_EXECUTION_SPACE", "device").c_str());

        auto mesh = sfem::Mesh::create_cube(
                context.communicator(), static_cast<smesh::ElemType>(type), resolution, resolution, resolution, 0, 0, 0, 1, 1, 1);
        if (level > 1) mesh = smesh::to_semistructured(level, mesh, true, false);
        auto fs = sfem::FunctionSpace::create(mesh, 3);
        auto op = sfem::create_op(fs, name.c_str(), es);
        if (!op) SFEM_ERROR("could not create %s\n", name.c_str());
        op->initialize();

        const ptrdiff_t ndofs = fs->n_dofs();
        auto            h     = sfem::create_host_buffer<real_t>(ndofs);
        for (ptrdiff_t i = 0; i < ndofs; ++i) h->data()[i] = std::sin(real_t(0.37) * real_t(i + 1));
        auto x = es == sfem::EXECUTION_SPACE_DEVICE ? smesh::to_device(h) : h;
        auto y = sfem::create_buffer<real_t>(ndofs, es);

        double best = std::numeric_limits<double>::max(), total = 0;
        for (int r = -1; r < repeat; ++r) {
            sfem::blas<real_t>(es)->zeros(ndofs, y->data());
            sfem::device_synchronize();
            const double start = smesh::time_seconds();
            if (op->apply(nullptr, x->data(), y->data()) != SFEM_SUCCESS) SFEM_ERROR("apply failed\n");
            sfem::device_synchronize();
            const double elapsed = smesh::time_seconds() - start;
            if (r < 0) continue;
            best = std::min(best, elapsed);
            total += elapsed;
        }

        const double norm = std::sqrt(sfem::blas<real_t>(es)->dot(ndofs, y->data(), y->data()));
        printf("op %s\nelement_type %s level %d\n#elements %ld\n#nodes %ld\n|y| %.17g\n",
               name.c_str(),
               type_to_string(mesh->element_type(0)),
               level,
               (long)mesh->n_elements(),
               (long)mesh->n_nodes(),
               norm);
        printf("apply best %.6g s mean %.6g s, %.4g Mdofs/s\n", best, total / repeat, 1e-6 * double(ndofs) / best);
    }
    return SFEM_SUCCESS;
}
