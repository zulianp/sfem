// bench_patch_assembly_cuda <mesh folder> [windows] [repeats]: the device assembly strategies of the
// Mooney-Rivlin Prony TET10 tangent on a TET10 mesh (bench_patch_assembly_cuda.hpp). Every block of the
// mesh is taken, as one connectivity.
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "bench_patch_assembly_cuda.hpp"
#include "sfem_API.hpp"
#include "smesh_mesh.hpp"

int main(int argc, char *argv[]) {
    auto context = sfem::initialize(argc, argv);
    if (argc < 2) {
        fprintf(stderr, "usage: %s <TET10 mesh folder> [windows] [repeats]\n", argv[0]);
        return EXIT_FAILURE;
    }
    auto comm = sfem::Communicator::world();
    auto mesh = sfem::Mesh::create_from_file(comm, smesh::Path(argv[1]));
    if (!mesh) return EXIT_FAILURE;
    auto fs    = sfem::FunctionSpace::create(mesh, 3);
    auto graph = fs->node_to_node_graph();

    const ptrdiff_t nnodes = mesh->n_nodes();
    ptrdiff_t       ne     = 0;
    for (size_t b = 0; b < mesh->n_blocks(); ++b) {
        if (mesh->block(b)->element_type() != smesh::TET10) {
            fprintf(stderr, "bench_patch_assembly: block %zu is not TET10\n", b);
            return EXIT_FAILURE;
        }
        ne += mesh->n_elements((block_idx_t)b);
    }
    // The blocks' connectivities as one.
    std::vector<std::vector<idx_t>> elements(10, std::vector<idx_t>(ne));
    ptrdiff_t                       offset = 0;
    for (size_t b = 0; b < mesh->n_blocks(); ++b) {
        const ptrdiff_t n = mesh->n_elements((block_idx_t)b);
        auto           *e = mesh->elements((block_idx_t)b)->data();
        for (int s = 0; s < 10; ++s)
            for (ptrdiff_t k = 0; k < n; ++k) elements[s][offset + k] = e[s][k];
        offset += n;
    }
    // Node -> elements, each node's elements ascending.
    std::vector<count_t>       n2e_ptr(nnodes + 1, 0);
    std::vector<element_idx_t> n2e_idx(10 * ne);
    for (int s = 0; s < 10; ++s)
        for (ptrdiff_t k = 0; k < ne; ++k) ++n2e_ptr[elements[s][k] + 1];
    for (ptrdiff_t i = 0; i < nnodes; ++i) n2e_ptr[i + 1] += n2e_ptr[i];
    std::vector<count_t> fill(n2e_ptr.begin(), n2e_ptr.end() - 1);
    for (ptrdiff_t k = 0; k < ne; ++k)
        for (int s = 0; s < 10; ++s) n2e_idx[fill[elements[s][k]]++] = (element_idx_t)k;

    std::vector<const idx_t *> eptr(10);
    for (int s = 0; s < 10; ++s) eptr[s] = elements[s].data();
    geom_t **const x = mesh->points()->data();
    const geom_t  *pptr[3] = {x[0], x[1], x[2]};

    PatchBenchInput in;
    in.nelements = ne;
    in.nnodes    = nnodes;
    in.elements  = eptr.data();
    in.points    = pptr;
    in.rowptr    = graph->rowptr()->data();
    in.colidx    = graph->colidx()->data();
    in.n2e_ptr   = n2e_ptr.data();
    in.n2e_idx   = n2e_idx.data();
    in.windows   = argc > 2 ? atoi(argv[2]) : 4;
    in.repeats   = argc > 3 ? atoi(argv[3]) : 5;
    in.patch     = !(getenv("SFEM_BENCH_PATCH") && atoi(getenv("SFEM_BENCH_PATCH")) == 0);
    printf("bench_patch_assembly %s: %ld TET10 elements, %ld nodes, %ld blocks (%.1f per row)\n", argv[1], (long)ne, (long)nnodes,
           (long)in.rowptr[nnodes], (double)in.rowptr[nnodes] / nnodes);
    return run_patch_bench(in);
}
