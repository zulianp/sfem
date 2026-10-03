#include "smesh_mesh.hpp"
#include "smesh_semistructured.hpp"
#include "smesh_sideset.hpp"
#include "sfem_context.hpp"
#include <cstdio>
#include <vector>
#include <cmath>
using namespace smesh;

// Every assertion below used to print OK/FAIL/MISMATCH and then `return 0` regardless, so
// the ctest could not fail. verdict() records the outcome as well as printing it.
static int g_failures = 0;
static const char *verdict(const bool ok, const char *bad = "FAIL") {
    if (!ok) ++g_failures;
    return ok ? "OK" : bad;
}

int main(int argc, char **argv) {
    auto ctx = sfem::initialize(argc, argv);
    const ptrdiff_t nx = 40, ny = 8, nz = 4;      // step at x=1 (4 cells), y=1 (4 cells)
    auto m = Mesh::create_hex8_lshape(ctx->communicator(), nx, ny, nz, 10, 2, 1, 1, 1);
    if (!m) { std::printf("FAIL: generator returned null\n"); return 1; }

    const ptrdiff_t nxs = 4, nys = 4;
    const ptrdiff_t want_e = nx*ny*nz - nxs*nys*nz;
    const ptrdiff_t grid_n = (nx+1)*(ny+1)*(nz+1);
    // A node is orphaned iff every element touching it lies in the notch. The elements
    // touching node (xi,yi) are {xi-1,xi} x {yi-1,yi} clipped to range, so the condition is
    // xi <= nxs-1 and yi <= nys-1 -- that is nxs*nys*(nz+1) nodes, not (nxs-1)*(nys-1)*(nz+1).
    const ptrdiff_t want_n = grid_n - nxs*nys*(nz+1);
    std::printf("elements: %td  (closed form %td)  %s\n", m->n_elements(0), want_e,
                verdict(m->n_elements(0) == want_e, "MISMATCH"));
    std::printf("nodes:    %td  (closed form %td)  %s\n", m->n_nodes(), want_n,
                verdict(m->n_nodes() == want_n, "MISMATCH"));

    // no orphans, and no node inside the open notch
    std::vector<char> seen((size_t)m->n_nodes(), 0);
    auto el = m->block(0)->elements()->data();
    for (ptrdiff_t e = 0; e < m->n_elements(0); ++e)
        for (int a = 0; a < 8; ++a) seen[(size_t)el[a][e]] = 1;
    ptrdiff_t orph = 0; for (auto c : seen) orph += !c;
    std::printf("orphan nodes: %td  %s\n", orph, verdict(orph == 0));

    auto p = m->points()->data();
    ptrdiff_t inside = 0;
    for (ptrdiff_t i = 0; i < m->n_nodes(); ++i)
        if (p[0][i] < 1.0-1e-9 && p[1][i] < 1.0-1e-9) ++inside;
    std::printf("nodes strictly inside the notch: %td  %s\n", inside, verdict(inside == 0));

    // bounding box
    double lo[3]={1e30,1e30,1e30}, hi[3]={-1e30,-1e30,-1e30};
    for (int d=0; d<3; ++d) for (ptrdiff_t i=0;i<m->n_nodes();++i){
        lo[d]=std::min(lo[d],(double)p[d][i]); hi[d]=std::max(hi[d],(double)p[d][i]); }
    std::printf("bbox: [%g,%g] x [%g,%g] x [%g,%g]\n", lo[0],hi[0],lo[1],hi[1],lo[2],hi[2]);

    auto ss = to_semistructured(2, m, true, false);
    std::printf("to_semistructured(2): %s\n", verdict((bool)ss));
    if (ss) std::printf("  ss nodes %td  elements %td\n", ss->n_nodes(), ss->n_elements(0));

    auto skin = skin_sideset(m);
    // closed form: the L cross-section perimeter is 10+2+9+1+1+1 = ... count faces directly
    // x-faces: inlet 1*4(y in [1,2])*4 ; outlet 2*4 ; step vertical 1*4
    // y-faces: bottom (x in [1,10]) 36*4 ; top 40*4 ; step horizontal (x in [0,1]) 4*4
    // z-faces: 2 * (kept cells per layer) = 2 * (40*8 - 4*4)
    const ptrdiff_t f_inlet = 4*nz, f_outlet = 8*nz, f_stepv = 4*nz;
    const ptrdiff_t f_bot = 36*nz, f_top = 40*nz, f_steph = 4*nz;
    const ptrdiff_t f_z = 2*(nx*ny - nxs*nys);
    const ptrdiff_t want_f = f_inlet+f_outlet+f_stepv+f_bot+f_top+f_steph+f_z;
    std::printf("skin faces: %td  (closed form %td)  %s\n",
                skin ? (ptrdiff_t)skin->parent()->size() : -1, want_f,
                verdict(skin && (ptrdiff_t)skin->parent()->size() == want_f, "MISMATCH"));
    if (g_failures) {
        std::fprintf(stderr, "tests_lshape_check: %d check(s) failed\n", g_failures);
        return 1;
    }
    return 0;
}
