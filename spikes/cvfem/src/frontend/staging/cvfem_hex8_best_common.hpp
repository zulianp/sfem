#ifndef CVFEM_HEX8_BEST_COMMON_HPP
#define CVFEM_HEX8_BEST_COMMON_HPP

// See the note at the top of cvfem_hex8_ns_core.hpp: this benchmark family and the
// solver core share sixteen names and disagree on the physics behind several of them.
#if defined(CVFEM_HEX8_NS_CORE_HPP)
#error "cvfem_hex8_best_*.hpp (benchmark) and cvfem_hex8_ns_core.hpp (solver) define the same names with different physics -- include one family per translation unit."
#endif

// Shared foundation for the HEX8 CVFEM Navier-Stokes assembly/apply layouts.
//
// Holds everything the layouts have in common: the mesh and matrix containers,
// the pack decomposition, the phase timers behind --breakdown, the per-thread
// scratch allocator, and the element gather/scatter primitives. Each layout then
// lives in its own header:
//
//   cvfem_hex8_best_atomic.hpp   element sweep, #pragma omp atomic per entry
//   cvfem_hex8_best_packed.hpp   pack-local buffer, reduced into the global one
//   cvfem_hex8_best_colored.hpp  colored pack sweep straight into the global one
//   cvfem_hex8_best_store.hpp    write-once packed assembly (packed variant)
//
// This header is self-contained: it pulls in the smesh/SFEM headers, the HEX8
// element kernels, and the scalar/index types the layouts are written against.

// The kernels trace through their own spelling, which is nothing unless a translation
// unit says what it means. See kernels/cvfem_phases.hpp.
#include "sfem_aliases.hpp"
#define CVFEM_TRACE_SCOPE(name) SFEM_TRACE_SCOPE(name)

#include "smesh_mesh.hpp"
#include "smesh_mesh_reorder.hpp"
#include "smesh_packed_mesh.hpp"
#include "smesh_buffer.hpp"
#include "sfem_BSR.hpp"

#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif




#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

using scalar_t = double;
// THE INDEX TYPES THE KERNELS NAME, declared here for the same reason scalar_t above is.
//
// DESIGN.md requires src/kernels/ to be "header only self-contained code with templated types"
// and to carry no library dependency. Those headers spelled smesh::idx_t and smesh::count_t,
// which named a library and a namespace; they now name idx_t and count_t unqualified, exactly as
// they have always named scalar_t, and the family header that includes them supplies the alias.
// That is what lets the vectorisation gate compile the lane kernels against nothing but the
// standard library, and it is what will let a caller with a different index width instantiate
// them without editing them.
using idx_t   = smesh::idx_t;
using count_t = smesh::count_t;
using geom_t  = smesh::geom_t;

// After the aliases above, not before: the flags header pulls in the microkernels, and those
// name scalar_t, idx_t and count_t on the contract that whoever includes them declared those
// first. Putting this at the top of the file breaks that contract, and the error it produces --
// "use of undeclared identifier 'scalar_t'" from inside the microkernel header -- points at the
// kernel rather than at the include that was too early.
#include "frontend/staging/cvfem_bnd_gather_build.hpp"
#include "kernels/microkernels/hex8/affine/cvfem_hex8_affine_geometry.hpp"
#include "kernels/microkernels/hex8/affine/cvfem_hex8_ns_upwind_affine.hpp"
#include "kernels/microkernels/hex8/isoparametric/cvfem_hex8_ns_upwind_isoparam.hpp"
#include "kernels/cvfem_bdf.hpp"
#include "kernels/cvfem_hex8_flags.hpp"
#include "kernels/cvfem_kernel_kind.hpp"  // KernelKind
#include "kernels/cvfem_scatter.hpp"      // atomic_add, find_bsr_slot, reset_residual, MIN
#include "kernels/cvfem_hex8_element_gather.hpp"   // Hex8RcConfig, Hex8Extras

static constexpr int N_FIELDS = 4;

#include "kernels/cvfem_portability.hpp"

// Needs scalar_t and N_FIELDS above.
#include "frontend/staging/cvfem_hex8_pack_common.hpp"

#include "kernels/microkernels/hex8/cvfem_hex8_ns_upwind_kernels.hpp"
#include "kernels/microkernels/hex8/affine/generated/cvfem_hex8_ns_upwind_sympy_affine.hpp"
#include "kernels/microkernels/hex8/isoparametric/generated/cvfem_hex8_ns_upwind_sympy_isoparam.hpp"

// The boundary sub-control-surface terms, for --boundary. The benchmark closes no control
// volumes by default and this header contributes nothing unless a face mask is supplied,
// so including it costs an unused function per kernel and nothing at run time. It also
// supplies cvfem_hex8_grad_scalar, which the nodal pressure gradient below needs, so it
// must precede cvfem_hex8_pack_helpers.hpp.
#include "kernels/microkernels/hex8/cvfem_hex8_boundary_scs.hpp"

#include "frontend/staging/cvfem_hex8_pack_helpers.hpp"

// The rowwise and facewise CSE arrangements lost the saturated evaluation (see
// subpar/README.md) and were moved to subpar/. Building with -DCVFEM_ENABLE_SUBPAR puts
// them back so `--kernel sympy_row|sympy_face` can be measured again.
//
// Without the option they are rejected by name during CLI validation, which is what
// makes the stubs below unreachable. They exist so that the layout dispatch chains --
// which are long else-if ladders in four headers -- keep compiling untouched, rather
// than being carved up with preprocessor branches. Reaching one is a bug, and says so.
#ifdef CVFEM_ENABLE_SUBPAR
#include "cvfem_hex8_ns_upwind_sympy_subpar.hpp"
#else
#define CVFEM_SUBPAR_STUB(name)                                                        \
    template <typename... Args>                                                        \
    static SFEM_INLINE void name(Args &&...) {                                         \
        std::fprintf(stderr,                                                           \
                     "%s was moved to subpar/: it is not the fastest kernel in any "    \
                     "measured configuration. Rebuild with -DCVFEM_ENABLE_SUBPAR to "  \
                     "use it.\n",                                                      \
                     #name);                                                           \
        std::abort();                                                                  \
    }
CVFEM_SUBPAR_STUB(cvfem_hex8_ns_upwind_sympy_jacobian_add_bsr_slots_rowwise)
CVFEM_SUBPAR_STUB(cvfem_hex8_ns_upwind_sympy_jacobian_add_bsr_slots_facewise)
CVFEM_SUBPAR_STUB(cvfem_hex8_ns_upwind_sympy_jacobian_add_local_slots_rowwise)
CVFEM_SUBPAR_STUB(cvfem_hex8_ns_upwind_sympy_jacobian_add_local_slots_facewise)
// The generated affine residual, measured on Grace at 8,586,756 dof
// (perf/campaign_generated_arms.csv): it gives up 21.6% on the packed layout and ties
// sumfact on the atomic one.
//
// The six generated Jacobian-action arrangements -- 0.37x to 0.57x of the hand-written
// atomic action in the same campaign -- need no stub any more. They had one while the
// atomic sweep still carried an `if constexpr` arm per arrangement; that selector is gone
// (DESIGN.md's correction: one micro-kernel per kernel), so nothing names them outside
// subpar/ and the default build does not have to declare them to compile.
CVFEM_SUBPAR_STUB(cvfem_hex8_ns_upwind_sympy_residual)
// The eight generated deferred-correction kernels, which were the packed default. Grace job
// 4981920 measured the hand-written lane-blocked kernel faster in all seven pairs (0.775x to
// 0.976x); see subpar/README.md.
CVFEM_SUBPAR_STUB(cvfem_hex8_ns_upwind_sympy_residual_defcor_lim0_simd)
CVFEM_SUBPAR_STUB(cvfem_hex8_ns_upwind_sympy_residual_defcor_lim1_simd)
CVFEM_SUBPAR_STUB(cvfem_hex8_ns_upwind_sympy_residual_defcor_lim2_simd)
CVFEM_SUBPAR_STUB(cvfem_hex8_ns_upwind_sympy_residual_defcor_lim3_simd)
CVFEM_SUBPAR_STUB(cvfem_hex8_ns_upwind_sympy_residual_defcor_rc_lim0_simd)
CVFEM_SUBPAR_STUB(cvfem_hex8_ns_upwind_sympy_residual_defcor_rc_lim1_simd)
CVFEM_SUBPAR_STUB(cvfem_hex8_ns_upwind_sympy_residual_defcor_rc_lim2_simd)
CVFEM_SUBPAR_STUB(cvfem_hex8_ns_upwind_sympy_residual_defcor_rc_lim3_simd)
// The packed higher-order residual's SCALAR sweep, now subpar/cvfem_hex8_packed_defcor_scalar.hpp.
CVFEM_SUBPAR_STUB(apply_residual_packed_defcor_scalar)
#undef CVFEM_SUBPAR_STUB
#endif

// KernelKind moved to kernels/cvfem_kernel_kind.hpp, so that the sweeps can be TEMPLATED on
// it rather than branching on it per element. parse_kernel and the predicates below stay:
// turning --kernel into a value is front-end work.

static KernelKind parse_kernel(const std::string &name) {
    if (name == "current") return KernelKind::Current;
    if (name == "fd") return KernelKind::Fd;
    if (name == "sumfact") return KernelKind::Sumfact;
    if (name == "sympy") return KernelKind::Sympy;
    if (name == "sympy_block") return KernelKind::SympyBlock;
    if (name == "sympy_row") return KernelKind::SympyRow;
    if (name == "sympy_face") return KernelKind::SympyFace;
    if (name == "split") return KernelKind::Split;
    // The generated Jacobian-action arrangements. These exist only for the action -- the
    // residual and the assembly have their own arrangements under the names above -- so
    // the driver refuses them for any other operation rather than mapping them onto
    // something that did run.
    if (name == "sympy_action") return KernelKind::SympyAction;
    if (name == "sympy_action_node") return KernelKind::SympyActionNode;
    if (name == "sympy_action_comp") return KernelKind::SympyActionComp;
    if (name == "sympy_action_face") return KernelKind::SympyActionFace;
    if (name == "sympy_action_geom") return KernelKind::SympyActionGeom;
    if (name == "sympy_action_geomface") return KernelKind::SympyActionGeomFace;
    return KernelKind::Sumfact;
}

// kernel_uses_sympy_residual moved to kernels/cvfem_kernel_kind.hpp: it is a pure function of
// the enum, and the sweeps need it as a compile-time test on their template argument.

static bool kernel_is_valid(const std::string &name) {
    return name == "current" || name == "fd" || name == "sumfact" || name == "sympy" || name == "sympy_block" ||
           name == "sympy_row" || name == "sympy_face" || name == "split" || name == "sympy_action" ||
           name == "sympy_action_node" || name == "sympy_action_comp" ||
           name == "sympy_action_face" || name == "sympy_action_geom" ||
           name == "sympy_action_geomface";
}

// The three generated Jacobian-action CSE arrangements, which are the only kernels the
// action dispatches on. Everything else ignores --kernel for that operation.
static bool kernel_is_action_only(const KernelKind k) {
    return k == KernelKind::SympyAction || k == KernelKind::SympyActionNode ||
           k == KernelKind::SympyActionComp || k == KernelKind::SympyActionFace ||
           k == KernelKind::SympyActionGeom || k == KernelKind::SympyActionGeomFace;
}

enum class GeomKind { Affine, Isoparam };

static GeomKind parse_geom(const std::string &name) {
    if (name == "isoparam") return GeomKind::Isoparam;
    return GeomKind::Affine;
}

static constexpr scalar_t CVFEM_HEX8_UNIT_CUBE[CVFEM_HEX8_N_NODES][3] = {
        {0, 0, 0},
        {1, 0, 0},
        {1, 1, 0},
        {0, 1, 0},
        {0, 0, 1},
        {1, 0, 1},
        {1, 1, 1},
        {0, 1, 1}};

struct MeshData {
    std::shared_ptr<smesh::Mesh> mesh;
    ptrdiff_t                    nnodes{0};
    ptrdiff_t                    nelements{0};
    smesh::idx_t               **elems{nullptr};
    smesh::geom_t              **points{nullptr};

    std::vector<scalar_t> ux, uy, uz, p;
    std::vector<scalar_t> rx, ry, rz, rc;
    std::vector<scalar_t> jacobian_adjugate[9];
    std::vector<scalar_t> jacobian_determinant;

    // THE SAME ADJUGATE AND DETERMINANT, AS RAW POINTERS, so that a kernel can be handed the
    // affine geometry without being handed this object. jacobian_adjugate is nine separate
    // vectors, so there is no single pointer into it and the alternative was building a
    // nine-element array of pointers at thirty-one call sites. PackedData already carries raw
    // pointers beside its vectors for the same reason.
    //
    // Filled by precompute_affine_geometry, which is the only thing that sizes those vectors.
    // They are valid until it runs again, which is the same lifetime the vectors themselves have.
    const scalar_t *adj_ptr[9]{};
    const scalar_t *det_ptr{nullptr};

    // --- optional physics, off unless the corresponding option is passed ---------------
    //
    // The benchmark measures the element kernel in isolation by default, which is why
    // these are empty and zero rather than always present: a residual with no Rhie-Chow
    // and no closed control volumes is a smaller, faster operator than the one the solver
    // runs, and isolating it is the point of this driver. See the note at the top of
    // cvfem_hex8_ns_core.hpp on why the two families differ in physics and not only in
    // layout.
    //
    // --rhie-chow fills pgx/pgy/pgz and sets rhie_chow_scale, at which point
    // cvfem_hex8_rhie_chow_active() starts returning true inside the element kernels and
    // the pressure-pressure coupling appears. --boundary fills face_mask and Lx/Ly/Lz, at
    // which point the boundary sub-control-surface terms close the boundary control
    // volumes. Both together are the operator the solver actually evaluates.
    std::vector<scalar_t> pgx, pgy, pgz;   // nodal pressure gradient (Rhie-Chow)
    scalar_t              rhie_chow_scale{0};

    // The same reconstruction applied to the Krylov DIRECTION's pressure, which only the
    // Jacobian action reads. It is kept separate from pgx/pgy/pgz because the two have
    // opposite lifetimes: pg is a function of the state and is hoisted across a whole
    // Krylov solve, whereas qg changes with every direction and so cannot be hoisted out
    // of anything. That asymmetry is the point of measuring it -- see the note on
    // --rhie-chow-jac in the benchmark driver.
    std::vector<scalar_t> qgx, qgy, qgz;   // nodal gradient of the direction's pressure

    // The Rhie-Chow coefficient, hoisted out of the element loop -- twelve values per
    // element, one per sub-control surface, rebuilt by cvfem_hex8_build_rc_coeff only when
    // rho, mu, the scale or the mesh change. See Hex8RhieChowPack::coeff for why it is not
    // computed where it is used.
    // ONE array, element-major: coefficient s of element e is rc_coeff[e * N_SCS + s]. See the
    // twin in cvfem_hex8_ns_core.hpp for the measurement -- it was twelve separate arrays, the
    // twelve-stream explanation for the Rhie-Chow arm's excess DRAM traffic was tested here and
    // falsified, and this shape is kept for simplicity rather than for speed.
    std::vector<scalar_t> rc_coeff;
    // The coefficient's velocity-sensitivity weight, per surface. Purely geometric plus the two
    // Rhie-Chow uniforms, so the Jacobian face loop need not recompute three dot products and a
    // division for it. Same shape and lifetime as rc_coeff.
    std::vector<scalar_t> rc_w;
    scalar_t              rc_coeff_rho{0}, rc_coeff_mu{0}, rc_coeff_scale{0};
    // The coefficient carries the advecting velocity now, so rho, mu and the scale no
    // longer span everything it depends on. state_stamp is bumped by whoever moves the
    // state; the cache compares it and rebuilds once per Newton step rather than once per
    // matvec. A harness that sets the state once and never moves it leaves the stamp at
    // zero and gets the same single build it always got.
    uint64_t              state_stamp{0};
    uint64_t              rc_coeff_stamp{~0ull};

    std::vector<uint8_t>  face_mask;       // per element, bits 0..5 = the six CVFEM faces

    // The effective boundary face mask -- see cvfem_hex8_build_face_mask_eff. Built once so
    // the boundary sweeps can skip the elements that have no boundary face at all, which on
    // any refined mesh is nearly all of them.
    std::vector<uint8_t> face_mask_eff;
    bool                 face_mask_eff_valid{false};
    scalar_t             face_mask_eff_lx{0}, face_mask_eff_ly{0}, face_mask_eff_lz{0};
    // The subset of elements face_mask_eff marks, compacted so the boundary sweeps are
    // load balanced rather than merely short.
    std::vector<ptrdiff_t> bnd_elems;
    // The boundary shell's node gather map -- see cvfem_hex8_build_bnd_gather. It lets the
    // closure be summed per node in a fixed order instead of scattered atomically, which
    // is what makes the operator bit-reproducible across thread counts.
    std::vector<ptrdiff_t>   bnd_gather_ptr;
    std::vector<int32_t>     bnd_gather_slot;
    std::vector<smesh::idx_t> bnd_gather_dest;
    std::vector<scalar_t>    bnd_r;
    ptrdiff_t                bnd_gather_n_bnd{-1};
    bool                     bnd_gather_valid{false};
    scalar_t              Lx{0}, Ly{0}, Lz{0};

    // --transient dt makes the operator unsteady. In a control-volume scheme the mass
    // matrix IS the control volume, so the term is diagonal: it needs a nodal volume and
    // one velocity history level per BDF order, and nothing else. dt <= 0 means steady,
    // which is the default and is what every existing measurement was taken with.
    scalar_t              dt{0};
    // The PREVIOUS step size, for variable-step BDF2. Zero means none recorded, which is the
    // uniform case and every existing caller. Mirrors the solver core's MeshData so the two
    // bdf_coeffs stay the same function -- the benchmark is the performance gate for these
    // kernels and it measuring a different scheme would make that gate meaningless.
    scalar_t              dt_prev{0};
    int                   bdf_order{1};
    std::vector<scalar_t> u_prev, u_prev2;  // 3 * nnodes, interleaved
    std::vector<scalar_t> node_vol;
    // The reconstruction's denominator: 1 / sum_{e in i} |det J_e|, one scalar per node.
    // Pure geometry, so it is constant for the whole solve -- and it was being rebuilt from
    // scratch on every matvec, with its own heap allocation and one atomic per node per
    // element, inside the pass that is 69% of that matvec. Keyed on the mesh and the
    // geometry rule, because the affine and isoparametric sweeps evaluate det differently.
    std::vector<scalar_t> grad_w_inv;
    // The partially assembled element tangent: five scalars per sub-control surface, sixty
    // per element, SoA by (surface, component). The complete dependence of the Jacobian
    // action on the Newton iterate -- see Hex8TangentPack. Built once per Newton step and
    // read on every matvec, so it is invalidated by the STATE changing, which no key made
    // of rho, mu and the mesh can see: pa_valid is cleared by whoever moves the state.
    std::vector<scalar_t> pa_tangent;
    scalar_t              pa_rho{0}, pa_mu{0}, pa_scale{0}, pa_ueps{0};
    ptrdiff_t             pa_nelements{0};
    bool                  pa_valid{false};
    int                   grad_w_isoparam{-1};
    ptrdiff_t             grad_w_nelements{0};
};

struct BSR4 {
    std::shared_ptr<smesh::Mesh::NodeToNodeGraph> graph;
    const smesh::count_t                         *rowptr{nullptr};
    const smesh::idx_t                           *colidx{nullptr};
    smesh::SharedBuffer<scalar_t>                 values;
    std::vector<smesh::count_t>                   element_slots;
    ptrdiff_t                                     nnz{0};
};


static int threads_active() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

static double wall_time() {
    return std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

// ---- lightweight phase breakdown (enabled with --breakdown) -------------------
static int g_breakdown = 0;
static int g_dense_flush = 0;  // --dense-flush: stage ke densely, then flush 64 contiguous blocks
static int g_kernel_only = 0;  // --kernel-only: element kernel writes to a dense stack buffer (no scatter)
static int g_identity_slots[64];
// PH_QGRAD is not a phase of an element sweep like the others: it is the separate
// reconstruction pass that the exact Rhie-Chow Jacobian action runs before the sweep, and it
// is the largest single item in that matvec -- 5.06 ms of 7.34 at 4,121,204 dof on Grace,
// 69%. It was invisible here because the breakdown covered packed *assembly* and the
// *colored* matvec but neither packed action, so the pass that dominates had no phase at all.
enum PhaseId {
    PH_ZERO = 0,
    PH_LOCAL_MEMSET,
    PH_GATHER,
    PH_KERNEL,
    PH_LOCAL_TO_GLOBAL,
    PH_GHOST,
    PH_QGRAD,
    PH_N
};
static const char *const g_phase_name[PH_N] = {"zero_global",
                                               "zero_local",
                                               "gather_u",
                                               "element_kernel",
                                               "local_to_global",
                                               "ghost_reduce",
                                               "qgrad"};
static double g_phase[PH_N] = {0};
struct PhaseAcc {
    double t[PH_N] = {0};
    void   flush() {
        if (!g_breakdown) return;
#pragma omp critical
        for (int i = 0; i < PH_N; ++i) g_phase[i] += t[i];
    }
};
static SFEM_INLINE double phase_now() { return g_breakdown ? wall_time() : 0.0; }
static void phase_reset() {
    for (int i = 0; i < PH_N; ++i) g_phase[i] = 0;
}
static void phase_report(const char *tag, const int repeat, const int nthreads) {
    if (!g_breakdown) return;
    // PH_QGRAD is WALL time for a whole pass; every other phase is thread time summed over
    // the team. Putting them in one percentage column would divide a wall-clock number by a
    // thread-summed total and report the largest pass in the matvec as a rounding error, so
    // it is kept out of the total and printed on its own terms.
    double total = 0;
    for (int i = 0; i < PH_N; ++i)
        if (i != PH_QGRAD) total += g_phase[i];
    std::printf("  breakdown_%s (ms/call, summed over %d threads):\n", tag, nthreads);
    for (int i = 0; i < PH_N; ++i) {
        if (g_phase[i] == 0 || i == PH_QGRAD) continue;
        std::printf("    %-16s %8.3f  (%5.1f%%)\n",
                    g_phase_name[i],
                    1000.0 * g_phase[i] / double(repeat),
                    100.0 * g_phase[i] / total);
    }
    std::printf("    %-16s %8.3f\n", "TOTAL", 1000.0 * total / double(repeat));
    if (g_phase[PH_QGRAD] != 0)
        std::printf("    %-16s %8.3f  (wall, one pass, not thread-summed)\n",
                    g_phase_name[PH_QGRAD], 1000.0 * g_phase[PH_QGRAD] / double(repeat));
}


static void fill_fields(MeshData &d) {
    d.ux.resize(d.nnodes);
    d.uy.resize(d.nnodes);
    d.uz.resize(d.nnodes);
    d.p.resize(d.nnodes);
    d.rx.assign(d.nnodes, 0.0);
    d.ry.assign(d.nnodes, 0.0);
    d.rz.assign(d.nnodes, 0.0);
    d.rc.assign(d.nnodes, 0.0);

    const auto *const x = d.points[0];
    const auto *const y = d.points[1];
    const auto *const z = d.points[2];

#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        d.ux[i] = 1.0 + 0.3 * x[i] - 0.2 * y[i] + 0.1 * z[i];
        d.uy[i] = -0.4 + 0.2 * x[i] + 0.5 * y[i] - 0.15 * z[i];
        d.uz[i] = 0.2 - 0.1 * x[i] + 0.25 * y[i] + 0.4 * z[i];
        d.p[i]  = 1.0 + 0.1 * x[i] + 0.2 * y[i] - 0.05 * z[i];
    }
}

// Takes the arrays and the count, not the mesh: it is called from inside the standard-layout
// sweeps, where a MeshData parameter is what keeps src/kernels/ dependent on this header.



static BSR4 make_bsr4(const std::shared_ptr<smesh::Mesh> &mesh) {
    BSR4 b;
    b.graph  = mesh->node_to_node_graph();
    b.rowptr = b.graph->rowptr()->data();
    b.colidx = b.graph->colidx()->data();
    b.nnz    = b.graph->nnz();
    b.values = smesh::create_host_buffer<scalar_t>((size_t)b.nnz * 16);
    return b;
}

// The assembled values, down-cast to single precision, for the SpMV storage comparison.
//
// h_bsr_spmv separates storage from compute -- BSR<R, C, TStorage, T> up-converts each entry in
// bsr_spmv_block_fma_4x4 -- so a float-storage operator still accumulates in double. What it
// halves is the matrix traffic, which at these sizes is what the apply is bound by, and that is
// the whole subject of the measurement.
//
// A separate buffer rather than a templated BSR4 on purpose: this family already collides with the
// solver's on MeshData and BSR4, and adding a second BSR4 type would widen a collision that
// tests/cvfem_ns_op_isolation.cpp exists to keep narrow.
static smesh::SharedBuffer<float> bsr4_values_f32(const BSR4 &b) {
    auto              out = smesh::create_host_buffer<float>((size_t)b.nnz * 16);
    float *const      dst = out->data();
    const scalar_t *const src = b.values->data();
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < b.nnz * 16; ++i) dst[i] = (float)src[i];
    return out;
}

static void zero_bsr4(BSR4 &b) {
    const double t0 = phase_now();
    cvfem_zero_scalars(b.values->data(), b.nnz * 16);
    if (g_breakdown) g_phase[PH_ZERO] += wall_time() - t0;
}


// The bench is driven by flags rather than by the environment, so it has no smesh::Env to
// read through; this is for the one escape hatch that has to match the solver's spelling.
static int cvfem_env_flag(const char *const name) {
    const char *const v = std::getenv(name);
    return v && *v && *v != '0' ? 1 : 0;
}







static SFEM_INLINE void bsr4_add16(scalar_t *const SFEM_RESTRICT dst, const scalar_t *const SFEM_RESTRICT src) {
#pragma omp simd
    for (int i = 0; i < 16; ++i) dst[i] += src[i];
}

static void precompute_element_bsr_slots(const MeshData &d, BSR4 &b) {
    b.element_slots.resize((size_t)d.nelements * CVFEM_HEX8_N_NODES * CVFEM_HEX8_N_NODES);

    smesh::idx_t **const SFEM_RESTRICT elems = d.elems;
    smesh::count_t *const SFEM_RESTRICT slots = b.element_slots.data();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const smesh::idx_t row = elems[a][e];
            for (int bnode = 0; bnode < CVFEM_HEX8_N_NODES; ++bnode) {
                const smesh::idx_t col = elems[bnode][e];
                slots[(size_t)e * 64 + a * 8 + bnode] = find_bsr_slot(b.rowptr, b.colidx, row, col);
            }
        }
    }
}


// Take the arrays, not the mesh: both are called from inside the standard-layout sweeps in
// src/kernels/, where naming MeshData is what keeps that directory dependent on this one. The
// sources are suffixed because the destinations already own the short names.


// ---- boundary closure as a post-pass -----------------------------------------
//
// The boundary sub-control-surface terms are applied as their own element sweep rather
// than inside each layout's kernel loop. That is the solver's own arrangement
// (apply_boundary_scs_residual in cvfem_hex8_ns_core.hpp) and it is the right one here for
// the same reason apply_body_force gives: the term touches no sub-control-surface flux, so
// one pass covers the atomic, packed, colored and store sweeps at once instead of being
// threaded into four of them -- and the packed and colored sweeps are SIMD over a pack,
// which a per-element face test does not fit.
//
// It costs an extra pass over the elements, which is why it is only run with --boundary.
// The early-out on an all-zero element contribution keeps interior elements to a gather
// and a compare.
static SFEM_NOINLINE void apply_boundary_scs_residual_pass(MeshData &d, const scalar_t rho, const scalar_t mu,
                                                           const int isoparam) {
    if (d.face_mask.empty()) return;
    // Over the listed boundary elements, not over the mesh with a filter -- the shell
    // clusters into a few static chunks, so filtering cuts the work without cutting the
    // wall time. See cvfem_hex8_compact_boundary_elems.
    cvfem_hex8_build_face_mask_eff(d);
    // The gather is the default: scattering the closure atomically made the operator's
    // result depend on thread timing. SFEM_BND_ATOMIC=1 restores the old scatter, as a
    // measurement escape hatch rather than a supported mode.
    static const int bnd_atomic = cvfem_env_flag("SFEM_BND_ATOMIC");
    if (!bnd_atomic) cvfem_hex8_build_bnd_gather(d);
    const ptrdiff_t n_bnd = (ptrdiff_t)d.bnd_elems.size();
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < n_bnd; ++i) {
        const ptrdiff_t e     = d.bnd_elems[(size_t)i];
        const int       fmask = (int)d.face_mask_eff[(size_t)e];
        scalar_t x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8], r[CVFEM_HEX8_N_DOF];
        gather_element_coords(d.elems, d.points, e, x, y, z);
        gather_element_fields(d.elems, d.ux.data(), d.uy.data(), d.uz.data(), d.p.data(), e, ux, uy, uz, p);
        std::memset(r, 0, sizeof(r));
        scalar_t adj[9], det = scalar_t(0);
        if (!isoparam) load_hex8_adj(d.adj_ptr, d.det_ptr, e, adj, &det);
        if (isoparam)
            boundary_scs_add_residual<true>(rho, mu, (const scalar_t *)nullptr, det, d.Lx, d.Ly, d.Lz, x, y, z,
                                            ux, uy, uz, p, r, fmask, 0);
        else
            boundary_scs_add_residual<false>(rho, mu, adj, det, d.Lx, d.Ly, d.Lz, x, y, z,
                                             ux, uy, uz, p, r, fmask, 0);
        // The caller picks the instantiation, so the staged-or-atomic choice is made once per
        // pass rather than once per boundary element.
        if (bnd_atomic)
            cvfem_hex8_bnd_commit<true>(d.elems, d.bnd_r.data(), i, e, r, d.rx.data(),
                                        d.ry.data(), d.rz.data(), d.rc.data());
        else
            cvfem_hex8_bnd_commit<false>(d.elems, d.bnd_r.data(), i, e, r, d.rx.data(),
                                         d.ry.data(), d.rz.data(), d.rc.data());
    }
    if (!bnd_atomic) cvfem_hex8_drain_boundary_soa(d, d.rx.data(), d.ry.data(), d.rz.data(), d.rc.data());
}

static SFEM_NOINLINE void apply_boundary_scs_jacobian_action_pass(MeshData &d, const scalar_t rho, const scalar_t mu,
                                                                  const int                           isoparam,
                                                                  const scalar_t *const SFEM_RESTRICT dir,
                                                                  scalar_t *const SFEM_RESTRICT       jv) {
    if (d.face_mask.empty()) return;
    // Over the listed boundary elements, not over the mesh with a filter -- the shell
    // clusters into a few static chunks, so filtering cuts the work without cutting the
    // wall time. See cvfem_hex8_compact_boundary_elems.
    cvfem_hex8_build_face_mask_eff(d);
    // The gather is the default: scattering the closure atomically made the operator's
    // result depend on thread timing. SFEM_BND_ATOMIC=1 restores the old scatter, as a
    // measurement escape hatch rather than a supported mode.
    static const int bnd_atomic = cvfem_env_flag("SFEM_BND_ATOMIC");
    if (!bnd_atomic) cvfem_hex8_build_bnd_gather(d);
    const ptrdiff_t n_bnd = (ptrdiff_t)d.bnd_elems.size();
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < n_bnd; ++i) {
        const ptrdiff_t e     = d.bnd_elems[(size_t)i];
        const int       fmask = (int)d.face_mask_eff[(size_t)e];
        scalar_t x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8], vx[8], vy[8], vz[8], q[8], r[CVFEM_HEX8_N_DOF];
        gather_element_coords(d.elems, d.points, e, x, y, z);
        gather_element_fields(d.elems, d.ux.data(), d.uy.data(), d.uz.data(), d.p.data(), e, ux, uy, uz, p);
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const ptrdiff_t g = (ptrdiff_t)d.elems[a][e] * N_FIELDS;
            vx[a]             = dir[g + 0];
            vy[a]             = dir[g + 1];
            vz[a]             = dir[g + 2];
            q[a]              = dir[g + 3];
        }
        std::memset(r, 0, sizeof(r));
        scalar_t adj[9], det = scalar_t(0);
        if (!isoparam) load_hex8_adj(d.adj_ptr, d.det_ptr, e, adj, &det);
        if (isoparam)
            boundary_scs_add_jacobian_action<true>(rho, mu, (const scalar_t *)nullptr, det, d.Lx, d.Ly, d.Lz, x, y, z,
                                                   ux, uy, uz, vx, vy, vz, q, r, fmask, 0);
        else
            boundary_scs_add_jacobian_action<false>(rho, mu, adj, det, d.Lx, d.Ly, d.Lz, x, y, z,
                                                    ux, uy, uz, vx, vy, vz, q, r, fmask, 0);
        if (bnd_atomic)
            cvfem_hex8_bnd_commit_interleaved<true>(d.elems, d.bnd_r.data(), i, e, r, jv);
        else
            cvfem_hex8_bnd_commit_interleaved<false>(d.elems, d.bnd_r.data(), i, e, r, jv);
    }
    if (!bnd_atomic) cvfem_hex8_drain_boundary_interleaved(d, jv);
}

// The assembled counterpart of the two passes above, and it exists for the same reason
// they do: the closure touches no sub-control-surface flux, so one sweep over the boundary
// shell covers every kernel and every layout at once. Threading it into each assembly
// entry point instead would mean nine copies -- and would still be impossible for the
// generated ones, whose element kernel is emitted text. That is why `--boundary` with
// `--assemble` used to be refused for every kernel but `sumfact` on affine geometry.
//
// It scatters through b.element_slots, the global element-to-slot map, which is the same
// whatever layout assembled the interior. `--kernel split` therefore gets the closure in
// its nonlinear half by construction: the linear half is a memcpy restore that runs first,
// and this pass runs after both.
static SFEM_NOINLINE void assemble_boundary_scs_jacobian_pass(MeshData      &d,
                                                              BSR4          &b,
                                                              const scalar_t rho,
                                                              const scalar_t mu,
                                                              const int      isoparam) {
    if (d.face_mask.empty()) return;
    cvfem_hex8_build_face_mask_eff(d);
    scalar_t *const SFEM_RESTRICT             values = b.values->data();
    // The global element-to-slot map exists for the layouts that scatter through it. The
    // pack-based ones do not build it -- they carry their own per-pack maps -- and it is
    // 64 counts per element, a gigabyte at the sizes this driver runs, so building it just
    // for this pass would be a poor trade. The slots for one element are found instead,
    // and only for the boundary shell: a binary search per entry over 6/n of the mesh,
    // once per assembly, against a term that is already the smaller part of the cost.
    const smesh::count_t *const SFEM_RESTRICT slots  = b.element_slots.empty() ? nullptr : b.element_slots.data();
    const ptrdiff_t                           n_bnd  = (ptrdiff_t)d.bnd_elems.size();
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < n_bnd; ++i) {
        const ptrdiff_t e     = d.bnd_elems[(size_t)i];
        const int       fmask = (int)d.face_mask_eff[(size_t)e];
        scalar_t        x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8];
        gather_element_coords(d.elems, d.points, e, x, y, z);
        gather_element_fields(d.elems, d.ux.data(), d.uy.data(), d.uz.data(), d.p.data(), e, ux, uy, uz, p);
        scalar_t adj[9], det = scalar_t(0);
        if (!isoparam) load_hex8_adj(d.adj_ptr, d.det_ptr, e, adj, &det);
        smesh::count_t              found[64];
        const smesh::count_t *const esl = slots ? slots + (size_t)e * 64 : found;
        if (!slots)
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a)
                for (int bn = 0; bn < CVFEM_HEX8_N_NODES; ++bn)
                    found[a * 8 + bn] = find_bsr_slot(b.rowptr, b.colidx, d.elems[a][e], d.elems[bn][e]);
        if (isoparam)
            boundary_scs_add_jacobian<true, true>(rho, mu, (const scalar_t *)nullptr, det, d.Lx, d.Ly, d.Lz,
                                                  x, y, z, ux, uy, uz, esl, values, fmask, 0);
        else
            boundary_scs_add_jacobian<true, false>(rho, mu, adj, det, d.Lx, d.Ly, d.Lz,
                                                   x, y, z, ux, uy, uz, esl, values, fmask, 0);
        (void)p;
    }
}

// ---------------------------------------------------------------- the transient term
//
// A third post-pass, for the third reason the other two are post-passes: in a
// control-volume scheme the mass matrix IS the control volume. The momentum equation
// integrated over CV_i carries d/dt of (rho u_i V_i), so the term is diagonal in the nodes
// and touches no sub-control-surface flux -- one pass covers the atomic, packed, colored
// and store sweeps at once. There is no consistent mass matrix to assemble and none should
// be introduced; an FEM mass matrix here would be a different discretisation.
//
// This mirrors apply_transient / apply_transient_action / build_node_volume in
// cvfem_hex8_ns_core.hpp. The two families cannot include each other -- each #errors on the
// other's guard -- so this is a duplicate, and a duplicate drifts. What stops it is
// tests/cvfem_bench_transient_test, which pins the node volume against a closed form (the
// volumes sum to the domain volume, and on a uniform box every interior node has exactly
// h^3) and the action against a central difference of the residual term.

static void build_node_volume(const MeshData &d, std::vector<scalar_t> &node_vol) {
    node_vol.assign((size_t)d.nnodes, scalar_t(0));
    // jacobian_determinant is precomputed for affine geometry only, so evaluate it here
    // when it is absent rather than reading an empty array.
    const bool have_det = (ptrdiff_t)d.jacobian_determinant.size() >= d.nelements;
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t det;
        if (have_det) {
            det = d.jacobian_determinant[(size_t)e];
        } else {
            scalar_t x[8], y[8], z[8], adj[9];
            gather_element_coords(d.elems, d.points, e, x, y, z);
            cvfem_hex8_geom_at(x, y, z, scalar_t(0.5), scalar_t(0.5), scalar_t(0.5), adj, &det);
        }
        const scalar_t v = std::fabs(det) / scalar_t(8);
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) node_vol[d.elems[a][e]] += v;
    }
}

// The staging read for the BDF coefficients. The rule is in kernels/cvfem_bdf.hpp;
// what belongs here is which members hold the step sizes and the history.
static BdfCoeffs bdf_coeffs(const MeshData &d) {
    return cvfem_bdf_coeffs(d.bdf_order, d.dt, d.dt_prev,
                            (ptrdiff_t)d.u_prev2.size() == 3 * d.nnodes);
}

static SFEM_NOINLINE void apply_transient_pass(MeshData &d, const scalar_t rho) {
    if (d.dt <= scalar_t(0)) return;
    if ((ptrdiff_t)d.u_prev.size() != 3 * d.nnodes) return;
    if ((ptrdiff_t)d.node_vol.size() != d.nnodes) build_node_volume(d, d.node_vol);
    const BdfCoeffs c   = bdf_coeffs(d);
    const scalar_t  inv = scalar_t(1) / d.dt;
    const bool      two = c.order == 2;
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        const scalar_t w   = rho * d.node_vol[(size_t)i] * inv;
        const size_t   k   = (size_t)i * 3;
        const scalar_t p2x = two ? d.u_prev2[k + 0] : scalar_t(0);
        const scalar_t p2y = two ? d.u_prev2[k + 1] : scalar_t(0);
        const scalar_t p2z = two ? d.u_prev2[k + 2] : scalar_t(0);
        d.rx[(size_t)i] += w * (c.a0 * d.ux[(size_t)i] + c.a1 * d.u_prev[k + 0] + c.a2 * p2x);
        d.ry[(size_t)i] += w * (c.a0 * d.uy[(size_t)i] + c.a1 * d.u_prev[k + 1] + c.a2 * p2y);
        d.rz[(size_t)i] += w * (c.a0 * d.uz[(size_t)i] + c.a1 * d.u_prev[k + 2] + c.a2 * p2z);
    }
}

// The derivative is rho V a0 / dt on each velocity diagonal entry and nothing on pressure,
// so the saddle-point structure is untouched -- which is why the block diagonal still needs
// Rhie-Chow to have an invertible pressure entry no matter how small the timestep is.
//
// The history is deliberately NOT required here: the coefficient is rho a0 / dt whatever
// u^n holds. The solver learned this the hard way -- requiring it gave every coarse level,
// which never receives a history, a steady Jacobian.
static scalar_t transient_diag_weight(const MeshData &d, const scalar_t rho) {
    if (d.dt <= scalar_t(0)) return scalar_t(0);
    if ((ptrdiff_t)d.u_prev.size() == 3 * d.nnodes) return bdf_coeffs(d).a0 * rho / d.dt;
    return (d.bdf_order >= 2 ? scalar_t(1.5) : scalar_t(1)) * rho / d.dt;
}

static SFEM_NOINLINE void apply_transient_action_pass(MeshData                           &d,
                                                      const scalar_t                      rho,
                                                      const scalar_t *const SFEM_RESTRICT dir,
                                                      scalar_t *const SFEM_RESTRICT       jv) {
    const scalar_t a = transient_diag_weight(d, rho);
    if (a == scalar_t(0)) return;
    if ((ptrdiff_t)d.node_vol.size() != d.nnodes) build_node_volume(d, d.node_vol);
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        const scalar_t w = a * d.node_vol[(size_t)i];
        for (int c = 0; c < 3; ++c) jv[(ptrdiff_t)i * N_FIELDS + c] += w * dir[(ptrdiff_t)i * N_FIELDS + c];
    }
}

// The same term on the assembled matrix and on the block diagonal. Added here rather than
// in the element kernels for the reason above, and consistent with the residual by
// construction because both read transient_diag_weight.
static SFEM_NOINLINE void assemble_transient_diag_pass(MeshData &d, const scalar_t rho, BSR4 &b) {
    const scalar_t a = transient_diag_weight(d, rho);
    if (a == scalar_t(0)) return;
    if ((ptrdiff_t)d.node_vol.size() != d.nnodes) build_node_volume(d, d.node_vol);
    scalar_t *const SFEM_RESTRICT values = b.values->data();
#pragma omp parallel for schedule(static)
    for (ptrdiff_t r = 0; r < d.nnodes; ++r) {
        const scalar_t w = a * d.node_vol[(size_t)r];
        for (smesh::count_t j = b.rowptr[r]; j < b.rowptr[r + 1]; ++j) {
            if (b.colidx[j] != (smesh::idx_t)r) continue;
            for (int c = 0; c < 3; ++c) values[(ptrdiff_t)j * 16 + c * 4 + c] += w;
        }
    }
}

static SFEM_NOINLINE void assemble_diag_transient_pass(MeshData &d, const scalar_t rho,
                                                       std::vector<scalar_t> &diag) {
    const scalar_t a = transient_diag_weight(d, rho);
    if (a == scalar_t(0)) return;
    if ((ptrdiff_t)d.node_vol.size() != d.nnodes) build_node_volume(d, d.node_vol);
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        const scalar_t w = a * d.node_vol[(size_t)i];
        for (int c = 0; c < 3; ++c) diag[(size_t)i * 16 + (size_t)c * 4 + (size_t)c] += w;
    }
}

// ---- optional terms: --rhie-chow and --boundary ------------------------------
//
// Both are off by default, and the default path must stay exactly as fast as it was --
// the throughput regression gate compares against numbers measured without them. So the
// coordinate and pressure-gradient gathers they need sit behind a flag hoisted out of the
// element loop rather than being done unconditionally. This is the same `with_pg` idiom
// the solver uses in cvfem_hex8_ns_packed.hpp.
//
// With both off, `rc` keeps its null pointers, cvfem_hex8_rhie_chow_active() is false and
// the element kernel's Rhie-Chow branch folds away, `fmask` stays 0 and the caller skips
// the boundary term entirely. Nothing is gathered and nothing is written.
// The Rhie-Chow time-scale configuration for this mesh. Mirrors the solver's
// cvfem_hex8_rc_config_for in cvfem_hex8_ns_core.hpp; the two MeshData types never appear in
// one translation unit, so the name is shared deliberately.
inline Hex8RcConfig cvfem_hex8_rc_config_for(const MeshData &d) {
    return cvfem_hex8_rc_config(d.rhie_chow_scale, transient_diag_weight(d, scalar_t(1)));
}

// Hex8Extras moved to kernels/cvfem_hex8_flags.hpp; the resolution from a mesh stays here.
inline Hex8Extras cvfem_hex8_extras_of(const MeshData &d) {
    Hex8Extras x;
    x.with_rc  = !d.pgx.empty() && d.rhie_chow_scale != scalar_t(0);
    x.with_bnd = !d.face_mask.empty();
    x.with_qg  = !d.pgx.empty() && d.rhie_chow_scale != scalar_t(0) && !d.qgx.empty();
    x.rcfg     = cvfem_hex8_rc_config_for(d);
    return x;
}

// Per-element scratch for the above. Declared inside the element loop; `rc` points into
// this object, so it must outlive the kernel call -- which it does, being a local.


static SFEM_INLINE void gather_hex8_simd_from_pack(pack_idx_t **const SFEM_RESTRICT   elems,
                                                   const scalar_t *const SFEM_RESTRICT pack_u,
                                                   const scalar_t *const *const SFEM_RESTRICT adj_ptr,
                                                   const scalar_t *const SFEM_RESTRICT        det_ptr,
                                                   const ptrdiff_t                     begin,
                                                   const int                           nlanes,
                                                   Hex8InputPack                      &in,
                                                   scalar_t *const SFEM_RESTRICT       cof0,
                                                   scalar_t *const SFEM_RESTRICT       cof1,
                                                   scalar_t *const SFEM_RESTRICT       cof2,
                                                   scalar_t *const SFEM_RESTRICT       cof3,
                                                   scalar_t *const SFEM_RESTRICT       cof4,
                                                   scalar_t *const SFEM_RESTRICT       cof5,
                                                   scalar_t *const SFEM_RESTRICT       cof6,
                                                   scalar_t *const SFEM_RESTRICT       cof7,
                                                   scalar_t *const SFEM_RESTRICT       cof8,
                                                   scalar_t *const SFEM_RESTRICT       det) {
    gather_hex8_adj_soa(adj_ptr, det_ptr, begin, nlanes, cof0, cof1, cof2, cof3, cof4, cof5, cof6, cof7, cof8, det);
    for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane) {
        if (lane < nlanes) {
            const ptrdiff_t e = begin + lane;
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                const scalar_t *const SFEM_RESTRICT u = pack_u + (ptrdiff_t)elems[a][e] * N_FIELDS;
                in.ux[a][lane]                        = u[0];
                in.uy[a][lane]                        = u[1];
                in.uz[a][lane]                        = u[2];
                in.p[a][lane]                         = u[3];
            }
        } else {
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                in.ux[a][lane] = in.uy[a][lane] = in.uz[a][lane] = in.p[a][lane] = scalar_t(0);
            }
        }
    }
}


static SFEM_INLINE void gather_hex8_action_simd_from_pack(pack_idx_t **const SFEM_RESTRICT   elems,
                                                          const scalar_t *const SFEM_RESTRICT pack_u,
                                                          const scalar_t *const SFEM_RESTRICT pack_dir,
                                                          const scalar_t *const *const SFEM_RESTRICT adj_ptr,
                                                          const scalar_t *const SFEM_RESTRICT        det_ptr,
                                                          const ptrdiff_t                     begin,
                                                          const int                           nlanes,
                                                          Hex8InputPack                      &u,
                                                          Hex8InputPack                      &du,
                                                          scalar_t *const SFEM_RESTRICT       cof0,
                                                          scalar_t *const SFEM_RESTRICT       cof1,
                                                          scalar_t *const SFEM_RESTRICT       cof2,
                                                          scalar_t *const SFEM_RESTRICT       cof3,
                                                          scalar_t *const SFEM_RESTRICT       cof4,
                                                          scalar_t *const SFEM_RESTRICT       cof5,
                                                          scalar_t *const SFEM_RESTRICT       cof6,
                                                          scalar_t *const SFEM_RESTRICT       cof7,
                                                          scalar_t *const SFEM_RESTRICT       cof8,
                                                          scalar_t *const SFEM_RESTRICT       det) {
    gather_hex8_simd_from_pack(elems, pack_u, adj_ptr, det_ptr, begin, nlanes, u, cof0, cof1, cof2, cof3, cof4, cof5, cof6, cof7, cof8, det);
    for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane) {
        if (lane < nlanes) {
            const ptrdiff_t e = begin + lane;
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                const scalar_t *const SFEM_RESTRICT dvec = pack_dir + (ptrdiff_t)elems[a][e] * N_FIELDS;
                du.ux[a][lane]                           = dvec[0];
                du.uy[a][lane]                           = dvec[1];
                du.uz[a][lane]                           = dvec[2];
                du.p[a][lane]                            = dvec[3];
            }
        } else {
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                du.ux[a][lane] = du.uy[a][lane] = du.uz[a][lane] = du.p[a][lane] = scalar_t(0);
            }
        }
    }
}

// Takes the arrays, not the staging objects: it is called from inside the pack sweeps, and a
// PackedData or MeshData parameter here is a staging dependency in src/kernels/, which DESIGN.md
// does not allow there.
static SFEM_INLINE void fill_pack_xyz(const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
                                      geom_t **const SFEM_RESTRICT       points,
                                      const ptrdiff_t                    pack,
                                      const ptrdiff_t                    n_contiguous,
                                      const ptrdiff_t                    n_ghost,
                                      const smesh::idx_t *const SFEM_RESTRICT ghosts,
                                      scalar_t *const SFEM_RESTRICT      pack_x,
                                      scalar_t *const SFEM_RESTRICT      pack_y,
                                      scalar_t *const SFEM_RESTRICT      pack_z) {
    const auto *const px    = points[0];
    const auto *const py    = points[1];
    const auto *const pz    = points[2];
    const ptrdiff_t   owned = owned_nodes_ptr[pack];
    for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
        const ptrdiff_t g = owned + k;
        pack_x[k]         = scalar_t(px[g]);
        pack_y[k]         = scalar_t(py[g]);
        pack_z[k]         = scalar_t(pz[g]);
    }
    for (ptrdiff_t k = 0; k < n_ghost; ++k) {
        const smesh::idx_t g = ghosts[k];
        pack_x[n_contiguous + k] = scalar_t(px[g]);
        pack_y[n_contiguous + k] = scalar_t(py[g]);
        pack_z[n_contiguous + k] = scalar_t(pz[g]);
    }
}

static SFEM_INLINE void gather_hex8_isoparam_simd_from_pack(pack_idx_t **const SFEM_RESTRICT     elems,
                                                            const scalar_t *const SFEM_RESTRICT pack_u,
                                                            const scalar_t *const SFEM_RESTRICT pack_x,
                                                            const scalar_t *const SFEM_RESTRICT pack_y,
                                                            const scalar_t *const SFEM_RESTRICT pack_z,
                                                            const ptrdiff_t                     begin,
                                                            const int                           nlanes,
                                                            Hex8InputPack                      &in,
                                                            Hex8CoordPack                      &xyz) {
    for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane) {
        if (lane < nlanes) {
            const ptrdiff_t e = begin + lane;
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                const pack_idx_t                    loc = elems[a][e];
                const scalar_t *const SFEM_RESTRICT u   = pack_u + (ptrdiff_t)loc * N_FIELDS;
                in.ux[a][lane]                          = u[0];
                in.uy[a][lane]                          = u[1];
                in.uz[a][lane]                          = u[2];
                in.p[a][lane]                           = u[3];
                xyz.x[a][lane]                          = pack_x[loc];
                xyz.y[a][lane]                          = pack_y[loc];
                xyz.z[a][lane]                          = pack_z[loc];
            }
        } else {
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                in.ux[a][lane] = in.uy[a][lane] = in.uz[a][lane] = in.p[a][lane] = scalar_t(0);
                xyz.x[a][lane]                                                   = CVFEM_HEX8_UNIT_CUBE[a][0];
                xyz.y[a][lane]                                                   = CVFEM_HEX8_UNIT_CUBE[a][1];
                xyz.z[a][lane]                                                   = CVFEM_HEX8_UNIT_CUBE[a][2];
            }
        }
    }
}

static SFEM_INLINE void gather_hex8_isoparam_action_simd_from_pack(pack_idx_t **const SFEM_RESTRICT     elems,
                                                                   const scalar_t *const SFEM_RESTRICT pack_u,
                                                                   const scalar_t *const SFEM_RESTRICT pack_dir,
                                                                   const scalar_t *const SFEM_RESTRICT pack_x,
                                                                   const scalar_t *const SFEM_RESTRICT pack_y,
                                                                   const scalar_t *const SFEM_RESTRICT pack_z,
                                                                   const ptrdiff_t                     begin,
                                                                   const int                           nlanes,
                                                                   Hex8InputPack                      &u,
                                                                   Hex8InputPack                      &du,
                                                                   Hex8CoordPack                      &xyz) {
    for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane) {
        if (lane < nlanes) {
            const ptrdiff_t e = begin + lane;
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                const pack_idx_t                    loc  = elems[a][e];
                const scalar_t *const SFEM_RESTRICT usrc = pack_u + (ptrdiff_t)loc * N_FIELDS;
                const scalar_t *const SFEM_RESTRICT dsrc = pack_dir + (ptrdiff_t)loc * N_FIELDS;
                u.ux[a][lane]                            = usrc[0];
                u.uy[a][lane]                            = usrc[1];
                u.uz[a][lane]                            = usrc[2];
                u.p[a][lane]                             = usrc[3];
                du.ux[a][lane]                           = dsrc[0];
                du.uy[a][lane]                           = dsrc[1];
                du.uz[a][lane]                           = dsrc[2];
                du.p[a][lane]                            = dsrc[3];
                xyz.x[a][lane]                           = pack_x[loc];
                xyz.y[a][lane]                           = pack_y[loc];
                xyz.z[a][lane]                           = pack_z[loc];
            }
        } else {
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                u.ux[a][lane] = u.uy[a][lane] = u.uz[a][lane] = u.p[a][lane] = scalar_t(0);
                du.ux[a][lane] = du.uy[a][lane] = du.uz[a][lane] = du.p[a][lane] = scalar_t(0);
                xyz.x[a][lane]                                                   = CVFEM_HEX8_UNIT_CUBE[a][0];
                xyz.y[a][lane]                                                   = CVFEM_HEX8_UNIT_CUBE[a][1];
                xyz.z[a][lane]                                                   = CVFEM_HEX8_UNIT_CUBE[a][2];
            }
        }
    }
}

static SFEM_INLINE void gather_hex8_coords_from_pack(pack_idx_t **const SFEM_RESTRICT     elems,
                                                     const scalar_t *const SFEM_RESTRICT pack_x,
                                                     const scalar_t *const SFEM_RESTRICT pack_y,
                                                     const scalar_t *const SFEM_RESTRICT pack_z,
                                                     const ptrdiff_t                     e,
                                                     scalar_t *const SFEM_RESTRICT       x,
                                                     scalar_t *const SFEM_RESTRICT       y,
                                                     scalar_t *const SFEM_RESTRICT       z) {
    for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
        const pack_idx_t loc = elems[a][e];
        x[a]                 = pack_x[loc];
        y[a]                 = pack_y[loc];
        z[a]                 = pack_z[loc];
    }
}

// ---------------------------------------------------------------------------
// Pack staging
// ---------------------------------------------------------------------------

// Copy a pack's nodal fields into an interleaved pack-local buffer. Indexing the
// element kernels through pack-local ids turns four scattered global reads per
// node into one contiguous read, which is why the packed and colored layouts both
// stage through this buffer rather than gathering from d.ux/uy/uz/p directly.
// Takes the arrays, not the staging objects: it is called from inside the pack sweeps, and a
// PackedData or MeshData parameter here is a staging dependency in src/kernels/, which DESIGN.md
// does not allow there.
static SFEM_INLINE void fill_pack_fields(const ptrdiff_t *const SFEM_RESTRICT    owned_nodes_ptr,
                                         const scalar_t *const SFEM_RESTRICT     ux,
                                         const scalar_t *const SFEM_RESTRICT     uy,
                                         const scalar_t *const SFEM_RESTRICT     uz,
                                         const scalar_t *const SFEM_RESTRICT     pr,
                                         const ptrdiff_t                         pack,
                                         const ptrdiff_t                         n_contiguous,
                                         const ptrdiff_t                         n_ghost,
                                         const smesh::idx_t *const SFEM_RESTRICT ghosts,
                                         scalar_t *const SFEM_RESTRICT           pack_u) {
    const ptrdiff_t                     owned = owned_nodes_ptr[pack];
    for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
        scalar_t *const SFEM_RESTRICT dst = pack_u + k * N_FIELDS;
        const ptrdiff_t               g   = owned + k;
        dst[0]                            = ux[g];
        dst[1]                            = uy[g];
        dst[2]                            = uz[g];
        dst[3]                            = pr[g];
    }
    for (ptrdiff_t k = 0; k < n_ghost; ++k) {
        scalar_t *const SFEM_RESTRICT dst = pack_u + (n_contiguous + k) * N_FIELDS;
        const smesh::idx_t            g   = ghosts[k];
        dst[0]                            = ux[g];
        dst[1]                            = uy[g];
        dst[2]                            = uz[g];
        dst[3]                            = pr[g];
    }
}

// Same, for an already-interleaved global vector (a Krylov direction).
// Takes the arrays, not the staging objects: it is called from inside the pack sweeps, and a
// PackedData or MeshData parameter here is a staging dependency in src/kernels/, which DESIGN.md
// does not allow there.
static SFEM_INLINE void fill_pack_interleaved(const ptrdiff_t *const SFEM_RESTRICT    owned_nodes_ptr,
                                              const ptrdiff_t                         pack,
                                              const ptrdiff_t                         n_contiguous,
                                              const ptrdiff_t                         n_ghost,
                                              const smesh::idx_t *const SFEM_RESTRICT ghosts,
                                              const scalar_t *const SFEM_RESTRICT     src,
                                              scalar_t *const SFEM_RESTRICT           pack_v) {
    const ptrdiff_t owned = owned_nodes_ptr[pack];
    for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
        std::memcpy(pack_v + k * N_FIELDS, src + (owned + k) * N_FIELDS, N_FIELDS * sizeof(scalar_t));
    }
    for (ptrdiff_t k = 0; k < n_ghost; ++k) {
        std::memcpy(pack_v + (n_contiguous + k) * N_FIELDS,
                    src + (ptrdiff_t)ghosts[k] * N_FIELDS,
                    N_FIELDS * sizeof(scalar_t));
    }
}

// Pack-local node id -> global node id. The colored layout scatters into the
// global arrays directly, so it needs this to translate the pack-local element
// table it gathers through.
static SFEM_INLINE void fill_pack_l2g(const PackedData                       &p,
                                      const ptrdiff_t                         pack,
                                      const ptrdiff_t                         n_contiguous,
                                      const ptrdiff_t                         n_ghost,
                                      const smesh::idx_t *const SFEM_RESTRICT ghosts,
                                      smesh::idx_t *const SFEM_RESTRICT       l2g) {
    const ptrdiff_t owned = p.owned_nodes_ptr[pack];
    for (ptrdiff_t k = 0; k < n_contiguous; ++k) l2g[k] = (smesh::idx_t)(owned + k);
    for (ptrdiff_t k = 0; k < n_ghost; ++k) l2g[n_contiguous + k] = ghosts[k];
}

// ---------------------------------------------------------------------------
// Element matrix -> target matrix
// ---------------------------------------------------------------------------

// Flush a dense block-major element matrix ke[(i*8+k)*16 + c] into the target
// matrix: 64 contiguous 16-double adds instead of ~768 scattered scalar updates.
static SFEM_INLINE void hex8_blocks_to_slots(const int *const SFEM_RESTRICT      slots,
                                             const scalar_t *const SFEM_RESTRICT ke,
                                             scalar_t *const SFEM_RESTRICT       values) {
    for (int blk = 0; blk < 64; ++blk) {
        scalar_t *const SFEM_RESTRICT       dst = values + (ptrdiff_t)slots[blk] * 16;
        const scalar_t *const SFEM_RESTRICT src = ke + blk * 16;
#pragma omp simd
        for (int c = 0; c < 16; ++c) dst[c] += src[c];
    }
}

static SFEM_INLINE void hex8_local_slots_to_bsr4(const int *const SFEM_RESTRICT      slots,
                                                 const scalar_t *const SFEM_RESTRICT ke,
                                                 scalar_t *const SFEM_RESTRICT       values) {
    for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
        for (int bnode = 0; bnode < CVFEM_HEX8_N_NODES; ++bnode) {
            scalar_t *const SFEM_RESTRICT blk = values + (ptrdiff_t)slots[a * 8 + bnode] * 16;
            for (int rf = 0; rf < 4; ++rf) {
                for (int cf = 0; cf < 4; ++cf) {
                    blk[rf * 4 + cf] += ke[(a * 4 + rf) * CVFEM_HEX8_N_DOF + (bnode * 4 + cf)];
                }
            }
        }
    }
}

#endif  // CVFEM_HEX8_BEST_COMMON_HPP
