// Newton with the compressed quadrature-point store as the Krylov operator.
//
//   newton_qp case.yaml [--set key.path=value]... [--qp f64|f16|i8|i8md|f16md|eaf64|eaf16|ead32|ead16|eadi8|eadi8p|bsrdi8|bsrfi8|bsrd16|bsrf16|bsrui8|bsru16|bsroi8|bsro16]
//
// The case runs exactly as `playground` runs it.  With --qp, the material operator is wrapped
// before the run: the wrapper answers everything by delegating to the generated operator, except
// the inexact pair, which it implements with the spike's set-up and application kernels.  Run it
// with `solver.linear.format: INEXACT`; the other formats (BSR, MATRIX_FREE) are the references
// and need no --qp.  The element is the one this build of the spike was configured for.
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "sfem_API.hpp"
#include "sfem_ElasticityCase.hpp"
#include "sfem_Playground.hpp"
#include "sfem_Function.hpp"
#include "sfem_FunctionSpace.hpp"

#include "kernel_math.hpp"
#include "qp_setup.hpp"
#include "qp_apply.hpp"
#include "bsr_quant.hpp"

#include <sys/resource.h>

#ifndef LANE_VS
#define LANE_VS 16
#endif

namespace {

using El = qp::Element;
using qp::Layout;
using qp::Scheme;

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// What the wrapper needs from a store, whatever its encoding.
struct QpKernels {
    virtual ~QpKernels() = default;
    virtual int update(const real_t *const x) = 0;
    virtual int apply(const real_t *const h, real_t *const out) = 0;
    virtual size_t bytes() const = 0;
};

// The library's packed layout, in the shape `qp::apply_packed` reads.
struct PackedView {
    ptrdiff_t n_packs = 0, n_elements_per_pack = 0, max_nodes_per_pack = 0;
    ptrdiff_t n_ghost_entries = 0, n_ghost_reduce_rows = 0;
    const ptrdiff_t *owned_nodes_ptr = nullptr, *ghost_ptr = nullptr, *ghost_reduce_ptr = nullptr,
                    *ghost_reduce_idx = nullptr;
    const idx_t *ghost_idx = nullptr, *ghost_reduce_dest = nullptr;
    std::vector<uint16_t *> kernel_elements;  // pack-local, in the kernels' node order
    std::shared_ptr<void> keep;               // the packed mesh the pointers belong to
    mutable std::vector<double> ghost_buf;
};

struct Geometry {
    ptrdiff_t nelements = 0;
    std::unique_ptr<PackedView> packed;
    std::vector<idx_t *> kernel_elements;  // in the order the kernels are written against
    std::vector<std::vector<geom_t>> adj;  // 9 components, per element
    std::vector<geom_t> det;
    real_t mu = 1, lmbda = 1;
    std::shared_ptr<sfem::CRSGraph> graph;  // the block graph BSR assembles on (quantized BSR)
};

template <class St, qp::Form FORM = qp::Form::Lanes>
struct QpKernelsT final : QpKernels {
    using code_t = typename St::code_type;
    const Geometry &g;
    St st;
    explicit QpKernelsT(const Geometry &geometry) : g(geometry), st(geometry.nelements) {}

    int update(const real_t *const x) override {
        idx_t **elements = const_cast<idx_t **>(g.kernel_elements.data());
        return qp::qp_setup_impl<real_t, geom_t, St, St::VS>(
            g.nelements, elements, g.adj[0].data(), g.adj[1].data(), g.adj[2].data(), g.adj[3].data(),
            g.adj[4].data(), g.adj[5].data(), g.adj[6].data(), g.adj[7].data(), g.adj[8].data(), g.det.data(), g.lmbda,
            g.mu, 3, x, x + 1, x + 2, st);
    }
    int apply(const real_t *const h, real_t *const out) override {
        idx_t **elements = const_cast<idx_t **>(g.kernel_elements.data());
        using tau = std::conditional_t<std::is_same<code_t, double>::value, double, float>;
        if (g.packed) {
            const PackedView &pk = *g.packed;
            return qp::apply_packed<St, tau, FORM>(st, pk, const_cast<uint16_t **>(pk.kernel_elements.data()),
                                                   g.nelements, 3, h, h + 1, h + 2, 3, out, out + 1, out + 2,
                                                   pk.ghost_buf.data());
        }
        return qp::apply<St, tau, FORM>(st, g.nelements, elements, 3, h, h + 1, h + 2, 3, out, out + 1, out + 2);
    }
    size_t bytes() const override { return St::bytes_per_element() * (size_t)g.nelements; }
};

// Quantized BSR (bsr_quant.hpp): the generated operator's single-precision assembly on the
// space's block graph, as the library's BSR f32 tangent assembles it, encoded after it.  The f32
// values stay allocated between updates; the store's bytes are what the application reads.  The
// upper form encodes over the values (encode_in_place), so its codes live in that buffer.
template <typename code_t, qp::BsrForm F, typename col_t = idx_t>
struct BsrKernels final : QpKernels {
    std::shared_ptr<sfem::Op> op;
    std::shared_ptr<sfem::CRSGraph> graph;
    qp::QuantBsr<code_t, F, LANE_VS, col_t> q;
    std::vector<float> values;
    BsrKernels(const std::shared_ptr<sfem::Op> &o, const std::shared_ptr<sfem::CRSGraph> &g)
        : op(o), graph(g), q(g->n_nodes(), g->rowptr()->data(), g->colidx()->data()),
          values((size_t)g->nnz() * 9) {}

    int update(const real_t *const x) override {
        float *const v = values.data();
        const ptrdiff_t n = (ptrdiff_t)values.size();
#pragma omp parallel for
        for (ptrdiff_t k = 0; k < n; ++k) v[k] = 0;
        const count_t *const rp = graph->rowptr()->data();
        const idx_t *const ci = graph->colidx()->data();
        if (op->hessian_bsr_typed(x, rp, ci, smesh::SMESH_FLOAT32, v) != SFEM_SUCCESS) return SFEM_FAILURE;
        if constexpr (F == qp::BsrForm::Upper)
            q.encode_in_place(rp, ci, v);
        else
            q.encode(rp, ci, v);
        return SFEM_SUCCESS;
    }
    int apply(const real_t *const h, real_t *const out) override {
        q.template apply<float, true>(h, out);
        return SFEM_SUCCESS;
    }
    size_t bytes() const override { return q.bytes(); }
};

template <typename c, Scheme S>
using Q = qp::Store<El, c, S, Layout::Blocked, LANE_VS>;
template <typename c, qp::EaForm F>
using EA = qp::EaStore<El, c, F, LANE_VS>;

// The material operator, with the inexact pair replaced.
class QpOp final : public sfem::Op {
public:
    QpOp(const std::shared_ptr<sfem::Op> &inner, std::unique_ptr<Geometry> geometry, const std::string &variant)
        : inner_(inner), g_(std::move(geometry)) {
        using qp::EaForm;
        if (variant == "f64")
            k_ = std::make_unique<QpKernelsT<Q<double, Scheme::Plain>>>(*g_);
        else if (variant == "f16")
            k_ = std::make_unique<QpKernelsT<Q<half_t, Scheme::Plain>>>(*g_);
        else if (variant == "i8")
            k_ = std::make_unique<QpKernelsT<Q<int8_t, Scheme::Plain>>>(*g_);
        else if (variant == "i8md" && El::NV > 1)
            k_ = std::make_unique<QpKernelsT<Q<int8_t, Scheme::MeanDeviation>>>(*g_);
        else if (variant == "f16md" && El::NV > 1)
            k_ = std::make_unique<QpKernelsT<Q<half_t, Scheme::MeanDeviation>>>(*g_);
        // Element matrices: full (eaf*) and difference form (ead*).
        else if (variant == "eaf64")
            k_ = std::make_unique<QpKernelsT<EA<double, EaForm::Full>>>(*g_);
        else if (variant == "eaf16")
            k_ = std::make_unique<QpKernelsT<EA<half_t, EaForm::Full>>>(*g_);
        else if (variant == "ead32")
            k_ = std::make_unique<QpKernelsT<EA<float, EaForm::Difference>>>(*g_);
        else if (variant == "ead16")
            k_ = std::make_unique<QpKernelsT<EA<half_t, EaForm::Difference>>>(*g_);
        else if (variant == "eadi8")
            k_ = std::make_unique<QpKernelsT<EA<int8_t, EaForm::Difference>>>(*g_);
        else if (variant == "eadi8p")
            k_ = std::make_unique<QpKernelsT<EA<int8_t, EaForm::PairScale>>>(*g_);
        // Quantized BSR: difference (bsrd*) and full (bsrf*) form, int8 per block or f16.
        else if (variant == "bsrdi8")
            k_ = std::make_unique<BsrKernels<int8_t, qp::BsrForm::Difference>>(inner_, g_->graph);
        else if (variant == "bsrfi8")
            k_ = std::make_unique<BsrKernels<int8_t, qp::BsrForm::Full>>(inner_, g_->graph);
        else if (variant == "bsrd16")
            k_ = std::make_unique<BsrKernels<half_t, qp::BsrForm::Difference>>(inner_, g_->graph);
        else if (variant == "bsrf16")
            k_ = std::make_unique<BsrKernels<half_t, qp::BsrForm::Full>>(inner_, g_->graph);
        else if (variant == "bsrui8")
            k_ = std::make_unique<BsrKernels<int8_t, qp::BsrForm::Upper>>(inner_, g_->graph);
        else if (variant == "bsru16")
            k_ = std::make_unique<BsrKernels<half_t, qp::BsrForm::Upper>>(inner_, g_->graph);
        else if (variant == "bsroi8")
            k_ = std::make_unique<BsrKernels<int8_t, qp::BsrForm::Upper, uint16_t>>(inner_, g_->graph);
        else if (variant == "bsro16")
            k_ = std::make_unique<BsrKernels<half_t, qp::BsrForm::Upper, uint16_t>>(inner_, g_->graph);
        else
            SFEM_ERROR("newton_qp: unknown --qp variant '%s'\n", variant.c_str());
    }
    ~QpOp() override {
        printf("qp: %s store %.1f MB, %d updates %.3f s, %d applies %.3f s\n", El::name, k_->bytes() / 1048576.0,
               n_update_, t_update_, n_apply_, t_apply_);
    }

    bool inexact_supported() const override { return true; }
    int inexact_update(const real_t *const x) override {
        const double t0 = now();
        const int status = k_->update(x);
        t_update_ += now() - t0;
        ++n_update_;
        return status;
    }
    int inexact_apply(const real_t *const h, real_t *const out) override {
        const double t0 = now();
        const int status = k_->apply(h, out);
        t_apply_ += now() - t0;
        ++n_apply_;
        return status;
    }

    // Everything else is the generated operator's.
    const char *name() const override { return inner_->name(); }
    bool is_linear() const override { return inner_->is_linear(); }
    int initialize(const std::vector<std::string> &b) override { return inner_->initialize(b); }
    int update(const real_t *const x) override { return inner_->update(x); }
    int update(const real_t *const SFEM_RESTRICT a, const real_t *const SFEM_RESTRICT b) override { return inner_->update(a, b); }
    int hessian_crs(const real_t *const x, const count_t *const r, const idx_t *const c, real_t *const v) override {
        return inner_->hessian_crs(x, r, c, v);
    }
    int hessian_crs_sym(const real_t *const x, const count_t *const r, const idx_t *const c, real_t *const d,
                        real_t *const o) override {
        return inner_->hessian_crs_sym(x, r, c, d, o);
    }
    int hessian_bcrs_sym(const real_t *const x, const count_t *const r, const idx_t *const c, const ptrdiff_t s,
                         real_t **const d, real_t **const o) override {
        return inner_->hessian_bcrs_sym(x, r, c, s, d, o);
    }
    int hessian_block_diag_sym(const real_t *const x, real_t *const v) override { return inner_->hessian_block_diag_sym(x, v); }
    int hessian_block_diag_sym_soa(const real_t *const x, real_t **const v) override {
        return inner_->hessian_block_diag_sym_soa(x, v);
    }
    int hessian_bsr(const real_t *const x, const count_t *const r, const idx_t *const c, real_t *const v) override {
        return inner_->hessian_bsr(x, r, c, v);
    }
    int hessian_bsr_typed(const real_t *const x, const count_t *const r, const idx_t *const c,
                          const smesh::PrimitiveType t, void *const v) override {
        return inner_->hessian_bsr_typed(x, r, c, t, v);
    }
    int hessian_dia(const real_t *const x, const int *const o, const ptrdiff_t n, real_t *const v) override {
        return inner_->hessian_dia(x, o, n, v);
    }
    int hessian_diag(const real_t *const x, real_t *const v) override { return inner_->hessian_diag(x, v); }
    int gradient(const real_t *const x, real_t *const out) override { return inner_->gradient(x, out); }
    int apply(const real_t *const x, const real_t *const h, real_t *const out) override { return inner_->apply(x, h, out); }
    int value(const real_t *x, real_t *const out) override { return inner_->value(x, out); }
    int gradient(const real_t *const x, real_t *const out, const sfem::ElementScope s) override {
        return inner_->gradient(x, out, s);
    }
    int apply(const real_t *const x, const real_t *const h, real_t *const out, const sfem::ElementScope s) override {
        return inner_->apply(x, h, out, s);
    }
    int value(const real_t *x, real_t *const out, const sfem::ElementScope s) override { return inner_->value(x, out, s); }
    int apply_scope_flat_range(const real_t *const x, const real_t *const h, real_t *const out,
                               const sfem::ElementScope s, const ptrdiff_t b, const ptrdiff_t e) override {
        return inner_->apply_scope_flat_range(x, h, out, s, b, e);
    }
    bool energy_or_potential_based() const override { return inner_->energy_or_potential_based(); }
    bool residual_depends_on_state() const override { return inner_->residual_depends_on_state(); }
    bool contracts_residual_merit() const override { return inner_->contracts_residual_merit(); }
    int residual_merit_steps(const real_t *const x, const real_t *const h, const int n, const real_t *const s,
                             const real_t *const a, real_t *const out) override {
        return inner_->residual_merit_steps(x, h, n, s, a, out);
    }
    int value_steps(const real_t *x, const real_t *h, const int n, const real_t *const s, real_t *const out) override {
        return inner_->value_steps(x, h, n, s, out);
    }
    int report(const real_t *const x) override { return inner_->report(x); }
    void set_option(const std::string &n, bool v) override { inner_->set_option(n, v); }
    void set_value_in_block(const std::string &b, const std::string &n, const real_t v) override {
        inner_->set_value_in_block(b, n, v);
    }
    sfem::ExecutionContext execution_context() const override { return inner_->execution_context(); }
    void set_field(const char *n, const std::shared_ptr<sfem::Buffer<real_t>> &v, const int c) override {
        inner_->set_field(n, v, c);
    }
    bool is_no_op() const override { return inner_->is_no_op(); }
    ptrdiff_t n_dofs_domain() const override { return inner_->n_dofs_domain(); }
    ptrdiff_t n_dofs_image() const override { return inner_->n_dofs_image(); }

private:
    std::shared_ptr<sfem::Op> inner_;
    std::unique_ptr<Geometry> g_;
    std::unique_ptr<QpKernels> k_;
    int n_update_ = 0, n_apply_ = 0;
    double t_update_ = 0, t_apply_ = 0;
};

// The process's peak resident memory so far.
static double peak_resident_mb() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
#ifdef __APPLE__
    return usage.ru_maxrss / 1048576.0;  // bytes
#else
    return usage.ru_maxrss / 1024.0;  // KiB
#endif
}

// The affine geometry the kernels take: adjugate and determinant of the map spanned by the
// element's first vertex and its three neighbours along the reference axes.
static std::unique_ptr<Geometry> affine_geometry(smesh::Mesh &mesh) {
    if (mesh.n_blocks() != 1) SFEM_ERROR("newton_qp: one block only\n");
    auto block = mesh.block(0);
    auto g = std::make_unique<Geometry>();
    g->nelements = block->n_elements();
    idx_t **ev = block->elements()->data();
    const geom_t *const *x = mesh.points()->data();
#ifdef KERNEL_SHAPE_ORDER
    static const int order[] = {KERNEL_SHAPE_ORDER};
    for (int p = 0; p < El::N; ++p) g->kernel_elements.push_back(ev[order[p]]);
#else
    for (int p = 0; p < El::N; ++p) g->kernel_elements.push_back(ev[p]);
#endif
    // Reference axes from node 0: tetrahedra 1, 2, 3; hexahedra (mesh order) 1, 3, 4; PROTEUS_HEX27
    // (lexicographic) 2, 6, 18.
    const int axis[3] = {El::N == 27 ? 2 : 1, El::N == 27 ? 6 : (El::N == 8 ? 3 : 2), El::N == 27 ? 18 : (El::N == 8 ? 4 : 3)};
    g->adj.assign(9, std::vector<geom_t>(g->nelements));
    g->det.assign(g->nelements, 0);
#pragma omp parallel for
    for (ptrdiff_t e = 0; e < g->nelements; ++e) {
        double J[9];
        for (int c = 0; c < 3; ++c)
            for (int d = 0; d < 3; ++d) J[d * 3 + c] = (double)x[d][ev[axis[c]][e]] - (double)x[d][ev[0][e]];
        const double A[9] = {J[4] * J[8] - J[5] * J[7], J[2] * J[7] - J[1] * J[8], J[1] * J[5] - J[2] * J[4],
                             J[5] * J[6] - J[3] * J[8], J[0] * J[8] - J[2] * J[6], J[2] * J[3] - J[0] * J[5],
                             J[3] * J[7] - J[4] * J[6], J[1] * J[6] - J[0] * J[7], J[0] * J[4] - J[1] * J[3]};
        for (int c = 0; c < 9; ++c) g->adj[c][e] = (geom_t)A[c];
        g->det[e] = (geom_t)(J[0] * A[0] + J[1] * A[3] + J[2] * A[6]);
    }
    return g;
}

// mu and lmbda from the case, as the generated operator reads them.
static bool material_parameters(const sfem::ElasticityCase &config, real_t &mu, real_t &lmbda) {
    auto m = config.root()["material"];
    auto p = m.has_child("parameters") ? m["parameters"] : m;
    auto read = [&](const char *key, real_t &v) {
        if (!p.has_child(ryml::to_csubstr(key))) return false;
        double d = 0;
        p[ryml::to_csubstr(key)] >> d;
        v = (real_t)d;
        return true;
    };
    if (read("mu", mu) && read("lmbda", lmbda)) return true;
    real_t E = 0, nu = 0;
    if (read("young_modulus", E) && read("poisson_ratio", nu)) {
        mu = E / (2 * (1 + nu));
        lmbda = E * nu / ((1 + nu) * (1 - 2 * nu));
        return true;
    }
    return false;
}

}  // namespace

int main(int argc, char *argv[]) {
    auto ctx = sfem::initialize(argc, argv);
    std::string path, variant;
    std::vector<std::string> overrides;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--set") && i + 1 < argc)
            overrides.push_back(argv[++i]);
        else if (!strcmp(argv[i], "--qp") && i + 1 < argc)
            variant = argv[++i];
        else if (path.empty())
            path = argv[i];
    }
    if (path.empty()) {
        fprintf(stderr, "usage: %s case.yaml [--set key.path=value]... [--qp f64|f16|i8|i8md|f16md|eaf64|eaf16|ead32|ead16|eadi8|eadi8p|bsrdi8|bsrfi8|bsrd16|bsrf16|bsrui8|bsru16|bsroi8|bsro16]\n", argv[0]);
        return SFEM_FAILURE;
    }
    auto config = sfem::ElasticityCase::read(path, overrides);
    if (!config) return SFEM_FAILURE;
    auto playground = sfem::Playground::create(ctx->communicator(), *config);
    if (!playground) return SFEM_FAILURE;
    printf("newton_qp: peak resident after set-up %.1f MB\n", peak_resident_mb());

    if (!variant.empty()) {
        auto f = playground->function();
        auto material = playground->material();
        // Quantized BSR needs only the block graph; the element stores only the geometry and packing.
        const bool assembled = variant.rfind("bsr", 0) == 0;
        std::unique_ptr<Geometry> geometry;
        if (assembled) {
            geometry = std::make_unique<Geometry>();
            geometry->nelements = f->space()->mesh()->block(0)->n_elements();
            geometry->graph = f->space()->element_node_graph();
        } else {
            geometry = affine_geometry(*f->space()->mesh());
        }
        if (!assembled && f->space()->has_packed_mesh()) {
            // The store is indexed by element, and packing keeps the element order, so the same
            // store serves the packed traversal; only the scatter changes.
            auto pm = f->space()->packed_mesh();
            auto v = std::make_unique<PackedView>();
            v->n_packs = pm->n_packs(0);
            v->n_elements_per_pack = pm->n_elements_per_pack(0);
            v->max_nodes_per_pack = pm->max_nodes_per_pack();
            v->n_ghost_entries = pm->n_ghost_entries(0);
            v->n_ghost_reduce_rows = pm->n_ghost_reduce_rows(0);
            v->owned_nodes_ptr = pm->owned_nodes_ptr(0)->data();
            v->ghost_ptr = pm->ghost_ptr(0)->data();
            v->ghost_idx = pm->ghost_idx(0)->data();
            v->ghost_reduce_ptr = pm->ghost_reduce_ptr(0)->data();
            v->ghost_reduce_idx = pm->ghost_reduce_idx(0)->data();
            v->ghost_reduce_dest = pm->ghost_reduce_dest(0)->data();
            uint16_t **pe = pm->elements(0)->data();
#ifdef KERNEL_SHAPE_ORDER
            static const int order[] = {KERNEL_SHAPE_ORDER};
            for (int p = 0; p < El::N; ++p) v->kernel_elements.push_back(pe[order[p]]);
#else
            for (int p = 0; p < El::N; ++p) v->kernel_elements.push_back(pe[p]);
#endif
            v->ghost_buf.assign((size_t)3 * v->n_ghost_entries, 0.0);
            v->keep = pm;
            if (pm->n_packed_elements(0) != geometry->nelements) {
                fprintf(stderr, "newton_qp: the packed mesh has %ld elements, the mesh %ld\n",
                        (long)pm->n_packed_elements(0), (long)geometry->nelements);
                return SFEM_FAILURE;
            }
            geometry->packed = std::move(v);
        }
        if (!material_parameters(*config, geometry->mu, geometry->lmbda)) {
            fprintf(stderr, "newton_qp: material needs mu and lmbda, or young_modulus and poisson_ratio\n");
            return SFEM_FAILURE;
        }
        auto wrapped = std::make_shared<QpOp>(material, std::move(geometry), variant);
        f->remove_operator(material);
        f->add_operator(wrapped);
        printf("newton_qp: %s wrapped with the %s qp store (%s, %s scatter)\n", material->name(), variant.c_str(),
               El::name, f->space()->has_packed_mesh() ? "packed" : "atomic");
        printf("newton_qp: peak resident after the store %.1f MB\n", peak_resident_mb());
    }
    const double t0 = now();
    const int status = playground->run();
    printf("newton_qp: run %.3f s, status %d\n", now() - t0, status);
    printf("newton_qp: peak resident %.1f MB\n", peak_resident_mb());
    return status;
}
