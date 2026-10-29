// What the Newton drivers share, on the CPU (newton_qp) and the GPU (cuda/newton_qp_cuda): the
// wrapper that replaces the material operator's inexact pair with a store's kernels, the affine
// geometry and material parameters the kernels take, and the timing and memory reports.
#pragma once

#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include <sys/resource.h>

#include "sfem_API.hpp"
#include "sfem_ElasticityCase.hpp"
#include "sfem_Function.hpp"
#include "sfem_FunctionSpace.hpp"

namespace {

using El = qp::Element;

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

// The material operator, with the inexact pair replaced.
class QpOp final : public sfem::Op {
public:
    QpOp(const std::shared_ptr<sfem::Op> &inner, std::unique_ptr<Geometry> geometry, std::unique_ptr<QpKernels> kernels,
         const char *element)
        : inner_(inner), g_(std::move(geometry)), k_(std::move(kernels)), element_(element) {}
    ~QpOp() override {
        printf("qp: %s store %.1f MB, %d updates %.3f s, %d applies %.3f s\n", element_, k_->bytes() / 1048576.0,
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
    const char *element_;
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
