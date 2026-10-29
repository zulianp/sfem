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
#include "qp_op.hpp"

#ifndef LANE_VS
#define LANE_VS 16
#endif

namespace {

using El = qp::Element;
using qp::Layout;
using qp::Scheme;

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

// The kernels of a --qp variant.
static std::unique_ptr<QpKernels> make_kernels(const std::string &variant, const std::shared_ptr<sfem::Op> &inner,
                                               const Geometry &g) {
        using qp::EaForm;
        if (variant == "f64")
            return std::make_unique<QpKernelsT<Q<double, Scheme::Plain>>>(g);
        else if (variant == "f16")
            return std::make_unique<QpKernelsT<Q<half_t, Scheme::Plain>>>(g);
        else if (variant == "i8")
            return std::make_unique<QpKernelsT<Q<int8_t, Scheme::Plain>>>(g);
        else if (variant == "i8md" && El::NV > 1)
            return std::make_unique<QpKernelsT<Q<int8_t, Scheme::MeanDeviation>>>(g);
        else if (variant == "f16md" && El::NV > 1)
            return std::make_unique<QpKernelsT<Q<half_t, Scheme::MeanDeviation>>>(g);
        // Element matrices: full (eaf*) and difference form (ead*).
        else if (variant == "eaf64")
            return std::make_unique<QpKernelsT<EA<double, EaForm::Full>>>(g);
        else if (variant == "eaf16")
            return std::make_unique<QpKernelsT<EA<half_t, EaForm::Full>>>(g);
        else if (variant == "ead32")
            return std::make_unique<QpKernelsT<EA<float, EaForm::Difference>>>(g);
        else if (variant == "ead16")
            return std::make_unique<QpKernelsT<EA<half_t, EaForm::Difference>>>(g);
        else if (variant == "eadi8")
            return std::make_unique<QpKernelsT<EA<int8_t, EaForm::Difference>>>(g);
        else if (variant == "eadi8p")
            return std::make_unique<QpKernelsT<EA<int8_t, EaForm::PairScale>>>(g);
        // Quantized BSR: difference (bsrd*) and full (bsrf*) form, int8 per block or f16.
        else if (variant == "bsrdi8")
            return std::make_unique<BsrKernels<int8_t, qp::BsrForm::Difference>>(inner, g.graph);
        else if (variant == "bsrfi8")
            return std::make_unique<BsrKernels<int8_t, qp::BsrForm::Full>>(inner, g.graph);
        else if (variant == "bsrd16")
            return std::make_unique<BsrKernels<half_t, qp::BsrForm::Difference>>(inner, g.graph);
        else if (variant == "bsrf16")
            return std::make_unique<BsrKernels<half_t, qp::BsrForm::Full>>(inner, g.graph);
        else if (variant == "bsrui8")
            return std::make_unique<BsrKernels<int8_t, qp::BsrForm::Upper>>(inner, g.graph);
        else if (variant == "bsru16")
            return std::make_unique<BsrKernels<half_t, qp::BsrForm::Upper>>(inner, g.graph);
        else if (variant == "bsroi8")
            return std::make_unique<BsrKernels<int8_t, qp::BsrForm::Upper, uint16_t>>(inner, g.graph);
        else if (variant == "bsro16")
            return std::make_unique<BsrKernels<half_t, qp::BsrForm::Upper, uint16_t>>(inner, g.graph);
        else
    SFEM_ERROR("newton_qp: unknown --qp variant '%s'\n", variant.c_str());
    return nullptr;
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
        auto kernels = make_kernels(variant, material, *geometry);
        auto wrapped = std::make_shared<QpOp>(material, std::move(geometry), std::move(kernels), El::name);
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
