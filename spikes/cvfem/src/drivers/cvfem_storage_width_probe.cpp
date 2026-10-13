#include "frontend/op/cvfem_hex8_ns_core.hpp"
#include "sfem_context.hpp"
#include "smesh_mesh_reorder.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

static uint32_t narrow_offset(const ptrdiff_t value) {
    if (value < 0 || uint64_t(value) > std::numeric_limits<uint32_t>::max())
        throw std::out_of_range("reduction offset does not fit uint32_t");
    return uint32_t(value);
}

struct Probe {
    MeshData mesh;
    PackedData packed;
    std::vector<uint32_t> offsets, indices;
    std::array<std::vector<float>, 10> geometry32;
    std::array<const float *, 9> adj32{};
};

static void setup(Probe &probe, sfem::Context &ctx, const int n, const int pack_size, const bool shear) {
    auto &d = probe.mesh;
    d.mesh = smesh::Mesh::create_hex8_cube(ctx.communicator(), n, n, n, 0, 0, 0, 1, 1, 1);
    auto sfc = smesh::SFC::create_from_env();
    sfc->reorder(*d.mesh);
    if (shear) {
        auto points = d.mesh->points()->data();
        for (ptrdiff_t i = 0; i < d.mesh->n_nodes(); ++i) {
            const double x = points[0][i], y = points[1][i], z = points[2][i];
            points[0][i] = smesh::geom_t(1.17*x + .13*y - .04*z);
            points[1][i] = smesh::geom_t(.07*x + .91*y + .09*z);
            points[2][i] = smesh::geom_t(.03*x - .06*y + 1.08*z);
        }
    }
    probe.packed = make_packed(d.mesh, pack_size);
    d.packed = &probe.packed;
    d.nnodes = d.mesh->n_nodes(); d.nelements = d.mesh->n_elements(0);
    d.elems = d.mesh->elements(0)->data(); d.points = d.mesh->points()->data();
    d.Lx = d.Ly = d.Lz = 1;
    d.rhie_chow_scale = 0;
    cvfem_hex8_precompute_affine_geometry(d);
    d.ux.resize(d.nnodes); d.uy.resize(d.nnodes); d.uz.resize(d.nnodes); d.p.resize(d.nnodes);
    d.rx.assign(d.nnodes, 0); d.ry.assign(d.nnodes, 0); d.rz.assign(d.nnodes, 0); d.rc.assign(d.nnodes, 0);
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        const double x = d.points[0][i], y = d.points[1][i], z = d.points[2][i];
        d.ux[i] = .3 + .1*x + .04*y;
        d.uy[i] = .2 + .07*y - .02*z;
        d.uz[i] = .15 + .06*z + .02*x;
        d.p[i] = .1 + .03*x - .01*y + .07*z;
    }
    // Capture actual ghost contributions once; only their graph reduction is timed below.
    cvfem_hex8_apply_residual_packed(d, probe.packed, 1, .01);
    const auto &p = probe.packed;
    if (!p.n_ghost_entries || !p.n_ghost_reduce_rows) throw std::runtime_error("probe needs a nonempty ghost graph");
    const ptrdiff_t count = p.ghost_reduce_ptr[p.n_ghost_reduce_rows];
    (void)narrow_offset(count);
    if (p.ghost_reduce_ptr[0] != 0) throw std::runtime_error("reduction offsets must start at zero");
    probe.offsets.resize(p.n_ghost_reduce_rows + 1);
    probe.indices.resize(count);
    std::vector<uint8_t> seen(d.nnodes, 0);
    for (ptrdiff_t r = 0; r <= p.n_ghost_reduce_rows; ++r) {
        const ptrdiff_t value = p.ghost_reduce_ptr[r];
        if (value > count || (r && value < p.ghost_reduce_ptr[r-1])) throw std::runtime_error("invalid reduction offsets");
        probe.offsets[r] = narrow_offset(value);
        if (r < p.n_ghost_reduce_rows) {
            const auto dest = p.ghost_reduce_dest[r];
            if (dest < 0 || dest >= d.nnodes || seen[dest]++) throw std::runtime_error("invalid or repeated reduction destination");
        }
    }
    for (ptrdiff_t j = 0; j < count; ++j) {
        const ptrdiff_t value = p.ghost_reduce_idx[j];
        if (value >= p.n_ghost_entries) throw std::runtime_error("invalid reduction index");
        probe.indices[j] = narrow_offset(value);
    }
    for (int c = 0; c < 10; ++c) {
        probe.geometry32[c].resize(d.nelements);
        const auto &values = c < 9 ? d.jacobian_adjugate[c] : d.jacobian_determinant;
        for (ptrdiff_t e = 0; e < d.nelements; ++e) {
            if (!std::isfinite(values[e]) || (c == 9 && values[e] <= 0)) throw std::runtime_error("invalid cached geometry");
            probe.geometry32[c][e] = float(values[e]);
        }
        if (c < 9) probe.adj32[c] = probe.geometry32[c].data();
    }
}

template <typename IndexType>
static SFEM_NOINLINE void reduce(const Probe &probe, const IndexType *offsets, const IndexType *indices,
                                 double *out) {
    const auto &p = probe.packed;
#pragma omp parallel
    {
        const auto rows = cvfem_range_split(0, p.n_ghost_reduce_rows, 1, cvfem_thread_index(), cvfem_n_threads());
        for (ptrdiff_t row = rows.begin; row < rows.end; ++row) {
            const auto dest = p.ghost_reduce_dest[row];
            const ptrdiff_t begin = offsets[row], end = offsets[row+1];
            for (int f = 0; f < 4; ++f) {
                const double *ghost = p.ghost_buf.data() + f*p.n_ghost_entries;
                double sum = 0;
                for (ptrdiff_t j = begin; j < end; ++j) sum += ghost[indices[j]];
                out[ptrdiff_t(dest)*4+f] += sum;
            }
        }
    }
    asm volatile("" : : "m"(*out) : "memory");
}

static constexpr int lanes = CVFEM_HEX8_VEC_SIZE;
using Geometry = std::array<double, 10*lanes>;

template <typename GeometryType>
static SFEM_INLINE void gather(const GeometryType *const *adj, const GeometryType *det,
                               const ptrdiff_t begin, const int nlanes, Geometry &out) {
    if constexpr (std::is_same<GeometryType, double>::value) {
        gather_hex8_adj_soa(adj, det, begin, nlanes, out.data(), out.data()+lanes, out.data()+2*lanes,
            out.data()+3*lanes, out.data()+4*lanes, out.data()+5*lanes, out.data()+6*lanes,
            out.data()+7*lanes, out.data()+8*lanes, out.data()+9*lanes);
    } else {
        std::array<float, 10*lanes> values;
        gather_hex8_adj_soa(adj, det, begin, nlanes, values.data(), values.data()+lanes, values.data()+2*lanes,
            values.data()+3*lanes, values.data()+4*lanes, values.data()+5*lanes, values.data()+6*lanes,
            values.data()+7*lanes, values.data()+8*lanes, values.data()+9*lanes);
        for (size_t k = 0; k < values.size(); ++k) out[k] = double(values[k]);
    }
}

template <typename GeometryType>
static SFEM_NOINLINE void geometry_sweep(const Probe &probe, const GeometryType *const *adj,
                                         const GeometryType *det) {
#pragma omp parallel
    {
        const auto elements = cvfem_range_split(0, probe.mesh.nelements, lanes,
                                               cvfem_thread_index(), cvfem_n_threads());
        Geometry values;
        for (ptrdiff_t e = elements.begin; e < elements.end; e += lanes) {
            gather(adj, det, e, int(std::min<ptrdiff_t>(lanes, elements.end-e)), values);
            // Consume every loaded coefficient in both modes, including the FP32 conversion.
            asm volatile("" : : "m"(values) : "memory");
        }
    }
}

static double validate(const Probe &probe, const char *label) {
    const auto &d = probe.mesh; const auto &p = probe.packed;
    std::vector<double> wide(d.nnodes*4, 0), narrow(d.nnodes*4, 0), reference(d.nnodes*4, 0);
    reduce(probe, p.ghost_reduce_ptr, p.ghost_reduce_idx, wide.data());
    reduce(probe, probe.offsets.data(), probe.indices.data(), narrow.data());
#pragma omp parallel
    cvfem_hex8_ghost_reduce_interleaved_range(cvfem_range_split(0, p.n_ghost_reduce_rows, 1,
        cvfem_thread_index(), cvfem_n_threads()), p.ghost_reduce_dest, p.ghost_reduce_ptr,
        p.ghost_reduce_idx, p.n_ghost_entries, p.ghost_buf.data(), reference.data());
    double signal = 0;
    for (const auto value : wide) {
        if (!std::isfinite(value)) throw std::runtime_error("nonfinite graph result");
        signal = std::max(signal, std::fabs(value));
    }
    if (signal == 0 || std::memcmp(wide.data(), narrow.data(), wide.size()*sizeof(double)) ||
        std::memcmp(wide.data(), reference.data(), wide.size()*sizeof(double)))
        throw std::runtime_error("nontrivial graph reduction parity failed");
    double worst = 0;
    for (ptrdiff_t e = 0; e < d.nelements; e += lanes) {
        Geometry a, b;
        const int count = int(std::min<ptrdiff_t>(lanes, d.nelements-e));
        gather(d.adj_ptr, d.det_ptr, e, count, a);
        gather(probe.adj32.data(), probe.geometry32[9].data(), e, count, b);
        for (int c = 0; c < 10; ++c) {
            for (int lane = 0; lane < count; ++lane) {
                const auto k = c*lanes+lane;
                if (!std::isfinite(b[k]) || (c == 9 && b[k] <= 0)) throw std::runtime_error("invalid FP32 geometry");
                const double error = std::fabs(b[k]-a[k]) / std::max(std::fabs(a[k]), std::numeric_limits<double>::min());
                worst = std::max(worst, error);
            }
        }
    }
    if (worst > 2e-7) throw std::runtime_error("FP32 geometry conversion exceeds rounding gate");
    std::printf("validation,%s,graph_bit_equal=1,geometry_max_relative=%.17g\n", label, worst);
    return worst;
}

template <typename Function>
static double timed(Function &&function) {
    const auto before = std::chrono::steady_clock::now();
    function();
    return std::chrono::duration<double>(std::chrono::steady_clock::now()-before).count();
}

int main(int argc, char **argv) {
    try {
        int n=16, pack=64, samples=12, repeat=5, warmup=3;
        bool self_test=false;
        for (int i=1; i<argc; ++i) {
            const std::string arg=argv[i];
            if (arg=="--self-test") { self_test=true; continue; }
            if (i+1==argc) throw std::runtime_error("missing argument value");
            const int value=std::stoi(argv[++i]);
            if (arg=="--n") n=value; else if (arg=="--pack-size") pack=value;
            else if (arg=="--samples") samples=value; else if (arg=="--repeat") repeat=value;
            else if (arg=="--warmup") warmup=value; else throw std::runtime_error("unknown argument");
        }
        if (n<2 || pack<1 || samples<1 || repeat<1 || warmup<0) throw std::runtime_error("invalid probe size");
        for (const auto value : {ptrdiff_t(-1), ptrdiff_t(uint64_t(std::numeric_limits<uint32_t>::max())+1)}) {
            bool rejected=false;
            try { (void)narrow_offset(value); } catch (const std::out_of_range &) { rejected=true; }
            if (!rejected) throw std::runtime_error("invalid narrowing was accepted");
        }
        auto ctx=sfem::initialize(argc, argv);
        if (ctx->communicator()->size() != 1) throw std::runtime_error("storage-width probe requires one MPI rank");
        for (const bool shear : {false, true}) {
            Probe fixture; setup(fixture, *ctx, 7, 16, shear);
            validate(fixture, shear ? "sheared_affine_at_coordinate_precision" : "cube");
        }
        if (self_test) return 0;
        Probe probe; setup(probe, *ctx, n, pack, false); validate(probe, "timed_cube");
        const auto &d=probe.mesh; const auto &p=probe.packed;
        const size_t wide_graph=(probe.offsets.size()+probe.indices.size())*sizeof(ptrdiff_t);
        const size_t narrow_graph=(probe.offsets.size()+probe.indices.size())*sizeof(uint32_t);
        const size_t wide_geom=size_t(d.nelements)*10*sizeof(double), narrow_geom=size_t(d.nelements)*10*sizeof(float);
        std::printf("scope,graph_reduction_and_escaped_geometry_gather_only,no_operator_speedup,no_total_footprint_saving\n");
        int threads = 1;
#pragma omp parallel
        {
#pragma omp single
            threads = cvfem_n_threads();
        }
        std::printf("metadata,n=%d,pack=%d,threads=%d,elements=%td,ghost_entries=%td,rows=%td,ordering=SFC\n", n, pack, threads, d.nelements, p.n_ghost_entries, p.n_ghost_reduce_rows);
        std::printf("allocations,retained_graph64=%zu,additional_graph32=%zu,retained_geometry64=%zu,additional_geometry32=%zu\n",wide_graph,narrow_graph,wide_geom,narrow_geom);
        std::printf("timing_contract,serial_output_clearing_excluded_from_both_graph_arms,geometry_float_to_double_included,warmup=%d,repeat=%d,repeated_warm_cache_probe\n",warmup,repeat);
        std::vector<double> out(d.nnodes*4, 0);
        auto graph64=[&] { reduce(probe,p.ghost_reduce_ptr,p.ghost_reduce_idx,out.data()); };
        auto graph32=[&] { reduce(probe,probe.offsets.data(),probe.indices.data(),out.data()); };
        auto geom64=[&] { geometry_sweep(probe,d.adj_ptr,d.det_ptr); };
        auto geom32=[&] { geometry_sweep(probe,probe.adj32.data(),probe.geometry32[9].data()); };
        for(int k=0;k<warmup;++k) { graph64();graph32();geom64();geom32(); }
        std::printf("sample,phase,first,wide_seconds,narrow_seconds,wide_over_narrow\n");
        for(int sample=0;sample<samples;++sample) {
            for(int phase=0;phase<2;++phase) {
                double seconds[2]={0,0};
                for(int r=0;r<repeat;++r) {
                    for(int slot=0;slot<2;++slot) {
                        const int mode=slot^(sample&1);
                        if(phase==0) std::fill(out.begin(),out.end(),0);
                        if(phase==0) seconds[mode]+=timed([&]{if(mode) graph32();else graph64();});
                        else seconds[mode]+=timed([&]{if(mode) geom32();else geom64();});
                    }
                }
                seconds[0]/=repeat;seconds[1]/=repeat;
                if(!(seconds[0]>0 && seconds[1]>0) || !std::isfinite(seconds[0]+seconds[1])) throw std::runtime_error("invalid timing");
                std::printf("%d,%s,%s,%.17g,%.17g,%.17g\n",sample,phase ? "geometry" : "graph",sample&1 ? "narrow" : "wide",seconds[0],seconds[1],seconds[0]/seconds[1]);
            }
        }
        validate(probe, "after_timing");
        std::printf("probe_complete,1\n");
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr,"storage_width_probe: %s\n",error.what());
        return 2;
    }
}
