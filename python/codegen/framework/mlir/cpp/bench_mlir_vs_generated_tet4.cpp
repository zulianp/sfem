#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

using idx_t = int32_t;
using geom_t = float;

extern "C" int linear_elasticity_tet4_tet4_apply_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **elements,
        const geom_t *g_jacobian_adjugate0,
        const geom_t *g_jacobian_adjugate1,
        const geom_t *g_jacobian_adjugate2,
        const geom_t *g_jacobian_adjugate3,
        const geom_t *g_jacobian_adjugate4,
        const geom_t *g_jacobian_adjugate5,
        const geom_t *g_jacobian_adjugate6,
        const geom_t *g_jacobian_adjugate7,
        const geom_t *g_jacobian_adjugate8,
        const geom_t *g_jacobian_determinant0,
        const float mu,
        const float lmbda,
        const ptrdiff_t h_stride,
        const float *hx,
        const float *hy,
        const float *hz,
        const ptrdiff_t out_stride,
        float *outx,
        float *outy,
        float *outz);

extern "C" void linear_elasticity_tet4_mlir_apply_openmp_c(
        int64_t *connectivity,
        float *direction,
        float *adjugate,
        float *determinant,
        float lmbda,
        float mu,
        float *scratch,
        int64_t *node_degree,
        int64_t *node_to_element_map,
        int64_t *node_to_local_idx,
        float *output);

extern "C" void linear_elasticity_tet4_mlir_apply_openmp(
        void *connectivity_alloc,
        void *connectivity_aligned,
        int64_t connectivity_offset,
        int64_t connectivity_size,
        int64_t connectivity_stride,
        void *direction_alloc,
        void *direction_aligned,
        int64_t direction_offset,
        int64_t direction_size,
        int64_t direction_stride,
        void *adjugate_alloc,
        void *adjugate_aligned,
        int64_t adjugate_offset,
        int64_t adjugate_size,
        int64_t adjugate_stride,
        void *determinant_alloc,
        void *determinant_aligned,
        int64_t determinant_offset,
        int64_t determinant_size,
        int64_t determinant_stride,
        float lmbda,
        float mu,
        void *scratch_alloc,
        void *scratch_aligned,
        int64_t scratch_offset,
        int64_t scratch_size,
        int64_t scratch_stride,
        void *node_degree_alloc,
        void *node_degree_aligned,
        int64_t node_degree_offset,
        int64_t node_degree_size,
        int64_t node_degree_stride,
        void *node_to_element_map_alloc,
        void *node_to_element_map_aligned,
        int64_t node_to_element_map_offset,
        int64_t node_to_element_map_size,
        int64_t node_to_element_map_stride,
        void *node_to_local_idx_alloc,
        void *node_to_local_idx_aligned,
        int64_t node_to_local_idx_offset,
        int64_t node_to_local_idx_size,
        int64_t node_to_local_idx_stride,
        void *output_alloc,
        void *output_aligned,
        int64_t output_offset,
        int64_t output_size,
        int64_t output_stride);

template <typename T>
static std::vector<T> read_raw(const std::string &path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        std::fprintf(stderr, "Unable to open %s\n", path.c_str());
        std::exit(2);
    }
    const std::streamsize bytes = in.tellg();
    if (bytes % static_cast<std::streamsize>(sizeof(T)) != 0) {
        std::fprintf(stderr, "Invalid raw size for %s\n", path.c_str());
        std::exit(2);
    }
    std::vector<T> values(bytes / sizeof(T));
    in.seekg(0, std::ios::beg);
    in.read(reinterpret_cast<char *>(values.data()), bytes);
    return values;
}

static double seconds_now() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

static void call_mlir_openmp(const ptrdiff_t nelements,
                             const ptrdiff_t nnodes,
                             const ptrdiff_t max_node_degree,
                             std::vector<int64_t> &connectivity,
                             std::vector<float> &direction,
                             std::vector<float> &adjugate,
                             std::vector<float> &determinant,
                             const float lmbda,
                             const float mu,
                             std::vector<float> &scratch,
                             std::vector<int64_t> &node_degree,
                             std::vector<int64_t> &node_to_element_map,
                             std::vector<int64_t> &node_to_local_idx,
                             std::vector<float> &output) {
    linear_elasticity_tet4_mlir_apply_openmp(connectivity.data(),
                                             connectivity.data(),
                                             0,
                                             4 * nelements,
                                             1,
                                             direction.data(),
                                             direction.data(),
                                             0,
                                             3 * nnodes,
                                             1,
                                             adjugate.data(),
                                             adjugate.data(),
                                             0,
                                             9 * nelements,
                                             1,
                                             determinant.data(),
                                             determinant.data(),
                                             0,
                                             nelements,
                                             1,
                                             lmbda,
                                             mu,
                                             scratch.data(),
                                             scratch.data(),
                                             0,
                                             12 * nelements,
                                             1,
                                             node_degree.data(),
                                             node_degree.data(),
                                             0,
                                             nnodes,
                                             1,
                                             node_to_element_map.data(),
                                             node_to_element_map.data(),
                                             0,
                                             nnodes * max_node_degree,
                                             1,
                                             node_to_local_idx.data(),
                                             node_to_local_idx.data(),
                                             0,
                                             nnodes * max_node_degree,
                                             1,
                                             output.data(),
                                             output.data(),
                                             0,
                                             3 * nnodes,
                                             1);
}

static void compute_adjugate_and_det(const std::vector<int64_t> &connectivity,
                                     const std::vector<float> &x,
                                     const std::vector<float> &y,
                                     const std::vector<float> &z,
                                     std::vector<float> &mlir_adj,
                                     std::vector<float> &mlir_det,
                                     std::vector<std::vector<geom_t>> &generated_adj,
                                     std::vector<geom_t> &generated_det) {
    const ptrdiff_t nelements = static_cast<ptrdiff_t>(mlir_det.size());
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        const int64_t n0 = connectivity[4 * e + 0];
        const int64_t n1 = connectivity[4 * e + 1];
        const int64_t n2 = connectivity[4 * e + 2];
        const int64_t n3 = connectivity[4 * e + 3];

        const double j00 = static_cast<double>(x[n1] - x[n0]);
        const double j10 = static_cast<double>(y[n1] - y[n0]);
        const double j20 = static_cast<double>(z[n1] - z[n0]);
        const double j01 = static_cast<double>(x[n2] - x[n0]);
        const double j11 = static_cast<double>(y[n2] - y[n0]);
        const double j21 = static_cast<double>(z[n2] - z[n0]);
        const double j02 = static_cast<double>(x[n3] - x[n0]);
        const double j12 = static_cast<double>(y[n3] - y[n0]);
        const double j22 = static_cast<double>(z[n3] - z[n0]);

        double adj[9];
        adj[0] = j11 * j22 - j12 * j21;
        adj[1] = j02 * j21 - j01 * j22;
        adj[2] = j01 * j12 - j02 * j11;
        adj[3] = j12 * j20 - j10 * j22;
        adj[4] = j00 * j22 - j02 * j20;
        adj[5] = j02 * j10 - j00 * j12;
        adj[6] = j10 * j21 - j11 * j20;
        adj[7] = j01 * j20 - j00 * j21;
        adj[8] = j00 * j11 - j01 * j10;
        const double det = j00 * adj[0] + j01 * adj[3] + j02 * adj[6];

        mlir_det[e] = static_cast<float>(det);
        generated_det[e] = det;
        for (int k = 0; k < 9; ++k) {
            mlir_adj[9 * e + k] = static_cast<float>(adj[k]);
            generated_adj[k][e] = adj[k];
        }
    }
}

int main(int argc, char **argv) {
    if (argc != 6) {
        std::fprintf(stderr, "usage: %s <mesh_dir> <nelements> <nnodes> <max_node_degree> <repeat>\n", argv[0]);
        return 2;
    }

    const std::string mesh_dir = argv[1];
    const ptrdiff_t nelements = std::strtoll(argv[2], nullptr, 10);
    const ptrdiff_t nnodes = std::strtoll(argv[3], nullptr, 10);
    const ptrdiff_t max_node_degree = std::strtoll(argv[4], nullptr, 10);
    const int repeat = std::atoi(argv[5]);
    const float mu = 3.0f;
    const float lmbda = 2.0f;

    auto i0_32 = read_raw<int32_t>(mesh_dir + "/i0.int32");
    auto i1_32 = read_raw<int32_t>(mesh_dir + "/i1.int32");
    auto i2_32 = read_raw<int32_t>(mesh_dir + "/i2.int32");
    auto i3_32 = read_raw<int32_t>(mesh_dir + "/i3.int32");
    auto x = read_raw<float>(mesh_dir + "/x.float32");
    auto y = read_raw<float>(mesh_dir + "/y.float32");
    auto z = read_raw<float>(mesh_dir + "/z.float32");

    if (static_cast<ptrdiff_t>(i0_32.size()) != nelements || static_cast<ptrdiff_t>(x.size()) != nnodes) {
        std::fprintf(stderr, "mesh size mismatch\n");
        return 2;
    }

    std::vector<idx_t> e0(nelements), e1(nelements), e2(nelements), e3(nelements);
    std::vector<int64_t> connectivity(4 * nelements);
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        e0[e] = static_cast<idx_t>(i0_32[e]);
        e1[e] = static_cast<idx_t>(i1_32[e]);
        e2[e] = static_cast<idx_t>(i2_32[e]);
        e3[e] = static_cast<idx_t>(i3_32[e]);
        connectivity[4 * e + 0] = i0_32[e];
        connectivity[4 * e + 1] = i1_32[e];
        connectivity[4 * e + 2] = i2_32[e];
        connectivity[4 * e + 3] = i3_32[e];
    }
    idx_t *elements[4] = {e0.data(), e1.data(), e2.data(), e3.data()};

    std::vector<int64_t> node_degree(nnodes, 0);
    std::vector<int64_t> node_to_element_map(nnodes * max_node_degree, 0);
    std::vector<int64_t> node_to_local_idx(nnodes * max_node_degree, 0);
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        for (ptrdiff_t local = 0; local < 4; ++local) {
            const int64_t node = connectivity[4 * e + local];
            const int64_t slot = node_degree[node]++;
            if (slot >= max_node_degree) {
                std::fprintf(stderr, "node degree exceeds max_node_degree\n");
                return 2;
            }
            node_to_element_map[node * max_node_degree + slot] = e;
            node_to_local_idx[node * max_node_degree + slot] = local;
        }
    }

    std::vector<float> direction(3 * nnodes);
    for (ptrdiff_t i = 0; i < 3 * nnodes; ++i) {
        direction[i] = static_cast<float>((static_cast<int>(i % 97) + 1) / 97.0);
    }

    std::vector<float> mlir_adj(9 * nelements);
    std::vector<float> mlir_det(nelements);
    std::vector<std::vector<geom_t>> generated_adj(9, std::vector<geom_t>(nelements));
    std::vector<geom_t> generated_det(nelements);
    compute_adjugate_and_det(connectivity, x, y, z, mlir_adj, mlir_det, generated_adj, generated_det);

    std::vector<float> scratch(12 * nelements, 0.0f);
    std::vector<float> openmp_scratch(12 * nelements, 0.0f);
    std::vector<float> mlir_out(3 * nnodes, 0.0f);
    std::vector<float> openmp_out(3 * nnodes, 0.0f);
    std::vector<float> generated_out(3 * nnodes, 0.0f);

    linear_elasticity_tet4_mlir_apply_openmp_c(connectivity.data(),
                                               direction.data(),
                                               mlir_adj.data(),
                                               mlir_det.data(),
                                               lmbda,
                                               mu,
                                               scratch.data(),
                                               node_degree.data(),
                                               node_to_element_map.data(),
                                               node_to_local_idx.data(),
                                               mlir_out.data());
    call_mlir_openmp(nelements,
                     nnodes,
                     max_node_degree,
                     connectivity,
                     direction,
                     mlir_adj,
                     mlir_det,
                     lmbda,
                     mu,
                     openmp_scratch,
                     node_degree,
                     node_to_element_map,
                     node_to_local_idx,
                     openmp_out);
    linear_elasticity_tet4_tet4_apply_affine_mesh_soa_float(nelements,
                                                            nnodes,
                                                            elements,
                                                            generated_adj[0].data(),
                                                            generated_adj[1].data(),
                                                            generated_adj[2].data(),
                                                            generated_adj[3].data(),
                                                            generated_adj[4].data(),
                                                            generated_adj[5].data(),
                                                            generated_adj[6].data(),
                                                            generated_adj[7].data(),
                                                            generated_adj[8].data(),
                                                            generated_det.data(),
                                                            mu,
                                                            lmbda,
                                                            3,
                                                            direction.data() + 0,
                                                            direction.data() + 1,
                                                            direction.data() + 2,
                                                            3,
                                                            generated_out.data() + 0,
                                                            generated_out.data() + 1,
                                                            generated_out.data() + 2);

    double diff2 = 0.0;
    double openmp_diff2 = 0.0;
    double ref2 = 0.0;
    double max_abs = 0.0;
    double openmp_max_abs = 0.0;
    for (ptrdiff_t i = 0; i < 3 * nnodes; ++i) {
        const double diff = static_cast<double>(mlir_out[i] - generated_out[i]);
        const double openmp_diff = static_cast<double>(openmp_out[i] - generated_out[i]);
        diff2 += diff * diff;
        openmp_diff2 += openmp_diff * openmp_diff;
        ref2 += static_cast<double>(generated_out[i]) * static_cast<double>(generated_out[i]);
        max_abs = std::max(max_abs, std::abs(diff));
        openmp_max_abs = std::max(openmp_max_abs, std::abs(openmp_diff));
    }
    const double rel_l2 = std::sqrt(diff2) / std::max(std::sqrt(ref2), 1.0);
    const double openmp_rel_l2 = std::sqrt(openmp_diff2) / std::max(std::sqrt(ref2), 1.0);

    for (int i = 0; i < repeat / 100 && i < 1000; ++i) {
        linear_elasticity_tet4_mlir_apply_openmp_c(connectivity.data(),
                                                   direction.data(),
                                                   mlir_adj.data(),
                                                   mlir_det.data(),
                                                   lmbda,
                                                   mu,
                                                   scratch.data(),
                                                   node_degree.data(),
                                                   node_to_element_map.data(),
                                                   node_to_local_idx.data(),
                                                   mlir_out.data());
        linear_elasticity_tet4_tet4_apply_affine_mesh_soa_float(nelements,
                                                                nnodes,
                                                                elements,
                                                                generated_adj[0].data(),
                                                                generated_adj[1].data(),
                                                                generated_adj[2].data(),
                                                                generated_adj[3].data(),
                                                                generated_adj[4].data(),
                                                                generated_adj[5].data(),
                                                                generated_adj[6].data(),
                                                                generated_adj[7].data(),
                                                                generated_adj[8].data(),
                                                                generated_det.data(),
                                                                mu,
                                                                lmbda,
                                                                3,
                                                                direction.data() + 0,
                                                                direction.data() + 1,
                                                                direction.data() + 2,
                                                                3,
                                                                generated_out.data() + 0,
                                                                generated_out.data() + 1,
                                                                generated_out.data() + 2);
        call_mlir_openmp(nelements,
                         nnodes,
                         max_node_degree,
                         connectivity,
                         direction,
                         mlir_adj,
                         mlir_det,
                         lmbda,
                         mu,
                         openmp_scratch,
                         node_degree,
                         node_to_element_map,
                         node_to_local_idx,
                         openmp_out);
    }

    const double t0 = seconds_now();
    for (int r = 0; r < repeat; ++r) {
        linear_elasticity_tet4_mlir_apply_openmp_c(connectivity.data(),
                                                   direction.data(),
                                                   mlir_adj.data(),
                                                   mlir_det.data(),
                                                   lmbda,
                                                   mu,
                                                   scratch.data(),
                                                   node_degree.data(),
                                                   node_to_element_map.data(),
                                                   node_to_local_idx.data(),
                                                   mlir_out.data());
    }
    const double t1 = seconds_now();

    const double t2 = seconds_now();
    for (int r = 0; r < repeat; ++r) {
        call_mlir_openmp(nelements,
                         nnodes,
                         max_node_degree,
                         connectivity,
                         direction,
                         mlir_adj,
                         mlir_det,
                         lmbda,
                         mu,
                         openmp_scratch,
                         node_degree,
                         node_to_element_map,
                         node_to_local_idx,
                         openmp_out);
    }
    const double t3 = seconds_now();

    const double t4 = seconds_now();
    for (int r = 0; r < repeat; ++r) {
        linear_elasticity_tet4_tet4_apply_affine_mesh_soa_float(nelements,
                                                                nnodes,
                                                                elements,
                                                                generated_adj[0].data(),
                                                                generated_adj[1].data(),
                                                                generated_adj[2].data(),
                                                                generated_adj[3].data(),
                                                                generated_adj[4].data(),
                                                                generated_adj[5].data(),
                                                                generated_adj[6].data(),
                                                                generated_adj[7].data(),
                                                                generated_adj[8].data(),
                                                                generated_det.data(),
                                                                mu,
                                                                lmbda,
                                                                3,
                                                                direction.data() + 0,
                                                                direction.data() + 1,
                                                                direction.data() + 2,
                                                                3,
                                                                generated_out.data() + 0,
                                                                generated_out.data() + 1,
                                                                generated_out.data() + 2);
    }
    const double t5 = seconds_now();

    const double mlir_time = (t1 - t0) / repeat;
    const double openmp_time = (t3 - t2) / repeat;
    const double generated_time = (t5 - t4) / repeat;
    const double ndofs = static_cast<double>(3 * nnodes);
    const double mlir_melem = 1e-6 * static_cast<double>(nelements) / mlir_time;
    const double openmp_melem = 1e-6 * static_cast<double>(nelements) / openmp_time;
    const double generated_melem = 1e-6 * static_cast<double>(nelements) / generated_time;
    const double mlir_mdof = 1e-6 * ndofs / mlir_time;
    const double openmp_mdof = 1e-6 * ndofs / openmp_time;
    const double generated_mdof = 1e-6 * ndofs / generated_time;

    std::printf("mesh_dir %s\n", mesh_dir.c_str());
    std::printf("nelements %ld\n", static_cast<long>(nelements));
    std::printf("nnodes %ld\n", static_cast<long>(nnodes));
    std::printf("repeat %d\n", repeat);
    std::printf("max_abs %.9e\n", max_abs);
    std::printf("rel_l2 %.9e\n", rel_l2);
    std::printf("openmp_max_abs %.9e\n", openmp_max_abs);
    std::printf("openmp_rel_l2 %.9e\n", openmp_rel_l2);
    std::printf("\n%-32s %14s %14s %14s %12s\n", "kernel", "time/call [s]", "MElem/s", "MDOF/s", "speedup");
    std::printf("%-32s %14.6e %14.3f %14.3f %12.3f\n", "mlir_emitc_apply", mlir_time, mlir_melem, mlir_mdof, mlir_melem / generated_melem);
    std::printf("%-32s %14.6e %14.3f %14.3f %12.3f\n", "mlir_openmp_apply", openmp_time, openmp_melem, openmp_mdof, openmp_melem / generated_melem);
    std::printf("%-32s %14.6e %14.3f %14.3f %12.3f\n", "generated_apply", generated_time, generated_melem, generated_mdof, 1.0);

    return ((max_abs <= 1e-4 || rel_l2 <= 1e-4) && (openmp_max_abs <= 1e-4 || openmp_rel_l2 <= 1e-4)) ? 0 : 1;
}
