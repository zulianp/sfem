#include <OpenCL/opencl.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
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

template <typename T>
static std::vector<T> read_raw(const std::string &path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        std::fprintf(stderr, "Unable to open %s\n", path.c_str());
        std::exit(2);
    }
    const std::streamsize bytes = in.tellg();
    std::vector<T> values(bytes / sizeof(T));
    in.seekg(0, std::ios::beg);
    in.read(reinterpret_cast<char *>(values.data()), bytes);
    return values;
}

static std::string read_text(const std::string &path) {
    std::ifstream in(path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static void check_cl(const cl_int status, const char *what) {
    if (status != CL_SUCCESS) {
        std::fprintf(stderr, "%s failed with OpenCL status %d\n", what, status);
        std::exit(3);
    }
}

static void compute_adjugate_and_det(const std::vector<int> &connectivity,
                                     const std::vector<float> &x,
                                     const std::vector<float> &y,
                                     const std::vector<float> &z,
                                     std::vector<float> &adj,
                                     std::vector<float> &det,
                                     std::vector<std::vector<geom_t>> &generated_adj,
                                     std::vector<geom_t> &generated_det) {
    for (ptrdiff_t e = 0; e < static_cast<ptrdiff_t>(det.size()); ++e) {
        const int n0 = connectivity[4 * e + 0];
        const int n1 = connectivity[4 * e + 1];
        const int n2 = connectivity[4 * e + 2];
        const int n3 = connectivity[4 * e + 3];
        const float j00 = x[n1] - x[n0], j10 = y[n1] - y[n0], j20 = z[n1] - z[n0];
        const float j01 = x[n2] - x[n0], j11 = y[n2] - y[n0], j21 = z[n2] - z[n0];
        const float j02 = x[n3] - x[n0], j12 = y[n3] - y[n0], j22 = z[n3] - z[n0];
        const float a[9] = {
            j11 * j22 - j12 * j21,
            j02 * j21 - j01 * j22,
            j01 * j12 - j02 * j11,
            j12 * j20 - j10 * j22,
            j00 * j22 - j02 * j20,
            j02 * j10 - j00 * j12,
            j10 * j21 - j11 * j20,
            j01 * j20 - j00 * j21,
            j00 * j11 - j01 * j10,
        };
        const float d = j00 * a[0] + j01 * a[3] + j02 * a[6];
        det[e] = d;
        generated_det[e] = d;
        for (int k = 0; k < 9; ++k) {
            adj[9 * e + k] = a[k];
            generated_adj[k][e] = a[k];
        }
    }
}

struct OpenCLEventProfile {
    double queued_to_submit;
    double submit_to_start;
    double start_to_end;
};

static OpenCLEventProfile event_profile(cl_event event) {
    cl_ulong queued = 0, submit = 0, begin = 0, end = 0;
    check_cl(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED, sizeof(queued), &queued, nullptr), "event queued");
    check_cl(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_SUBMIT, sizeof(submit), &submit, nullptr), "event submit");
    check_cl(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(begin), &begin, nullptr), "event start");
    check_cl(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr), "event end");
    return {
        static_cast<double>(submit - queued) * 1e-9,
        static_cast<double>(begin - submit) * 1e-9,
        static_cast<double>(end - begin) * 1e-9,
    };
}

static double wall_seconds() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv) {
    if (argc != 9) {
        std::fprintf(stderr, "usage: %s <mesh_dir> <kernel.cl> <nelements> <nnodes> <max_node_degree> <repeat> <generated_melem> <opencl_build_options>\n", argv[0]);
        return 2;
    }
    const std::string mesh_dir = argv[1];
    const std::string kernel_path = argv[2];
    const ptrdiff_t nelements = std::strtoll(argv[3], nullptr, 10);
    const ptrdiff_t nnodes = std::strtoll(argv[4], nullptr, 10);
    const int max_node_degree = std::atoi(argv[5]);
    const int repeat = std::atoi(argv[6]);
    const double generated_melem = std::atof(argv[7]);
    const std::string build_options = argv[8];
    const float mu = 3.0f;
    const float lmbda = 2.0f;

    auto i0 = read_raw<int32_t>(mesh_dir + "/i0.int32");
    auto i1 = read_raw<int32_t>(mesh_dir + "/i1.int32");
    auto i2 = read_raw<int32_t>(mesh_dir + "/i2.int32");
    auto i3 = read_raw<int32_t>(mesh_dir + "/i3.int32");
    auto x = read_raw<float>(mesh_dir + "/x.float32");
    auto y = read_raw<float>(mesh_dir + "/y.float32");
    auto z = read_raw<float>(mesh_dir + "/z.float32");

    std::vector<int> connectivity(4 * nelements);
    std::vector<idx_t> e0(nelements), e1(nelements), e2(nelements), e3(nelements);
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        connectivity[4 * e + 0] = i0[e];
        connectivity[4 * e + 1] = i1[e];
        connectivity[4 * e + 2] = i2[e];
        connectivity[4 * e + 3] = i3[e];
        e0[e] = i0[e]; e1[e] = i1[e]; e2[e] = i2[e]; e3[e] = i3[e];
    }
    idx_t *elements[4] = {e0.data(), e1.data(), e2.data(), e3.data()};

    std::vector<int> node_degree(nnodes, 0), node_to_element_map(nnodes * max_node_degree, 0), node_to_local_idx(nnodes * max_node_degree, 0);
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        for (int local = 0; local < 4; ++local) {
            const int node = connectivity[4 * e + local];
            const int slot = node_degree[node]++;
            node_to_element_map[node * max_node_degree + slot] = static_cast<int>(e);
            node_to_local_idx[node * max_node_degree + slot] = local;
        }
    }

    std::vector<float> direction(3 * nnodes), output(3 * nnodes), gpu_output(3 * nnodes), scratch(12 * nelements);
    for (ptrdiff_t i = 0; i < 3 * nnodes; ++i) direction[i] = static_cast<float>((static_cast<int>(i % 97) + 1) / 97.0);
    std::vector<float> adj(9 * nelements), det(nelements);
    std::vector<std::vector<geom_t>> generated_adj(9, std::vector<geom_t>(nelements));
    std::vector<geom_t> generated_det(nelements);
    compute_adjugate_and_det(connectivity, x, y, z, adj, det, generated_adj, generated_det);

    linear_elasticity_tet4_tet4_apply_affine_mesh_soa_float(nelements, nnodes, elements,
                                                            generated_adj[0].data(), generated_adj[1].data(), generated_adj[2].data(),
                                                            generated_adj[3].data(), generated_adj[4].data(), generated_adj[5].data(),
                                                            generated_adj[6].data(), generated_adj[7].data(), generated_adj[8].data(),
                                                            generated_det.data(), mu, lmbda, 3,
                                                            direction.data(), direction.data() + 1, direction.data() + 2,
                                                            3, output.data(), output.data() + 1, output.data() + 2);

    cl_int status = CL_SUCCESS;
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_uint nplatforms = 0;
    check_cl(clGetPlatformIDs(0, nullptr, &nplatforms), "clGetPlatformIDs(count)");
    std::vector<cl_platform_id> platforms(nplatforms);
    check_cl(clGetPlatformIDs(nplatforms, platforms.data(), nullptr), "clGetPlatformIDs(list)");
    for (cl_platform_id candidate_platform : platforms) {
        cl_uint ndevices = 0;
        status = clGetDeviceIDs(candidate_platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &ndevices);
        if (status != CL_SUCCESS || ndevices == 0) {
            continue;
        }
        std::vector<cl_device_id> devices(ndevices);
        check_cl(clGetDeviceIDs(candidate_platform, CL_DEVICE_TYPE_ALL, ndevices, devices.data(), nullptr), "clGetDeviceIDs(list)");
        for (cl_device_id candidate_device : devices) {
            cl_device_type type = 0;
            check_cl(clGetDeviceInfo(candidate_device, CL_DEVICE_TYPE, sizeof(type), &type, nullptr), "clGetDeviceInfo(type)");
            if (type & CL_DEVICE_TYPE_GPU) {
                platform = candidate_platform;
                device = candidate_device;
                break;
            }
        }
        if (device) {
            break;
        }
    }
    if (!device) {
        std::fprintf(stderr, "No OpenCL GPU device was found\n");
        return 3;
    }
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &status);
    check_cl(status, "clCreateContext");
    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
    check_cl(status, "clCreateCommandQueue");

    const std::string source = read_text(kernel_path);
    const char *src = source.c_str();
    const size_t src_size = source.size();
    cl_program program = clCreateProgramWithSource(context, 1, &src, &src_size, &status);
    check_cl(status, "clCreateProgramWithSource");
    status = clBuildProgram(program, 1, &device, build_options.c_str(), nullptr, nullptr);
    if (status != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string log(log_size, '\0');
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log.size(), log.data(), nullptr);
        std::fprintf(stderr, "%s\n", log.c_str());
        check_cl(status, "clBuildProgram");
    }
    cl_kernel map_kernel = clCreateKernel(program, "tet4_le_map", &status);
    check_cl(status, "clCreateKernel(map)");
    cl_kernel reduce_kernel = clCreateKernel(program, "tet4_le_reduce", &status);
    check_cl(status, "clCreateKernel(reduce)");

    auto buffer = [&](size_t bytes, void *host, cl_mem_flags flags) {
        cl_int st = CL_SUCCESS;
        cl_mem mem = clCreateBuffer(context, flags | CL_MEM_COPY_HOST_PTR, bytes, host, &st);
        check_cl(st, "clCreateBuffer");
        return mem;
    };
    cl_mem d_conn = buffer(connectivity.size() * sizeof(int), connectivity.data(), CL_MEM_READ_ONLY);
    cl_mem d_dir = buffer(direction.size() * sizeof(float), direction.data(), CL_MEM_READ_ONLY);
    cl_mem d_adj = buffer(adj.size() * sizeof(float), adj.data(), CL_MEM_READ_ONLY);
    cl_mem d_det = buffer(det.size() * sizeof(float), det.data(), CL_MEM_READ_ONLY);
    cl_mem d_scratch = buffer(scratch.size() * sizeof(float), scratch.data(), CL_MEM_READ_WRITE);
    cl_mem d_degree = buffer(node_degree.size() * sizeof(int), node_degree.data(), CL_MEM_READ_ONLY);
    cl_mem d_map = buffer(node_to_element_map.size() * sizeof(int), node_to_element_map.data(), CL_MEM_READ_ONLY);
    cl_mem d_local = buffer(node_to_local_idx.size() * sizeof(int), node_to_local_idx.data(), CL_MEM_READ_ONLY);
    cl_mem d_out = buffer(gpu_output.size() * sizeof(float), gpu_output.data(), CL_MEM_WRITE_ONLY);

    int arg = 0;
    clSetKernelArg(map_kernel, arg++, sizeof(d_conn), &d_conn);
    clSetKernelArg(map_kernel, arg++, sizeof(d_dir), &d_dir);
    clSetKernelArg(map_kernel, arg++, sizeof(d_adj), &d_adj);
    clSetKernelArg(map_kernel, arg++, sizeof(d_det), &d_det);
    clSetKernelArg(map_kernel, arg++, sizeof(lmbda), &lmbda);
    clSetKernelArg(map_kernel, arg++, sizeof(mu), &mu);
    clSetKernelArg(map_kernel, arg++, sizeof(d_scratch), &d_scratch);
    arg = 0;
    clSetKernelArg(reduce_kernel, arg++, sizeof(d_scratch), &d_scratch);
    clSetKernelArg(reduce_kernel, arg++, sizeof(d_degree), &d_degree);
    clSetKernelArg(reduce_kernel, arg++, sizeof(d_map), &d_map);
    clSetKernelArg(reduce_kernel, arg++, sizeof(d_local), &d_local);
    clSetKernelArg(reduce_kernel, arg++, sizeof(max_node_degree), &max_node_degree);
    clSetKernelArg(reduce_kernel, arg++, sizeof(d_out), &d_out);

    const size_t map_global = static_cast<size_t>(nelements);
    const size_t reduce_global = static_cast<size_t>(nnodes);
    double map_queued_to_submit = 0.0, map_submit_to_start = 0.0, map_start_to_end = 0.0;
    double reduce_queued_to_submit = 0.0, reduce_submit_to_start = 0.0, reduce_start_to_end = 0.0;
    double host_elapsed = 0.0;
    for (int r = 0; r < repeat; ++r) {
        cl_event ev_map = nullptr, ev_reduce = nullptr;
        const double iter_begin = wall_seconds();
        check_cl(clEnqueueNDRangeKernel(queue, map_kernel, 1, nullptr, &map_global, nullptr, 0, nullptr, &ev_map), "enqueue map");
        check_cl(clEnqueueNDRangeKernel(queue, reduce_kernel, 1, nullptr, &reduce_global, nullptr, 1, &ev_map, &ev_reduce), "enqueue reduce");
        check_cl(clFinish(queue), "clFinish");
        host_elapsed += wall_seconds() - iter_begin;
        const OpenCLEventProfile map_profile = event_profile(ev_map);
        const OpenCLEventProfile reduce_profile = event_profile(ev_reduce);
        map_queued_to_submit += map_profile.queued_to_submit;
        map_submit_to_start += map_profile.submit_to_start;
        map_start_to_end += map_profile.start_to_end;
        reduce_queued_to_submit += reduce_profile.queued_to_submit;
        reduce_submit_to_start += reduce_profile.submit_to_start;
        reduce_start_to_end += reduce_profile.start_to_end;
        clReleaseEvent(ev_map);
        clReleaseEvent(ev_reduce);
    }
    check_cl(clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0, gpu_output.size() * sizeof(float), gpu_output.data(), 0, nullptr, nullptr), "read output");

    double diff2 = 0.0, ref2 = 0.0, max_abs = 0.0;
    for (ptrdiff_t i = 0; i < 3 * nnodes; ++i) {
        const double diff = static_cast<double>(gpu_output[i] - output[i]);
        diff2 += diff * diff;
        ref2 += static_cast<double>(output[i]) * static_cast<double>(output[i]);
        max_abs = std::max(max_abs, std::abs(diff));
    }
    const double rel_l2 = std::sqrt(diff2) / std::max(std::sqrt(ref2), 1.0);
    const double inv_repeat = 1.0 / static_cast<double>(repeat);
    const double map_kernel_time = map_start_to_end * inv_repeat;
    const double reduce_kernel_time = reduce_start_to_end * inv_repeat;
    const double event_kernel_time = map_kernel_time + reduce_kernel_time;
    const double time = host_elapsed * inv_repeat;
    const double melem = 1e-6 * static_cast<double>(nelements) / time;
    const double mdof = 1e-6 * static_cast<double>(3 * nnodes) / time;
    std::printf("opencl_max_abs %.9e\n", max_abs);
    std::printf("opencl_rel_l2 %.9e\n", rel_l2);
    std::printf("opencl_build_options %s\n", build_options.c_str());
    std::printf("opencl_map_queued_to_submit_s %.9e\n", map_queued_to_submit * inv_repeat);
    std::printf("opencl_map_submit_to_start_s %.9e\n", map_submit_to_start * inv_repeat);
    std::printf("opencl_map_kernel_s %.9e\n", map_kernel_time);
    std::printf("opencl_reduce_queued_to_submit_s %.9e\n", reduce_queued_to_submit * inv_repeat);
    std::printf("opencl_reduce_submit_to_start_s %.9e\n", reduce_submit_to_start * inv_repeat);
    std::printf("opencl_reduce_kernel_s %.9e\n", reduce_kernel_time);
    std::printf("opencl_event_kernel_time_per_call_s %.9e\n", event_kernel_time);
    std::printf("opencl_host_loop_time_per_call_s %.9e\n", time);
    std::printf("opencl_host_over_event_ratio %.9e\n", time / std::max(event_kernel_time, 1.0e-30));
    std::printf("%-32s %14.6e %14.3f %14.3f %12.3f\n", "mlir_opencl_apply", time, melem, mdof, melem / generated_melem);
    return (max_abs <= 1e-4 || rel_l2 <= 1e-4) ? 0 : 1;
}
