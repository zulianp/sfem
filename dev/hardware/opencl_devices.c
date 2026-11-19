/*
 * opencl_devices - List OpenCL platforms and devices.
 *
 * Standalone utility; build from this directory:
 *   cmake -S . -B build && cmake --build build
 */

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif

#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define KV_WIDTH 28

static void die(const char *msg, cl_int err)
{
    fprintf(stderr, "error: %s (cl_int=%d)\n", msg, (int)err);
    exit(EXIT_FAILURE);
}

static void trim_trailing(char *s)
{
    if (!s) {
        return;
    }
    size_t n = strlen(s);
    while (n > 0 && (s[n - 1] == ' ' || s[n - 1] == '\t' || s[n - 1] == '\n' ||
                     s[n - 1] == '\r')) {
        s[--n] = '\0';
    }
}

static void banner(void)
{
    printf("================================================================================\n");
    printf(" OpenCL Device Report\n");
    printf("================================================================================\n");
}

static void rule(void)
{
    printf("--------------------------------------------------------------------------------\n");
}

static void section(const char *title)
{
    printf("\n%s\n", title);
    rule();
}

static void kv_str(const char *key, const char *value)
{
    printf("  %-*s %s\n", KV_WIDTH, key, value);
}

static void kv_u32(const char *key, cl_uint value)
{
    printf("  %-*s %u\n", KV_WIDTH, key, value);
}

static void kv_size(const char *key, size_t value)
{
    printf("  %-*s %zu\n", KV_WIDTH, key, value);
}

static void kv_bool(const char *key, cl_bool value)
{
    kv_str(key, value ? "yes" : "no");
}

static void format_bytes(char *buf, size_t buflen, cl_ulong bytes)
{
    const double b = (double)bytes;
    if (bytes >= (cl_ulong)1024 * 1024 * 1024) {
        snprintf(buf, buflen, "%.2f GiB (%" PRIu64 " B)", b / (1024.0 * 1024.0 * 1024.0),
                 (uint64_t)bytes);
    } else if (bytes >= (cl_ulong)1024 * 1024) {
        snprintf(buf, buflen, "%.2f MiB (%" PRIu64 " B)", b / (1024.0 * 1024.0),
                 (uint64_t)bytes);
    } else if (bytes >= 1024) {
        snprintf(buf, buflen, "%.2f KiB (%" PRIu64 " B)", b / 1024.0, (uint64_t)bytes);
    } else {
        snprintf(buf, buflen, "%" PRIu64 " B", (uint64_t)bytes);
    }
}

static void kv_bytes(const char *key, cl_ulong bytes)
{
    char buf[80];
    format_bytes(buf, sizeof(buf), bytes);
    kv_str(key, buf);
}

static void kv_mhz(const char *key, cl_uint mhz)
{
    char buf[32];
    snprintf(buf, sizeof(buf), "%u MHz", mhz);
    kv_str(key, buf);
}

static const char *device_type_str(cl_device_type type)
{
    if (type & CL_DEVICE_TYPE_GPU) {
        return "GPU";
    }
    if (type & CL_DEVICE_TYPE_CPU) {
        return "CPU";
    }
    if (type & CL_DEVICE_TYPE_ACCELERATOR) {
        return "Accelerator";
    }
    if (type & CL_DEVICE_TYPE_CUSTOM) {
        return "Custom";
    }
    if (type & CL_DEVICE_TYPE_DEFAULT) {
        return "Default";
    }
    return "Unknown";
}

static const char *mem_cache_type_str(cl_device_mem_cache_type t)
{
    switch (t) {
    case CL_NONE: return "none";
    case CL_READ_ONLY_CACHE: return "read-only";
    case CL_READ_WRITE_CACHE: return "read-write";
    default: return "unknown";
    }
}

static const char *local_mem_type_str(cl_device_local_mem_type t)
{
    switch (t) {
    case CL_LOCAL: return "local";
    case CL_GLOBAL: return "global";
    default: return "unknown";
    }
}

static bool get_string(cl_device_id dev, cl_device_info param, char *buf, size_t buflen)
{
    size_t nbytes = 0;
    if (clGetDeviceInfo(dev, param, 0, NULL, &nbytes) != CL_SUCCESS || nbytes == 0) {
        return false;
    }
    if (nbytes > buflen) {
        nbytes = buflen;
    }
    if (clGetDeviceInfo(dev, param, nbytes, buf, NULL) != CL_SUCCESS) {
        return false;
    }
    trim_trailing(buf);
    return true;
}

static bool get_platform_string(cl_platform_id platform, cl_platform_info param, char *buf,
                                size_t buflen)
{
    size_t nbytes = 0;
    if (clGetPlatformInfo(platform, param, 0, NULL, &nbytes) != CL_SUCCESS || nbytes == 0) {
        return false;
    }
    if (nbytes > buflen) {
        nbytes = buflen;
    }
    if (clGetPlatformInfo(platform, param, nbytes, buf, NULL) != CL_SUCCESS) {
        return false;
    }
    trim_trailing(buf);
    return true;
}

static bool get_u32(cl_device_id dev, cl_device_info param, cl_uint *out)
{
    return clGetDeviceInfo(dev, param, sizeof(*out), out, NULL) == CL_SUCCESS;
}

static bool get_u64(cl_device_id dev, cl_device_info param, cl_ulong *out)
{
    return clGetDeviceInfo(dev, param, sizeof(*out), out, NULL) == CL_SUCCESS;
}

static bool get_bool(cl_device_id dev, cl_device_info param, cl_bool *out)
{
    return clGetDeviceInfo(dev, param, sizeof(*out), out, NULL) == CL_SUCCESS;
}

static bool get_size(cl_device_id dev, cl_device_info param, size_t *out)
{
    return clGetDeviceInfo(dev, param, sizeof(*out), out, NULL) == CL_SUCCESS;
}

static bool get_device_type(cl_device_id dev, cl_device_type *out)
{
    return clGetDeviceInfo(dev, CL_DEVICE_TYPE, sizeof(*out), out, NULL) == CL_SUCCESS;
}

static void print_extension_list(const char *heading, const char *extensions)
{
    if (!extensions || extensions[0] == '\0') {
        return;
    }

    printf("  %s\n", heading);
    const char *p = extensions;
    while (*p != '\0') {
        while (*p == ' ') {
            ++p;
        }
        if (*p == '\0') {
            break;
        }
        const char *start = p;
        while (*p != '\0' && *p != ' ') {
            ++p;
        }
        printf("    - %.*s\n", (int)(p - start), start);
    }
}

static void print_fp_config(const char *key, cl_device_fp_config cfg)
{
    char buf[512];
    size_t pos = 0;
    bool any = false;

    buf[0] = '\0';
#define ADD_FP_FLAG(flag, label)                                              \
    do {                                                                      \
        if ((cfg) & (flag)) {                                                 \
            pos += (size_t)snprintf(buf + pos, sizeof(buf) - pos, "%s%s",      \
                                    any ? ", " : "", (label));                \
            any = true;                                                       \
        }                                                                     \
    } while (0)

    ADD_FP_FLAG(CL_FP_DENORM, "denorm");
    ADD_FP_FLAG(CL_FP_INF_NAN, "inf/nan");
    ADD_FP_FLAG(CL_FP_ROUND_TO_NEAREST, "round-to-nearest");
    ADD_FP_FLAG(CL_FP_ROUND_TO_ZERO, "round-to-zero");
    ADD_FP_FLAG(CL_FP_ROUND_TO_INF, "round-to-inf");
#ifdef CL_FP_ROUND_TO_NEG_INF
    ADD_FP_FLAG(CL_FP_ROUND_TO_NEG_INF, "round-to-neg-inf");
#endif
    ADD_FP_FLAG(CL_FP_FMA, "fma");
#ifdef CL_FP_SOFT_FLOAT
    ADD_FP_FLAG(CL_FP_SOFT_FLOAT, "soft-float");
#endif
    ADD_FP_FLAG(CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT, "cr-divide-sqrt");

#undef ADD_FP_FLAG

    kv_str(key, any ? buf : "none");
}

static void print_queue_props(const char *key, cl_command_queue_properties props)
{
    char buf[256];
    size_t pos = 0;
    bool any = false;

    buf[0] = '\0';
    if (props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) {
        pos += (size_t)snprintf(buf + pos, sizeof(buf) - pos, "%sout-of-order",
                                any ? ", " : "");
        any = true;
    }
    if (props & CL_QUEUE_PROFILING_ENABLE) {
        pos += (size_t)snprintf(buf + pos, sizeof(buf) - pos, "%sprofiling", any ? ", " : "");
        any = true;
    }
#if CL_TARGET_OPENCL_VERSION >= 200
    if (props & CL_QUEUE_ON_DEVICE) {
        pos += (size_t)snprintf(buf + pos, sizeof(buf) - pos, "%son-device", any ? ", " : "");
        any = true;
    }
    if (props & CL_QUEUE_ON_DEVICE_DEFAULT) {
        pos +=
            (size_t)snprintf(buf + pos, sizeof(buf) - pos, "%son-device-default",
                             any ? ", " : "");
        any = true;
    }
#endif
    kv_str(key, any ? buf : "none");
}

static void print_vector_widths(cl_device_id dev)
{
    static const char *names[] = {"char",  "short", "int",  "long",
                                  "float", "double", "half"};
    static const cl_device_info preferred[] = {
        CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR,  CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT,
        CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT,   CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG,
        CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE,
        CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF,
    };
    static const cl_device_info native[] = {
        CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR,  CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT,
        CL_DEVICE_NATIVE_VECTOR_WIDTH_INT,   CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG,
        CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT, CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE,
        CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF,
    };

    printf("  %-*s  %8s  %8s\n", KV_WIDTH, "Vector width (type)", "preferred", "native");
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
        cl_uint pref = 0;
        cl_uint nat = 0;
        if (!get_u32(dev, preferred[i], &pref) || !get_u32(dev, native[i], &nat)) {
            continue;
        }
        printf("  %-*s  %8u  %8u\n", KV_WIDTH, names[i], pref, nat);
    }
}

static void print_platform(cl_platform_id platform, cl_uint index, cl_uint device_count)
{
    char name[1024];
    char vendor[1024];
    char version[1024];
    char profile[256];
    char extensions[8192];

    get_platform_string(platform, CL_PLATFORM_NAME, name, sizeof(name));
    get_platform_string(platform, CL_PLATFORM_VENDOR, vendor, sizeof(vendor));
    get_platform_string(platform, CL_PLATFORM_VERSION, version, sizeof(version));
    get_platform_string(platform, CL_PLATFORM_PROFILE, profile, sizeof(profile));
    get_platform_string(platform, CL_PLATFORM_EXTENSIONS, extensions, sizeof(extensions));

    printf("\n");
    rule();
    printf(" Platform %u: %s\n", index, name);
    rule();
    kv_str("Vendor", vendor);
    kv_str("Version", version);
    kv_str("Profile", profile);
    kv_u32("Devices", device_count);
    print_extension_list("Extensions:", extensions);
}

static void print_device_identity(cl_device_id dev)
{
    char name[1024];
    char vendor[1024];
    char driver[1024];
    char profile[256];
    char version[256];
    char opencl_c[256];
    cl_device_type type = 0;

    if (get_string(dev, CL_DEVICE_NAME, name, sizeof(name))) {
        kv_str("Name", name);
    }
    if (get_string(dev, CL_DEVICE_VENDOR, vendor, sizeof(vendor))) {
        kv_str("Vendor", vendor);
    }
#ifdef CL_DEVICE_DRIVER_VERSION
    if (get_string(dev, CL_DEVICE_DRIVER_VERSION, driver, sizeof(driver))) {
        kv_str("Driver", driver);
    }
#else
    if (get_string(dev, CL_DRIVER_VERSION, driver, sizeof(driver))) {
        kv_str("Driver", driver);
    }
#endif
    if (get_device_type(dev, &type)) {
        kv_str("Type", device_type_str(type));
    }
    if (get_string(dev, CL_DEVICE_PROFILE, profile, sizeof(profile))) {
        kv_str("Profile", profile);
    }
    if (get_string(dev, CL_DEVICE_VERSION, version, sizeof(version))) {
        kv_str("OpenCL version", version);
    }
    if (get_string(dev, CL_DEVICE_OPENCL_C_VERSION, opencl_c, sizeof(opencl_c))) {
        kv_str("OpenCL C version", opencl_c);
    }

#if CL_TARGET_OPENCL_VERSION >= 300
    cl_version numeric = 0;
    if (clGetDeviceInfo(dev, CL_DEVICE_NUMERIC_VERSION, sizeof(numeric), &numeric, NULL) ==
        CL_SUCCESS) {
        char buf[32];
        snprintf(buf, sizeof(buf), "%u.%u.%u", CL_VERSION_MAJOR(numeric),
                 CL_VERSION_MINOR(numeric), CL_VERSION_PATCH(numeric));
        kv_str("Numeric version", buf);
    }
#endif
}

static void print_device_compute(cl_device_id dev)
{
    cl_uint units = 0;
    cl_uint clock = 0;
    cl_uint dims = 0;
    size_t max_wg = 0;
    size_t max_param = 0;
    size_t max_items[3] = {0, 0, 0};
    size_t item_bytes = 0;

    if (get_u32(dev, CL_DEVICE_MAX_COMPUTE_UNITS, &units)) {
        kv_u32("Compute units", units);
    }
    if (get_u32(dev, CL_DEVICE_MAX_CLOCK_FREQUENCY, &clock)) {
        kv_mhz("Max clock", clock);
    }
    if (get_u32(dev, CL_DEVICE_ADDRESS_BITS, &units)) {
        kv_u32("Address bits", units);
    }
    if (get_u32(dev, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, &dims)) {
        kv_u32("Work-item dimensions", dims);
    }
    if (get_size(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, &max_wg)) {
        kv_size("Max work-group size", max_wg);
    }
    if (get_size(dev, CL_DEVICE_MAX_PARAMETER_SIZE, &max_param)) {
        kv_size("Max parameter size", max_param);
    }
    if (clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_ITEM_SIZES, 0, NULL, &item_bytes) ==
            CL_SUCCESS &&
        item_bytes > 0) {
        size_t count = item_bytes / sizeof(size_t);
        if (count <= 3 &&
            clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_ITEM_SIZES, item_bytes, max_items, NULL) ==
                CL_SUCCESS) {
            char buf[64];
            if (count == 1) {
                snprintf(buf, sizeof(buf), "%zu", max_items[0]);
            } else if (count == 2) {
                snprintf(buf, sizeof(buf), "%zu x %zu", max_items[0], max_items[1]);
            } else {
                snprintf(buf, sizeof(buf), "%zu x %zu x %zu", max_items[0], max_items[1],
                         max_items[2]);
            }
            kv_str("Max work-item sizes", buf);
        }
    }

    print_vector_widths(dev);
}

static void print_device_memory(cl_device_id dev)
{
    cl_ulong global = 0;
    cl_ulong max_alloc = 0;
    cl_ulong constant = 0;
    cl_ulong local = 0;
    cl_ulong cache = 0;
    cl_uint cacheline = 0;
    cl_device_mem_cache_type cache_type = CL_NONE;
    cl_device_local_mem_type local_type = CL_LOCAL;

    if (get_u64(dev, CL_DEVICE_GLOBAL_MEM_SIZE, &global)) {
        kv_bytes("Global memory", global);
    }
    if (get_u64(dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE, &max_alloc)) {
        kv_bytes("Max allocation", max_alloc);
    }
    if (get_u64(dev, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, &constant)) {
        kv_bytes("Max constant buffer", constant);
    }
    if (get_u64(dev, CL_DEVICE_LOCAL_MEM_SIZE, &local)) {
        kv_bytes("Local memory", local);
    }
    if (clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, sizeof(cache_type), &cache_type,
                        NULL) == CL_SUCCESS) {
        kv_str("Global memory cache", mem_cache_type_str(cache_type));
    }
    if (cache_type != CL_NONE && get_u64(dev, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, &cache)) {
        kv_bytes("Cache size", cache);
    }
    if (get_u32(dev, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, &cacheline) && cacheline > 0) {
        kv_u32("Cache line size", cacheline);
    }
    if (clGetDeviceInfo(dev, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(local_type), &local_type,
                        NULL) == CL_SUCCESS) {
        kv_str("Local memory type", local_mem_type_str(local_type));
    }
    if (get_u32(dev, CL_DEVICE_MEM_BASE_ADDR_ALIGN, &cacheline)) {
        kv_u32("Mem base addr align (bits)", cacheline);
    }
#if CL_TARGET_OPENCL_VERSION >= 200
    {
        cl_bool unified = CL_FALSE;
        if (get_bool(dev, CL_DEVICE_HOST_UNIFIED_MEMORY, &unified)) {
            kv_bool("Unified host memory", unified);
        }
    }
#endif
}

static void print_device_images(cl_device_id dev)
{
    cl_bool support = CL_FALSE;
    if (!get_bool(dev, CL_DEVICE_IMAGE_SUPPORT, &support) || !support) {
        kv_str("Image support", "no");
        return;
    }

    kv_str("Image support", "yes");

    cl_uint value = 0;
    if (get_u32(dev, CL_DEVICE_MAX_READ_IMAGE_ARGS, &value)) {
        kv_u32("Max read image args", value);
    }
    if (get_u32(dev, CL_DEVICE_MAX_WRITE_IMAGE_ARGS, &value)) {
        kv_u32("Max write image args", value);
    }
    if (get_u32(dev, CL_DEVICE_MAX_SAMPLERS, &value)) {
        kv_u32("Max samplers", value);
    }

    if (get_u32(dev, CL_DEVICE_IMAGE2D_MAX_WIDTH, &value)) {
        kv_u32("Image2D max width", value);
    }
    if (get_u32(dev, CL_DEVICE_IMAGE2D_MAX_HEIGHT, &value)) {
        kv_u32("Image2D max height", value);
    }
    if (get_u32(dev, CL_DEVICE_IMAGE3D_MAX_WIDTH, &value)) {
        kv_u32("Image3D max width", value);
    }
    if (get_u32(dev, CL_DEVICE_IMAGE3D_MAX_HEIGHT, &value)) {
        kv_u32("Image3D max height", value);
    }
    if (get_u32(dev, CL_DEVICE_IMAGE3D_MAX_DEPTH, &value)) {
        kv_u32("Image3D max depth", value);
    }
}

static void print_device_queues(cl_device_id dev)
{
    cl_device_exec_capabilities exec = 0;
    cl_command_queue_properties props = 0;
    size_t timer_res = 0;

    if (clGetDeviceInfo(dev, CL_DEVICE_EXECUTION_CAPABILITIES, sizeof(exec), &exec, NULL) ==
        CL_SUCCESS) {
        char buf[64];
        size_t pos = 0;
        bool any = false;
        buf[0] = '\0';
        if (exec & CL_EXEC_KERNEL) {
            pos += (size_t)snprintf(buf + pos, sizeof(buf) - pos, "%skernel", any ? ", " : "");
            any = true;
        }
        if (exec & CL_EXEC_NATIVE_KERNEL) {
            pos += (size_t)snprintf(buf + pos, sizeof(buf) - pos, "%snative kernel",
                                     any ? ", " : "");
            any = true;
        }
        kv_str("Execution", any ? buf : "none");
    }

    if (clGetDeviceInfo(dev, CL_DEVICE_QUEUE_PROPERTIES, sizeof(props), &props, NULL) ==
        CL_SUCCESS) {
        print_queue_props("Queue properties", props);
    }
#if CL_TARGET_OPENCL_VERSION >= 200
    if (clGetDeviceInfo(dev, CL_DEVICE_QUEUE_ON_HOST_PROPERTIES, sizeof(props), &props,
                        NULL) == CL_SUCCESS) {
        print_queue_props("On-host queue props", props);
    }
    if (clGetDeviceInfo(dev, CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES, sizeof(props), &props,
                        NULL) == CL_SUCCESS) {
        print_queue_props("On-device queue props", props);
    }
#endif
    if (get_size(dev, CL_DEVICE_PROFILING_TIMER_RESOLUTION, &timer_res)) {
        char buf[32];
        snprintf(buf, sizeof(buf), "%zu ns", timer_res);
        kv_str("Profiling timer res.", buf);
    }
}

static void print_device_fp(cl_device_id dev)
{
    cl_device_fp_config single = 0;
    cl_device_fp_config doub = 0;

    if (clGetDeviceInfo(dev, CL_DEVICE_SINGLE_FP_CONFIG, sizeof(single), &single, NULL) ==
        CL_SUCCESS) {
        print_fp_config("Single precision", single);
    }
    if (clGetDeviceInfo(dev, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(doub), &doub, NULL) ==
        CL_SUCCESS) {
        print_fp_config("Double precision", doub);
    }
#if CL_TARGET_OPENCL_VERSION >= 200
    {
        cl_device_fp_config half = 0;
        if (clGetDeviceInfo(dev, CL_DEVICE_HALF_FP_CONFIG, sizeof(half), &half, NULL) ==
            CL_SUCCESS) {
            print_fp_config("Half precision", half);
        }
    }
#endif
}

static void print_device_capabilities(cl_device_id dev)
{
    cl_bool value = CL_FALSE;

    if (get_bool(dev, CL_DEVICE_AVAILABLE, &value)) {
        kv_bool("Available", value);
    }
    if (get_bool(dev, CL_DEVICE_COMPILER_AVAILABLE, &value)) {
        kv_bool("Online compiler", value);
    }
#if CL_TARGET_OPENCL_VERSION >= 200
    if (get_bool(dev, CL_DEVICE_LINKER_AVAILABLE, &value)) {
        kv_bool("Online linker", value);
    }
#endif
    if (get_bool(dev, CL_DEVICE_ENDIAN_LITTLE, &value)) {
        kv_str("Endianness", value ? "little" : "big");
    }
    if (get_bool(dev, CL_DEVICE_ERROR_CORRECTION_SUPPORT, &value) && value) {
        kv_bool("Error correction", value);
    }
#if CL_TARGET_OPENCL_VERSION >= 210
    if (get_bool(dev, CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT, &value) && value) {
        kv_bool("Non-uniform work-groups", value);
    }
#endif
}

#if CL_TARGET_OPENCL_VERSION >= 300
static void print_device_version_lists(cl_device_id dev)
{
    size_t nbytes = 0;

    if (clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS_WITH_VERSION, 0, NULL, &nbytes) ==
            CL_SUCCESS &&
        nbytes > 0) {
        size_t count = nbytes / sizeof(cl_name_version);
        cl_name_version *items = (cl_name_version *)malloc(nbytes);
        if (items &&
            clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS_WITH_VERSION, nbytes, items, NULL) ==
                CL_SUCCESS) {
            printf("  Extensions (versioned):\n");
            for (size_t i = 0; i < count; ++i) {
                printf("    - %s (%u.%u)\n", items[i].name,
                       CL_VERSION_MAJOR(items[i].version),
                       CL_VERSION_MINOR(items[i].version));
            }
        }
        free(items);
    }

    if (clGetDeviceInfo(dev, CL_DEVICE_OPENCL_C_ALL_VERSIONS, 0, NULL, &nbytes) ==
            CL_SUCCESS &&
        nbytes > 0) {
        size_t count = nbytes / sizeof(cl_name_version);
        cl_name_version *items = (cl_name_version *)malloc(nbytes);
        if (items &&
            clGetDeviceInfo(dev, CL_DEVICE_OPENCL_C_ALL_VERSIONS, nbytes, items, NULL) ==
                CL_SUCCESS) {
            printf("  Supported OpenCL C versions:\n");
            for (size_t i = 0; i < count; ++i) {
                printf("    - %u.%u\n", CL_VERSION_MAJOR(items[i].version),
                       CL_VERSION_MINOR(items[i].version));
            }
        }
        free(items);
    }
}
#endif

static void print_device(cl_device_id dev, cl_uint platform_index, cl_uint device_index)
{
    char name[1024];
    char extensions[8192];
    cl_device_type type = 0;

    if (!get_string(dev, CL_DEVICE_NAME, name, sizeof(name))) {
        snprintf(name, sizeof(name), "device %u", device_index);
    }
    get_device_type(dev, &type);

    printf("\n");
    rule();
    printf(" Device %u.%u: %s  [%s]\n", platform_index, device_index, name,
           device_type_str(type));
    rule();

    section("Identity");
    print_device_identity(dev);

    section("Compute");
    print_device_compute(dev);

    section("Memory");
    print_device_memory(dev);

    section("Images");
    print_device_images(dev);

    section("Queues & Execution");
    print_device_queues(dev);

    section("Floating Point");
    print_device_fp(dev);

    section("Capabilities");
    print_device_capabilities(dev);

    if (get_string(dev, CL_DEVICE_EXTENSIONS, extensions, sizeof(extensions))) {
        print_extension_list("Extensions:", extensions);
    }

#if CL_TARGET_OPENCL_VERSION >= 300
    print_device_version_lists(dev);
#endif
}

int main(void)
{
    cl_uint num_platforms = 0;
    cl_int err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS) {
        die("clGetPlatformIDs(count)", err);
    }

    banner();

    if (num_platforms == 0) {
        printf("\nNo OpenCL platforms found.\n\n");
        return EXIT_SUCCESS;
    }

    cl_platform_id *platforms = (cl_platform_id *)calloc((size_t)num_platforms,
                                                         sizeof(cl_platform_id));
    if (!platforms) {
        die("calloc", CL_OUT_OF_HOST_MEMORY);
    }

    err = clGetPlatformIDs(num_platforms, platforms, NULL);
    if (err != CL_SUCCESS) {
        free(platforms);
        die("clGetPlatformIDs", err);
    }

    cl_uint total_devices = 0;
    for (cl_uint p = 0; p < num_platforms; ++p) {
        cl_uint n = 0;
        if (clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, NULL, &n) == CL_SUCCESS) {
            total_devices += n;
        }
    }

    printf("\nFound %u platform%s, %u device%s\n", num_platforms,
           num_platforms == 1 ? "" : "s", total_devices, total_devices == 1 ? "" : "s");

    for (cl_uint p = 0; p < num_platforms; ++p) {
        cl_uint num_devices = 0;
        err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        if (err == CL_DEVICE_NOT_FOUND) {
            print_platform(platforms[p], p, 0);
            continue;
        }
        if (err != CL_SUCCESS) {
            free(platforms);
            die("clGetDeviceIDs(count)", err);
        }

        print_platform(platforms[p], p, num_devices);

        cl_device_id *devices = (cl_device_id *)calloc((size_t)num_devices, sizeof(cl_device_id));
        if (!devices) {
            free(platforms);
            die("calloc", CL_OUT_OF_HOST_MEMORY);
        }

        err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
        if (err != CL_SUCCESS) {
            free(devices);
            free(platforms);
            die("clGetDeviceIDs", err);
        }

        for (cl_uint d = 0; d < num_devices; ++d) {
            print_device(devices[d], p, d);
        }

        free(devices);
    }

    printf("\n");
    rule();
    printf("\n");

    free(platforms);
    return EXIT_SUCCESS;
}
