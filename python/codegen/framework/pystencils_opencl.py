from dataclasses import dataclass
import argparse
import ctypes
import importlib.util

import numpy as np

try:
    from .kernel_ast import (
        AssignmentNode,
        GatherNode,
        KernelAST,
        ScatterNode,
        buffer_access,
        expr_ref,
        symbol_ref,
    )
    from .kernel_ast_pystencils import PystencilsKernelASTAdapter
except ImportError:
    from kernel_ast import (
        AssignmentNode,
        GatherNode,
        KernelAST,
        ScatterNode,
        buffer_access,
        expr_ref,
        symbol_ref,
    )
    from kernel_ast_pystencils import PystencilsKernelASTAdapter


@dataclass(frozen=True)
class OpenCLSmokeResult:
    success: bool
    reason: str = ""
    platform_name: str = ""
    device_name: str = ""
    source: str = ""
    max_error: float = 0.0

    def to_dict(self):
        return {
            "success": self.success,
            "reason": self.reason,
            "platform_name": self.platform_name,
            "device_name": self.device_name,
            "max_error": self.max_error,
        }


class PystencilsOpenCLFloat32Smoke:
    kernel_name = "sfem_pystencils_float32_add"

    def __init__(self, n=1024):
        self.n = int(n)

    def generate_kernel_source(self):
        body = self._generate_body()
        return "\n".join(
            (
                "__kernel void %s(" % self.kernel_name,
                "    __global const float *a,",
                "    __global const float *b,",
                "    __global float *out) {",
                "    const int gid = get_global_id(0);",
                body,
                "}",
            )
        )

    def run(self):
        if importlib.util.find_spec("pyopencl") is None:
            return OpenCLSmokeResult(False, "pyopencl is not installed")

        import pyopencl as cl

        device_result = self._find_apple_gpu_device(cl)
        if not device_result.success:
            return device_result

        platform, device = device_result.platform_name, device_result.device_name
        cl_platform, cl_device = device_result._platform, device_result._device
        source = self.generate_kernel_source()

        context = cl.Context(devices=[cl_device])
        queue = cl.CommandQueue(context)
        program = cl.Program(context, source).build()

        a = np.linspace(0.0, 1.0, self.n, dtype=np.float32)
        b = np.linspace(1.0, 2.0, self.n, dtype=np.float32)
        out = np.empty_like(a)

        flags = cl.mem_flags
        a_buf = cl.Buffer(context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=a)
        b_buf = cl.Buffer(context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=b)
        out_buf = cl.Buffer(context, flags.WRITE_ONLY, out.nbytes)

        kernel = getattr(program, self.kernel_name)
        kernel(queue, (self.n,), None, a_buf, b_buf, out_buf)
        cl.enqueue_copy(queue, out, out_buf).wait()

        expected = a + b
        max_error = float(np.max(np.abs(out - expected)))
        if not np.allclose(out, expected, rtol=0, atol=0):
            return OpenCLSmokeResult(
                False,
                "float32 OpenCL result mismatch",
                platform,
                device,
                source,
                max_error,
            )

        return OpenCLSmokeResult(True, "", platform, device, source, max_error)

    def _generate_body(self):
        ast = KernelAST(
            "pystencils_opencl_float32_body",
            nodes=(
                GatherNode(symbol_ref("av"), symbol_ref("a"), symbol_ref("gid")),
                GatherNode(symbol_ref("bv"), symbol_ref("b"), symbol_ref("gid")),
                AssignmentNode(symbol_ref("ov"), expr_ref("av + bv", "float32_add")),
                ScatterNode(buffer_access("out", symbol_ref("gid")), symbol_ref("ov"), "=", False),
            ),
        )
        result = PystencilsKernelASTAdapter(default_float_type="float32").generate_c(ast)
        if not result.success:
            raise RuntimeError("; ".join(result.diagnostics))
        return _strip_c_block(result.lowered, indent="    ")

    def _find_apple_gpu_device(self, cl):
        for platform in cl.get_platforms():
            if "apple" not in (platform.vendor + " " + platform.name).lower():
                continue
            try:
                devices = platform.get_devices(device_type=cl.device_type.GPU)
            except Exception as exc:
                native = _native_apple_opencl_device_query()
                native_reason = ""
                if native:
                    native_reason = "; native OpenCL device query: %s" % native
                return OpenCLSmokeResult(
                    False,
                    "Apple OpenCL platform is present but GPU device query failed: %s%s"
                    % (exc, native_reason),
                    platform.name,
                    "",
                )
            for device in devices:
                if device.type & cl.device_type.GPU:
                    result = OpenCLSmokeResult(
                        True,
                        "",
                        platform.name,
                        device.name,
                    )
                    object.__setattr__(result, "_platform", platform)
                    object.__setattr__(result, "_device", device)
                    return result
        return OpenCLSmokeResult(False, "no Apple OpenCL GPU device found")


def _strip_c_block(source, indent):
    lines = source.splitlines()
    if lines and lines[0].strip() == "{":
        lines = lines[1:]
    if lines and lines[-1].strip() == "}":
        lines = lines[:-1]

    stripped = []
    for line in lines:
        if line.startswith("   "):
            line = line[3:]
        stripped.append(indent + line)
    return "\n".join(stripped)


def _native_apple_opencl_device_query():
    try:
        lib = ctypes.CDLL("/System/Library/Frameworks/OpenCL.framework/OpenCL")
    except OSError as exc:
        return "OpenCL framework load failed: %s" % exc

    cl_uint = ctypes.c_uint
    cl_int = ctypes.c_int
    cl_platform_id = ctypes.c_void_p
    cl_device_id = ctypes.c_void_p
    cl_device_type = ctypes.c_ulong

    lib.clGetPlatformIDs.argtypes = [
        cl_uint,
        ctypes.POINTER(cl_platform_id),
        ctypes.POINTER(cl_uint),
    ]
    lib.clGetPlatformIDs.restype = cl_int
    lib.clGetPlatformInfo.argtypes = [
        cl_platform_id,
        cl_uint,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.clGetPlatformInfo.restype = cl_int
    lib.clGetDeviceIDs.argtypes = [
        cl_platform_id,
        cl_device_type,
        cl_uint,
        ctypes.POINTER(cl_device_id),
        ctypes.POINTER(cl_uint),
    ]
    lib.clGetDeviceIDs.restype = cl_int

    platform_count = cl_uint()
    err = lib.clGetPlatformIDs(0, None, ctypes.byref(platform_count))
    if err != 0:
        return "clGetPlatformIDs count err=%d" % err
    if platform_count.value == 0:
        return "clGetPlatformIDs reported zero platforms"

    platforms = (cl_platform_id * platform_count.value)()
    err = lib.clGetPlatformIDs(platform_count.value, platforms, None)
    if err != 0:
        return "clGetPlatformIDs list err=%d" % err

    apple_platform = None
    for platform in platforms:
        name = _native_platform_string(lib, platform, 0x0902)
        vendor = _native_platform_string(lib, platform, 0x0903)
        if "apple" in ("%s %s" % (name, vendor)).lower():
            apple_platform = platform
            break

    if apple_platform is None:
        return "no native Apple OpenCL platform found"

    query_parts = []
    for name, device_type in (
        ("DEFAULT", 1 << 0),
        ("CPU", 1 << 1),
        ("GPU", 1 << 2),
        ("ACCELERATOR", 1 << 3),
        ("ALL", 0xFFFFFFFF),
    ):
        device_count = cl_uint()
        count_err = lib.clGetDeviceIDs(
            apple_platform,
            device_type,
            0,
            None,
            ctypes.byref(device_count),
        )
        device = cl_device_id()
        one_device_count = cl_uint()
        one_err = lib.clGetDeviceIDs(
            apple_platform,
            device_type,
            1,
            ctypes.byref(device),
            ctypes.byref(one_device_count),
        )
        query_parts.append(
            "%s count_only_err=%d count=%d one_device_err=%d one_device_count=%d"
            % (name, count_err, device_count.value, one_err, one_device_count.value)
        )
    return ", ".join(query_parts)


def _native_platform_string(lib, platform, key):
    size = ctypes.c_size_t()
    err = lib.clGetPlatformInfo(platform, key, 0, None, ctypes.byref(size))
    if err != 0:
        return ""
    buffer = ctypes.create_string_buffer(size.value)
    err = lib.clGetPlatformInfo(platform, key, size.value, buffer, None)
    if err != 0:
        return ""
    return buffer.value.decode(errors="replace")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run or print a pystencils-generated float32 OpenCL smoke kernel."
    )
    parser.add_argument("--n", type=int, default=1024, help="number of float32 entries")
    parser.add_argument(
        "--source",
        action="store_true",
        help="print generated OpenCL source before running",
    )
    args = parser.parse_args(argv)

    smoke = PystencilsOpenCLFloat32Smoke(args.n)
    if args.source:
        print(smoke.generate_kernel_source())

    result = smoke.run()
    print(result.to_dict())
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
