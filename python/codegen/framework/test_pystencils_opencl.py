import unittest

from codegen.framework.pystencils_opencl import PystencilsOpenCLFloat32Smoke


class PystencilsOpenCLTest(unittest.TestCase):
    def test_generates_float32_opencl_kernel_source(self):
        source = PystencilsOpenCLFloat32Smoke(16).generate_kernel_source()

        self.assertIn("__kernel void sfem_pystencils_float32_add", source)
        self.assertIn("__global const float *a", source)
        self.assertIn("const int gid = get_global_id(0);", source)
        self.assertIn("const float av = a[gid];", source)
        self.assertIn("const float bv = b[gid];", source)
        self.assertIn("const float ov = av + bv;", source)
        self.assertIn("out[gid] = ov;", source)
        self.assertNotIn("double", source)

    def test_runs_float32_kernel_on_apple_opencl_gpu_when_available(self):
        result = PystencilsOpenCLFloat32Smoke(1024).run()

        if not result.success and _opencl_device_unavailable(result.reason):
            self.skipTest(result.reason)

        self.assertTrue(result.success, result.reason)
        self.assertEqual(result.platform_name, "Apple")
        self.assertIn("Apple", result.device_name)
        self.assertEqual(result.max_error, 0.0)


def _opencl_device_unavailable(reason):
    known_reasons = (
        "pyopencl is not installed",
        "no Apple OpenCL GPU device found",
        "Apple OpenCL platform is present but GPU device query failed",
    )
    return any(reason.startswith(known) for known in known_reasons)


if __name__ == "__main__":
    unittest.main()
