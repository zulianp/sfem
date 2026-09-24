"""The generated device kernels compile, and their templates instantiate.

This environment builds with `SFEM_ENABLE_CUDA=OFF` and has no `nvcc`, so
every `.cu` and `.cuh` in the generated tree was going out unchecked: the whole
device surface -- fifty element headers, four hundred entry points -- had no
gate of any kind behind it, and a wide edit to it would have been made blind.

A device compile is not available, but a *compile* is.  The generated device
headers are ordinary C++ once the CUDA decorations are defined away, so this
defines them away, includes each header and takes the address of every entry
point it declares.  Taking the address is the part that matters:
`-fsyntax-only` parses a template but does not instantiate it, and the body is
where the mistakes are.

What this does not cover, and what still wants a real `nvcc` run on Alps:
anything that is genuinely device-only -- shared memory, warp intrinsics,
launch bounds -- and of course whether the kernels are correct rather than
merely well-formed.  It is a syntax and instantiation gate, which is a great
deal more than none.

The include path matters and is easy to get wrong.  The build puts every
material's `op/` directory on it, and the device tree's `op/cuda/` is that
directory's twin; the generated relative includes are written to resolve from
there, not from the file that contains them.  A check that omits them fails
thirty-eight headers out of fifty and looks like a tree-wide defect.  It is
not one -- it is the check being wrong -- and the first version of this test
made exactly that mistake.
"""

import glob
import os
import re
import shutil
import subprocess
import tempfile
import unittest


def _generated_tree():
    return os.path.normpath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "..", "..", "frontend", "ops", "generated",
        )
    )


#: The CUDA decorations, defined away, plus the one builtin the kernels call.
_PROLOGUE = (
    "#define __host__",
    "#define __device__",
    "#define __forceinline__ inline",
    "#include <cstddef>",
    "#include <cmath>",
    "static inline double atomicAdd(double *p, double v) { *p += v; return *p; }",
    "static inline float atomicAdd(float *p, float v) { *p += v; return *p; }",
)

#: An entry point of a device element header: the template head the emitter
#: gives them, then the function name.
_ENTRY = re.compile(r"^template <typename s_t(?:, int VS)?>\n[^\n]*\bint (\w+)\(", re.M)


def _include_arguments(root):
    arguments = ["-I", root]
    for directory in sorted(glob.glob(os.path.join(root, "*", "op"))) + sorted(
        glob.glob(os.path.join(root, "*", "op", "cuda"))
    ):
        arguments += ["-I", directory]
    return arguments


def _device_element_headers(root):
    return sorted(
        os.path.join(directory, name)
        for directory, _, names in os.walk(root)
        for name in names
        if name.endswith("_element.hpp") and os.path.basename(directory) == "cuda"
    )


class DeviceKernelsCompile(unittest.TestCase):
    def setUp(self):
        if shutil.which("c++") is None:
            self.skipTest("no c++ compiler")
        self.root = _generated_tree()
        self.includes = _include_arguments(self.root)

    def test_every_device_element_header_compiles_and_instantiates(self):
        headers = _device_element_headers(self.root)
        self.assertGreater(len(headers), 0, "no device element headers found")
        instantiated = 0
        failures = []
        for header in headers:
            names = _ENTRY.findall(open(header, encoding="utf-8").read())
            instantiated += len(names)
            relative = os.path.relpath(header, self.root)
            # `<double>` where the kernel is scalar, `<double, 1>` where it
            # still carries a width -- one element per thread either way.
            arguments = "<double, 1>" if ", int VS>" in open(
                header, encoding="utf-8"
            ).read() else "<double>"
            source = list(_PROLOGUE) + ['#include "%s"' % relative] + [
                "auto *probe%d = &sfem::codegen::%s%s;" % (index, name, arguments)
                for index, name in enumerate(names)
            ]
            with tempfile.NamedTemporaryFile(
                "w", suffix=".cpp", delete=False
            ) as handle:
                handle.write("\n".join(source) + "\n")
                path = handle.name
            try:
                result = subprocess.run(
                    ["c++", "-std=c++17", "-fsyntax-only"] + self.includes + [path],
                    capture_output=True,
                    text=True,
                )
            finally:
                os.unlink(path)
            if result.returncode != 0:
                errors = [
                    line for line in result.stderr.splitlines() if "error:" in line
                ]
                failures.append("%s: %s" % (relative, errors[0] if errors else "?"))
        self.assertEqual(failures, [], "\n".join(failures))
        self.assertGreater(
            instantiated, 0, "no entry points were instantiated; the gate is inert"
        )


if __name__ == "__main__":
    unittest.main()
