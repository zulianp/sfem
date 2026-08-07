#!/usr/bin/env python3
import os
import sys


FRAMEWORK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if FRAMEWORK_DIR not in sys.path:
    sys.path.insert(0, FRAMEWORK_DIR)

try:
    from ._script_common import bootstrap_python_path, generated_output_dir  # noqa: E402
except ImportError:
    from _script_common import bootstrap_python_path, generated_output_dir  # noqa: E402


bootstrap_python_path(__file__, 3)

from codegen.framework.materials.two_phase_flow import material  # noqa: E402
from sfem import gen  # noqa: E402


if __name__ == "__main__":
    gen.run(material, generated_output_dir(__file__, "two_phase_flow", 4))
