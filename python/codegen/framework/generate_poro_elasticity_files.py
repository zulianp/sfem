#!/usr/bin/env python3
try:
    from ._script_common import bootstrap_python_path, generated_output_dir
except ImportError:
    from _script_common import bootstrap_python_path, generated_output_dir


bootstrap_python_path(__file__, 2)

from codegen.framework.materials.poro_hyperelasticity import material  # noqa: E402
from sfem import gen  # noqa: E402


if __name__ == "__main__":
    gen.run(material, generated_output_dir(__file__, "poro_hyperelasticity", 3))
