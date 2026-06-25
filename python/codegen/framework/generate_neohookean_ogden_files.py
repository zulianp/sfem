#!/usr/bin/env python3
import os
import sys


PYTHON_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PYTHON_DIR)

from codegen.framework.materials.neohookean_ogden import material  # noqa: E402
from sfem import gen  # noqa: E402


if __name__ == "__main__":
    gen.run(material, os.path.join(os.path.dirname(__file__), "../../../frontend/ops/generated/neohookean_ogden"))
