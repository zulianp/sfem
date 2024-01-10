# __init__.py

import sys
import os

sys.path.append(f'{os.path.dirname(os.path.realpath(__file__))}/..')

from sfem_codegen import *
