# __init__.py

import sys
import os

sys.path.append(f"{os.path.dirname(os.path.realpath(__file__))}/..")

# from sfem.sfem_config import *
from .distance_point_to_triangle import *
from .smesh import *
from .mesh_to_sdf import *
from .mesh_to_udf import *
