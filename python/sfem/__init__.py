try:
    from .sfem_config import *
except ModuleNotFoundError as error:
    if error.name != "sfem.sfem_config":
        raise
