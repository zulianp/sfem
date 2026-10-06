"""Compatibility shim: the generator driver now lives inside the framework.

``sfem.gen`` was the entry point for every material and generator script, but
the four pipeline stages it defines -- ``UserInputStage``,
``SpecializedFormManipulationStage``, ``CodeGenerationStage`` and ``generate``
-- are the framework's own architecture, not runtime-library surface.  They
moved to ``codegen.framework.pipeline.driver``; this module re-exports them so
existing callers keep working unchanged.
"""

from codegen.framework.pipeline.driver import *  # noqa: F401,F403
from codegen.framework.pipeline import driver as _driver

# `import *` honours __all__, which deliberately excludes the lower-level names
# the tests and the framework's own modules reach for.  Re-export the module's
# full public surface so the shim is a faithful stand-in.
globals().update(
    {
        _name: _value
        for _name, _value in vars(_driver).items()
        if not _name.startswith("__")
    }
)

__all__ = list(getattr(_driver, "__all__", ()))
