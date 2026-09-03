"""Target platforms: the syntax and execution facts a backend lowers to.

This is layer L5. It sits *below* planning and the IR and *above* emission: an
emitter asks a target how this platform spells an inline qualifier, a vectorize
pragma, an atomic update or a work-item index, and prints accordingly.

It used to live inside ``backends/``, which conflated two unrelated things --
these target definitions, and the orchestrators (``backends/openmp.py``,
``backends/cuda.py``) that drive the emitters. That conflation is why
``emitters -> backends`` looked like a layering violation: an emitter naming a
target is an upward edge and always was legal; an emitter naming an
orchestrator would not be. Splitting the package makes the distinction
enforceable instead of a matter of interpretation.
"""

from codegen.framework.targets.context import (  # noqa: F401
    current_target,
    use_target,
)
from codegen.framework.targets.targets import *  # noqa: F401,F403
from codegen.framework.targets.targets import (  # noqa: F401
    CUDATarget,
    HIPTarget,
    OpenMPTarget,
    TargetLanguage,
)
