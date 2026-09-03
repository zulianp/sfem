"""The target in effect for one code-generation run.

L5 owns the choice of target and L6 prints what that choice implies.  Until
now L6 made the choice itself: three emitter modules each defined a private
``_target()`` returning ``OpenMPTarget()``.  A backend that had already been
built with a target -- ``AVX512_SOA_BACKEND``, ``CUDA_SOA_BACKEND`` and the
rest all carry one -- passed it to a language check and to the energy emitter,
and the residual path discarded it.  That is the same defect the IR work
found in the scheduler: a decision made in the right layer and then thrown
away by the layer that needed it.

The target is bound for the duration of an emission rather than threaded
through every function that prints a pragma.  That is a deliberate choice and
worth stating.  It is ambient because it is a property of the run: 112 call
sites in 49 functions ask for a qualifier, a pragma or a work-item index, and
none of them chooses the target or would ever pass anything but the value it
received.  A parameter in all 49 signatures would be forwarded unchanged
through every one of them, which records the plumbing without recording a
decision.  Binding it once, explicitly, at the backend boundary keeps the
decision in L5 and keeps it visible: ``use_target`` marks exactly where the
choice takes effect, and ``current_target`` is the only way to read it.

The binding is a :class:`contextvars.ContextVar`, so concurrent generation
runs cannot see each other's target, and a run that never binds one still
gets OpenMP -- which is what every caller got when the emitters hardcoded it.
"""

import contextlib
import contextvars

from codegen.framework.targets.targets import OpenMPTarget


_ACTIVE_TARGET = contextvars.ContextVar("sfem_active_target", default=None)


def current_target():
    """The target this emission must print for.

    Defaults to OpenMP when nothing has bound one, which is the behaviour the
    emitters had when each of them constructed ``OpenMPTarget()`` directly.
    """
    target = _ACTIVE_TARGET.get()
    return OpenMPTarget() if target is None else target


@contextlib.contextmanager
def use_target(target):
    """Bind ``target`` for the duration of the block.

    ``None`` leaves the current binding alone, so a backend that was built
    without an explicit target does not clobber an outer one.
    """
    if target is None:
        yield current_target()
        return
    token = _ACTIVE_TARGET.set(target)
    try:
        yield target
    finally:
        _ACTIVE_TARGET.reset(token)
