#!/usr/bin/env python3
"""Mooney-Rivlin with Prony relaxation, its powers of J continued below Jc.

`mooney_rivlin_prony`'s law with J^(-2/3) and J^(-4/3) replaced by h_m(J)
(`gen.continued_power`): equal for J >= Jc, a quartic C2 continuation for
Jmin < J < Jc and constant below Jmin.  The energy, its derivatives and the
relaxed stress the history reads stay finite through J = 0, so a Newton trial
that inverts an element is evaluated instead of rejected.  Below Jc the law is
the continuation, not the physical one; nothing rejects a state that ends there.

A material of its own rather than an option of the original, so the original's
kernels cannot change and the two run side by side on one case.  The
continuation is that of fsi-bootcamp's 2D Mooney-Rivlin
(`mooney_rivlin/native/model.h`, `mr_continued_power`).
"""

from pathlib import Path

from sfem import gen

try:
    from . import mooney_rivlin_prony as original
except ImportError:
    import mooney_rivlin_prony as original


Jc = gen.material_parameter("Jc")
Jmin = gen.material_parameter("Jmin")


def continued(J, m):
    return gen.continued_power(J, m, Jc, Jmin)


material = original.code_generator(
    "continued_mooney_rivlin_prony",
    "GeneratedContinuedMooneyRivlinProny",
    power=continued,
    parameter_defaults=original.PARAMETER_DEFAULTS + (("Jc", 0.2), ("Jmin", -1.0)),
)


if __name__ == "__main__":
    gen.run(material, Path(__file__).with_name("generated") / material.name)
