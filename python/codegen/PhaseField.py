#!/usr/bin/env python3

from PhaseFieldBase import *


def linear_strain(gradu):
    return (gradu + gradu.T) / 2


class AT2:
    def g(self, c):
        return (1 - c) ** 2

    def omega(self, c):
        return c**2

    def comega(self, c):
        return 2


class IsotropicPhaseField(PhaseFieldBase):
    def __init__(self, name, degradation, elem_trial, elem_test):
        super().__init__(elem_trial, elem_test)

        mu, lmbda = sp.symbols("mu lambda", real=True)
        Gc, ls = sp.symbols("Gc ls", real=True)

        self.kernel_name = name
        self.params = [mu, lmbda, Gc, ls]

        gradu = self.displacement.grad()
        gradc = self.phase.grad()
        c = self.phase.value()

        # Energy
        epsu = linear_strain(gradu)
        eu = lmbda / 2 * tr(epsu) * tr(epsu) + mu * inner(epsu, epsu)
        ec = (Gc / degradation.comega(c)) * (
            degradation.omega(c) / ls + ls * inner(gradc, gradc)
        )
        energy = degradation.g(c) * eu + ec

        self.initialize(energy)


pf2 = IsotropicPhaseField(
    "IsotropicPhaseField_2D_AT2", AT2(), GenericFE("trial", 2), GenericFE("test", 2)
)
pf2.generate_code()

pf3 = IsotropicPhaseField(
    "IsotropicPhaseField_3D_AT2", AT2(), GenericFE("trial", 3), GenericFE("test", 3)
)
pf3.generate_code()

#############################################################
