from fe import FE
import sympy as sp
from sfem_codegen import *


class Real1(FE):
    def __init__(self):
        super().__init__()

        self.A_ = 1
        self.Ainv_ = 1

    def name(self):
        return "Real1"

    def f0(self, x, y):
        return 1

    def fun(self, p):
        return [1]

    def n_nodes(self):
        return 1
