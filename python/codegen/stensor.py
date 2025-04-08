import sympy as sp
from sfem_codegen import inner
import sympy.codegen.ast as ast


class Tensor3:
    def __init__(self, n0, n1, n2):
        self.n0 = n0
        self.n1 = n1
        self.n2 = n2
        self.d = [0] * n0

        for i in range(0, n0):
            self.d[i] = sp.zeros(n1, n2)

    def nnz_symbolic(self, prefix):
        ret = Tensor3(self.n0, self.n1, self.n2)

        next_idx = 0
        for i in range(0, self.n0):
            for d1 in range(0, self.n1):
                for d2 in range(0, self.n2):

                    if self[i, d1, d2] != 0:
                        ret[i, d1, d2] = sp.symbols(f"{prefix}_diff3[{next_idx}]")
                        next_idx += 1

        return ret

    def compress_nnz(self, t3):
        ret = Tensor3(self.n0, self.n1, self.n2)

        vals = []
        for i in range(0, self.n0):
            for d1 in range(0, self.n1):
                for d2 in range(0, self.n2):

                    if self[i, d1, d2] != 0:
                        vals.append(t3[i, d1, d2])

        return sp.Matrix(len(vals), 1, vals)

    def assign_nnz(self, t3):
        expr = []
        for i in range(0, self.n0):
            for d1 in range(0, self.n1):
                for d2 in range(0, self.n2):
                    if self[i, d1, d2] != 0:
                        expr.append(ast.Assignment(self[i, d1, d2], t3[i, d1, d2]))
        return expr

    def __setitem__(self, idx, val):
        self.d[idx[0]][idx[1], idx[2]] = val

    def __getitem__(self, idx):
        return self.d[idx[0]][idx[1], idx[2]]

    def __mul__(self, x):
        ret = [0] * self.n0

        r, c = x.shape

        if c == 1:
            ret = sp.zeros(self.n0, self.n1)
            for i in range(0, self.n0):
                v = self.d[i] * x

                for j in range(0, self.n1):
                    ret[i, j] = v[j]
        else:
            for i in range(0, self.n0):
                ret[i] = inner(self.d[i], x)

        return ret

    def iadd(self, right):
        that = self

        for i in range(0, self.n0):
            for d1 in range(0, self.n1):
                for d2 in range(0, self.n2):
                    that[i, d1, d2] += right[i, d1, d2]
        return self

    def nnz_op(self):
        ret = Tensor3(self.n0, self.n1, self.n2)

        that = self

        for i in range(0, self.n0):
            for d1 in range(0, self.n1):
                for d2 in range(0, self.n2):
                    ret[i, d1, d2] = (that[i, d1, d2] != 0) * 1
        return ret
