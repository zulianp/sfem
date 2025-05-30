import sympy as sp
from sfem_codegen import adjugate
from sfem_codegen import norm2
from sfem_codegen import c_gen
from sfem_codegen import c_log
from sfem_codegen import q as quadrature_point
from sfem_codegen import real_t
from sfem_codegen import coeffs
from sfem_codegen import det2
from sfem_codegen import det3
from sfem_codegen import matrix_coeff
from sfem_codegen import subsmat
from sfem_codegen import inner
import sympy.codegen.ast as ast
import sympy as sp
import numpy as np
from stensor import *


def read_file(path):
    with open(path, "r") as f:
        tpl = f.read()
        return tpl
    assert False
    return ""


def str_to_file(path, mystr):
    with open(path, "w") as f:
        f.write(mystr)
        f.close()


class FE:
    SoA = True
    strided = False
    use_adjugate = False

    def subparam_n_nodes(self):
        return self.n_nodes()

    def is_symbolic(self):
        return False

    def grad(self, p):
        fx = self.fun(p)
        dims, __ = p.shape
        nn = len(fx)

        g = [0] * nn

        for i in range(0, nn):
            gi = []

            for d in range(0, dims):
                gg = sp.diff(fx[i], p[d])
                # gg = sp.simplify(gg)
                gi.append(gg)

            g[i] = sp.Matrix(dims, 1, gi)
        return g

    def hessian(self, p):
        gx = self.grad(p)
        dims, __ = p.shape
        nn = len(gx)

        H = [0] * nn

        for i in range(0, nn):
            Hi = sp.zeros(dims, dims)
            for d1 in range(0, dims):
                for d2 in range(0, dims):
                    HH = sp.diff(gx[i][d1], p[d2])
                    Hi[d1, d2] = HH
            H[i] = Hi
        return H

    def diff3(self, p):
        Hx = self.hessian(p)
        dims, __ = p.shape
        nn = len(Hx)

        T = [0] * nn
        for i in range(0, nn):
            t3 = Tensor3(dims, dims, dims)
            for d1 in range(0, dims):
                for d2 in range(0, dims):
                    for d3 in range(0, dims):
                        val = Hx[i][d1, d2]
                        t3[d1, d2, d3] = sp.diff(val, p[d3])
            T[i] = t3

        return T

    def interpolate(self, p, c):
        f = self.fun(p)

        ret = 0
        for i in range(0, len(f)):
            ret += f[i] * c[i]

        return ret

    def grad_interpolate(self, p, c):
        g = self.physical_grad(p)

        ret = sp.Matrix(self.spatial_dim(), 1, [0] * self.spatial_dim())

        for i in range(0, self.n_nodes()):
            gi = g[i]
            for d in range(0, self.spatial_dim()):
                # print(gi)
                ret[d] += gi[d] * c[i]
        return ret

    def tfun(self, p, ncomp=0):
        return self.tensorize(self.fun(p), ncomp)

    def tensorize(self, f, ncomp=0):
        if ncomp == 0:
            ncomp = self.manifold_dim()

        if ncomp == 1:
            return f

        nnodes = len(f)
        ndofs = nnodes * ncomp
        ret = [0] * ndofs

        for i in range(0, nnodes):
            for d in range(0, ncomp):
                F = sp.Matrix(ncomp, 1, [0] * ncomp)
                F[d] = f[i]

                if self.SoA:
                    ret[d * nnodes + i] = F
                else:
                    ret[i * ncomp + d] = F
        return ret

    def eval_grad(self, p):
        q = self.quadrature_point()
        g = self.grad(q)
        dim = (self.manifold_dim(),)

        dim = self.spatial_dim()
        for i in range(0, len(g)):
            for d in range(0, dim):
                g[i][d] = subsmat(g[i][d], q, p)

        return g

    def eval_hessian(self, p):
        q = self.quadrature_point()
        dim = (self.manifold_dim(),)
        H = self.hessian(q)

        dim = self.spatial_dim()
        for i in range(0, len(H)):
            for d in range(0, dim):
                for d2 in range(0, dim):
                    H[i][d, d2] = subsmat(H[i][d, d2], q, p)
        return H

    def eval_diff3(self, p):
        q = self.quadrature_point()
        dim = (self.manifold_dim(),)
        T = self.diff3(q)

        dim = self.spatial_dim()
        for i in range(0, len(T)):
            for d1 in range(0, dim):
                for d2 in range(0, dim):
                    for d2 in range(0, dim):
                        T[i][d1, d2, d2] = subsmat(T[i][d1, d2, d2], q, p)

        return T

    def taylor_expand_fun(self, center, point, order=-1):
        f = self.fun(center)
        if order == 0:
            return f

        g = self.eval_grad(center)
        H = self.eval_hessian(center)
        T3 = self.eval_diff3(center)

        if order == -1:
            order = 100

        h = point - center
        for i in range(0, self.n_nodes()):
            f[i] += inner(g[i], h)

            if order >= 2:
                f[i] += inner(H[i] * h, h) / 2

            if order >= 3:
                temp = T3[i] * h
                f[i] += inner(h, temp.T * h) / 6

            f[i] = sp.simplify(f[i])

        return f

    def taylor_expand_grad(self, center, point, order=-1):
        g = self.eval_grad(center)

        if order == 0:
            return g

        H = self.eval_hessian(center)
        T3 = self.eval_diff3(center)

        if order == -1:
            order = 100

        h = point - center
        for i in range(0, self.n_nodes()):
            g[i] += H[i] * h

            if order >= 2:
                temp = T3[i] * h
                g[i] += temp.T * h / 2

            for d in range(0, len(g[i])):
                g[i][d] = sp.simplify(g[i][d])

        return g

    def taylor_tgrad_symbolic(self, prefix, c, point, order, ncomp=0):
        return self.grad_tensorize(
            [self.taylor_grad_symbolic(prefix, c, point, order)], ncomp
        )

    def grad_nnz(self, prefix):
        dim = self.spatial_dim()
        g = coeffs(f"{prefix}_g", dim)
        return g

    def hessian_nnz(self, prefix):
        point = self.quadrature_point()
        dim = self.spatial_dim()
        H = sp.zeros(dim, dim)

        T2 = self.eval_hessian(point)
        T2_nnz = sp.zeros(dim, dim)

        for t2 in T2:
            for d1 in range(0, dim):
                for d2 in range(0, dim):
                    T2_nnz[d1, d2] += (t2[d1, d2] != 0) * 1

        next_idx = 0
        for d1 in range(0, dim):
            for d2 in range(d1, dim):
                if T2_nnz[d1, d2] != 0:
                    H[d1, d2] = sp.symbols(f"{prefix}_H[{next_idx}]")
                    H[d2, d1] = H[d1, d2]
                    next_idx += 1
        return H

    def diff3_nnz(self, prefix):
        point = self.quadrature_point()
        dim = self.spatial_dim()
        T3_eval = self.eval_diff3(point)
        T3 = Tensor3(dim, dim, dim)

        for t3 in T3_eval:
            T3.iadd(t3.nnz_op())

        return T3.nnz_symbolic(prefix)

    def taylor_grad_symbolic(self, prefix, center, point, order=-1):
        if order == -1:
            order = 100

        dim = self.spatial_dim()
        g = coeffs(f"{prefix}_g", dim)
        if order == 0:
            return g

        H = self.hessian_nnz(prefix)

        h = point - center

        g += H * h

        if order > 1:
            T3 = self.diff3_nnz(prefix)
            temp = T3 * h
            g += temp.T * h / 2
        return g

    def grad_tensorize(self, g, ncomp=0):
        if ncomp == 0:
            ncomp = self.manifold_dim()

        if ncomp == 1:
            return g

        tensor_size = ncomp * self.spatial_dim()
        ndofs = len(g) * ncomp
        nnodes = len(g)

        ret = [0] * ndofs
        for i in range(0, nnodes):
            gi = g[i]
            for d1 in range(0, ncomp):
                G = sp.Matrix(ncomp, self.spatial_dim(), [0] * tensor_size)
                for d2 in range(0, self.spatial_dim()):
                    G[d1, d2] = gi[d2]

                if self.SoA:
                    ret[d1 * nnodes + i] = G
                else:
                    ret[i * ncomp + d1] = G

        return ret

    def tgrad(self, p, ncomp=0):
        return self.grad_tensorize(self.grad(p), ncomp)

    def physical_tgrad(self, p, ncomp=0):
        return self.grad_tensorize(self.physical_grad(p), ncomp)

    def physical_grad(self, p):
        fx = self.fun(p)
        dims, __ = p.shape
        nn = len(fx)

        g = [0] * nn

        J_inv = self.symbol_jacobian_inverse()

        rg = self.grad(p)

        for i in range(0, nn):
            # gi = J_inv * gi
            gi = J_inv.T * rg[i]  # ATTENTION!
            g[i] = gi

        return g

    def is_isoparametric(self):
        return False

    def isoparametric_transform(self, q):
        p = self.coords()
        f = self.fun(q)
        ret = sp.zeros(self.spatial_dim(), 1)

        for i in range(0, self.n_nodes()):
            for d in range(0, self.spatial_dim()):
                ret[d] += p[d][i] * f[i]

        return ret

    def isoparametric_jacobian(self, q):
        p = self.coords()
        g = self.grad(q)

        ret = sp.zeros(self.spatial_dim(), self.manifold_dim())

        for i in range(0, self.n_nodes()):
            for d1 in range(0, self.spatial_dim()):
                for d2 in range(0, self.manifold_dim()):
                    ret[d1, d2] += p[d1][i] * g[i][d2]
        return ret

    def adj(self, q):
        return self.jacobian_inverse(q) * self.jacobian_determinant(q)

    def quadrature_weight(self):
        return sp.symbols("qw")

    def quadrature_point(self):
        qx, qy, qz = sp.symbols("qx qy qz")
        q_temp = [qx, qy, qz]
        return sp.Matrix(self.manifold_dim(), 1, q_temp[0 : self.manifold_dim()])

    def symbol_jacobian_inverse(self):
        if self.use_adjugate:
            return self.symbol_jacobian_inverse_as_adjugate()

        rows = self.manifold_dim()
        cols = self.spatial_dim()

        sls = []

        for i in range(0, rows):
            for j in range(0, cols):
                if self.strided:
                    var = sp.symbols(
                        f"jacobian_inverse[{i*cols + j}*stride_jacobian_inverse]"
                    )
                else:
                    var = sp.symbols(f"jacobian_inverse[{i*cols + j}]")
                # var = sp.symbols(f'jac_inv_{i*cols + j}]')
                sls.append(var)
        return sp.Matrix(rows, cols, sls)

    def symbol_adjugate(self):
        rows = self.manifold_dim()
        cols = self.spatial_dim()

        coff = matrix_coeff("adjugate", rows, cols)
        return coff

    def symbol_jacobian_inverse_as_adjugate(self):
        coff = self.symbol_adjugate()
        return coff / self.symbol_jacobian_determinant()

    def symbol_normal(self):
        A = self.symbol_adjugate().T
        n = A[:, self.manifold_dim() - 1]
        print(A)
        n /= norm2(n)
        return n

    def symbol_jacobian(self):
        rows = self.spatial_dim()
        cols = self.manifold_dim()

        sls = []

        for i in range(0, rows):
            for j in range(0, cols):
                var = sp.symbols(f"jacobian[{i*cols + j}*stride_jacobian]")
                # var = sp.symbols(f'jac_inv_{i*cols + j}]')
                sls.append(var)
        return sp.Matrix(rows, cols, sls)

    def symbol_jacobian_determinant(self):
        return sp.symbols("jacobian_determinant")

    def symbol_fff(self):
        rows = self.spatial_dim()
        cols = self.spatial_dim()
        FFF_symbolic = sp.Matrix(rows, cols, [0] * (rows * cols))

        varidx = 0
        for i in range(0, rows):
            for j in range(i, cols):
                var = sp.symbols(f"fff[{varidx}*stride]")
                varidx += 1

                # if FFF[i, j] != 0:
                FFF_symbolic[i, j] = var
                FFF_symbolic[j, i] = var

        return FFF_symbolic

    def generate_det_code(self):
        mat = sp.Matrix(
            self.manifold_dim(),
            self.manifold_dim(),
            [0] * (self.manifold_dim() * self.manifold_dim()),
        )

        for d1 in range(0, self.manifold_dim()):
            for d2 in range(0, self.manifold_dim()):
                if self.strided:
                    mat[d1, d2] = sp.symbols(f"a[{d1*self.manifold_dim() + d2}*stride]")
                else:
                    mat[d1, d2] = sp.symbols(f"a[{d1*self.manifold_dim() + d2}]")

        expr = []
        if self.manifold_dim() == 2:
            expr.append((det2(mat)))
        elif self.manifold_dim() == 3:
            expr.append((det3(mat)))
        else:
            assert False

        print(expr)

        det_body = c_gen(expr)

        det_code = f"static SFEM_CUDA_INLINE real_t {self.name()}_mk_det_{self.manifold_dim()}("
        det_code += "const count_t stride,\n const real_t *const SFEM_RESTRICT a\n"
        det_code += ")"
        det_code += "{\n"
        det_code += f"return {det_body};\n"
        det_code += "}\n"
        return det_code

    def generate_qp_based_code(self):
        expr = []

        if not self.is_isoparametric():
            qp = sp.Matrix(
                self.manifold_dim(), 1, quadrature_point[0 : self.spatial_dim()]
            )

            tqp = self.inverse_transform(qp)
            for i in range(0, len(tqp)):
                expr.append(ast.Assignment(sp.symbols(f"res[{i}]"), tqp[i]))

            print("------ inverse transform -------")
            c_log(c_gen(expr))
        else:
            print("------ TODO inverse transform -------")

        qp = self.quadrature_point()
        expr = []
        tqp = self.transform(qp)

        for i in range(0, len(tqp)):
            expr.append(ast.Assignment(sp.symbols(f"res[{i}]"), tqp[i]))

        print("------ transform -------")
        c_log(c_gen(expr))

        expr = []
        tqp = self.measure(qp)
        expr.append(ast.Assignment(sp.symbols(f"res[{0}]"), tqp))

        print("------ measure -------")
        c_log(c_gen(expr))

        f = self.fun(qp)
        nfun = len(f)

        fun_expr = []
        for i in range(0, nfun):
            if self.strided:
                fx = ast.Assignment(sp.symbols(f"f[{i}*stride_fun]"), f[i])
            else:
                fx = ast.Assignment(sp.symbols(f"f[{i}]"), f[i])

            fun_expr.append(fx)

        print(f"------ basis functions ({nfun}) -------")
        c_log(c_gen(fun_expr))

    def generate_c_code(self):
        self.generate_kernels_c_code()

        tpl = read_file("tpl/FE_CUDA_impl_tpl.cu")

        coordname = ["x", "y", "z", "t"]
        coordinate_read = ""
        coordinates = ""
        csp = self.coords_sub_parametric()
        for d in range(0, len(csp)):
            comment = f"// {coordname[d]}-coordinates\n"
            coordinate_read += comment
            coordinates += comment

            # coordnum = 0
            for x in csp[d]:
                coordnum = f"{x}".replace(f"p{coordname[d]}", "")
                coordinate_read += f"const {real_t} {x} = this_xyz[({d} * fe_subparam_n_nodes + {coordnum}) * nelements];\n"
                coordinates += f"{x},\n"
                # coordnum+=1

        c_log(coordinate_read)
        c_log(coordinates)

        code = tpl.format(
            NAME=self.name(),
            MK_FILE_CU=f'#include "{self.name()}_kernels.cu"',
            COORDINATES_READ=coordinate_read,
            COORDINATES=coordinates,
        )

        c_log(code)
        str_to_file(f"{self.name()}_impl.cu", code)

        test = f'#include "{self.name()}_impl.cu"'
        str_to_file(f"{self.name()}_kernels.cpp", code)

    def generate_kernels_c_code(self):
        output = ""
        coordname = ["x", "y", "z"]

        singature_prefix = f"static void SFEM_INLINE {self.name()}"
        qp = sp.Matrix(
            self.manifold_dim(), 1, quadrature_point[0 : self.manifold_dim()]
        )

        f = self.fun(qp)
        nfun = len(f)

        fun_expr = []
        for i in range(0, nfun):
            if self.strided:
                fx = ast.Assignment(sp.symbols(f"f[{i}*stride_fun]"), f[i])
            else:
                fx = ast.Assignment(sp.symbols(f"f[{i}]"), f[i])

            fun_expr.append(fx)

        g = self.physical_grad(qp)

        grad_expr = []

        for d in range(0, self.spatial_dim()):
            gx = []
            for i in range(0, nfun):
                if self.strided:
                    fx = ast.Assignment(
                        sp.symbols(f"g{coordname[d]}[{i}*stride_grad]"), g[i][d]
                    )
                else:
                    fx = ast.Assignment(sp.symbols(f"g{coordname[d]}[{i}]"), g[i][d])
                gx.append(fx)
            grad_expr.append(gx)

        jac = self.jacobian(qp)
        rows, cols = jac.shape
        jac_expr = []
        for r in range(0, rows):
            for c in range(0, cols):
                if self.strided:
                    jac_expr.append(
                        ast.Assignment(
                            sp.symbols(f"jacobian[{r*cols+c}*stride_jacobian]"),
                            jac[r, c],
                        )
                    )
                else:
                    jac_expr.append(
                        ast.Assignment(sp.symbols(f"jacobian[{r*cols+c}]"), jac[r, c])
                    )

        jac_inv = self.jacobian_inverse(qp)
        rows, cols = jac_inv.shape
        jac_inv_expr = []
        for r in range(0, rows):
            for c in range(0, cols):
                if self.strided:
                    jac_inv_expr.append(
                        ast.Assignment(
                            sp.symbols(
                                f"jacobian_inverse[{r*cols+c}*stride_jacobian_inverse]"
                            ),
                            jac_inv[r, c],
                        )
                    )
                else:
                    jac_inv_expr.append(
                        ast.Assignment(
                            sp.symbols(f"jacobian_inverse[{r*cols+c}]"), jac_inv[r, c]
                        )
                    )

        jacobian_determinant = self.jacobian_determinant(qp)

        jacobian_determinant_expr = []
        jacobian_determinant_expr.append(
            ast.Assignment(sp.symbols(f"jacobian_determinant[0]"), jacobian_determinant)
        )

        jacobian_determinant_and_inverse_expr = np.append(
            jacobian_determinant_expr, jac_inv_expr
        )

        constants = f"static const int fe_spatial_dim = {self.spatial_dim()};\n"
        constants += f"static const int fe_manifold_dim = {self.manifold_dim()};\n"
        constants += f"static const int fe_n_nodes = {self.n_nodes()};\n"
        constants += f'static const char * fe_name = "{self.name()}";\n'
        constants += f"static const int fe_n_nodes_for_jacobian = {len(self.coords_sub_parametric()[0])};\n"
        constants += (
            f"static const int fe_subparam_n_nodes = {self.subparam_n_nodes()};\n"
        )
        constants += f"static const float fe_reference_measure = {c_gen(self.reference_measure())};\n"

        coordinates = ""
        for c in self.coords_sub_parametric():
            for x in c:
                coordinates += f"const {real_t} {x},\n"
        quadrature_point_str = ""

        for x in qp:
            quadrature_point_str += f"const {real_t} {x},\n"

        if len(grad_expr) > 2:
            partial_z = c_gen(grad_expr[2])
        else:
            partial_z = "//TODO\n"

        c = sp.Matrix(self.n_nodes(), 1, [0] * self.n_nodes())
        for i in range(0, self.n_nodes()):
            c[i] = sp.symbols(f"c[{i}*stride_coeff]")

        # interp = ast.Assignment(sp.symbols("ret"), self.interpolate(qp, c))
        interp = self.interpolate(qp, c)

        grad_interp = []

        g = self.grad_interpolate(qp, c)
        for d in range(0, self.spatial_dim()):
            if self.strided:
                grad_interp.append(
                    ast.Assignment(sp.symbols(f"grad[{d}*stride_grad]"), g[d])
                )
            else:
                grad_interp.append(ast.Assignment(sp.symbols(f"grad[{d}]"), g[d]))

        utilities = ""
        utilities += self.generate_det_code()

        tpl = read_file("tpl/FE_CUDA_tpl.cu")
        code = tpl.format(
            NAME=self.name(),
            CONSTANTS=constants,
            COORDINATES=coordinates,
            QUADRATURE_POINT=quadrature_point_str,
            JACOBIAN=c_gen(jac_expr),
            JACOBIAN_INVERSE=c_gen(jac_inv_expr),
            JACOBIAN_DETERMINANT=c_gen(jacobian_determinant_expr),
            JACOBIAN_DETERMINANT_AND_INVERSE=c_gen(
                jacobian_determinant_and_inverse_expr
            ),
            FUN=c_gen(fun_expr),
            PARTIAL_X=c_gen(grad_expr[0]),
            PARTIAL_Y=c_gen(grad_expr[1]),
            PARTIAL_Z=partial_z,
            INTERPOLATE=c_gen(interp),
            GRAD_INTERPOLATE=c_gen(grad_interp),
            UTILITIES=utilities,
        )

        str_to_file(f"{self.name()}_kernels.cu", code)
