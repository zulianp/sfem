#!/usr/bin/env python3

try:
    from sfem_codegen import *
except Exception:
    # Minimal shims to allow file generation without SymPy at runtime
    import sympy as sp  # will fail if sympy missing; we guard uses
    def simplify(expr):
        return expr
from quad4 import *
from tri3 import *
from tri6 import *
from tet4 import *
from tet10 import *
from tet20 import *
from hex8 import *
from aahex8 import *

import sys
import os

from time import perf_counter


def simplify(expr):
    return expr
    # return sp.simplify(expr)


def assign_matrix(name, mat):
    rows, cols = mat.shape
    expr = []
    for i in range(0, rows):
        for j in range(0, cols):
            var = sp.symbols(f"{name}[{i*cols + j}]")
            expr.append(ast.Assignment(var, mat[i, j]))
    return expr


def assign_add_matrix(name, mat):
    rows, cols = mat.shape
    expr = []
    for i in range(0, rows):
        for j in range(0, cols):
            var = sp.symbols(f"{name}[{i*cols + j}]")
            expr.append(ast.AddAugmentedAssignment(var, mat[i, j]))
    return expr


class SRHyperelasticity:
    @staticmethod
    def create_from_string(fe, str_expr: str):
        # Allow users to pass expressions in terms of invariants I1b, I2b, J
        I1b, I2b, J = sp.symbols("I1b I2b J", real=True)
        local_ns = {"I1b": I1b, "I2b": I2b, "J": J}
        fun = sp.parse_expr(str_expr, local_dict=local_ns)
        return SRHyperelasticity(fe, fun)

    def init_templates(self):
        """Initialize the C and CUDA C templates for placing the generated code."""
        import os

        self.tpl = {}
        # Templates live at repo_root/tpl/hyperelasticity
        tpl_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "tpl", "hyperelasticity"))
        if not os.path.isdir(tpl_dir):
            raise FileNotFoundError(f"Template directory not found: {tpl_dir}")

        for fname in os.listdir(tpl_dir):
            fpath = os.path.join(tpl_dir, fname)
            if os.path.isfile(fpath):
                with open(fpath, "r") as f:
                    basename = os.path.splitext(fname)[0]
                    # Normalize keys: strip optional "_tpl" suffix
                    key = basename
                    if key.endswith("_tpl"):
                        key = key[:-4]
                    self.tpl[key] = f.read()
      

    def init_symbols(self):
        # Invariants
        self.I1b = sp.symbols("I1b", real=True)
        self.I2b = sp.symbols("I2b", real=True)
        self.J = sp.symbols("J", real=True)
        self.safe_log = lambda x: sp.log(sp.Max(1e-8, x))
        self.safe_sqrt = lambda x: sp.sqrt(sp.Max(1e-8, x))

    def __init__(self, fe, fun):
        self.init_symbols()
        self.init_templates()
        self.fe = fe
        self.fun = simplify(fun)

    def value(self):
        dfdI1b = sp.diff(self.fun, self.I1b)
        dfdI2b = sp.diff(self.fun, self.I2b)
        dfdJ = sp.diff(self.fun, self.J)
        return dfdI1b, dfdI2b, dfdJ


    def apply(self):
        pass

    # ---------------- Symbolic core (P_tXJinv_t idiom) ----------------
    def _build_sym(self, fe):
        dims = fe.manifold_dim()
        q = fe.quadrature_point()

        fe.use_adjugate = True
        gref = fe.tgrad(q, ncomp=dims)
        jac_inv = fe.symbol_jacobian_inverse_as_adjugate()

        u = coeffs_SoA("u", dims, fe.n_nodes())
        disp_grad = sp.zeros(dims, dims)
        for i in range(0, dims * fe.n_nodes()):
            disp_grad += u[i] * gref[i]
        disp_grad = disp_grad * jac_inv

        Id = sp.eye(dims)
        F = Id + disp_grad
        J = determinant(F)
        B = F * F.T
        trB = sp.trace(B)
        I2 = (trB * trB - sp.trace(B * B)) / 2
        Jm23 = J ** (-sp.Rational(2, 3))
        B_bar = Jm23 * B
        I1b = sp.trace(B_bar)
        I2b = (I1b * I1b - sp.trace(B_bar * B_bar)) / 2

        W = self.fun.subs({self.I1b: I1b, self.I2b: I2b, self.J: J})
        P = sp.Matrix(dims, dims, [0] * (dims * dims))
        for i in range(dims):
            for j in range(dims):
                P[i, j] = sp.diff(W, F[i, j])

        dV = fe.reference_measure() * fe.symbol_jacobian_determinant() * fe.quadrature_weight()
        P_tXJinv_t = P * jac_inv.T * dV

        return {
            'dims': dims,
            'gref': gref,
            'jac_inv': jac_inv,
            'disp_grad': disp_grad,
            'F': F,
            'J': J,
            'W': W,
            'P': P,
            'P_tXJinv_t': P_tXJinv_t,
            'dV': dV,
        }

    def kernel_value(self, fe):
        s = self._build_sym(fe)
        expr = []
        form = sp.symbols("element_scalar[0]")
        expr.append(ast.AddAugmentedAssignment(form, s['W'] * s['dV']))
        return c_gen(expr)

    def kernel_gradient(self, fe):
        s = self._build_sym(fe)
        dims = s['dims']
        expr = []
        for i in range(0, dims * fe.n_nodes()):
            lform = sp.symbols(f"element_vector[{i}*stride]")
            expr.append(ast.AddAugmentedAssignment(lform, inner(s['P_tXJinv_t'], s['gref'][i])))
        return c_gen(expr)

    def kernel_apply(self, fe):
        s = self._build_sym(fe)
        dims = s['dims']
        F = s['F']
        P = s['P']
        jac_inv = s['jac_inv']
        dV = s['dV']

        H = sp.Matrix(dims, dims, coeffs("grad_trial", dims * dims))
        dP = sp.Matrix(dims, dims, [0] * (dims * dims))
        for a in range(dims):
            for b in range(dims):
                acc = 0
                for i in range(dims):
                    for j in range(dims):
                        acc += sp.diff(P[a, b], F[i, j]) * H[i, j]
                dP[a, b] = simplify(acc)
        Lop = dP * jac_inv.T * dV

        expr = []
        for i in range(0, dims * fe.n_nodes()):
            lform = sp.symbols(f"element_vector[{i}*stride]")
            expr.append(ast.AddAugmentedAssignment(lform, inner(Lop, s['gref'][i])))
        return c_gen(expr)

    def emit_tet4_all(self, out_dir: str = None, opname: str = "hyperelasticity"):
        if out_dir is None:
            out_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "operators", "generated"))
        os.makedirs(out_dir, exist_ok=True)

        fe = Tet4()
        k_value = self.kernel_value(fe)
        k_grad = self.kernel_gradient(fe)
        k_apply = self.kernel_apply(fe)

        params = dict(
            NODES=4,
            INCLUDES="#include \\\"tet4_inline_cpu.h\\\"",
            DECLARE_GEOM_LOCALS="scalar_t lx[4]; scalar_t ly[4]; scalar_t lz[4];",
            DECLARE_STATE_LOCALS="scalar_t element_ux[4]; scalar_t element_uy[4]; scalar_t element_uz[4];",
            DECLARE_DIR_LOCALS="scalar_t element_hx[4]; scalar_t element_hy[4]; scalar_t element_hz[4];",
            DECLARE_OUTPUT_LOCALS=(
                "accumulator_t element_outx[4]; accumulator_t element_outy[4]; accumulator_t element_outz[4];\n"
                "for (int d = 0; d < 4; ++d) {{ element_outx[d]=0; element_outy[d]=0; element_outz[d]=0; }}"
            ),
            DECLARE_JACOBIAN_LOCALS="scalar_t jacobian_adjugate[9]; scalar_t jacobian_determinant = 0;",
            QUAD_SETUP="/* single-point tet quadrature */",
            QUAD_LOOP_BEGIN="{{",
            QUAD_LOOP_END="}}",
            GATHER_CONN="for (int v = 0; v < 4; ++v) {{ ev[v] = elements[v][i]; }}",
            GATHER_GEOM="for (int v = 0; v < 4; ++v) {{ lx[v]=x[ev[v]]; ly[v]=y[ev[v]]; lz[v]=z[ev[v]]; }}",
            GATHER_U=(
                "for (int v = 0; v < 4; ++v) {{ const ptrdiff_t gi = ev[v]*u_stride;"
                " element_ux[v]=ux[gi]; element_uy[v]=uy[gi]; element_uz[v]=uz[gi]; }}"
            ),
            GATHER_H=(
                "for (int v = 0; v < 4; ++v) {{ const ptrdiff_t gi = ev[v]*h_stride;"
                " element_hx[v]=hx[gi]; element_hy[v]=hy[gi]; element_hz[v]=hz[gi]; }}"
            ),
            JACOBIAN_AT_QP=(
                "tet4_adjugate_and_det_s(\n"
                "    lx[0], lx[1], lx[2], lx[3],\n"
                "    ly[0], ly[1], ly[2], ly[3],\n"
                "    lz[0], lz[1], lz[2], lz[3],\n"
                "    jacobian_adjugate, &jacobian_determinant);"
            ),
        )

        # Prelude: build jac_inv, grad_ref, disp_grad and grad_trial
        prelude = (
            "scalar_t jac_inv[9];\n"
            "for (int r=0;r<3;++r) for (int c=0;c<3;++c) jac_inv[r*3+c] = jacobian_adjugate[c*3+r] / jacobian_determinant;\n"
            "const scalar_t grad_ref[12] = {-1,-1,-1, 1,0,0, 0,1,0, 0,0,1};\n"
            "scalar_t disp_grad[9] = {0};\n"
            "for (int a=0;a<4;++a){ const scalar_t ux=element_ux[a], uy=element_uy[a], uz=element_uz[a];\n"
            "  const scalar_t gx=grad_ref[a*3+0], gy=grad_ref[a*3+1], gz=grad_ref[a*3+2];\n"
            "  disp_grad[0]+=ux*gx; disp_grad[1]+=ux*gy; disp_grad[2]+=ux*gz;\n"
            "  disp_grad[3]+=uy*gx; disp_grad[4]+=uy*gy; disp_grad[5]+=uy*gz;\n"
            "  disp_grad[6]+=uz*gx; disp_grad[7]+=uz*gy; disp_grad[8]+=uz*gz; }\n"
            "scalar_t tmp[9];\n"
            "for (int r=0;r<3;++r){ for (int c=0;c<3;++c){ tmp[r*3+c]=disp_grad[r*3+0]*jac_inv[0*3+c]+disp_grad[r*3+1]*jac_inv[1*3+c]+disp_grad[r*3+2]*jac_inv[2*3+c]; }}\n"
            "for (int k=0;k<9;++k) disp_grad[k]=tmp[k];\n"
        )

        # Gradient
        k_grad_c = prelude + ("scalar_t element_vector[12]={0}; const int stride=1;\n" + self.kernel_gradient(fe) +
                                "for (int a=0;a<4;++a){ element_outx[a]+=element_vector[0*4+a]; element_outy[a]+=element_vector[1*4+a]; element_outz[a]+=element_vector[2*4+a]; }\n")

        # Apply
        k_apply_c = prelude + (
            "scalar_t grad_trial[9]={0};\n"
            "for (int a=0;a<4;++a){ const scalar_t hx=element_hx[a], hy=element_hy[a], hz=element_hz[a];\n"
            "  const scalar_t gx=grad_ref[a*3+0], gy=grad_ref[a*3+1], gz=grad_ref[a*3+2];\n"
            "  grad_trial[0]+=hx*gx; grad_trial[1]+=hx*gy; grad_trial[2]+=hx*gz;\n"
            "  grad_trial[3]+=hy*gx; grad_trial[4]+=hy*gy; grad_trial[5]+=hy*gz;\n"
            "  grad_trial[6]+=hz*gx; grad_trial[7]+=hz*gy; grad_trial[8]+=hz*gz; }\n"
            "for (int r=0;r<3;++r){ for (int c=0;c<3;++c){ tmp[r*3+c]=grad_trial[r*3+0]*jac_inv[0*3+c]+grad_trial[r*3+1]*jac_inv[1*3+c]+grad_trial[r*3+2]*jac_inv[2*3+c]; }}\n"
            "for (int k=0;k<9;++k) grad_trial[k]=tmp[k];\n"
            "scalar_t element_vector[12]={0}; const int stride=1;\n" + self.kernel_apply(fe) +
            "for (int a=0;a<4;++a){ element_outx[a]+=element_vector[0*4+a]; element_outy[a]+=element_vector[1*4+a]; element_outz[a]+=element_vector[2*4+a]; }\n")

        # Value
        k_value_c = prelude + ("scalar_t element_scalar[1]={0};\n" + self.kernel_value(fe) + "accumulator_t energy = element_scalar[0];\n")

        grad_tpl = self.tpl['gradient']
        apply_tpl = self.tpl['apply']
        value_tpl = self.tpl['value']

        grad_code = grad_tpl.format(FUNC_NAME=f"{opname}_tet4_gradient", KERNEL_GRADIENT=k_grad_c, **params)
        apply_code = apply_tpl.format(FUNC_NAME=f"{opname}_tet4_apply", KERNEL_APPLY=k_apply_c, **params)

        vparams = params.copy()
        vparams.pop('DECLARE_DIR_LOCALS', None)
        vparams.pop('GATHER_H', None)
        value_code = value_tpl.format(FUNC_NAME=f"{opname}_tet4_value", KERNEL_VALUE=k_value_c, **vparams)

        with open(os.path.join(out_dir, f"{opname}_tet4_gradient.c"), "w") as f:
            f.write(grad_code)
        with open(os.path.join(out_dir, f"{opname}_tet4_apply.c"), "w") as f:
            f.write(apply_code)
        with open(os.path.join(out_dir, f"{opname}_tet4_value.c"), "w") as f:
            f.write(value_code)

    # -------- Rendering (stubs) --------
    def render_gradient_stub_hex8(self, func_name: str = "hyperelasticity_hex8_gradient"):
        tpl = self.tpl["gradient"]

        dI1b, dI2b, dJ = self.value()
        deriv_comment = (
            f"/* dW/dI1b = {sp.sstr(dI1b)}; dW/dI2b = {sp.sstr(dI2b)}; dW/dJ = {sp.sstr(dJ)} */"
        )

        code = tpl.format(
            FUNC_NAME=func_name,
            NODES=8,
            INCLUDES="#include \"line_quadrature.h\"\n#include \"hex8_inline_cpu.h\"",
            DECLARE_GEOM_LOCALS="scalar_t lx[8]; scalar_t ly[8]; scalar_t lz[8];",
            DECLARE_STATE_LOCALS="scalar_t element_ux[8]; scalar_t element_uy[8]; scalar_t element_uz[8];",
            DECLARE_OUTPUT_LOCALS=(
                "accumulator_t element_outx[8]; accumulator_t element_outy[8]; accumulator_t element_outz[8];\n"
                "for (int d = 0; d < 8; ++d) {{ element_outx[d]=0; element_outy[d]=0; element_outz[d]=0; }}"
            ),
            DECLARE_JACOBIAN_LOCALS="scalar_t jacobian_adjugate[9]; scalar_t jacobian_determinant = 0;",
            QUAD_SETUP=(
                "int SFEM_HEX8_QUADRATURE_ORDER = 2; SFEM_READ_ENV(SFEM_HEX8_QUADRATURE_ORDER, atoi);\n"
                "int n_qp = line_q3_n; const scalar_t *qx = line_q3_x; const scalar_t *qw = line_q3_w;\n"
                "if (SFEM_HEX8_QUADRATURE_ORDER == 1) { n_qp = line_q2_n; qx = line_q2_x; qw = line_q2_w; }\n"
                "else if (SFEM_HEX8_QUADRATURE_ORDER == 5) { n_qp = line_q6_n; qx = line_q6_x; qw = line_q6_w; }"
            ),
            QUAD_LOOP_BEGIN=(
                "for (int kz = 0; kz < n_qp; ++kz) {{\n"
                "    for (int ky = 0; ky < n_qp; ++ky) {{\n"
                "        for (int kx = 0; kx < n_qp; ++kx) {{\n"
            ),
            QUAD_LOOP_END="        }}}",
            GATHER_CONN="for (int v = 0; v < 8; ++v) {{ ev[v] = elements[v][i]; }}",
            GATHER_GEOM="for (int v = 0; v < 8; ++v) {{ lx[v]=x[ev[v]]; ly[v]=y[ev[v]]; lz[v]=z[ev[v]]; }}",
            GATHER_U=(
                "for (int v = 0; v < 8; ++v) {{ const ptrdiff_t gi = ev[v]*u_stride;"
                " element_ux[v]=ux[gi]; element_uy[v]=uy[gi]; element_uz[v]=uz[gi]; }}"
            ),
            JACOBIAN_AT_QP="hex8_adjugate_and_det(lx, ly, lz, qx[kx], qx[ky], qx[kz], jacobian_adjugate, &jacobian_determinant);",
            KERNEL_GRADIENT=(
                f"{deriv_comment}\n"
                "/* TODO: compute invariants at QP and accumulate residual using derivatives */\n"
                "(void)mu; (void)lambda; (void)qw; (void)qx;\n"
            ),
            SCATTER_OUT=(
                "for (int k = 0; k < 8; ++k) {{ const ptrdiff_t gi = ev[k]*out_stride;\n"
                "    #pragma omp atomic update\n"
                "    outx[gi] += element_outx[k];\n"
                "    #pragma omp atomic update\n"
                "    outy[gi] += element_outy[k];\n"
                "    #pragma omp atomic update\n"
                "    outz[gi] += element_outz[k];\n"
                "}}"
            ),
        )
        return code

    def render_apply_stub_hex8(self, func_name: str = "hyperelasticity_hex8_apply"):
        tpl = self.tpl["apply"]

        code = tpl.format(
            FUNC_NAME=func_name,
            NODES=8,
            INCLUDES="#include \"line_quadrature.h\"\n#include \"hex8_inline_cpu.h\"",
            DECLARE_GEOM_LOCALS="scalar_t lx[8]; scalar_t ly[8]; scalar_t lz[8];",
            DECLARE_STATE_LOCALS="scalar_t element_ux[8]; scalar_t element_uy[8]; scalar_t element_uz[8];",
            DECLARE_DIR_LOCALS="scalar_t element_hx[8]; scalar_t element_hy[8]; scalar_t element_hz[8];",
            DECLARE_OUTPUT_LOCALS=(
                "accumulator_t element_outx[8]; accumulator_t element_outy[8]; accumulator_t element_outz[8];\n"
                "for (int d = 0; d < 8; ++d) {{ element_outx[d]=0; element_outy[d]=0; element_outz[d]=0; }}"
            ),
            DECLARE_JACOBIAN_LOCALS="scalar_t jacobian_adjugate[9]; scalar_t jacobian_determinant = 0;",
            QUAD_SETUP=(
                "int SFEM_HEX8_QUADRATURE_ORDER = 2; SFEM_READ_ENV(SFEM_HEX8_QUADRATURE_ORDER, atoi);\n"
                "int n_qp = line_q3_n; const scalar_t *qx = line_q3_x; const scalar_t *qw = line_q3_w;\n"
                "if (SFEM_HEX8_QUADRATURE_ORDER == 1) { n_qp = line_q2_n; qx = line_q2_x; qw = line_q2_w; }\n"
                "else if (SFEM_HEX8_QUADRATURE_ORDER == 5) { n_qp = line_q6_n; qx = line_q6_x; qw = line_q6_w; }"
            ),
            QUAD_LOOP_BEGIN=(
                "for (int kz = 0; kz < n_qp; ++kz) {{\n"
                "    for (int ky = 0; ky < n_qp; ++ky) {{\n"
                "        for (int kx = 0; kx < n_qp; ++kx) {{\n"
            ),
            QUAD_LOOP_END="        }}}",
            GATHER_CONN="for (int v = 0; v < 8; ++v) {{ ev[v] = elements[v][i]; }}",
            GATHER_GEOM="for (int v = 0; v < 8; ++v) {{ lx[v]=x[ev[v]]; ly[v]=y[ev[v]]; lz[v]=z[ev[v]]; }}",
            GATHER_U=(
                "for (int v = 0; v < 8; ++v) {{ const ptrdiff_t gi = ev[v]*u_stride;"
                " element_ux[v]=ux[gi]; element_uy[v]=uy[gi]; element_uz[v]=uz[gi]; }}"
            ),
            GATHER_H=(
                "for (int v = 0; v < 8; ++v) {{ const ptrdiff_t gi = ev[v]*h_stride;"
                " element_hx[v]=hx[gi]; element_hy[v]=hy[gi]; element_hz[v]=hz[gi]; }}"
            ),
            JACOBIAN_AT_QP="hex8_adjugate_and_det(lx, ly, lz, qx[kx], qx[ky], qx[kz], jacobian_adjugate, &jacobian_determinant);",
            KERNEL_APPLY=(
                "/* TODO: compute Hessian apply at QP (A(u) * h) and accumulate into element_out* */\n"
                "(void)mu; (void)lambda; (void)qw; (void)qx;\n"
            ),
            SCATTER_OUT=(
                "for (int k = 0; k < 8; ++k) {{ const ptrdiff_t gi = ev[k]*out_stride;\n"
                "    #pragma omp atomic update\n"
                "    outx[gi] += element_outx[k];\n"
                "    #pragma omp atomic update\n"
                "    outy[gi] += element_outy[k];\n"
                "    #pragma omp atomic update\n"
                "    outz[gi] += element_outz[k];\n"
                "}}"
            ),
        )
        return code

    def write_hex8_stubs(self, out_dir: str = None):
        if out_dir is None:
            out_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "operators", "generated"))
        os.makedirs(out_dir, exist_ok=True)

        grad_code = self.render_gradient_stub_hex8()
        apply_code = self.render_apply_stub_hex8()

        with open(os.path.join(out_dir, "hyperelasticity_hex8_gradient.c"), "w") as f:
            f.write(grad_code)

        with open(os.path.join(out_dir, "hyperelasticity_hex8_apply.c"), "w") as f:
            f.write(apply_code)

    # ------------------- Neo-Hookean concrete emit (TET4) -------------------
    def render_neohookean_tet4_gradient(self, func_name: str = "neohookean_tet4_gradient"):
        tpl = self.tpl["gradient"]
        code = tpl.format(
            FUNC_NAME=func_name,
            NODES=4,
            INCLUDES="#include \\\"tet4_inline_cpu.h\\\"\n#include \\\"tet4_neohookean_ogden_inline_cpu.h\\\"",
            DECLARE_GEOM_LOCALS="scalar_t lx[4]; scalar_t ly[4]; scalar_t lz[4];",
            DECLARE_STATE_LOCALS="scalar_t element_ux[4]; scalar_t element_uy[4]; scalar_t element_uz[4];",
            DECLARE_OUTPUT_LOCALS=(
                "accumulator_t element_outx[4]; accumulator_t element_outy[4]; accumulator_t element_outz[4];\n"
                "for (int d = 0; d < 4; ++d) {{ element_outx[d]=0; element_outy[d]=0; element_outz[d]=0; }}"
            ),
            DECLARE_JACOBIAN_LOCALS="scalar_t jacobian_adjugate[9]; scalar_t jacobian_determinant = 0;",
            QUAD_SETUP="/* single-point tet quadrature */",
            QUAD_LOOP_BEGIN="{{",
            QUAD_LOOP_END="}}",
            GATHER_CONN="for (int v = 0; v < 4; ++v) {{ ev[v] = elements[v][i]; }}",
            GATHER_GEOM="for (int v = 0; v < 4; ++v) {{ lx[v]=x[ev[v]]; ly[v]=y[ev[v]]; lz[v]=z[ev[v]]; }}",
            GATHER_U=(
                "for (int v = 0; v < 4; ++v) {{ const ptrdiff_t gi = ev[v]*u_stride;"
                " element_ux[v]=ux[gi]; element_uy[v]=uy[gi]; element_uz[v]=uz[gi]; }}"
            ),
            JACOBIAN_AT_QP=(
                "tet4_adjugate_and_det_s(\n"
                "    lx[0], lx[1], lx[2], lx[3],\n"
                "    ly[0], ly[1], ly[2], ly[3],\n"
                "    lz[0], lz[1], lz[2], lz[3],\n"
                "    jacobian_adjugate, &jacobian_determinant);"
            ),
            KERNEL_GRADIENT=(
                "tet4_neohookean_gradient_adj(jacobian_adjugate, jacobian_determinant, mu, lambda,"
                " element_ux, element_uy, element_uz, element_outx, element_outy, element_outz);"
            ),
            SCATTER_OUT=(
                "for (int k = 0; k < 4; ++k) {{ const ptrdiff_t gi = ev[k]*out_stride;\n"
                "    #pragma omp atomic update\n"
                "    outx[gi] += element_outx[k];\n"
                "    #pragma omp atomic update\n"
                "    outy[gi] += element_outy[k];\n"
                "    #pragma omp atomic update\n"
                "    outz[gi] += element_outz[k];\n"
                "}}"
            ),
        )
        return code

    def render_neohookean_tet4_apply(self, func_name: str = "neohookean_tet4_apply"):
        tpl = self.tpl["apply"]
        code = tpl.format(
            FUNC_NAME=func_name,
            NODES=4,
            INCLUDES="#include \\\"tet4_inline_cpu.h\\\"\n#include \\\"tet4_neohookean_ogden_inline_cpu.h\\\"",
            DECLARE_GEOM_LOCALS="scalar_t lx[4]; scalar_t ly[4]; scalar_t lz[4];",
            DECLARE_STATE_LOCALS="scalar_t element_ux[4]; scalar_t element_uy[4]; scalar_t element_uz[4];",
            DECLARE_DIR_LOCALS="scalar_t element_hx[4]; scalar_t element_hy[4]; scalar_t element_hz[4];",
            DECLARE_OUTPUT_LOCALS=(
                "accumulator_t element_outx[4]; accumulator_t element_outy[4]; accumulator_t element_outz[4];\n"
                "for (int d = 0; d < 4; ++d) {{ element_outx[d]=0; element_outy[d]=0; element_outz[d]=0; }}"
            ),
            DECLARE_JACOBIAN_LOCALS="scalar_t jacobian_adjugate[9]; scalar_t jacobian_determinant = 0;",
            QUAD_SETUP="/* single-point tet quadrature */",
            QUAD_LOOP_BEGIN="{{",
            QUAD_LOOP_END="}}",
            GATHER_CONN="for (int v = 0; v < 4; ++v) {{ ev[v] = elements[v][i]; }}",
            GATHER_GEOM="for (int v = 0; v < 4; ++v) {{ lx[v]=x[ev[v]]; ly[v]=y[ev[v]]; lz[v]=z[ev[v]]; }}",
            GATHER_U=(
                "for (int v = 0; v < 4; ++v) {{ const ptrdiff_t gi = ev[v]*u_stride;"
                " element_ux[v]=ux[gi]; element_uy[v]=uy[gi]; element_uz[v]=uz[gi]; }}"
            ),
            GATHER_H=(
                "for (int v = 0; v < 4; ++v) {{ const ptrdiff_t gi = ev[v]*h_stride;"
                " element_hx[v]=hx[gi]; element_hy[v]=hy[gi]; element_hz[v]=hz[gi]; }}"
            ),
            JACOBIAN_AT_QP=(
                "tet4_adjugate_and_det_s(\n"
                "    lx[0], lx[1], lx[2], lx[3],\n"
                "    ly[0], ly[1], ly[2], ly[3],\n"
                "    lz[0], lz[1], lz[2], lz[3],\n"
                "    jacobian_adjugate, &jacobian_determinant);"
            ),
            KERNEL_APPLY=(
                "tet4_neohookean_hessian_apply_adj(jacobian_adjugate, jacobian_determinant, mu, lambda,"
                " element_ux, element_uy, element_uz, element_hx, element_hy, element_hz, element_outx, element_outy, element_outz);"
            ),
            SCATTER_OUT=(
                "for (int k = 0; k < 4; ++k) {{ const ptrdiff_t gi = ev[k]*out_stride;\n"
                "    #pragma omp atomic update\n"
                "    outx[gi] += element_outx[k];\n"
                "    #pragma omp atomic update\n"
                "    outy[gi] += element_outy[k];\n"
                "    #pragma omp atomic update\n"
                "    outz[gi] += element_outz[k];\n"
                "}}"
            ),
        )
        return code

    def write_neohookean_tet4(self, out_dir: str = None):
        if out_dir is None:
            out_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "operators", "generated"))
        os.makedirs(out_dir, exist_ok=True)

        grad_code = self.render_neohookean_tet4_gradient()
        apply_code = self.render_neohookean_tet4_apply()

        with open(os.path.join(out_dir, "neohookean_tet4_gradient.c"), "w") as f:
            f.write(grad_code)

        with open(os.path.join(out_dir, "neohookean_tet4_apply.c"), "w") as f:
            f.write(apply_code)

    def render_gradient_stub_tet4(self, func_name: str = "hyperelasticity_tet4_gradient"):
        tpl = self.tpl["gradient"]

        dI1b, dI2b, dJ = self.value()
        deriv_comment = (
            f"/* dW/dI1b = {sp.sstr(dI1b)}; dW/dI2b = {sp.sstr(dI2b)}; dW/dJ = {sp.sstr(dJ)} */"
        )

        code = tpl.format(
            FUNC_NAME=func_name,
            NODES=4,
            INCLUDES="#include \"tet4_inline_cpu.h\"",
            DECLARE_GEOM_LOCALS="scalar_t lx[4]; scalar_t ly[4]; scalar_t lz[4];",
            DECLARE_STATE_LOCALS="scalar_t element_ux[4]; scalar_t element_uy[4]; scalar_t element_uz[4];",
            DECLARE_OUTPUT_LOCALS=(
                "accumulator_t element_outx[4]; accumulator_t element_outy[4]; accumulator_t element_outz[4];\n"
                "for (int d = 0; d < 4; ++d) {{ element_outx[d]=0; element_outy[d]=0; element_outz[d]=0; }}"
            ),
            DECLARE_JACOBIAN_LOCALS="scalar_t jacobian_adjugate[9]; scalar_t jacobian_determinant = 0;",
            QUAD_SETUP="/* single-point tet quadrature (handled in kernel/Jacobian if needed) */",
            QUAD_LOOP_BEGIN="{{",
            QUAD_LOOP_END="}}",
            GATHER_CONN="for (int v = 0; v < 4; ++v) {{ ev[v] = elements[v][i]; }}",
            GATHER_GEOM="for (int v = 0; v < 4; ++v) {{ lx[v]=x[ev[v]]; ly[v]=y[ev[v]]; lz[v]=z[ev[v]]; }}",
            GATHER_U=(
                "for (int v = 0; v < 4; ++v) {{ const ptrdiff_t gi = ev[v]*u_stride;"
                " element_ux[v]=ux[gi]; element_uy[v]=uy[gi]; element_uz[v]=uz[gi]; }}"
            ),
            JACOBIAN_AT_QP=(
                "tet4_adjugate_and_det_s(\n"
                "    lx[0], lx[1], lx[2], lx[3],\n"
                "    ly[0], ly[1], ly[2], ly[3],\n"
                "    lz[0], lz[1], lz[2], lz[3],\n"
                "    jacobian_adjugate, &jacobian_determinant);"
            ),
            KERNEL_GRADIENT=(
                f"{deriv_comment}\n"
                "/* TODO: compute invariants at element (or mapped QP) and accumulate residual using derivatives */\n"
                "(void)mu; (void)lambda;\n"
            ),
            SCATTER_OUT=(
                "for (int k = 0; k < 4; ++k) {{ const ptrdiff_t gi = ev[k]*out_stride;\n"
                "    #pragma omp atomic update\n"
                "    outx[gi] += element_outx[k];\n"
                "    #pragma omp atomic update\n"
                "    outy[gi] += element_outy[k];\n"
                "    #pragma omp atomic update\n"
                "    outz[gi] += element_outz[k];\n"
                "}}"
            ),
        )
        return code

    def render_apply_stub_tet4(self, func_name: str = "hyperelasticity_tet4_apply"):
        tpl = self.tpl["apply"]

        code = tpl.format(
            FUNC_NAME=func_name,
            NODES=4,
            INCLUDES="#include \"tet4_inline_cpu.h\"",
            DECLARE_GEOM_LOCALS="scalar_t lx[4]; scalar_t ly[4]; scalar_t lz[4];",
            DECLARE_STATE_LOCALS="scalar_t element_ux[4]; scalar_t element_uy[4]; scalar_t element_uz[4];",
            DECLARE_DIR_LOCALS="scalar_t element_hx[4]; scalar_t element_hy[4]; scalar_t element_hz[4];",
            DECLARE_OUTPUT_LOCALS=(
                "accumulator_t element_outx[4]; accumulator_t element_outy[4]; accumulator_t element_outz[4];\n"
                "for (int d = 0; d < 4; ++d) {{ element_outx[d]=0; element_outy[d]=0; element_outz[d]=0; }}"
            ),
            DECLARE_JACOBIAN_LOCALS="scalar_t jacobian_adjugate[9]; scalar_t jacobian_determinant = 0;",
            QUAD_SETUP="/* single-point tet quadrature (handled in kernel/Jacobian if needed) */",
            QUAD_LOOP_BEGIN="{{",
            QUAD_LOOP_END="}}",
            GATHER_CONN="for (int v = 0; v < 4; ++v) {{ ev[v] = elements[v][i]; }}",
            GATHER_GEOM="for (int v = 0; v < 4; ++v) {{ lx[v]=x[ev[v]]; ly[v]=y[ev[v]]; lz[v]=z[ev[v]]; }}",
            GATHER_U=(
                "for (int v = 0; v < 4; ++v) {{ const ptrdiff_t gi = ev[v]*u_stride;"
                " element_ux[v]=ux[gi]; element_uy[v]=uy[gi]; element_uz[v]=uz[gi]; }}"
            ),
            GATHER_H=(
                "for (int v = 0; v < 4; ++v) {{ const ptrdiff_t gi = ev[v]*h_stride;"
                " element_hx[v]=hx[gi]; element_hy[v]=hy[gi]; element_hz[v]=hz[gi]; }}"
            ),
            JACOBIAN_AT_QP=(
                "tet4_adjugate_and_det_s(\n"
                "    lx[0], lx[1], lx[2], lx[3],\n"
                "    ly[0], ly[1], ly[2], ly[3],\n"
                "    lz[0], lz[1], lz[2], lz[3],\n"
                "    jacobian_adjugate, &jacobian_determinant);"
            ),
            KERNEL_APPLY=(
                "/* TODO: compute Hessian apply at element and accumulate into element_out* */\n"
                "(void)mu; (void)lambda;\n"
            ),
            SCATTER_OUT=(
                "for (int k = 0; k < 4; ++k) {{ const ptrdiff_t gi = ev[k]*out_stride;\n"
                "    #pragma omp atomic update\n"
                "    outx[gi] += element_outx[k];\n"
                "    #pragma omp atomic update\n"
                "    outy[gi] += element_outy[k];\n"
                "    #pragma omp atomic update\n"
                "    outz[gi] += element_outz[k];\n"
                "}}"
            ),
        )
        return code

    def write_tet4_stubs(self, out_dir: str = None):
        if out_dir is None:
            out_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "operators", "generated"))
        os.makedirs(out_dir, exist_ok=True)

        grad_code = self.render_gradient_stub_tet4()
        apply_code = self.render_apply_stub_tet4()

        with open(os.path.join(out_dir, "hyperelasticity_tet4_gradient.c"), "w") as f:
            f.write(grad_code)

        with open(os.path.join(out_dir, "hyperelasticity_tet4_apply.c"), "w") as f:
            f.write(apply_code)


def main():
    start = perf_counter()

    fes = {
        "AAHEX8": AAHex8(),
        # "AAQUAD4": AxisAlignedQuad4(),
        "HEX8": Hex8(),
        # Additional element types can be re-enabled as needed
        # "TET10": Tet10(),
        # "TET20": Tet20(),
        "TET4": Tet4(),
        # "TRI3": Tri3(),
        # "TRI6": Tri6(),
    }

    if len(sys.argv) < 2:
        print("Usage: python3 sr_hyperelasticity.py \"<strain_energy_function>\" [EType]")
        print("Example: python3 sr_hyperelasticity.py \"J * (I2b + (I1b * 0.8341331275382947))\" HEX8")
        exit(1)

    strain_energy_function = sys.argv[1]
    fe = Hex8()
    if len(sys.argv) >= 3:
        et = sys.argv[2].upper()
        if et in fes:
            fe = fes[et]
        else:
            print(f"[WARN] Unknown EType '{et}', defaulting to HEX8")

    op = SRHyperelasticity.create_from_string(fe, strain_energy_function)
    # Neo-Hookean use-case: emit concrete TET4 kernels using inlined adjoint functions
    if isinstance(fe, Tet4):
        op.emit_tet4_all(opname="hyperelasticity")
        print("Generated: operators/generated/hyperelasticity_tet4_gradient.c")
        print("Generated: operators/generated/hyperelasticity_tet4_apply.c")
        print("Generated: operators/generated/hyperelasticity_tet4_value.c")
    else:
        # Fall back: still compute and print derivatives for visibility
        dI1b, dI2b, dJ = op.value()
        print("dW/dI1b =", dI1b)
        print("dW/dI2b =", dI2b)
        print("dW/dJ   =", dJ)


if __name__ == "__main__":
    main()
