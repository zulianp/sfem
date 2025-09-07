#!/usr/bin/env python3

import os


class TPLHyperelasticity:
    """Loads and provides access to hyperelasticity C templates.

    - Reads all files in tpl/hyperelasticity
    - Normalizes keys by stripping an optional `_tpl` suffix and file extension
    - Exposes `get(name)` to retrieve a template string by logical name
    """

    def __init__(self, fe, base_dir: str = None):
        """Initialize templates and FE-specific boilerplate.

        - fe: finite element descriptor (e.g., Tet4())
        - base_dir: path to tpl/hyperelasticity
        """
        self.fe = fe
        if base_dir is None:
            # repo_root/python/codegen/.. -> repo_root/tpl/hyperelasticity
            here = os.path.dirname(__file__)
            base_dir = os.path.normpath(os.path.join(here, "..", "..", "tpl", "hyperelasticity"))

        if not os.path.isdir(base_dir):
            raise FileNotFoundError(f"Template directory not found: {base_dir}")

        self._tpl = {}
        for fname in os.listdir(base_dir):
            fpath = os.path.join(base_dir, fname)
            if not os.path.isfile(fpath):
                continue
            name, _ = os.path.splitext(fname)
            if name.endswith("_tpl"):
                name = name[:-4]
            with open(fpath, "r") as f:
                self._tpl[name] = f.read()

        # Precompute FE-specific placeholders
        self._params = self._build_params()

    def _build_params(self):
        name = getattr(self.fe, 'name', lambda: '')()
        nodes = self.fe.n_nodes()
        dims = self.fe.manifold_dim()

        # Default placeholders; extend per FE as needed
        params = {
            'NODES': nodes,
            'INCLUDES': '',
            'INLINE_HEADERS': '',
            'DECLARE_GEOM_LOCALS': '',
            'DECLARE_STATE_LOCALS': '',
            'DECLARE_DIR_LOCALS': '',
            'DECLARE_OUTPUT_LOCALS': '',
            'DECLARE_JACOBIAN_LOCALS': '',
            'QUAD_SETUP': '',
            'QUAD_LOOP_BEGIN': '{{',
            'QUAD_LOOP_END': '}}',
            'GATHER_CONN': '',
            'GATHER_GEOM': '',
            'GATHER_U': '',
            'GATHER_H': '',
            'JACOBIAN_AT_QP': '',
            'SCATTER_OUT': '',
        }

        # TET4 setup (3D, constant reference gradients)
        if name.upper() == 'TET4' or (nodes == 4 and dims == 3):
            params.update({
                'INCLUDES': '#include "tet4_inline_cpu.h"',
                'DECLARE_GEOM_LOCALS': 'scalar_t lx[4]; scalar_t ly[4]; scalar_t lz[4];',
                'DECLARE_STATE_LOCALS': 'scalar_t element_ux[4]; scalar_t element_uy[4]; scalar_t element_uz[4];',
                'DECLARE_DIR_LOCALS': 'scalar_t element_hx[4]; scalar_t element_hy[4]; scalar_t element_hz[4];',
                'DECLARE_OUTPUT_LOCALS': (
                    'accumulator_t element_outx[4]; accumulator_t element_outy[4]; accumulator_t element_outz[4];\n'
                    'for (int d = 0; d < 4; ++d) {{ element_outx[d]=0; element_outy[d]=0; element_outz[d]=0; }}'
                ),
                'DECLARE_JACOBIAN_LOCALS': 'scalar_t jacobian_adjugate[9]; scalar_t jacobian_determinant = 0;',
                'QUAD_SETUP': 'const scalar_t qw = 1;',
                'QUAD_LOOP_BEGIN': '{{',
                'QUAD_LOOP_END': '}}',
                'GATHER_CONN': 'for (int v = 0; v < 4; ++v) {{ ev[v] = elements[v][i]; }}',
                'GATHER_GEOM': 'for (int v = 0; v < 4; ++v) {{ lx[v]=x[ev[v]]; ly[v]=y[ev[v]]; lz[v]=z[ev[v]]; }}',
                'GATHER_U': 'for (int v = 0; v < 4; ++v) {{ const ptrdiff_t gi = ev[v]*u_stride; element_ux[v]=ux[gi]; element_uy[v]=uy[gi]; element_uz[v]=uz[gi]; }}',
                'GATHER_H': 'for (int v = 0; v < 4; ++v) {{ const ptrdiff_t gi = ev[v]*h_stride; element_hx[v]=hx[gi]; element_hy[v]=hy[gi]; element_hz[v]=hz[gi]; }}',
                'JACOBIAN_AT_QP': (
                    'tet4_adjugate_and_det_s(\n'
                    '    lx[0], lx[1], lx[2], lx[3],\n'
                    '    ly[0], ly[1], ly[2], ly[3],\n'
                    '    lz[0], lz[1], lz[2], lz[3],\n'
                    '    jacobian_adjugate, &jacobian_determinant);'
                ),
            })

        return params

    def prelude(self, need_trial: bool = False) -> str:
        """Emit FE-specific prelude code to compute jac_inv, disp_grad, and optionally grad_trial.

        Currently implemented for Tet4.
        """
        name = getattr(self.fe, 'name', lambda: '')()
        nodes = self.fe.n_nodes()
        dims = self.fe.manifold_dim()
        if name.upper() == 'TET4' or (nodes == 4 and dims == 3):
            body = (
                'scalar_t jac_inv[9];\n'
                'for (int r=0;r<3;++r) for (int c=0;c<3;++c) jac_inv[r*3+c] = jacobian_adjugate[c*3+r] / jacobian_determinant;\n'
                'const scalar_t grad_ref[12] = {-1,-1,-1, 1,0,0, 0,1,0, 0,0,1};\n'
                'scalar_t disp_grad[9] = {0};\n'
                'for (int a=0;a<4;++a){ const scalar_t ux=element_ux[a], uy=element_uy[a], uz=element_uz[a];\n'
                '  const scalar_t gx=grad_ref[a*3+0], gy=grad_ref[a*3+1], gz=grad_ref[a*3+2];\n'
                '  disp_grad[0]+=ux*gx; disp_grad[1]+=ux*gy; disp_grad[2]+=ux*gz;\n'
                '  disp_grad[3]+=uy*gx; disp_grad[4]+=uy*gy; disp_grad[5]+=uy*gz;\n'
                '  disp_grad[6]+=uz*gx; disp_grad[7]+=uz*gy; disp_grad[8]+=uz*gz; }\n'
                'scalar_t tmp[9];\n'
                'for (int r=0;r<3;++r){ for (int c=0;c<3;++c){ tmp[r*3+c]=disp_grad[r*3+0]*jac_inv[0*3+c]+disp_grad[r*3+1]*jac_inv[1*3+c]+disp_grad[r*3+2]*jac_inv[2*3+c]; }}\n'
                'for (int k=0;k<9;++k) disp_grad[k]=tmp[k];\n'
            )
            if need_trial:
                body += (
                    'scalar_t grad_trial[9]={0};\n'
                    'for (int a=0;a<4;++a){ const scalar_t hx=element_hx[a], hy=element_hy[a], hz=element_hz[a];\n'
                    '  const scalar_t gx=grad_ref[a*3+0], gy=grad_ref[a*3+1], gz=grad_ref[a*3+2];\n'
                    '  grad_trial[0]+=hx*gx; grad_trial[1]+=hx*gy; grad_trial[2]+=hx*gz;\n'
                    '  grad_trial[3]+=hy*gx; grad_trial[4]+=hy*gy; grad_trial[5]+=hy*gz;\n'
                    '  grad_trial[6]+=hz*gx; grad_trial[7]+=hz*gy; grad_trial[8]+=hz*gz; }\n'
                    'for (int r=0;r<3;++r){ for (int c=0;c<3;++c){ tmp[r*3+c]=grad_trial[r*3+0]*jac_inv[0*3+c]+grad_trial[r*3+1]*jac_inv[1*3+c]+grad_trial[r*3+2]*jac_inv[2*3+c]; }}\n'
                    'for (int k=0;k<9;++k) grad_trial[k]=tmp[k];\n'
                )
            return body
        return ''

    def scatter_vector(self, vec_name: str = 'element_vector') -> str:
        name = getattr(self.fe, 'name', lambda: '')()
        nodes = self.fe.n_nodes()
        dims = self.fe.manifold_dim()
        if name.upper() == 'TET4' or (nodes == 4 and dims == 3):
            return (
                f'for (int a=0;a<4;++a){{ element_outx[a]+={vec_name}[0*4+a]; element_outy[a]+={vec_name}[1*4+a]; element_outz[a]+={vec_name}[2*4+a]; }}\n'
            )
        return ''

    def render_gradient(self, func_name: str, kernel: str) -> str:
        tpl = self.get('gradient')
        return tpl.format(FUNC_NAME=func_name, KERNEL_GRADIENT=kernel, **self._params)

    def render_apply(self, func_name: str, kernel: str) -> str:
        tpl = self.get('apply')
        return tpl.format(FUNC_NAME=func_name, KERNEL_APPLY=kernel, **self._params)

    def render_value(self, func_name: str, kernel: str) -> str:
        tpl = self.get('value')
        # Ensure no direction locals appear in value
        params = dict(self._params)
        params.pop('DECLARE_DIR_LOCALS', None)
        params.pop('GATHER_H', None)
        return tpl.format(FUNC_NAME=func_name, KERNEL_VALUE=kernel, **params)

    def fe_tag(self) -> str:
        n = getattr(self.fe, 'name', lambda: '')()
        return (n or '').lower()

    def emit_all(self, opname: str, kernels: dict, out_dir: str = None):
        """Emit gradient/apply/value C files for the current FE.

        kernels: {'gradient': str, 'apply': str, 'value': str}
        """
        if out_dir is None:
            here = os.path.dirname(__file__)
            out_dir = os.path.normpath(os.path.join(here, "..", "..", "operators", "generated"))
        os.makedirs(out_dir, exist_ok=True)

        tag = self.fe_tag()

        pre_grad = self.prelude(need_trial=False)
        pre_apply = self.prelude(need_trial=True)

        k_grad_c = pre_grad + "scalar_t element_vector[12]={0}; const int stride=1;\n" + kernels['gradient'] + self.scatter_vector('element_vector')
        k_apply_c = pre_apply + "scalar_t element_vector[12]={0}; const int stride=1;\n" + kernels['apply'] + self.scatter_vector('element_vector')
        k_value_c = pre_grad + "scalar_t element_scalar[1]={0};\n" + kernels['value']

        grad_code = self.render_gradient(f"{opname}_{tag}_gradient", k_grad_c)
        apply_code = self.render_apply(f"{opname}_{tag}_apply", k_apply_c)
        value_code = self.render_value(f"{opname}_{tag}_value", k_value_c)

        with open(os.path.join(out_dir, f"{opname}_{tag}_gradient.c"), "w") as f:
            f.write(grad_code)
        with open(os.path.join(out_dir, f"{opname}_{tag}_apply.c"), "w") as f:
            f.write(apply_code)
        with open(os.path.join(out_dir, f"{opname}_{tag}_value.c"), "w") as f:
            f.write(value_code)

    def get(self, name: str) -> str:
        if name not in self._tpl:
            raise KeyError(f"Template '{name}' not found (available: {list(self._tpl.keys())})")
        return self._tpl[name]
