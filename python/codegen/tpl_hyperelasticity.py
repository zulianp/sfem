#!/usr/bin/env python3

import os
import sympy as sp
from sfem_codegen import c_gen


class TPLHyperelasticity:
    """Modern template system for hyperelasticity C code generation.
    
    Works with expression lists instead of pre-generated C strings.
    Provides clean separation between FEM boilerplate and mathematical expressions.
    """

    def __init__(self, fe, base_dir: str = None):
        """Initialize templates and FE-specific configuration.
        
        Args:
            fe: finite element descriptor (e.g., Tet4())
            base_dir: path to tpl/hyperelasticity directory
        """
        self.fe = fe
        if base_dir is None:
            # repo_root/python/codegen/.. -> repo_root/tpl/hyperelasticity
            here = os.path.dirname(__file__)
            base_dir = os.path.normpath(os.path.join(here, "..", "..", "tpl", "hyperelasticity"))

        if not os.path.isdir(base_dir):
            raise FileNotFoundError(f"Template directory not found: {base_dir}")

        self._templates = {}
        for fname in os.listdir(base_dir):
            fpath = os.path.join(base_dir, fname)
            if not os.path.isfile(fpath):
                continue
            name, _ = os.path.splitext(fname)
            if name.endswith("_tpl"):
                name = name[:-4]
            with open(fpath, "r") as f:
                self._templates[name] = f.read()

        # FE-specific configuration
        self._config = self._build_fe_config()

    def _build_fe_config(self):
        """Build FE-specific configuration."""
        name = getattr(self.fe, 'name', lambda: '')()
        nodes = self.fe.n_nodes()
        dims = self.fe.manifold_dim()

        config = {
            'name': name,
            'nodes': nodes,
            'dims': dims,
            'is_tet4': name.upper() == 'TET4' or (nodes == 4 and dims == 3),
        }

        # TET4 specific configuration
        if config['is_tet4']:
            config.update({
                'includes': ['#include "tet4_inline_cpu.h"', "#iinclude <math.h>"],
                'geom_locals': ['scalar_t lx[4];', 'scalar_t ly[4];', 'scalar_t lz[4];'],
                'state_locals': [
                    'scalar_t element_ux[4];',
                    'scalar_t element_uy[4];', 
                    'scalar_t element_uz[4];'
                ],
                'dir_locals': [
                    'scalar_t element_hx[4];',
                    'scalar_t element_hy[4];',
                    'scalar_t element_hz[4];'
                ],
                'output_locals': [
                    'accumulator_t element_outx[4];',
                    'accumulator_t element_outy[4];',
                    'accumulator_t element_outz[4];',
                    'for (int d = 0; d < 4; ++d) { element_outx[d]=0; element_outy[d]=0; element_outz[d]=0; }'
                ],
                'jacobian_locals': [
                    'scalar_t jacobian_adjugate[9];',
                    'scalar_t jacobian_determinant = 0;'
                ],
                'quad_setup': 'const scalar_t qw = 1;',
                'gather_conn': 'for (int v = 0; v < 4; ++v) { ev[v] = elements[v][i]; }',
                'gather_geom': [
                    'for (int v = 0; v < 4; ++v) { lx[v]=x[ev[v]]; ly[v]=y[ev[v]]; lz[v]=z[ev[v]]; }'
                ],
                'gather_u': [
                    'for (int v = 0; v < 4; ++v) {',
                    '    const ptrdiff_t gi = ev[v]*u_stride;',
                    '    element_ux[v]=ux[gi]; element_uy[v]=uy[gi]; element_uz[v]=uz[gi];',
                    '}'
                ],
                'gather_h': [
                    'for (int v = 0; v < 4; ++v) {',
                    '    const ptrdiff_t gi = ev[v]*h_stride;',
                    '    element_hx[v]=hx[gi]; element_hy[v]=hy[gi]; element_hz[v]=hz[gi];',
                    '}'
                ],
                'jacobian_at_qp': [
                    'tet4_adjugate_and_det_s(',
                    '    lx[0], lx[1], lx[2], lx[3],',
                    '    ly[0], ly[1], ly[2], ly[3],',
                    '    lz[0], lz[1], lz[2], lz[3],',
                    '    jacobian_adjugate, &jacobian_determinant);'
                ],
                'scatter_output': [
                    'for (int a=0;a<4;++a){',
                    '    element_outx[a]+=element_vector[0*4+a];',
                    '    element_outy[a]+=element_vector[1*4+a];',
                    '    element_outz[a]+=element_vector[2*4+a];',
                    '}'
                ]
            })

        return config

    def _generate_fem_boilerplate(self, need_trial=False):
        """Generate FEM boilerplate code as expression list."""
        if not self._config['is_tet4']:
            return []

        # Generate jacobian inverse computation
        jac_inv_expr = [
            'scalar_t jac_inv[9];',
            'for (int r=0;r<3;++r) for (int c=0;c<3;++c) jac_inv[r*3+c] = jacobian_adjugate[c*3+r] / jacobian_determinant;'
        ]

        # Generate reference gradients
        grad_ref_expr = [
            'const scalar_t grad_ref[12] = {-1,-1,-1, 1,0,0, 0,1,0, 0,0,1};'
        ]

        # Generate displacement gradient computation
        disp_grad_expr = [
            'scalar_t disp_grad[9] = {0};',
            'for (int a=0;a<4;++a){',
            '    const scalar_t ux=element_ux[a], uy=element_uy[a], uz=element_uz[a];',
            '    const scalar_t gx=grad_ref[a*3+0], gy=grad_ref[a*3+1], gz=grad_ref[a*3+2];',
            '    disp_grad[0]+=ux*gx; disp_grad[1]+=ux*gy; disp_grad[2]+=ux*gz;',
            '    disp_grad[3]+=uy*gx; disp_grad[4]+=uy*gy; disp_grad[5]+=uy*gz;',
            '    disp_grad[6]+=uz*gx; disp_grad[7]+=uz*gy; disp_grad[8]+=uz*gz;',
            '}',
            'scalar_t tmp[9];',
            'for (int r=0;r<3;++r){',
            '    for (int c=0;c<3;++c){',
            '        tmp[r*3+c]=disp_grad[r*3+0]*jac_inv[0*3+c]+disp_grad[r*3+1]*jac_inv[1*3+c]+disp_grad[r*3+2]*jac_inv[2*3+c];',
            '    }',
            '}',
            'for (int k=0;k<9;++k) disp_grad[k]=tmp[k];'
        ]

        # Generate trial gradient computation if needed
        trial_grad_expr = []
        if need_trial:
            trial_grad_expr = [
                'scalar_t grad_trial[9]={0};',
                'for (int a=0;a<4;++a){',
                '    const scalar_t hx=element_hx[a], hy=element_hy[a], hz=element_hz[a];',
                '    const scalar_t gx=grad_ref[a*3+0], gy=grad_ref[a*3+1], gz=grad_ref[a*3+2];',
                '    grad_trial[0]+=hx*gx; grad_trial[1]+=hx*gy; grad_trial[2]+=hx*gz;',
                '    grad_trial[3]+=hy*gx; grad_trial[4]+=hy*gy; grad_trial[5]+=hy*gz;',
                '    grad_trial[6]+=hz*gx; grad_trial[7]+=hz*gy; grad_trial[8]+=hz*gz;',
                '}',
                'for (int r=0;r<3;++r){',
                '    for (int c=0;c<3;++c){',
                '        tmp[r*3+c]=grad_trial[r*3+0]*jac_inv[0*3+c]+grad_trial[r*3+1]*jac_inv[1*3+c]+grad_trial[r*3+2]*jac_inv[2*3+c];',
                '    }',
                '}',
                'for (int k=0;k<9;++k) grad_trial[k]=tmp[k];'
            ]

        return jac_inv_expr + grad_ref_expr + disp_grad_expr + trial_grad_expr

    def _inject_expressions_into_template(self, template_name, kernel_expressions, need_trial=False):
        """Inject expression list into template and generate final C code."""
        template = self._templates[template_name]
        
        # Generate FEM boilerplate (as C code strings)
        fem_boilerplate = self._generate_fem_boilerplate(need_trial)
        
        # Convert expression list to C code
        if kernel_expressions:
            kernel_code = c_gen(kernel_expressions)
        else:
            kernel_code = ""
        
        # Combine FEM boilerplate with kernel code
        if kernel_code:
            full_kernel_code = '\n'.join(fem_boilerplate) + '\n' + kernel_code
        else:
            full_kernel_code = '\n'.join(fem_boilerplate)
        
        # Prepare template parameters
        params = {
            'FUNC_NAME': f"hyperelasticity_{self._config['name'].lower()}",
            'NODES': self._config['nodes'],
            'INCLUDES': '\n'.join(self._config.get('includes', [])),
            'INLINE_HEADERS': '',  # Add missing parameter
            'DECLARE_GEOM_LOCALS': '\n        '.join(self._config.get('geom_locals', [])),
            'DECLARE_STATE_LOCALS': '\n        '.join(self._config.get('state_locals', [])),
            'DECLARE_DIR_LOCALS': '\n        '.join(self._config.get('dir_locals', [])),
            'DECLARE_OUTPUT_LOCALS': '\n        '.join(self._config.get('output_locals', [])),
            'DECLARE_JACOBIAN_LOCALS': '\n        '.join(self._config.get('jacobian_locals', [])),
            'QUAD_SETUP': self._config.get('quad_setup', ''),
            'QUAD_LOOP_BEGIN': '{',  # Add missing parameter
            'QUAD_LOOP_END': '}',    # Add missing parameter
            'GATHER_CONN': self._config.get('gather_conn', ''),
            'GATHER_GEOM': '\n        '.join(self._config.get('gather_geom', [])),
            'GATHER_U': '\n        '.join(self._config.get('gather_u', [])),
            'GATHER_H': '\n        '.join(self._config.get('gather_h', [])),
            'JACOBIAN_AT_QP': '\n            '.join(self._config.get('jacobian_at_qp', [])),
            'SCATTER_OUT': '\n        '.join(self._config.get('scatter_output', [])),
        }
        
        # Add kernel-specific placeholders
        if template_name == 'gradient':
            params['KERNEL_GRADIENT'] = full_kernel_code
        elif template_name == 'apply':
            params['KERNEL_APPLY'] = full_kernel_code
        elif template_name == 'value':
            params['KERNEL_VALUE'] = full_kernel_code
        
        return template.format(**params)

    def render_gradient(self, kernel_expressions):
        """Render gradient template with expression list."""
        return self._inject_expressions_into_template('gradient', kernel_expressions, need_trial=False)

    def render_apply(self, kernel_expressions):
        """Render apply template with expression list."""
        return self._inject_expressions_into_template('apply', kernel_expressions, need_trial=True)

    def render_value(self, kernel_expressions):
        """Render value template with expression list."""
        return self._inject_expressions_into_template('value', kernel_expressions, need_trial=False)

    def emit_all(self, opname, kernels, out_dir=None):
        """Emit all C files for the current FE.
        
        Args:
            opname: operation name (e.g., 'hyperelasticity')
            kernels: dict with 'gradient', 'apply', 'value' keys containing expression lists
            out_dir: output directory (defaults to operators/generated)
        """
        if out_dir is None:
            here = os.path.dirname(__file__)
            out_dir = os.path.normpath(os.path.join(here, "..", "..", "operators", "generated"))
        os.makedirs(out_dir, exist_ok=True)

        tag = self._config['name'].lower()

        # Generate C code for each kernel
        grad_code = self.render_gradient(kernels.get('gradient', []))
        apply_code = self.render_apply(kernels.get('apply', []))
        value_code = self.render_value(kernels.get('value', []))

        # Write files
        with open(os.path.join(out_dir, f"{opname}_{tag}_gradient.c"), "w") as f:
            f.write(grad_code)
        with open(os.path.join(out_dir, f"{opname}_{tag}_apply.c"), "w") as f:
            f.write(apply_code)
        with open(os.path.join(out_dir, f"{opname}_{tag}_value.c"), "w") as f:
            f.write(value_code)

    def get_template(self, name):
        """Get raw template string."""
        if name not in self._templates:
            raise KeyError(f"Template '{name}' not found (available: {list(self._templates.keys())})")
        return self._templates[name]