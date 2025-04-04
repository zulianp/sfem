#!/usr/bin/env python3

from sfem_codegen import *

simplify_expr = False
# simplify_expr = True

# class Tet4:
# 	def __init__(self):
# 		self.type = 'TET4'
# 		self.dim = 3


class GenericFE:
    def __init__(self, name, dim):
        self.type = "GENERIC"
        self.dim = dim

        self.value_ = sp.symbols(name)

        temp_grad = []
        for d in range(0, dim):
            s = sp.symbols(f"grad_{name}[{d}]")
            temp_grad.append(s)

        self.grad_ = sp.Matrix(dim, 1, temp_grad)

    def is_generic(self):
        return True

    def grad(self):
        return self.grad_

    def value(self):
        return self.value_


class TensorProductFE:
    def __init__(self, elem_type, block_size):
        self.elem_type = elem_type
        self.block_size = block_size

        if elem_type.is_generic():
            temp_grad = []

            g = elem_type.grad()

            for d1 in range(0, block_size):
                z = [0] * (block_size * elem_type.dim)
                G = sp.Matrix(block_size, elem_type.dim, z)

                for d2 in range(0, elem_type.dim):
                    G[d1, d2] = g[d2]

                temp_grad.append(G)

            self.grad_ = temp_grad

            #

            temp_value = []
            v = elem_type.value()

            for d1 in range(0, block_size):
                z = [0] * (block_size)
                V = sp.Matrix(block_size, 1, z)

                temp_value.append(V)

            self.value_ = temp_value

        else:
            # Implement me
            assert False

    def is_generic(self):
        return self.elem_type.is_generic()

    def grad(self):
        return self.grad_

    def value(self):
        return self.value_


class Function:
    def __init__(self, name, elem_type):
        self.name = name
        self.grad_name = f"grad_{name}"
        self.elem_type = elem_type
        self.block_size = 1

        self.value_ = sp.symbols(name, real=True)

        grad_list = []

        dim = elem_type.dim
        for d1 in range(0, dim):
            s = sp.symbols(f"{self.grad_name}[{d1}]", real=True)
            grad_list.append(s)

        self.grad_ = sp.Matrix(dim, 1, grad_list)

    def grad(self):
        return self.grad_

    def value(self):
        return self.value_

    def elem(self):
        return self.elem_type


class VectorFunction:
    def __init__(self, name, elem_type, block_size):
        self.name = name
        self.grad_name = f"grad_{name}"
        self.block_size = block_size
        self.elem_type = TensorProductFE(elem_type, block_size)

        dim = elem_type.dim

        grad_list = []
        for d1 in range(0, block_size):
            for d2 in range(0, elem_type.dim):
                s = sp.symbols(f"{self.grad_name}[{d1*elem_type.dim + d2}]", real=True)
                grad_list.append(s)

        self.grad_ = sp.Matrix(block_size, dim, grad_list)

        #

        value_list = []
        for d1 in range(0, block_size):
            s = sp.symbols(f"{self.name}[{d1}]", real=True)
            value_list.append(s)

        self.value_ = sp.Matrix(block_size, 1, value_list)

    def grad(self):
        return self.grad_

    def value(self):
        return self.value_


def is_matrix(expr):
    return sp.matrices.immutable.ImmutableDenseMatrix == type(
        expr
    ) or sp.matrices.dense.MutableDenseMatrix == type(expr)


def derivative(f, var):
    if is_matrix(var):
        rows, cols = var.shape
        z = [0] * (rows * cols)
        ret = sp.Matrix(rows, cols, z)

        for d1 in range(0, rows):
            for d2 in range(0, cols):
                ret[d1, d2] = sp.diff(f, var[d1, d2])
        return ret
    else:
        return sp.diff(f, var)


def directional_derivative(f, var, h):
    deriv = derivative(f, var)
    return inner(deriv, h)


class Model:
    def __init__(self, elem_trial, elem_test):
        self.elem_trial = elem_trial
        self.elem_test = elem_test
        self.dim = elem_test.dim

        self.dV = sp.symbols("dV")

    def set_energy(self, e):
        self.energy_ = e

    def energy(self):
        return self.energy_

    def makeenergy(self):
        integr = self.energy() * self.dV

        if simplify_expr:
            integr = sp.simplify(integr)

        form = sp.symbols(f"element_scalar[0]")
        energy_expr = ast.AddAugmentedAssignment(form, integr)
        return energy_expr


class PhaseFieldBase(Model):
    def __init__(self, elem_trial, elem_test):
        super().__init__(elem_trial, elem_test)
        dim = elem_test.dim
        self.elem_test = elem_test
        self.tp_element_test = TensorProductFE(elem_test, dim)

        self.phase = Function("c", elem_trial)
        self.displacement = VectorFunction("u", elem_trial, dim)
        self.increment_phase = Function("inc_c", elem_trial)
        self.increment_displacement = VectorFunction("inc_u", elem_trial, dim)

    def initialize(self, energy):
        self.set_energy(energy)

        trial = self.phase.elem_type
        test = self.elem_test
        phase = self.phase

        displacement = self.displacement
        tp_test = self.tp_element_test

        d_trial_grad = self.displacement.elem_type.grad()
        d_test_grad = tp_test.grad()
        n = len(d_test_grad)
        assert n == tp_test.block_size

        dedc = derivative(energy, phase.value())
        dedgradc = derivative(energy, phase.grad())

        # Gradient
        self.grad_wrt_c = (dedc * test.value()) + inner(dedgradc, test.grad())

        dedu = derivative(energy, displacement.grad())
        self.grad_wrt_u = sp.Matrix(n, 1, [0] * n)

        for d1 in range(0, n):
            self.grad_wrt_u[d1] = inner(dedu, d_test_grad[d1])

        # Hessian
        self.hessian_wrt_uu = sp.Matrix(n, n, [0] * (n * n))

        H_grad_u = sp.Matrix(n, 1, [0] * n)
        for d1 in range(0, n):
            H_grad_u[d1] = inner(dedu, d_trial_grad[d1])

        for i_trial in range(0, n):
            d2edu2 = derivative(H_grad_u[i_trial], displacement.grad())

            for i_test in range(0, n):
                self.hessian_wrt_uu[i_trial, i_test] = inner(
                    d2edu2, d_test_grad[i_test]
                )

        self.hessian_wrt_uc = sp.Matrix(n, 1, [0] * n)

        for i_trial in range(0, n):
            d2edudc = derivative(H_grad_u[i_trial], phase.grad())
            self.hessian_wrt_uc[i_trial] = (
                inner(d2edudc, test.grad())
                + sp.diff(H_grad_u[i_trial], phase.value()) * test.value()
            )

        # cc
        grad_wrt_c = inner(dedgradc, trial.grad()) + dedc * trial.value()
        self.hessian_wrt_cc = (
            inner(derivative(grad_wrt_c, phase.grad()), test.grad())
            + derivative(grad_wrt_c, phase.value()) * test.value()
        )

    def makegradu(self, idx, i):
        integr = self.grad_wrt_u[i] * self.dV

        if simplify_expr:
            integr = sp.simplify(integr)

        lform = sp.symbols(f"element_vector[{idx}]")
        expr = ast.AddAugmentedAssignment(lform, integr)
        return expr

    def makegradc(self, idx):
        integr = self.grad_wrt_c * self.dV

        if simplify_expr:
            integr = sp.simplify(integr)

        lform = sp.symbols(f"element_vector[{idx}]")
        expr = ast.AddAugmentedAssignment(lform, integr)
        return expr

    def makegrad(self):
        ndisp = self.tp_element_test.block_size
        n = ndisp + 1

        expr = [0] * n

        for i in range(0, ndisp):
            expr[i] = self.makegradu(i, i)

        expr[ndisp] = self.makegradc(ndisp)
        return expr

    def makehessianuu(self, idx, i, j):
        integr = self.hessian_wrt_uu[i, j] * self.dV

        if simplify_expr:
            integr = sp.simplify(integr)

        bform = sp.symbols(f"element_matrix[{idx}]")
        return ast.AddAugmentedAssignment(bform, integr)

    def makehessianuc(self, idx, i, j):
        integr = self.hessian_wrt_uc[i] * self.dV

        if simplify_expr:
            integr = sp.simplify(integr)

        bform = sp.symbols(f"element_matrix[{idx}]")
        return ast.AddAugmentedAssignment(bform, integr)

    def makehessiancu(self, idx, j, i):
        integr = self.hessian_wrt_uc[i] * self.dV

        if simplify_expr:
            integr = sp.simplify(integr)

        bform = sp.symbols(f"element_matrix[{idx}]")
        return ast.AddAugmentedAssignment(bform, integr)

    def makehessiancc(self, idx):
        integr = self.hessian_wrt_cc * self.dV

        if simplify_expr:
            integr = sp.simplify(integr)

        bform = sp.symbols(f"element_matrix[{idx}]")
        return ast.AddAugmentedAssignment(bform, integr)

    def makehessian(self):
        ndisp = self.tp_element_test.block_size
        n = ndisp + 1

        expr = [0] * (n * n)

        for i in range(0, ndisp):
            expr[i * n + ndisp] = self.makehessianuc(i * n + ndisp, i, ndisp)
            expr[ndisp * n + i] = self.makehessiancu(ndisp * n + i, ndisp, i)

            for j in range(0, ndisp):
                expr[i * n + j] = self.makehessianuu(i * n + j, i, j)

        expr[ndisp * n + ndisp] = self.makehessiancc(ndisp * n + ndisp)
        return expr

    def makeapply_blocks(self):
        ndisp = self.tp_element_test.block_size

        expr_uu = [0] * ndisp
        expr_uc = [0] * ndisp
        expr_cu = [0]
        expr_cc = [0]

        u_inc = self.increment_displacement.value()
        c_inc = self.increment_phase.value()

        for i in range(0, ndisp):
            for j in range(0, ndisp):
                integr = self.hessian_wrt_uu[i, j] * u_inc[j] * self.dV

                if simplify_expr:
                    integr = sp.simplify(integr)

                expr_uu[i] = integr

        for i in range(0, ndisp):
            integr = self.hessian_wrt_uc[i] * c_inc * self.dV

            if simplify_expr:
                integr = sp.simplify(integr)

            expr_uc[i] += integr

        for j in range(0, ndisp):
            integr = self.hessian_wrt_uc[j] * u_inc[j] * self.dV

            if simplify_expr:
                integr = sp.simplify(integr)

            expr_cu[0] += integr

        expr_cc[0] = self.hessian_wrt_cc * c_inc * self.dV

        return expr_uu, expr_uc, expr_cu, expr_cc

    def makeapply(self):
        ndisp = self.tp_element_test.block_size
        n = ndisp + 1

        expr_uu, expr_uc, expr_cu, expr_cc = self.makeapply_blocks()

        expr = []

        for i in range(0, ndisp):
            integr = expr_uu[i] + expr_uc[i]

            if simplify_expr:
                integr = sp.simplify(integr)

            lform = sp.symbols(f"element_vector[{i}]")
            expr.append(ast.AddAugmentedAssignment(lform, integr))

        integr = expr_cc[0] + expr_cu[0]

        if simplify_expr:
            integr = sp.simplify(integr)

        lform = sp.symbols(f"element_vector[{ndisp}]")
        expr.append(ast.AddAugmentedAssignment(lform, integr))

        return expr

    def split_apply_code(self):
        ndisp = self.tp_element_test.block_size
        n = ndisp + 1

        expr_uu, expr_uc, expr_cu, expr_cc = self.makeapply_blocks()

        expr_apply_uu = []

        for i in range(0, ndisp):
            integr = expr_uu[i]  # + expr_uc[i]

            if simplify_expr:
                integr = sp.simplify(integr)

            lform = sp.symbols(f"element_vector[{i}]")
            expr_apply_uu.append(ast.AddAugmentedAssignment(lform, integr))

        integr = expr_cc[0]  # + expr_cu[0]

        if simplify_expr:
            integr = sp.simplify(integr)

        lform = sp.symbols(f"element_vector[{0}]")
        expr_apply_cc = [ast.AddAugmentedAssignment(lform, integr)]

        code_apply_uu = c_gen(expr_apply_uu)
        code_apply_cc = c_gen(expr_apply_cc)
        return code_apply_uu, code_apply_cc

    def make_split_hessian_uu(self):
        ndisp = self.tp_element_test.block_size

        expr = [0] * (ndisp * ndisp)
        for i in range(0, ndisp):
            for j in range(0, ndisp):
                expr[i * ndisp + j] = self.makehessianuu(i * ndisp + j, i, j)

        return expr

    def make_split_hessian_uc(self):
        ndisp = self.tp_element_test.block_size

        expr = [0] * ndisp
        for i in range(0, ndisp):
            expr[i] = self.makehessianuc(i, i, 0)
        return expr

    def make_split_hessian_cu(self):
        ndisp = self.tp_element_test.block_size

        expr = [0] * ndisp
        for i in range(0, ndisp):
            expr[i] = self.makehessiancu(i, 0, i)
        return expr

    def make_split_hessian_cc(self):
        expr = [self.makehessiancc(0)]
        return expr

    def genreate_split_code(self):
        energy_code = self.energy_code()
        args_energy = self.energy_args()

        # Split operators
        gradient_c_expr = self.makegradc(0)
        gradient_c_code = c_gen(gradient_c_expr)

        ndisp = self.tp_element_test.block_size
        gradient_u_expr = []

        for d1 in range(0, ndisp):
            gradient_u_expr.append(self.makegradu(d1, d1))

        gradient_u_code = c_gen(gradient_u_expr)

        hessian_uu_expr = self.make_split_hessian_uu()
        hessian_uu_code = c_gen(hessian_uu_expr)

        hessian_cc_expr = self.make_split_hessian_cc()
        hessian_cc_code = c_gen(hessian_cc_expr)

        hessian_uc_expr = self.make_split_hessian_uc()
        hessian_uc_code = c_gen(hessian_uc_expr)

        hessian_cu_expr = self.make_split_hessian_cu()
        hessian_cu_code = c_gen(hessian_cu_expr)

        params = self.param_string()

        args_gradient_u = (
            params
            + f"""const real_t {self.phase.name}, 
			const real_t *{self.phase.grad_name}, 
			const real_t *{self.displacement.grad_name},
			const real_t *grad_test,
			const real_t dV,
			real_t *element_vector
			"""
        )
        args_gradient_c = (
            params
            + f"""const real_t {self.phase.name}, 
			const real_t *{self.phase.grad_name}, 
			const real_t *{self.displacement.grad_name},
			const real_t test,
			const real_t *grad_test,
			const real_t dV,
			real_t *element_vector
			"""
        )
        args_hessian_uu = (
            params
            + f"""const real_t {self.phase.name},  
			const real_t *grad_test,
			const real_t *grad_trial,
			const real_t dV,
			real_t *element_matrix"""
        )

        args_hessian_uc = (
            params
            + f"""const real_t {self.phase.name},  
			const real_t *{self.displacement.grad_name},
			const real_t test,
			const real_t *grad_trial,
			const real_t dV,
			real_t *element_matrix"""
        )

        args_hessian_cu = args_hessian_uc

        args_hessian_cc = (
            params
            + f"""const real_t *{self.displacement.grad_name},
			const real_t test,
			const real_t trial,
			const real_t *grad_test,
			const real_t *grad_trial,
			const real_t dV,
			real_t *element_matrix"""
        )

        tpl = self.read_field_split_tpl()

        code_apply_uu, code_apply_cc = self.split_apply_code()

        args_apply_uu = (
            params
            + f"""const real_t {self.phase.name}, 
			const real_t *{self.increment_displacement.name},
			const real_t *grad_trial,
			const real_t *grad_test,
			const real_t dV,
			real_t *element_vector
			"""
        )

        args_apply_cc = (
            params
            + f"""const real_t {self.phase.name}, 
			const real_t *{self.phase.grad_name}, 
			const real_t *{self.displacement.grad_name},
			const real_t {self.increment_phase.name},
			const real_t test,
			const real_t trial,
			const real_t *grad_trial,
			const real_t *grad_test,
			const real_t dV,
			real_t *element_vector
			"""
        )

        output = tpl.format(
            kernel_name=self.kernel_name,
            energy=energy_code,
            gradient_u=gradient_u_code,
            gradient_c=gradient_c_code,
            hessian_uu=hessian_uu_code,
            hessian_uc=hessian_uc_code,
            hessian_cu=hessian_cu_code,
            hessian_cc=hessian_cc_code,
            apply_uu=code_apply_uu,
            apply_cc=code_apply_cc,
            args_energy=args_energy,
            args_gradient_u=args_gradient_u,
            args_gradient_c=args_gradient_c,
            args_hessian_uu=args_hessian_uu,
            args_hessian_uc=args_hessian_uc,
            args_hessian_cu=args_hessian_cu,
            args_hessian_cc=args_hessian_cc,
            args_apply_uu=args_apply_uu,
            args_apply_cc=args_apply_cc,
        )

        return output

    def param_string(self):
        params = ""
        for p in self.params:
            params += f"const real_t {p},\n"
        return params

    def energy_code(self):
        energy_expr = self.makeenergy()
        energy_code = c_gen(energy_expr)
        return energy_code

    def energy_args(self):
        params = self.param_string()
        args_energy = (
            params
            + f"""const real_t {self.phase.name}, 
			const real_t *{self.phase.grad_name}, 
			const real_t *{self.displacement.grad_name},
			const real_t dV,
			real_t *element_scalar
			"""
        )
        return args_energy

    def generate_monolithic_code(self):
        energy_code = self.energy_code()
        args_energy = self.energy_args()

        gradient_expr = self.makegrad()
        gradient_code = c_gen(gradient_expr)

        hessian_expr = self.makehessian()
        hessian_code = c_gen(hessian_expr)

        apply_expr = self.makeapply()
        apply_code = c_gen(apply_expr)

        params = self.param_string()
        args_gradient = (
            params
            + f"""const real_t {self.phase.name}, 
			const real_t *{self.phase.grad_name}, 
			const real_t *{self.displacement.grad_name},
			const real_t test,
			const real_t *grad_test,
			const real_t dV,
			real_t *element_vector
			"""
        )

        args_hessian = (
            params
            + f"""const real_t {self.phase.name},  
			const real_t *{self.displacement.grad_name},
			const real_t test,
			const real_t trial,
			const real_t *grad_test,
			const real_t *grad_trial,
			const real_t dV,
			real_t *element_matrix"""
        )

        args_apply = (
            params
            + f"""const real_t {self.phase.name},  
			const real_t *{self.displacement.grad_name},
			const real_t {self.increment_phase.name},
			const real_t *{self.increment_displacement.name},
			const real_t test,
			const real_t trial,
			const real_t *grad_test,
			const real_t *grad_trial,
			const real_t dV,
			real_t *element_vector"""
        )

        tpl = self.read_tpl()

        output = tpl.format(
            kernel_name=self.kernel_name,
            energy=energy_code,
            gradient=gradient_code,
            hessian=hessian_code,
            apply=apply_code,
            args_energy=args_energy,
            args_gradient=args_gradient,
            args_hessian=args_hessian,
            args_apply=args_apply,
        )

        return output

    def read_field_split_tpl(self):
        tpl_path = "tpl/PhaseFieldSplit_tpl.c"

        tpl = None
        with open(tpl_path, "r") as f:
            tpl = f.read()
            return tpl

    def read_tpl(self):
        tpl_path = "tpl/PhaseField_tpl.c"

        tpl = None
        with open(tpl_path, "r") as f:
            tpl = f.read()
            return tpl

    def generate_code(self):
        with open(f"{self.kernel_name}.c", "w") as f:
            output = self.generate_monolithic_code()
            f.write(output)
            f.close()

        with open(f"{self.kernel_name}_split.c", "w") as f:
            output = self.genreate_split_code()
            f.write(output)
            f.close()
