import sympy as sp

from sympy.printing.c import ccode
import sympy.codegen.ast as ast

# Symbols
a, b, c, d, e, f = sp.symbols("a b c d e f")
fma = sp.Function("fma")

def _decompose_simple_product(t):
    """Return (x, y) if t is a product of >=2 factors with coeff separated."""
    if not isinstance(t, sp.Mul):
        return None

    coeff, rest = t.as_coeff_Mul()
    factors = list(rest.args) if isinstance(rest, sp.Mul) else [rest]
    if len(factors) < 2:
        return None
    x = sp.simplify(coeff * sp.Mul(*factors[:-1]))
    y = factors[-1]
    return (x, y)

def _pack_fmas_balanced(terms):
    """Build a balanced FMA tree from a list of Add terms."""
    if len(terms) == 1:
        return terms[0]

    if len(terms) == 2:
        left, right = terms
        xy = _decompose_simple_product(left)
        if xy:
            return fma(xy[0], xy[1], right)
        xy = _decompose_simple_product(right)
        if xy:
            return fma(xy[0], xy[1], left)
        return sp.Add(left, right)

    mid = len(terms) // 2
    left_tree = _pack_fmas_balanced(terms[:mid])
    right_tree = _pack_fmas_balanced(terms[mid:])
    return _pack_fmas_balanced([left_tree, right_tree])

def fma_rewrite(expr):
    """Recursively rewrite Add nodes into balanced FMA trees, factoring common terms first."""
    if not expr.args:
        return expr

    # Recurse into children
    expr2 = expr.func(*[fma_rewrite(arg) for arg in expr.args])

    # Factor common terms from Add
    if isinstance(expr2, sp.Add):
        factored = sp.factor_terms(expr2)
        if isinstance(factored, sp.Mul):  # something was factored out
            coeff, inside = factored.as_coeff_Mul()
            # coeff may include numbers and symbols
            if inside.is_Add:
                return coeff * _pack_fmas_balanced(list(inside.args))
            else:
                return coeff * inside

        # otherwise no common factor
        return _pack_fmas_balanced(list(expr2.args))

    return expr2

def cleanup(expr):
    """Remove degenerate FMAs."""
    if isinstance(expr, sp.Function) and expr.func == fma:
        x, y, z = map(cleanup, expr.args)
        if z == 0:
            return x * y
        if y == 1:
            return x + z
        return fma(x, y, z)
    if expr.args:
        return expr.func(*[cleanup(arg) for arg in expr.args])
    return expr

def generate_c_code(expr, result_name="out"):


    # Apply FMA rewrite and cleanup
    expr_opt = cleanup(fma_rewrite(expr))

    # Run common subexpression elimination
    cse_repls, reduced = sp.cse(expr_opt, symbols=sp.numbered_symbols("t"))

    lines = []
    for sym, rhs in cse_repls:
        lines.append(f"double {sym} = {ccode(rhs)};")
    lines.append(f"double {result_name} = {ccode(reduced[0])};")

    return "\n".join(lines)


# ----------------
# Tests
# ----------------
examples = [
    2*a*b + 3*a**2*b*c*(c/b) + a/b/(3*a**2*b*c),
    4*a*b + 6*a*b*d + e,
    3*a*b + c,
    -2*a*b*c + d,
    a*b + c + d*e,
    a*b*(c + d*e)*(a*b + c) + c,
    a*b*(c + d*e)*(a*b + c) + c
]

for ex in examples:
    print("orig :", ex)
    print("fma  :", generate_c_code(ex))
    print()

expr = []
idx = 0
for ex in examples:
    expr.append(ast.Assignment(sp.symbols(f"v[{idx}]"), cleanup(fma_rewrite(ex))))
    idx += 1


cse_repls, reduced = sp.cse(expr, symbols=sp.numbered_symbols("t"))

lines = []
for sym, rhs in cse_repls:
    lines.append(f"double {sym} = {ccode(rhs)};")

for r in reduced:
    lines.append(ccode(r));

print( "\n".join(lines))
