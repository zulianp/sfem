"""C and C++ syntax for SFEM kernels: the printer and its naming helpers.

Everything here turns a SymPy expression or a symbol into text.  It lived in
``symbolic/core.py`` until the layering work, which put a ``C99CodePrinter``
subclass inside the layer that is supposed to hold no syntax at all.  Nothing
above the emission layer needs any of it.
"""

import sympy as sp
from sympy.printing.c import C99CodePrinter


_SFEM_SPECIALIZED_POW_MAX_EXPONENT = 16

# Printers are cached per scalar type; building one is not cheap and the
# emitters call _sfem_ccode once per generated expression.
_SFEM_CCODE_PRINTERS = {}


class _SfemCCodePrinter(C99CodePrinter):
    def __init__(self, scalar_type="scalar_t", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scalar_type = scalar_type

    def _scalar_literal(self, value):
        return "%s(%s)" % (self._scalar_type, value)

    def _print_Integer(self, expr):
        return self._scalar_literal("%d" % int(expr))

    def _print_Float(self, expr):
        return self._scalar_literal(super()._print_Float(expr))

    def _print_Rational(self, expr):
        return "(%s / %s)" % (
            self._scalar_literal("%d" % int(expr.p)),
            self._scalar_literal("%d" % int(expr.q)),
        )

    def _print_Pow(self, expr):
        base, exponent = expr.as_base_exp()
        if exponent.is_Integer:
            exponent_value = int(exponent)
            if abs(exponent_value) <= _SFEM_SPECIALIZED_POW_MAX_EXPONENT:
                if exponent_value == 0:
                    return self._scalar_literal("1")
                if exponent_value == 1:
                    return self._print(base)
                suffix = "m%d" % abs(exponent_value) if exponent_value < 0 else "%d" % exponent_value
                return "pow_%s(%s)" % (suffix, self._print(base))
        return super()._print_Pow(expr)


def _sfem_ccode(expression, scalar_type="scalar_t"):
    printer = _SFEM_CCODE_PRINTERS.get(scalar_type)
    if printer is None:
        printer = _SfemCCodePrinter(scalar_type)
        _SFEM_CCODE_PRINTERS[scalar_type] = printer
    return printer.doprint(expression)


def _sfem_pow_function_name(exponent):
    exponent = int(exponent)
    if exponent < 0:
        return "pow_m%d" % abs(exponent)
    return "pow_%d" % exponent


def _sfem_pow_product_expression(exponent):
    exponent = int(exponent)
    if exponent == 0:
        return "T(1)"
    return " * ".join("x" for _ in range(exponent))


def _sfem_math_function_lines(inline_qualifier="SFEM_INLINE"):
    lines = []
    for exponent in range(2, _SFEM_SPECIALIZED_POW_MAX_EXPONENT + 1):
        lines.extend(
            [
                "template <typename T>",
                "static %s T %s(const T x) {"
                % (inline_qualifier, _sfem_pow_function_name(exponent)),
                "    return %s;" % _sfem_pow_product_expression(exponent),
                "}",
                "",
            ]
        )
    for exponent in range(1, _SFEM_SPECIALIZED_POW_MAX_EXPONENT + 1):
        lines.extend(
            [
                "template <typename T>",
                "static %s T %s(const T x) {"
                % (inline_qualifier, _sfem_pow_function_name(-exponent)),
                "    return T(1) / %s(x);" % _sfem_pow_function_name(exponent)
                if exponent > 1
                else "    return T(1) / x;",
                "}",
                "",
            ]
        )
    return lines


def _sfem_math_header_source(
    header_guard_suffix="HPP",
    inline_qualifier="SFEM_INLINE",
    define_sfem_inline=True,
):
    guard = "SFEM_CODEGEN_KERNEL_MATH_%s" % header_guard_suffix
    lines = [
        "#ifndef %s" % guard,
        "#define %s" % guard,
        "",
    ]
    if define_sfem_inline:
        lines.extend(
            [
                "#ifndef SFEM_INLINE",
                "#define SFEM_INLINE inline",
                "#endif",
                "",
            ]
        )
    lines.extend(["namespace sfem {", "namespace codegen {", ""])
    lines.extend(_sfem_math_function_lines(inline_qualifier))
    lines.extend(["} // namespace codegen", "} // namespace sfem", "", "#endif", ""])
    return "\n".join(lines)


def _component_name(component):
    return ("x", "y", "z")[component]


def _cpp_symbol(symbol, output_name):
    if isinstance(symbol, str) and symbol.startswith("output:"):
        index = symbol.rsplit(":", 1)[-1]
        return "%s[%s]" % (output_name, index)
    return str(symbol)


def _cpp_lvalue(symbol, output_name):
    base, _ = _indexed_symbol(symbol)
    if isinstance(symbol, str) and symbol.startswith("output:"):
        return _cpp_symbol(symbol, output_name)
    if base is not None:
        return _cpp_symbol(symbol, output_name)
    return "*%s" % _cpp_symbol(symbol, output_name)


def _cpp_argument_name(argument):
    return argument.replace("*", " ").split()[-1]


def _cpp_macro_name(name):
    chars = []
    for char in str(name):
        if char.isalnum():
            chars.append(char.upper())
        else:
            chars.append("_")
    return "".join(chars)


def _indexed_symbol(symbol):
    text = str(symbol)
    if not text.endswith("]"):
        return None, None
    bracket = text.rfind("[")
    if bracket <= 0:
        return None, None
    index = text[bracket + 1 : -1]
    if not index.isdigit():
        return None, None
    return text[:bracket], int(index)


def _group_kernel_symbols(symbols):
    arrays = {}
    scalars = []
    for symbol in symbols:
        base, index = _indexed_symbol(symbol)
        if base is None:
            scalars.append(symbol)
        else:
            arrays.setdefault(base, set()).add(index)
    return arrays, tuple(scalars)


def _direct_output_targets(output_targets):
    direct_output_targets = tuple(
        target
        for target in output_targets
        if not (isinstance(target, str) and target.startswith("output:"))
    )
    return direct_output_targets, len(direct_output_targets) != len(output_targets)
