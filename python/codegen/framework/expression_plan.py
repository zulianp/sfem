from dataclasses import dataclass

from .forms import FormOrder


@dataclass(frozen=True)
class KernelExpressionPlan:
    name: str
    form_order: FormOrder
    role: object
    expression_graph: object = None
    weak_form: object = None
    coefficients: tuple = ()
    dependencies: object = None
    required_streams: tuple = ()
    diagnostics: object = None
    fields: tuple = ()
    blocks: tuple = ()
    source: object = None
    output_mode: str = ""
    has_direction: bool = False

    def __post_init__(self):
        name = str(self.name)
        if not name:
            raise ValueError("kernel expression plan requires a name")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "form_order", FormOrder(self.form_order))
        object.__setattr__(self, "coefficients", tuple(self.coefficients))
        object.__setattr__(self, "required_streams", tuple(self.required_streams))
        object.__setattr__(self, "fields", tuple(self.fields))
        object.__setattr__(self, "blocks", tuple(self.blocks))
        object.__setattr__(self, "output_mode", str(self.output_mode))
        object.__setattr__(self, "has_direction", bool(self.has_direction))

    def to_dict(self):
        return {
            "name": self.name,
            "form_order": self.form_order.value,
            "role": _role_name(self.role),
            "coefficients": [_symbol_name(symbol) for symbol in self.coefficients],
            "dependencies": None
            if self.dependencies is None
            else {
                "current": bool(getattr(self.dependencies, "current", False)),
                "previous": bool(getattr(self.dependencies, "previous", False)),
                "direction": bool(getattr(self.dependencies, "direction", False)),
                "geometry": bool(getattr(self.dependencies, "geometry", False)),
                "parameters": [
                    _symbol_name(symbol)
                    for symbol in getattr(self.dependencies, "parameters", ())
                ],
            },
            "required_streams": [_stream_name(stream) for stream in self.required_streams],
            "fields": [_field_name(field) for field in self.fields],
            "blocks": [_block_name(block) for block in self.blocks],
            "output_mode": self.output_mode,
            "has_direction": self.has_direction,
        }


def _role_name(role):
    return getattr(role, "value", getattr(role, "name", str(role)))


def _symbol_name(symbol):
    return getattr(symbol, "name", str(symbol))


def _stream_name(stream):
    return getattr(stream, "name", str(stream))


def _field_name(field):
    return getattr(field, "name", str(field))


def _block_name(block):
    return getattr(block, "name", str(block))
