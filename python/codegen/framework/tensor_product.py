import codegen.framework.fem.tensor_product as _impl
globals().update({
    _name: _value
    for _name, _value in vars(_impl).items()
    if not _name.startswith("__")
})
