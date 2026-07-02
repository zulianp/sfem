if __name__ == "__main__":
    import runpy

    runpy.run_module("codegen.framework.generators.neohookean_ogden", run_name="__main__")
else:
    import codegen.framework.generators.neohookean_ogden as _impl

    globals().update({
        _name: _value
        for _name, _value in vars(_impl).items()
        if not _name.startswith("__")
    })
