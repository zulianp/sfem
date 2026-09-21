"""Every test module must import cleanly.

A module that fails to import is reported by ``unittest`` as a single
``_FailedTest`` and its real tests simply never run.  During the layering work
this happened twice -- a relocation removed a name from a package's re-export
surface, seventeen tests silently stopped running, and both the generated-source
snapshot and the targeted unit tests stayed green.  Watching the failure list
does not catch it; watching the test count does, and so does this.
"""

import importlib
import os
import pkgutil
import unittest


def _test_module_names():
    package = os.path.dirname(os.path.abspath(__file__))
    return sorted(
        name
        for _, name, _ in pkgutil.iter_modules([package])
        if name.startswith("test_")
    )


class ModuleImportTest(unittest.TestCase):
    def test_every_test_module_imports(self):
        failures = []
        for name in _test_module_names():
            if name == "test_module_imports":
                continue
            try:
                importlib.import_module("codegen.framework.tests.%s" % name)
            except Exception as error:  # noqa: BLE001 - report, do not mask
                failures.append("%s: %s: %s" % (name, type(error).__name__, error))
        self.assertEqual(
            failures,
            [],
            "test module(s) failed to import, so their tests never ran:\n  "
            + "\n  ".join(failures),
        )

    def test_discovers_the_expected_test_modules(self):
        """A module that disappears entirely is also a silent loss of coverage."""
        self.assertGreaterEqual(len(_test_module_names()), 10)
