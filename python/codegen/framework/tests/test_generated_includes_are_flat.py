"""The generated tree survives being installed flat.

`make install` copies every header into one `include/` directory, so an
`#include` between generated files must be a bare file name -- a directory in
it names a path that does not exist once installed -- and no two generated
files may share a name, or one overwrites the other.  The build tree resolves
the bare names because CMake puts every generated folder holding a header on
the include path.
"""

import os
import re
import unittest


def _generated_tree():
    return os.path.normpath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "..", "..", "frontend", "ops", "generated",
        )
    )


_EXTENSIONS = (".hpp", ".cuh", ".cpp", ".cu")
_QUOTED_INCLUDE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.M)


def _generated_files(root):
    for directory, _, names in os.walk(root):
        for name in names:
            if name.endswith(_EXTENSIONS):
                yield os.path.join(directory, name)


class GeneratedIncludesAreFlat(unittest.TestCase):
    def setUp(self):
        self.root = _generated_tree()
        self.files = sorted(_generated_files(self.root))
        self.assertGreater(len(self.files), 0, "no generated files found")
        self.names = {}
        for path in self.files:
            self.names.setdefault(os.path.basename(path), []).append(path)

    def test_includes_between_generated_files_are_bare_names(self):
        offenders = []
        for path in self.files:
            with open(path, encoding="utf-8") as handle:
                includes = _QUOTED_INCLUDE.findall(handle.read())
            for include in includes:
                if os.path.basename(include) in self.names and include != os.path.basename(include):
                    offenders.append("%s: %s" % (os.path.relpath(path, self.root), include))
        self.assertEqual(offenders, [], "\n".join(offenders))

    def test_generated_headers_have_unique_names(self):
        clashes = [
            "%s: %s" % (name, ", ".join(os.path.relpath(path, self.root) for path in paths))
            for name, paths in sorted(self.names.items())
            if len(paths) > 1 and name.endswith((".hpp", ".cuh"))
        ]
        self.assertEqual(clashes, [], "\n".join(clashes))


if __name__ == "__main__":
    unittest.main()
