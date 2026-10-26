"""Read the external, byte-pinned mesh and refinement evidence."""
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import tempfile
import zipfile

SHA256 = '44baa5b5943bd4a72f750d3aa390d2aba0bd4a9671bd7b86efa4de43cb0f4ec3'

@contextmanager
def evidence():
    archive = Path(os.environ.get('SFEM_VALIDATION_ARCHIVE', 'pasc27-mesh-validation.zip'))
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise ValueError('mesh/refinement archive hash differs')
    with tempfile.TemporaryDirectory() as temporary, zipfile.ZipFile(archive) as zipped:
        names = zipped.namelist()
        if len(names) != 117 or len(set(names)) != 117:
            raise ValueError('incomplete or duplicate archive members')
        root = Path(temporary)
        for name in names:
            path = Path(name)
            if path.is_absolute() or '..' in path.parts or path.parts[0] not in (
                    'unstructured_5052301', 'mms_5077627', 'mms_5077906'):
                raise ValueError('invalid archive path')
            target = root/path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zipped.read(name))
        yield root
