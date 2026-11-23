"""Nonfinite and malformed dumps must fail even when both files would be identical."""
import math
from pathlib import Path
import struct
import subprocess
import tempfile


def main():
    checker = Path(__file__).with_name('cvfem_check_dump.sh')
    with tempfile.TemporaryDirectory() as work:
        path = Path(work) / 'values.bin'
        good = struct.pack('=4d', 0.0, -1.5, 2.0, 3.25)
        cases = [('finite', good, True), ('empty', b'', False),
                 ('truncated', good[:-1], False),
                 ('missing_value', good[:-8], False),
                 ('extra_value', good + good[:8], False)]
        for value in (float('nan'), float('inf'), -float('inf')):
            for index in (0, 3):
                bad = bytearray(good)
                struct.pack_into('=d', bad, 8 * index, value)
                cases.append(('nonfinite', bad, False))
        for name, payload, accepted in cases:
            path.write_bytes(payload)
            result = subprocess.run(['bash', str(checker), str(path), '4'],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if (result.returncode == 0) != accepted:
                raise AssertionError('wrong acceptance for ' + name)
            if accepted:
                values = [float(x) for x in result.stdout.split()]
                if values != [0.0, -1.5, 2.0, 3.25] or not all(map(math.isfinite, values)):
                    raise AssertionError('changed finite values')
    print('dump validation: finite values, NaN/Inf, truncation and size controls PASS')


if __name__ == '__main__':
    main()
