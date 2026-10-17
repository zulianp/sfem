#!/usr/bin/env python3
"""Check the retained boundary/transient result without rerunning Grace."""
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent


def check(root=ROOT):
    manifest = (root / 'RAW-SHA256.txt').read_text().splitlines()
    names = []
    for line in manifest:
        digest, name = line.split('  ', 1)
        path = Path(name)
        assert path.parts[0] == 'raw' and '..' not in path.parts, name
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest, name
        names.append(name)
    actual = {str(p.relative_to(root)) for p in (root / 'raw').iterdir() if p.is_file()}
    assert len(names) == len(set(names)) == 12 and set(names) == actual

    for name, count in [('solver-terms-T1.log', 5), ('solver-terms-T72.log', 5),
                        ('fixed-input.log', 1), ('derivatives-and-cache.log', 5)]:
        log = (root / 'raw' / name).read_text()
        assert log.count('Test Passed.') == count and 'Test Failed.' not in log, name

    log = (root / 'raw/derivatives-and-cache.log').read_text()
    log = log.split('Test: cvfem_jacobian_composed\n', 1)[1].split('Test Passed.', 1)[0]
    blocks = re.split(r'^threads=(\d+) scheme=(first|[0-3]) mode=([a-z0-9-]+) rc_time=([01])$',
                      log, flags=re.M)
    expected = {(t, s, mode, rc) for t in (1, 72) for s in ('first', '0', '1', '2', '3')
                for rc in (0, 1) for mode in ('boundary', 'bdf1', 'bdf2', 'boundary-bdf1', 'boundary-bdf2')
                if not (rc == 1 and mode == 'boundary')}
    seen, errors, fields = set(), [], []
    controls = {k: [] for k in ('no_qgrad', 'no_coeff_du', 'no_boundary', 'no_transient')}
    for i in range(1, len(blocks), 5):
        t, s, mode, rc, text = blocks[i:i + 5]
        key = (int(t), s, mode, int(rc))
        assert key in expected and key not in seen, key
        seen.add(key)
        boundary = int(mode.startswith('boundary'))
        order = int(mode[-1]) if 'bdf' in mode else 0
        inv_dt = (10 if order == 1 else 15) if int(rc) and order else 0
        scope = 'boundary {} bdf {} dt {} rc_inv_dt_a0 {} threads {}'.format(
            boundary, order, '0.1' if order else '0', inv_dt, t)
        assert 'verify_jac_coupled_scope: ' + scope in text, key
        assert text.count('verify_jac_coupled: PASSED') == 1, key
        rows = [line.split() for line in text.splitlines()
                if line.startswith('verify_jac_coupled: ho ') and ' eps ' in line]
        assert len(rows) == 4, key
        steps = set()
        for row in rows:
            if row[row.index('required') + 1] != '1':
                continue
            eps = float(row[row.index('eps') + 1])
            assert eps not in steps
            steps.add(eps)
            assert int(row[row.index('ho') + 1]) == int(s != 'first')
            assert int(row[row.index('limiter') + 1]) == (0 if s == 'first' else int(s))
            assert int(row[row.index('boundary') + 1]) == boundary
            assert int(row[row.index('bdf') + 1]) == order
            error = float(row[row.index('rel') + 1])
            field = list(map(float, row[row.index('fields') + 1:row.index('fields') + 5]))
            assert 0 <= error <= 1e-6 and all(0 <= v <= 1e-6 for v in field), key
            errors.append(error)
            fields.extend(field)
            for name in controls:
                active = name in ('no_qgrad', 'no_coeff_du') or (
                    name == 'no_boundary' and boundary) or (name == 'no_transient' and order)
                if active:
                    value = float(row[row.index(name) + 1])
                    assert 1e-5 < value < float('inf'), (key, name)
                    controls[name].append(value)
        assert steps == {1e-6, 1e-7}, key
    assert seen == expected and len(errors) == 180
    return dict(cases=len(seen), required_steps=len(errors), ctest_invocations=16,
                max_relative_error=max(errors), max_field_relative_error=max(fields),
                minimum_omitted_term_errors={k: min(v) for k, v in controls.items()})


if __name__ == '__main__':
    result = check()
    assert result == json.loads((ROOT / 'verification.json').read_text())
    print(json.dumps(result, indent=2, sort_keys=True))
    print('Retained composed checks: PASS (no new execution)')
