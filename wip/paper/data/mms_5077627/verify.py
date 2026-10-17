#!/usr/bin/env python3
"""Replay the retained MMS study without running the solver."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def hashes(path):
    result = {}
    for line in path.read_text().splitlines():
        digest, name = line.split('  ', 1)
        require(re.fullmatch('[0-9a-f]{64}', digest), 'invalid hash')
        require(name not in result, 'duplicate hash entry')
        result[name] = digest
    return result


def check(root=ROOT):
    archived = hashes(root/'RAW-SHA256.txt')
    actual = {str(p.relative_to(root)) for folder in ('raw', 'sources')
              for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    require(len(archived) == 32 and set(archived) == actual, 'archive coverage changed')
    for name, digest in archived.items():
        require(Path(name).parts[0] in ('raw', 'sources') and '..' not in Path(name).parts,
                'invalid archive path')
        require(hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, 'hash mismatch: ' + name)

    source = hashes(root/'raw/SOURCE-SHA256.txt')
    checks = (root/'raw/source-check.log').read_text().splitlines()
    require(len(checks) == len(source) and set(checks) == {name + ': OK' for name in source},
            'source manifest check incomplete')
    for local, remote in (
            ('raw/run-mms-on-grace.sh', 'run-mms-on-grace.sh'),
            ('sources/verify_report.sh', 'sfem/spikes/cvfem/scripts/verify_report.sh'),
            ('sources/original_summary.py', 'sfem/spikes/cvfem/scripts/mms_refinement_summary.py')):
        require(archived[local] == source[remote], 'run source identity mismatch')
    settings = dict(line.split('=', 1) for line in (root/'raw/settings.txt').read_text().splitlines())
    expected = {'CONV_ARMS': 'lo ho_lim0 ho_lim1 ho_lim2_k0 ho_lim3', 'CONV_LADDER': '16 24 32',
                'OMP_NUM_THREADS': '8', 'OMP_DYNAMIC': 'FALSE', 'OMP_PLACES': 'cores',
                'OMP_PROC_BIND': 'close', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                'VERIFY_GROUPS': 'conv', 'VERIFY_LEVEL': '1', 'VERIFY_RESTART': '480',
                'VERIFY_SOLVER': 'fgmres', 'SFEM_NL_RTOL': '1e-10', 'SFEM_LSOLVE_RTOL': '1e-6',
                'SFEM_RC_EXACT_JAC': '1', 'SFEM_RC_TAU': '1', 'SFEM_RC_TAU_TRANSIENT': '0',
                'SFEM_VERIFY_TOL': '0.01', 'SFEM_PACK_SIZE': '1024', 'SFEM_GEOM': 'affine',
                'SFEM_LX': '2', 'SFEM_LY': '2', 'SFEM_LZ': '2', 'SFEM_U': '1',
                'NL_MAX_IT': '40', 'VERIFY_RUN_TIMEOUT': '300'}
    require(all(settings.get(k) == v for k, v in expected.items()), 'changed run settings')
    binary = hashes(root/'raw/BUILD-SHA256.txt')
    require(list(binary) == ['build/cvfem/cvfem_hex8_ns_ssgmg'], 'unexpected binary identity')
    manifest = json.loads((root/'raw/study/manifest.json').read_text())
    require(manifest['machine'] == 'nid006551' and manifest['threads'] == '8' and
            manifest['level'] == '1' and manifest['solver'] == 'fgmres, restart 480',
            'wrong machine or solver metadata')
    spec = importlib.util.spec_from_file_location('retained_mms_reader', root/'sources/mms_refinement_summary.py')
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    reported = json.loads((root/'raw/study/summary.json').read_text())
    require(len(reported) == len(manifest['runs']) == 15, 'wrong result count')
    expected_cases = {(arm, n) for arm in ('lo', 'ho_lim0', 'ho_lim1', 'ho_lim2_k0', 'ho_lim3')
                      for n in (16, 24, 32)}
    seen, rows = set(), []
    tsv = []
    for line in (root/'raw/study/runs.tsv').read_text().splitlines():
        group, label, log, extra = line.split('\t')
        arm, n = label.rsplit('_n', 1)
        n = int(n)
        require(group == 'conv' and (arm, n) in expected_cases and (arm, n) not in seen,
                'unexpected or duplicate run')
        require(log == 'conv_' + label + '.log' and extra == 'conv_arm=' + arm + ',exit_code=1',
                'wrong log or exit metadata')
        tsv.append(dict(group=group, label=label, log=log, conv_arm=arm, exit_code=1.0))
        seen.add((arm, n))
        row = reader.read_case(root/'raw/study'/log, n, 1)
        row['scheme'] = arm
        rows.append(row)
    require(seen == expected_cases and tsv == manifest['runs'], 'manifest/run coverage mismatch')
    for row, original in zip(rows, reported):
        for key, value in original.items():
            if key != 'adjacent_velocity_l2_rate':
                require(row[key] == value, 'raw summary differs from replay')
    rates = {}
    import math
    for arm in sorted({r['scheme'] for r in rows}):
        ladder = sorted((r for r in rows if r['scheme'] == arm), key=lambda r: r['n'])
        rates[arm] = [math.log(a['u_l2']/b['u_l2']) / math.log(a['h']/b['h'])
                      for a, b in zip(ladder, ladder[1:])]
        saved = [r['adjacent_velocity_l2_rate'] for r in reported if r['scheme'] == arm]
        require(saved == [None] + rates[arm], 'rate replay differs')
    return dict(cases=len(rows), configured_threads=8, source_checks=len(source),
                maximum_final_relative_residual=max(r['nonlinear_relative'] for r in rows),
                absolute_accuracy_passes=sum(r['absolute_accuracy_pass'] for r in rows),
                intermediate_floor_cases=[r['scheme']+'_n'+str(r['n']) for r in rows if r['intermediate_floor']],
                velocity_l2_rates=rates, recorded_binary_sha256=next(iter(binary.values())))


if __name__ == '__main__':
    result = check()
    require(result == json.loads((ROOT/'verification.json').read_text()), 'stored verification differs')
    print(json.dumps(result, indent=2, sort_keys=True))
    print('MMS archive verified; all 15 absolute-accuracy failures retained (no new solve).')
