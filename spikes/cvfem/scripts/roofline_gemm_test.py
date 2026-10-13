#!/usr/bin/env python3
"""Check DGEMM collection, refusals and optional real BLAS sessions without a PMU."""
import argparse
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

import roofline_gemm as gemm
from roofline_reuse_diagnostic_test import MOCK


def refused(call):
    try:
        call()
    except (ValueError, KeyError, OSError, subprocess.SubprocessError):
        return
    raise AssertionError('invalid DGEMM evidence accepted')


def output(kind, repeat, n=8192, threads=72, seconds=None):
    seconds = seconds if seconds is not None else .5*repeat
    result = ''.join('GEMM_CHECK stage={} finite={} samples=21 relative=0 tolerance=1e-12 PASS\n'.format(stage, n*n)
                     for stage in ('before', 'after'))
    if kind != 'normal':
        selected = gemm.phase(kind)
        result += 'PERF_PHASE phase={} controllers=2 transitions=4 PASS\n'.format(selected)
        result += 'PERF_CONTROL phase={} controllers=2 enable_seconds=0 disable_seconds=0 inner_seconds={} outer_seconds={}\n'.format(selected, seconds, seconds)
    return result+'ROOFLINE_GEMM window={} n={} repeats={} warmup=3 threads={} seconds={} useful_flops={} precision=fp64 layout=column_major transa=N transb=N alpha=1 beta=0 PASS\n'.format(gemm.phase(kind), n, repeat, threads, seconds, 2*n**3)


def synthetic_records():
    rows = []
    for group in gemm.groups():
        for index, (block, kind, repeat) in enumerate(group):
            row = dict(block=block, kind=kind, repeat=repeat, seconds=.5*repeat,
                       useful_flops=2*8192**3, session='gemm-b{}-reuse'.format(block), session_index=index,
                       identity=dict(pid=100+block, buffers=['0x1', '0x2', '0x3']),
                       library=dict(backend='openblas', actual='72', library='/mock/libblas.so', config='OpenBLAS mock'))
            if kind != 'normal':
                factor = repeat if kind == 'sweep' else .0001
                row.update(read_bytes=1e9*factor, write_bytes=5e8*factor, component_write_bytes=5e8*factor,
                           fp_ops=2*8192**3*factor, scf_seconds=.5*repeat, core_seconds=.5*repeat)
            rows.append(row)
    return rows


def accounting_tests():
    rows = synthetic_records()
    assert len(rows) == 36 and gemm.analyze(rows)[1]
    for changed in (rows[:-1], rows+[rows[0]], list(reversed(rows))):
        refused(lambda: gemm.analyze(changed))
    for field, value in (('seconds', float('nan')), ('useful_flops', 1), ('library', {}), ('identity', {})):
        changed = copy.deepcopy(rows)
        changed[1][field] = value
        refused(lambda: gemm.analyze(changed))
    for kind, field, value in (('sweep', 'scf_seconds', 99), ('sweep', 'fp_ops', 1),
                               ('sweep', 'component_write_bytes', 0), ('empty-before', 'fp_ops', 1e20),
                               ('normal', 'seconds', .001)):
        changed = copy.deepcopy(rows)
        next(r for r in changed if r['kind'] == kind)[field] = value
        assert not gemm.analyze(changed)[1], (kind, field)
    for kind in ('normal', 'sweep', 'empty-before'):
        valid = output(kind, 1)
        assert gemm.parse_output(valid, kind, 1)['useful_flops'] == 2*8192**3
        for old, new in (('stage=before', 'stage=after'), ('threads=72', 'threads=71'), ('finite=67108864', 'finite=1'),
                         ('relative=0', 'relative=nan'), ('relative=0', 'relative=1e-5'), ('warmup=3', 'warmup=30'),
                         ('useful_flops=1099511627776', 'useful_flops=1'), ('GEMM_CHECK', 'OMITTED_CHECK')):
            refused(lambda: gemm.parse_output(valid.replace(old, new), kind, 1))
    identity = 'GEMM_IDENTITY backend=openblas requested=72 actual=72 thread_scope=runtime\nGEMM_CONFIG OpenBLAS 0.3 mock\nGEMM_LIBRARY /mock/libblas.so\n'
    assert gemm.library_identity(identity, 72, Path('/mock/libblas.so'), True)['backend'] == 'openblas'
    for changed in (identity.replace('actual=72', 'actual=1'), identity.replace('GEMM_CONFIG', 'MISSING'),
                    identity.replace('OpenBLAS 0.3 mock', 'OpenBLAS USE64BITINT'), identity.replace('GEMM_LIBRARY', 'MISSING')):
        refused(lambda: gemm.library_identity(changed, 72, production=True))
    refused(lambda: gemm.library_identity(identity, 72, Path('/other/library'), True))
    try:
        gemm.library_identity(identity.replace('/mock/libblas.so', '/mock/gemm'),
                              72, Path('/mock/libblas.so'), True)
    except ValueError as error:
        assert '/mock/gemm' in str(error) and '/mock/libblas.so' in str(error)
    else:
        raise AssertionError('executable mistaken for the BLAS library')
    print('DGEMM accounting, scope, library identity and failed screens PASS (synthetic)')


def collection_tests():
    topo = dict(cpus=list(range(72)), node=0, scf_cpu=0, pmu='nvidia_scf_pmu_0')
    fixture = synthetic_records()
    with tempfile.TemporaryDirectory() as temp:
        library = Path(temp)/'library.so'; library.write_text('test library')
        for fault in (None, 'coverage', 'check', 'identity', 'shutdown', 'timeout', 'stability', 'topology'):
            out = Path(temp)/(fault or 'good')
            sessions, seen = [], []
            def prepare(binary, path, elements, mode, events):
                assert elements == 8192 and mode == 'gemm-reuse'
                path.mkdir(); (path/'environment.json').write_text('{}')
                return topo, Path('/binary'), {}
            class FakeSession:
                def __init__(self, topology, binary, env, path, name, library=None, production=False):
                    assert all(s.closed for s in sessions) and production and library.is_file()
                    self.name, self.index, self.closed = name, 0, False
                    self.identity = dict(pid=100+len(sessions), buffers=['0x1', '0x2', '0x3'])
                    self.library_identity = fixture[0]['library']
                    sessions.append(self)
                def capture(self, topology, binary, stem, kind, repeat):
                    if fault == 'timeout': raise subprocess.TimeoutExpired('mock', 1)
                    row = dict(fixture[len(seen)]); seen.append(row)
                    assert row['session'] == self.name and row['session_index'] == self.index
                    if fault == 'identity' and self.index == 1: self.identity = dict(self.identity, pid=9999)
                    if fault == 'stability' and kind == 'sweep' and repeat == 2 and row['block'] == 0: row['write_bytes'] *= 1.3
                    result = output(kind, repeat)
                    if fault == 'check': result = result.replace('GEMM_CHECK', 'MISSING')
                    stem.with_suffix('.out').write_text(result)
                    if kind != 'normal':
                        scf, core = gemm.counter_events(topo)
                        scf += [topo['pmu']+'/'+e+'/' for e in ('cmem_wb_data', 'cmem_wr_data')]
                        for suffix, events, values, seconds in (
                                ('.scf.perf', scf, [row['read_bytes']/32, row['write_bytes'], 0, row['write_bytes']], row['scf_seconds']),
                                ('.core.perf', core, [0, row['fp_ops']], row['core_seconds']*72)):
                            stem.with_suffix(suffix).write_text(''.join('{:.0f},,{}, {:.0f},{},,\n'.format(v, event, seconds*1e9, '4' if fault == 'coverage' else '100.00') for event, v in zip(events, values)))
                    self.index += 1
                    return result
                def close(self, abort=False):
                    self.closed = True
                    if fault == 'shutdown' and not abort: raise ValueError('mock shutdown failure')
            reference_topo = {} if fault == 'topology' else topo
            with patch.object(gemm, 'prepare_capture', prepare), patch.object(gemm, 'GemmSession', FakeSession), patch.object(gemm, 'calibration_reference', return_value=(reference_topo, 'control fixture\n')), contextlib.redirect_stdout(io.StringIO()) as log:
                if fault:
                    refused(lambda: gemm.collect(Path('/binary'), out, Path('/control'), library))
                    assert 'CAMPAIGN PASSED' not in log.getvalue()
                else:
                    gemm.collect(Path('/binary'), out, Path('/control'), library)
                    assert len(json.loads((out/'records.json').read_text())) == 36
                    assert len(sessions) == 6 and all(s.index == 6 for s in sessions)
                assert (out/'files.sha256.json').exists() and all(s.closed for s in sessions)
    print('DGEMM collection: 36 captures, six retained workers, all failure cleanup screens PASS (synthetic)')


def discovery_tests():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        one = root/'libopenblas-a.so'; one.write_text('LP64')
        alias = root/'libopenblas-a.so.0'; alias.symlink_to(one)
        def probe(path):
            if path.read_text() != 'LP64': raise ValueError('not LP64')
            return dict(config='OpenBLAS mock', threads=72)
        assert gemm.discover_library(root, probe=probe) == one.resolve()
        two = root/'libopenblas-b.so'; two.write_text('LP64')
        refused(lambda: gemm.discover_library(root, probe=probe))
        assert gemm.discover_library(root, override=str(two), probe=probe) == two.resolve()
        two.write_text('ILP64')
        assert gemm.discover_library(root, probe=probe) == one.resolve()
        refused(lambda: gemm.discover_library(root, override=str(two), probe=probe))
        refused(lambda: gemm.discover_library(root/'absent', probe=probe))
        refused(lambda: gemm.discover_library(root, override=str(root/'missing.so'), probe=probe))
        refused(lambda: gemm.discover_library(root, probe=lambda path: (_ for _ in ()).throw(AttributeError('missing API'))))
    session = gemm.GemmSession.__new__(gemm.GemmSession)
    session.fds = list(range(3, 11))
    session.exchange = lambda stem, request, command, timeout: (request, command)
    topo = dict(cpus=list(range(72)), node=0, scf_cpu=0, pmu='mock')
    request, command = session.capture(topo, Path('/binary'), Path('/capture'), 'sweep', 2)
    assert request == 'sweep 2\n' and command.count('--delay=-1') == 2
    assert [command[i+1] for i, part in enumerate(command) if part == '-C'] == ['0', ','.join(map(str, range(72)))]
    assert session.capture(topo, Path('/binary'), Path('/capture'), 'normal', 2) == ('normal 2\n', None)
    print('Library discovery: explicit override, deduplicated aliases, ambiguity/missing/API/LP64 refusal and monitor scopes PASS')


def binary_tests(binary):
    for threads in (1, 2):
        env = dict(os.environ, VECLIB_MAXIMUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads), OMP_NUM_THREADS=str(threads))
        for n in (64, 65, 127, 257):
            result = subprocess.run([str(binary), 'normal', str(n), '2', str(threads)], env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, timeout=30)
            assert result.returncode == 0, result.stdout
            identity = gemm.library_identity(result.stdout, threads)
            assert Path(identity['library']).resolve() != binary.resolve(), identity
            gemm.parse_output(result.stdout, 'normal', 2, n, threads)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp); mock = out/'monitor.py'; mock.write_text(MOCK)
            topo = dict(cpus=list(range(threads)), node=0, scf_cpu=0, pmu='mock')
            for fault in (None, 'bad', 'timeout'):
                session = gemm.GemmSession(topo, binary, env, out, 'gemm-'+str(fault)+'-reuse', n=128,
                                           threads=threads, launcher=False)
                try:
                    assert Path(session.library_identity['library']).resolve() != binary.resolve()
                    for index, (kind, repeat) in enumerate((('normal', 2), ('empty-before', 1), ('sweep', 3), ('sweep', 1), ('sweep', 2), ('empty-after', 1))):
                        if fault and kind != 'normal':
                            refused(lambda: session.capture(topo, binary, out/(str(fault)+str(index)), kind, repeat,
                                    monitor_prefix=[sys.executable, str(mock), 'bad' if fault == 'bad' else 'silent'], timeout=.5 if fault == 'timeout' else 8))
                            break
                        result = session.capture(topo, binary, out/(str(fault)+str(index)), kind, repeat,
                                                 monitor_prefix=[sys.executable, str(mock), 'good'])
                        gemm.parse_output(result, kind, repeat, n=128, threads=threads)
                    if fault is None: session.close()
                finally:
                    session.close(abort=True)
    env = dict(os.environ, VECLIB_MAXIMUM_THREADS='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    for command in (('preflight', '8', '2'), ('normal', '0', '1', '1'), ('normal', '8193', '1', '1'),
                    ('normal', '8', '0', '1'), ('normal', '8', '4', '1')):
        result = subprocess.run([str(binary)]+list(command), env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, universal_newlines=True, timeout=30)
        assert result.returncode != 0 and 'ROOFLINE_GEMM' not in result.stdout, result.stdout
    print('Actual BLAS: eight checked sizes/teams, retained sessions, bad/missing acknowledgements and scope refusal PASS (no PMU)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path)
    args = parser.parse_args()
    accounting_tests()
    collection_tests()
    discovery_tests()
    if args.binary:
        binary_tests(args.binary.resolve())
