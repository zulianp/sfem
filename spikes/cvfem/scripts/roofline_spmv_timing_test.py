#!/usr/bin/env python3
"""Check timing traces, retained workers and diagnostic refusal paths."""
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

import roofline_spmv_timing as timing
import roofline_process_observation as observation
from roofline_operator_test import refused
from roofline_reuse_diagnostic_test import MOCK


def output(kind, trace, resources=False):
    text = ''.join('ROOFLINE_CHECK stage={} field={} relative=0 PASS\n'.format(s, f)
                   for s in ('before', 'after') for f in range(4))
    if kind == 'sweep':
        text += 'PERF_PHASE phase=sweep controllers=2 transitions=4 PASS\n'
        text += 'PERF_CONTROL phase=sweep controllers=2 enable_seconds=0 disable_seconds=0 inner_seconds=0.6 outer_seconds=0.6\n'
    text += 'ROOFLINE_ACTION operation=spmv direction=coordinate_fixed reconstruction=none PASS\n'
    text += ('ROOFLINE_OPERATOR window={} layout=packed limiter=-1 elements={} nodes={} pack=1024 '
             'repeats=60 warmup=30 threads=72 seconds=0.6 precision=fp64 geometry=affine rc=1 '
             'state_gradients=cached boundary=0 transient=0 PASS\n').format(kind, 128**3, 129**3)
    if trace:
        for phase, count in (('warmup', 30), ('apply', 60)):
            text += ''.join('ROOFLINE_TIMING phase={} index={} seconds=0.01\n'.format(phase, i) for i in range(count))
            if resources:
                text += ''.join('ROOFLINE_RESOURCE phase={} index={} user_seconds=0.1 system_seconds=0.001 minor_faults=2 major_faults=0 voluntary_switches=3 involuntary_switches=1\n'.format(phase, i) for i in range(count))
    return text


def collection_tests(resources=False):
    topo = dict(cpus=list(range(72)), node=0, scf_cpu=0, pmu='nvidia_scf_pmu_0')
    with tempfile.TemporaryDirectory() as temp:
        for fault in (None, 'trace', 'numerical', 'coverage', 'identity', 'capture', 'shutdown'):
            out = Path(temp)/(fault or 'good')
            sessions = []
            def prepare(binary, path, elements, mode, events):
                assert mode == 'spmv-timing-diagnostic'
                path.mkdir()
                return topo, Path('/binary'), {}
            class Worker:
                def __init__(self, topo, binary, env, path, name, operation, scheme, layout):
                    assert (operation, scheme, layout) == ('spmv', 'fo', 'packed')
                    self.index, self.closed = 0, False
                    self.identity = dict(pid=1+len(sessions), buffers=['0x1', '0x2', '0x3'])
                    self.direction = dict(nodes=129**3, xor='1234567890abcdef', sum='abcdef1234567890')
                    sessions.append(self)
                def capture(self, topo, binary, stem, kind, repeat, trace=False, resources=False):
                    assert repeat == 60
                    if fault == 'capture':
                        raise subprocess.TimeoutExpired('mock', 1)
                    if fault == 'identity' and self.index == 1:
                        self.identity = dict(self.identity, pid=999)
                    self.index += 1
                    text = output(kind, trace, resources)
                    if fault == 'trace' and trace:
                        text = text.replace('index=59 ', 'index=58 ')
                    if fault == 'numerical':
                        text = text.replace('relative=0', 'relative=1')
                    stem.with_suffix('.out').write_text(text)
                    if kind == 'sweep':
                        scf, core = timing.counter_events(topo)
                        scf += [topo['pmu']+'/'+e+'/' for e in ('cmem_wb_data', 'cmem_wr_data')]
                        for ext, events, values, runtime in (('.scf.perf', scf, [60e8/32, 60e8, 0, 60e8], .6),
                                                             ('.core.perf', core, [0, 60e8], .6*72)):
                            stem.with_suffix(ext).write_text(''.join('{},,{}, {},{},,\n'.format(v, e, runtime*1e9,
                                '4.0' if fault == 'coverage' else '100.00') for e, v in zip(events, values)))
                    return text
                def close(self, abort=False):
                    self.closed = True
                    if fault == 'shutdown' and not abort:
                        raise ValueError('failed shutdown')
            def observe(pid):
                return dict(pid=pid, monotonic_start=1, monotonic_end=1, cgroups={})
            with patch.object(timing, 'prepare_capture', prepare), patch.object(timing.actions, 'ActionSession', Worker), patch.object(timing, 'snapshot', observe), contextlib.redirect_stdout(io.StringIO()) as log:
                if fault:
                    try:
                        timing.collect(Path('/binary'), out, resources)
                    except (ValueError, subprocess.SubprocessError):
                        pass
                    else:
                        raise AssertionError('invalid collection accepted: '+fault)
                    assert 'DIAGNOSTIC COLLECTED' not in log.getvalue()
                else:
                    timing.collect(Path('/binary'), out, resources)
                    records = json.loads((out/'records.json').read_text())
                    assert len(records) == 64 and len(sessions) == 8
                    assert all(s.index == 8 for s in sessions)
                    for bad in (records[:-1], list(reversed(records)), records+[records[0]]):
                        refused(lambda: timing.analyze(bad, resources))
                    for field, value in (('seconds', float('nan')), ('apply_times', [0]*60)):
                        bad = copy.deepcopy(records)
                        next(r for r in bad if r['variant'] == 'normal-trace')[field] = value
                        refused(lambda: timing.analyze(bad, resources))
                    if resources:
                        slow = copy.deepcopy(records)
                        row = next(r for r in slow if r['variant'] == 'normal-trace')
                        row['apply_times'][0] += .05
                        row['seconds'] += .05
                        row['observations']['before']['cgroups'] = {'/job': {'cpu.stat': dict(text='nr_throttled 2\n')}}
                        row['observations']['after']['cgroups'] = {'/job': {'cpu.stat': dict(text='nr_throttled 5\n')}}
                        report = timing.analyze(slow, True)
                        assert 'apply 0 60.000000 100.000000 1.000000 2 0 3 1' in report
                        assert 'path=/job nr_throttled delta=3' in report
                        assert 'no campaign acceptance' in report and len(slow) == 64
                        bad = copy.deepcopy(records)
                        next(r for r in bad if r['variant'] == 'normal-trace')['apply_resources'][0]['major_faults'] = -1
                        refused(lambda: timing.analyze(bad, True))
                        bad = copy.deepcopy(records)
                        bad[0]['observations']['before']['pid'] = -1
                        refused(lambda: timing.analyze(bad, True))
                    assert 'no campaign acceptance' in log.getvalue()
                assert all(s.closed for s in sessions) and (out/'files.sha256.json').exists()
    text = output('normal', True)
    refused(lambda: timing.parse_trace(text, .6, False))
    refused(lambda: timing.parse_trace(text, .7, True))
    refused(lambda: timing.parse_trace(text.replace('seconds=0.01', 'seconds=nan'), .6, True))
    text = output('normal', True, True)
    assert timing.parse_trace(text, .6, True, resources=True)['apply_resources'][0]['minor_faults'] == 2
    refused(lambda: timing.parse_trace(text, .6, True))
    refused(lambda: timing.parse_trace(text.replace('minor_faults=2', 'minor_faults=-1'), .6, True, resources=True))
    refused(lambda: timing.parse_trace(text.replace('ROOFLINE_RESOURCE', 'MISSING'), .6, True, resources=True))
    print('SpMV diagnostic: 64 captures, complete traces, failed checks/coverage/identity/capture/shutdown refusal PASS (synthetic)')


def binary_tests(binary, resources=False):
    topo = dict(cpus=[0, 1], node=0, scf_cpu=0, pmu='mock')
    with tempfile.TemporaryDirectory() as temp:
        out = Path(temp)
        mock = out/'mock.py'
        mock.write_text(MOCK)
        for threads in (1, 2):
            env = dict(os.environ, OMP_NUM_THREADS=str(threads), OMP_DYNAMIC='FALSE',
                       OMPI_MCA_btl='self', OMPI_MCA_pml='ob1')
            worker = timing.actions.ActionSession(topo, binary, env, out, 't{}-reuse'.format(threads),
                                                  'spmv', 'fo', 'packed', n=4, threads=threads, launcher=False)
            try:
                for index, variant in enumerate(timing.VARIANTS + timing.VARIANTS[::-1]):
                    kind = 'sweep' if variant.startswith('counted') else 'normal'
                    traced = variant.endswith('-trace')
                    text = worker.capture(topo, binary, out/'t{}-i{}'.format(threads, index), kind, 60,
                        trace=traced, resources=resources and traced,
                        monitor_prefix=[sys.executable, str(mock), 'good'])
                    row = timing.actions.parse_output(text, 'spmv', 'fo', 'packed', kind, 60, n=4, threads=threads)
                    timing.parse_trace(text, row['seconds'], traced, resources=resources and traced)
                worker.close()
            finally:
                worker.close(abort=True)
        for index, request in enumerate(('normal 60 junk\n', 'normal 60 trace junk\n', 'normal 0 trace\n', 'normal 60 resources junk\n')):
            worker = timing.actions.ActionSession(topo, binary, env, out, 'bad{}-reuse'.format(index),
                                                  'spmv', 'fo', 'packed', n=4, threads=2, launcher=False)
            try:
                refused(lambda: worker.exchange(out/'bad{}'.format(index), request, None))
                assert worker.process.poll() is not None
            finally:
                worker.close(abort=True)
    print('SpMV diagnostic: 16 real plain/traced normal/counted windows at 1/2 threads and invalid requests PASS (no PMU)')


def observation_tests():
    mounts = '31 24 0:28 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n'
    assert observation.cgroup_paths('0::/job/step\n', mounts) == ['/sys/fs/cgroup', '/sys/fs/cgroup/job', '/sys/fs/cgroup/job/step']
    mounts = '31 24 0:28 /job /sys/fs/cgroup/cpu rw - cgroup cgroup rw,cpu,cpuacct\n'
    assert observation.cgroup_paths('4:cpu,cpuacct:/job/step\n', mounts) == ['/sys/fs/cgroup/cpu', '/sys/fs/cgroup/cpu/step']
    assert observation.cgroup_paths('4:cpu,cpuacct:/elsewhere\n', mounts) == []
    assert observation.cgroup_paths('4:cpu,cpuacct:/job/../../secret\n', mounts) == []
    assert observation.cgroup_paths('broken', 'broken') == []
    snap = observation.snapshot(123, read=lambda path: dict(error='unavailable'))
    assert snap['pid'] == 123 and snap['cgroups'] == {} and observation.throttle_changes(snap, snap) == []
    before = dict(cgroups={'/job': {'cpu.stat': dict(text='nr_throttled 2\nthrottled_usec 40\n')}})
    after = dict(cgroups={'/job': {'cpu.stat': dict(text='nr_throttled 5\nthrottled_usec 100\n')}})
    assert [r['delta'] for r in observation.throttle_changes(before, after)] == [3, 60]
    assert observation.throttle_changes(before, dict(cgroups={})) == []
    assert observation.read_optional('/does/not/exist')['error'] == 'FileNotFoundError'
    print('Process observations: v1/v2 CPU ancestry, mount roots, malformed paths, missing files and throttle deltas PASS')


if __name__ == '__main__':
    collection_tests()
    collection_tests(True)
    observation_tests()
    if len(sys.argv) > 1:
        binary_tests(Path(sys.argv[1]).resolve())
        binary_tests(Path(sys.argv[1]).resolve(), True)
