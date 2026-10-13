#!/usr/bin/env python3
"""Check THP intervention, complete paired collection and refusal paths."""
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
import roofline_thp_exec as launch
import roofline_thp_study as thp
from roofline_operator_test import refused
from roofline_reuse_diagnostic_test import MOCK
from roofline_spmv_timing_test import output


def launch_tests():
    for policy in thp.POLICIES:
        calls = []
        state = [0]
        def call(option, value, a, b, c):
            assert (a, b, c) == (0, 0, 0)
            calls.append((option, value))
            if option == 41:
                state[0] = value
                return 0
            return state[0]
        assert launch.configure(policy, call) == int(policy == 'disabled')
        assert (41, 1) in calls if policy == 'disabled' else all(c[0] == 42 for c in calls)
    refused(lambda: launch.configure('unknown', call))
    refused(lambda: launch.configure('default', lambda *args: 1))
    refused(lambda: launch.configure('disabled', lambda *args: 0))
    try:
        launch.configure('disabled', lambda *args: -1)
    except OSError:
        pass
    else:
        raise AssertionError('failed syscall accepted')
    for args in ([], ['disabled'], ['default', '--', 'relative']):
        refused(lambda: launch.main(args))
    with patch.object(launch, 'configure', return_value=1), patch.object(launch.os, 'execv') as execute, contextlib.redirect_stdout(io.StringIO()):
        launch.main(['disabled', '--', '/bench', '--perf-fds', '3', '4'])
        execute.assert_called_once_with('/bench', ['/bench', '--perf-fds', '3', '4'])
    assert thp.observed_count(dict(text='AnonHugePages: 42 kB\n'), 'AnonHugePages:') == 42
    assert thp.observed_count(dict(error='unavailable'), 'AnonHugePages:') is None
    assert thp.observed_count(dict(text='thp_collapse_alloc bad\n'), 'thp_collapse_alloc') is None
    print('THP launch: scoped control, readback, failed calls and exact exec arguments PASS (mock syscalls)')


def collection_tests():
    topo = dict(cpus=list(range(72)), node=0, scf_cpu=0, pmu='mock')
    with tempfile.TemporaryDirectory() as temp:
        for fault in (None, 'policy', 'system', 'numerical', 'capture', 'shutdown'):
            out = Path(temp)/(fault or 'good')
            sessions = []
            def prepare(binary, path, *args):
                path.mkdir()
                return topo, Path('/binary'), {}
            class Worker:
                def __init__(self, topo, binary, env, path, name, operation, scheme, layout, policy):
                    self.policy, self.closed = policy, False
                    self.identity = dict(pid=len(sessions)+1, buffers=['0x1', '0x2', '0x3'])
                    self.direction = dict(nodes=129**3, xor='1234567890abcdef', sum='abcdef1234567890')
                    sessions.append(self)
                def capture(self, topo, binary, stem, kind, repeat, trace, resources):
                    assert repeat == 60 and resources == trace
                    if fault == 'capture':
                        raise subprocess.TimeoutExpired('mock', 1)
                    text = output(kind, trace, resources)
                    if fault == 'numerical':
                        text = text.replace('relative=0', 'relative=1')
                    stem.with_suffix('.out').write_text(text)
                    if kind == 'sweep':
                        scf, core = timing.counter_events(topo)
                        scf += [topo['pmu']+'/'+e+'/' for e in ('cmem_wb_data', 'cmem_wr_data')]
                        for ext, events, values, runtime in (('.scf.perf', scf, [60e8/32, 60e8, 0, 60e8], .6),
                                                             ('.core.perf', core, [0, 60e8], .6*72)):
                            stem.with_suffix(ext).write_text(''.join('{},,{}, {},100.00,,\n'.format(v, e, runtime*1e9)
                                for e, v in zip(events, values)))
                    return text
                def close(self, abort=False):
                    self.closed = True
                    if fault == 'shutdown' and not abort:
                        raise ValueError('failed shutdown')
            def observe(pid):
                enabled = int(sessions[pid-1].policy == 'default')
                if fault == 'policy':
                    enabled = 1-enabled
                return dict(pid=pid, monotonic_start=1, monotonic_end=1, cgroups={},
                    process=dict(status=dict(text='THP_enabled:\t{}\n'.format(enabled))),
                    system={'/sys/kernel/mm/transparent_hugepage/'+k:dict(text=v) for k,v in (
                        ('enabled','always [madvise] never' if fault == 'system' else '[always] madvise never'),
                        ('defrag','always [madvise] never'))})
            with patch.object(timing, 'prepare_capture', prepare), patch.object(thp, 'ThpSession', Worker), patch.object(timing, 'snapshot', observe), contextlib.redirect_stdout(io.StringIO()) as log:
                if fault:
                    try:
                        timing.collect(Path('/binary'), out, thp_study=True)
                    except (ValueError, subprocess.SubprocessError):
                        pass
                    else:
                        raise AssertionError('invalid THP study accepted: '+fault)
                    assert 'DIAGNOSTIC COLLECTED' not in log.getvalue()
                else:
                    timing.collect(Path('/binary'), out, thp_study=True)
                    rows = json.loads((out/'records.json').read_text())
                    assert len(rows) == 128 and len(sessions) == 16
                    for bad in (rows[:-1], list(reversed(rows)), rows+rows[:1]):
                        refused(lambda: thp.analyze(bad, timing.groups(), timing.analyze))
                    for field, value in (('direction', {}), ('identity', rows[0]['identity'])):
                        bad = copy.deepcopy(rows)
                        bad[8][field] = value
                        refused(lambda: thp.analyze(bad, timing.groups(), timing.analyze))
                    assert 'no performance acceptance' in log.getvalue()
                    assert json.loads((out/'collection.json').read_text())['thp_study']
                assert all(s.closed for s in sessions) and (out/'files.sha256.json').exists()
    print('THP paired collection: 128 captures, policy/order/direction/control/result/failure refusal PASS (synthetic)')


def binary_tests(binary):
    topo = dict(cpus=[0, 1], node=0, scf_cpu=0, pmu='mock')
    with tempfile.TemporaryDirectory() as temp:
        out = Path(temp)
        mock = out/'monitor.py'
        mock.write_text(MOCK)
        original = thp.ThpSession.worker_command
        def command(worker, *args):
            result = original(worker, *args)
            if sys.platform != 'linux':
                result[1] = str(out/'mock-policy.py')
            return result
        (out/'mock-policy.py').write_text('import sys\nsys.path.insert(0, '+repr(str(Path(__file__).resolve().parent))+')\nimport roofline_thp_exec as launch\nlaunch.configure=lambda policy: int(policy == "disabled")\nlaunch.main(sys.argv[1:])\n')
        for threads in (1, 2):
            env = dict(os.environ, OMP_NUM_THREADS=str(threads), OMP_DYNAMIC='FALSE', OMPI_MCA_btl='self', OMPI_MCA_pml='ob1')
            for policy in thp.POLICIES:
                with patch.object(thp.ThpSession, 'worker_command', command):
                    worker = thp.ThpSession(topo, binary, env, out, 't{}-{}-reuse'.format(threads, policy),
                        'spmv', 'fo', 'packed', policy, n=4, threads=threads, launcher=False)
                try:
                    if sys.platform == 'linux':
                        status = Path('/proc/{}/status'.format(worker.identity['pid'])).read_text()
                        assert 'THP_enabled:\t'+str(int(policy == 'default')) in status
                    for index, variant in enumerate(timing.VARIANTS):
                        kind = 'sweep' if variant.startswith('counted') else 'normal'
                        traced = variant.endswith('-trace')
                        text = worker.capture(topo, binary, out/'t{}-{}-i{}'.format(threads, policy, index), kind, 60,
                            trace=traced, resources=traced, monitor_prefix=[sys.executable, str(mock), 'good'])
                        row = timing.actions.parse_output(text, 'spmv', 'fo', 'packed', kind, 60, n=4, threads=threads)
                        timing.parse_trace(text, row['seconds'], traced, resources=traced)
                    worker.close()
                finally:
                    worker.close(abort=True)
    print('THP sessions: 16 real checked windows, exec/PID/FD inheritance PASS; '+
          ('Linux prctl and live policy readback PASS' if sys.platform == 'linux' else 'Linux policy mocked on this host')+'; no PMU')


if __name__ == '__main__':
    launch_tests()
    collection_tests()
    if len(sys.argv) > 1:
        binary_tests(Path(sys.argv[1]).resolve())
