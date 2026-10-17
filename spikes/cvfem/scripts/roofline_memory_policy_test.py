#!/usr/bin/env python3
"""Check matched worker policies, reference refusal and real session inheritance."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import roofline_memory_policy as policy
import roofline_operator as residual
import roofline_actions as actions
import roofline_gemm as gemm
import roofline_reuse_diagnostic as reuse
from roofline_reuse_diagnostic_test import MOCK


def refused(call):
    try:
        call()
    except (ValueError, OSError, KeyError):
        return
    raise AssertionError('invalid memory policy accepted')


def reference_tests():
    assert not policy.selected({}) and policy.selected({policy.VARIABLE: 'disabled'})
    refused(lambda: policy.selected({policy.VARIABLE: 'default'}))
    marker = 'THP_LAUNCH policy=disabled disabled=1 pid=123 PASS\n'
    policy.check_launch(marker, 123)
    for output in ('', marker+marker, marker.replace('123', '124'), marker.replace('disabled=1', 'disabled=0')):
        refused(lambda: policy.check_launch(output, 123))
    for status in ('', 'THP_enabled:\t1\n', 'THP_enabled:\t0\nTHP_enabled:\t0\n'):
        with patch.object(Path, 'read_text', return_value=status):
            refused(lambda: policy.observe(123))
    with patch.object(Path, 'read_text', return_value='THP_enabled:\t0\n'):
        assert policy.observe(123) == dict(pid=123, thp_enabled=0)
    with tempfile.TemporaryDirectory() as temp:
        folder = Path(temp)
        identity = dict(pid=123, buffers=['0x1', '0x2', '0x3'])
        records = [dict(session='control-reuse', session_index=0, identity=identity)]
        with patch.dict(os.environ, {policy.VARIABLE: ''}):
            policy.record(folder, os.environ)
            assert not (folder/'memory-policy.json').exists()
            policy.verify_reference(folder, records)
        with patch.dict(os.environ, {policy.VARIABLE: 'disabled'}):
            refused(lambda: policy.verify_reference(folder, records))
            policy.record(folder, os.environ)
            (folder/'control.request.json').write_text(json.dumps(dict(session='control-reuse', index=0, identity=identity)))
            (folder/'control-reuse.ready.out').write_text(marker)
            for name in ('control-reuse.policy-ready.json', 'control.policy-before.json', 'control.policy-after.json'):
                (folder/name).write_text(json.dumps(dict(pid=123, thp_enabled=0)))
            hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}
            (folder/'files.sha256.json').write_text(json.dumps(hashes))
            policy.verify_reference(folder, records)
            for name, bad in (
                    ('memory-policy.json', '{}'),
                    ('control.request.json', json.dumps(dict(session='control-reuse', index=1, identity=identity))),
                    ('control.policy-before.json', '{"pid":124,"thp_enabled":0}'),
                    ('control.policy-after.json', '{"pid":123,"thp_enabled":1}'),
                    ('control-reuse.policy-ready.json', '{}'),
                    ('control-reuse.ready.out', marker.replace('123', '124')),
                    ('files.sha256.json', '{}')):
                path = folder/name
                original = path.read_text()
                path.write_text(bad)
                refused(lambda: policy.verify_reference(folder, records))
                path.unlink()
                refused(lambda: policy.verify_reference(folder, records))
                path.write_text(original)
            refused(lambda: policy.verify_reference(folder, records+[dict(records[0], session_index=1)]))
            for mode in ('phase-diagnostic', 'sweep-validation', 'spmv-timing'):
                refused(lambda: residual.prepare_capture(Path('/missing'), folder/'unused', 4096, mode))
        with patch.dict(os.environ, {policy.VARIABLE: ''}):
            refused(lambda: policy.verify_reference(folder, records))
    print('Memory policy: metadata, launch/readback, missing/changed/unhashed/mixed references and unsupported recipe refusal PASS')


def binary_tests(binary, calibration=None, gemm_binary=None):
    with tempfile.TemporaryDirectory() as temp, contextlib.ExitStack() as stack:
        out = Path(temp)
        mock = out/'monitor.py'
        mock.write_text(MOCK)
        if sys.platform != 'linux':
            shim = out/'mock-policy.py'
            shim.write_text('import sys\nsys.path.insert(0, '+repr(str(Path(__file__).resolve().parent))+')\nimport roofline_thp_exec as launch\nlaunch.configure=lambda policy: 1\nlaunch.main(sys.argv[1:])\n')
            original = policy.command
            def command(arguments):
                result = original(arguments)
                result[1] = str(shim)
                return result
            stack.enter_context(patch.object(policy, 'command', command))
            stack.enter_context(patch.object(policy, 'observe', side_effect=lambda pid: dict(pid=pid, thp_enabled=0)))
        captures = 0
        for threads in (1, 2):
            topo = dict(cpus=list(range(threads)), node=0, scf_cpu=0, pmu='mock')
            env = dict(os.environ, OMP_NUM_THREADS=str(threads), OMP_DYNAMIC='FALSE',
                       OPENBLAS_NUM_THREADS=str(threads), GOTO_NUM_THREADS=str(threads),
                       VECLIB_MAXIMUM_THREADS=str(threads), OMPI_MCA_btl='self', OMPI_MCA_pml='ob1',
                       CVFEM_CALIBRATION_WARMUP='30', **{policy.VARIABLE: 'disabled'})
            names = ['residual', 'jacobian', 'lagged', 'spmv']
            if calibration: names.append('calibration')
            if gemm_binary: names.append('gemm')
            for operation in names:
                name = '{}-t{}-reuse'.format(operation, threads)
                if operation == 'calibration':
                    worker = reuse.Session(topo, calibration, env, out, name, 4096, threads, False)
                elif operation == 'gemm':
                    worker = gemm.GemmSession(topo, gemm_binary, env, out, name, n=128, threads=threads, launcher=False)
                elif operation == 'residual':
                    worker = residual.OperatorSession(topo, binary, env, out, name, 'fo', 'packed', n=4, threads=threads, launcher=False)
                else:
                    worker = actions.ActionSession(topo, binary, env, out, name, operation, 'fo', 'packed', n=4, threads=threads, launcher=False)
                try:
                    kinds = ('empty-before', 'sweep') if operation == 'calibration' else ('normal', 'sweep')
                    for kind in kinds:
                        stem = out/(name+'-'+kind)
                        monitor = [sys.executable, str(mock), 'good']
                        if operation == 'calibration':
                            result = worker.capture(topo, calibration, stem, 'triad', 4096, kind, 10, monitor_prefix=monitor)
                            prefix = 'CALIBRATION mode=triad elements=4096 repeats=10 threads={} '.format(threads)
                            rows = [line for line in result.splitlines() if line.startswith(prefix)]
                            assert len(rows) == 1 and rows[0].endswith(' PASS')
                            values = dict(word.split('=') for word in rows[0].split()[1:-1])
                            assert float(values['seconds']) > 0 and float(values['useful_bytes']) == 4096*24
                            assert float(values['flops']) == 4096*2 and 'CALIBRATION_WARMUP sweeps=30' in result
                            phase = 'sweep' if kind == 'sweep' else 'sweep-empty'
                            assert 'PERF_PHASE phase='+phase+' controllers=2 transitions=4 PASS' in result
                        else:
                            result = worker.capture(topo, gemm_binary if operation == 'gemm' else binary, stem, kind, 3, monitor_prefix=monitor)
                            if operation == 'gemm': gemm.parse_output(result, kind, 3, n=128, threads=threads)
                            elif operation == 'residual': residual.parse_output(result, 'fo', 'packed', kind, 3, n=4, threads=threads, warmup=30)
                            else: actions.parse_output(result, operation, 'fo', 'packed', kind, 3, n=4, threads=threads)
                        for suffix in ('.policy-before.json', '.policy-after.json'):
                            assert json.loads(stem.with_suffix(suffix).read_text()) == dict(pid=worker.identity['pid'], thp_enabled=0)
                        captures += 1
                    worker.close()
                finally:
                    worker.close(abort=True)
        for failure in ('before', 'after'):
            worker = residual.OperatorSession(topo, binary, env, out, 'failure-'+failure, 'fo', 'packed', n=4, threads=threads, launcher=False)
            observed = dict(pid=worker.identity['pid'], thp_enabled=0)
            effects = [ValueError('changed policy')] if failure == 'before' else [observed, ValueError('changed policy')]
            try:
                with patch.object(policy, 'observe', side_effect=effects):
                    refused(lambda: worker.capture(topo, binary, out/('failure-'+failure), 'normal', 3))
                assert worker.process.poll() is not None and not worker.fds
            finally:
                worker.close(abort=True)
    print('Memory policy sessions: {} real numerical windows and policy-change cleanup PASS; {}; no PMU'.format(
        captures, 'Linux prctl/live policy PASS' if sys.platform == 'linux' else 'Linux policy mocked on this host'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binary', type=Path, nargs='?')
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--gemm', type=Path)
    args = parser.parse_args()
    reference_tests()
    if args.binary:
        binary_tests(args.binary.resolve(), args.calibration, args.gemm)
