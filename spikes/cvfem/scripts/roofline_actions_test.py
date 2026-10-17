#!/usr/bin/env python3
"""Check action identity, counter accounting and the retained-worker protocol locally."""
import ast
import copy
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import roofline_actions as actions
from roofline_operator_test import refused
from roofline_reuse_diagnostic_test import MOCK


def synthetic_records(repeat_counts=actions.residual.REPEAT_COUNTS):
    records = []
    for i, group in enumerate(actions.groups(repeat_counts)):
        for index, (block, operation, scheme, layout, kind, repeat) in enumerate(group):
            row = dict(block=block, operation=operation, scheme=scheme, layout=layout,
                       kind=kind, repeat=repeat, seconds=.01*repeat,
                       session=actions.session_name(block, operation, scheme, layout),
                       session_index=index, identity=dict(pid=i+1, buffers=['0x1', '0x2', '0x3']),
                       direction=dict(nodes=129**3, xor='1234567890abcdef', sum='abcdef1234567890'))
            if kind != 'normal':
                factor = repeat if kind == 'sweep' else .001
                row.update(read_bytes=2e8*factor, write_bytes=1e8*factor,
                           component_write_bytes=1e8*factor, fp_ops=4e8*factor,
                           scf_seconds=.01*repeat, core_seconds=.01*repeat)
            records.append(row)
    return records


def accounting_tests():
    records = synthetic_records()
    assert len(records) == 252 and actions.analyze(records)[1]
    for invalid in (records[:-1], records+[records[0]], list(reversed(records))):
        refused(lambda: actions.analyze(invalid))
    bad = copy.deepcopy(records)
    bad[1]['identity']['buffers'][2] = '0xff'
    refused(lambda: actions.analyze(bad))
    bad = copy.deepcopy(records)
    bad[1]['operation'] = 'residual'
    refused(lambda: actions.analyze(bad))
    bad = copy.deepcopy(records)
    bad[1]['direction']['sum'] = '0000000000000000'
    refused(lambda: actions.analyze(bad))
    marker = 'ROOFLINE_DIRECTION source=coordinates nodes=125 xor=1234567890abcdef sum=abcdef1234567890'
    assert actions.direction_identity(marker, 4)['nodes'] == 125
    for output in ('', marker+'\n'+marker, marker.replace('125', '126'), marker.replace('coordinates', 'indices')):
        refused(lambda: actions.direction_identity(output, 4))
    for key in ('seconds', 'read_bytes', 'write_bytes', 'fp_ops', 'scf_seconds', 'core_seconds'):
        bad = copy.deepcopy(records)
        next(r for r in bad if r['kind'] == 'sweep')[key] = float('nan')
        refused(lambda: actions.analyze(bad))
    for kind, key, value in (('sweep', 'scf_seconds', 99), ('sweep', 'component_write_bytes', 0),
                             ('sweep', 'fp_ops', 1), ('empty-before', 'fp_ops', 1e20),
                             ('normal', 'seconds', .01)):
        bad = copy.deepcopy(records)
        next(r for r in bad if r['kind'] == kind)[key] = value
        assert not actions.analyze(bad)[1], (kind, key)
    for name in ('roofline_actions.py', 'roofline_actions_test.py'):
        source = Path(__file__).with_name(name).read_text()
        if sys.version_info >= (3, 8):
            ast.parse(source, feature_version=(3, 6))
        else:
            ast.parse(source)
    counts = actions.residual.LONG_REPEAT_COUNTS
    longer = synthetic_records(counts)
    assert actions.analyze(longer, counts)[1]
    refused(lambda: actions.analyze(longer))
    refused(lambda: actions.analyze(records, counts))
    for kind, key, value in (('sweep', 'scf_seconds', 99), ('sweep', 'component_write_bytes', 0),
                             ('sweep', 'fp_ops', 1), ('empty-before', 'fp_ops', 1e20),
                             ('normal', 'seconds', .01)):
        bad = copy.deepcopy(longer)
        next(r for r in bad if r['kind'] == kind)[key] = value
        assert not actions.analyze(bad, counts)[1], (kind, key)
    print('Action accounting: seven arms, complete blocks, identities, finite counts and unchanged refusal screens PASS')


def binary_tests(binary):
    topo = dict(cpus=[0, 1], node=0, scf_cpu=0, pmu='mock')
    with tempfile.TemporaryDirectory() as temp:
        out = Path(temp)
        mock = out/'mock.py'
        mock.write_text(MOCK)
        directions = []
        for threads in (1, 2):
            env = dict(os.environ, OMP_NUM_THREADS=str(threads), OMP_DYNAMIC='FALSE',
                       OMPI_MCA_btl='self', OMPI_MCA_pml='ob1')
            low, middle, high = (actions.residual.REPEAT_COUNTS if threads == 1
                                 else actions.residual.LONG_REPEAT_COUNTS)
            for operation, scheme, layout in actions.ARMS:
                name = '{}-{}-{}-t{}-reuse'.format(operation, scheme, layout, threads)
                worker = actions.ActionSession(topo, binary, env, out, name, operation, scheme, layout,
                                               n=4, threads=threads, launcher=False)
                directions.append(worker.direction)
                try:
                    for kind, repeat in (('normal', middle), ('empty-before', low), ('sweep', high),
                                         ('sweep', middle), ('sweep', low), ('empty-after', low)):
                        output = worker.capture(topo, binary, out/(name+'-'+kind+str(repeat)), kind, repeat,
                                                monitor_prefix=[sys.executable, str(mock), 'good'])
                        args = (operation, scheme, layout, kind, repeat)
                        actions.parse_output(output, *args, n=4, threads=threads)
                        refused(lambda: actions.parse_output(output.replace('ROOFLINE_ACTION ', 'MISSING '),
                                                             *args, n=4, threads=threads))
                        refused(lambda: actions.parse_output(output.replace('state_gradients=cached', 'state_gradients=per_apply'),
                                                             *args, n=4, threads=threads))
                    worker.close()
                finally:
                    worker.close(abort=True)
        assert all(direction == directions[0] for direction in directions)
        for operation, scheme, layout in (('jacobian', 'darwish', 'packed'), ('spmv', 'fo', 'packed')):
            worker = actions.ActionSession(topo, binary, env, out, 'bad-'+operation+'-reuse',
                                           operation, scheme, layout, n=4, threads=2, launcher=False)
            try:
                try:
                    worker.capture(topo, binary, out/('bad-'+operation), 'sweep', 10,
                                   monitor_prefix=[sys.executable, str(mock), 'bad'])
                except (ValueError, subprocess.SubprocessError):
                    pass
                else:
                    raise AssertionError('bad controller accepted')
                assert worker.process.poll() is not None
            finally:
                worker.close(abort=True)
        common = [str(binary), '--n', '4', '--rhie-chow', '--perturb', '.017', '--roofline-window', 'normal']
        for extra in (['--jac-action', '--rebuild-state-gradients'], ['--jac-action', '--live-vectors', '7'],
                      ['--jac-action', '--conv-ho', '3', '--lagged-ho'],
                      ['--bsr-apply', '--conv-ho', '3'], ['--bsr-apply', '--layout', 'atomic']):
            result = subprocess.run(common+extra, cwd=out, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, universal_newlines=True, timeout=30)
            assert result.returncode != 0 and 'ROOFLINE_OPERATOR window=' not in result.stdout, result.stdout
    print('Action binary: 84 checked windows on one/two threads; controller and unsupported-scope refusals PASS (no PMU)')


if __name__ == '__main__':
    accounting_tests()
    if len(sys.argv) > 1:
        binary_tests(Path(sys.argv[1]).resolve())
