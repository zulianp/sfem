"""Keep the selected worker memory policy consistent across a performance campaign."""
import json
import os
from pathlib import Path
import sys

VARIABLE = 'CVFEM_ROOFLINE_THP_POLICY'
DISABLED = dict(worker_thp='disabled', scope='benchmark_workers_only', system_settings='unchanged')


def selected(env=None):
    value = (os.environ if env is None else env).get(VARIABLE, '')
    if value not in ('', 'disabled'):
        raise ValueError('unsupported campaign THP policy: '+value)
    return value == 'disabled'


def record(out, env):
    if selected(env):
        (out/'memory-policy.json').write_text(json.dumps(DISABLED, indent=2)+'\n')


def command(arguments):
    return [sys.executable, str(Path(__file__).with_name('roofline_thp_exec.py')),
            'disabled', '--'] + arguments


def check_launch(output, pid):
    expected = 'THP_LAUNCH policy=disabled disabled=1 pid={} PASS'.format(pid)
    if [s for s in output.splitlines() if s.startswith('THP_LAUNCH ')] != [expected]:
        raise ValueError('worker did not confirm the selected THP policy before exec')


def observe(pid):
    status = Path('/proc/{}/status'.format(pid)).read_text()
    values = [line.split()[1:] for line in status.splitlines() if line.startswith('THP_enabled:')]
    if values != [['0']]:
        raise ValueError('worker THP disable is missing or changed')
    return dict(pid=pid, thp_enabled=0)


def save(path, pid):
    value = observe(pid)
    if value != dict(pid=pid, thp_enabled=0):
        raise ValueError('invalid worker THP observation')
    path.write_text(json.dumps(value)+'\n')


def verify_reference(folder, records):
    path = folder/'memory-policy.json'
    if not path.exists():
        if selected():
            raise ValueError('THP-disabled operators require THP-disabled calibration')
        return
    if not selected() or json.loads(path.read_text()) != DISABLED:
        raise ValueError('calibration and operators use different memory policies')
    expected = {(r['session'], r['session_index']): r for r in records}
    required = {'memory-policy.json'}
    seen = set()
    for request in folder.glob('*.request.json'):
        value = json.loads(request.read_text())
        key = (value['session'], value['index'])
        if key in seen or key not in expected or value['identity'] != expected[key]['identity']:
            raise ValueError('memory policy request does not match calibration records')
        seen.add(key)
        pid = value['identity']['pid']
        stem = request.name[:-len('.request.json')]
        required.add(request.name)
        for name in (stem+'.policy-before.json', stem+'.policy-after.json', value['session']+'.policy-ready.json'):
            required.add(name)
            if json.loads((folder/name).read_text()) != dict(pid=pid, thp_enabled=0):
                raise ValueError('calibration worker memory policy is missing or changed')
        name = value['session']+'.ready.out'
        required.add(name)
        check_launch((folder/name).read_text(), pid)
    if seen != set(expected):
        raise ValueError('incomplete calibration memory policy observations')
    if not required <= set(json.loads((folder/'files.sha256.json').read_text())):
        raise ValueError('calibration memory policy observations are not bound by capture hashes')
