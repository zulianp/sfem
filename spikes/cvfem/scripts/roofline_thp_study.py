"""Compare default and disabled process THP policies without accepting performance."""
from pathlib import Path
import statistics
import sys

from roofline_actions import ActionSession

POLICIES = ('default', 'disabled')


def groups(timing_groups):
    for group in timing_groups:
        order = POLICIES if group[0][0] % 2 == 0 else POLICIES[::-1]
        for policy in order:
            yield policy, group


class ThpSession(ActionSession):
    def __init__(self, topo, binary, env, out, name, operation, scheme, layout, policy, **options):
        if policy not in POLICIES:
            raise ValueError('unknown THP policy')
        self.policy = policy
        super().__init__(topo, binary, env, out, name, operation, scheme, layout, **options)
        try:
            ready = (out/(name+'.ready.out')).read_text()
            expected = 'THP_LAUNCH policy={} disabled={} pid={} PASS'.format(
                policy, int(policy == 'disabled'), self.identity['pid'])
            if [s for s in ready.splitlines() if s.startswith('THP_LAUNCH ')] != [expected]:
                raise ValueError('missing or incorrect THP launch readback')
        except BaseException:
            self.close(abort=True)
            raise

    def worker_command(self, binary, elements, threads):
        command = super().worker_command(binary, elements, threads)
        return [sys.executable, str(Path(__file__).with_name('roofline_thp_exec.py')),
                self.policy, '--'] + command


def check_observation(observation, policy):
    status = observation['process']['status'].get('text', '')
    values = [line.split()[1:] for line in status.splitlines() if line.startswith('THP_enabled:')]
    if values != [[str(int(policy == 'default'))]]:
        raise ValueError('live worker THP policy is missing or differs from request')
    control = observation['system'].get('/sys/kernel/mm/transparent_hugepage/enabled', {}).get('text', '')
    if '[always]' not in control.split():
        raise ValueError('THP study requires the observed system always policy')


def observed_count(entry, key):
    rows = [line.split()[1:] for line in entry.get('text', '').splitlines() if line.split()[:1] == [key]]
    if len(rows) != 1 or not rows[0] or not rows[0][0].isdigit():
        return None
    return int(rows[0][0])


def analyze(records, timing_groups, analyze_timing):
    expected = [(policy, *case) for policy, group in groups(timing_groups) for case in group]
    if [tuple(r[k] for k in ('policy', 'block', 'index', 'variant')) for r in records] != expected:
        raise ValueError('incomplete or reordered THP comparison')
    if any(r['direction'] != records[0]['direction'] for r in records):
        raise ValueError('THP comparison changed the physical direction')
    if len({r['identity']['pid'] for r in records if r['index'] == 0}) != 16:
        raise ValueError('THP arms did not use separate workers')
    controls = set()
    for r in records:
        for stage in ('before', 'after'):
            obs = r['observations'][stage]
            check_observation(obs, r['policy'])
            controls.add(tuple(obs['system'].get('/sys/kernel/mm/transparent_hugepage/'+name, {}).get('text', '')
                               for name in ('enabled', 'defrag')))
    if len(controls) != 1 or not all(next(iter(controls))):
        raise ValueError('system THP controls changed or are unavailable')
    reports = []
    for policy in POLICIES:
        rows = [r for r in records if r['policy'] == policy]
        reports.append('Process THP policy: '+policy+'\n'+analyze_timing(rows, True))
    lines = ['THP policy diagnostic only; no performance acceptance.',
             'Eight paired blocks, 16 workers, 128 retained captures; policy order alternates by block.',
             'Each policy retains the same plain/traced normal/counted order, 30 warmups and 60 applies.',
             'Policies use separate processes and allocations; physical pages are not held identical.',
             'The 40 ms threshold is descriptive; all intervals and captures are retained.',
             'block variant default_ms disabled_ms disabled_minus_default_ms']
    for block in range(8):
        for variant in ('normal', 'counted', 'normal-trace', 'counted-trace'):
            means = [statistics.mean(1000*r['seconds']/60 for r in records
                      if (r['block'], r['variant'], r['policy']) == (block, variant, policy)) for policy in POLICIES]
            lines.append('{} {} {:.6f} {:.6f} {:+.6f}'.format(block, variant, *means, means[1]-means[0]))
    for policy in POLICIES:
        rows = [r for r in records if r['policy'] == policy]
        for phase in ('warmup', 'apply'):
            times = [t for r in rows for t in r.get(phase+'_times', [])]
            lines.append('{} {}: intervals={} above40ms={} max_ms={:.6f}'.format(
                policy, phase, len(times), sum(t > .04 for t in times), 1000*max(times)))
        huge = []
        collapse = []
        for r in rows:
            obs = [r['observations'][s] for s in ('before', 'after')]
            for sample in obs:
                value = observed_count(sample['process'].get('smaps_rollup', {}), 'AnonHugePages:')
                if value is not None:
                    huge.append(value)
            values = [observed_count(s['system'].get('/proc/vmstat', {}), 'thp_collapse_alloc') for s in obs]
            if all(v is not None for v in values):
                collapse.append(values[1]-values[0])
        lines.append('{} process AnonHugePages: available={}/128 range_kB={}'.format(
            policy, len(huge), [min(huge), max(huge)] if huge else 'unavailable'))
        lines.append('{} capture-wide system collapse counts: available={}/64 increases={} decreases={}'.format(
            policy, len(collapse), sum(v > 0 for v in collapse), sum(v < 0 for v in collapse)))
    lines.append('Huge-page snapshots cover the whole process/capture; system collapse counts may include other processes.')
    lines += ['An improvement with reproduced baseline bursts supports a process THP policy effect, not a specific kernel call.',
              'If baseline bursts do not recur, their absence in the disabled arm is inconclusive.',
              'No recipe is adopted automatically; matched operators and calibration would require separate validation.', '']
    return '\n'.join(lines)+'\n'+'\n'.join(reports)
