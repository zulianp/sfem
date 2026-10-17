#!/usr/bin/env python3
"""Keep nonlinear convergence and the fixed MMS accuracy target separate."""
import argparse
import json
import math
from pathlib import Path
import re


def read_case(path, n, code):
    text = path.read_text()
    assert 'case: mms  geom: affine  refine_level: 1  semi_structured: 0' in text
    assert 'channel: L=(2,2,2)  cells=({0},{0},{0})'.format(n) in text
    assert 'rho: 1  mu: 0.01  mms_Re (=1/mu, the parameter in p): 100' in text
    assert 'pressure gauge: zero mean' in text and 'stage 5/5: rho: 1  Re: 200' in text
    assert text.count('newton_converged:') == 1
    assert re.search(r'newton_converged: 1\s', text)
    residuals = re.findall(r'^newton \d+\s+\|\|R\|\|: (\S+)\s+rel: (\S+)', text, re.M)
    absolute, relative = map(float, residuals[-1])
    assert math.isfinite(absolute) and math.isfinite(relative)
    assert absolute >= 0 and 0 <= relative <= 1e-10
    match = re.search(r'^mms_err: u_l2 (\S+)\s+p_l2 (\S+)\s+p_l2_shifted (\S+)\s+\(dp_mean (\S+), vol (\S+)\)', text, re.M)
    u_l2, p_l2, p_shifted, dp, volume = map(float, match.groups())
    assert all(math.isfinite(v) for v in (u_l2, p_l2, p_shifted, dp, volume))
    assert u_l2 > 0 and p_l2 > 0 and p_shifted > 0 and abs(volume - 8) < 1e-6
    norms = re.findall(r'^u_linf: (\S+)\s+p_linf: (\S+)', text, re.M)
    assert len(norms) == 1
    u_inf, p_inf = map(float, norms[0])
    assert all(math.isfinite(v) and v >= 0 for v in (u_inf, p_inf))
    accuracy = u_inf <= 0.01
    assert code == (0 if accuracy else 1), (path, code, accuracy)
    if accuracy:
        assert 'verification failed' not in text
    else:
        fail = re.search(r'verification failed \(converged=1, u_linf=(\S+), tol=0.01\)', text)
        assert fail and float(fail.group(1)) == u_inf
    return dict(n=n, h=2.0/n, nonlinear_relative=relative, u_l2=u_l2,
                p_l2_shifted=p_shifted, u_linf=u_inf, p_linf=p_inf,
                absolute_accuracy_pass=accuracy, driver_exit=code,
                intermediate_floor='residual floor, accepting as converged' in text)


def summarize(root):
    arms = ('lo', 'ho_lim0', 'ho_lim1', 'ho_lim2_k0', 'ho_lim3')
    expected = {(arm, n) for arm in arms for n in (16, 24, 32)}
    seen, rows = set(), []
    for line in (root/'runs.tsv').read_text().splitlines():
        group, label, log, extra = line.split('\t')
        assert group == 'conv' and Path(log).name == log
        arm, n_text = label.rsplit('_n', 1)
        n = int(n_text)
        assert (arm, n) in expected and (arm, n) not in seen
        metadata = dict(item.split('=', 1) for item in extra.split(','))
        assert metadata['conv_arm'] == arm
        row = read_case(root/log, n, int(metadata['exit_code']))
        row['scheme'] = arm
        rows.append(row)
        seen.add((arm, n))
    assert seen == expected, 'missing or unexpected case'
    lines = ['All 15 final nonlinear targets met; spatial accuracy is reported separately.',
             'The fixed maximum-velocity-error limit remains 0.01.',
             'Adjacent-grid rates are observations, not a common asymptotic-order guarantee.', '',
             '| Scheme | n | Velocity L2 | Shifted pressure L2 | Velocity max | Accuracy gate | Velocity L2 rate |',
             '|---|---:|---:|---:|---:|---|---:|']
    for arm in arms:
        prior = None
        for row in sorted((r for r in rows if r['scheme'] == arm), key=lambda r:r['n']):
            rate = None if prior is None else math.log(prior['u_l2']/row['u_l2'])/math.log(prior['h']/row['h'])
            row['adjacent_velocity_l2_rate'] = rate
            lines.append('| {} | {} | {:.7g} | {:.7g} | {:.7g} | {} | {} |'.format(
                arm, row['n'], row['u_l2'], row['p_l2_shifted'], row['u_linf'],
                'PASS' if row['absolute_accuracy_pass'] else 'FAIL', '-' if rate is None else '{:.4f}'.format(rate)))
            prior = row
    lines += ['', 'All logs and exit codes are retained, including failed accuracy gates.',
              'Intermediate residual-floor acceptance, where present, is recorded in summary.json.',
              'This is an affine MMS study, not an unstructured-mesh or solver-performance result.']
    (root/'summary.json').write_text(json.dumps(rows, indent=2, sort_keys=True)+'\n')
    (root/'summary.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    summarize(args.directory)
