#!/usr/bin/env python3
"""Keep nonlinear convergence and the fixed MMS accuracy target separate."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re


def require(condition, message):
    if not condition:
        raise ValueError(message)


def single_match(pattern, text, description):
    matches = list(re.finditer(pattern, text, re.M))
    require(len(matches) == 1, 'expected one ' + description)
    return matches[0]


def read_case(path, n, code):
    text = path.read_text()
    require(single_match(r'^case:.*$', text, 'case header').group() ==
            'case: mms  geom: affine  refine_level: 1  semi_structured: 0',
            'wrong manufactured-solution geometry')
    require(single_match(r'^channel:.*$', text, 'grid header').group() ==
            'channel: L=(2,2,2)  cells=({0},{0},{0})'.format(n), 'wrong domain or grid')
    require(single_match(r'^rho:.*$', text, 'physical parameters').group().startswith(
            'rho: 1  mu: 0.01  mms_Re (=1/mu, the parameter in p): 100 '),
            'wrong density, viscosity or manufactured pressure parameter')
    require(single_match(r'^pressure gauge:.*$', text, 'pressure gauge').group().startswith(
            'pressure gauge: zero mean '), 'wrong pressure gauge')
    stages = list(re.finditer(r'^stage (\d+)/5: rho: (\S+)  Re: (\S+)$', text, re.M))
    require([(int(s[1]), float(s[2]), float(s[3])) for s in stages] ==
            [(1, .005, 1), (2, .02, 4), (3, .08, 16), (4, .32, 64), (5, 1, 200)],
            'missing or changed continuation stages')
    convergence = single_match(r'^newton_converged: (\d+)\s.*$', text, 'nonlinear verdict')
    require(convergence[1] == '1', 'nonlinear solve did not converge')
    require(stages[-1].end() < convergence.start(), 'misplaced nonlinear verdict')
    final_stage = text[stages[-1].end():convergence.start()]
    residuals = re.findall(r'^newton \d+\s+\|\|R\|\|: (\S+)\s+rel: (\S+)', final_stage, re.M)
    require(residuals, 'missing final-stage nonlinear residual')
    absolute, relative = map(float, residuals[-1])
    require(math.isfinite(absolute) and math.isfinite(relative) and
            absolute >= 0 and 0 <= relative <= 1e-10, 'final nonlinear target not met')
    match = single_match(r'^mms_err: u_l2 (\S+)\s+p_l2 (\S+)\s+p_l2_shifted (\S+)\s+\(dp_mean (\S+), vol (\S+)\)',
                         text, 'MMS error record')
    require(match.start() > convergence.end(), 'errors precede nonlinear verdict')
    u_l2, p_l2, p_shifted, dp, volume = map(float, match.groups())
    require(all(math.isfinite(v) for v in (u_l2, p_l2, p_shifted, dp, volume)) and
            u_l2 > 0 and p_l2 > 0 and p_shifted > 0 and abs(volume - 8) < 1e-6,
            'invalid error norms or volume')
    norms = single_match(r'^u_linf: (\S+)\s+p_linf: (\S+)', text, 'maximum-error record')
    require(norms.start() > convergence.end(), 'maximum errors precede nonlinear verdict')
    u_inf, p_inf = map(float, norms.groups())
    require(all(math.isfinite(v) and v >= 0 for v in (u_inf, p_inf)), 'invalid maximum errors')
    accuracy = u_inf <= 0.01
    require(code == (0 if accuracy else 1), 'driver exit disagrees with accuracy gate')
    if accuracy:
        require('verification failed' not in text, 'unexpected verification failure')
    else:
        fail = single_match(r'^verification failed \(converged=1, u_linf=(\S+), tol=0.01\)$',
                            text, 'accuracy-failure verdict')
        require(float(fail[1]) == u_inf, 'accuracy-failure error disagrees with norm')
    return dict(n=n, h=2.0/n, nonlinear_relative=relative, u_l2=u_l2,
                p_l2_shifted=p_shifted, u_linf=u_inf, p_linf=p_inf,
                absolute_accuracy_pass=accuracy, driver_exit=code,
                intermediate_floor='residual floor, accepting as converged' in text,
                log=path.name, log_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def summarize(root, grids=(16, 24, 32)):
    grids = tuple(grids)
    require(grids and all(isinstance(n, int) and n > 0 for n in grids) and
            grids == tuple(sorted(set(grids))), 'grids must be positive, unique and increasing')
    arms = ('lo', 'ho_lim0', 'ho_lim1', 'ho_lim2_k0', 'ho_lim3')
    expected = {(arm, n) for arm in arms for n in grids}
    seen, rows = set(), []
    for line in (root/'runs.tsv').read_text().splitlines():
        fields = line.split('\t')
        require(len(fields) == 4, 'malformed run record')
        group, label, log, extra = fields
        require(group == 'conv', 'unexpected study group')
        arm, n_text = label.rsplit('_n', 1)
        n = int(n_text)
        require((arm, n) in expected and (arm, n) not in seen, 'unexpected or duplicate case')
        require(label == '{}_n{}'.format(arm, n) and log == 'conv_' + label + '.log',
                'log path does not match the scheme and grid')
        items = [item.split('=', 1) for item in extra.split(',')]
        require(len(items) == 2 and all(len(item) == 2 for item in items) and
                {item[0] for item in items} == {'conv_arm', 'exit_code'}, 'invalid run metadata')
        metadata = dict(items)
        require(metadata['conv_arm'] == arm, 'scheme label disagrees with metadata')
        try:
            row = read_case(root/log, n, int(metadata['exit_code']))
        except ValueError as error:
            raise ValueError('{}: {}'.format(log, error)) from error
        row['scheme'] = arm
        rows.append(row)
        seen.add((arm, n))
    require(seen == expected, 'missing or unexpected case')
    lines = ['All {} final nonlinear targets met; spatial accuracy is reported separately.'.format(len(expected)),
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
              'Scheme labels come from runs.tsv; source, binary and launch provenance need separate review.',
              'Intermediate residual-floor acceptance, where present, is recorded in summary.json.',
              'This is an affine MMS study, not an unstructured-mesh or solver-performance result.']
    (root/'summary.json').write_text(json.dumps(rows, indent=2, sort_keys=True)+'\n')
    (root/'summary.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--grids', nargs='+', type=int, default=(16, 24, 32),
                        help='complete grid list expected for each of the five schemes')
    args = parser.parse_args()
    summarize(args.directory, args.grids)
