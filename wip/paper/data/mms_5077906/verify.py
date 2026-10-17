#!/usr/bin/env python3
"""Verify the n=40 follow-up and combine its accuracy data with the earlier grids."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def check(root=ROOT):
    previous = root.parent/'mms_5077627'
    prior = module('previous_mms_verification', previous/'verify.py')
    prior_result = prior.check(previous)
    require, hashes = prior.require, prior.hashes
    archived = hashes(root/'RAW-SHA256.txt')
    actual = {str(p.relative_to(root)) for folder in ('raw', 'sources')
              for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    require(len(archived) == 20 and set(archived) == actual, 'fine archive coverage changed')
    for name, digest in archived.items():
        require(Path(name).parts[0] in ('raw', 'sources') and '..' not in Path(name).parts,
                'invalid archive path')
        require(hashlib.sha256((root/name).read_bytes()).hexdigest() == digest,
                'fine archive hash mismatch: ' + name)
    before = hashes(previous/'raw/SOURCE-SHA256.txt')
    after = hashes(root/'raw/SOURCE-SHA256.txt')
    allowed = {'SOURCE-REVISIONS.txt', 'run-mms-on-grace.sh',
               'sfem/spikes/cvfem/scripts/mms_refinement_summary.py',
               'sfem/spikes/cvfem/scripts/test_mms_refinement_summary.py',
               'sfem/spikes/cvfem/jobs/MMS_REFINEMENT.md'}
    changed = {name for name in set(before) | set(after) if before.get(name) != after.get(name)}
    require(changed == allowed, 'unexpected source change between refinement jobs')
    checks = (root/'raw/source-check.log').read_text().splitlines()
    require(len(checks) == len(after) and set(checks) == {name + ': OK' for name in after},
            'fine source manifest check incomplete')
    for local, remote in (('raw/run-mms-on-grace.sh', 'run-mms-on-grace.sh'),
                          ('sources/mms_refinement_summary.py', 'sfem/spikes/cvfem/scripts/mms_refinement_summary.py')):
        require(archived[local] == after[remote], 'fine launch/reader source identity differs')
    require(hashes(root/'raw/BUILD-SHA256.txt') == hashes(previous/'raw/BUILD-SHA256.txt'),
            'different solver binary between refinement jobs')
    def settings(directory):
        return dict(line.split('=', 1) for line in (directory/'raw/settings.txt').read_text().splitlines())
    a, b = settings(previous), settings(root)
    require({k for k in set(a) | set(b) if a.get(k) != b.get(k)} == {'CONV_LADDER', 'OUT'},
            'physical, solver or thread settings changed')
    require(b['CONV_LADDER'] == '40' and b['OUT'].endswith('/mms-fine-5077906-azilru/study'),
            'wrong follow-up grid or output')
    manifest = json.loads((root/'raw/study/manifest.json').read_text())
    old_manifest = json.loads((previous/'raw/study/manifest.json').read_text())
    require(manifest['machine'] == 'nid006550', 'wrong follow-up node')
    require(all(manifest[k] == old_manifest[k] for k in ('threads', 'level', 'solver', 'conv_solver')),
            'solver manifest differs')
    reader = module('fine_mms_reader', root/'sources/mms_refinement_summary.py')
    expected = {'lo', 'ho_lim0', 'ho_lim1', 'ho_lim2_k0', 'ho_lim3'}
    seen, rows, tsv = set(), [], []
    for line in (root/'raw/study/runs.tsv').read_text().splitlines():
        group, label, log, extra = line.split('\t')
        arm, n = label.rsplit('_n', 1)
        require(group == 'conv' and n == '40' and arm in expected and arm not in seen,
                'missing or duplicate fine case')
        code = int(arm == 'lo')
        require(log == 'conv_' + label + '.log' and
                extra == 'conv_arm={},exit_code={}'.format(arm, code), 'fine log/exit metadata differs')
        seen.add(arm)
        tsv.append(dict(group=group, label=label, log=log, conv_arm=arm, exit_code=float(code)))
        row = reader.read_case(root/'raw/study'/log, 40, code)
        row.update(scheme=arm, adjacent_velocity_l2_rate=None)
        rows.append(row)
    require(seen == expected and tsv == manifest['runs'], 'fine manifest coverage differs')
    require(rows == json.loads((root/'raw/study/summary.json').read_text()), 'fine raw summary differs')
    combined = []
    for job, directory, data in (
            ('5077627', previous, json.loads((previous/'raw/study/summary.json').read_text())),
            ('5077906', root, rows)):
        for original in data:
            row = dict(original)
            name = 'conv_{}_n{}.log'.format(row['scheme'], row['n'])
            row.update(source_job=job, log=name,
                       log_sha256=hashlib.sha256((directory/'raw/study'/name).read_bytes()).hexdigest())
            combined.append(row)
    ordered = []
    for arm in ('lo', 'ho_lim0', 'ho_lim1', 'ho_lim2_k0', 'ho_lim3'):
        ladder = sorted((r for r in combined if r['scheme'] == arm), key=lambda r: r['n'])
        require([r['n'] for r in ladder] == [16, 24, 32, 40], 'combined grid coverage differs')
        for index, row in enumerate(ladder):
            row['adjacent_velocity_l2_rate'] = (None if index == 0 else
                math.log(ladder[index-1]['u_l2']/row['u_l2']) / math.log(ladder[index-1]['h']/row['h']))
            ordered.append(row)
    return dict(cases=len(ordered), fine_cases=len(rows),
                source_checks=prior_result['source_checks']+len(after),
                identical_solver_binary=True, configured_threads=8,
                maximum_final_relative_residual=max(r['nonlinear_relative'] for r in ordered),
                absolute_accuracy_passes=sum(r['absolute_accuracy_pass'] for r in ordered), rows=ordered)


if __name__ == '__main__':
    result = check()
    if result != json.loads((ROOT/'verification.json').read_text()):
        raise ValueError('stored combined verification differs')
    print('Verified {} cases across four grids; {} fixed accuracy passes, {} retained failures.'.format(
        result['cases'], result['absolute_accuracy_passes'], result['cases']-result['absolute_accuracy_passes']))
    print('Identical solver binary; no new solve or performance comparison.')
