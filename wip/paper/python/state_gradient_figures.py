"""Generate PR60 state-reuse results from the pinned, external Grace archive."""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import statistics
import zipfile

PAPER = Path(__file__).resolve().parents[1]
ARCHIVE_NAME = 'pasc27-state-gradients-5114021.zip'
ARCHIVE_SHA256 = '436fd698025351559618b2c50bb9b6a38fcb269c52061f1dc2323d05e6104d6c'
PREFIX = 'state-gradients-5114021/'
COMMIT = '9cfefc2a67390cdf6e7aadc84c5db3244d2a5704'
SCHEMES = [('first_order', 'First-order', None), ('unlimited', 'Unlimited', 0),
           ('barth_jespersen', 'Barth–Jespersen', 1), ('venkatakrishnan', 'Venkatakrishnan', 2),
           ('darwish', 'Darwish', 3)]
BEGIN = '<!-- BEGIN GENERATED STATE GRADIENT RESULTS -->'
END = '<!-- END GENERATED STATE GRADIENT RESULTS -->'

def require(condition, message):
    if not condition:
        raise ValueError(message)

def validate(data, read):
    require(data['status'] == 'collected', 'incomplete campaign')
    require(data['config'] == dict(n=128, pack=1024, threads=72, warmup=30, repeat=70, local=False), 'wrong scope')
    require(data['environment'] == dict(OMP_NUM_THREADS='72', OMP_DYNAMIC='FALSE', OMP_PROC_BIND='close',
            OMP_PLACES='cores', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1'), 'changed thread settings')
    rows, pairs = data['captures'], data['pairs']
    expected = []
    for block in range(6):
        schemes = SCHEMES[block % 5:] + SCHEMES[:block % 5]
        if block % 2:
            schemes = list(reversed(schemes))
        for scheme, _, _ in schemes:
            order = ['per_apply', 'cached']
            if (block + [s[0] for s in SCHEMES].index(scheme)) % 2:
                order.reverse()
            expected.extend((block, scheme, mode) for mode in order)
    require([(r['block'], r['scheme'], r['mode']) for r in rows] == expected, 'missing, duplicate or reordered captures')
    for row in rows:
        mode, scheme, stem = row['mode'], row['scheme'], row['stem']
        require(stem == '{}-b{}-{}'.format(scheme, row['block'], mode), 'wrong capture identity')
        log, csv_data = [read('captures/' + stem + suffix) for suffix in ('.log', '.csv')]
        require(hashlib.sha256(log).hexdigest() == row['log_sha256'] and
                hashlib.sha256(csv_data).hexdigest() == row['csv_sha256'], 'capture hash changed')
        records = list(csv.DictReader(io.StringIO(csv_data.decode())))
        require(len(records) == 1 and row['exit'] == 0, 'failed or ambiguous capture')
        record = records[0]
        fields = dict(operation='jac_action', layout='packed', ran_layout='packed', geom='affine',
                      threads='72', pack_size='1024', cube_n='128', repeat='70', rhie_chow='1',
                      boundary='0', exact_rc='1', ran_rc='exact', live_vectors='7', jac_state_gradients=mode)
        require(all(record[k] == value for k, value in fields.items()), 'wrong CSV scope')
        seconds = float(record['seconds_per_call'])
        require(math.isfinite(seconds) and seconds > 0 and seconds == row['seconds'], 'invalid timing')
        text = log.decode()
        relative = re.findall(r'^jac_action_rc_vs_atomic_rel: (\S+)$', text, re.M)
        require(len(relative) == 1 and 0 <= float(relative[0]) < 1e-10 and
                float(relative[0]) == row['layout_relative'], 'failed layout check')
        require(re.findall(r'^  state_gradients: (\S+)$', text, re.M) == [mode] and
                re.findall(r'^  scalar: (\S+)$', text, re.M) == ['f64'], 'wrong arithmetic or mode')
        require(len(re.findall(r'^THP_LAUNCH policy=disabled disabled=1 pid=\d+ PASS$', text, re.M)) == 1, 'missing THP readback')
        flags = ['--n', '128', '--pack-size', '1024', '--layout', 'packed', '--geom', 'affine',
                 '--jac-action', '--rhie-chow', '--perturb', '0.017', '--live-vectors', '7',
                 '--warmup', '30', '--repeat', '70']
        limiter = next(s[2] for s in SCHEMES if s[0] == scheme)
        if limiter is not None:
            flags += ['--conv-ho', str(limiter)]
        if mode == 'per_apply':
            flags += ['--rebuild-state-gradients']
        capture = data['capture_root'] + '/' + stem
        require(row['command'] == data['prefix'] + [data['binary']] + flags +
                ['--csv', capture + '.csv', '--dump', capture + '.bin'], 'changed command')
    require(len(pairs) == 30 and {(p['block'], p['scheme']) for p in pairs} ==
            {(b, s[0]) for b in range(6) for s in SCHEMES}, 'missing vector comparisons')
    for pair in pairs:
        require(pair['count'] == 4 * 129**3 and 0 <= pair['relative'] <= 64 * 2**-52 and
                pair['tolerance'] == 64 * 2**-52, 'failed vector comparison')
    return rows

def read_archive(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    require(h.hexdigest() == ARCHIVE_SHA256, 'state-gradient archive checksum mismatch')
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), 'duplicate archive members')
        read = lambda name: archive.read(PREFIX + name)
        data = json.loads(read('captures/records.json'))
        require(json.loads(read('kit/SOURCE.json'))['commit'] == COMMIT, 'wrong source commit')
        require(json.loads(read('binding.json')) == dict(cpus=list(range(72)), numa=0), 'wrong NUMA scope')
        require(data['binary_sha256'] == read('binaries.sha256').decode().split()[0], 'binary identity mismatch')
        tests = read('build/Testing/Temporary/LastTest.log').decode()
        require(tests.count('Test Passed.') == 5 and 'Test Failed.' not in tests, 'failed numerical regressions')
        require('\nnid006553\n' in read('run.log').decode(), 'wrong host')
        return validate(data, read)

def results(rows):
    output = []
    for scheme, label, _ in SCHEMES:
        groups = [[r['seconds'] * 1000 for r in rows if r['scheme'] == scheme and r['mode'] == m]
                  for m in ('per_apply', 'cached')]
        a, b = groups
        ratios = []
        for block in range(6):
            pair = {r['mode']: r['seconds'] for r in rows if r['scheme'] == scheme and r['block'] == block}
            ratios.append(pair['per_apply'] / pair['cached'])
        output.append(dict(scheme=scheme, label=label, rebuild=a, reuse=b, ratios=ratios,
                           saved=100 * (1 - statistics.mean(b) / statistics.mean(a))))
    return output

def outputs(rows):
    result = results(rows)
    values = dict(sgFirstSaved=result[0]['saved'], sgHigherMin=min(r['saved'] for r in result[1:]),
                  sgHigherMax=max(r['saved'] for r in result[1:]))
    macros = '% Generated by python/state_gradient_figures.py; do not edit.\n'
    macros += ''.join('\\newcommand{\\%s}{%.2f}\n' % (name, value) for name, value in values.items())
    macros += '\\newcommand{\\sgPairs}{%d}\n' % sum(len(r['ratios']) for r in result)
    macros += '\\newcommand{\\sgPositive}{%d}\n' % sum(sum(x > 1 for x in r['ratios']) for r in result)
    for mode in ('rebuild', 'reuse'):
        sample = result[-1][mode]
        for suffix, value in [('Mean', statistics.mean(sample)), ('Min', min(sample)), ('Max', max(sample))]:
            macros += '\\newcommand{\\sgDarwish%s%s}{%.3f}\n' % (mode.title(), suffix, value)
    lines = ['| Scheme | Rebuild mean ms [range] | Reuse mean ms [range] | Time saved | Paired time-ratio range | Pairs favor reuse |',
             '|---|---|---|---:|---|---:|']
    for r in result:
        a, b, ratios = r['rebuild'], r['reuse'], r['ratios']
        lines.append('| {} | {:.6f} [{:.6f}, {:.6f}] | {:.6f} [{:.6f}, {:.6f}] | {:.2f}% | [{:.6f}, {:.6f}] | {}/6 |'.format(
            r['label'], statistics.mean(a), min(a), max(a), statistics.mean(b), min(b), max(b),
            r['saved'], min(ratios), max(ratios), sum(x > 1 for x in ratios)))
    return macros, '\n'.join(lines)

def selftest():
    rows = [dict(scheme=s[0], block=b, mode=m, seconds=1.0 if m == 'per_apply' else 2.0)
            for b in range(6) for s in SCHEMES for m in ('per_apply', 'cached')]
    macros, table = outputs(rows)
    require(table.count('-100.00%') == 5 and table.count('0/6') == 5, 'negative results were filtered')
    require('\\sgPositive}{0}' in macros and '\\sgPairs}{30}' in macros, 'wrong pair counts')
    require(all(r['saved'] == -100 for r in results(rows)), 'wrong time reduction')
    print('State-gradient generation: mean ratios, reductions and negative-pair retention PASS')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--selftest', action='store_true')
    args = parser.parse_args()
    if args.selftest:
        selftest()
    else:
        archive = args.archive or Path(os.environ.get('SFEM_STATE_GRADIENT_ARCHIVE', str(PAPER / 'data' / ARCHIVE_NAME)))
        rows = read_archive(archive)
        macros, table = outputs(rows)
        readme_path = PAPER / 'README.md'
        readme = readme_path.read_text()
        require(readme.count(BEGIN) == 1 and readme.count(END) == 1, 'missing or duplicate result markers')
        readme = readme.split(BEGIN)[0] + BEGIN + '\n\n' + table + '\n\n' + END + readme.split(END)[1]
        (PAPER / 'figures' / 'state_gradient_macros.tex').write_text(macros)
        readme_path.write_text(readme)
        print('State-gradient macros and reproducibility table generated from all 60 captures')
