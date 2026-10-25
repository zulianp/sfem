"""Update the Jacobian scheme figure from the complete, pinned Grace study."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import zipfile
import perf_figures as perf

PAPER = Path(__file__).resolve().parents[1]
ARCHIVE = 'pasc27-jacobian-layouts-5138025.zip'
SHA256 = 'f6414f76657344d9d5b1e3041fd49010d3152355b22655237fc2e28eb3117e26'
PREFIX = 'jacobian-layouts-5138025/'
ROOT = '/capstor/scratch/cscs/gandanie/git/pasc27-jacobian-layout-study/results/jacobian-layouts-5138025-sXv4uz'
SCHEMES = [('upwind', 'first_order', None), ('unlimited', 'ho_unlimited', 0),
           ('barth', 'ho_clip', 1), ('venkat', 'ho_venkatakrishnan', 2),
           ('darwish', 'ho_darwish_moukalled', 3)]
LAYOUTS = ['packed', 'ecolor', 'atomic']
DOFS = 4 * 129**3
BEGIN = '<!-- BEGIN GENERATED JACOBIAN LAYOUT RESULTS -->'
END = '<!-- END GENERATED JACOBIAN LAYOUT RESULTS -->'

def require(ok, message):
    if not ok:
        raise ValueError(message)

def one(log, key):
    values = re.findall(r'^\s*' + re.escape(key) + r': (\S+)\s*$', log, re.M)
    require(len(values) == 1, 'missing or ambiguous ' + key)
    return values[0]

def arms():
    return [(s, l, 'cached' if s == 'darwish' and l == 'packed' else 'direct')
            for s, _, _ in SCHEMES for l in LAYOUTS] + [('darwish', 'packed', 'direct'), ('spmv', 'packed', 'direct')]

def validate(data, read):
    require(data['status'] == 'collected', 'incomplete campaign')
    require(data['config'] == dict(n=128, pack=1024, threads=72, warmup=30, repeat=70, local=False), 'wrong scope')
    require(data['root'] == ROOT + '/captures', 'wrong allocation')
    require(data['environment'] == dict(OMP_NUM_THREADS='72', OMP_DYNAMIC='FALSE', OMP_PROC_BIND='close',
            OMP_PLACES='cores', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1'), 'wrong thread settings')
    prefix = ['/user-environment/env/default/bin/python3', ROOT + '/kit/thp_exec.py', 'disabled', '--',
              '/usr/bin/numactl', '--physcpubind=' + ','.join(map(str, range(72))), '--membind=0']
    require(data['prefix'] == prefix, 'changed worker policy')
    hashes = {Path(line.split()[1]).name: line.split()[0] for line in read('binaries.sha256').decode().splitlines()}
    for precision in ('f64', 'f32'):
        require(data['binaries'][precision] == dict(path=ROOT + '/build/benchmark_' + precision,
                hash=hashes['benchmark_' + precision]), 'changed binary identity')
    require(data['checker_hash'] == hashes['compare'], 'changed checker identity')
    expected = []
    for b in range(6):
        order = arms()[b:] + arms()[:b]
        if b % 2:
            order.reverse()
        for arm in order:
            precisions = ['f64', 'f32'] if (arms().index(arm) + b) % 2 == 0 else ['f32', 'f64']
            expected.extend((b, list(arm), p) for p in precisions)
    rows = data['captures']
    require([(r['block'], r['arm'], r['precision']) for r in rows] == expected, 'missing, duplicate or reordered captures')
    for r in rows:
        s, layout, mode = r['arm']; p = r['precision']
        stem = 'b{}-{}-{}-{}-{}'.format(r['block'], s, layout, mode, p)
        require(r['stem'] == stem and r['exit'] == 0, 'failed or misidentified capture')
        raw = read('captures/' + stem + '.log'); log = raw.decode()
        require(hashlib.sha256(raw).hexdigest() == r['log_hash'], 'changed log')
        fields = dict(layout=layout, scalar=p, geom='affine', cube_n='128', dofs=str(DOFS), repeat='70',
                      OpenMP_threads='72', operation='bsr_apply' if s == 'spmv' else 'jacobian_action')
        require(all(one(log, k) == v for k, v in fields.items()), 'wrong printed scope')
        require(len(re.findall(r'^THP_LAUNCH policy=disabled disabled=1 pid=\d+ PASS$', log, re.M)) == 1, 'missing THP readback')
        if layout == 'packed':
            require(one(log, 'pack_size') == '1024', 'wrong pack size')
        if s != 'spmv':
            require(one(log, 'state_gradients') == 'cached' and one(log, 'pipeline_nodal_gradient') == layout, 'wrong gradient scope')
            if layout != 'atomic':
                error = float(one(log, 'jac_action_rc_vs_atomic_rel'))
                require(0 <= error < (1e-10 if p == 'f64' else 1e-5), 'failed layout check')
        if mode == 'cached':
            require(one(log, 'darwish_cache_mode') == 'read' and one(log, 'darwish_cache_checks') == '12', 'missing cache checks')
            require(0 <= float(one(log, 'darwish_cache_max_relative')) <= (5e-12 if p == 'f64' else 5e-5), 'failed cache check')
        else:
            require('darwish_cache_mode:' not in log, 'unexpected cache')
        key = 'seconds_per_bsr_apply' if s == 'spmv' else 'seconds_per_jac_action'
        seconds = float(one(log, key))
        require(math.isfinite(seconds) and seconds > 0 and seconds == r['seconds'], 'invalid or changed timing')
        cmd = prefix + [data['binaries'][p]['path'], '--n', '128', '--pack-size', '1024', '--geom', 'affine',
                '--layout', layout, '--rhie-chow', '--perturb', '0.017', '--live-vectors', '7', '--warmup', '30',
                '--repeat', '70', '--dump', ROOT + '/captures/' + stem + '.bin']
        if s == 'spmv':
            cmd += ['--bsr-apply', '--bsr-precision', 'double' if p == 'f64' else 'single']
        else:
            cmd += ['--jac-action']
            limiter = next(l for name, _, l in SCHEMES if name == s)
            if limiter is not None:
                cmd += ['--conv-ho', str(limiter)]
        if mode == 'cached':
            cmd += ['--darwish-cache-experiment']
        require(r['command'] == cmd, 'changed launch command')
    checks = data['comparisons']
    require(len(checks) == 102 and {(c['block'], tuple(c['arm'])) for c in checks} ==
            {(b, a) for b in range(6) for a in arms()}, 'missing or duplicate precision checks')
    for c in checks:
        raw = read('captures/' + c['log']); result = json.loads(raw)
        require(c['exit'] == 0 and hashlib.sha256(raw).hexdigest() == c['log_hash'] and result == c['result'], 'failed or changed precision check')
        require(result['pass'] is True and result['count'] == DOFS, 'wrong vector check')
        e, f = result['relative'], result['fraction_past_1e4']
        require(math.isfinite(e) and e >= 0 and math.isfinite(f) and 0 <= f <= 1, 'nonfinite precision check')
        require(f < .01 if c['arm'][0] in ('barth', 'venkat', 'darwish') else e < 2e-5, 'failed precision bound')
    return rows

def values(rows, arm, precision):
    return [r['seconds'] for r in rows if r['arm'] == list(arm) and r['precision'] == precision]

def products(rows):
    means = {(a, p): statistics.mean(values(rows, a, p)) for a in arms() for p in ('f64', 'f32')}
    chart = []
    for s, name, _ in SCHEMES:
        for layout in LAYOUTS:
            mode = 'cached' if s == 'darwish' and layout == 'packed' else 'direct'
            for p in ('f64', 'f32'):
                chart.append(perf.ConvHo(name, layout, DOFS / means[((s, layout, mode), p)] / 1e6, DOFS, 128**3, 0, 0, p))
    references = {p: DOFS / means[(('spmv', 'packed', 'direct'), p)] / 1e6 for p in ('f64', 'f32')}
    figure = perf.fig_scheme_bars(chart, references).replace("python/perf_figures.py", "python/jacobian_layout_figures.py", 1)
    macros = perf.macros_jacho(chart)
    for p, suffix in [('f64', 'Double'), ('f32', 'Single')]:
        direct = means[(('darwish', 'packed', 'direct'), p)]
        cached = means[(('darwish', 'packed', 'cached'), p)]
        spmv = means[(('spmv', 'packed', 'direct'), p)]
        atomic = means[(('darwish', 'atomic', 'direct'), p)]
        for tag, val in [('DirectMs', direct*1e3), ('CacheMs', cached*1e3), ('SpmvMs', spmv*1e3),
                         ('CacheSaved', 100*(1-cached/direct)), ('SpmvRatio', spmv/cached),
                         ('SpmvExtra', 100*(cached/spmv-1)), ('DirectLayoutRatio', atomic/direct)]:
            macros += r'\newcommand{\jachoDm%s%s}{%.2f}' % (suffix, tag, val) + '\n'
    lines = ['| Scheme | Layout | Coefficients | Precision | Mean ms [range] | MDOF/s from mean time |',
             '|---|---|---|---|---|---:|']
    for arm in arms():
        for p in ('f64', 'f32'):
            ts = [t*1e3 for t in values(rows, arm, p)]
            lines.append('| {} | {} | {} | {} | {:.6f} [{:.6f}, {:.6f}] | {:.3f} |'.format(*arm, p, statistics.mean(ts), min(ts), max(ts), DOFS/means[(arm,p)]/1e6))
    lines += ['', '| Comparison (numerator / cached Darwish) | Precision | Ratio of mean times | Paired range | Pairs above one |', '|---|---|---:|---|---:|']
    for name, arm in [('Direct Darwish', ('darwish','packed','direct')), ('Lagged SpMV (different operator)', ('spmv','packed','direct'))]:
        for p in ('f64','f32'):
            ratios = [a/b for a,b in zip(values(rows, arm, p), values(rows, ('darwish','packed','cached'), p))]
            lines.append('| {} | {} | {:.6f} | [{:.6f}, {:.6f}] | {}/6 |'.format(name,p,means[(arm,p)]/means[(('darwish','packed','cached'),p)],min(ratios),max(ratios),sum(r>1 for r in ratios)))
    return figure, macros, '\n'.join(lines)


PRECISION_SHA = '71e9ea6fc5163313dda2d55ffcd63c7ae3b1ba130e3baa013c7d3115eff22a48'
PRECISION_PREFIX = 'precision-5140975-HyOKbQ/'
PRECISION_ARMS = ['cached64','cached32','direct64','direct32','spmv64','spmv32','spmv_mixed']
PRECISION_BEGIN = '<!-- BEGIN GENERATED PRECISION DIAGNOSTIC -->'
PRECISION_END = '<!-- END GENERATED PRECISION DIAGNOSTIC -->'

def validate_precision(data, screens, read):
    require(data['status'] == 'collected' and data['config'] ==
            dict(n=128,pack=1024,threads=72,warmup=30,repeat=70,local=False), 'wrong precision scope')
    require(data['topology']['job'] == '5140975' and data['topology']['host'] == 'nid005555', 'wrong precision allocation')
    expected = []
    for b in range(6):
        order = PRECISION_ARMS[b:]+PRECISION_ARMS[:b]
        if b % 2: order.reverse()
        for arm in order:
            kinds = ['normal','live','scf','core']; k=(b+PRECISION_ARMS.index(arm))%4
            kinds = kinds[k:]+kinds[:k]
            if b % 2: kinds.reverse()
            expected.extend((b,arm,kind) for kind in kinds)
            if b in (0,5): expected.append((b,arm,'empty'))
    rows = data['captures']
    require([(r['block'],r['arm'],r['kind']) for r in rows] == expected, 'missing precision capture')
    require(set(screens) == set(PRECISION_ARMS) and all(len(v)==12 and all(x is True for x in v.values())
            for v in screens.values()), 'failed precision reliability screen')
    for r in rows + data['controls']:
        require(r['exit'] == 0, 'failed precision capture')
        for suffix, digest in r['hashes'].items():
            require(hashlib.sha256(read('captures/'+r['stem']+suffix)).hexdigest() == digest, 'changed precision log')
        require(math.isfinite(r['seconds']) and r['seconds'] >= 0 and
                (r['kind']=='empty' or r['seconds']>0), 'invalid precision time')
        if 'counters' in r:
            require(r['monitor_exit']==0 and all(math.isfinite(v['count']) and v['count']>=0 and
                    math.isfinite(v['seconds']) and v['seconds']>0 for v in r['counters'].values()), 'invalid precision counter')
    require(len(data['controls']) == 18 and len(data['checks']) == 146, 'missing precision controls/checks')
    for check in data['checks']:
        require(check['pass'] is True, 'failed precision numerical check')
        if check['against']=='identical':
            require(check['left_hash']==check['right_hash'], 'instrumentation changed result')
        else:
            raw=read('captures/'+check['log'])
            require(check['exit']==0 and hashlib.sha256(raw).hexdigest()==check['log_hash'] and
                    json.loads(raw)==check['result'], 'changed precision comparison')
    return rows

def precision_products(rows):
    selected=lambda arm,kind:[r for r in rows if r['arm']==arm and r['kind']==kind]
    mean=lambda arm,key:statistics.mean(r[key] for r in selected(arm,'normal'))
    metric=lambda arm,kind,key:statistics.mean(r['metrics'][key]/70 for r in selected(arm,kind))
    bandwidth=lambda arm:statistics.mean(r['metrics']['total_bytes']/70/r['seconds']/1e9 for r in selected(arm,'scf'))
    macros='\n% Separate precision diagnostic; does not replace figure measurements.\n'
    vals={}
    for arm,tag in [('spmv64','SpmvDouble'),('spmv32','SpmvSingle'),('spmv_mixed','SpmvMixed'),('cached64','CachedDouble'),('cached32','CachedSingle')]:
        vals[tag+'Ms']=(mean(arm,'seconds')*1000,2)
        vals[tag+'GB']=(metric(arm,'scf','total_bytes')/1e9,2)
        vals[tag+'Bandwidth']=(bandwidth(arm),0)
    vals['CachedSpeedup']=(mean('cached64','seconds')/mean('cached32','seconds'),2)
    for arm,tag in [('cached32','CachedLiveMs'),('spmv32','SpmvLiveMs')]:
        vals[tag]=(statistics.mean(r['seconds'] for r in selected(arm,'live'))*1000,2)
    for bits,tag in [('64','Double'),('32','Single')]:
        vals[tag+'TrafficExtra']=(100*(metric('cached'+bits,'scf','total_bytes')/metric('direct'+bits,'scf','total_bytes')-1),1)
        vals[tag+'InstructionsSaved']=(100*(1-metric('cached'+bits,'core','instructions')/metric('direct'+bits,'core','instructions')),1)
    for name,(value,digits) in vals.items():
        macros+=r'\newcommand{\pd%s}{%s}'%(name,format(value,'.'+str(digits)+'f'))+'\n'
    fmt=lambda vs:'{:.6f} [{:.6f}, {:.6f}]'.format(statistics.mean(vs),min(vs),max(vs))
    lines=['| Arm | Normal ms [range] | Live7 ms [range] | Transported GB/apply [range] | Transported GB/s [range] | Retired instructions, billions/apply [range] |',
           '|---|---|---|---|---|---|']
    for arm in PRECISION_ARMS:
        normal=selected(arm,'normal');live=selected(arm,'live');scf=selected(arm,'scf');core=selected(arm,'core')
        lines.append('| {} | {} | {} | {} | {} | {} |'.format(arm,
            fmt([r['seconds']*1000 for r in normal]),fmt([r['seconds']*1000 for r in live]),
            fmt([r['metrics']['total_bytes']/70/1e9 for r in scf]),
            fmt([r['metrics']['total_bytes']/70/r['seconds']/1e9 for r in scf]),
            fmt([r['metrics']['instructions']/70/1e9 for r in core])))
    return macros,'\n'.join(lines)

def precision_evidence(path, selftest=False):
    require(hashlib.sha256(Path(path).read_bytes()).hexdigest()==PRECISION_SHA,'wrong precision archive')
    with zipfile.ZipFile(path) as archive:
        require(len(archive.namelist())==len(set(archive.namelist())),'duplicate precision archive member')
        read=lambda name:archive.read(PRECISION_PREFIX+name)
        data=json.loads(read('captures/records.json'));screens=json.loads(read('captures/screens.json'))
        rows=validate_precision(data,screens,read)
        if selftest:
            mutations=[lambda d:d.update(status='failed'),lambda d:d['captures'].pop(),
                lambda d:d['captures'][0].update(seconds=float('nan')),
                lambda d:d['controls'].pop(),lambda d:d['checks'][0].update({'pass':False})]
            for mutate in mutations:
                bad=copy.deepcopy(data);mutate(bad)
                try:validate_precision(bad,screens,read)
                except ValueError:pass
                else:raise AssertionError('invalid precision evidence accepted')
            bad=copy.deepcopy(screens);bad['cached32']['scf_windows']=False
            try:validate_precision(data,bad,read)
            except ValueError:pass
            else:raise AssertionError('failed precision screen accepted')
            print('Precision evidence: pinned archive, complete records and six refusal checks PASS')
        return precision_products(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', default=os.environ.get('SFEM_JACOBIAN_LAYOUT_ARCHIVE', ARCHIVE))
    parser.add_argument('--selftest', action='store_true')
    parser.add_argument('--precision-archive', default=os.environ.get('SFEM_PRECISION_ARCHIVE', 'pasc27-precision-5140975.zip'))
    args = parser.parse_args()
    precision_macros, precision_table = precision_evidence(args.precision_archive, args.selftest)
    path = Path(args.archive)
    require(hashlib.sha256(path.read_bytes()).hexdigest() == SHA256, 'wrong evidence archive')
    with zipfile.ZipFile(path) as archive:
        require(len(archive.namelist()) == len(set(archive.namelist())), 'duplicate archive members')
        read = lambda name: archive.read(PREFIX + name)
        data = json.loads(read('captures/records.json'))
        rows = validate(data, read)
        if args.selftest:
            mutations = [lambda d:d.update(status='failed'), lambda d:d['captures'].pop(),
                lambda d:d['captures'].reverse(), lambda d:d['config'].update(local=True),
                lambda d:d['prefix'].append('changed'), lambda d:d['captures'][0].update(seconds=1),
                lambda d:d['captures'][0].update(exit=1), lambda d:d['captures'][0]['command'].append('--kernel-only'),
                lambda d:d['comparisons'].pop(), lambda d:d['comparisons'][0]['result'].update(relative=float('nan'))]
            for mutate in mutations:
                changed = copy.deepcopy(data); mutate(changed)
                try: validate(changed, read)
                except ValueError: pass
                else: raise AssertionError('malformed evidence accepted')
            print('Jacobian layout evidence: complete replay and ten refusal checks PASS')
            return
    figure, macros, table = products(rows)
    (PAPER/'figures/jacho.tex').write_text(figure)
    (PAPER/'figures/jacho_macros.tex').write_text(macros + precision_macros)
    readme = PAPER/'README.md'; text = readme.read_text()
    require(text.count(BEGIN) == text.count(END) == 1, 'missing documentation markers')
    text = text[:text.index(BEGIN)+len(BEGIN)] + '\n\n' + table + '\n\n' + text[text.index(END):]
    require(text.count(PRECISION_BEGIN)==text.count(PRECISION_END)==1,'missing precision documentation markers')
    text=text[:text.index(PRECISION_BEGIN)+len(PRECISION_BEGIN)]+'\n\n'+precision_table+'\n\n'+text[text.index(PRECISION_END):]
    readme.write_text(text)
    print('Jacobian figure and ranges generated from all 204 captures; original plot styling retained')

if __name__ == '__main__':
    main()
