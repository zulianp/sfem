"""Companion plots of counted work and transported traffic, with measured references."""
import copy
import json
import math
from pathlib import Path
import statistics


ARMS = [
    ('residual-fo-atomic', 'First-order, atomic', 'rlBlue', '*', False),
    ('residual-fo-packed', 'First-order, packed', 'rlBlue', '*', True),
    ('residual-darwish-atomic', 'Darwish, atomic', 'rlOrange', 'square*', False),
    ('residual-darwish-packed', 'Darwish, packed', 'rlOrange', 'square*', True),
    ('jacobian-fo-atomic', r'$J_{\rm exact}$, atomic', 'rlBlue', '*', False),
    ('jacobian-fo-packed', r'$J_{\rm exact}$, packed', 'rlBlue', '*', True),
    ('jacobian-darwish-atomic', r'$J^{\rm ho}_{\rm exact}$, atomic', 'rlOrange', 'triangle*', False),
    ('jacobian-darwish-packed', r'$J^{\rm ho}_{\rm exact}$, packed', 'rlOrange', 'triangle*', True),
    ('lagged-fo-atomic', r'$J_{\rm lagged}$, atomic', 'rlGreen', 'square*', False),
    ('lagged-fo-packed', r'$J_{\rm lagged}$, packed', 'rlGreen', 'square*', True),
    ('spmv-fo-packed', r'$J_{\rm lagged}$, BSR SpMV', 'black', 'diamond*', True),
]
PREAMBLE = '% Generated from the verified job 5047137 archive; do not edit.\n'


def project(row, floating=True):
    values = [row[key] for key in ('seconds', 'fp_ops', 'read_bytes', 'write_bytes')]
    if any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError('nonfinite or negative roofline input')
    seconds, work, read, write = values
    traffic = read + write
    if seconds <= 0 or traffic <= 0 or (floating and work <= 0):
        raise ValueError('zero roofline denominator or work')
    return dict(ops_per_byte=work/traffic, gops=work/seconds/1e9,
                gbps=traffic/seconds/1e9)


def summary(values):
    return dict(mean=statistics.mean(values), minimum=min(values), maximum=max(values))


def plot_data(meta, result):
    if result['job'] != meta['job'] or result['host'] != meta['host'] or result['policy'] != meta['thp']:
        raise ValueError('plot scope differs from verified campaign')
    if len(result['stages']) != 4 or not all(stage['accepted'] for stage in result['stages'].values()):
        raise ValueError('incomplete accepted campaign')
    samples = result['counted_samples']
    expected = {arm[0] for arm in ARMS} | {'copy', 'triad', 'dgemm'}
    if set(samples) != expected:
        raise ValueError('incomplete roofline arms')
    data = dict(schema=1, job=meta['job'], host=meta['host'], source_archive_sha256=meta['sha256'],
                scope={key: meta[key] for key in ('n', 'pack_size', 'threads', 'precision', 'geometry',
                       'rhie_chow', 'state_gradients', 'direction', 'boundary', 'transient', 'thp')},
                axes=dict(x='speculative FP operations / transported byte', y='speculative FP Gop/s'),
                center='arithmetic mean of per-capture x and y; not an additional capture',
                variation='all captures shown; ranges are not confidence intervals',
                operators={}, references={})
    for name in sorted(expected):
        rows = samples[name]
        repeats = (1, 2, 3) if name == 'dgemm' else ((10, 40, 70) if name in ('copy', 'triad') else (30, 60, 90))
        identities = [(row['block'], row['repeat']) for row in rows]
        if len(rows) != 18 or set(identities) != {(block, repeat) for block in range(6) for repeat in repeats}:
            raise ValueError('missing or duplicate roofline captures: '+name)
        projected = [dict(row, **project(row, name != 'copy')) for row in rows]
        if name in ('copy', 'triad', 'dgemm'):
            quantity = 'gops' if name == 'dgemm' else 'gbps'
            data['references'][name] = dict(quantity=quantity, samples=projected,
                                            **summary([row[quantity] for row in projected]))
        else:
            data['operators'][name] = dict(samples=projected,
                center=dict(ops_per_byte=statistics.mean(row['ops_per_byte'] for row in projected),
                            gops=statistics.mean(row['gops'] for row in projected)),
                normal_ms=result['metrics'][name]['normal_ms'],
                normal_range=result['metrics'][name]['normal_range'])
    return data


def coordinates(rows):
    return ' '.join('({:.12g},{:.12g})'.format(row['ops_per_byte'], row['gops']) for row in rows)


def panel(data, arms, title):
    text = [PREAMBLE, r'\begin{tikzpicture}', r'\begin{axis}[',
            r'width=\linewidth,height=7.9cm,scale only axis=false,',
            r'xmode=log,ymode=log,log basis x=10,log basis y=10,',
            r'xmin=0.16,xmax=10,ymin=60,ymax=3300,',
            r'xtick={0.2,0.5,1,2,5,10},xticklabels={0.2,0.5,1,2,5,10},',
            r'ytick={100,200,500,1000,2000},yticklabels={100,200,500,1000,2000},',
            r'xlabel={Speculative FP operations / transported byte},',
            r'ylabel={Speculative FP Gop/s},title={'+title+'},',
            r'grid=major,major grid style={gray!18},tick label style={font=\footnotesize},',
            r'label style={font=\footnotesize},legend columns=2,',
            r'legend style={at={(0.5,-0.27)},anchor=north,draw=none,font=\footnotesize,',
            r'cells={anchor=west},column sep=6pt},]']
    refs = data['references']
    low = min(refs[name]['minimum'] for name in ('copy', 'triad'))
    high = max(refs[name]['maximum'] for name in ('copy', 'triad'))
    text.append(r'\addplot[draw=none,fill=gray!15,forget plot] coordinates {'+
                '(.16,{}) (10,{}) (10,{}) (.16,{})'.format(.16*low,10*low,10*high,.16*high)+r'} \closedcycle;')
    gemm = refs['dgemm']
    text.append(r'\addplot[draw=none,fill=gray!15,forget plot] coordinates {'+
                '(.16,{lo}) (10,{lo}) (10,{hi}) (.16,{hi})'.format(lo=gemm['minimum'],hi=gemm['maximum'])+r'} \closedcycle;')
    for name, style in (('copy', 'dashed'), ('triad', 'dotted')):
        text.append(r'\addplot[gray,'+style+r',thick,no marks,forget plot] coordinates {'+
                    '(.16,{}) (10,{})'.format(.16*refs[name]['mean'],10*refs[name]['mean'])+'};')
    text.append(r'\addplot[gray,dashdotted,thick,no marks,forget plot] coordinates {'+
                '(.16,{0}) (10,{0})'.format(gemm['mean'])+'};')
    for name, label, color, mark, filled in arms:
        fill = color if filled else 'white'
        style = 'only marks,color='+color+',mark='+mark+',mark options={fill='+fill+'}'
        value = data['operators'][name]
        text.append(r'\addplot['+style+r',mark size=1.25pt,opacity=0.45,forget plot] coordinates {'+coordinates(value['samples'])+'};')
        text.append(r'\addplot['+style+r',mark size=2.7pt,thick] coordinates {'+coordinates([value['center']])+'};')
        text.append(r'\addlegendentry{'+label+'}')
    text += [r'\end{axis}',r'\end{tikzpicture}','']
    return '\n'.join(text)


def values(data, result):
    refs = data['references']
    metrics = result['metrics']
    lagged, spmv = (metrics[name] for name in ('lagged-fo-packed', 'spmv-fo-packed'))
    values = dict(Copy='{:.1f}'.format(refs['copy']['mean']), Triad='{:.1f}'.format(refs['triad']['mean']),
                  Gemm='{:.0f}'.format(refs['dgemm']['mean']),
                  LaggedTime='{:.2f}'.format(lagged['normal_ms']), SpmvTime='{:.2f}'.format(spmv['normal_ms']),
                  Speedup='{:.2f}'.format(spmv['normal_ms']/lagged['normal_ms']),
                  WorkRatio='{:.2f}'.format(lagged['gop']/spmv['gop']),
                  RateRatio='{:.2f}'.format(lagged['gops']/spmv['gops']))
    return PREAMBLE+''.join(r'\newcommand{\rlComp%s}{%s}'%(key,value)+'\n' for key,value in values.items())


def build_outputs(meta, result, directory):
    data = plot_data(meta, result)
    directory = Path(directory)
    return [(directory/'roofline-data.json',json.dumps(data,indent=2,sort_keys=True)+'\n'),
            (directory/'roofline-residuals.tex',panel(data,ARMS[:4],'Residuals')),
            (directory/'roofline-actions.tex',panel(data,ARMS[4:],'Jacobian and lagged actions')),
            (directory/'roofline-values.tex',values(data,result))]


def selftest(meta, result):
    data = plot_data(meta, result)
    assert len(data['operators']) == 11 and len(data['references']) == 3
    for value in data['operators'].values():
        for row in value['samples']:
            assert math.isclose(row['gops']/row['ops_per_byte'], row['gbps'], rel_tol=1e-12)
    row = dict(block=0, repeat=30, seconds=2., fp_ops=6e9, read_bytes=1e9, write_bytes=2e9)
    assert project(row) == dict(ops_per_byte=2.,gops=3.,gbps=1.5)
    assert project(dict(row, seconds=4.)) == dict(ops_per_byte=2.,gops=1.5,gbps=.75)
    assert project(dict(row, fp_ops=12e9)) == dict(ops_per_byte=4.,gops=6.,gbps=1.5)
    assert project(dict(row, write_bytes=5e9)) == dict(ops_per_byte=1.,gops=3.,gbps=3.)
    for field, value in (('seconds',0.),('fp_ops',0.),('write_bytes',-1.),('read_bytes',float('nan'))):
        try:
            project(dict(row, **{field:value}))
        except ValueError:
            pass
        else:
            raise AssertionError('invalid axis input accepted')
    for kind in ('missing','duplicate','failed'):
        bad = copy.deepcopy(result)
        if kind == 'failed':
            next(iter(bad['stages'].values()))['accepted'] = False
        elif kind == 'missing':
            bad['counted_samples']['residual-fo-packed'].pop()
        else:
            bad['counted_samples']['residual-fo-packed'][-1] = bad['counted_samples']['residual-fo-packed'][0]
        try:
            plot_data(meta,bad)
        except ValueError:
            pass
        else:
            raise AssertionError('incomplete roofline accepted: '+kind)
    # Changing a normal timing must not move counter-based plot coordinates.
    changed = copy.deepcopy(result)
    changed['metrics']['lagged-fo-packed']['normal_ms'] *= 2
    assert plot_data(meta,changed)['operators']['lagged-fo-packed']['samples'] == data['operators']['lagged-fo-packed']['samples']
    return True
