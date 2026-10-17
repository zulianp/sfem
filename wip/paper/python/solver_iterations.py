"""Generate the small CPU iteration table from its checked public-cavity logs."""
import importlib.util
import json
import math
from pathlib import Path
import re as regex

ROOT = Path(__file__).resolve().parents[3]
PARSER = ROOT / 'spikes/cvfem/scripts/solver_iteration_study.py'
spec = importlib.util.spec_from_file_location('solver_iteration_study', str(PARSER))
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)
DATASET = 'solver_iterations_grace_5039923'


def read_study(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if (manifest['readback_collector_sha256'] != study.digest(PARSER)
            or manifest['driver_patch_sha256'] != study.digest(directory / 'solver-driver.patch')):
        raise ValueError('solver evidence source hash mismatch')
    platform = manifest['measurement_platform']
    chip = {'local_cpu': 'Apple M2 Pro', 'grace_cpu': 'NVIDIA GH200 Grace CPU'}.get(platform)
    if (chip is None or manifest['threads'] != 1 or manifest['hardware']['chip'] != chip):
        raise ValueError('solver table platform differs from its caption')
    if platform == 'grace_cpu':
        if (manifest['host'] != 'nid006545' or manifest['hardware']['node'] != manifest['host']
                or manifest['binary_readback_sha256'] != manifest['binary_sha256']
                or manifest['campaign'] != 'validation-5039923-vfXOQo'
                or study.digest(directory / 'SOURCE-REVISIONS') != manifest['supplied_revisions_sha256']):
            raise ValueError('Grace campaign or binary provenance mismatch')
        for path, digest in manifest['sources'].items():
            if study.digest(directory / 'sources' / path) != digest:
                raise ValueError('archived solver source hash mismatch')
        for path, digest in manifest['provenance_files'].items():
            if study.digest(directory / path) != digest:
                raise ValueError('solver build provenance mismatch')
    rows = json.loads((directory / 'table.json').read_text())
    expected = {(n, re, warp, pc) for n in (4, 8, 12) for re in (1, 10)
                for warp in (0, .025) for pc in ('bjacobi', 'vanka')}
    keys = [(r['fine_cells'], r['reynolds'], r['warp'], r['preconditioner']) for r in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError('missing or duplicate solver case')
    for row in rows:
        relative = Path(row['log'])
        case = Path(row['directory'])
        if (relative.is_absolute() or '..' in relative.parts
                or case.is_absolute() or '..' in case.parts):
            raise ValueError('invalid solver log path')
        log = directory / row['directory'] / relative if platform == 'grace_cpu' else directory / relative
        if study.digest(log) != row['log_sha256']:
            raise ValueError('solver log hash mismatch')
        parsed = study.parse_log(log.read_text(), row['returncode'])
        if parsed['status'] != 'converged' or any(row.get(k) != v for k, v in parsed.items()):
            raise ValueError('solver counts differ from converged raw evidence')
        config = study.configuration(row['fine_cells'], 2, row['reynolds'],
                                     row['preconditioner'], 1, 1000, 60, row['warp'])
        if (row['environment'] != config or row['process_status'] != 'exited'
                or row['level'] != 2 or row['macro_cells'] * 2 != row['fine_cells']):
            raise ValueError('solver configuration differs from the table scope')
        summary = row['driver_summary']
        if (summary['nonlinear_rtol'] != float(config['SFEM_NL_RTOL'])
                or summary['nonlinear_atol'] != float(config['SFEM_NL_ATOL'])
                or summary['dofs'] != 4 * (row['fine_cells'] + 1)**3):
            raise ValueError('solver tolerance or mesh mismatch')
        for check in row['linear_checks']:
            target = max(float(config['SFEM_LSOLVE_ATOL']),
                         float(config['SFEM_LSOLVE_RTOL']) * check['rhs_norm'])
            if (not math.isclose(check['target'], target, rel_tol=1e-14, abs_tol=0)
                    or check['iterations'] > 1000
                    or check['at_limit'] != int(check['iterations'] >= 1000)):
                raise ValueError('linear tolerance or cap mismatch')
        if row['warp']:
            geometry, valid = study.geometry_record(log.read_text(), row['warp'])
            if (not valid or row['geometry'] != geometry
                    or not math.isclose(geometry['determinant_ratio_lower_bound'],
                                        (1 - geometry['jacobian_perturbation_bound'])**3,
                                        rel_tol=1e-14)):
                raise ValueError('deformed mesh certificate failed')
        if platform == 'grace_cpu':
            raw_result = json.loads((log.parent / 'result.json').read_text())
            if any(row.get(key) != value for key, value in raw_result.items()):
                raise ValueError('solver table differs from captured result')
            if len(row['linear_checks']) != row['newton_iterations']:
                raise ValueError('missing Newton correction checks')
            if row['preconditioner'] == 'vanka':
                setup = regex.search(r'^\s+vanka_setup\s+\S+\s+(\d+)\s+', log.read_text(), regex.M)
                if setup is None or int(setup[1]) != row['newton_iterations']:
                    raise ValueError('Vanka rebuild count differs from Newton count')
    by_key = dict(zip(keys, rows))
    for n, re, warp, _ in expected:
        left, right = (by_key[n, re, warp, pc] for pc in ('bjacobi', 'vanka'))
        if (left['newton_iterations'] != right['newton_iterations']
                or left.get('geometry') != right.get('geometry')):
            raise ValueError('paired Newton count or geometry differs')
    return by_key


def table(rows):
    lines = ['% Generated by python/solver_iterations.py via make figures; do not edit.',
             r'\begin{tabular}{@{}rrrrrrrr@{}}', r'\toprule',
             r'& & \multicolumn{3}{c}{Regular} & \multicolumn{3}{c}{Deformed} \\',
             r'\cmidrule(lr){3-5}\cmidrule(l){6-8}',
             r'$n$ & $Re$ & Newton & BJ & V & Newton & BJ & V \\', r'\midrule']
    for n in (4, 8, 12):
        for re in (1, 10):
            values = [n, re]
            for warp in (0, .025):
                bj, v = (rows[n, re, warp, pc] for pc in ('bjacobi', 'vanka'))
                values += [bj['newton_iterations'], bj['linear_iterations'], v['linear_iterations']]
            lines.append(' & '.join(map(str, values)) + r' \\')
    return '\n'.join(lines + [r'\bottomrule', r'\end{tabular}', ''])


def build(data, figures, tables):
    rows = read_study(Path(data) / DATASET)
    output = Path(tables) / 'solver_iterations.tex'
    output.write_text(table(rows))
    return [str(output)]


def selftest():
    import tempfile
    directory = ROOT / 'wip/paper/data' / DATASET
    read_study(directory)
    read_study(ROOT / 'wip/paper/data/solver_iterations_local_20261013')
    with tempfile.TemporaryDirectory() as temp:
        target = Path(temp)
        for source in directory.iterdir():
            if source.name != 'table.json':
                (target / source.name).symlink_to(source)
        original = json.loads((directory / 'table.json').read_text())
        for mutation in ('count', 'hash', 'config', 'missing'):
            rows = json.loads(json.dumps(original))
            if mutation == 'count':
                rows[0]['linear_iterations'] += 1
            elif mutation == 'hash':
                rows[0]['log_sha256'] = '0' * 64
            elif mutation == 'config':
                rows[0]['environment']['SFEM_LSOLVE_RTOL'] = '1e-3'
            else:
                rows.pop()
            (target / 'table.json').write_text(json.dumps(rows))
            try:
                read_study(target)
            except ValueError:
                pass
            else:
                raise AssertionError('modified solver evidence accepted: ' + mutation)
    return True
