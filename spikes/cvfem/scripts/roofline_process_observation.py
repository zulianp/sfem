"""Read process and scheduler observations without changing system settings."""
from pathlib import Path, PurePosixPath
import re
import time


def read_optional(path):
    try:
        with Path(path).open() as stream:
            text = stream.read(1048577)
        if len(text) > 1048576:
            return dict(error='observation exceeds size limit')
        return dict(text=text)
    except OSError as error:
        return dict(error=type(error).__name__, errno=error.errno)


def cgroup_paths(membership, mounts):
    paths = set()
    def unescape(value):
        return re.sub(r'\\([0-7]{3})', lambda m: chr(int(m.group(1), 8)), value)
    for line in mounts.splitlines():
        fields = line.split()
        if '-' not in fields:
            continue
        separator = fields.index('-')
        if separator < 6 or len(fields) < separator+4:
            continue
        fs, options = fields[separator+1], set(fields[separator+3].split(','))
        if fs != 'cgroup2' and not (fs == 'cgroup' and 'cpu' in options):
            continue
        mount_root, mount = (PurePosixPath(unescape(fields[i])) for i in (3, 4))
        if not mount_root.is_absolute() or not mount.is_absolute() or '..' in mount_root.parts or '..' in mount.parts:
            continue
        for entry in membership.splitlines():
            parts = entry.split(':', 2)
            if len(parts) != 3:
                continue
            controllers = set(parts[1].split(','))
            if not ((fs == 'cgroup2' and parts[0:2] == ['0', '']) or
                    (fs == 'cgroup' and 'cpu' in controllers)):
                continue
            group = PurePosixPath(parts[2])
            if not group.is_absolute() or '..' in group.parts:
                continue
            try:
                relative = group.relative_to(mount_root)
            except ValueError:
                continue
            path = mount/relative
            while True:
                paths.add(str(path))
                if path == mount:
                    break
                path = path.parent
    return sorted(paths)


def snapshot(pid, read=read_optional):
    start = time.monotonic()
    proc = '/proc/{}'.format(pid)
    process = {name: read(proc+'/'+name) for name in ('status', 'cgroup', 'mountinfo', 'numa_maps', 'smaps_rollup')}
    system_paths = ('/proc/pressure/cpu', '/proc/pressure/memory', '/proc/pressure/io',
                    '/proc/vmstat', '/proc/sys/kernel/numa_balancing',
                    '/sys/kernel/mm/transparent_hugepage/enabled',
                    '/sys/kernel/mm/transparent_hugepage/defrag',
                    '/sys/kernel/mm/transparent_hugepage/hpage_pmd_size',
                    '/sys/kernel/mm/transparent_hugepage/khugepaged/pages_collapsed',
                    '/sys/kernel/mm/transparent_hugepage/khugepaged/full_scans')
    system = {path: read(path) for path in system_paths}
    membership, mounts = (process[name].get('text', '') for name in ('cgroup', 'mountinfo'))
    groups = {}
    for path in cgroup_paths(membership, mounts):
        groups[path] = {name: read(path+'/'+name) for name in (
            'cpu.stat', 'cpu.max', 'cpu.pressure', 'cpu.cfs_quota_us', 'cpu.cfs_period_us',
            'cpuset.cpus.effective', 'cpuset.mems.effective')}
    return dict(pid=pid, monotonic_start=start, monotonic_end=time.monotonic(),
                wall_seconds=time.time(), process=process, system=system, cgroups=groups)


def throttle_changes(before, after):
    rows = []
    for path, entry in before['cgroups'].items():
        values = []
        for sample in (entry, after['cgroups'].get(path, {})):
            text = sample.get('cpu.stat', {}).get('text', '')
            parsed = dict(line.split() for line in text.splitlines() if len(line.split()) == 2)
            values.append(parsed)
        for key in ('nr_throttled', 'throttled_usec', 'throttled_time'):
            if all(key in value for value in values):
                a, b = (int(value[key]) for value in values)
                rows.append(dict(path=path, counter=key, delta=b-a))
    return rows
