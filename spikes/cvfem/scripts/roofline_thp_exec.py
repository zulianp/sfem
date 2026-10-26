#!/usr/bin/env python3
"""Set a worker's THP policy before exec; never change system-wide settings."""
import ctypes
import os
import sys


def configure(policy, call=None):
    if policy not in ('default', 'disabled'):
        raise ValueError('unknown THP policy')
    if call is None:
        if sys.platform != 'linux':
            raise ValueError('THP intervention requires Linux')
        libc = ctypes.CDLL(None, use_errno=True)
        call = libc.prctl
        call.argtypes = [ctypes.c_int] + [ctypes.c_ulong]*4
        call.restype = ctypes.c_int
    # Linux UAPI: PR_SET_THP_DISABLE=41 and PR_GET_THP_DISABLE=42.
    def checked(option, value=0):
        result = call(option, value, 0, 0, 0)
        if result < 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        return result
    if checked(42) != 0:
        raise ValueError('worker inherited a non-default THP policy')
    expected = int(policy == 'disabled')
    if expected and checked(41, 1) != 0:
        raise ValueError('failed to disable THP for worker')
    if checked(42) != expected:
        raise ValueError('THP policy readback differs from request')
    return expected


def main(argv):
    if len(argv) < 3 or argv[1] != '--' or not os.path.isabs(argv[2]):
        raise ValueError('usage: roofline_thp_exec.py default|disabled -- /absolute/executable [args]')
    policy = argv[0]
    state = configure(policy)
    print('THP_LAUNCH policy={} disabled={} pid={} PASS'.format(policy, state, os.getpid()), flush=True)
    os.execv(argv[2], argv[2:])


if __name__ == '__main__':
    try:
        main(sys.argv[1:])
    except (ValueError, OSError) as error:
        print('THP LAUNCH FAILED: '+str(error), file=sys.stderr)
        sys.exit(1)
