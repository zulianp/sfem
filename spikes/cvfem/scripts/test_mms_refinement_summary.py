#!/usr/bin/env python3
"""Synthetic acceptance/refusal checks; these are not numerical measurements."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import mms_refinement_summary as summary


ARMS = ('lo', 'ho_lim0', 'ho_lim1', 'ho_lim2_k0', 'ho_lim3')


def fixture(root):
    records = []
    for arm in ARMS:
        for n in (16, 24, 32):
            error = .03 * (16.0 / n) ** 2
            code = int(error > .01)
            label = '{}_n{}'.format(arm, n)
            log = 'conv_' + label + '.log'
            text = (
                'case: mms  geom: affine  refine_level: 1  semi_structured: 0\n'
                'channel: L=(2,2,2)  cells=({0},{0},{0})\n'
                'rho: 1  mu: 0.01  mms_Re (=1/mu, the parameter in p): 100   [continuation ramps rho to 1]\n'
                'pressure gauge: zero mean  (pressure dofs free)\n'
                'stage 1/5: rho: 0.005  Re: 1\n'
                'stage 2/5: rho: 0.02  Re: 4\n'
                'stage 3/5: rho: 0.08  Re: 16\n'
                'stage 4/5: rho: 0.32  Re: 64\n'
                'stage 5/5: rho: 1  Re: 200\n'
                'newton 0  ||R||: 1e-4  rel: 1\n'
                'newton 1  ||R||: 1e-16  rel: 1e-12\n'
                'newton_converged: 1  newton_it: 1 (last stage)\n'
                'mms_err: u_l2 {1} p_l2 .5 p_l2_shifted .4 (dp_mean 0, vol 8)\n'
                'u_linf: {1} p_linf: .2\n'
            ).format(n, error)
            if code:
                text += 'verification failed (converged=1, u_linf={}, tol=0.01)\n'.format(error)
            (root/log).write_text(text)
            records.append('conv\t{}\t{}\tconv_arm={},exit_code={}'.format(label, log, arm, code))
    (root/'runs.tsv').write_text('\n'.join(records) + '\n')


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        fixture(self.root)

    def summarize(self):
        with contextlib.redirect_stdout(io.StringIO()):
            summary.summarize(self.root)

    def test_accuracy_failures_are_retained(self):
        self.summarize()
        rows = json.loads((self.root/'summary.json').read_text())
        self.assertEqual(len(rows), 15)
        self.assertEqual(sum(r['absolute_accuracy_pass'] for r in rows), 5)
        self.assertEqual({r['driver_exit'] for r in rows}, {0, 1})
        for row in rows:
            self.assertEqual(len(row['log_sha256']), 64)
            if row['n'] != 16:
                self.assertAlmostEqual(row['adjacent_velocity_l2_rate'], 2)

    def test_invalid_evidence_is_rejected(self):
        log_name = 'conv_lo_n16.log'
        changes = [
            ('runs.tsv', 'conv_ho_lim1_n16.log', 'conv_ho_lim0_n16.log'),
            ('runs.tsv', 'conv_arm=lo,exit_code=1', 'conv_arm=lo,exit_code=0,exit_code=1'),
            ('runs.tsv', 'conv_arm=lo,exit_code=1', 'conv_arm=lo,exit_code=0'),
            ('runs.tsv', 'conv_arm=lo,exit_code=1', 'conv_arm=ho_lim0,exit_code=1'),
            (log_name, 'stage 3/5: rho: 0.08  Re: 16\n', ''),
            (log_name, 'stage 5/5: rho: 1  Re: 200', 'stage 5/5: rho: 1  Re: 100'),
            (log_name, 'mu: 0.01', 'mu: 0.02'),
            (log_name, 'cells=(16,16,16)', 'cells=(32,32,32)'),
            (log_name, 'pressure gauge: zero mean', 'pressure gauge: pinned'),
            (log_name, 'newton_converged: 1', 'newton_converged: 0'),
            (log_name, 'rel: 1e-12', 'rel: 1e-7'),
            (log_name, 'u_l2 0.03', 'u_l2 nan'),
            (log_name, 'u_linf: 0.03', 'u_linf: inf'),
            (log_name, 'vol 8)', 'vol 1)'),
            (log_name, 'tol=0.01)', 'tol=0.1)'),
            (log_name, 'u_linf: 0.03 p_linf: .2\n', ''),
            (log_name, 'u_linf: 0.03 p_linf: .2\n', 'u_linf: 0.03 p_linf: .2\nu_linf: 0.03 p_linf: .2\n'),
        ]
        for name, old, new in changes:
            with self.subTest(file=name, change=old):
                path = self.root/name
                original = path.read_text()
                self.assertIn(old, original)
                path.write_text(original.replace(old, new))
                with self.assertRaises(ValueError):
                    self.summarize()
                path.write_text(original)

    def test_final_stage_cannot_reuse_an_earlier_residual(self):
        path = self.root/'conv_lo_n16.log'
        text = path.read_text().replace('stage 5/5: rho: 1  Re: 200\n', '')
        text = text.replace('newton_converged:', 'stage 5/5: rho: 1  Re: 200\nnewton_converged:')
        path.write_text(text)
        with self.assertRaisesRegex(ValueError, 'missing final-stage'):
            self.summarize()

    def test_missing_and_duplicate_cases_are_rejected(self):
        path = self.root/'runs.tsv'
        lines = path.read_text().splitlines()
        for invalid in (lines[:-1], lines + lines[:1]):
            path.write_text('\n'.join(invalid) + '\n')
            with self.assertRaises(ValueError):
                self.summarize()

    def test_python_optimization_cannot_disable_validation(self):
        path = self.root/'conv_lo_n16.log'
        path.write_text(path.read_text().replace('rel: 1e-12', 'rel: 1e-7'))
        result = subprocess.run([sys.executable, '-O', summary.__file__, str(self.root)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                universal_newlines=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('final nonlinear target not met', result.stderr)
        self.assertFalse((self.root/'summary.json').exists())

    def test_n40_followup_requires_all_five_schemes(self):
        lines = []
        for arm in ARMS:
            old = self.root/('conv_{}_n32.log'.format(arm))
            new = self.root/('conv_{}_n40.log'.format(arm))
            new.write_text(old.read_text().replace('cells=(32,32,32)', 'cells=(40,40,40)'))
            lines.append('conv\t{0}_n40\tconv_{0}_n40.log\tconv_arm={0},exit_code=0'.format(arm))
        (self.root/'runs.tsv').write_text('\n'.join(lines) + '\n')
        with contextlib.redirect_stdout(io.StringIO()):
            summary.summarize(self.root, (40,))
        rows = json.loads((self.root/'summary.json').read_text())
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(r['n'] == 40 and r['adjacent_velocity_l2_rate'] is None for r in rows))
        with self.assertRaises(ValueError):
            self.summarize()
        (self.root/'runs.tsv').write_text('\n'.join(lines[:-1]) + '\n')
        with self.assertRaises(ValueError):
            summary.summarize(self.root, (40,))

    def test_invalid_grid_lists_are_rejected(self):
        for grids in ((), (0,), (-1,), (32, 16), (16, 16)):
            with self.subTest(grids=grids), self.assertRaises(ValueError):
                summary.summarize(self.root, grids)


if __name__ == '__main__':
    unittest.main()
