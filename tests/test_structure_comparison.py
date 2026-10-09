"""Regression checks for structure-selection verdicts and missing evaluations."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'sim'/'rl'))
from compare_structures import robust, summarize


def heat_row(speed=4., sustainable=True, heat=1.):
    return dict(cmd=4., fell=False, sustainable=sustainable, sustainable_20hz=True,
                max_heat=heat, max_heat_20hz=.6, speed_last40_mps=speed,
                burst3s_mps=speed, steady_heat={'knee_pitch': heat}, mass_kg=11.29)


class StructureComparisonTest(unittest.TestCase):
    def test_display_rounding_cannot_override_failed_heat_verdict(self):
        result = summarize([heat_row(sustainable=False, heat=round(1.0004, 3))])
        self.assertEqual(result['S'], 0.)
        self.assertEqual(result['S_20hz'], 4.)

    def test_every_start_must_pass_and_speed_uses_slowest_start(self):
        self.assertEqual(summarize([heat_row(4.2), heat_row(3.8)])['S'], 3.8)
        self.assertEqual(summarize([heat_row(), heat_row(sustainable=False)])['S'], 0.)

    def test_fall_disqualifies_even_a_true_heat_flag(self):
        row = heat_row(); row['fell'] = True
        self.assertEqual(summarize([row])['S'], 0.)

    def test_40kmh_uses_exact_actual_speed_threshold(self):
        self.assertFalse(summarize([heat_row(11.111)])['reached_40kmh'])
        self.assertTrue(summarize([heat_row(100/9)])['reached_40kmh'])

    @patch('compare_structures.subprocess.run')
    def test_failed_subprocess_is_not_a_missing_display_cell(self, run):
        run.return_value = SimpleNamespace(returncode=1, stdout='', stderr='test failure')
        with self.assertRaisesRegex(RuntimeError, 'evaluation failed'):
            robust('run')
        self.assertEqual(run.call_count, 1)

    @patch('compare_structures.subprocess.run')
    def test_missing_cases_fail_even_when_exit_status_is_zero(self, run):
        run.return_value = SimpleNamespace(returncode=0, stdout='{}\n', stderr='')
        with self.assertRaisesRegex(RuntimeError, 'incomplete or duplicate'):
            robust('run')

    @patch('compare_structures.subprocess.run')
    def test_duplicate_cases_do_not_replace_missing_conditions(self, run):
        rows = [{'episodes': 5}] + [{'episodes': 5, 'impulse_nms': .5}]*3
        run.return_value = SimpleNamespace(returncode=0, stdout='\n'.join(map(json.dumps, rows)), stderr='')
        with self.assertRaisesRegex(RuntimeError, 'incomplete or duplicate'):
            robust('run')

    @patch('compare_structures.subprocess.run')
    def test_complete_evaluations_are_preserved(self, run):
        def response(cmd, **kwargs):
            turn = '--turns' in cmd
            values = [1., 2.] if not turn and cmd[cmd.index('--impulse-axis')+1] == 'y' else [.5, 1., 1.5]
            key = 'turn_cmd_rad_s' if turn else 'impulse_nms'
            rows = [{'episodes': 5, 'falls': 0}] + [{'episodes': 5, 'falls': 0, key: v} for v in values]
            return SimpleNamespace(returncode=0, stdout='\n'.join(map(json.dumps, rows)), stderr='')
        run.side_effect = response
        result = robust('run')
        self.assertEqual({k: len(v) for k, v in result.items()}, {'yaw': 4, 'pitch': 3, 'turn': 4})


if __name__ == '__main__':
    unittest.main()
