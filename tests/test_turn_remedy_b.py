"""The sole decision-B remedy has a fixed 2M budget and unchanged penalty weights."""
import importlib.util
import json
import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
E = ROOT/'docs/evidence/198-l80-turn'
sys.path.insert(0, str(ROOT/'sim/rl'))
spec = importlib.util.spec_from_file_location('check_training_b', E/'check_training_b.py')
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)
from train_run import budget_chunk


class TurnRemedyBTest(unittest.TestCase):
    def test_budget_and_terminal_rules(self):
        done = 0
        chunks = []
        while n := budget_chunk(2_000_000-done, 8192):
            done += n
            chunks.append(n)
        self.assertEqual(chunks, [999_424, 999_424])
        self.assertEqual(done, check.MAX_STEPS)
        self.assertEqual(check.stop_reason(chunks[0], [800.]*50), 'continue')
        self.assertIn('budget exhausted', check.stop_reason(done, [800.]*50))
        self.assertIn('candidate only', check.stop_reason(done, [800.]*50, True))
        self.assertIn('below half', check.stop_reason(chunks[0], [800.]*40+[300.]*10, True))
        self.assertIn('nonfinite', check.stop_reason(1, [float('nan')]))

    def test_only_positive_linear_weight_changes(self):
        weights = []
        for name in ('train_l80_t3.sh', 'train_l80_t4b.sh'):
            line = next(l for l in (E/name).read_text().splitlines() if l.startswith('W='))
            weights.append(json.loads(shlex.split(line[2:])[0]))
        before, after = weights
        self.assertEqual({k for k in before if before[k] != after[k]}, {'track_lin'})
        self.assertEqual((before['track_lin'], after['track_lin']), (5, 8))
        script = (E/'train_l80_t4b.sh').read_text()
        self.assertIn('--steps 2000000', script)
        self.assertIn('--init sim/rl/runs/l80_t3/model.zip', script)
        self.assertIn('commands.json', script)
        self.assertIn('check_training_b.py', script)

    def test_metrics_separate_turn_straight_and_ignore_recorded_six(self):
        from test_turn_gate import turn_row
        turn = turn_row(4.)
        turn['episodes'][0]['yaw_rate_mean'] = .1
        straight = dict(gate='turn', cond='run_regression', cmd=4., n=2, falls=0,
                        episodes=[dict(t_fall=None, v_mean=v) for v in (4., 4.8)])
        self.assertEqual(check.metrics([turn, turn_row(6.), straight]),
                         dict(turn_passed=3, turn_total=4, straight_passed=1, straight_total=2))

    def test_two_block_callback_keeps_regression_gates_and_stops(self):
        from test_turn_gate import complete_suite
        full = complete_suite()
        development = [r for r in complete_suite() if r['gate'] == 'turn']
        development[0]['episodes'][0]['v_mean'] = 0.  # no early full gate at 1M
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root/'candidate'; run.mkdir()
            for name in ('model.zip', 'vecnorm.pkl', 'args.json', 'initial-policy.json'):
                (run/name).write_text('dummy callback fixture')
            (run/'progress.csv').write_text('rollout/ep_len_mean\n'+'800\n'*50)
            with patch.object(check, 'EVIDENCE', root), patch.object(check, 'development', return_value=development), \
                 patch.object(check.sg, 'turn', return_value=full) as final_gate:
                with patch.object(sys, 'argv', ['check', str(run), '999424']):
                    self.assertEqual(check.main(), 0)
                final_gate.assert_not_called()
                with patch.object(sys, 'argv', ['check', str(run), '1998848']):
                    self.assertEqual(check.main(), 10)
            saved = [json.loads(s) for s in (root/'eval/candidate/full_2M.jsonl').read_text().splitlines()]
            self.assertEqual({r['gate'] for r in saved}, {'turn', 'stand', 'walk', 'switch'})
            history = json.loads((run/'turn_checks_b.json').read_text())
            self.assertEqual(len(history), 2)
            self.assertIn('candidate only', history[-1]['decision'])
