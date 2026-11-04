"""Evidence 198: categorical commands and strict training budget."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'sim/rl'))
try:
    import mujoco
    import numpy as np
except ImportError:
    mujoco = None


@unittest.skipIf(mujoco is None, 'mujoco not installed')
class TurnCommandsTest(unittest.TestCase):
    def test_distribution_and_copy(self):
        from run_env import RunEnv
        mix = [[2, 0, 1, 3], [6, 0, -1, 1]]
        env = RunEnv(cmd_max=[6, 0, 1.5], cmd_mix=mix, seed=12)
        rows = np.array([env._sample_command() for _ in range(4000)])
        self.assertTrue(np.all((rows[:, 0] == 2) | (rows[:, 0] == 6)))
        self.assertAlmostEqual(float((rows[:, 0] == 2).mean()), .75, delta=.03)
        row = env._sample_command(); row[:] = 100
        self.assertLess(env._sample_command().max(), 100)

    def test_default_rng_unchanged(self):
        from run_env import RunEnv
        env = RunEnv(cmd_max=[6, .3, 1.5], cmd_turn=.1, zero_cmd=.1, top_cmd=.3, seed=42)
        ref = np.random.default_rng()
        ref.bit_generator.state = env.rng.bit_generator.state
        for _ in range(100):
            c = ref.uniform(-1, 1, 3)*env.cmd_max
            u = ref.random()
            c[0] = 0. if u < .1 else 6. if u < .4 else ref.uniform(0., 6.)
            if c[0] == 0.:
                c[1:] = 0.
            if ref.random() < .1:
                c[:] = 0.
                c[2] = ref.choice([-1., 1.])*ref.uniform(.3, 1.)*1.5
            np.testing.assert_array_equal(env._sample_command(), c)

    def test_invalid_mixtures(self):
        from run_env import RunEnv
        for mix in ([], [[1, 0, 0]], [[1, 0, 0, 0]], [[1, 0, 0, -1]],
                    [[1, 0, float('nan'), 1]], [[7, 0, 0, 1]]):
            with self.subTest(mix=mix), self.assertRaises(ValueError):
                RunEnv(cmd_max=[6, 0, 1.5], cmd_mix=mix)

    def test_budget_never_overshoots(self):
        from train_run import budget_chunk
        done, blocks = 0, []
        while True:
            n = budget_chunk(8_000_000-done, 8192)
            if not n:
                break
            blocks.append(n); done += n
        self.assertEqual(done, 7_995_392)
        self.assertEqual(len(blocks), 8)
        self.assertTrue(all(n <= 1_000_000 and n % 8192 == 0 for n in blocks))


@unittest.skipIf(mujoco is None, 'mujoco not installed')
class TurnTrainingChecksTest(unittest.TestCase):
    def test_kill_rules(self):
        import importlib.util
        path = Path(__file__).resolve().parents[1]/'docs/evidence/198-l80-turn/check_training.py'
        spec = importlib.util.spec_from_file_location('turn_training_checks', path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        reason = module.stop_reason
        prior = [dict(block=2, passed=1, yaw_mae=.4)]
        current = dict(block=4, passed=1, yaw_mae=.395)
        self.assertTrue(reason(current, prior, [900]*50).startswith('stop: no extra'))
        self.assertEqual(reason({**current, 'yaw_mae': .37}, prior, [900]*50), 'continue')
        self.assertEqual(reason({**current, 'passed': 2}, prior, [900]*50), 'continue')
        self.assertTrue(reason({**current, 'passed': 0}, [], [900]*50).startswith('stop: zero'))
        self.assertTrue(reason(current, [], [900]*40+[100]*10).startswith('stop: episode'))
        self.assertEqual(reason(dict(block=2, passed=0, yaw_mae=1.), [], [900]*50), 'continue')


if __name__ == '__main__':
    unittest.main()
