"""Stage 2 command options (evidence 196): cmd_turn samples in-place turns, hold_settle delays the hold anchor until
the base has slowed down; both off by default."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'sim'/'rl'))
try:
    import gymnasium  # noqa: F401
    import mujoco  # noqa: F401
    import numpy as np
except ImportError:  # .venv-ai has no simulator stack; use .venv-sim
    mujoco = None

KW = dict(arch8=True, real_mass=True, randomize=False, seed=1, model_rev=3, springs='c', cmd_max=[1.5, .2, .5], zero_cmd=.25)


@unittest.skipIf(mujoco is None, 'mujoco/gymnasium not installed')
class WalkCommandTest(unittest.TestCase):
    def test_in_place_turns(self):
        from run_env import RunEnv
        off, on = RunEnv(**KW), RunEnv(**KW, cmd_turn=.2)
        turns = lambda e: sum(c[0] == 0. and c[2] != 0. for c in (e._sample_command() for _ in range(2000)))
        self.assertEqual(turns(off), 0)  # the default sampler never turns in place
        n = turns(on)
        self.assertGreater(n, 300); self.assertLess(n, 500)

    def test_hold_settle(self):
        from run_env import RunEnv
        for settle, expect_none in ((None, False), (.1, True)):
            e = RunEnv(**KW, hold_settle=settle, weights={'hold': -3.})
            e.reset(seed=1)
            e.command = np.zeros(3); e.hold_anchor = None
            e.data.qvel[0] = 1.  # still moving when the zero command begins
            self.assertEqual(e._anchor() is None, expect_none)
            e.data.qvel[:2] = 0.
            self.assertIsNotNone(e._anchor())


if __name__ == '__main__':
    unittest.main()
