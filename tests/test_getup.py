"""Stage 1b get-up starts (RunEnv getup, run_env._drop): the robot starts at rest on the ground, the ankle on its knee
coupling, and step() pays the get-up reward without terminating on body contact."""
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

KW = dict(arch8=True, real_mass=True, randomize=False, seed=1, model_rev=3, cmd_max=[0., 0., 0.], springs='c')


@unittest.skipIf(mujoco is None, 'mujoco/gymnasium not installed')
class GetupTest(unittest.TestCase):
    def test_start_at_rest_on_ground(self):
        from run_env import RunEnv
        c = RunEnv(**KW)
        c.reset(seed=1)
        ka = lambda e, s: sum(e.data.qpos[e.q_adr][e.active.index(f'{s}_{j}_pitch_joint')] for j in ('knee', 'ankle'))
        couple = ka(c, 'left')  # ankle = couple - knee, the standing reset's coupling
        for mode in ('easy', 'sit', 'side', 'back'):
            e = RunEnv(**KW, getup={'modes': {mode: 1.}})
            for seed in range(3):
                e.reset(seed=seed)
                d = e.data
                self.assertEqual(e.getup_mode, mode)
                self.assertLess(d.xpos[e.base][2], .9, mode)  # not frozen in mid-air (the old fixed settle did)
                self.assertTrue(any(e.model.geom_bodyid[c.geom1] == 0 or e.model.geom_bodyid[c.geom2] == 0
                                    for c in d.contact[:d.ncon]), mode)
                if mode != 'easy':  # easy hands over while the crouch still moves; the others start at rest
                    self.assertLess(np.abs(d.qvel).max(), .5, mode)
                for s in ('left', 'right'):
                    self.assertAlmostEqual(ka(e, s), couple, delta=.02)

    def test_fall_mode_spans_the_fall(self):
        from run_env import RunEnv
        tilt = lambda e: float(np.arccos(np.clip(e.data.xmat[e.base].reshape(3, 3)[2, 2], -1, 1)))
        early = RunEnv(**KW, getup={'modes': {'fall': 1.}, 'fall_t': .05})
        late = RunEnv(**KW, getup={'modes': {'fall': 1.}, 'fall_t': 3., 'fall_v': [2., 2.5]})
        a, b = [], []
        for seed in range(6):
            early.reset(seed=seed); a.append(tilt(early))
            late.reset(seed=seed); b.append(tilt(late))
        self.assertLess(max(a), .2)  # handed over right after the kick: still standing
        self.assertGreater(np.median(b), 1.)  # handed over late: on the ground
        self.assertGreater(np.abs(early.data.qvel[:2]).max(), .5)  # the kick velocity is kept, not zeroed

    def test_step_reward_and_no_contact_termination(self):
        from run_env import RunEnv
        e = RunEnv(**KW, getup={'modes': {'back': 1.}, 'weights': {'potential': 1.}})
        e.reset(seed=1)
        for _ in range(10):
            _, r, term, _, info = e.step(np.zeros(e.action_space.shape))
            self.assertTrue(np.isfinite(r))
            self.assertFalse(term)  # lying on the back is the start, not a fall
        self.assertIn('potential', info['getup']['terms'])
        self.assertIsNone(e.hold_anchor)  # not standing: the hold place follows the robot
        self.assertFalse(info['getup']['standing'])

    def test_fixed_start_seed(self):
        from run_env import RunEnv
        e = RunEnv(**KW, getup={'modes': {'side': 1.}, 'start_seed': 7000})
        poses = []
        for seed in range(3):
            e.reset(seed=seed)
            poses.append(e.data.qpos[:7].copy())
        for p in poses[1:]:  # the same fallen start whatever the episode seed
            np.testing.assert_allclose(p, poses[0], atol=1e-9)


if __name__ == '__main__':
    unittest.main()
