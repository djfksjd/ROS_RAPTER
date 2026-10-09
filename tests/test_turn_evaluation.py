"""Command-step timing, full angular integration and raw gate boundary."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'sim'/'rl'))
try:
    import numpy as np
    from evaluate_run import episode, turn_pass
except ImportError:
    episode = None


class FakeEnv:
    base = 0
    weight = 1.
    tail = 'active'
    peaks = []
    init_seed = 0

    def __init__(self, fail_at=None):
        self.data = SimpleNamespace(time=0., xmat=np.eye(3).reshape(1,9), xpos=np.zeros((1,3)))
        self.command = np.zeros(3)
        self.angle = 0.
        self.executed = []
        self.fail_at = fail_at

    def _obs(self):
        return np.array([self.command[2]])

    def step(self, action):
        self.executed.append(float(action[0]))
        self.angle += float(action[0])*.02
        c, s = np.cos(self.angle), np.sin(self.angle)
        self.data.xmat[0] = [c,-s,0,s,c,0,0,0,1]
        self.data.time += .02
        info = dict(v_body=np.array([4.,0,0]), tilt=0., power=0., flight=False,
                    loaded={'left':True,'right':True}, L=[0,0,0], L_tail=[0,0,0],
                    L_legs=[0,0,0], tail_qd=[0,0])
        return self._obs(), 0., self.fail_at is not None and self.data.time >= self.fail_at, False, info

    def body_velocity(self, body):
        return [0,0,self.executed[-1]], [4,0,0]


class FakeNorm:
    def __init__(self, env):
        self.env = env

    def reset(self):
        return self.env._obs()[None]

    def normalize_obs(self, obs):
        return obs


class FakePolicy:
    def predict(self, obs, deterministic=True):
        return obs, None


@unittest.skipIf(episode is None, 'simulator dependencies unavailable')
class TurnEvaluationTest(unittest.TestCase):
    def test_command_timing_and_first_increment(self):
        env = FakeEnv()
        result = episode(FakePolicy(), FakeNorm(env), env, [4,0,0], .14, turn=(1.,.04,.06))
        self.assertEqual(env.executed, [0.,0.,1.,1.,1.,0.,0.])
        self.assertAlmostEqual(result['raw']['turn_rad'], .06, places=12)
        self.assertTrue(turn_pass(result, .06))

    def test_fall_before_turn_completion_fails(self):
        env = FakeEnv(fail_at=.08)
        result = episode(FakePolicy(), FakeNorm(env), env, [4,0,0], .14, turn=(1.,.04,.06))
        self.assertIsNone(result['raw']['turn_rad'])
        self.assertFalse(turn_pass(result, .06))

    def test_turn_completed_at_last_evaluation_step_is_retained(self):
        env = FakeEnv()
        result = episode(FakePolicy(), FakeNorm(env), env, [4,0,0], .1, turn=(1.,.04,.06))
        self.assertAlmostEqual(result['raw']['turn_rad'], .06, places=12)
        self.assertTrue(turn_pass(result, .06))

    def test_turn_speed_interval_excludes_before_and_after_turn(self):
        class ChangingSpeed(FakeEnv):
            def step(self,action):
                raw,reward,term,trunc,info=super().step(action)
                info['v_body'][0]=4. if self.command[2] else .25
                return raw,reward,term,trunc,info
        env=ChangingSpeed()
        r=episode(FakePolicy(),FakeNorm(env),env,[4,0,0],.14,turn=(1.,.04,.06))
        self.assertEqual(r['raw']['turn_interval_samples'],3)
        self.assertEqual(r['raw']['turn_interval_mean_vx'],4.)
        self.assertEqual(r['raw']['turn_interval_min_vx'],4.)
        self.assertEqual(r['raw']['turn_min_vx'],.25)

    def test_raw_values_determine_boundary_for_both_directions(self):
        for sign in [-1,1]:
            for raw, passes in [(1.19996,True),(1.20004,False)]:
                result = {'fell':False,'turn_rad':sign*1.2,'raw':{'turn_rad':sign*raw}}
                self.assertEqual(turn_pass(result, sign), passes)
        self.assertFalse(turn_pass({'fell':True,'raw':{'turn_rad':1.}}, 1.))


if __name__ == '__main__':
    unittest.main()
