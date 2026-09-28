"""Numpy policy runner (ROS side) must reproduce the trained SB3 policy and the env observation."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'sim'/'rl'), str(ROOT/'src/raptor_control/scripts')]
MODEL = ROOT/'sim/rl/runs/walk12_rocker_track/model.zip'
try:
    import stable_baselines3  # noqa: F401
    import mujoco  # noqa: F401
except ImportError:
    stable_baselines3 = None


@unittest.skipIf(stable_baselines3 is None or not MODEL.exists(), 'needs .venv-sim and a local trained run')
class PolicyRunnerTest(unittest.TestCase):
    def test_runner_matches_sb3_and_env_observation(self):
        from export_policy import export
        from rl_policy import PolicyRunner
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'p.npz'
            env, model, norm = export(str(MODEL), out, dof=12, sole='rocker')
            runner = PolicyRunner(out)
        env.resample_steps = 0
        env.command = np.array([.3, 0., 0.])
        obs_env = env._obs()
        for _ in range(100):
            d = env.data
            obs = runner.observation(d.qvel[3:6], d.qpos[3:7], env.command, d.qpos[env.q_adr], d.qvel[env.v_adr])
            # gravity (3:6): the env reads xmat from before the last 1 ms integration step, the runner the
            # current quaternion (as an IMU would); everything else must match exactly
            np.testing.assert_allclose(obs[3:6], obs_env[3:6], atol=5e-3)
            np.testing.assert_allclose(np.delete(obs, [3, 4, 5]), np.delete(obs_env, [3, 4, 5]), atol=1e-5)
            ref, _ = model.predict(norm.normalize_obs(obs_env[None]), deterministic=True)
            action = runner.act(obs_env)
            np.testing.assert_allclose(action, ref[0], atol=1e-4)
            obs_env, _, done, _, _ = env.step(ref[0])
            self.assertFalse(done)


if __name__ == '__main__':
    unittest.main()
