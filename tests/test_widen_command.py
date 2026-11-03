"""train_run.widen_command_inputs (evidence 196): continuing the zero-command stand policy with commands keeps its actions
at a zero command exactly and gives the command inputs a usable normalisation."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'sim'/'rl'))
RUN = ROOT/'sim/rl/runs/l80_s1p'
WALK = ROOT/'sim/rl/runs/l80_w2'
try:
    import gymnasium  # noqa: F401
    import mujoco  # noqa: F401
    import numpy as np
    import stable_baselines3  # noqa: F401
except ImportError:  # .venv-ai has no simulator stack; use .venv-sim
    mujoco = None


@unittest.skipIf(mujoco is None or not (RUN/'model.zip').exists(), 'needs the simulator stack and the local run l80_s1p')
class WidenCommandTest(unittest.TestCase):
    def test_zero_command_actions_unchanged(self):
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from mirror_probe import ObservationReflection
        from sym_policy import SymPolicy, freeze_obs, reflection_constants
        from train_run import env_kwargs, widen_command_inputs
        from run_env import RunEnv
        kw = env_kwargs(json.loads((RUN/'args.json').read_text()))
        env = VecNormalize.load(str(RUN/'vecnorm.pkl'), DummyVecEnv([lambda: RunEnv(seed=0, **kw)]))
        model = PPO.load(str(RUN/'model.zip'), env=env, device='cpu')
        rng = np.random.default_rng(0)
        raw = rng.normal(0., 1., (64, env.observation_space.shape[0]))*np.sqrt(env.obs_rms.var)+env.obs_rms.mean
        raw[:, 6:9] = 0.
        before = model.predict(env.normalize_obs(raw), deterministic=True)[0]
        self.assertTrue(widen_command_inputs(env, model, [1.5, .2, .5]))
        freeze_obs(env)
        probe = RunEnv(seed=0, **kw)
        sym = reflection_constants(ObservationReflection(probe.active, probe.policy_idx, probe.q0, probe.obs_vel, probe.obs_contact,
                                                         len(probe.scan_dx), bool(probe.obs_hold)), env)
        new = PPO(SymPolicy, env, device='cpu', policy_kwargs={**model.policy_kwargs, 'sym': sym}, seed=0, verbose=0)
        new.policy.load_state_dict(model.policy.state_dict(), strict=False)
        after = new.predict(env.normalize_obs(raw), deterministic=True)[0]
        np.testing.assert_allclose(after, before, atol=1e-5)
        self.assertAlmostEqual(float(np.sqrt(env.obs_rms.var[6])), .075, places=6)
        cmd = raw.copy(); cmd[:, 6] = .15  # 1.5 m/s scaled: inside the clip now
        self.assertLess(np.abs(env.normalize_obs(cmd)[:, 6]).max(), 2.01)


    @unittest.skipIf(not (WALK/'model.zip').exists(), 'needs the local run l80_w2')
    def test_narrow_normalisation_rescaled(self):
        """A walking policy (std 0.075 on the scaled forward command) continued for running: identical actions for the
        walking commands, 11.1 m/s inside the clip."""
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from mirror_probe import ObservationReflection
        from sym_policy import SymPolicy, freeze_obs, reflection_constants
        from train_run import env_kwargs, widen_command_inputs
        from run_env import RunEnv
        kw = env_kwargs(json.loads((WALK/'args.json').read_text()))
        env = VecNormalize.load(str(WALK/'vecnorm.pkl'), DummyVecEnv([lambda: RunEnv(seed=0, **kw)]))
        model = PPO.load(str(WALK/'model.zip'), env=env, device='cpu')
        rng = np.random.default_rng(1)
        raw = rng.normal(0., 1., (64, env.observation_space.shape[0]))*np.sqrt(env.obs_rms.var)+env.obs_rms.mean
        raw[:, 6] = rng.uniform(0., .15, 64); raw[:, 7] = rng.uniform(-.4, .4, 64); raw[:, 8] = rng.uniform(-.25, .25, 64)
        before = model.predict(env.normalize_obs(raw), deterministic=True)[0]
        self.assertTrue(widen_command_inputs(env, model, [11.1, .3, 1.]))
        freeze_obs(env)
        probe = RunEnv(seed=0, **kw)
        sym = reflection_constants(ObservationReflection(probe.active, probe.policy_idx, probe.q0, probe.obs_vel, probe.obs_contact,
                                                         len(probe.scan_dx), bool(probe.obs_hold)), env)
        new = PPO(SymPolicy, env, device='cpu', policy_kwargs={**model.policy_kwargs, 'sym': sym}, seed=0, verbose=0)
        new.policy.load_state_dict(model.policy.state_dict(), strict=False)
        np.testing.assert_allclose(new.predict(env.normalize_obs(raw), deterministic=True)[0], before, atol=1e-4)
        fast = raw.copy(); fast[:, 6] = 1.11
        self.assertLess(np.abs(env.normalize_obs(fast)[:, 6]).max(), 2.5)


if __name__ == '__main__':
    unittest.main()
