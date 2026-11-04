"""Resume action coordinates: symmetric means, std, and clipped target limits."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'sim/rl'))
try:
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from train_run import rescale_action_outputs, env_kwargs, widen_command_inputs
    from run_env import RunEnv
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'simulation dependencies unavailable')
class ActionScaleResumeTest(unittest.TestCase):
    def test_rows_std_reset_and_noop(self):
        policy = SimpleNamespace(action_net=torch.nn.Linear(3, 3),
                                 log_std=torch.nn.Parameter(torch.tensor([-.7, -.8, -.9])),
                                 sym_a_idx=torch.tensor([2, 1, 0]))
        model = SimpleNamespace(policy=policy)
        names = ['left_hip_yaw', 'tail_pitch', 'right_hip_yaw']
        original = copy.deepcopy(policy)
        args = {'scale_override': '{"hip_yaw": 0.4}'}
        factors = rescale_action_outputs(model, names, np.arange(3), args, [.5, .8, .5])
        np.testing.assert_array_equal(factors, [.8, 1., .8])
        torch.testing.assert_close(policy.action_net.weight, original.action_net.weight*torch.tensor([.8, 1., .8])[:, None])
        torch.testing.assert_close(policy.action_net.bias, original.action_net.bias*torch.tensor([.8, 1., .8]))
        torch.testing.assert_close(policy.log_std, original.log_std+torch.log(torch.tensor([.8, 1., .8])))
        before = copy.deepcopy(policy)
        rescale_action_outputs(model, names, np.arange(3), args, [.4, .8, .4])
        torch.testing.assert_close(policy.action_net.weight, before.action_net.weight, rtol=0, atol=0)
        torch.testing.assert_close(policy.log_std, before.log_std, rtol=0, atol=0)
        rescale_action_outputs(model, names, np.arange(3), args, [.5, .8, .5], .15)
        # The existing reset runs after symmetric rebuild; no log-ratio is added here.
        torch.testing.assert_close(policy.log_std, before.log_std, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            rescale_action_outputs(model, names, np.arange(3), args, [.5, .8, .6])
        with self.assertRaises(ValueError):
            rescale_action_outputs(model, names, np.arange(3), args, [0., .8, 0.])

    @unittest.skipUnless((ROOT/'sim/rl/runs/l80_t1/model.zip').exists(), 'l80_t1 absent')
    def test_l80_t1_unsaturated_targets_and_symmetric_rebuild(self):
        from evaluate import load
        from sym_policy import SymPolicy, reflection_constants, freeze_obs
        from mirror_probe import ObservationReflection
        torch.set_num_threads(1)
        run = ROOT/'sim/rl/runs/l80_t1'
        args = json.loads((run/'args.json').read_text())
        env = RunEnv(**env_kwargs(args), randomize=False)
        old, vn = load(str(run/'model.zip'), env)
        try:
            new = copy.deepcopy(old)
            self.assertFalse(widen_command_inputs(vn, new, args['cmd_final']))
            scale = env.scale.copy()
            yaw = [j for j, i in enumerate(env.policy_idx) if 'hip_yaw' in env.active[i]]
            scale[env.policy_idx[yaw]] = .5
            before = old.policy.log_std.detach().clone()
            f = rescale_action_outputs(new, env.active, env.policy_idx, args, scale)
            self.assertEqual(len(yaw), 2)
            np.testing.assert_allclose(f[yaw], .6)
            torch.testing.assert_close(new.policy.log_std, before+torch.tensor(np.log(f), dtype=before.dtype))
            freeze_obs(vn)
            sym = reflection_constants(ObservationReflection(env.active, env.policy_idx, env.q0, env.obs_vel,
                env.obs_contact, len(env.scan_dx), bool(env.obs_hold)), vn)
            rebuilt = PPO(SymPolicy, vn, policy_kwargs={**new.policy_kwargs, 'sym': sym}, device='cpu')
            rebuilt.policy.load_state_dict(new.policy.state_dict(), strict=False)
            obs = torch.tensor(np.random.default_rng(812).normal(size=(8192, env.observation_space.shape[0])), dtype=torch.float32)
            with torch.no_grad():
                a = old.policy.get_distribution(obs).distribution.mean.numpy()
                b = rebuilt.policy.get_distribution(obs).distribution.mean.numpy()
            np.testing.assert_allclose(b, a*f, atol=2e-6, rtol=2e-6)
            valid = np.all(np.abs(a) < 1., axis=1)
            self.assertGreater(int(valid.sum()), 20)
            q0 = env.q0[env.policy_idx]
            old_targets = q0+env.scale[env.policy_idx]*np.clip(a[valid], -1, 1)
            new_targets = q0+scale[env.policy_idx]*np.clip(b[valid], -1, 1)
            np.testing.assert_allclose(new_targets, old_targets, atol=1e-6, rtol=1e-6)
            # A saturated mean is a counterexample to unconditional preservation.
            saturated = np.abs(a[:, yaw]) > 1.1
            delta = np.abs(.3*np.clip(a[:, yaw], -1, 1)-.5*np.clip(b[:, yaw], -1, 1))
            self.assertTrue(np.any(saturated))
            self.assertGreater(float(delta[saturated].max()), .02)
            untouched = [i for i in range(len(f)) if i not in yaw]
            torch.testing.assert_close(new.policy.action_net.weight[untouched], old.policy.action_net.weight[untouched], rtol=0, atol=0)
        finally:
            vn.close()


if __name__ == '__main__':
    unittest.main()
