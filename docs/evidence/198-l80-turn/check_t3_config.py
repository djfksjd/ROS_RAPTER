"""Check reward-only t3 and paired six-second rollouts; no learn(), no runs/ writes."""
import contextlib
import hashlib
import io
import json
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'sim/rl'))
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from run_env import RunEnv
from sym_policy import SymPolicy, freeze_obs
from train_run import budget_chunk, env_kwargs, rescale_action_outputs, widen_command_inputs

EVIDENCE = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(1)
    old = (EVIDENCE/'train_l80_t2.sh').read_text()
    new = (EVIDENCE/'train_l80_t3.sh').read_text()
    expected = old.replace('NAME=${NAME:-l80_t2}', 'NAME=${NAME:-l80_t3}').replace(
        '--init sim/rl/runs/l80_t1/model.zip', '--init sim/rl/runs/l80_t2/model.zip').replace(
        '"track_yaw": 4', '"track_yaw": 8')
    assert new == expected, 't3 changed more than name, init, track_yaw'
    argv = shlex.split(new[new.index('exec "$PY"'):].replace('\\\n', ' '))
    source = ROOT/argv[argv.index('--init')+1]
    files = [source, source.with_name('vecnorm.pkl'), source.with_name('args.json')]
    files += [ROOT/'sim/rl'/n for n in ('train_run.py', 'run_env.py', 'sym_policy.py')]
    def digest():
        return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    before = digest()
    args = json.loads(source.with_name('args.json').read_text())
    weights = json.loads(shlex.split(next(line[2:] for line in new.splitlines() if line.startswith('W=')))[0])
    old_weights = json.loads(args['weights'])
    assert old_weights['track_yaw'] == 4
    assert weights == {**old_weights, 'track_yaw': 8}
    assert json.loads(args['cmd_mix']) == json.loads((EVIDENCE/'commands.json').read_text())
    assert json.loads(argv[argv.index('--scale-override')+1]) == json.loads(args['scale_override']) == {'hip_roll': .5, 'hip_yaw': .5}
    budgets, done = [], 0
    while n := budget_chunk(int(argv[argv.index('--steps')+1])-done, 8192):
        budgets.append(n); done += n
    assert done == 3_997_696 and budgets == [999424]*4
    paired = []
    normalization = None
    for yaw in (.5, -.5):
        common = {**env_kwargs(args), 'level': 0., 'episode_s': 20., 'disturb_items': [],
                  'roll_impulse': None, 'yaw_impulse': None, 'cmd_max': [0., 0., 0.],
                  'cmd_mix': None, 'init_speed': False}
        envs = [RunEnv(**{**common, 'weights': w}, randomize=False, seed=7200) for w in (old_weights, weights)]
        for env in envs:
            env.init_seed = 7200
        vns = [VecNormalize.load(str(source.with_name('vecnorm.pkl')), DummyVecEnv([lambda e=e: e])) for e in envs]
        try:
            vn = vns[0]
            assert vn.training and vn.norm_reward
            initial_count = float(vn.ret_rms.count)
            std = float(np.sqrt(vn.ret_rms.var+vn.epsilon))
            normalization = dict(training=vn.training, norm_reward=vn.norm_reward, norm_obs=vn.norm_obs,
                obs_rms_class=type(vn.obs_rms).__name__, ret_mean=float(vn.ret_rms.mean),
                ret_var=float(vn.ret_rms.var), ret_count=initial_count, reward_std=std,
                gamma=vn.gamma, clip_reward=vn.clip_reward, max_added_raw_reward=.08,
                max_added_normalized_reward_at_start=.08/std,
                max_added_discounted_return_fixed_scale=.08/std/(1-vn.gamma),
                new_sample_fraction_after_budget=done/(initial_count+done))
            # Match the training resume branch, not evaluate.load (which disables reward normalization).
            model = PPO.load(str(source), env=vn, device='cpu')
            original = {k: v.detach().clone() for k, v in model.policy.state_dict().items()}
            with contextlib.redirect_stdout(io.StringIO()):
                widened = widen_command_inputs(vn, model, args['cmd_final'])
                factors = rescale_action_outputs(model, envs[0].active, envs[0].policy_idx, args, envs[0].scale, .15)
            freeze_obs(vn)
            assert isinstance(model.policy, SymPolicy) and not widened
            np.testing.assert_array_equal(factors, np.ones(10))
            for k,v in model.policy.state_dict().items():
                torch.testing.assert_close(v, original[k], rtol=0, atol=0)
            with torch.no_grad():
                model.policy.log_std.fill_(float(np.log(.15)))
            for k,v in model.policy.state_dict().items():
                if k != 'log_std':
                    torch.testing.assert_close(v, original[k], rtol=0, atol=0)
            assert model.clip_range_vf is None and model.normalize_advantage
            normalization.update(clip_range_vf=None, normalize_advantage=True, vf_coef=model.vf_coef,
                                 critic_warmup=False, actor_critic_parameters_preserved=True)
            obs_stats = [(v.obs_rms.mean.copy(), v.obs_rms.var.copy(), v.obs_rms.count) for v in vns]
            raw_obs=[]
            for v,e in zip(vns,envs):
                # PPO.load(env=vn) schedules the saved training seed on its vector env.
                # Override BOTH comparison resets, including the second wrapper without a model.
                v.seed(7200)
                v.reset(); e.resample_steps=0
                raw_obs.append(v.get_original_obs()[0].copy())
            max_reward_error = 0.
            max_normalized_reward = 0.
            max_added_reward = 0.
            for k in range(300):
                actions=[]
                for v,e,raw in zip(vns,envs,raw_obs):
                    e.command = np.array([4., 0., yaw if k >= 250 else 0.])
                    raw[6:9] = e.command*np.array([.1, 2., .5])
                    actions.append(model.predict(v.normalize_obs(raw[None]), deterministic=True)[0])
                np.testing.assert_array_equal(actions[0], actions[1])
                results = [v.step(action) for v,action in zip(vns,actions)]
                raw_obs=[v.get_original_obs()[0].copy() for v in vns]
                np.testing.assert_array_equal(results[0][0],results[1][0])
                np.testing.assert_array_equal(envs[0].data.qpos,envs[1].data.qpos)
                np.testing.assert_array_equal(envs[0].data.qvel,envs[1].data.qvel)
                assert not any(bool(r[2][0]) for r in results)
                raw_rewards = [float(v.get_original_reward()[0]) for v in vns]
                expected_delta = .08*results[0][3][0]['terms']['track_yaw']
                max_reward_error=max(max_reward_error,abs(raw_rewards[1]-raw_rewards[0]-expected_delta))
                max_added_reward=max(max_added_reward,raw_rewards[1]-raw_rewards[0])
                max_normalized_reward=max(max_normalized_reward,*[abs(float(r[1][0])) for r in results])
            # DummyVecEnv stores rewards in float32; avoid pretending its round-off is a physical difference.
            assert max_reward_error < 1e-6
            assert max_normalized_reward < vn.clip_reward
            for v,stats in zip(vns,obs_stats):
                np.testing.assert_array_equal(v.obs_rms.mean,stats[0])
                np.testing.assert_array_equal(v.obs_rms.var,stats[1])
                assert v.obs_rms.count == stats[2] and v.ret_rms.count == initial_count+300
            paired.append(dict(speed=4., yaw=yaw, seed=7200, steps_per_environment=300,
                state_observation_action_exact=True, falls=0, reward_delta_max_error=max_reward_error,
                max_added_raw_reward=max_added_reward, max_abs_normalized_reward=max_normalized_reward,
                ret_stats_updated=True, obs_stats_unchanged=True))
        finally:
            for v in vns:
                v.close()
    after=digest()
    assert before==after
    assert not (ROOT/'sim/rl/runs/l80_t3').exists()
    print(json.dumps(dict(script_changes_only=['name','init','track_yaw 4 -> 8'],
        source=str(source.relative_to(ROOT)), all_other_weights_equal=True, commands_equal=True,
        scale_override={'hip_roll':.5,'hip_yaw':.5}, resume_action_std=.15,
        blocks=budgets, actual_budget=done, normalization=normalization, paired_rollouts=paired,
        training_started=False, before=before, after=after, inputs_unchanged=True),indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
