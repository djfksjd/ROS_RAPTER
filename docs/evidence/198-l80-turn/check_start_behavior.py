"""Bounded non-RL comparison: t1 vs the requested linear action rescaling.

Two fixed seeds, 5 s straight + 10 s yaw .5 at 4 m/s, deterministic policy.
No learning, checkpoint writes, or changes to RunEnv. Exit 1 if behaviour differs.
"""
import argparse
import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'sim/rl'))
import numpy as np
import torch
from evaluate import load
from run_env import RunEnv
from train_run import env_kwargs, widen_command_inputs, rescale_action_outputs


def rollout(run, seed, scale_override=None):
    args = json.loads((run/'args.json').read_text())
    config = args if scale_override is None else {**args, 'scale_override': json.dumps(scale_override)}
    kw = {**env_kwargs(config), 'level': 0., 'episode_s': 20., 'disturb_items': [],
          'roll_impulse': None, 'yaw_impulse': None, 'cmd_max': [0., 0., 0.],
          'cmd_mix': None, 'init_speed': False}
    env = RunEnv(**kw, randomize=False, seed=seed)
    env.init_seed = seed
    model, vn = load(str(run/'model.zip'), env)
    reference = None
    if scale_override is not None:
        from stable_baselines3 import PPO
        reference = PPO.load(run/'model.zip', device='cpu')
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            assert not widen_command_inputs(vn, model, args['cmd_final'])
            factors = rescale_action_outputs(model, env.active, env.policy_idx, args, env.scale, .15)
        with torch.no_grad():
            model.policy.log_std.fill_(float(np.log(.15)))
    vn.reset()
    env.resample_steps = 0
    raw, fell = env._obs(), None
    trace, deltas, saturated = [], [], []
    yaw = [j for j, i in enumerate(env.policy_idx) if 'hip_yaw' in env.active[i]]
    try:
        for k in range(750):
            env.command = np.array([4., 0., .5 if k >= 250 else 0.])
            raw[6:9] = env.command*np.array([.1, 2., .5])
            obs = vn.normalize_obs(raw[None])
            action = model.predict(obs, deterministic=True)[0][0]
            if reference is not None:
                old_action = reference.predict(obs, deterministic=True)[0][0]
                deltas.append(float(np.max(np.abs(.3*old_action[yaw]-.5*action[yaw]))))
                saturated.append(float(np.mean(np.abs(old_action[yaw]) >= 1.)))
            raw, _, term, trunc, info = env.step(action)
            w, v = env.body_velocity(env.base)
            trace.append([env.data.time, *v, *w])
            if term or trunc:
                fell = float(env.data.time) if term else None
                break
    finally:
        vn.close()
    data = np.asarray(trace)
    steady = data[data[:, 0] > 8.+1e-6]
    row = dict(seed=seed, variant='t1' if scale_override is None else 'rescaled',
        steps=len(data), fell_at_s=fell, vx=float(steady[:, 1].mean()) if len(steady) else None,
        yaw_rate=float(steady[:, 6].mean()) if len(steady) else None)
    if reference is not None:
        row.update(factors=factors.tolist(), rescale_log=captured.getvalue().splitlines(),
            same_observation_target_max_delta_rad=max(deltas), old_yaw_saturated_share=float(np.mean(saturated)))
    return row, data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scale-override', default='{"hip_roll": 0.5, "hip_yaw": 0.5}')
    p.add_argument('--out', type=Path, default=Path(__file__).resolve().parent/'eval/t2_start_behavior.json')
    a = p.parse_args()
    torch.set_num_threads(1)
    run = ROOT/'sim/rl/runs/l80_t1'
    paths = [run/name for name in ('model.zip', 'vecnorm.pkl', 'args.json')]
    digest = lambda: {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in paths}
    before = digest()
    # Declared before measurement: same deterministic conditions; tolerate .01 rad/s yaw,
    # .05 m/s speed and require both to finish. This is not a stage-4 gate.
    rows, comparisons = [], []
    for seed in (7200, 7201):
        old, old_trace = rollout(run, seed)
        new, new_trace = rollout(run, seed, json.loads(a.scale_override))
        rows.extend([old, new])
        dy = abs(new['yaw_rate']-old['yaw_rate']) if new['yaw_rate'] is not None and old['yaw_rate'] is not None else None
        dv = abs(new['vx']-old['vx']) if new['vx'] is not None and old['vx'] is not None else None
        ok = old['steps'] == new['steps'] == 750 and old['fell_at_s'] is new['fell_at_s'] is None and dy <= .01 and dv <= .05
        comparisons.append(dict(seed=seed, yaw_abs_delta=dy, vx_abs_delta=dv, passed=ok))
        print(json.dumps(comparisons[-1]), flush=True)
        np.savez_compressed(a.out.with_name(f't2_start_s{seed}.npz'),
            baseline=old_trace, rescaled=new_trace, columns=['time', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz'])
    after = digest()
    assert before == after
    result = dict(training_started=False, scale_override=json.loads(a.scale_override),
        command=[4., 0., .5], straight_s=5, turn_s=10, steady_window_s=[8, 15],
        randomize=False, disturbances=False, deterministic=True, resume_action_std=.15,
        tolerance=dict(yaw_rad_s=.01, vx_m_s=.05), rows=rows, comparisons=comparisons,
        passed=all(c['passed'] for c in comparisons), source_sha256=before, source_unchanged=before == after)
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
