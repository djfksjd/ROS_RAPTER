"""Counterfactual sagittal reflection diagnosis; does not execute reflected actions.

Reflect raw observations before frozen VecNormalize. Exchange legs and shift the
contact clock by half a cycle. This is a policy equivariance check, not proof of
full dynamics/reward symmetry or a trained symmetric locomotion controller.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def joint_reflection(names):
    names = list(names)
    if len(set(names)) != len(names):
        raise ValueError('duplicate joint names')
    index, sign = [], []
    for name in names:
        if name.startswith('left_'):
            other = 'right_' + name[5:]
        elif name.startswith('right_'):
            other = 'left_' + name[6:]
        elif name in ('tail_yaw_joint', 'tail_pitch_joint'):
            other = name
        else:
            raise ValueError(f'unsupported joint: {name}')
        if not name.endswith(('_roll_joint', '_pitch_joint', '_yaw_joint')):
            raise ValueError(f'unsupported axis: {name}')
        if other not in names:
            raise ValueError(f'missing mirrored joint: {other}')
        index.append(names.index(other))
        sign.append(-1. if name.endswith(('_roll_joint', '_yaw_joint')) else 1.)
    return np.array(index), np.array(sign)


class ObservationReflection:
    def __init__(self, active, policy_idx, q0, obs_vel, obs_contact=False):
        self.joint_index, self.joint_sign = joint_reflection(active)
        self.action_index, self.action_sign = joint_reflection([active[i] for i in policy_idx])
        self.q0 = np.asarray(q0)
        self.obs_vel = obs_vel
        self.obs_contact = obs_contact
        self.n, self.na = len(active), len(policy_idx)
        self.size = 9 + 3*int(obs_vel) + 2*self.n + self.na + 2 + 2*int(obs_contact)

    def action(self, action):
        a = np.asarray(action)
        if a.shape[-1] != self.na:
            raise ValueError('action dimension mismatch')
        return a[..., self.action_index]*self.action_sign

    def observation(self, observation):
        x = np.asarray(observation)
        if x.shape[-1] != self.size:
            raise ValueError('observation dimension mismatch')
        y = x.copy()
        y[..., :3] *= [-1, 1, -1]  # angular velocity, axial vector
        y[..., 3:6] *= [1, -1, 1]  # gravity, polar vector
        y[..., 6:9] *= [1, -1, -1]  # vx, vy, yaw command
        start = 9
        if self.obs_vel:
            y[..., 9:12] *= [1, -1, 1]
            start += 3
        # Joint observation is q-q0, so reflect absolute q before subtracting q0.
        y[..., start:start+self.n] = ((x[..., start:start+self.n]+self.q0)[..., self.joint_index]
                                      *self.joint_sign-self.q0)
        start += self.n
        y[..., start:start+self.n] = x[..., start:start+self.n][..., self.joint_index]*self.joint_sign
        start += self.n
        y[..., start:start+self.na] = self.action(x[..., start:start+self.na])
        start += self.na
        y[..., start:start+2] *= -1  # phase += 0.5: sine and cosine change sign
        if self.obs_contact:
            y[..., start+2:start+4] = x[..., start+2:start+4][..., [1,0]]
        return y


def symmetric_action(model, norm, reflection, raw):
    """Inference average only; this is not PPO augmentation or retraining."""
    raw = np.asarray(raw)
    a, _ = model.predict(norm.normalize_obs(raw[None]), deterministic=True)
    b, _ = model.predict(norm.normalize_obs(reflection.observation(raw)[None]), deterministic=True)
    return .5*(a[0]+reflection.action(b[0]))


def probe(run, seed):
    from evaluate import load
    from run_env import RunEnv
    from train_run import env_kwargs
    run = Path(run)
    args = json.loads((run/'args.json').read_text())
    env = RunEnv(**{**env_kwargs(args), 'episode_s': 12., 'level': 0.}, randomize=False, seed=7)
    env.init_seed = seed
    model, norm = load(str(run/'model.zip'), env)
    reflection = ObservationReflection(env.active, env.policy_idx, env.q0, env.obs_vel,env.obs_contact)
    rows = []
    try:
        norm.reset()
        env.command = np.array([4., 0., 0.]); env.resample_steps = 0
        fell = False
        for k in range(400):
            raw = env._obs()
            actual, _ = model.predict(norm.normalize_obs(raw[None]), deterministic=True)
            if k >= 200:
                for yaw in [-1., 0., 1.]:
                    env.command[2] = yaw
                    obs = env._obs()
                    mirrored = reflection.observation(obs)
                    a, _ = model.predict(norm.normalize_obs(obs[None]), deterministic=True)
                    b, _ = model.predict(norm.normalize_obs(mirrored[None]), deterministic=True)
                    returned = reflection.action(b[0])
                    rows.append(dict(t=k*.02, yaw=yaw, phase=float(env.phase),
                                     action=a[0].tolist(), reflected_action_returned=returned.tolist(),
                                     residual=(a[0]-returned).tolist()))
                env.command[2] = 0.
            _, _, fell, _, _ = env.step(actual[0])
            if fell:
                break
        summary = {}
        for yaw in [-1., 0., 1.]:
            selected = [r for r in rows if r['yaw'] == yaw]
            if selected:
                residual = np.array([r['residual'] for r in selected])
                summary[str(yaw)] = dict(states=len(selected), rms=np.sqrt(np.mean(residual**2, axis=0)).tolist(),
                                         mean_abs=np.mean(np.abs(residual), axis=0).tolist())
        return dict(run=str(run), initial_seed=seed, fell=bool(fell), time_s=float(env.data.time),
                    policy_joints=[env.active[i] for i in env.policy_idx], summary=summary, rows=rows,
                    counterfactual_actions_executed=False, full_dynamics_symmetry_verified=False)
    finally:
        norm.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--seeds', type=int, nargs='+', default=[14001, 14002])
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    path = Path(args.out); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        for run in args.runs:
            for seed in args.seeds:
                result = probe(run, seed)
                f.write(json.dumps(result)+'\n'); f.flush()
                print(json.dumps({k:v for k,v in result.items() if k != 'rows'}), flush=True)


if __name__ == '__main__':
    main()
