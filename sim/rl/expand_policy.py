"""Warm-start a 12-DOF (ankle roll) policy from a 10-DOF one.

The network input/output grow by the two ankle-roll joints: new input weights are zero (the policy
initially ignores roll readings), new action rows are zero (roll targets start at the default 0) and
their log-std copies the ankle-pitch value. VecNormalize statistics are expanded the same way.
At the start the 12-DOF policy therefore acts like the 10-DOF one with the ankle rolls held at 0.
Usage: .venv-sim/bin/python sim/rl/expand_policy.py runs/SRC/model.zip runs/DST/model.zip
"""
import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from raptor_env import RaptorEnv  # noqa: E402


def index_map():
    old, new = RaptorEnv(dof=10).active, RaptorEnv(dof=12).active
    return [new.index(n) for n in old], len(old), len(new)


def obs_map(joint_map, n_old, n_new):
    """Old observation index -> new index (layout: gyro, gravity, command, q, qd, last action, clock)."""
    head = list(range(9))
    blocks = [9+k*n_new+j for k in range(3) for j in joint_map]
    clock = [9+3*n_new, 10+3*n_new]
    return head+blocks+clock, 11+3*n_new


def main(src, dst):
    joints, n_old, n_new = index_map()
    omap, obs_new = obs_map(joints, n_old, n_new)
    model = PPO.load(src, device='cpu')
    p = model.policy
    new = PPO('MlpPolicy', RaptorEnv(dof=12), policy_kwargs=model.policy_kwargs, device='cpu',
              **{k: getattr(model, k) for k in ('n_steps', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda',
                                                  'ent_coef', 'max_grad_norm')},
              learning_rate=3e-4, clip_range=.2)
    q = new.policy
    with torch.no_grad():
        for name, param in p.state_dict().items():
            target = q.state_dict()[name]
            if name in ('mlp_extractor.policy_net.0.weight', 'mlp_extractor.value_net.0.weight'):
                target.zero_(); target[:, omap] = param
            elif name == 'action_net.weight':
                target.zero_(); target[joints] = param
            elif name == 'action_net.bias':
                target.zero_(); target[joints] = param
            elif name == 'log_std':
                pitch = [i for i, n in enumerate(RaptorEnv(dof=12).active) if 'ankle_pitch' in n]
                target[joints] = param
                target[[i+1 for i in pitch]] = param[[joints.index(i) for i in pitch]]
            else:
                target.copy_(param)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    new.save(dst)
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    old = VecNormalize.load(str(Path(src).with_name('vecnorm.pkl')), DummyVecEnv([lambda: RaptorEnv(dof=10)]))
    norm = VecNormalize(DummyVecEnv([lambda: RaptorEnv(dof=12)]), clip_obs=old.clip_obs, gamma=old.gamma)
    mean, var = np.zeros(obs_new), np.ones(obs_new)
    mean[omap], var[omap] = old.obs_rms.mean, old.obs_rms.var
    roll_q = [9+j for j in range(n_new) if j not in joints]
    var[roll_q] = .01  # roll angles are small; keeps the (zero-weighted) inputs well scaled once learned
    norm.obs_rms.mean, norm.obs_rms.var, norm.obs_rms.count = mean, var, old.obs_rms.count
    norm.ret_rms = old.ret_rms
    norm.save(str(Path(dst).with_name('vecnorm.pkl')))
    print(f'expanded {src} ({n_old} joints) -> {dst} ({n_new} joints)')


if __name__ == '__main__':
    main(*sys.argv[1:3])
