"""Export an SB3 PPO policy + VecNormalize to a numpy .npz for the ROS runner (rl_policy.py).

Usage: .venv-sim/bin/python sim/rl/export_policy.py runs/NAME/model.zip OUT.npz --dof 12 --sole rocker
"""
import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from raptor_env import CONTROL_DT, RaptorEnv  # noqa: E402

CLOCK_HZ = 1.6  # raptor_env.step advances the phase by CONTROL_DT*1.6


def export(model_path, out, dof=12, sole='flat'):
    env = RaptorEnv(dof=dof, sole=sole, randomize=False)
    env.reset()
    model = PPO.load(model_path, device='cpu')
    norm = VecNormalize.load(str(Path(model_path).with_name('vecnorm.pkl')), DummyVecEnv([lambda: env]))
    sd = {k: v.numpy() for k, v in model.policy.state_dict().items()}
    names = sorted({k.rsplit('.', 1)[0] for k in sd if k.startswith('mlp_extractor.policy_net')},
                   key=lambda n: int(n.rsplit('.', 1)[1]))+['action_net']
    arrays = {f'w{i}': sd[f'{n}.weight'] for i, n in enumerate(names)}
    arrays |= {f'b{i}': sd[f'{n}.bias'] for i, n in enumerate(names)}
    np.savez(out, n_layers=len(names), obs_mean=norm.obs_rms.mean, obs_var=norm.obs_rms.var, obs_clip=norm.clip_obs,
             joints=np.array(env.active), q0=env.q0, action_scale=env.scale, joint_lo=env.lo, joint_hi=env.hi,
             clock_hz=CLOCK_HZ, control_dt=CONTROL_DT, **arrays)
    return env, model, norm


def main():
    p = argparse.ArgumentParser()
    p.add_argument('model')
    p.add_argument('out')
    p.add_argument('--dof', type=int, default=12)
    p.add_argument('--sole', default='flat')
    a = p.parse_args()
    export(a.model, a.out, a.dof, a.sole)
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
