"""PPO training for RaptorEnv (Stable-Baselines3, CPU, local only).

Curriculum: terrain level and command range grow when the recent fall rate is low.
Usage: .venv-sim/bin/python sim/rl/train.py --name walk_flat --terrain flat --steps 10e6
Outputs go to sim/rl/runs/<name>/ (git-ignored): model.zip, vecnorm.pkl, progress.csv, curriculum.jsonl.
"""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.logger import configure  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize  # noqa: E402

from raptor_env import RaptorEnv  # noqa: E402


class Curriculum(BaseCallback):
    """Raise level/command range when fewer than 20% of recent episodes end in a fall."""

    def __init__(self, path, start_level, cmd_max, cmd_final, every=200_000):
        super().__init__()
        self.path, self.level, self.every = path, start_level, every
        self.cmd, self.cmd_final = np.array(cmd_max, float), np.array(cmd_final, float)
        self.falls, self.ends, self.last = 0, 0, 0

    def _on_step(self):
        for done, info in zip(self.locals['dones'], self.locals['infos']):
            if done:
                self.ends += 1
                self.falls += not info.get('TimeLimit.truncated', False)
        if self.num_timesteps-self.last >= self.every and self.ends >= 20:
            rate = self.falls/self.ends
            if rate < .2:
                self.level = min(1., self.level+.1)
                self.cmd = np.minimum(self.cmd_final, self.cmd*1.15)
            elif rate > .5:
                self.level = max(0., self.level-.1)
            self.training_env.env_method('set_curriculum', self.level, self.cmd.tolist())
            with open(self.path, 'a') as f:
                f.write(json.dumps({'steps': self.num_timesteps, 'fall_rate': round(rate, 3),
                                    'level': round(self.level, 2), 'cmd_max': self.cmd.round(3).tolist()})+'\n')
            self.logger.record('curriculum/level', self.level)
            self.logger.record('curriculum/fall_rate', rate)
            self.falls = self.ends = 0
            self.last = self.num_timesteps
        return True


def make(terrain, level, cmd, vel_scale, seed, actuator='urdf', dof=10, sole='flat', weights=None, zero_cmd=.1):
    def thunk():
        return RaptorEnv(terrain=terrain, level=level, cmd_max=cmd, vel_scale=vel_scale, seed=seed, actuator=actuator,
                         dof=dof, sole=sole, weights=weights, zero_cmd=zero_cmd)
    return thunk


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--terrain', nargs='+', default=['flat'])
    p.add_argument('--steps', type=float, default=10e6)
    p.add_argument('--envs', type=int, default=8)
    p.add_argument('--level', type=float, default=0.)
    p.add_argument('--cmd', type=float, nargs=3, default=[.2, .1, .3])
    p.add_argument('--cmd-final', type=float, nargs=3, default=[.6, .2, .6])
    p.add_argument('--vel-scale', type=float, default=1.)
    p.add_argument('--actuator', default='urdf', help='joint speed spec: urdf or r01a (hypothetical)')
    p.add_argument('--dof', type=int, choices=[10, 12], default=10, help='12 = with ankle roll')
    p.add_argument('--sole', choices=['flat', 'rocker'], default='flat')
    p.add_argument('--weights', default='{}', help='JSON reward-weight overrides, e.g. {"track_lin": 4}')
    p.add_argument('--zero-cmd', type=float, default=.1, help='fraction of zero (stand) commands')
    p.add_argument('--init', help='model.zip to continue from (its vecnorm.pkl is loaded too)')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    out = HERE/'runs'/a.name
    out.mkdir(parents=True, exist_ok=True)
    (out/'args.json').write_text(json.dumps(vars(a), indent=1))
    torch.set_num_threads(1)
    env = VecMonitor(SubprocVecEnv([make(a.terrain, a.level, a.cmd, a.vel_scale, a.seed*100+i, a.actuator, a.dof, a.sole,
                                          json.loads(a.weights), a.zero_cmd) for i in range(a.envs)]))
    if a.init:
        env = VecNormalize.load(str(Path(a.init).with_name('vecnorm.pkl')), env)
        model = PPO.load(a.init, env=env, device='cpu')
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., gamma=.99)
        model = PPO('MlpPolicy', env, n_steps=1024, batch_size=4096, n_epochs=5, learning_rate=3e-4,
                    gamma=.99, gae_lambda=.95, clip_range=.2, ent_coef=.003, max_grad_norm=1., device='cpu',
                    policy_kwargs=dict(net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64]),
                                       activation_fn=torch.nn.ELU, log_std_init=-2.),
                    seed=a.seed, verbose=0)
    model.set_logger(configure(str(out), ['csv']))
    cb = Curriculum(out/'curriculum.jsonl', a.level, a.cmd, a.cmd_final)
    chunk = 1_000_000
    done = 0
    while done < a.steps:
        n = int(min(chunk, a.steps-done))
        model.learn(n, callback=cb, reset_num_timesteps=False, progress_bar=False)
        done += n
        model.save(out/'model.zip')
        env.save(str(out/'vecnorm.pkl'))
        print(json.dumps({'steps': model.num_timesteps, 'level': cb.level, 'cmd': cb.cmd.round(3).tolist()}), flush=True)
    env.close()


if __name__ == '__main__':
    main()
