"""PPO training for RunEnv (T1 running, Stable-Baselines3, CPU, local only).

Curriculum (same rule as train.py): when fewer than 20 % of recent episodes end in a fall the commanded speed range
grows x1.15 towards --cmd-final and the disturbance level rises +0.1; above 50 % the level drops.
Usage: .venv-sim/bin/python sim/rl/train_run.py --name run_t1 --steps 20e6
Outputs go to sim/rl/runs/<name>/ (git-ignored): model.zip, vecnorm.pkl, progress.csv, curriculum.jsonl.
Every result is a T1 (virtual actuator) result: report it with the actuator spec of run_env.ACTUATOR_T1.
"""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.logger import configure  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize  # noqa: E402

from recover_env import RecoverEnv  # noqa: E402
from run_env import RunEnv  # noqa: E402
from train import Curriculum  # noqa: E402


def make(kw, seed, cls=RunEnv):
    def thunk():
        return cls(seed=seed, **kw)
    return thunk


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--env', choices=['run', 'recover'], default='run', help='recover = get-up task (recover_env.py)')
    p.add_argument('--steps', type=float, default=20e6)
    p.add_argument('--envs', type=int, default=8)
    p.add_argument('--mass', type=float, default=5.)
    p.add_argument('--springs', choices=['none', 'b', 'c'], default='c')
    p.add_argument('--tail', choices=['active', 'locked'], default='active')
    p.add_argument('--level', type=float, default=0., help='disturbance level at start (0..1)')
    p.add_argument('--cmd', type=float, nargs=3, default=[2., .2, .5])
    p.add_argument('--cmd-final', type=float, nargs=3, default=[11.1, .3, .8])
    p.add_argument('--weights', default='{}', help='JSON reward-weight overrides')
    p.add_argument('--zero-cmd', type=float, default=.1)
    p.add_argument('--top-cmd', type=float, default=.3, help='fraction of commands at the current maximum speed')
    p.add_argument('--episode-s', type=float, default=10.)
    p.add_argument('--disturb-items', nargs='+', help='subset of pitch yaw trip touchdown push')
    p.add_argument('--no-obs-vel', action='store_true', help='drop the body velocity from the observation')
    p.add_argument('--no-init-speed', action='store_true')
    p.add_argument('--init', help='model.zip to continue from (its vecnorm.pkl is loaded too)')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    out = HERE/'runs'/a.name
    out.mkdir(parents=True, exist_ok=True)
    (out/'args.json').write_text(json.dumps(vars(a), indent=1))
    kw = dict(mass=a.mass, springs=a.springs, tail=a.tail, level=a.level, cmd_max=a.cmd, episode_s=a.episode_s,
              weights=json.loads(a.weights), zero_cmd=a.zero_cmd, top_cmd=a.top_cmd, disturb_items=a.disturb_items,
              obs_vel=not a.no_obs_vel, init_speed=not a.no_init_speed)
    if a.env == 'recover':
        kw = dict(mass=a.mass, springs=a.springs, tail=a.tail, episode_s=a.episode_s, weights=json.loads(a.weights))
    torch.set_num_threads(1)
    env = VecMonitor(SubprocVecEnv([make(kw, a.seed*100+i, RecoverEnv if a.env == 'recover' else RunEnv) for i in range(a.envs)]))
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
    cb = Curriculum(out/'curriculum.jsonl', a.level, a.cmd, a.cmd_final) if a.env == 'run' else None
    chunk, done = 1_000_000, 0
    while done < a.steps:
        n = int(min(chunk, a.steps-done))
        model.learn(n, callback=cb, reset_num_timesteps=False, progress_bar=False)
        done += n
        model.save(out/'model.zip')
        env.save(str(out/'vecnorm.pkl'))
        print(json.dumps({'steps': model.num_timesteps, **({'level': round(cb.level, 2), 'cmd': cb.cmd.round(3).tolist()} if cb else {})}), flush=True)
    env.close()


if __name__ == '__main__':
    main()
