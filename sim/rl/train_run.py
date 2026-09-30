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
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
import numpy as np  # noqa: E402


class EvalCurriculum(BaseCallback):
    """Advance on deterministic evaluation, not on the stochastic training fall rate (Codex review, evidence 86 §11).
    Every `every` steps: `n` episodes of `seconds` s at the current maximum forward speed, with the training
    randomisation and disturbances kept, actions deterministic. Pass = at most one fall, mean speed >= 80 % of the
    command and left/right touchdown phase within 0.5 +- 0.15. Speed rises first (x1.15 per pass) at a fixed
    disturbance level; the level rises (+0.1 per pass) only once the speed has reached its final value."""

    def __init__(self, path, kw, level, cmd, cmd_final, every=300_000, n=4, seconds=8.):
        super().__init__()
        self.path, self.kw, self.every, self.n, self.seconds = path, kw, every, n, seconds
        self.level, self.cmd, self.cmd_final = level, np.array(cmd, float), np.array(cmd_final, float)
        self.last = 0

    def _evaluate(self):
        env = RunEnv(seed=int(self.num_timesteps) % 100000, **{**self.kw, 'level': self.level, 'cmd_max': self.cmd.tolist()})
        venv = self.training_env
        falls, speeds, phases = 0, [], []
        for ep in range(self.n):
            env.reset(seed=int(self.num_timesteps)+ep)
            env.command = np.array([self.cmd[0], 0., 0.]); env.resample_steps = 0
            obs = venv.normalize_obs(env._obs()[None])
            vx, tl, tr = [], [], []
            for k in range(int(self.seconds/.02)):
                a, _ = self.model.predict(obs, deterministic=True)
                raw, _, term, _, info = env.step(a[0])
                obs = venv.normalize_obs(raw[None])
                if k > 100:
                    vx.append(info['v_body'][0])
                if term:
                    falls += 1
                    break
            speeds.append(np.mean(vx) if vx else 0.)
            L = [t for t, s_ in env.touchdowns if s_ == 'left' and t > 2.]; R = [t for t, s_ in env.touchdowns if s_ == 'right' and t > 2.]
            if len(L) > 2 and R:
                T = np.mean(np.diff(L))
                phases += [((r-max(x for x in L if x <= r))/T) % 1. for r in R if any(x <= r for x in L)]
        ph = float(np.median(phases)) if phases else 0.
        ok = falls <= 1 and np.mean(speeds) >= .8*self.cmd[0] and abs(ph-.5) <= .15
        return ok, falls, float(np.mean(speeds)), ph

    def _on_step(self):
        if self.num_timesteps-self.last < self.every:
            return True
        self.last = self.num_timesteps
        ok, falls, v, ph = self._evaluate()
        if ok:
            if self.cmd[0] < self.cmd_final[0]-1e-6:
                self.cmd = np.minimum(self.cmd_final, self.cmd*1.15)
            else:
                self.level = min(1., self.level+.1)
            self.training_env.env_method('set_curriculum', self.level, self.cmd.tolist())
        with open(self.path, 'a') as f:
            f.write(json.dumps({'steps': self.num_timesteps, 'eval_pass': bool(ok), 'eval_falls': falls, 'eval_speed': round(v, 2),
                                'eval_lr_phase': round(ph, 2), 'level': round(self.level, 2), 'cmd_max': self.cmd.round(3).tolist()})+'\n')
        self.logger.record('curriculum/level', self.level)
        self.logger.record('curriculum/speed', self.cmd[0])
        return True


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
    p.add_argument('--ankle-clutch', action='store_true', help='ankle spring engaged only while that foot is loaded')
    p.add_argument('--kin', default='{}', help='JSON overrides of run_env.KIN, e.g. {"fold": [65, 15, 0.1, 0.45]}')
    p.add_argument('--eval-curriculum', action='store_true', help='advance on deterministic evaluation (speed first, then level)')
    p.add_argument('--track-sigma-frac', type=float, default=.15, help='speed-tracking width as a fraction of the command')
    p.add_argument('--springs-override', help='JSON {joint: [k, q0, mode, engaged]} replacing the spring set (spring_refit.py)')
    p.add_argument('--arch8-override', help='JSON {joint: [class, gear]} for the arch8 motors')
    p.add_argument('--arch8', action='store_true', help='8-axis hybrid: four-bar knee, coupled ankle, locked ankle roll, class motor curves')
    p.add_argument('--couple-ankle', action='store_true', help='horse-style knee-ankle coupling, ankle pitch motors removed')
    p.add_argument('--yaw-impulse', type=float, nargs=3, metavar=('RATE', 'LO', 'HI'), help='training yaw impulses independent of the level')
    p.add_argument('--init', help='model.zip to continue from (its vecnorm.pkl is loaded too)')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    out = HERE/'runs'/a.name
    out.mkdir(parents=True, exist_ok=True)
    (out/'args.json').write_text(json.dumps(vars(a), indent=1))
    kw = dict(mass=a.mass, springs=a.springs, tail=a.tail, level=a.level, cmd_max=a.cmd, episode_s=a.episode_s,
              weights=json.loads(a.weights), zero_cmd=a.zero_cmd, top_cmd=a.top_cmd, disturb_items=a.disturb_items,
              obs_vel=not a.no_obs_vel, init_speed=not a.no_init_speed, ankle_clutch=a.ankle_clutch,
              kin=json.loads(a.kin), yaw_impulse=a.yaw_impulse, couple_ankle=a.couple_ankle,
              track_sigma_frac=a.track_sigma_frac, arch8=a.arch8,
              springs_override=json.loads(a.springs_override) if a.springs_override else None,
              arch8_override=json.loads(a.arch8_override) if a.arch8_override else None)
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
    if a.env == 'run' and a.eval_curriculum:
        cb = EvalCurriculum(out/'curriculum.jsonl', {k: v for k, v in kw.items() if k not in ('level', 'cmd_max')}, a.level, a.cmd, a.cmd_final)
    else:
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
