"""PPO training for RunEnv (T1 running, Stable-Baselines3, CPU, local only).

Curriculum (same rule as train.py): when fewer than 20 % of recent episodes end in a fall the commanded speed range
grows x1.15 towards --cmd-final and the disturbance level rises +0.1; above 50 % the level drops.
Usage: .venv-sim/bin/python sim/rl/train_run.py --name run_t1 --steps 20e6
Outputs go to sim/rl/runs/<name>/ (git-ignored): model.zip, vecnorm.pkl, progress.csv, curriculum.jsonl.
Every result is a T1 (virtual actuator) result: report it with the actuator spec of run_env.ACTUATOR_T1.
"""
import argparse
import json
import pickle
import subprocess
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.logger import configure  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize  # noqa: E402

from recover_env import RecoverEnv  # noqa: E402
from run_env import RunEnv, SCALE_RUN, joint_type  # noqa: E402
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


def widen_command_inputs(env, model, cmd_max):
    """Command inputs (obs 6-8, RunEnv._compose scales the command by 0.1, 2, 0.5) whose frozen normalisation cannot
    hold the new command range. (1) A policy trained at a zero command only (stage 1a) has zero variance there: continued
    with commands, every non-zero command became +-10 after clipping and the walking run l80_w1 collapsed (evidence 196).
    (2) A walking policy normalised for 1.5 m/s sees an 11.1 m/s command at 15 sigma, again clipped. Each command input
    whose std is below half its scaled command range gets that std; the first-layer weights of the input are scaled by
    new/old std (zero when the old variance was ~0), so the policy acts exactly as before for every command it could
    already read. The normalisation mean of the input is kept. Returns True if anything changed."""
    rms = env.obs_rms
    want = np.maximum(np.abs(np.asarray(cmd_max, float))*np.array([.1, 2., .5])/2, 1e-3)
    old = np.sqrt(rms.var[6:9]+env.epsilon)
    idx = [6+i for i in range(3) if old[i] < want[i]*(1-1e-9)]
    if not idx:
        return False
    with torch.no_grad():
        for k, w in model.policy.state_dict(keep_vars=True).items():
            if w.ndim == 2 and k.endswith('.0.weight') and w.shape[1] == rms.mean.shape[0]:
                for i in idx:
                    w[:, i] *= 0. if rms.var[i] < 1e-6 else float(want[i-6]/old[i-6])
    for i in idx:
        if rms.var[i] < 1e-6:
            rms.mean[i] = 0.
        rms.var[i] = want[i-6]**2-env.epsilon
    print('widened command inputs', idx, 'std', [round(float(want[i-6]), 3) for i in idx], flush=True)
    return True


def rescale_action_outputs(model, active, policy_idx, old_args, new_scale, resume_action_std=None):
    """Change action coordinates on resume, before rebuilding a symmetric policy.

    The mean rows and Gaussian std get old/new scale. This preserves unbounded
    target angles (same observation and q0), and clipped targets where neither
    action saturates. It cannot preserve old saturated targets: clip(f*a) !=
    f*clip(a). RunEnv also observes last_action in these changed coordinates;
    closed-loop behaviour therefore needs a separate rollout check.
    """
    old_scales = {**SCALE_RUN, **json.loads(old_args.get('scale_override') or '{}')}
    names = [active[i] for i in policy_idx]
    old = np.array([old_scales[joint_type(name)] for name in names], dtype=float)
    new = np.asarray(new_scale, dtype=float)[policy_idx]
    if not (np.all(np.isfinite(old)) and np.all(np.isfinite(new)) and np.all(old > 0) and np.all(new > 0)):
        raise ValueError('action scales must be finite and positive')
    factors = old/new
    policy = model.policy
    if factors.shape != tuple(policy.log_std.shape) or policy.action_net.out_features != len(names):
        raise ValueError('resume action layout differs from the new environment')
    if hasattr(policy, 'sym_a_idx'):
        mirrored = policy.sym_a_idx.detach().cpu().numpy()
        if not np.array_equal(factors, factors[mirrored]):
            raise ValueError('action rescaling must commute with the signed symmetry permutation')
    changed = np.flatnonzero(factors != 1.)
    with torch.no_grad():
        for i in changed:
            factor = float(factors[i])
            policy.action_net.weight[i].mul_(factor)
            policy.action_net.bias[i].mul_(factor)
            if resume_action_std is None:
                policy.log_std[i].add_(float(np.log(factor)))
            print(f'rescaled action {i} {names[i]}: scale {old[i]:g} -> {new[i]:g}, '
                  f'mean x {factor:g}, ' + (f'log_std += {np.log(factor):.9f}' if resume_action_std is None
                  else f'std reset to {resume_action_std:g} after symmetric rebuild'), flush=True)
    return factors


def expand_inputs(path, venv, std):
    """Continue a policy on an environment with observations appended at the end (obs_scan). Returns the VecNormalize
    wrapper and the model. The first layer of the actor and critic gets zero weights for the new inputs, so the policy acts
    exactly as the old one until training uses them; their normalisation is mean 0, std `std` (the symmetric policy
    freezes it, so it has to be set here)."""
    from stable_baselines3.common.running_mean_std import RunningMeanStd
    with open(Path(path).with_name('vecnorm.pkl'), 'rb') as f:
        env = pickle.load(f)
    rms, n_new = env.obs_rms, venv.observation_space.shape[0]-env.obs_rms.mean.shape[0]
    if n_new <= 0:
        raise ValueError('expand_inputs requires appended observation coordinates')
    new = RunningMeanStd(shape=(rms.mean.shape[0]+n_new,))
    new.mean = np.r_[rms.mean, np.zeros(n_new)]
    new.var = np.r_[rms.var, np.full(n_new, std**2)]
    new.count = rms.count
    env.obs_rms, env.observation_space = new, venv.observation_space
    env.set_venv(venv)
    old = PPO.load(path, device='cpu')
    # a symmetric policy's reflection constants have the old size: rebuild as a plain policy, main() wraps it again
    model = PPO('MlpPolicy', env, n_steps=old.n_steps, batch_size=old.batch_size, n_epochs=old.n_epochs,
                learning_rate=old.learning_rate, gamma=old.gamma, gae_lambda=old.gae_lambda, clip_range=old.clip_range,
                ent_coef=old.ent_coef, max_grad_norm=old.max_grad_norm, device='cpu',
                policy_kwargs={k: v for k, v in old.policy_kwargs.items() if k != 'sym'}, seed=0, verbose=0)
    sd = old.policy.state_dict()
    for k, w in sd.items():
        if w.ndim == 2 and k.endswith('.0.weight') and w.shape[1] == rms.mean.shape[0]:
            sd[k] = torch.cat([w, torch.zeros(w.shape[0], n_new)], 1)
    model.policy.load_state_dict(sd, strict=False)
    model.num_timesteps = old.num_timesteps
    return env, model


def validate_jump_resume(old, new):
    """A jump-only extension must keep the old input layout and symmetric inference."""
    if not new.get('jump'):
        return
    if old.get('symmetric') and not new.get('symmetric'):
        raise ValueError('jump resume from a symmetric policy requires --symmetric')
    old_jump = json.loads(old['jump']) if old.get('jump') else {}
    new_jump = json.loads(new['jump'])
    if old_jump.get('obs_phase', False) and not new_jump.get('obs_phase', False):
        raise ValueError('jump resume cannot remove the existing phase input')
    for key, default in [('obs_scan', None), ('obs_hold', None), ('no_obs_vel', False), ('obs_contact', False)]:
        if old.get(key, default) != new.get(key, default):
            raise ValueError('jump resume must preserve existing observation layout: '+key)


def env_kwargs(a):
    """RunEnv keyword arguments from a train_run argument dict (also args.json of a finished run; keys added later default)."""
    g = a.get
    return dict(mass=a['mass'], springs=a['springs'], tail=a['tail'], level=a['level'], cmd_max=a['cmd'], episode_s=a['episode_s'],
                weights=json.loads(a['weights']), zero_cmd=a['zero_cmd'], top_cmd=a['top_cmd'], disturb_items=a['disturb_items'],
                obs_vel=not a['no_obs_vel'], init_speed=not a['no_init_speed'], ankle_clutch=a['ankle_clutch'],
                kin=json.loads(g('kin') or '{}'), yaw_impulse=g('yaw_impulse'), couple_ankle=g('couple_ankle', False),
                track_sigma_frac=g('track_sigma_frac', .15), arch8=g('arch8', False),
                springs_override=json.loads(a['springs_override']) if g('springs_override') else None,
                arch8_override=json.loads(a['arch8_override']) if g('arch8_override') else None,
                arch12=g('arch12', False), real_mass=g('real_mass', False), hip_back=g('hip_back', 0.), nominal=g('nominal'),
                target_ramp=g('target_ramp', False),obs_contact=g('obs_contact',False), model_rev=g('model_rev', 1), rev3_config=g('rev3_config'), heat_limit=g('heat_limit', 1.), roll_impulse=g('roll_impulse'), init_noise=g('init_noise', .02),
                realism=g('realism'), scale_override=json.loads(a['scale_override']) if g('scale_override') else None,
                range_override=json.loads(a['range_override']) if g('range_override') else None,
                gait_duty_min=g('gait_duty_min'), contact_tc=g('contact_tc'), kp_err=g('kp_err'), action_lpf=g('action_lpf'),
                terrain=json.loads(a['terrain']) if g('terrain') else None,
                obs_scan=json.loads(a['obs_scan']) if g('obs_scan') else None,
                obs_hold=json.loads(a['obs_hold']) if g('obs_hold') else None,
                getup=json.loads(a['getup']) if g('getup') else None, cmd_turn=g('cmd_turn') or 0., hold_settle=g('hold_settle'),
                cmd_mix=json.loads(a['cmd_mix']) if g('cmd_mix') else None,
                jump=json.loads(a['jump']) if g('jump') else None)


def budget_chunk(remaining, rollout):
    """Whole PPO rollouts, at most 1M and never above the declared remaining budget."""
    return (min(1_000_000, int(remaining))//rollout)*rollout


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
    p.add_argument('--fixed-curriculum', action='store_true', help='hold the supplied level and command range fixed for controlled ablations')
    p.add_argument('--track-sigma-frac', type=float, default=.15, help='speed-tracking width as a fraction of the command')
    p.add_argument('--springs-override', help='JSON {joint: [k, q0, mode, engaged]} replacing the spring set (spring_refit.py)')
    p.add_argument('--arch8-override', help='JSON {joint: [class, gear]} for the arch8 motors')
    p.add_argument('--arch8', action='store_true', help='8-axis hybrid: four-bar knee, coupled ankle, locked ankle roll, class motor curves')
    p.add_argument('--couple-ankle', action='store_true', help='horse-style knee-ankle coupling, ankle pitch motors removed')
    p.add_argument('--arch12', action='store_true', help='12-axis leg with the arch8 motor classes (+ankle pitch B 1.5:1, ankle roll A)')
    p.add_argument('--real-mass', action='store_true', help='explicit part masses (run_env.real_masses); --mass is ignored')
    p.add_argument('--hip-back', type=float, default=0., help='R-03: hip mount moved rearward along the torso (m)')
    p.add_argument('--target-ramp', action='store_true', help='interpolate each PD target over the 20 ms control step')
    p.add_argument('--rev3-config', help='model rev 3 axis config replacing sim/rl/rev3_b.json (e.g. sim/rl/rev3_fold.json)')
    p.add_argument('--model-rev', type=int, default=1, help='2 = physically consistent model options of run_env (latch without fallback, foot friction draw, self-collision, knee band, ground start, rotor inertia)')
    p.add_argument('--nominal', type=float, nargs=3, metavar=('HIP', 'KNEE', 'ANKLE'), help='nominal pose (nominal_scan.py)')
    p.add_argument('--yaw-impulse', type=float, nargs=3, metavar=('RATE', 'LO', 'HI'), help='training yaw impulses independent of the level')
    p.add_argument('--init-noise', type=float, default=.02, help='std (rad) of the joint-angle noise at a training reset')
    p.add_argument('--symmetric', action='store_true', help='left-right symmetric policy (sym_policy.py); needs --init, freezes the observation normalisation')
    p.add_argument('--roll-impulse', type=float, nargs=3, metavar=('RATE', 'LO', 'HI'), help='training roll impulses independent of the level')
    p.add_argument('--heat-limit', type=float, default=1., help='the heat term penalises max(0, heat - this); 1 = continuous rating')
    p.add_argument('--scale-override', help='JSON {joint type: rad} action half-range replacing SCALE_RUN, e.g. {"hip_roll": 0.5}')
    p.add_argument('--range-override', help='JSON {joint type: rad} symmetric joint stop replacing the MJCF range, e.g. {"hip_roll": 0.65}')
    p.add_argument('--gait-duty-min', type=float, help='floor of the scheduled stance duty at running speed (default GAIT_DUTY 0.28)')
    p.add_argument('--action-lpf', type=float, help='cutoff Hz of a first-order low-pass on the policy action (default off)')
    p.add_argument('--kp-err', type=float, help='rad of PD position error at the torque limit (kp = tau_max / kp_err; default 0.1)')
    p.add_argument('--contact-tc', type=float, help='solref time constant of every contact (default: MJCF, MuJoCo 0.02)')
    p.add_argument('--obs-scan', help='JSON {"dx": [m ahead], "noise": m, "bias": m, "delay": steps}: ground-height profile '
                   'ahead appended to the observation (run_env.py). With --init from a policy without it, the new inputs start '
                   'with zero weights (the policy acts as before) and normalisation mean 0, std --scan-std')
    p.add_argument('--cmd-turn', type=float, default=0., help='fraction of commands that turn in place (0, 0, yaw)')
    p.add_argument('--cmd-mix', help='JSON rows [vx, vy, yaw, weight]; replaces flat command sampling; requires fixed curriculum')
    p.add_argument('--checkpoint-check', help='Python script called with RUN_DIR ADDED_STEPS after each <=1M block; '
                   'exit 0 continues, 10 stops, other errors abort; enforces a whole-rollout step budget')
    p.add_argument('--hold-settle', type=float, help='m/s: set the zero-command hold anchor once the base is slower than this')
    p.add_argument('--obs-hold', help='JSON {"noise": m, "walk": m/sqrt(s)}: odometry offset from the zero-command hold anchor '
                   '(x, y, heading) appended after the scan (run_env.py); with --init from a policy without it, as --obs-scan')
    p.add_argument('--getup', help='JSON {"modes": {"sit"|"side"|"back"|"easy": weight}, "hold_s": s, "weights": {}}: stage 1b '
                   'episodes start fallen, get-up reward (run_env.py _drop, _getup_reward)')
    p.add_argument('--jump', help='JSON opt-in jump task: auto_request_s (2 or null), flight_bonus (10), success_bonus (30), suspend_stand_penalties (false); height_target_m (0.05 legacy), obs_phase (false); request then optional landing-phase input')
    p.add_argument('--scan-std', type=float, default=.15, help='frozen normalisation std (m) of the new scan / hold inputs')
    p.add_argument('--terrain', help='JSON {"kinds": {kind: weight}, "max_level": L, "vmax": m/s, "size": m}: terrain drawn per episode (run_env.py)')
    p.add_argument('--realism', help='realism config JSON path for RunEnv(realism=...) (realism.py, docs/SIM_REALISM.ko.md); omitted = off')
    p.add_argument('--init', help='model.zip to continue from (its vecnorm.pkl is loaded too)')
    p.add_argument('--resume-action-std', type=float, help='optional Gaussian exploration std reset for a resumed PPO policy; actor mean is preserved')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    if a.cmd_mix and not a.fixed_curriculum:
        p.error('cmd-mix requires --fixed-curriculum')
    if a.resume_action_std is not None and (not a.init or not np.isfinite(a.resume_action_std) or a.resume_action_std <= 0):
        p.error('resume-action-std requires --init and a finite positive value')
    if a.jump and a.init:
        try:
            validate_jump_resume(json.loads(Path(a.init).with_name('args.json').read_text()), vars(a))
        except ValueError as error:
            p.error(str(error))
    out = HERE/'runs'/a.name
    out.mkdir(parents=True, exist_ok=True)
    (out/'args.json').write_text(json.dumps(vars(a), indent=1))
    kw = env_kwargs(vars(a))
    if a.env == 'recover':
        kw = dict(mass=a.mass, springs=a.springs, tail=a.tail, episode_s=a.episode_s, weights=json.loads(a.weights))
    torch.set_num_threads(1)
    env = VecMonitor(SubprocVecEnv([make(kw, a.seed*100+i, RecoverEnv if a.env == 'recover' else RunEnv) for i in range(a.envs)]))
    if a.init:
        vn_path = str(Path(a.init).with_name('vecnorm.pkl'))
        with open(vn_path, 'rb') as f:
            saved_shape = pickle.load(f).observation_space.shape
        if env.observation_space.shape != saved_shape:
            env, model = expand_inputs(a.init, env, 1. if a.jump else a.scan_std)
        else:
            env = VecNormalize.load(vn_path, env)
            model = PPO.load(a.init, env=env, device='cpu')
        widened = a.env == 'run' and widen_command_inputs(env, model, a.cmd_final)
        if a.env == 'run':
            old_args = json.loads(Path(a.init).with_name('args.json').read_text())
            rescale_action_outputs(model, env.get_attr('active')[0], env.get_attr('policy_idx')[0],
                                   old_args, env.get_attr('scale')[0], a.resume_action_std)
        if a.symmetric:
            from mirror_probe import ObservationReflection
            from sym_policy import SymPolicy, freeze_obs, reflection_constants
            probe = RunEnv(seed=0, **kw)
            freeze_obs(env)
            sym = reflection_constants(ObservationReflection(probe.active, probe.policy_idx, probe.q0, probe.obs_vel, probe.obs_contact,
                                                             len(probe.scan_dx), bool(probe.obs_hold), probe.jump_task is not None,
                                                             bool(probe.jump_task and probe.jump_task.obs_phase)), env)
            if not isinstance(model.policy, SymPolicy) or widened:  # rebuild: the reflection stores the normalisation
                old = model
                model = PPO(SymPolicy, env, n_steps=old.n_steps, batch_size=old.batch_size, n_epochs=old.n_epochs, learning_rate=old.learning_rate,
                            gamma=old.gamma, gae_lambda=old.gae_lambda, clip_range=old.clip_range, ent_coef=old.ent_coef, max_grad_norm=old.max_grad_norm,
                            device='cpu', policy_kwargs={**old.policy_kwargs, 'sym': sym}, seed=a.seed, verbose=0)
                model.policy.load_state_dict(old.policy.state_dict(), strict=False)
                model.num_timesteps = old.num_timesteps
        if a.resume_action_std is not None:
            with torch.no_grad():
                model.policy.log_std.fill_(float(np.log(a.resume_action_std)))
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., gamma=.99)
        model = PPO('MlpPolicy', env, n_steps=1024, batch_size=4096, n_epochs=5, learning_rate=3e-4,
                    gamma=.99, gae_lambda=.95, clip_range=.2, ent_coef=.003, max_grad_norm=1., device='cpu',
                    policy_kwargs=dict(net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64]),
                                       activation_fn=torch.nn.ELU, log_std_init=-2.),
                    seed=a.seed, verbose=0)
    model.set_logger(configure(str(out), ['csv']))
    (out/'initial-policy.json').write_text(json.dumps(dict(
        steps=model.num_timesteps, action_std=model.policy.log_std.detach().exp().cpu().numpy().tolist(),
        resume_action_std=a.resume_action_std), indent=2))
    if a.fixed_curriculum:
        cb = None
    elif a.env == 'run' and a.eval_curriculum:
        cb = EvalCurriculum(out/'curriculum.jsonl', {k: v for k, v in kw.items() if k not in ('level', 'cmd_max')}, a.level, a.cmd, a.cmd_final)
    else:
        cb = Curriculum(out/'curriculum.jsonl', a.level, a.cmd, a.cmd_final) if a.env == 'run' else None
    chunk, done = 1_000_000, 0
    while done < a.steps:
        n = budget_chunk(a.steps-done, model.n_steps*a.envs) if a.checkpoint_check else int(min(chunk, a.steps-done))
        if n == 0:
            break
        before = model.num_timesteps
        model.learn(n, callback=cb, reset_num_timesteps=False, progress_bar=False)
        done += model.num_timesteps-before if a.checkpoint_check else n
        model.save(out/'model.zip')
        env.save(str(out/'vecnorm.pkl'))
        print(json.dumps({'steps': model.num_timesteps, **({'level': round(cb.level, 2), 'cmd': cb.cmd.round(3).tolist()} if cb else {})}), flush=True)
        if a.checkpoint_check:
            rc = subprocess.run([sys.executable, a.checkpoint_check, str(out), str(done)], check=False).returncode
            if rc == 10:
                break
            if rc:
                env.close()
                raise RuntimeError(f'checkpoint check failed ({rc}); training stopped')
    env.close()


if __name__ == '__main__':
    main()
