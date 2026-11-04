"""5a jump gate: retained walk -> requested jump -> one second standing; 1 ms raw traces local only."""
import json
from pathlib import Path
import mujoco
import numpy as np
from evaluate import load
from run_env import RunEnv
from train_run import env_kwargs
from jump_task import (SECONDS, NOMINAL_SEEDS, DR_SEEDS, QUICK_NOMINAL_SEEDS, QUICK_DR_SEEDS)

DEFAULT_TRACES = Path(__file__).resolve().parents[2]/'docs/evidence/199-l80-jump/eval'


def jump_rollout(run, walk_dir, seed, randomize, trace_dir=None):
    args = json.loads((run/'args.json').read_text())
    if not args.get('jump'):
        raise ValueError('jump gate requires a policy with the appended jump request input')
    kw = {**env_kwargs(args), 'episode_s': SECONDS+2., 'level': 0., 'disturb_items': [], 'roll_impulse': None,
          'yaw_impulse': None, 'init_speed': False, 'cmd_max': [0.,0.,0.], 'cmd_mix': None,
          'jump': {'auto_request_s': 2., 'flight_bonus': 0., 'success_bonus': 0.}}
    env = RunEnv(**kw, randomize=randomize, seed=seed)
    if env.rz is None or not env.rz.on['motor']:
        raise ValueError('jump gate requires motor realism and current measurements')
    wargs = json.loads((walk_dir/'args.json').read_text())
    wenv = RunEnv(**env_kwargs(wargs), randomize=False, seed=seed)
    model, vn = load(str(run/'model.zip'), env)
    walk, wn = load(str(walk_dir/'model.zip'), wenv)
    try:
        if env.active != wenv.active or not np.array_equal(env.q0, wenv.q0) or not np.array_equal(env.scale, wenv.scale):
            raise ValueError('jump handoff requires matching joint layout, nominal and action scales')
        if wn.observation_space.shape[0]+1 != vn.observation_space.shape[0]:
            raise ValueError('jump handoff must append exactly one input to the walk observation')
        env.init_seed = seed
        env.reset(seed=seed)
        # RunEnv.init_seed adds initial vx too; this gate explicitly starts stationary.
        env.data.qvel[:6] = 0.; mujoco.mj_forward(env.model, env.data)
        if env.rz is not None:
            env.rz.reset(env)
        env.command = np.zeros(3); env.resample_steps = 0
        task = env.jump_task; task.capture = True
        raw = env._observe(); fell = None; returned_at = None
        for k in range(round((SECONDS+1.)/.02)):
            use_jump = task.request_time is not None and task.success_time is None
            if task.success_time is not None and returned_at is None:
                returned_at = float(env.data.time)
            policy, norm, obs = (model, vn, raw) if use_jump else (walk, wn, raw[:-1])
            action = policy.predict(norm.normalize_obs(obs[None]), deterministic=True)[0][0]
            raw, _, term, trunc, _ = env.step(action)
            if term:
                fell = task.fall_time if task.fall_time is not None else float(env.data.time)
                break
            if trunc or (returned_at is not None and env.data.time-returned_at >= 1.-1e-9):
                break
            if task.success_time is None and env.data.time >= SECONDS-1e-9:
                break
        result = task.result()
        result.update(seed=seed, t_fall=fell, elapsed_s=float(env.data.time), handoff_from=walk_dir.name,
                      sample_dt_s=.001, return_to=walk_dir.name if returned_at is not None else None,
                      return_observed_s=float(env.data.time-returned_at) if returned_at is not None else 0.)
        path = Path(trace_dir) if trace_dir is not None else DEFAULT_TRACES
        path.mkdir(parents=True, exist_ok=True)
        path = path/f'{run.name}_jump_{"dr" if randomize else "nominal"}_{seed}.npz'
        if path.exists():
            raise FileExistsError(f'preserve existing 1 ms trace: {path}')
        np.savez_compressed(path, **{key:np.asarray([s[key] for s in task.trace]) for key in task.trace[0]},
                            weight_n=env.weight, active=np.array(env.active), motor_indices=env.rz.idx)
        result['raw_trace'] = str(path)
        return result
    finally:
        vn.close(); wn.close()


def jump_conditions(run, walk_dir, quick=False, trace_dir=None):
    for cond, randomize, seeds in (
        ('in_place', False, QUICK_NOMINAL_SEEDS if quick else NOMINAL_SEEDS),
        ('in_place_dr', True, QUICK_DR_SEEDS if quick else DR_SEEDS)):
        episodes = [jump_rollout(run, walk_dir, s, randomize, trace_dir) for s in seeds]
        yield dict(gate='jump', cond=cond, quick=quick, n=len(episodes), walk=walk_dir.name,
                   falls=sum(e['t_fall'] is not None for e in episodes), episodes=episodes)
