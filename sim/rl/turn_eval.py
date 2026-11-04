"""Moving-turn measurements shared by evidence 198 and the stage-4 gate.

Five seconds straight from standing, then ten seconds turning. Statistics over
8--15 s; heading is unwrapped from the body rotation, not integrated local gyro.
Hip yaw drive torque/speed/limits are sampled after every 1 ms physics step.
"""
import json
from pathlib import Path

import numpy as np

from evaluate import load
from run_env import RunEnv
from train_run import env_kwargs


def heading_delta(previous, current):
    return float((current-previous+np.pi) % (2*np.pi)-np.pi)


def turn_rollout(d, seed, speed, yaw, instrument=True):
    d = Path(d)
    a = json.loads((d/'args.json').read_text())
    kw = {**env_kwargs(a), 'level': 0., 'episode_s': 20., 'disturb_items': [],
          'roll_impulse': None, 'yaw_impulse': None, 'cmd_max': [0., 0., 0.], 'cmd_mix': None, 'init_speed': False}
    env = RunEnv(**kw, randomize=False, seed=seed)
    env.init_seed = seed
    model, vn = load(str(d/'model.zip'), env)
    vn.reset()
    env.resample_steps = 0
    dd, m = env.data, env.model
    idx = [i for i, name in enumerate(env.active) if 'hip_yaw' in name]
    forces, speeds, limits = [], [], []
    original = env.rz.after_substep

    def sample(e, dt):
        original(e, dt)
        if e.steps >= 250:  # entire turning interval, including onset and any fall
            f = dd.actuator_force[env.act[idx]].copy()
            lim = m.actuator_forcerange[env.act[idx]]
            forces.append(f)
            speeds.append(dd.qvel[env.v_adr[idx]].copy())
            limits.append(np.where(f >= 0, lim[:, 1], -lim[:, 0]).copy())

    if instrument:
        env.rz.after_substep = sample
    heading = lambda: float(np.arctan2(dd.xmat[env.base][3], dd.xmat[env.base][0]))
    prev, angle, t90, fell = heading(), 0., None, None
    log = []
    raw = env._obs()
    try:
        for k in range(750):
            env.command = np.array([speed, 0., yaw if k >= 250 else 0.])
            # Command is not a delayed sensor. Refresh only its observation slots.
            raw[6:9] = env.command*np.array([.1, 2., .5])
            raw, _, term, _, _ = env.step(model.predict(vn.normalize_obs(raw[None]), deterministic=True)[0][0])
            current = heading()
            if k >= 250:
                angle += heading_delta(prev, current)
                if t90 is None and np.sign(yaw)*angle >= np.pi/2:
                    t90 = (k+1)*.02-5.
            prev = current
            w, v = env.body_velocity(env.base)
            log.append(((k+1)*.02, float(v[0]), float(w[2]), angle))
            if term:
                fell = (k+1)*.02
                break
    finally:
        env.rz.after_substep = original
        vn.close()
    L = np.asarray(log)
    steady = L[L[:, 0] > 8.]
    complete = fell is None and len(log) == 750
    vm, wm = (map(float, steady[:, 1:3].mean(axis=0)) if complete else (None, None))
    joints = {}
    if forces:
        T, W, B = np.array(forces), np.array(speeds), np.array(limits)
        for j, i in enumerate(idx):
            joints[env.active[i]] = dict(rms_nm=float(np.sqrt(np.mean(T[:, j]**2))),
                peak_nm=float(np.max(np.abs(T[:, j]))), peak_rad_s=float(np.max(np.abs(W[:, j]))),
                at_limit=float(np.mean((B[:, j] > 1e-6) & (np.abs(T[:, j]) >= .98*B[:, j]))))
    return dict(seed=seed, t_fall=fell, v_mean=vm, yaw_rate_mean=wm,
                radius_m=abs(vm/wm) if wm is not None and abs(wm) > 1e-6 else None,
                heading_change_rad=angle, t_90_s=t90, load_samples=len(forces), joints=joints)


def turn_condition(d, speed, yaw, seeds):
    rows = [turn_rollout(d, s, speed, yaw) for s in seeds]
    return dict(cond='moving_turn', cmd=speed, yaw_cmd=yaw, policy=Path(d).name,
                n=len(rows), falls=sum(r['t_fall'] is not None for r in rows), episodes=rows)
