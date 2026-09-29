"""State at each fall of a policy under all dr-2 groups (pulses 20 N): reason, time since pulse/shove,
gravity in the body frame now and 1 s before, command, servo kv, observation delay, torso CoM shift (evidence 82).
Usage: .venv-sim/bin/python sim/rl/fall_modes.py runs/<name>/model.zip   (run from sim/rl)
"""
import sys, numpy as np
from multiprocessing import Pool
sys.path.insert(0, '.')
from evaluate import load
from raptor_env import RaptorEnv
def run(seed):
    env = RaptorEnv('flat', dof=12, sole='flat', randomize=True, seed=seed, cmd_max=(.6,.2,.6), zero_cmd=.25,
                    jtc_horizon=.02, kv_range=(20.,80.), slew=.85, dr_items=['delay','noise','toe','com','pulses','initvel'], pulse_force=20.)
    model, venv = load(sys.argv[1], env); raw, _ = env.reset(seed=seed)
    obs = venv.normalize_obs(raw[None])
    last_pulse = -9.; hist = []
    for k in range(1000):
        a, _ = model.predict(obs, deterministic=True)
        raw, _, term, trunc, info = env.step(a[0]); obs = venv.normalize_obs(raw[None])
        d = env.data
        if np.any(d.xfrc_applied[env.base, :2] != 0) and d.time-last_pulse > .35: last_pulse = d.time
        g = d.xmat[env.base].reshape(3,3).T @ np.array([0,0,-1.])
        hist.append(g[:2].copy())
        if term:
            feet, hit = env._contacts()
            bz = d.xpos[env.base][2]
            h = np.array(hist[-50:])
            return dict(seed=seed, t=round(d.time,2), since_pulse=round(d.time-last_pulse,2), since_shove=round(d.time-env.push_at,2),
                        reason=('body_hit ' if hit else '')+('tilt ' if info['tilt'] > .8 else '')+('low' if bz < .5 else ''),
                        grav_xy_now=g[:2].round(2).tolist(), grav_xy_1s_ago=h[0].round(2).tolist(),
                        cmd=env.command.round(2).tolist(), kv=round(float(env.model.actuator_gainprm[0,0]),1), delay=env.obs_delay,
                        com=(env.model.body_ipos[env.base]-env.nominal_ipos)[:2].round(3).tolist())
        if trunc: return None
    return None
if __name__ == '__main__':
    with Pool(9) as p: res = [r for r in p.map(run, range(5000, 5048)) if r]
    print(len(res), 'falls')
    for r in res: print(r)
