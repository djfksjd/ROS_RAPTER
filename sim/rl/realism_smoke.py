"""Short rollout with a realism config (no training, no policy): NaN check, winding temperature and current from info,
control steps per second. Smoothed random actions with a fixed seed; a fall resets the episode.

  python sim/rl/realism_smoke.py [--config sim/rl/realism_robstride.json] [--steps 1000] [--extra backlash,battery]
Prints one JSON object. Temperatures after a few seconds say nothing about continuous ratings (time constants are
minutes); this only checks that the model runs and reports.
"""
import argparse
import json
from pathlib import Path
import time
import warnings

import numpy as np

from run_env import RunEnv

HERE = Path(__file__).resolve().parent


def rollout(realism, steps, seed, randomize, kw):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        env = RunEnv(seed=seed, realism=realism, randomize=randomize, **kw)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    a = np.zeros(env.action_space.shape)
    t_max, i_max, resets, bad = None, None, 0, 0
    impacts, overcurrent = 0, 0
    t0 = time.perf_counter()
    for _ in range(steps):
        a = .8*a+.2*rng.uniform(-1, 1, a.shape)
        obs, r, term, trunc, info = env.step(a)
        rz = info.get('realism')
        finite = np.all(np.isfinite(obs)) and np.isfinite(r) and np.all(np.isfinite(env.data.qpos)) \
            and np.all(np.isfinite(env.data.qvel))
        if rz is not None:
            idx = env.rz.idx
            for k in ('t_winding', 'current'):
                if k in rz:
                    finite &= bool(np.all(np.isfinite(rz[k][idx])))
            if 't_winding' in rz:
                t_max = rz['t_winding'][idx] if t_max is None else np.maximum(t_max, rz['t_winding'][idx])
            if 'current' in rz:
                i_max = rz['current'][idx] if i_max is None else np.maximum(i_max, rz['current'][idx])
            impacts = rz.get('backlash_impacts', impacts)
            overcurrent += bool(np.any(rz.get('overcurrent', False)))
        bad += not finite
        if term or trunc:
            resets += 1
            env.reset()
    sps = steps/(time.perf_counter()-t0)
    out = {'steps': steps, 'resets': resets, 'nonfinite_steps': int(bad), 'steps_per_s': round(sps, 1)}
    if rz is not None:
        names = [env.active[i] for i in env.rz.idx]
        out['last_info_keys'] = sorted(rz)
        if t_max is not None:
            out['t_winding_max_c'] = dict(zip(names, np.round(t_max, 3).tolist()))
        if i_max is not None:
            out['current_max_a'] = dict(zip(names, np.round(i_max, 2).tolist()))
        out['backlash_impacts_last_episode'] = impacts
        out['overcurrent_steps'] = overcurrent  # control steps with a voltage-forced current past i_lim (end of step)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(HERE/'realism_robstride.json'))
    ap.add_argument('--steps', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--randomize', action='store_true')
    ap.add_argument('--extra', default='', help='comma list of extra blocks to switch on, e.g. backlash,battery')
    a = ap.parse_args()
    kw = dict(arch8=True, model_rev=2)
    cfg = json.loads(Path(a.config).read_text())
    for b in filter(None, a.extra.split(',')):
        cfg.setdefault(b, {})['enabled'] = True
    cfg['motor']['spec_file'] = str((Path(a.config).resolve().parent/cfg['motor']['spec_file']).resolve())
    res = {'config': a.config, 'extra': a.extra, 'randomize': a.randomize, 'env': kw,
           'off': rollout(None, a.steps, a.seed, a.randomize, kw),
           'on': rollout(cfg, a.steps, a.seed, a.randomize, kw)}
    res['on_vs_off_speed'] = round(res['on']['steps_per_s']/res['off']['steps_per_s'], 2)
    print(json.dumps(res, ensure_ascii=False))


if __name__ == '__main__':
    main()
