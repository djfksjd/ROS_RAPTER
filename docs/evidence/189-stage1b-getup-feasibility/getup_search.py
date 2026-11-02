"""Evidence 189: can R-03 get up at all? Open-loop keyframe search (cross-entropy method), no policy.

Start poses: RunEnv getup starts (run_env._drop), settled with the joints held until at rest on the ground:
  sit   upright torso orientation, legs folded (knee 1.6-2.2 rad, hip pitch forward); the leg springs open the fold
        and the robot ends on its belly, side or back
  side  lying on the left side (roll +pi/2)
  back  upside down (roll pi)
A candidate is K keyframes of the 10 policy axes (action units, target = q0 + scale*a as RunEnv), each held `dt` s with
linear blending, then the nominal pose. Score: best handoff value over the rollout; handoff state = torso tilt < 0.3 rad,
base above 0.8 of the nominal stand height, both feet loaded > 15 % of the weight, base speed < 0.5 m/s (the state
from which the stage 1a stand policy would take over; reaching it is NOT standing).
Motors: the run's realism motor limits through RunEnv._clamp (current/T-N/voltage caps); delays, sensor noise and
heating are not stepped (short open-loop motion).
Usage: .venv-sim/bin/python docs/evidence/189-stage1b-getup-feasibility/getup_search.py MODE [--iters 20 --pop 64]
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'sim/rl'))
from run_env import RunEnv, CONTROL_DT, PHYSICS_DT  # noqa: E402
from train_run import env_kwargs  # noqa: E402

K, DT, TAIL_S = 4, .6, 1.5
ORIENT = {'sit': (0., 0.), 'side': (np.pi/2, 0.), 'back': (np.pi, 0.)}
_env = None


def make():
    a = json.loads((ROOT/'sim/rl/runs/r3_s1ho/args.json').read_text())
    kw = {**env_kwargs(a), 'disturb_items': [], 'roll_impulse': None, 'yaw_impulse': None, 'cmd_max': [0., 0., 0.],
          'init_speed': False, 'action_lpf': None}
    return RunEnv(**kw, randomize=False, seed=1)


def drop(e, mode, seed):
    """Place the robot in the fallen pose with the env's own get-up start (RunEnv._drop: passive joints at the reset
    pose, ankle on the knee coupling, settled until at rest on the ground)."""
    e.getup = {'modes': {mode: 1.}}
    e.reset(seed=seed)
    e.getup = None
    return e.h_nom+.03


def rollout(e, mode, seed, x, trace=False):
    h_stand = drop(e, mode, seed)
    m, d = e.model, e.data
    keys = np.vstack([np.clip(x.reshape(K, -1), -1, 1), np.zeros(len(e.policy_idx))])
    n_ctrl = int(round(CONTROL_DT/PHYSICS_DT))
    steps = int((K*DT+TAIL_S)/CONTROL_DT)
    best, reached, log = 0., None, []
    pi = e.policy_idx  # blend from the settled joint pose
    prev = np.clip((d.qpos[e.q_adr][pi]-e.q0[pi])/e.scale[pi], -1, 1)
    for k in range(steps):
        t = k*CONTROL_DT
        i = min(int(t/DT), K)
        f = min((t-i*DT)/.2, 1.) if i < K else min((t-K*DT)/.2, 1.)  # 0.2 s blend into each keyframe
        a = prev+(keys[i]-prev)*f if f < 1 else keys[i]
        if f >= 1:
            prev = keys[i]
        target = e.q0.copy()
        target[e.policy_idx] = np.clip(e.q0[e.policy_idx]+e.scale[e.policy_idx]*a, e.lo[e.policy_idx], e.hi[e.policy_idx])
        d.ctrl[e.act] = target
        for _ in range(n_ctrl):
            e._clamp(); mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos).all():
            return -1., None, log
        R = d.xmat[e.base].reshape(3, 3)
        up = (1+R[2, 2])/2
        tilt = float(np.arccos(np.clip(R[2, 2], -1, 1)))
        bz = float(d.xpos[e.base][2])
        feet, _ = e._contacts()
        loaded = all(feet[s] > .15*e.weight for s in feet)
        speed = float(np.linalg.norm(d.qvel[:3]))
        h = min(bz/h_stand, 1.)
        v = up+h*(1. if loaded else .5)
        handoff = tilt < .3 and bz > .8*h_stand and loaded and speed < .5
        if handoff and reached is None:
            reached = round(t, 2)
        best = max(best, v+(2. if handoff else 0.))
        if trace:
            log.append((round(t, 2), round(tilt, 3), round(bz, 3), int(loaded), round(speed, 3)))
    return best, reached, log


def _init():
    global _env
    _env = make()


def _eval(args):
    mode, seeds, x = args
    r = [rollout(_env, mode, s, x)[:2] for s in seeds]
    return float(np.mean([b for b, _ in r])), sum(t is not None for _, t in r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=list(ORIENT))
    ap.add_argument('--iters', type=int, default=20)
    ap.add_argument('--pop', type=int, default=64)
    ap.add_argument('--procs', type=int, default=3)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    n = K*10
    mu, sd = np.zeros(n), np.full(n, .7)
    seeds = [11, 12, 13]  # three fallen starts per candidate (robust, not one lucky start)
    hist = []
    with Pool(a.procs, initializer=_init) as pool:
        for it in range(a.iters):
            X = np.clip(mu+sd*rng.standard_normal((a.pop, n)), -1, 1)
            res = pool.map(_eval, [(a.mode, seeds, x) for x in X])
            sc = np.array([r[0] for r in res]); ok = np.array([r[1] for r in res])
            elite = X[np.argsort(-sc)[:a.pop//8]]
            mu, sd = elite.mean(0), np.maximum(elite.std(0), .05)
            hist.append(dict(iter=it, best=round(float(sc.max()), 3), mean=round(float(sc.mean()), 3), handoff_max=int(ok.max())))
            print(json.dumps(hist[-1]), flush=True)
        best = X[int(np.argmax(sc))]
        # held-out starts
        test = pool.map(_eval, [(a.mode, [s], mu) for s in range(100, 120)])
    out = dict(mode=a.mode, hist=hist, mu=mu.round(4).tolist(), heldout_handoff=sum(t for _, t in test), heldout_n=len(test))
    (OUT/f'eval/search_{a.mode}.json').write_text(json.dumps(out))
    print(json.dumps({k: out[k] for k in ('mode', 'heldout_handoff', 'heldout_n')}))


if __name__ == '__main__':
    main()
