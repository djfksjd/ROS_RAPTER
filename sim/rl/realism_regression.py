"""Regression for the opt-in realism layer: RunEnv rollouts with a fixed seed and a fixed action sequence, realism off.

Run once per source tree and compare bit for bit (the realism branch must reproduce main exactly):
  git worktree add /tmp/main_wt origin/main
  python sim/rl/realism_regression.py run --root /tmp/main_wt --out main.npz
  python sim/rl/realism_regression.py run --root . --out branch.npz
  python sim/rl/realism_regression.py compare main.npz branch.npz
`run` imports RunEnv from <root>/sim/rl, so this script also drives trees that do not contain it. Actions are a smoothed
random sequence (the same for every tree); a fall resets the episode without a new seed, so resets are covered too.
"""
import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np

CONDITIONS = {
    'default_dr': dict(seed=0),
    'deterministic': dict(seed=1, randomize=False),
    'arch8_ramp': dict(seed=2, arch8=True, target_ramp=True, level=.5),
    'rev2_arch12': dict(seed=3, model_rev=2, arch12=True, real_mass=True),
    'locked_tail_b': dict(seed=4, tail='locked', springs='b', obs_contact=True, level=1.),
    'arch8_ratchet': dict(seed=5, arch8=True, model_rev=2, weights={'spring_store': .1, 'heat': -.5},
                          springs_override={'knee_pitch': [500., .956, 'uni+', 'ratchet']}),
}
STEPS = 200


def rollout(RunEnv, kw, steps=STEPS):
    env = RunEnv(**kw)
    obs, _ = env.reset(seed=kw.get('seed'))
    na = env.action_space.shape[0]
    rng = np.random.default_rng(1234)
    a = np.zeros(na)
    rec = {k: [] for k in ('obs', 'reward', 'done', 'qpos', 'qvel', 'ctrl', 'heat')}
    for _ in range(steps):
        a = .8*a+.2*rng.uniform(-1, 1, na)  # smoothed random actions: the robot moves without falling at once
        obs, r, term, trunc, info = env.step(a)
        rec['obs'].append(obs); rec['reward'].append(r); rec['done'].append((term, trunc))
        rec['qpos'].append(env.data.qpos.copy()); rec['qvel'].append(env.data.qvel.copy())
        rec['ctrl'].append(env.data.ctrl.copy())
        rec['heat'].append(info['heat'] if info['heat'] is not None else np.zeros(1))
        if term or trunc:
            obs, _ = env.reset()
            rec['obs'].append(obs)
    return {k: np.asarray(v) for k, v in rec.items()}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run'); r.add_argument('--root', required=True); r.add_argument('--out', required=True)
    c = sub.add_parser('compare'); c.add_argument('a'); c.add_argument('b')
    args = ap.parse_args()
    if args.cmd == 'run':
        sys.path.insert(0, str(Path(args.root).resolve()/'sim'/'rl'))
        from run_env import RunEnv
        out = {}
        for name, kw in CONDITIONS.items():
            rec = rollout(RunEnv, kw)
            for k, v in rec.items():
                out[f'{name}/{k}'] = v
            h = hashlib.sha256(b''.join(rec[k].tobytes() for k in sorted(rec))).hexdigest()[:16]
            print(f'{name:14s} steps {STEPS} resets {int(sum(any(x) for x in rec["done"]))} sha256 {h}')
        np.savez(args.out, **out)
    else:
        a, b = np.load(args.a), np.load(args.b)
        bad = sorted(set(a.files) ^ set(b.files)) + [k for k in sorted(set(a.files) & set(b.files))
                                                     if a[k].shape != b[k].shape or a[k].tobytes() != b[k].tobytes()]
        conds = sorted({k.split('/')[0] for k in a.files})
        for cnd in conds:
            print(f'{cnd:14s}', 'IDENTICAL' if not any(k.startswith(cnd+'/') for k in bad) else 'DIFFERENT')
        print('ALL IDENTICAL' if not bad else f'DIFFERENT: {bad}')
        sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
