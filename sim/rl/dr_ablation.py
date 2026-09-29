"""Fall rate of a policy with one dr-2 group switched on at a time (training-like randomization, fixed seeds).

Deterministic actions by default; --stochastic samples them as PPO training does (training fall rates).

Every case keeps the base randomization (mass, friction, force range, one shove, kv range) and adds one group,
so a group's effect is its fall rate minus the 'base' row. Commands are sampled as in training.
Usage: .venv-sim/bin/python sim/rl/dr_ablation.py runs/A/model.zip [runs/B/model.zip ...] --out result.json
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from evaluate import load  # noqa: E402
from raptor_env import RaptorEnv  # noqa: E402

CASES = {'base': [], 'delay': ['delay'], 'noise': ['noise'], 'toe': ['toe'], 'com': ['com'],
         'pulses': ['pulses'], 'initvel': ['initvel'], 'b2': ['delay', 'noise', 'toe', 'com'], 'all_b3': ['delay', 'noise', 'toe', 'com', 'pulses', 'initvel']}
# leave-one-out of the B2 combination: which group the combination's falls depend on
CASES.update({f'b2_no_{g}': [x for x in CASES['b2'] if x != g] for g in CASES['b2']})


def run(job):
    policy, case, seed, a = job
    env = RaptorEnv('flat', dof=12, sole='flat', randomize=True, seed=seed, cmd_max=(.6, .2, .6), zero_cmd=.25,
                    jtc_horizon=.02, kv_range=(20., 80.), slew=.85, dr_items=CASES[case] or None, model_path=a['design'], crouch=a['crouch'])
    env.dr_items = set(CASES[case])  # empty set = base randomization only
    env.pulse_force, env.com_shift = a['pulse_force'], a['com_shift']
    model, venv = load(policy, env)
    obs = venv.reset()
    steps = 0
    for steps in range(1, env.episode_steps+1):
        action, _ = model.predict(obs, deterministic=not a['stochastic'])
        obs, _, done, info = venv.step(action)
        if done[0]:
            fell = not info[0].get('TimeLimit.truncated', False) and steps < env.episode_steps
            return policy, case, fell, steps*.02
    return policy, case, False, steps*.02


def main():
    p = argparse.ArgumentParser()
    p.add_argument('models', nargs='+')
    p.add_argument('--cases', nargs='+', default=list(CASES))
    p.add_argument('--episodes', type=int, default=12)
    p.add_argument('--design', help='MJCF design variant')
    p.add_argument('--pulse-force', type=float, default=40., help='N per horizontal axis (pulses group)')
    p.add_argument('--crouch', type=float, nargs=2, metavar=('HIP', 'KNEE'), help='nominal leg pose used in training')
    p.add_argument('--com-shift', type=float, default=.03, help='m, torso CoM shift range (com group)')
    p.add_argument('--stochastic', action='store_true', help='sample actions as in training')
    p.add_argument('--procs', type=int, default=8)
    p.add_argument('--out')
    a = p.parse_args()
    jobs = [(m, c, 5000+e, {'design': a.design, 'pulse_force': a.pulse_force, 'stochastic': a.stochastic, 'com_shift': a.com_shift, 'crouch': a.crouch}) for m in a.models for c in a.cases for e in range(a.episodes)]
    with Pool(a.procs) as pool:
        res = pool.map(run, jobs, chunksize=1)
    rows = []
    for m in a.models:
        for c in a.cases:
            r = [x for x in res if x[0] == m and x[1] == c]
            row = {'model': m, 'case': c, 'falls': sum(x[2] for x in r), 'episodes': len(r),
                   'mean_time_s': round(float(np.mean([x[3] for x in r])), 1), 'pulse_force': a.pulse_force, 'stochastic': a.stochastic, 'com_shift': a.com_shift}
            rows.append(row)
            print(json.dumps(row), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rows, indent=1)+'\n')


if __name__ == '__main__':
    main()
