"""Control steps per second of RunEnv with the realism layer off, per block, and fully on (single process, fixed actions).

  python sim/rl/realism_bench.py [--steps 1000] [--repeat 3]
Each case runs `steps` control steps (1 s = 50 steps, 20 physics steps each) and resets on a fall; the best of
`repeat` runs is reported, so the numbers are machine-dependent throughput, not a training-time estimate.
"""
import argparse
import time
import warnings

import numpy as np

from run_env import RunEnv

CASES = {
    'off': None,
    'motor': {'enable': ['motor']},
    'motor+thermal': {'enable': ['motor', 'thermal']},
    'friction': {'enable': ['friction']},
    'delay(obs 2)': {'enable': ['delay']},
    'imu+encoder': {'enable': ['imu', 'encoder']},
    'all default': True,
    'all+backlash+battery': {'backlash': {'enabled': True}, 'battery': {'enabled': True}},
}


def run(realism, steps, kw):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        env = RunEnv(seed=0, realism=realism, randomize=False, **kw)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    acts = np.cumsum(rng.normal(0, .05, (steps, env.action_space.shape[0])), 0).clip(-1, 1)
    t0 = time.perf_counter()
    for a in acts:
        _, _, term, trunc, _ = env.step(a)
        if term or trunc:
            env.reset()
    return steps/(time.perf_counter()-t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=1000)
    ap.add_argument('--repeat', type=int, default=3)
    ap.add_argument('--arch8', action='store_true')
    a = ap.parse_args()
    kw = dict(arch8=True, model_rev=2) if a.arch8 else {}
    base = None
    for name, cfg in CASES.items():
        sps = max(run(cfg, a.steps, kw) for _ in range(a.repeat))
        base = base or sps
        print(f'{name:22s} {sps:8.1f} control steps/s  {sps/base:5.2f}x of off')


if __name__ == '__main__':
    main()
