#!/usr/bin/env python3
"""MuJoCo side of the evidence-73 lateral experiments (A: sway amplitude, B: swing window), 60 s, tail fixed.

Usage: python lateral_experiments.py OUT.json {A|B} [--kv 20 30 100] [--cycles 150]
Each row: fall time and roll_diagnosis.table_metrics over [2 s, fall or end]. Gazebo runs use rock_gz_probe.py.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import roll_diagnosis as rd

SWEEPS = {'A': [('amplitude', a) for a in (.08, .04, .02, .01, 0.)],
          'B': [('window_width', w) for w in (.35, .30, .25)]}


def one(job):
    kv, key, value, cycles = job
    series, result = rd.mujoco_series(kv, cycles, **{key: value})
    fall = result['stopped_by_guard_at_s']
    end = fall if fall else float(series['t'][-1])
    return {'kv': kv, key: value, 'fall_s': fall, 'x_m': result['final_xy'][0],
            **rd.table_metrics(series, 2., end, 2.5)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out')
    parser.add_argument('experiment', choices=sorted(SWEEPS))
    parser.add_argument('--kv', type=float, nargs='+', default=[20., 30., 100.])
    parser.add_argument('--cycles', type=int, default=150)
    args = parser.parse_args()
    jobs = [(kv, key, value, args.cycles) for kv in args.kv for key, value in SWEEPS[args.experiment]]
    with ProcessPoolExecutor(8) as pool:
        rows = list(pool.map(one, jobs))
    with open(args.out, 'w') as f:
        json.dump(rows, f, indent=1)
    print('wrote', args.out, len(rows), 'rows')


if __name__ == '__main__':
    main()
