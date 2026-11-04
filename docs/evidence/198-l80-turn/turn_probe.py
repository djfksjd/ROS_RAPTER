"""Evidence 198: no-training moving-turn baseline, 24 conditions x 4 seeds.
Run from the repository root with .venv-sim/bin/python; JSON lines on stdout.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'sim/rl'))
from turn_eval import turn_condition


def main():
    for speed in (1., 2., 4., 6.):
        d = ROOT/'sim/rl/runs'/('l80_w2' if speed == 1. else 'l80_r1_6M')
        for magnitude in (.5, 1., 1.5):
            for sign in (1., -1.):
                row = turn_condition(d, speed, sign*magnitude, range(7200, 7204))
                print(json.dumps(dict(gate='turn_baseline', **row), allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
