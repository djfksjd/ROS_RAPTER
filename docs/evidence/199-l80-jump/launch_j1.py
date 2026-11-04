"""Frozen j1 launch plan. --print-command is read-only; no training occurs on import."""
import argparse
import hashlib
import json
import shlex
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
E = Path(__file__).resolve().parent
INIT = ROOT/'sim/rl/runs/l80_s1p'
JUMP = dict(auto_request_s=2., flight_bonus=10., success_bonus=30.)


def plan():
    original = json.loads((INIT/'args.json').read_text())
    args = {**original, 'name': 'l80_j1', 'steps': 8_000_000, 'init': 'sim/rl/runs/l80_s1p/model.zip',
            'episode_s': 12., 'jump': json.dumps(JUMP), 'resume_action_std': None,
            'checkpoint_check': 'docs/evidence/199-l80-jump/check_training_j1.py'}
    return args


def command(args):
    out = [str(ROOT/'.venv-sim/bin/python'), '-u', str(ROOT/'sim/rl/train_run.py')]
    for key, value in args.items():
        if value is None or value is False:
            continue
        out.append('--'+key.replace('_', '-'))
        if value is not True:
            out.extend(str(x) for x in value) if isinstance(value, list) else out.append(str(value))
    return out


def verify_inputs():
    for path, digest in json.loads((E/'j1_inputs.json').read_text()).items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != digest:
            raise ValueError(f'predeclared init/config hash mismatch: {path}')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--print-command', action='store_true'); a = parser.parse_args()
    verify_inputs(); args = plan(); cmd = command(args)
    if a.print_command:
        print(shlex.join(cmd)); return 0
    if (ROOT/'sim/rl/runs'/args['name']).exists():
        raise FileExistsError('l80_j1 exists; no overwrite or implicit extension')
    os.chdir(ROOT)
    os.execv(cmd[0], cmd)  # keep the supervisor PID as the trainer; no orphan training child


if __name__ == '__main__': raise SystemExit(main())
