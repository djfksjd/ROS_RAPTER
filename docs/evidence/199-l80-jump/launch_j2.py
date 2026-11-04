"""j2: one task-conditional removal of anti-flight costs; no training on import."""
import argparse
import hashlib
import json
import os
import shlex
from pathlib import Path
from launch_j1 import command

ROOT = Path(__file__).resolve().parents[3]
E = Path(__file__).resolve().parent
INIT = ROOT/'sim/rl/runs/l80_j1'


def plan():
    original = json.loads((INIT/'args.json').read_text())
    jump = {**json.loads(original['jump']), 'suspend_stand_penalties': True}
    return {**original, 'name':'l80_j2', 'steps':4_000_000,
        'init':'sim/rl/runs/l80_j1/model.zip', 'jump':json.dumps(jump),
        'checkpoint_check':'docs/evidence/199-l80-jump/check_training_j2.py'}


def verify_inputs():
    for path, expected in json.loads((E/'j2_inputs.json').read_text()).items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != expected:
            raise ValueError('predeclared input/code hash mismatch: '+path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--print-command',action='store_true');a=p.parse_args()
    verify_inputs();args=plan();cmd=command(args)
    if a.print_command:
        print(shlex.join(cmd));return 0
    if (ROOT/'sim/rl/runs'/args['name']).exists():
        raise FileExistsError('l80_j2 exists; no overwrite or implicit extension')
    os.chdir(ROOT);os.execv(cmd[0],cmd)


if __name__=='__main__':raise SystemExit(main())
