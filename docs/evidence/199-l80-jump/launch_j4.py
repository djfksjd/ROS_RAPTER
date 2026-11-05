"""j4: 0.20 m task plus a zero-initialized landing-phase observation. No learning on import."""
import argparse
import hashlib
import json
import os
import shlex
from pathlib import Path
from launch_j1 import command
ROOT=Path(__file__).resolve().parents[3]
E=Path(__file__).resolve().parent
INIT=ROOT/'sim/rl/runs/l80_j3'


def plan():
    original=json.loads((INIT/'args.json').read_text())
    jump={**json.loads(original['jump']),'height_target_m':.20,'obs_phase':True}
    return {**original,'name':'l80_j4','steps':4_000_000,'init':'sim/rl/runs/l80_j3/model.zip',
            'jump':json.dumps(jump),'checkpoint_check':'docs/evidence/199-l80-jump/check_training_j4.py'}


def verify_inputs():
    for path,expected in json.loads((E/'j4_inputs.json').read_text()).items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=expected:
            raise ValueError('predeclared input/code hash mismatch: '+path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--print-command',action='store_true');args=p.parse_args()
    verify_inputs();config=plan();cmd=command(config)
    if args.print_command:print(shlex.join(cmd));return 0
    if (ROOT/'sim/rl/runs'/config['name']).exists():
        raise FileExistsError('l80_j4 exists; no overwrite or implicit extension')
    os.chdir(ROOT);os.execv(cmd[0],cmd)


if __name__=='__main__':raise SystemExit(main())
