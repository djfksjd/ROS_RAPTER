"""j3: positive completion bonus 30 -> 50 only. <=2M, no extension; never learn on import."""
import argparse
import hashlib
import json
import os
import shlex
from pathlib import Path
from launch_j1 import command
ROOT=Path(__file__).resolve().parents[3]
E=Path(__file__).resolve().parent
INIT=ROOT/'sim/rl/runs/l80_j2'


def plan():
    original=json.loads((INIT/'args.json').read_text())
    jump={**json.loads(original['jump']),'success_bonus':50.}
    return {**original,'name':'l80_j3','steps':2_000_000,'init':'sim/rl/runs/l80_j2/model.zip',
            'jump':json.dumps(jump),'checkpoint_check':'docs/evidence/199-l80-jump/check_training_j3.py'}


def verify_inputs():
    for path,expected in json.loads((E/'j3_inputs.json').read_text()).items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=expected:
            raise ValueError('predeclared input/code hash mismatch: '+path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--print-command',action='store_true');args=p.parse_args()
    verify_inputs();config=plan();cmd=command(config)
    if args.print_command:print(shlex.join(cmd));return 0
    if (ROOT/'sim/rl/runs'/config['name']).exists():
        raise FileExistsError('l80_j3 exists; no overwrite or implicit extension')
    os.chdir(ROOT);os.execv(cmd[0],cmd)


if __name__=='__main__':raise SystemExit(main())
