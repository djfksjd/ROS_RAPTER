"""No training: a fixed walk handoff at the first qualifying landing; unchanged 5a criteria."""
import argparse
import json
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[3]
E=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'sim/rl'))
from jump_eval import jump_conditions
from check_training_j1 import write_rows,judge_text,metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--full',action='store_true');args=p.parse_args()
    torch.set_num_threads(1)
    run=ROOT/'sim/rl/runs/l80_j2';walk=ROOT/'sim/rl/runs/l80_w2'
    out=E/'eval/j2_landing_walk';out.mkdir(exist_ok=True)
    tag='full' if args.full else 'quick'
    rows=write_rows(out/f'{tag}.jsonl',jump_conditions(run,walk,not args.full,out/f'{tag}_raw',landing_walk=True))
    judge_text(out/f'{tag}_judge.txt',rows,full=args.full)
    if not args.full:print(json.dumps(metrics(rows)))
    else:print('Full 32 jump episodes only; retained-policy regressions must be included for a suite verdict.')


if __name__=='__main__':main()
