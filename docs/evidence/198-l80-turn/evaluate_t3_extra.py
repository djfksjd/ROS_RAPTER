"""Finite, read-only extra t3 gates: 8 m/s, six 600 s heat runs, lateral kicks.

Required stage-4 regressions are already in stage_gates.py turn. This script uses
its unchanged run_rollout; no training, altered thresholds or runs/ writes.
"""
import concurrent.futures
import json
import multiprocessing
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'sim/rl'))
import torch
from stage_gates import run_rollout

EVIDENCE=Path(__file__).resolve().parent
RUN=ROOT/'sim/rl/runs/l80_t3'


def evaluate(task):
    torch.set_num_threads(1)
    kind,value=task
    common=dict(gate='run',run=RUN.name,cond=kind)
    if kind=='heat':
        cmd,mount=value
        fell,v,der,hot=run_rollout(RUN,6800,cmd,600.,mount=mount)
        return dict(**common,cmd=cmd,mount=mount,n=1,falls=int(fell is not None),t_fall=fell,
                    min_derate=der,t_winding_max_c=hot,v_mean=round(float(v[500:].mean()),3))
    if kind=='speed':
        rows=[]
        for seed in range(6700,6704):
            fell,v,der,hot=run_rollout(RUN,seed,8.,20.)
            rows.append(dict(seed=seed,t_fall=fell,v_mean=round(float(v[500:].mean()),3) if fell is None else None,
                             min_derate=der,t_winding_max_c=hot))
        return dict(**common,cmd=8.,n=4,falls=sum(r['t_fall'] is not None for r in rows),episodes=rows)
    assert kind=='lateral'
    kick=(0.,.5 if value=='left' else -.5)
    rows=[]
    for seed in range(6900,6908):
        fell,_,_,_=run_rollout(RUN,seed,4.,15.,kick=kick)
        rows.append(dict(seed=seed,t_fall=fell))
    return dict(**common,dir=value,cmd=4.,n=8,falls=sum(r['t_fall'] is not None for r in rows),episodes=rows)


def main():
    out=EVIDENCE/'eval/t3_extra.jsonl'
    # Submit long jobs first; each of three processes evaluates disjoint episodes.
    tasks=[('heat',(v,m)) for v in (4.,5.,6.) for m in ('plate','worst')]
    tasks += [('speed',8.),('lateral','left'),('lateral','right')]
    with out.open('x') as f:
        with concurrent.futures.ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('spawn')) as pool:
            pending={pool.submit(evaluate,t):t for t in tasks}
            for future in concurrent.futures.as_completed(pending):
                row=future.result()
                f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()
                print(json.dumps(dict(completed=pending[future],falls=row['falls'])),flush=True)


if __name__=='__main__':
    main()
