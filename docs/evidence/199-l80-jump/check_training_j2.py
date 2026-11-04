"""j2 <=4M contract. Reuse j1's unchanged judge, fixed quick seeds and stagnation rule.

Called by the supervisor-launched trainer only. Exit 10 stops; any other error aborts.
"""
import csv
import json
import math
import shutil
import sys
from pathlib import Path
from check_training_j1 import (BLOCK, E, ROOT, metrics, judge_text, write_rows,
                               stop_reason as j1_stop_reason, sg, jump_suite_status)
from launch_j2 import plan, verify_inputs

MAX_STEPS = 4*BLOCK


def stop_reason(block, lengths, history, current, eligible=False):
    reason = j1_stop_reason(block, lengths, history, current, eligible)
    if reason != 'continue':
        return reason
    if block >= 4:
        return 'stop: 4M budget exhausted; no extension'
    return 'continue'


def main():
    run,steps=Path(sys.argv[1]),int(sys.argv[2])
    verify_inputs()
    actual=json.loads((run/'args.json').read_text())
    for key,value in plan().items():
        if actual.get(key) != value:
            raise ValueError('predeclared run option changed: '+key)
    path=run/'jump_checks.json';history=json.loads(path.read_text()) if path.exists() else []
    block=len(history)+1
    if not 1 <= block <= 4 or steps != block*BLOCK:
        raise ValueError('unexpected interval/budget; no extension')
    with (run/'progress.csv').open() as f:log=list(csv.DictReader(f))
    lengths=[float(r['rollout/ep_len_mean']) for r in log if r.get('rollout/ep_len_mean')]
    for row in log:
        for key,value in row.items():
            if value and (key.startswith('train/') or key=='rollout/ep_rew_mean') and not math.isfinite(float(value)):
                raise ValueError('nonfinite training statistic: '+key)
    early=stop_reason(1,lengths,[],dict(passed=0,progress=0.))
    if early != 'continue':
        print(early,flush=True);return 10
    snap=run.with_name(f'{run.name}_{block}M');snap.mkdir()
    for name in ('model.zip','vecnorm.pkl','args.json','progress.csv','initial-policy.json'):
        shutil.copy2(run/name,snap/name)
    out=E/'eval'/run.name;out.mkdir(parents=True,exist_ok=True);traces=out/f'traces_{block}M'
    walk,runner=ROOT/'sim/rl/runs/l80_w2',ROOT/'sim/rl/runs/l80_r1_6M'
    quick=write_rows(out/f'quick_{block}M.jsonl',
        ({'gate':'jump','run':snap.name,**r} for r in sg.jump(snap,True,walk,trace_dir=traces/'quick')))
    judge_text(out/f'quick_{block}M_judge.txt',quick)
    entry=dict(block=block,added_steps=steps,**metrics(quick))
    decision=stop_reason(block,lengths,history,entry)
    if entry['passed']==6 or (block==4 and decision=='stop: 4M budget exhausted; no extension'):
        full=write_rows(out/f'full_{block}M.jsonl',
            ({'gate':'jump','run':snap.name,**r} for r in sg.jump(snap,False,walk,runner,traces/'full')))
        judge_text(out/f'full_{block}M_judge.txt',full,True)
        eligible,why=jump_suite_status(full)
        (out/f'adoption_{block}M.json').write_text(json.dumps(dict(eligible_5a=eligible,reason=why,stage5_passed=False),indent=2)+'\n')
        decision=stop_reason(block,lengths,history,entry,eligible)
    entry['decision']=decision;history.append(entry)
    path.write_text(json.dumps(history,indent=2)+'\n');print(json.dumps(entry),flush=True)
    return 0 if decision=='continue' else 10


if __name__=='__main__':raise SystemExit(main())
