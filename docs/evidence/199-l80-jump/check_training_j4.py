"""j4 <=4M: 0.20 m only; 0->2 / 1->3 / 2->4 same-set plateau comparisons."""
import csv
import json
import math
import shutil
import sys
from pathlib import Path
from check_training_j1 import BLOCK,E,ROOT,judge_text,write_rows,sg,jump_suite_status
from gate_judge import jump_episode_ok
from launch_j4 import verify_inputs,plan
MAX_STEPS=4*BLOCK
HEIGHT=.20


def metrics(rows):
    jumps=[r for r in rows if r['cond'] in ('in_place','in_place_dr')]
    if len(jumps)!=2 or any(r.get('height_target_m')!=HEIGHT for r in jumps):
        raise ValueError('j4 requires two 0.20 m quick rows; legacy 5 cm is not comparable')
    episodes=[e for r in jumps for e in r['episodes']]
    if len(episodes)!=6 or {e['seed'] for e in episodes}!={9300,9301,9310,9311,9312,9313}:
        raise ValueError('incomplete or changed quick seed set')
    if any(not math.isfinite(e['progress']) or e.get('height_target_m')!=HEIGHT for e in episodes):
        raise ValueError('nonfinite progress or wrong episode height')
    return dict(passed=sum(jump_episode_ok(e,HEIGHT) for e in episodes),total=6,
        progress=sum(e['progress'] for e in episodes)/6,falls=sum(e['t_fall'] is not None for e in episodes))


def baseline():
    return metrics([json.loads(l) for l in (E/'eval/j4_prepare/quick_0M.jsonl').read_text().splitlines()])


def stop_reason(block,lengths,history,current,initial,eligible=False):
    if not lengths or any(not math.isfinite(v) for v in lengths):
        return 'stop: missing/nonfinite episode lengths'
    if len(lengths)>=50 and sum(lengths[-10:])<.5*sum(lengths[:10]):
        return 'stop: episode length below half the first ten updates'
    if eligible:return 'stop: full 5a-1 0.20 m + policy regressions passed; supervisor adoption only'
    if block>=2:
        previous=initial if block==2 else history[block-3]
        if current['passed']<=previous['passed'] and current['progress']+1e-12<previous['progress']+.05:
            return 'stop: no success-count gain or >=0.05 progress gain over two blocks at 0.20 m'
    if block>=4:
        if current['passed']==0:return 'stop: zero quick successes at 4M; no extension'
        return 'stop: 4M budget exhausted; no extension'
    return 'continue'


def main():
    run,steps=Path(sys.argv[1]),int(sys.argv[2]);verify_inputs()
    actual=json.loads((run/'args.json').read_text())
    for key,value in plan().items():
        if actual.get(key)!=value:raise ValueError('predeclared run option changed: '+key)
    path=run/'jump_checks.json';history=json.loads(path.read_text()) if path.exists() else []
    block=len(history)+1
    if not 1<=block<=4 or steps!=block*BLOCK:raise ValueError('unexpected interval/budget; no extension')
    with (run/'progress.csv').open() as f:log=list(csv.DictReader(f))
    lengths=[float(r['rollout/ep_len_mean']) for r in log if r.get('rollout/ep_len_mean')]
    for row in log:
        for key,value in row.items():
            if value and (key.startswith('train/') or key=='rollout/ep_rew_mean') and not math.isfinite(float(value)):
                raise ValueError('nonfinite training statistic: '+key)
    initial=baseline();early=stop_reason(1,lengths,[],initial,initial)
    if early!='continue':print(early,flush=True);return 10
    snap=run.with_name(f'{run.name}_{block}M');snap.mkdir()
    for name in ('model.zip','vecnorm.pkl','args.json','progress.csv','initial-policy.json'):shutil.copy2(run/name,snap/name)
    out=E/'eval'/run.name;out.mkdir(parents=True,exist_ok=True);traces=out/f'traces_{block}M'
    walk,runner=ROOT/'sim/rl/runs/l80_w2',ROOT/'sim/rl/runs/l80_r1_6M'
    quick=write_rows(out/f'quick_{block}M.jsonl',
        ({'gate':'jump','run':snap.name,**r} for r in sg.jump(snap,True,walk,trace_dir=traces/'quick',height_target_m=HEIGHT)))
    judge_text(out/f'quick_{block}M_judge.txt',quick)
    entry=dict(block=block,added_steps=steps,**metrics(quick))
    decision=stop_reason(block,lengths,history,entry,initial)
    # Last normal-budget checkpoint gets a full gate even if the final plateau/zero rule also fires.
    if entry['passed']==6 or block==4:
        full=write_rows(out/f'full_{block}M.jsonl',
            ({'gate':'jump','run':snap.name,**r} for r in sg.jump(snap,False,walk,runner,traces/'full',height_target_m=HEIGHT)))
        judge_text(out/f'full_{block}M_judge.txt',full,True)
        eligible,why=jump_suite_status(full,height_target_m=HEIGHT)
        (out/f'adoption_{block}M.json').write_text(json.dumps(dict(eligible_5a1=eligible,height_target_m=HEIGHT,
            reason=why,stage5_passed=False,eligible_5a2=False),indent=2)+'\n')
        decision=stop_reason(block,lengths,history,entry,initial,eligible)
    entry['decision']=decision;history.append(entry)
    path.write_text(json.dumps(history,indent=2)+'\n');print(json.dumps(entry),flush=True)
    return 0 if decision=='continue' else 10


if __name__=='__main__':raise SystemExit(main())
