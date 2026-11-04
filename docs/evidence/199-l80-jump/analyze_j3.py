"""Full j3 gate audit, per-episode criteria and paired j2 comparison. No learning."""
import argparse
import collections
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
E=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'sim/rl'))
from analyze_j2 import episode
from gate_judge import judge,jump_suite_status
from check_training_j1 import metrics,BLOCK
from check_training_j3 import baseline,stop_reason,MAX_STEPS
from launch_j3 import plan,verify_inputs


def audit():
    verify_inputs()
    run=ROOT/'sim/rl/runs/l80_j3';history=json.loads((run/'jump_checks.json').read_text())
    actual=json.loads((run/'args.json').read_text());assert all(actual.get(k)==v for k,v in plan().items())
    with (run/'progress.csv').open() as f:log=list(csv.DictReader(f))
    lengths=[float(r['rollout/ep_len_mean']) for r in log if r.get('rollout/ep_len_mean')]
    stats=[float(v) for r in log for k,v in r.items() if v and (k.startswith('train/') or k=='rollout/ep_rew_mean')]
    assert np.isfinite(stats).all()
    for i,entry in enumerate(history):
        rows=[json.loads(l) for l in (E/f'eval/l80_j3/quick_{i+1}M.jsonl').read_text().splitlines()]
        m=metrics(rows);assert all(entry[k]==v for k,v in m.items())
        assert entry['added_steps']==(i+1)*BLOCK
        assert entry['decision']==stop_reason(i+1,lengths,m,baseline())
    initial=json.loads((run/'initial-policy.json').read_text())['steps'];final=int(log[-1]['time/total_timesteps'])
    assert final==initial+history[-1]['added_steps'] and history[-1]['added_steps']==MAX_STEPS
    hashes={n:hashlib.sha256((run/n).read_bytes()).hexdigest() for n in ('model.zip','vecnorm.pkl','args.json')}
    assert all(v==hashlib.sha256((run.with_name('l80_j3_2M')/n).read_bytes()).hexdigest() for n,v in hashes.items())
    return dict(checks=history,baseline=baseline(),initial_steps=initial,final_steps=final,
        added_steps=history[-1]['added_steps'],budget=MAX_STEPS,unused_steps=2_000_000-MAX_STEPS,
        input_hashes_verified=26,final_hashes=hashes,matches_2M_snapshot=True,
        training_statistics_finite=True,episode_length_range=[min(lengths),max(lengths)])


def summarize(rows):
    episodes=[episode(r,e,ROOT/'sim/rl/runs/l80_j3') for r in rows if r['gate']=='jump'
              and r['cond'] in ('in_place','in_place_dr') for e in r['episodes']]
    assert len(episodes)==32 and sum(r['n'] for r in rows)==240
    counts=dict(collections.Counter(e['primary_failure'] for e in episodes))
    return dict(audit=audit(),episodes=episodes,failure_distribution=counts,
        criterion_failure_counts={k:sum(e['criterion_failures'][k] for e in episodes)
                                  for k in ('flight','apex','landing_distance','hold','fall')},
        conditions=[dict(gate=r['gate'],cond=r['cond'],cmd=r.get('cmd'),n=r['n'],judge=judge(r)) for r in rows],
        suite=jump_suite_status(rows),rollouts=sum(r['n'] for r in rows))


def compare(old,new):
    old_by={e['seed']:e for e in old['episodes']};pairs=[]
    for b in new['episodes']:
        a=old_by[b['seed']]
        pairs.append(dict(seed=b['seed'],cond=b['cond'],j2_pass=a['passed'],j3_pass=b['passed'],
            j2_primary=a['primary_failure'],j3_primary=b['primary_failure'],
            j2_flight=a['representative_flight'],j3_flight=b['representative_flight'],
            j2_hold_s=a['gate_episode']['standing_hold_s'],j3_hold_s=b['gate_episode']['standing_hold_s']))
    return dict(j2_primary=old['failure_distribution'],j3_primary=new['failure_distribution'],
        j2_criteria=old['criterion_failure_counts'],j3_criteria=new['criterion_failure_counts'],
        newly_passed=[p['seed'] for p in pairs if not p['j2_pass'] and p['j3_pass']],
        newly_failed=[p['seed'] for p in pairs if p['j2_pass'] and not p['j3_pass']],pairs=pairs)


def reproduce(rows,automatic):
    assert len(rows)==len(automatic)
    def clean(row):
        row=json.loads(json.dumps(row));row.pop('run',None)
        for e in row.get('episodes',[]):e.pop('raw_trace',None)
        return row
    assert [clean(r) for r in rows]==[clean(r) for r in automatic]
    count=samples=0
    for row,old in zip(rows,automatic):
        if row['gate']!='jump' or row['cond'] not in ('in_place','in_place_dr'):continue
        for a,b in zip(row['episodes'],old['episodes']):
            with np.load(a['raw_trace']) as aa,np.load(b['raw_trace']) as bb:
                assert set(aa.files)==set(bb.files)
                for key in aa.files:np.testing.assert_array_equal(aa[key],bb[key],err_msg=f"{a['seed']} {key}")
                samples+=len(aa['t']);count+=1
    return dict(exact_rows=True,rollouts=240,exact_raw_episodes=count,raw_samples=samples,
        ignored_metadata=['run directory name','raw trace path'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--automatic',action='store_true');args=p.parse_args()
    out=E/'eval/j3_full';automatic=[json.loads(l) for l in (E/'eval/l80_j3/full_2M.jsonl').read_text().splitlines()]
    rows=automatic if args.automatic else [json.loads(l) for l in (out/'gate.jsonl').read_text().splitlines()]
    result=summarize(rows)
    old=json.loads((E/'eval/j2_full/analysis.json').read_text());result['comparison']=compare(old,result)
    if not args.automatic:result['reproduction']=reproduce(rows,automatic)
    path=out/('automatic_analysis.json' if args.automatic else 'analysis.json')
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(result['failure_distribution'],result['suite'])
    print('new passes',result['comparison']['newly_passed'],'new failures',result['comparison']['newly_failed'])
    for e in result['episodes']:
        if not e['passed']:print(e['seed'],e['primary_failure'],e['representative_flight'],e['hold'])


if __name__=='__main__':main()
