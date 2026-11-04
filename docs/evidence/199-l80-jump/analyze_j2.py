"""Reconstruct every landing and 1 ms hold failure without changing the frozen judge."""
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
from run_env import RunEnv
from train_run import env_kwargs
from gate_judge import jump_episode_ok,judge,jump_suite_status
from jump_probe import intervals,longest
from check_training_j2 import stop_reason,BLOCK,MAX_STEPS
from check_training_j1 import metrics
from launch_j2 import plan


def initial_height(seed,dr):
    args=json.loads((ROOT/'sim/rl/runs/l80_j2/args.json').read_text())
    kw={**env_kwargs(args),'episode_s':14.,'level':0.,'disturb_items':[],'roll_impulse':None,
        'yaw_impulse':None,'init_speed':False,'cmd_max':[0.,0.,0.],'cmd_mix':None,
        'jump':{'auto_request_s':2.,'flight_bonus':0.,'success_bonus':0.}}
    env=RunEnv(**kw,randomize=dr,seed=seed);env.init_seed=seed;env.reset(seed=seed)
    height=env.h_nom+.03;env.close();return height


def episode(row,e):
    with np.load(e['raw_trace']) as raw:a=dict(raw)
    t=a['t'];requested=t>=e['request_time_s'];off=(a['feet_n']==0).all(axis=1)&requested
    events=[]
    for s,b in intervals(off):
        if b>=len(t):continue
        rise=float(a['xyz'][s:b,2].max()-e['standing_height_m']);duration=(b-s)*.001
        distance=float(np.linalg.norm(a['xyz'][b,:2]-a['xyz'][s,:2]))
        events.append(dict(takeoff_s=float(t[s]),touchdown_s=float(t[b]),flight_s=duration,
            apex_rise_m=rise,landing_error_m=distance,dx_m=float(a['xyz'][b,0]-a['xyz'][s,0]),
            dy_m=float(a['xyz'][b,1]-a['xyz'][s,1]),
            takeoff_vxy_mps=float(np.linalg.norm(a['velocity'][s,:2])),
            touchdown_vxy_mps=float(np.linalg.norm(a['velocity'][b,:2])),
            q=min(duration/.1,1)*float(np.clip(rise/.05,0,1))))
    qualified=[f for f in events if f['flight_s']>=.1 and f['apex_rise_m']>=.05 and f['landing_error_m']<=.3]
    chosen=qualified[0] if qualified else max(events,key=lambda f:(f['q'],f['apex_rise_m']),default=None)
    # For an incomplete jump, distinguish an observed landing from the task's null (not qualified) field.
    primary='pass' if jump_episode_ok(e) else 'fall' if e['t_fall'] is not None else (
        'flight' if chosen is None or chosen['flight_s']<.1 else 'apex' if chosen['apex_rise_m']<.05 else
        'landing' if chosen['landing_error_m']>.3 else 'hold' if e['standing_hold_s']<1 else 'handoff')
    out=dict(cond=row['cond'],seed=e['seed'],passed=jump_episode_ok(e),primary_failure=primary,
        gate_episode=e,representative_flight=chosen,flights=events,hold=None,
        criterion_failures=dict(flight=bool(chosen is None or chosen['flight_s']<.1),
            apex=bool(chosen is None or chosen['apex_rise_m']<.05),
            landing_distance=bool(chosen is not None and chosen['landing_error_m']>.3),
            hold=bool(e['standing_hold_s']<1.),fall=e['t_fall'] is not None),
        hold_entered=bool(qualified))
    if qualified:
        end=e['success_time_s'] if e['success_time_s'] is not None else e['elapsed_s']
        mask=(t>=qualified[0]['touchdown_s'])&(t<=end+1e-9)
        h=initial_height(e['seed'],row['cond']=='in_place_dr');weight=float(a['weight_n'])
        tests=dict(foot_load=(a['feet_force_n']>.15*weight).all(axis=1),
            speed=np.linalg.norm(a['velocity'],axis=1)<.5,tilt=a['tilt']<.3,
            height=a['xyz'][:,2]-a['ground_z']>.8*h,body=~a['body_hit'])
        stable=np.logical_and.reduce(list(tests.values()))
        measured=longest(stable&mask)
        assert abs(measured-e['standing_hold_s'])<1e-9,(e['seed'],measured,e['standing_hold_s'])
        out['hold']=dict(observed_s=float(mask.sum()*.001),longest_s=measured,
            failure_samples={k:int((~v&mask).sum()) for k,v in tests.items()},
            unstable_samples=int((~stable&mask).sum()),stable_share=float(stable[mask].mean()),
            both_feet_off_s=float((off&mask).sum()*.001),
            max_horizontal_speed=float(np.linalg.norm(a['velocity'][mask,:2],axis=1).max()),
            max_vertical_speed=float(np.abs(a['velocity'][mask,2]).max()),
            max_tilt_rad=float(a['tilt'][mask].max()),
            horizontal_displacement_m=float(np.linalg.norm(a['xyz'][np.flatnonzero(mask)[-1],:2]-a['xyz'][np.flatnonzero(mask)[0],:2])))
    return out


def audit():
    run=ROOT/'sim/rl/runs/l80_j2';h=json.loads((run/'jump_checks.json').read_text())
    actual=json.loads((run/'args.json').read_text());assert all(actual.get(k)==v for k,v in plan().items())
    with (run/'progress.csv').open() as f:log=list(csv.DictReader(f))
    lengths=[float(r['rollout/ep_len_mean']) for r in log if r.get('rollout/ep_len_mean')]
    finite=[float(v) for r in log for k,v in r.items() if v and (k.startswith('train/') or k=='rollout/ep_rew_mean')]
    assert np.isfinite(finite).all()
    initial_steps=json.loads((run/'initial-policy.json').read_text())['steps']
    final_steps=int(log[-1]['time/total_timesteps'])
    assert final_steps==initial_steps+h[-1]['added_steps']
    for i,entry in enumerate(h):
        rows=[json.loads(l) for l in (E/f'eval/l80_j2/quick_{i+1}M.jsonl').read_text().splitlines()]
        m=metrics(rows);assert all(m[k]==entry[k] for k in m)
        assert entry['added_steps']==(i+1)*BLOCK
        assert entry['decision']==stop_reason(i+1,lengths,h[:i],m)
    hashes={n:hashlib.sha256((run/n).read_bytes()).hexdigest() for n in ('model.zip','vecnorm.pkl','args.json')}
    assert all(v==hashlib.sha256((run.with_name('l80_j2_4M')/n).read_bytes()).hexdigest() for n,v in hashes.items())
    return dict(checks=h,budget=MAX_STEPS,final_hashes=hashes,matches_4M_snapshot=True,
        initial_steps=initial_steps,final_steps=final_steps,
        episode_length_range=[min(lengths),max(lengths)],training_statistics_finite=True,
        unused_steps=4_000_000-h[-1]['added_steps'])


def main():
    out=E/'eval/j2_full';rows=[json.loads(l) for l in (out/'gate.jsonl').read_text().splitlines()]
    episodes=[episode(r,e) for r in rows if r['gate']=='jump' and r['cond'] in ('in_place','in_place_dr') for e in r['episodes']]
    assert len(episodes)==32
    result=dict(audit=audit(),episodes=episodes,
        failure_distribution=dict(collections.Counter(e['primary_failure'] for e in episodes)),
        criterion_failure_counts={key:sum(e['criterion_failures'][key] for e in episodes)
                                  for key in ('flight','apex','landing_distance','hold','fall')},
        conditions=[dict(gate=r['gate'],cond=r['cond'],cmd=r.get('cmd'),n=r['n'],judge=judge(r)) for r in rows],
        suite=jump_suite_status(rows),rollouts=sum(r['n'] for r in rows))
    (out/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(result['failure_distribution'],result['suite'],result['rollouts'])
    for e in episodes:
        if not e['passed']:print(e['seed'],e['primary_failure'],e['representative_flight'],e['hold'])


if __name__=='__main__':main()
