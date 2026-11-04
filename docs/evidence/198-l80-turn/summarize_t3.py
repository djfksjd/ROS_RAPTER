"""Offline t3 final evaluation summary and one same-seed training comparison chart."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'sim/rl'))
from gate_judge import judge,turn_episode_ok,turn_suite_status,sustained_speed
from turn_rules import TURN_CRITERIA

E=Path(__file__).resolve().parent


def read(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def chart():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    conditions=[(v,sign*r['yaw']) for v,r in TURN_CRITERIA.items() for sign in (1,-1)]
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
    x=np.arange(len(conditions))
    series=[('l80_t1',2,'#777777','t1 +2M'),('l80_t2',2,'#b87532','t2 +2M'),
            ('l80_t3',2,'#7ba9cf','t3 +2M'),('l80_t3',4,'#17496e','t3 +4M final')]
    values=[]
    for run,block,color,label in series:
        rows=read(E/f'eval/{run}/turn_{block}M.jsonl')
        ys,vs=[],[]
        for v,w in conditions:
            row=next(r for r in rows if (r['cmd'],r['yaw_cmd'])==(v,w))
            assert {e['seed'] for e in row['episodes']}=={7200,7201}
            assert row['falls']==0
            mean=lambda k:sum(e[k] for e in row['episodes'])/2
            ys.append(mean('yaw_rate_mean')/w);vs.append(mean('v_mean')/v)
            values.append(dict(run=run,block=block,cmd=v,yaw_cmd=w,yaw_ratio=ys[-1],speed_ratio=vs[-1]))
        for ax,vals in zip(axes,(ys,vs)):
            ax.plot(x,vals,'o-',color=color,lw=1,ms=4,label=label)
    for i,(v,w) in enumerate(conditions):
        r=TURN_CRITERIA[v]
        axes[0].plot([i-.35,i+.35],[1-r['yaw_tol']]*2,color='black',lw=1.5)
        for sign in (-1,1):
            axes[1].plot([i-.35,i+.35],[1+sign*r['v_tol']]*2,color='black',lw=1.5)
    for ax in axes:
        ax.axhline(1,color='.6',lw=.6,ls='--');ax.grid(axis='y',alpha=.2)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Mean yaw / command');axes[0].set_ylim(0,1.35)
    axes[1].set_ylabel('Mean forward speed / command')
    axes[0].legend(frameon=False,ncol=4,fontsize=9)
    axes[1].set_xticks(x,[f'{v:g} / {w:+g}' for v,w in conditions]);axes[1].set_xlabel('Forward / yaw command (m/s / rad/s)')
    fig.suptitle('Development seeds 7200/7201; 8–15 s means; 1 m/s always uses w2\nBlack bars: yaw lower bound / forward bounds. Radius and heading also required.',fontsize=10)
    fig.tight_layout();fig.savefig(E/'media/t1_t2_t3_turn.png',dpi=160);plt.close(fig)
    (E/'eval/t3_chart_data.json').write_text(json.dumps(values,indent=2)+'\n')


def summarize():
    rows=read(E/'eval/t3_full_turn.jsonl');extra=read(E/'eval/t3_extra.jsonl')
    assert len(rows)==36 and sum(r['n'] for r in rows)==240
    assert len(extra)==9 and sum(r['n'] for r in extra)==26
    assert {(r['cmd'],r['mount']) for r in extra if r['cond']=='heat'}=={(v,m) for v in (4,5,6) for m in ('plate','worst')}
    assert len([r for r in extra if r['cond']=='speed' and r['cmd']==8 and r['n']==4])==1
    assert {(r['dir'],r['n']) for r in extra if r['cond']=='lateral'}=={('left',8),('right',8)}
    turns=[]
    for row in rows:
        if row['cond']!='moving_turn':continue
        rule=TURN_CRITERIA[row['cmd']]
        assert row['quick'] is False and {e['seed'] for e in row['episodes']}==set(range(7300,7304))
        assert all(e['load_samples']==10000 for e in row['episodes'])
        failures=[]
        for e in row['episodes']:
            reasons=[]
            checks=[('fall',e['t_fall'] is None),('speed',e['v_mean'] is not None and abs(e['v_mean']-row['cmd'])<=rule['v_tol']*row['cmd']),
                    ('yaw',e['yaw_rate_mean'] is not None and abs(e['yaw_rate_mean']-row['yaw_cmd'])<=rule['yaw_tol']*rule['yaw']),
                    ('radius',e['radius_m'] is not None and 0<=e['radius_m']<=rule['radius']),
                    ('t90',e['t_90_s'] is not None and 0<e['t_90_s']<=rule['t90']),
                    ('heading',math.copysign(1,row['yaw_cmd'])*e['heading_change_rad']>=math.pi/2)]
            reasons=[k for k,ok in checks if not ok]
            assert (not reasons)==turn_episode_ok(row,e)
            if reasons:failures.append(dict(seed=e['seed'],reasons=reasons))
        interval=lambda k:[min(e[k] for e in row['episodes'] if e[k] is not None),max(e[k] for e in row['episodes'] if e[k] is not None)]
        turns.append(dict(cmd=row['cmd'],yaw_cmd=row['yaw_cmd'],policy=row['policy'],
             passed=sum(turn_episode_ok(row,e) for e in row['episodes']),n=row['n'],falls=row['falls'],
             speed=interval('v_mean'),yaw=interval('yaw_rate_mean'),radius=interval('radius_m'),
             t90=interval('t_90_s') if any(e['t_90_s'] is not None for e in row['episodes']) else None,
             failures=failures))
    before=json.loads((E/'eval/t3_final_inputs.json').read_text())
    after={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in before}
    assert before==after
    suites={}
    for gate in ('stand','walk','switch'):
        rr=[r for r in rows if r['gate']==gate]
        judged=[r for r in rr if judge(r)[0] is not None]
        suites[gate]=dict(conditions=len(rr),episodes=sum(r['n'] for r in rr),
            passing_conditions=sum(judge(r)[0] is True for r in judged),judged_conditions=len(judged),
            falls=sum(r['falls'] for r in rr),judged_falls=sum(r['falls'] for r in judged))
    result=dict(full_rows=len(rows),extra_rows=len(extra),episodes=266,turns=turns,
        turn_passing_episodes=sum(r['passed'] for r in turns),turn_falls=sum(r['falls'] for r in turns),
        full_suite=dict(zip(('passed','reason'),turn_suite_status(rows))),regressions=suites,
        extra_sustained_command=sustained_speed(extra),
        judgments=[dict(gate=r['gate'],cond=r['cond'],cmd=r.get('cmd'),yaw_cmd=r.get('yaw_cmd'),
                        direction=r.get('dir'),dv=r.get('dv'),mount=r.get('mount'),passed=judge(r)[0],reason=judge(r)[1]) for r in rows+extra],
        inputs_unchanged=True,after=after,decision='report B/C; no extension in t3 pre-declaration')
    result['switch_limitation'] = dict(harness_uses_walk_environment=True,
        walk_hip_yaw_scale=.3, standalone_t3_hip_yaw_scale=.5,
        scale_preserving_policy_switch_verified=False)
    (E/'eval/t3_final_summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('after','judgments')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--chart-only',action='store_true');a=p.parse_args()
    chart()
    if not a.chart_only:summarize()
