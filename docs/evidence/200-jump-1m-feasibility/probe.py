"""195 finite CEM/replay episodes, no learning; modified configs live only here."""
import copy
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import mujoco
from study import ROOT,E,REV,RZ,hashes,dump
sys.path.insert(0,str(ROOT/'docs/evidence/199-l80-jump'))
import jump_probe as jp
from audit_current import CurrentRecorder
from run_env import RunEnv
from train_run import env_kwargs


class Recorder(CurrentRecorder):
    def __init__(self,env):
        super().__init__(env)
        self.rows['com']=[]
    def sample(self,env,dt):
        super().sample(env,dt)
        self.rows['com'].append(np.sum(env.model.body_mass[:,None]*env.data.xipos,axis=0)/env.model.body_mass.sum())


def configs():
    sweep=json.loads((E/'eval/sweep.json').read_text());r=sweep['best_ratio']
    variants={}
    for name,scale in [('frozen',1.),('belts',1.),('long_belts',1.25)]:
        rev,rz=copy.deepcopy(REV),copy.deepcopy(RZ)
        if name!='frozen':
            for axis,g in [('hip_pitch',r['hip_ratio']),('knee_pitch',r['knee_ratio'])]:
                rev['axes'][axis]['ratio']=g
                rz['motor']['assign'][axis]['external_ratio']=g
        if scale!=1:
            for k in ('femur','tibia','meta'):rev['leg'][k]*=scale
            for k in ('a','b','c'):rev['fourbar'][k]*=scale
        rev['spec_file']='../../../../docs/design/motor_candidates.json'
        rz['motor']['spec_file']='../../../../docs/design/motor_candidates.json'
        # Strip stale design comments so an isolated variant is not mistaken for a frozen design.
        rev['_comment']=f'Evidence 200 isolated {name}; no adoption. Physical knee band retained even with belt replacement.'
        rz['_comment_l80']=f'Evidence 200 isolated {name}; all enabled realism blocks unchanged.'
        for axis in rev['axes'].values():axis.pop('why',None)
        rev['fourbar'].pop('why',None);rev['leg'].pop('why',None)
        files=[]
        for suffix,cfg in [('model',rev),('realism',rz)]:
            file=E/'configs'/f'{name}_{suffix}.json';file.write_text(json.dumps(cfg,indent=2)+'\n');files.append(file)
        variants[name]=files
    return variants


def make(paths):
    args=json.loads((jp.RUN/'args.json').read_text())
    kw={**env_kwargs(args),'level':0.,'episode_s':15.,'disturb_items':[], 'roll_impulse':None,
        'yaw_impulse':None,'cmd_max':[0.,0.,0.],'cmd_mix':None,'init_speed':False,'terrain':None,
        'rev3_config':str(paths[0]),'realism':str(paths[1])}
    return RunEnv(**kw,randomize=False,seed=0)


def main():
    out=E/'eval/probes.jsonl'
    if out.exists():raise RuntimeError('Do not overwrite CEM evidence')
    start=time.process_time();before=hashes();variants=configs()
    sources=[json.loads(l) for l in (jp.E/'eval/search.jsonl').read_text().splitlines()]
    source=max(sources,key=lambda r:r['result']['event']['apex_rise_m'] if r['result']['event'] else -100)
    all_summary=[]
    with out.open('x') as f:
        for name,paths in variants.items():
            env=make(paths);rng=np.random.default_rng(20001);mean=np.array(source['x']);std=np.full(len(mean),.25)
            best=None;top_score=-np.inf;count=0;valid_flights=0;falls=0
            for gen in range(-1,4):
                xs=[np.array(source['x'])] if gen<0 else np.clip(rng.normal(mean,std,(16,len(mean))),-1,1)
                if gen>=0:xs[0]=mean
                scores=[]
                for x in xs:
                    if time.process_time()-start>1200:raise RuntimeError('CPU guard')
                    with patch.object(jp,'Recorder',Recorder):result,a=jp.rollout(env,x,capture=True)
                    assert all(np.isfinite(v).all() for v in a.values())
                    ev=result['event'];s=jp.score(result,'vertical');scores.append(s);count+=1
                    valid_flights+=int(ev is not None);falls+=int(result['t_fall'] is not None)
                    row=dict(variant=name,generation=gen,candidate=count,x=x.tolist(),score=s,result=result)
                    if ev:
                        i,j=ev['start'],ev['stop'];ref=min(np.searchsorted(a['t'],.2),len(a['t'])-1)
                        row['com_rise_m']=float(a['com'][i:j,2].max()-a['com'][ref,2])
                        row['takeoff_velocity_mps']=a['velocity'][i].tolist()
                    f.write(json.dumps(row)+'\n');f.flush()
                    metric=ev['apex_rise_m'] if ev else -100
                    if best is None or metric>best.get('height',-100):
                        best=dict(height=metric,**row,loads=jp.loads(env,a,ev),
                          max_current_using_clamp_ratio=float((a['current_using_clamp_state_a']/a['current_peak_a']).max()))
                        np.savez_compressed(E/'eval'/f'{name}_best.npz',**a)
                    top_score=max(top_score,s)
                if gen>=0:
                    elite=np.asarray(xs)[np.argsort(scores)[-4:]]
                    mean=.25*mean+.75*elite.mean(axis=0);std=np.maximum(.10,.25*std+.75*elite.std(axis=0))
            all_summary.append(dict(variant=name,episodes=count,mass_kg=float(env.model.body_mass.sum()),
                valid_flights=valid_flights,falls=falls,best=best,best_score=top_score))
            env.close();print(name,valid_flights,falls,best['height'],flush=True)
    assert before==hashes()
    dump('probe_summary.json',dict(source=dict(mode=source['mode'],candidate=source['candidate']),
        variants=all_summary,cpu_s=time.process_time()-start,input_hashes=before,hashes_unchanged=True))


if __name__=='__main__':main()
