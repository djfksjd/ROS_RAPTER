"""Evidence 192: strict get-up probe of a fixed-start run (training leak off, 10 s, deterministic): the run's fixed
fallen start with randomize off (1) and on (10), and 10 unseen back starts (getup start_seed removed).
Usage: .venv-sim/bin/python docs/evidence/192-knee-fold-fourbar/strict_rate.py sim/rl/runs/r3_gxa_back"""
import sys, json, numpy as np
sys.path.insert(0,'sim/rl')
from pathlib import Path
import stage_gates as sg
d=Path(sys.argv[1]); a=json.loads((d/'args.json').read_text()); g=json.loads(a['getup'])
def run(rnd, s, gg):
    kw={**sg.env_kwargs(a),'level':0.,'episode_s':15.,'disturb_items':[],'roll_impulse':None,'yaw_impulse':None,'cmd_max':[0.,0.,0.],'init_speed':False,'getup':{**gg,'leak':None}}
    env=sg.RunEnv(**kw,randomize=rnd,seed=8000+s); env.init_seed=8000+s
    model,venv=sg.load(str(d/'model.zip'),env); venv.reset()
    obs=venv.normalize_obs(env._obs()[None]); t=None; best=9.; hmax=0.
    for k in range(500):
        raw,_,term,_,info=env.step(model.predict(obs,deterministic=True)[0][0]); obs=venv.normalize_obs(raw[None])
        R=env.data.xmat[env.base].reshape(3,3); best=min(best,float(np.arccos(np.clip(R[2,2],-1,1)))); hmax=max(hmax,float(env.data.xpos[env.base][2]))
        if info['getup']['success']: t=round((k+1)*.02,2); break
    return dict(randomize=rnd,seed=s,t_up=t,min_tilt=round(best,2),h_max=round(hmax,2))
fixed=[run(False,0,g)]+[run(True,s,g) for s in range(10)]
g2={k:v for k,v in g.items() if k!='start_seed'}
unseen=[run(False,s,g2) for s in range(10)]
ok=lambda L: sum(r['t_up'] is not None for r in L)
print(json.dumps(dict(run=d.name, fixed_nominal=ok(fixed[:1]), fixed_dr=ok(fixed[1:]), unseen_back=ok(unseen), fixed=fixed, unseen=unseen)))
