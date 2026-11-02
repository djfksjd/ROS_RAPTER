"""Evidence 191: fixed-start get-up probe. Strict success (training leak off) from the run's one fixed fallen start
(getup start_seed), randomize off (1 run) and on (10 runs); smallest tilt reached.
Usage: .venv-sim/bin/python docs/evidence/191-getup-ground-mechanism/fixed_rate.py sim/rl/runs/r3_gx_side"""
import sys, json, numpy as np
sys.path.insert(0,'sim/rl')
from pathlib import Path
import stage_gates as sg
d=Path(sys.argv[1]); a=json.loads((d/'args.json').read_text()); g=json.loads(a['getup'])
res=[]
for rnd, seeds in ((False,[0]),(True,list(range(10)))):
    for s in seeds:
        kw={**sg.env_kwargs(a),'level':0.,'episode_s':15.,'disturb_items':[],'roll_impulse':None,'yaw_impulse':None,'cmd_max':[0.,0.,0.],'init_speed':False,'getup':{**g,'leak':None}}
        env=sg.RunEnv(**kw,randomize=rnd,seed=8000+s); env.init_seed=8000+s
        model,venv=sg.load(str(d/'model.zip'),env); venv.reset()
        obs=venv.normalize_obs(env._obs()[None]); t=None; best=9.
        for k in range(500):
            raw,_,term,_,info=env.step(model.predict(obs,deterministic=True)[0][0]); obs=venv.normalize_obs(raw[None])
            R=env.data.xmat[env.base].reshape(3,3); best=min(best,float(np.arccos(np.clip(R[2,2],-1,1))))
            if info['getup']['success']: t=round((k+1)*.02,2); break
        res.append(dict(randomize=rnd,seed=s,t_up=t,min_tilt=round(best,2)))
print(json.dumps(dict(run=d.name, nominal_ok=res[0]['t_up'] is not None, dr_ok=sum(r['t_up'] is not None for r in res[1:]), res=res)))
