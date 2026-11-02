"""Evidence 191: from the stuck prone state of r3_gx_back (t = 10 s), scripted joint targets tuck the legs (hip pitch,
knee, hip roll) with the tail pitch down or up, 1 s ramp then hold; best and final tilt / height, foot loads.
Usage: .venv-sim/bin/python docs/evidence/191-getup-ground-mechanism/prone_tuck.py > eval/prone_tuck.jsonl"""
import sys, json, numpy as np, mujoco
sys.path.insert(0,'sim/rl')
from pathlib import Path
import stage_gates as sg
d=Path('sim/rl/runs/r3_gx_back'); a=json.loads((d/'args.json').read_text()); g=json.loads(a['getup'])
kw={**sg.env_kwargs(a),'level':0.,'episode_s':30.,'disturb_items':[],'roll_impulse':None,'yaw_impulse':None,'cmd_max':[0.,0.,0.],'init_speed':False,'getup':{**g,'leak':None},'action_lpf':None}
def prone():
    env=sg.RunEnv(**kw,randomize=False,seed=8000); env.init_seed=8000
    model,venv=sg.load(str(d/'model.zip'),env); venv.reset(); obs=venv.normalize_obs(env._obs()[None])
    for k in range(500):
        raw,*_=env.step(model.predict(obs,deterministic=True)[0][0]); obs=venv.normalize_obs(raw[None])
    return env
out=[]
for hip in (-1.4,-1.0,-0.6):
  for knee in (1.6,2.2):
    for roll in (0.,.45):
     for tailp in (-.8,.8):
        env=prone(); m,dd=env.model,env.data
        tgt=dd.qpos[env.q_adr].copy()
        for s in ('left','right'):
            for j,v in (('hip_pitch',hip),('knee_pitch',knee),('hip_roll',roll*(1 if s=='left' else -1)),('hip_yaw',0.)):
                tgt[env.active.index(f'{s}_{j}_joint')]=v
            tgt[env.active.index(f'{s}_ankle_pitch_joint')]=dd.qpos[env.q_adr[env.active.index(f'{s}_ankle_pitch_joint')]]+dd.qpos[env.q_adr[env.active.index(f'{s}_knee_pitch_joint')]]-knee
        tgt[env.active.index('tail_pitch_joint')]=tailp; tgt[env.active.index('tail_yaw_joint')]=0.
        start=dd.qpos[env.q_adr].copy(); best=(9,0)
        for k in range(2000):  # 2 s ramp then hold
            f=min(k/1000,1.); dd.ctrl[env.act]=np.clip(start+(tgt-start)*f,env.lo,env.hi)
            env._clamp(); mujoco.mj_step(m,dd)
            R=dd.xmat[env.base].reshape(3,3); tl=float(np.arccos(np.clip(R[2,2],-1,1)))
            if tl<best[0]: best=(round(tl,2),round(float(dd.xpos[env.base][2]),2))
        feet,_=env._contacts()
        out.append(dict(hip=hip,knee=knee,roll=roll,tail=tailp,end_tilt=round(tl,2),end_bz=round(float(dd.xpos[env.base][2]),2),best_tilt_bz=best,feet_loaded={s:round(float(v/env.weight),2) for s,v in feet.items()}))
        print(json.dumps(out[-1]),flush=True)
