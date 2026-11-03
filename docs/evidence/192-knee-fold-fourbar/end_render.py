"""Evidence 192: A/B fixed-start back get-up policies (r3_gxa_back current knee, r3_gxb_back deep-fold knee), frames at
t = 0, 2, 5, 10 s, plus the end state (base height, left-leg and tail joints, ground contacts, share of saturated actions).
Usage: .venv-sim/bin/python docs/evidence/192-knee-fold-fourbar/end_render.py OUT.png > eval/end_state.txt"""
import sys, json, numpy as np, mujoco
sys.path.insert(0,'sim/rl')
from pathlib import Path
import stage_gates as sg
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
RUNS=(('r3_gxa_back','A: current knee (band 0.64-2.26), full action range'),('r3_gxb_back','B: deep-fold knee (band 0.71-2.96)'))
imgs=[]
for run,_ in RUNS:
    d=Path(f'sim/rl/runs/{run}'); a=json.loads((d/'args.json').read_text()); g=json.loads(a['getup'])
    kw={**sg.env_kwargs(a),'level':0.,'episode_s':15.,'disturb_items':[],'roll_impulse':None,'yaw_impulse':None,'cmd_max':[0.,0.,0.],'init_speed':False,'getup':{**g,'leak':None}}
    env=sg.RunEnv(**kw,randomize=False,seed=8000,render_mode='rgb_array'); env.init_seed=8000
    model,venv=sg.load(str(d/'model.zip'),env); venv.reset(); env.model.vis.headlight.ambient[:]=.45
    obs=venv.normalize_obs(env._obs()[None]); r=mujoco.Renderer(env.model,360,480); M,D=env.model,env.data
    cam=mujoco.MjvCamera(); cam.type=mujoco.mjtCamera.mjCAMERA_TRACKING; cam.trackbodyid=env.base; cam.distance,cam.elevation=2.4,-28.
    snaps=[]; acts=[]
    for k in range(500):
        act=model.predict(obs,deterministic=True)[0][0]; acts.append(act)
        raw,_,_,_,_=env.step(act); obs=venv.normalize_obs(raw[None])
        if k in (0,100,250,499):
            R=D.xmat[env.base].reshape(3,3); cam.azimuth=float(np.degrees(np.arctan2(R[1,0],R[0,0])))+90.
            r.update_scene(D,cam); snaps.append(r.render().copy())
            cs=sorted({M.body(M.geom_bodyid[c.geom2 if M.geom_bodyid[c.geom1]==0 else c.geom1]).name for c in D.contact[:D.ncon] if 0 in (M.geom_bodyid[c.geom1],M.geom_bodyid[c.geom2])})
            q={n.replace('_joint',''):round(float(D.qpos[env.q_adr[i]]),2) for i,n in enumerate(env.active) if 'left' in n or 'tail' in n}
            print(f'{run} t={0.02*(k+1):.1f} tilt={np.arccos(np.clip(R[2,2],-1,1)):.2f} bz={D.xpos[env.base][2]:.2f} contacts={cs} q={q}')
    A=np.array(acts); print(f'{run} share |action|>0.98 after 5 s per policy axis', np.round((np.abs(A[250:])>.98).mean(0),2).tolist())
    imgs.append(snaps)
fig,ax=plt.subplots(2,4,figsize=(16,6.8))
for i,(run,lab) in enumerate(RUNS):
    for j,t in enumerate((0.02,2.02,5.02,10.0)):
        ax[i,j].imshow(imgs[i][j]); ax[i,j].axis('off'); ax[i,j].set_title(f'{lab}\nt={t:.0f} s' if j==0 else f't={t:.0f} s',fontsize=9)
fig.suptitle('Fixed back start (seed 7000), deterministic, MuJoCo + RobStride realism (not hardware); 3.2M of 8M steps',fontsize=10)
fig.tight_layout(); fig.savefig(sys.argv[1],dpi=100)
