"""Evidence 191: frames of the fixed-start get-up policies r3_gx_side / r3_gx_back (t = 0, 2, 5, 10 s) with contacts.
Usage: .venv-sim/bin/python docs/evidence/191-getup-ground-mechanism/stuck_render.py OUT.png"""
import sys, json, numpy as np, mujoco
sys.path.insert(0,'sim/rl')
from pathlib import Path
import stage_gates as sg
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
imgs=[]; info_txt=[]
for m in ('side','back'):
    d=Path(f'sim/rl/runs/r3_gx_{m}'); a=json.loads((d/'args.json').read_text()); g=json.loads(a['getup'])
    kw={**sg.env_kwargs(a),'level':0.,'episode_s':15.,'disturb_items':[],'roll_impulse':None,'yaw_impulse':None,'cmd_max':[0.,0.,0.],'init_speed':False,'getup':{**g,'leak':None}}
    env=sg.RunEnv(**kw,randomize=False,seed=8000,render_mode='rgb_array'); env.init_seed=8000
    model,venv=sg.load(str(d/'model.zip'),env); venv.reset(); env.model.vis.headlight.ambient[:]=.45
    obs=venv.normalize_obs(env._obs()[None]); r=mujoco.Renderer(env.model,360,480)
    cam=mujoco.MjvCamera(); cam.type=mujoco.mjtCamera.mjCAMERA_TRACKING; cam.trackbodyid=env.base; cam.distance,cam.elevation=2.4,-28.
    snaps=[]
    for k in range(500):
        raw,_,_,_,_=env.step(model.predict(obs,deterministic=True)[0][0]); obs=venv.normalize_obs(raw[None])
        if k in (0,100,250,499):
            R=env.data.xmat[env.base].reshape(3,3); cam.azimuth=float(np.degrees(np.arctan2(R[1,0],R[0,0])))+90.
            r.update_scene(env.data,cam); snaps.append(r.render().copy())
            M,D=env.model,env.data
            cs=sorted({M.body(M.geom_bodyid[c.geom2 if M.geom_bodyid[c.geom1]==0 else c.geom1]).name for c in D.contact[:D.ncon] if 0 in (M.geom_bodyid[c.geom1],M.geom_bodyid[c.geom2])})
            q={n:round(float(D.qpos[env.q_adr[i]]),2) for i,n in enumerate(env.active) if 'left' in n or 'tail' in n}
            info_txt.append(f'{m} t={0.02*(k+1):.1f} tilt={np.arccos(np.clip(R[2,2],-1,1)):.2f} bz={D.xpos[env.base][2]:.2f} contacts={cs} q={q}')
    imgs.append(snaps)
fig,ax=plt.subplots(2,4,figsize=(16,6.5))
for i,m in enumerate(('side','back')):
    for j,t in enumerate((0.02,2.02,5.02,10.0)):
        ax[i,j].imshow(imgs[i][j]); ax[i,j].axis('off'); ax[i,j].set_title(f'{m} start, t={t:.0f} s',fontsize=9)
fig.tight_layout(); fig.savefig(sys.argv[1],dpi=100)
print('\n'.join(info_txt))
