"""Evidence 191: lowest crouch the legs can hold. Torso level, both legs posed over a hip pitch x knee grid (ankle on
the knee coupling); reports base height and belly clearance above the lowest foot point for poses with the whole-body CoM
within 4 cm of the foot support centre. Kinematics only (no self-collision, torque or four-bar check).
Usage: .venv-sim/bin/python docs/evidence/191-getup-ground-mechanism/fold_geom.py [KNEE_MAX]"""
import sys, json, numpy as np, mujoco
sys.path.insert(0,'sim/rl')
from run_env import RunEnv
from train_run import env_kwargs
a=json.load(open('sim/rl/runs/r3_s1ho/args.json'))
e=RunEnv(**{**env_kwargs(a),'cmd_max':[0.,0.,0.]},randomize=False,seed=0); e.reset()
m,d=e.model,e.data; q0=d.qpos.copy()
ik={s:e.active.index(f'{s}_knee_pitch_joint') for s in ('left','right')}
ia={s:e.active.index(f'{s}_ankle_pitch_joint') for s in ('left','right')}
ih={s:e.active.index(f'{s}_hip_pitch_joint') for s in ('left','right')}
c=q0[e.q_adr[ia['left']]]+q0[e.q_adr[ik['left']]]
feet=[g for g in range(m.ngeom) if any(k in m.body(m.geom_bodyid[g]).name for k in ('foot','toe','metatars'))]
belly=[g for g in range(m.ngeom) if m.geom_bodyid[g]==e.base]
KMAX=float(sys.argv[1]) if len(sys.argv)>1 else e.hi[ik['left']]
print('knee max',KMAX,'body names with feet:',sorted({m.body(m.geom_bodyid[g]).name for g in feet}))
rows=[]
for hip in np.linspace(e.lo[ih['left']],e.hi[ih['left']],33):
  for knee in np.linspace(e.lo[ik['left']],KMAX,40):
    d.qpos[:]=q0; d.qpos[3:7]=[1,0,0,0]
    for s in ('left','right'):
        d.qpos[e.q_adr[ih[s]]]=hip; d.qpos[e.q_adr[ik[s]]]=knee; d.qpos[e.q_adr[ia[s]]]=c-knee
    mujoco.mj_kinematics(m,d); mujoco.mj_comPos(m,d)
    lo=min(d.geom_xpos[g][2]-m.geom_rbound[g] for g in feet)
    zf=[(d.geom_xpos[g][0],d.geom_xpos[g][2]) for g in feet]
    sup=[x for x,z in zf if z-lo<.04]
    bz=d.xpos[e.base][2]-lo
    belly_lo=min(d.geom_xpos[g][2]-m.geom_rbound[g] for g in belly)-lo
    rows.append((round(hip,2),round(knee,2),round(bz,3),round(float(d.subtree_com[e.root][0]-np.mean(sup)),3),round(belly_lo,3)))
ok=[r for r in rows if abs(r[3])<.04]
print('poses with CoM over feet:',len(ok))
for r in sorted(ok,key=lambda r:r[2])[:6]: print('hip %.2f knee %.2f base_h %.3f com-sup %.3f belly_clear %.3f'%r)
print('nominal-ish', min(rows,key=lambda r:abs(r[0]+.24)+abs(r[1]-1.18)))
