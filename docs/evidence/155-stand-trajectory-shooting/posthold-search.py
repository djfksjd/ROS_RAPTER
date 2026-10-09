from pathlib import Path
import sys,json,time
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,str(Path('sim/rl').resolve()))
from local_stand_model import LocalStandModel
from contact_wrench import contact_wrench
from stand_observer import StandObserverController
from motion_transition import stand_gate
source=Path('sim/rl/runs/kl005_lr3e5_update64k_20261009')
candidate=next(x for x in map(json.loads,Path('docs/evidence/148-static-equilibrium/margin-search.jsonl').read_text().splitlines()) if x['start_knee']==1.1)
plant=LocalStandModel(source,candidate);plant.set_stand_current_limit(.995);e=plant.env;m=e.model
plant.restore(np.zeros(plant.nx));base_reference=e.data.xpos[e.base].copy()
# Freeze reset-mutated model parameters so every objective evaluation starts equally.
stiffness=m.tendon_stiffness.copy();force_range=m.actuator_forcerange.copy()
knots=np.array([0.,.2,.4,.6]);horizon=80
K=np.load('sim/rl/runs/screen_logs/local_equilibrium_linearization/lqr_design.npz')['K']
out=Path('sim/rl/runs/screen_logs/stand_shooting_posthold_20261009');out.mkdir(exist_ok=False)
log=(out/'objective.jsonl').open('x');calls=0;best=np.inf
pi=e.policy_idx; names=[e.active[i] for i in pi]
# Five independent coordinates at each knot; still eight physical motor outputs.
# Common hip roll + bilateral spread, symmetric pitch/knee, tail pitch; tail yaw0.
def controls(parameters,t):
 z=parameters.reshape(4,5);v=np.array([np.interp(t,knots,z[:,j]) for j in range(5)])
 common,spread,hp,knee,tail=v
 lower=np.maximum(-1.,(e.lo[pi]-e.q0[pi])/e.scale[pi]);upper=np.minimum(1.,(e.hi[pi]-e.q0[pi])/e.scale[pi])
 return np.clip(np.array([common+spread,hp,knee,common-spread,hp,knee,0.,tail]),lower,upper)

def reset(seed):
 m.tendon_stiffness[:]=stiffness;m.actuator_forcerange[:]=force_range
 e.init_seed=seed;e.reset();e.command=np.zeros(3);e.resample_steps=0
 return e._obs()

def feet():
 _,_,contacts=contact_wrench(m,e.data,e.root);f=np.zeros(2)
 for c in contacts:
  name=m.body(c['body']).name
  if any(part in name for part in ['foot','toe','metatarsus']):
   for i,side in enumerate(['left','right']):
    if name.startswith(side):f[i]+=max(0.,c['force_world_N'][2])
 return f/e.weight

def rollout(parameters,seed=83001):
 reset(seed);states=[];rows=[];ever_fell=False
 for step in range(horizon):
  action=controls(parameters,step*.02) if step<30 else np.clip(plant.u-K@plant.state(),np.maximum(-1.,(e.lo[pi]-e.q0[pi])/e.scale[pi]),np.minimum(1.,(e.hi[pi]-e.q0[pi])/e.scale[pi]));_,_,fell,_,info=e.step(action);ever_fell|=fell
  x=plant.state();f=feet();states.append(x)
  rows.append(dict(t=(step+1)*.02,position=e.data.xpos[e.base].tolist(),tilt=float(info['tilt']),fell=bool(fell),flight=bool(info['flight']),foot_load_BW=f.tolist(),qpos=e.data.qpos.tolist(),qvel=e.data.qvel.tolist(),action=action.tolist()))
 return np.array(states),rows,ever_fell

weight=np.ones(plant.nx);weight[:plant.nv]=3.;weight[:3]=0.;weight[3:6]=12.;weight[plant.nv:2*plant.nv]=.3;weight[plant.nv:plant.nv+6]=3.;weight[2*plant.nv:]=.1

def objective(parameters):
 global calls,best
 states,rows,fell=rollout(parameters);terms=[]
 # Final .2s must approach a loaded equilibrium, rather than just match angles in flight.
 for i in [59,69,79]:
  terms.extend((weight*states[i]).tolist());terms.extend((20*(np.array(rows[i]['position'])-base_reference)).tolist());terms.extend((4*np.maximum(.35-np.array(rows[i]['foot_load_BW']),0.)).tolist())
 terms.extend([5*max(r['tilt']-.17,0.) for r in rows]);terms.extend([10*max(.5-r['position'][2],0.) for r in rows])
 # Smooth bounded action trajectory; fall is scored separately, never accepted as success.
 terms.extend((.05*np.diff(parameters.reshape(4,5),axis=0)).ravel().tolist())
 result=np.array(terms);cost=float(result@result);calls+=1
 if cost<best:
  best=cost;np.save(out/'best_parameters.npy',parameters)
  (out/'best_rollout.json').write_text(json.dumps(dict(cost=cost,ever_fell=bool(fell),rows=rows),allow_nan=False))
 log.write(json.dumps(dict(call=calls,cost=cost,best=best,ever_fell=bool(fell),terminal_foot_load_BW=rows[-1]['foot_load_BW'],terminal_tilt=rows[-1]['tilt']))+'\n');log.flush()
 if calls%25==0:print(dict(call=calls,cost=cost,best=best),flush=True)
 return result

def jacobian(parameters):
 epsilon=.002
 return np.column_stack([(objective(parameters+np.eye(len(parameters))[i]*epsilon)-objective(parameters-np.eye(len(parameters))[i]*epsilon))/(2*epsilon) for i in range(len(parameters))])

try:
 v=np.array([0.,plant.u[0],plant.u[1],plant.u[2],plant.u[7]]);initial=np.load('sim/rl/runs/screen_logs/stand_shooting_absolute_20261009/best_parameters.npy');np.save(out/'initial_parameters.npy',initial)
 start=time.monotonic();initial_residual=objective(initial)
 result=least_squares(objective,initial,bounds=(-.9,.9),jac=jacobian,max_nfev=20,ftol=1e-5,xtol=1e-5,gtol=1e-5)
 np.save(out/'final_parameters.npy',result.x)
 (out/'optimizer.json').write_text(json.dumps(dict(nfev=result.nfev,status=result.status,message=result.message,objective_calls=calls,seconds=time.monotonic()-start,initial_cost=float(initial_residual@initial_residual),final_cost=float(result.fun@result.fun),best_cost=best,catalogue_pass=False),indent=2))
 print(json.loads((out/'optimizer.json').read_text()),flush=True)
finally:log.close();plant.close()
