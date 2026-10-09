"""Read-only 1 ms torque-bound telemetry with an uninstrumented parity comparison.

No forward/kinematic recomputation on live data; existing saved physics remain unchanged.
"""
import argparse,json
from pathlib import Path
import numpy as np
import mujoco
from evaluate import load
from run_env import RunEnv
from train_run import env_kwargs
from yaw_feedback import heading_rate


def utilization(torque,low,high):
    torque,low,high=map(np.asarray,(torque,low,high))
    capacity=np.where(torque>=0,high,-low)
    return np.divide(np.abs(torque),capacity,out=np.zeros_like(torque,dtype=float),where=capacity>1e-12)


def probe(run,yaw,seed,instrument=True,yaw_scale=None):
    kw=env_kwargs(json.loads((Path(run)/'args.json').read_text()))
    env=RunEnv(**{**kw,'episode_s':15.,'level':0.},randomize=False,seed=7);env.init_seed=seed
    model,norm=load(str(Path(run)/'model.zip'),env)
    physics=[];controls=[];original=mujoco.mj_step
    def measured_step(m,d):
        time=round(float(d.time),12)
        recording=4.<=time<6.
        if recording:
            low,high=m.actuator_forcerange[ids].T.copy()
            target=d.ctrl[ids].copy();actual=d.qpos[qadr].copy();velocity=d.qvel[vadr].copy()
        original(m,d)
        if recording:
            torque=d.actuator_force[ids].copy()
            physics.append(dict(t=time,target=target.tolist(),actual=actual.tolist(),velocity=velocity.tolist(),
                                torque=torque.tolist(),low=low.tolist(),high=high.tolist(),
                                utilization=utilization(torque,low,high).tolist()))
    try:
        norm.reset()
        # RunEnv constructs actuator arrays in reset(), not in its constructor.
        ids=env.act[env.policy_idx];qadr=env.q_adr[env.policy_idx];vadr=env.v_adr[env.policy_idx]
        names=[env.active[i] for i in env.policy_idx]
        if yaw_scale is not None:
            from symmetric_rollout import RawPolicyAdapter
            model=RawPolicyAdapter(model,norm,env,yaw_scale)
        env.command=np.array([4.,0.,0.]);env.resample_steps=0
        if instrument:mujoco.mj_step=measured_step
        for _ in range(500):
            time=round(float(env.data.time),12);env.command[2]=yaw if 4.<=time<6. else 0.
            obs=norm.normalize_obs(env._obs()[None]);action,_=model.predict(obs,deterministic=True)
            _,_,term,_,info=env.step(action[0])
            R=env.data.xmat[env.base].reshape(3,3);omega=env.body_velocity(env.base)[0]
            controls.append(dict(t=round(float(env.data.time),12),before_t=time,yaw_command=float(env.command[2]),
                                 vx=float(info['v_body'][0]),gyro_z=float(omega[2]),heading_rate=heading_rate(R,omega),
                                 action=action[0].tolist(),qpos=env.data.qpos.tolist(),qvel=env.data.qvel.tolist(),
                                 tilt=float(info['tilt'])))
            if term:break
        summary=[]
        if physics:
            u=np.array([r['utilization'] for r in physics]);err=np.array([np.array(r['target'])-r['actual'] for r in physics])
            tq=np.array([r['torque'] for r in physics]);vel=np.array([r['velocity'] for r in physics])
            for i,n in enumerate(names):
                summary.append(dict(joint=n,near_bound_fraction=float(np.mean(u[:,i]>=.98)),
                                    max_utilization=float(u[:,i].max()),target_error_rms_rad=float(np.sqrt(np.mean(err[:,i]**2))),
                                    peak_torque_Nm=float(np.abs(tq[:,i]).max()),peak_velocity_rad_s=float(np.abs(vel[:,i]).max())))
        return dict(run=run,yaw=yaw,seed=seed,instrumented=instrument,yaw_scale=yaw_scale,
                    fell=bool(term),time_s=float(env.data.time),physics_samples=len(physics),summary=summary,
                    controls=controls,physics=physics)
    finally:mujoco.mj_step=original;norm.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run');p.add_argument('--out',required=True)
    p.add_argument('--yaw-scale',type=float);p.add_argument('--yaw',type=float,nargs='+',default=[-1.,1.])
    p.add_argument('--seed',type=int,default=41001);a=p.parse_args();out=Path(a.out);out.parent.mkdir(exist_ok=True,parents=True)
    with out.open('x') as f:
        for yaw in a.yaw:
            plain=probe(a.run,yaw,a.seed,False,a.yaw_scale);measured=probe(a.run,yaw,a.seed,True,a.yaw_scale)
            parity=plain['controls']==measured['controls'];assert parity,'instrumentation changed control trajectory'
            measured['uninstrumented_control_parity_exact']=parity
            f.write(json.dumps(measured,allow_nan=False)+'\n');f.flush()
            print(json.dumps({k:v for k,v in measured.items() if k not in ['physics','controls']}),flush=True)

if __name__=='__main__':main()
